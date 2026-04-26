"""Diffusion Tree Sampling (DTS) and Diffusion Tree Search (DTS*) for FLUX.

Reproduces the algorithm from Jain et al., arXiv:2506.20701, adapted to the
FLUX schnell / SenseFlow-FLUX runner (sampling_flux_unified.py). See
sampling_unified_sd35_dts.py for the SD3.5 counterpart.

Used as a drop-in replacement for the unified runner: monkey-patches
`base.run_mcts` and calls `base.main(argv)` after rewriting
`--search_method dts/dts_star` → `mcts`. The unified driver dispatches MCTS,
which we have replaced with our DTS implementation.

Branching at non-root nodes requires stochastic transitions:
    * FLUX schnell / senseflow_flux (use_euler=False, x0_sampler=False):
      per-step fresh noise gives natural stochasticity, so DTS branches
      everywhere.
    * Standard FLUX dev / Euler (use_euler=True): the ODE step is
      deterministic. Set `--dts_sde_noise_scale > 0` to inject Gaussian
      noise after each Euler step.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from PIL import Image

import sampling_flux_unified as base
from sampling_flux_unified import (
    FluxContext,
    PromptEmbed,
    SearchResult,
    _compute_dt,
    _final_decode_tensor,
    _pred_x0,
    build_t_schedule,
    decode_to_pil,
    flux_transformer_step,
    make_initial_latents,
    score_image,
)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_dts_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--dts_m_iter",
        type=int,
        default=64,
        help="Number of DTS rollouts (tree iterations).",
    )
    parser.add_argument(
        "--dts_lambda",
        type=float,
        default=1.0,
        help="Boltzmann temperature for selection / soft Bellman backup.",
    )
    parser.add_argument(
        "--dts_pw_c",
        type=float,
        default=1.0,
        help="Progressive widening C: B(N) = ceil(C * N**alpha).",
    )
    parser.add_argument(
        "--dts_pw_alpha",
        type=float,
        default=0.5,
        help="Progressive widening exponent alpha.",
    )
    parser.add_argument(
        "--dts_c_uct",
        type=float,
        default=1.0,
        help="UCB exploration constant for DTS* selection.",
    )
    parser.add_argument(
        "--dts_sde_noise_scale",
        type=float,
        default=0.0,
        help="SDE-style Gaussian noise added after each Euler step (use_euler=True only).",
    )
    parser.add_argument(
        "--dts_cfg_bank",
        type=str,
        default="",
        help="Whitespace-separated CFG values to sample per rollout. Empty → use --baseline_guidance_scale.",
    )
    return parser.parse_known_args(argv)


def _rewrite_search_method(argv: list[str]) -> list[str]:
    out = list(argv)
    for i, tok in enumerate(out):
        if tok == "--search_method" and i + 1 < len(out) and out[i + 1] in ("dts", "dts_star"):
            out[i + 1] = "mcts"
        elif tok.startswith("--search_method=") and tok.split("=", 1)[1] in ("dts", "dts_star"):
            out[i] = "--search_method=mcts"
    return out


def _detect_select_mode(argv: list[str]) -> str:
    """Return 'dts_star' if --search_method=dts_star was requested, else 'dts'."""
    for i, tok in enumerate(argv):
        if tok == "--search_method" and i + 1 < len(argv) and argv[i + 1] == "dts_star":
            return "dts_star"
        if tok.startswith("--search_method=") and tok.split("=", 1)[1] == "dts_star":
            return "dts_star"
    return "dts"


def _make_patched_parse_args(
    original_parse_args: Callable[[list[str] | None], argparse.Namespace],
) -> Callable[[list[str] | None], argparse.Namespace]:
    def _patched_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
        source = list(argv) if argv is not None else []
        if argv is None:
            import sys

            source = list(sys.argv[1:])
        dts_args, remaining = _parse_dts_flags(source)
        select_mode = _detect_select_mode(source)
        normalized = _rewrite_search_method(remaining)
        args = original_parse_args(normalized)
        for key, value in vars(dts_args).items():
            setattr(args, key, value)
        setattr(args, "dts_select_mode", select_mode)
        return args

    return _patched_parse_args


# ── Tree data structures ────────────────────────────────────────────────────


@dataclass
class DTSNode:
    """One node in the diffusion tree. See sampling_unified_sd35_dts.DTSNode."""

    depth: int
    parent: Optional["DTSNode"] = None
    children: list["DTSNode"] = field(default_factory=list)
    n_visits: int = 0
    v_hat: float = 0.0
    cfg: float = 0.0
    variant_idx: int = 0
    latents_cpu: Optional[torch.Tensor] = None
    dx_cpu: Optional[torch.Tensor] = None
    terminal_score: Optional[float] = None
    image: Optional[Image.Image] = None


def _branching_budget(node: DTSNode, c: float, alpha: float) -> int:
    n = max(1, int(node.n_visits))
    return max(1, int(math.ceil(float(c) * (n ** float(alpha)))))


def _is_terminal(node: DTSNode, total_steps: int) -> bool:
    return int(node.depth) >= int(total_steps)


# ── Selection ───────────────────────────────────────────────────────────────


def _boltzmann_sample(values: list[float], lam: float, rng: random.Random) -> int:
    if not values:
        raise ValueError("cannot sample from empty list")
    if len(values) == 1:
        return 0
    m = max(values)
    weights = [math.exp(float(lam) * (v - m)) for v in values]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return rng.randrange(len(values))
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(values) - 1


def _select_child_dts(node: DTSNode, lam: float, rng: random.Random) -> DTSNode:
    values = [float(c.v_hat) for c in node.children]
    return node.children[_boltzmann_sample(values, lam, rng)]


def _ucb_score(child: DTSNode, parent: DTSNode, c_uct: float) -> float:
    n_p = max(1, int(parent.n_visits))
    n_c = max(1, int(child.n_visits))
    return float(child.v_hat) + float(c_uct) * math.sqrt(math.log(n_p) / float(n_c))


def _select_child_dts_star(node: DTSNode, c_uct: float) -> DTSNode:
    return max(node.children, key=lambda c: _ucb_score(c, node, c_uct))


# ── Backup ──────────────────────────────────────────────────────────────────


def _soft_value(values: list[float], lam: float) -> float:
    if not values:
        return 0.0
    m = max(values)
    s = sum(math.exp(float(lam) * (v - m)) for v in values) / float(len(values))
    if s <= 0.0:
        return float(m)
    return float(m) + math.log(s) / float(lam)


def _backup(path: list[DTSNode], lam: float) -> None:
    for node in reversed(path):
        node.n_visits = int(node.n_visits) + 1
        if not node.children:
            if node.terminal_score is not None:
                node.v_hat = float(node.terminal_score)
            continue
        node.v_hat = _soft_value([float(c.v_hat) for c in node.children], lam)


# ── Expansion + rollout ─────────────────────────────────────────────────────


@torch.no_grad()
def _denoise_one_step(
    args: argparse.Namespace,
    ctx: FluxContext,
    embed: PromptEmbed,
    *,
    parent_latents: Optional[torch.Tensor],
    parent_dx: Optional[torch.Tensor],
    t_val: float,
    t_4d: torch.Tensor,
    dt: float,
    step_idx: int,
    guidance: float,
    use_euler: bool,
    fresh_noise: torch.Tensor,
    sde_noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a single denoiser step and return (latents_after, dx_after)."""
    if use_euler:
        latents = parent_latents
        if latents is None:
            raise ValueError("Euler step requires parent_latents")
    else:
        # Re-noise: latents = (1-t)*dx + t*noise
        if parent_dx is None:
            # First step from root: dx is implicitly zeros and "noise" is the
            # initial latent itself, so latents = init_latents = fresh_noise.
            latents = fresh_noise
        else:
            if step_idx == 0:
                latents = fresh_noise
            else:
                latents = (1.0 - t_4d) * parent_dx + t_4d * fresh_noise

    flow = flux_transformer_step(ctx, latents, embed, float(t_val), float(guidance))
    dx_next = _pred_x0(latents, t_4d, flow, args.x0_sampler)
    if use_euler:
        latents_next = latents + float(dt) * flow
        if float(sde_noise_scale) > 0.0:
            latents_next = latents_next + float(sde_noise_scale) * torch.randn_like(latents_next)
    else:
        # Non-euler: latents at next step are recomputed from dx + fresh noise.
        # Keep current latents on hand for storage; next step will re-noise.
        latents_next = latents
    return latents_next, dx_next


@torch.no_grad()
def _expand_and_rollout(
    args: argparse.Namespace,
    ctx: FluxContext,
    embed: PromptEmbed,
    reward_model: Any,
    prompt: str,
    parent: DTSNode,
    t_values: list[float],
    *,
    seed: int,
    iteration: int,
    variant_idx: int,
    guidance: float,
    use_euler: bool,
    sde_noise_scale: float,
) -> list[DTSNode]:
    """Add one new child to `parent`, then roll out to terminal."""
    total_steps = len(t_values)
    device = ctx.device
    chain: list[DTSNode] = []

    if parent.depth == 0:
        latent_seed = int(seed) * 1000003 + int(iteration)
        latents_now: Optional[torch.Tensor] = make_initial_latents(
            ctx, latent_seed, args.height, args.width, batch_size=1
        )
        dx_now: Optional[torch.Tensor] = None
        first_fresh = latents_now
    else:
        latents_now = parent.latents_cpu.to(device) if parent.latents_cpu is not None else None
        dx_now = parent.dx_cpu.to(device) if parent.dx_cpu is not None else None
        if use_euler:
            first_fresh = torch.zeros_like(latents_now) if latents_now is not None else None
        else:
            ref = dx_now if dx_now is not None else latents_now
            first_fresh = torch.randn_like(ref) if ref is not None else None

    cur_latents = latents_now
    cur_dx = dx_now
    for k in range(parent.depth, total_steps):
        t_val = float(t_values[k])
        ref_dtype = (
            cur_latents.dtype
            if cur_latents is not None
            else (cur_dx.dtype if cur_dx is not None else ctx.dtype)
        )
        t_4d = torch.tensor(t_val, device=ctx.device, dtype=ref_dtype).view(1, 1, 1, 1)
        dt = _compute_dt(t_values, k)
        if k == parent.depth:
            fresh = first_fresh
        else:
            if use_euler:
                fresh = torch.zeros_like(cur_latents)
            else:
                fresh = torch.randn_like(cur_dx if cur_dx is not None else cur_latents)
        cur_latents, cur_dx = _denoise_one_step(
            args,
            ctx,
            embed,
            parent_latents=cur_latents,
            parent_dx=cur_dx,
            t_val=t_val,
            t_4d=t_4d,
            dt=dt,
            step_idx=k,
            guidance=guidance,
            use_euler=use_euler,
            fresh_noise=fresh,
            sde_noise_scale=sde_noise_scale,
        )
        new_depth = k + 1
        node = DTSNode(
            depth=new_depth,
            parent=chain[-1] if chain else parent,
            children=[],
            n_visits=0,
            v_hat=0.0,
            cfg=float(guidance),
            variant_idx=int(variant_idx),
        )
        if new_depth < total_steps:
            node.latents_cpu = cur_latents.detach().to("cpu") if cur_latents is not None else None
            node.dx_cpu = cur_dx.detach().to("cpu") if cur_dx is not None else None
        else:
            decode_tensor = _final_decode_tensor(cur_latents, cur_dx, use_euler)
            img = decode_to_pil(ctx, decode_tensor)
            score = score_image(reward_model, prompt, img)
            node.terminal_score = float(score)
            node.v_hat = float(score)
            node.image = img
        if chain:
            chain[-1].children.append(node)
        chain.append(node)

    return chain


# ── Public entry points ─────────────────────────────────────────────────────


def _parse_cfg_bank(raw: Any, fallback: float) -> list[float]:
    if raw is None:
        return [float(fallback)]
    text = str(raw).strip()
    if not text:
        return [float(fallback)]
    out: list[float] = []
    for tok in text.replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out if out else [float(fallback)]


@torch.no_grad()
def _run_dts_core(
    args: argparse.Namespace,
    ctx: FluxContext,
    reward_model: Any,
    prompt: str,
    embeds: list[PromptEmbed],
    seed: int,
    *,
    select_mode: str,
) -> SearchResult:
    M = max(1, int(getattr(args, "dts_m_iter", 64)))
    lam = float(getattr(args, "dts_lambda", 1.0))
    pw_c = float(getattr(args, "dts_pw_c", 1.0))
    pw_alpha = float(getattr(args, "dts_pw_alpha", 0.5))
    c_uct = float(getattr(args, "dts_c_uct", 1.0))
    sde_noise_scale = float(getattr(args, "dts_sde_noise_scale", 0.0))
    baseline_cfg = float(getattr(args, "baseline_guidance_scale", 1.0))
    cfg_bank = _parse_cfg_bank(getattr(args, "dts_cfg_bank", ""), baseline_cfg)
    variant_idx = 0
    use_euler = bool(getattr(args, "euler_sampler", False))

    t_values = build_t_schedule(int(args.steps), getattr(args, "sigmas", None))
    total_steps = len(t_values)
    rng = random.Random(int(seed) * 7919 + hash(select_mode) % (2**31))

    root = DTSNode(depth=0)
    best_score = -float("inf")
    best_image: Optional[Image.Image] = None

    cfg_bank_str = " ".join(f"{c:.3g}" for c in cfg_bank)
    print(
        f"  {select_mode}: M={M} lam={lam:.3f} pw=(C={pw_c:.2f},a={pw_alpha:.2f}) "
        f"c_uct={c_uct:.2f} sde={sde_noise_scale:.3f} cfg_bank=[{cfg_bank_str}] "
        f"steps={total_steps} euler={use_euler}"
    )

    iter_log: list[dict[str, Any]] = []
    for it in range(M):
        # ── Selection ──
        path: list[DTSNode] = [root]
        node = root
        while True:
            if _is_terminal(node, total_steps):
                break
            budget = _branching_budget(node, pw_c, pw_alpha)
            if len(node.children) < budget:
                break
            if select_mode == "dts":
                child = _select_child_dts(node, lam, rng)
            else:
                child = _select_child_dts_star(node, c_uct)
            path.append(child)
            node = child

        if _is_terminal(node, total_steps):
            _backup(path, lam)
            score = float(node.terminal_score) if node.terminal_score is not None else -float("inf")
            cfg_used = float(node.cfg)
        else:
            cfg_used = (
                float(cfg_bank[rng.randrange(len(cfg_bank))])
                if len(cfg_bank) > 1
                else float(cfg_bank[0])
            )
            chain = _expand_and_rollout(
                args,
                ctx,
                embeds[variant_idx],
                reward_model,
                prompt,
                node,
                t_values,
                seed=seed,
                iteration=it,
                variant_idx=variant_idx,
                guidance=cfg_used,
                use_euler=use_euler,
                sde_noise_scale=sde_noise_scale,
            )
            node.children.append(chain[0])
            full_path = path + chain
            leaf = chain[-1]
            score = float(leaf.terminal_score) if leaf.terminal_score is not None else -float("inf")
            if score > best_score:
                best_score = score
                best_image = leaf.image
            _backup(full_path, lam)

        if (it + 1) % max(1, M // 8) == 0 or it == 0 or it == M - 1:
            print(
                f"    iter {it + 1:3d}/{M} best={best_score:.4f} root_v={root.v_hat:.4f} "
                f"root_children={len(root.children)} N={root.n_visits}"
            )
        iter_log.append(
            {
                "iter": int(it),
                "score": float(score),
                "best": float(best_score),
                "cfg": float(cfg_used),
            }
        )

    # ── Final selection: descend without expansion ──
    out_image: Optional[Image.Image]
    out_score: float
    final_node = root
    final_path: list[DTSNode] = []
    while final_node.children:
        if select_mode == "dts":
            final_node = _select_child_dts(final_node, lam, rng)
        else:
            final_node = max(final_node.children, key=lambda c: float(c.v_hat))
        final_path.append(final_node)

    if final_node.image is not None and final_node.terminal_score is not None:
        out_image = final_node.image
        out_score = float(final_node.terminal_score)
    else:
        out_image = best_image
        out_score = float(best_score)

    if out_image is None:
        raise RuntimeError("DTS produced no terminal sample.")

    if len(final_path) == total_steps:
        actions = [(int(n.variant_idx), float(n.cfg)) for n in final_path]
    else:
        actions = [(int(variant_idx), float(baseline_cfg)) for _ in range(total_steps)]

    diagnostics = {
        "select_mode": select_mode,
        "M": int(M),
        "lambda": float(lam),
        "pw_C": float(pw_c),
        "pw_alpha": float(pw_alpha),
        "c_uct": float(c_uct),
        "sde_noise_scale": float(sde_noise_scale),
        "cfg_bank": [float(c) for c in cfg_bank],
        "baseline_cfg": float(baseline_cfg),
        "best_seen_score": float(best_score) if math.isfinite(best_score) else None,
        "final_path_depths": [int(n.depth) for n in final_path],
        "final_path_cfgs": [float(n.cfg) for n in final_path],
        "root_v_hat": float(root.v_hat),
        "root_children": len(root.children),
        "iter_log_tail": iter_log[-min(8, len(iter_log)) :],
    }
    return SearchResult(
        image=out_image,
        score=out_score,
        actions=actions,
        diagnostics=diagnostics,
    )


def main(argv: list[str] | None = None) -> None:
    original_parse_args = base.parse_args
    original_run_mcts = base.run_mcts

    def _patched_run_mcts(args, ctx, reward_model, prompt, embeds, guidance_bank, seed):
        select_mode = str(getattr(args, "dts_select_mode", "dts"))
        return _run_dts_core(
            args=args,
            ctx=ctx,
            reward_model=reward_model,
            prompt=prompt,
            embeds=embeds,
            seed=seed,
            select_mode=select_mode,
        )

    base.parse_args = _make_patched_parse_args(original_parse_args)
    base.run_mcts = _patched_run_mcts
    try:
        base.main(argv)
    finally:
        base.parse_args = original_parse_args
        base.run_mcts = original_run_mcts


if __name__ == "__main__":
    main()
