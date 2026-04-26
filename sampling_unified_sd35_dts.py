"""Diffusion Tree Sampling (DTS) and Diffusion Tree Search (DTS*).

Reproduces the algorithm from
    Jain et al., "Diffusion Tree Sampling: Scalable inference-time alignment of
    diffusion models", arXiv:2506.20701.

Tree nodes correspond to noisy latent states (x_t, t). Edges correspond to
denoising transitions x_{t-1} ~ p_theta(. | x_t). Each iteration:

    1. Selection — descend from root by Boltzmann sampling children with
       weights proportional to exp(lam * v_hat) (DTS), or argmax + UCB
       exploration (DTS*).
    2. Expansion — at the first node whose child count is below
       B(N) = ceil(C * N**alpha), draw a fresh denoiser child and attach it.
    3. Rollout — denoise stochastically until t=0, scoring the terminal
       reward r(x_0). Every state along the rollout is added to the tree
       (single-child chain).
    4. Backup — propagate values along the visited path with the soft Bellman
       equation V_t = (1/lam) * log E_{p_theta}[exp(lam * V_{t-1})], increment
       visit counts.

The final returned sample is selected by descending from the root once more,
without expansion: Boltzmann sampling for DTS, argmax for DTS*.

Vanilla DTS uses the baseline CFG and the original prompt (no rewrites). Tree
branching at non-root nodes requires stochastic transitions:

    * SiD / SenseFlow (use_euler=False):  per-step fresh noise gives natural
      stochasticity, so DTS branches everywhere.
    * Standard Euler (use_euler=True, sd35_base default): the ODE step is
      deterministic. Only the root has true randomness (initial noise). Set
      `--dts_sde_noise_scale > 0` to inject Gaussian noise after each Euler
      step, recovering full DTS branching.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from PIL import Image

from sampling_unified_sd35 import (
    EmbeddingContext,
    PipelineContext,
    SearchResult,
    _apply_step,
    _final_decode_tensor,
    _prepare_latents,
    decode_to_pil,
    make_latents,
    score_image,
    step_schedule,
    transformer_step,
)
from reward_unified import UnifiedRewardScorer


# ── Tree data structures ────────────────────────────────────────────────────


@dataclass
class DTSNode:
    """One node in the diffusion tree.

    `depth` is the number of denoising steps taken from the root. Root has
    depth 0 and no state; its children at depth 1 correspond to different
    initial-noise samples followed by one denoiser step. Terminal nodes have
    depth == total_steps.
    """

    depth: int
    parent: Optional["DTSNode"] = None
    children: list["DTSNode"] = field(default_factory=list)
    n_visits: int = 0
    v_hat: float = 0.0
    # CFG used to produce the transition into this node (root: not applicable).
    cfg: float = 0.0
    # Latent state AFTER this node's step. Stored on CPU to bound GPU memory.
    latents_cpu: Optional[torch.Tensor] = None
    dx_cpu: Optional[torch.Tensor] = None
    # For terminal nodes only.
    terminal_score: Optional[float] = None
    image: Optional[Image.Image] = None


def _branching_budget(node: DTSNode, c: float, alpha: float) -> int:
    """Progressive widening: B(N) = ceil(C * max(1, N)**alpha)."""
    n = max(1, int(node.n_visits))
    return max(1, int(math.ceil(float(c) * (n ** float(alpha)))))


def _is_terminal(node: DTSNode, total_steps: int) -> bool:
    return int(node.depth) >= int(total_steps)


# ── Selection ───────────────────────────────────────────────────────────────


def _boltzmann_sample(values: list[float], lam: float, rng: random.Random) -> int:
    """Sample an index proportional to exp(lam * v). Numerically stable."""
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
    idx = _boltzmann_sample(values, lam, rng)
    return node.children[idx]


def _ucb_score(child: DTSNode, parent: DTSNode, c_uct: float) -> float:
    n_p = max(1, int(parent.n_visits))
    n_c = max(1, int(child.n_visits))
    return float(child.v_hat) + float(c_uct) * math.sqrt(math.log(n_p) / float(n_c))


def _select_child_dts_star(node: DTSNode, c_uct: float) -> DTSNode:
    return max(node.children, key=lambda c: _ucb_score(c, node, c_uct))


# ── Backup ──────────────────────────────────────────────────────────────────


def _soft_value(values: list[float], lam: float) -> float:
    """V = (1/lam) * log( mean_c exp(lam * v_c) ).

    Numerically stable using the standard log-mean-exp trick.
    """
    if not values:
        return 0.0
    m = max(values)
    s = sum(math.exp(float(lam) * (v - m)) for v in values) / float(len(values))
    if s <= 0.0:
        return float(m)
    return float(m) + math.log(s) / float(lam)


def _backup(path: list[DTSNode], lam: float) -> None:
    # Walk from leaf back to root: each internal node's v_hat is recomputed
    # from its children's current v_hat values.
    for node in reversed(path):
        node.n_visits = int(node.n_visits) + 1
        if not node.children:
            # Terminal: v_hat = terminal reward (set by caller).
            if node.terminal_score is not None:
                node.v_hat = float(node.terminal_score)
            continue
        node.v_hat = _soft_value([float(c.v_hat) for c in node.children], lam)


# ── Expansion + rollout ─────────────────────────────────────────────────────


@torch.no_grad()
def _denoise_one_step(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    *,
    parent_latents: Optional[torch.Tensor],
    parent_dx: Optional[torch.Tensor],
    sched_entry: tuple[torch.Tensor, torch.Tensor, float],
    step_idx: int,
    variant_idx: int,
    cfg: float,
    use_euler: bool,
    fresh_noise: torch.Tensor,
    sde_noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take a single denoiser step and return (latents_after, dx_after).

    For SiD (use_euler=False), `parent_dx` is required (zeros at the very
    first step). `fresh_noise` is the per-step noise sample to use at this
    step (for step 0 it should equal `parent_latents`, the initial latent).

    For Euler (use_euler=True), `parent_latents` is required; `fresh_noise`
    is added with scale `sde_noise_scale` to inject SDE-like stochasticity.
    """
    t_flat, t_4d, dt = sched_entry
    if use_euler:
        latents = parent_latents
        if latents is None:
            raise ValueError("Euler step requires parent_latents")
        dx = parent_dx if parent_dx is not None else torch.zeros_like(latents)
    else:
        # SiD re-noise: latents = (1-t)*dx + t*noise
        if parent_dx is None:
            # First step: dx is zeros, "noise" is the initial latent.
            dx = torch.zeros_like(fresh_noise)
            latents = fresh_noise
        else:
            dx = parent_dx
            if step_idx == 0:
                # The very first step's "noise" is the initial latent itself.
                latents = fresh_noise
            else:
                latents = (1.0 - t_4d) * dx + t_4d * fresh_noise

    flow = transformer_step(args, ctx, latents, emb, variant_idx, t_flat, float(cfg))
    latents_next, dx_next = _apply_step(latents, flow, dx, t_4d, dt, use_euler, args.x0_sampler)
    if use_euler and float(sde_noise_scale) > 0.0:
        latents_next = latents_next + float(sde_noise_scale) * torch.randn_like(latents_next)
    return latents_next, dx_next


@torch.no_grad()
def _expand_and_rollout(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    parent: DTSNode,
    sched: list[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    seed: int,
    iteration: int,
    variant_idx: int,
    cfg: float,
    use_euler: bool,
    sde_noise_scale: float,
    initial_latent_dtype: torch.dtype,
) -> list[DTSNode]:
    """Add one new child to `parent`, then roll out to terminal.

    Returns the chain of newly-created nodes from depth `parent.depth + 1`
    down to depth `len(sched)`. The last node gets its terminal score and
    image set.
    """
    total_steps = len(sched)
    device = ctx.device
    chain: list[DTSNode] = []

    # Restore parent's state on GPU (root has none).
    if parent.depth == 0:
        # Root: sample a fresh initial latent for this child. Each child of
        # root corresponds to a different initial-noise sample, so we use a
        # seed derived from (seed, iteration) for determinism.
        latent_seed = int(seed) * 1000003 + int(iteration)
        latents_now: Optional[torch.Tensor] = make_latents(
            ctx, latent_seed, args.height, args.width, initial_latent_dtype
        )
        dx_now: Optional[torch.Tensor] = None  # will be initialized to zeros inside _denoise_one_step
        # First step uses latents_now as the "fresh_noise" too.
        first_fresh = latents_now
    else:
        latents_now = parent.latents_cpu.to(device) if parent.latents_cpu is not None else None
        dx_now = parent.dx_cpu.to(device) if parent.dx_cpu is not None else None
        if use_euler:
            first_fresh = torch.zeros_like(latents_now)  # unused; SDE noise added inside step
        else:
            first_fresh = torch.randn_like(dx_now if dx_now is not None else latents_now)

    # Walk from depth `parent.depth` down to total_steps, creating one node
    # per step. Each node represents the state AFTER applying step
    # `parent.depth + k` of the schedule (1-indexed within the chain).
    cur_latents = latents_now
    cur_dx = dx_now
    for k in range(parent.depth, total_steps):
        sched_entry = sched[k]
        # For step 0 we use first_fresh; for later steps draw new noise.
        if k == parent.depth:
            fresh = first_fresh
        else:
            if use_euler:
                fresh = torch.zeros_like(cur_latents)  # SDE noise added inside step
            else:
                fresh = torch.randn_like(cur_latents if cur_dx is None else cur_dx)
        cur_latents, cur_dx = _denoise_one_step(
            args, ctx, emb,
            parent_latents=cur_latents,
            parent_dx=cur_dx,
            sched_entry=sched_entry,
            step_idx=k,
            variant_idx=variant_idx,
            cfg=cfg,
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
            cfg=float(cfg),
        )
        # Cache latents/dx on CPU for future expansions (skip terminal: we only
        # need the decoded image there).
        if new_depth < total_steps:
            node.latents_cpu = cur_latents.detach().to("cpu") if cur_latents is not None else None
            node.dx_cpu = cur_dx.detach().to("cpu") if cur_dx is not None else None
        else:
            # Terminal: decode + score.
            final_tensor = _final_decode_tensor(cur_latents, cur_dx, use_euler)
            img = decode_to_pil(ctx, final_tensor)
            score = score_image(reward_model, prompt, img)
            node.terminal_score = float(score)
            node.v_hat = float(score)
            node.image = img
        # Wire child into its parent (chain or `parent`).
        if chain:
            chain[-1].children.append(node)
        chain.append(node)

    # Note: we do NOT yet attach chain[0] to `parent.children` here — the
    # caller does it so it can also extend its descent path.
    return chain


# ── Public entry points ─────────────────────────────────────────────────────


@torch.no_grad()
def run_dts(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
) -> SearchResult:
    """Diffusion Tree Sampling — Boltzmann selection at every level."""
    return _run_dts_core(args, ctx, emb, reward_model, prompt, seed, select_mode="dts")


@torch.no_grad()
def run_dts_star(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
    seed: int,
) -> SearchResult:
    """Diffusion Tree Search — UCB selection (greedy with exploration)."""
    return _run_dts_core(args, ctx, emb, reward_model, prompt, seed, select_mode="dts_star")


def _parse_cfg_bank(raw: Any, fallback: float) -> list[float]:
    """Parse a whitespace-separated CFG bank string. Empty/None → [fallback]."""
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


def _run_dts_core(
    args: argparse.Namespace,
    ctx: PipelineContext,
    emb: EmbeddingContext,
    reward_model: UnifiedRewardScorer,
    prompt: str,
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
    baseline_cfg = float(args.baseline_cfg)
    cfg_bank = _parse_cfg_bank(getattr(args, "dts_cfg_bank", ""), baseline_cfg)
    variant_idx = 0  # vanilla DTS: original prompt only.
    use_euler = bool(getattr(args, "euler_sampler", False))

    dtype = emb.cond_text[0].dtype
    sched = step_schedule(
        ctx.device, dtype, int(args.steps), getattr(args, "sigmas", None), euler=use_euler
    )
    total_steps = len(sched)
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
            # All allowed children expanded — descend.
            if select_mode == "dts":
                child = _select_child_dts(node, lam, rng)
            else:
                child = _select_child_dts_star(node, c_uct)
            path.append(child)
            node = child

        if _is_terminal(node, total_steps):
            # Already a leaf — just visit it again (re-backup, no new rollout).
            _backup(path, lam)
            score = float(node.terminal_score) if node.terminal_score is not None else -float("inf")
            cfg_used = float(node.cfg)
        else:
            # ── Expansion + rollout ──
            cfg_used = float(cfg_bank[rng.randrange(len(cfg_bank))]) if len(cfg_bank) > 1 else float(cfg_bank[0])
            chain = _expand_and_rollout(
                args, ctx, emb, reward_model, prompt, node, sched,
                seed=seed,
                iteration=it,
                variant_idx=variant_idx,
                cfg=cfg_used,
                use_euler=use_euler,
                sde_noise_scale=sde_noise_scale,
                initial_latent_dtype=dtype,
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
            print(f"    iter {it + 1:3d}/{M} best={best_score:.4f} root_v={root.v_hat:.4f} "
                  f"root_children={len(root.children)} N={root.n_visits}")
        iter_log.append({
            "iter": int(it),
            "score": float(score),
            "best": float(best_score),
            "cfg": float(cfg_used),
        })

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
        # Final descent didn't reach a terminal (shouldn't happen if rollouts
        # always reach depth=total_steps). Fall back to best-seen.
        out_image = best_image
        out_score = float(best_score)

    if out_image is None:
        raise RuntimeError("DTS produced no terminal sample.")

    if len(final_path) == total_steps:
        actions = [(variant_idx, float(n.cfg), 0.0) for n in final_path]
    else:
        # Fallback: use baseline_cfg if final descent was incomplete.
        actions = [(variant_idx, baseline_cfg, 0.0) for _ in range(total_steps)]
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
    return SearchResult(image=out_image, score=out_score, actions=actions, diagnostics=diagnostics)
