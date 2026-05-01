"""Search-over-Paths (SoP) inference-time search for SD3.5 / SiD.

Path-space search that holds prompt variant and CFG fixed and searches only
over noise/path perturbations. At each branching step, every kept path is
expanded into M children by injecting fresh noise into the current
intermediate state, the children are denoised one step forward, and the top K
are retained based on x0-pred reward (or only at the end if --sop_score_decode
final). Used as a noise-search baseline against MCTS/SMC under matched NFE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from PIL import Image

import sampling_unified_sd35 as su


# ── CLI args ────────────────────────────────────────────────────────────────


def add_sop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sop_init_paths", type=int, default=8,
                        help="Number of initial Gaussian noises N.")
    parser.add_argument("--sop_branch_factor", type=int, default=4,
                        help="Children per kept path at each branching step (M).")
    parser.add_argument("--sop_keep_top", type=int, default=4,
                        help="Top-K paths kept after each branching step.")
    parser.add_argument("--sop_branch_every", type=int, default=1,
                        help="Branch at every N denoising steps (1 = every step in window).")
    parser.add_argument("--sop_start_frac", type=float, default=0.25,
                        help="Progress fraction at which branching begins (below: just denoise).")
    parser.add_argument("--sop_end_frac", type=float, default=1.0,
                        help="Progress fraction at which branching ends.")
    parser.add_argument("--sop_score_decode", choices=["x0_pred", "final"], default="x0_pred",
                        help="x0_pred: decode + score x0_pred at every branch step. final: score only at end.")
    parser.add_argument("--sop_variant_idx", type=int, default=0,
                        help="Prompt variant index used for the entire search.")


# ── Path container ──────────────────────────────────────────────────────────


@dataclass
class _Path:
    latents: torch.Tensor
    dx: torch.Tensor
    score: float = 0.0


# ── Driver ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_sop_search(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    seed: int,
    cfg_scale: float = 1.0,
) -> su.SearchResult:
    """Search-over-Paths: K paths, branch by fresh-noise re-injection, prune by x0_pred reward."""
    n_init = max(1, int(getattr(args, "sop_init_paths", 8)))
    branch_M = max(1, int(getattr(args, "sop_branch_factor", 4)))
    keep_K = max(1, int(getattr(args, "sop_keep_top", 4)))
    branch_every = max(1, int(getattr(args, "sop_branch_every", 1)))
    start_frac = float(getattr(args, "sop_start_frac", 0.25))
    end_frac = float(getattr(args, "sop_end_frac", 1.0))
    score_x0 = (str(getattr(args, "sop_score_decode", "x0_pred")) == "x0_pred")
    variant_idx = int(getattr(args, "sop_variant_idx", 0))
    variant_idx = max(0, min(len(emb.cond_text) - 1, variant_idx))

    use_euler = bool(getattr(args, "euler_sampler", False))
    x0_sampler = bool(getattr(args, "x0_sampler", False))

    # Build schedule and progress bounds.
    init_latents = su.make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    sched = su.step_schedule(
        ctx.device, init_latents.dtype, args.steps,
        getattr(args, "sigmas", None), euler=use_euler,
    )
    sigma_vals = [float(s[0].item()) for s in sched]
    sigma_min = float(min(sigma_vals)) if sigma_vals else 0.0
    sigma_max = float(max(sigma_vals)) if sigma_vals else 1.0
    span = max(1e-8, sigma_max - sigma_min)
    n_steps = len(sched)

    # N initial paths: first reuses the seed-derived latent, others use independent draws.
    paths: list[_Path] = [
        _Path(latents=init_latents, dx=torch.zeros_like(init_latents), score=0.0)
    ]
    for _ in range(n_init - 1):
        paths.append(
            _Path(
                latents=torch.randn_like(init_latents),
                dx=torch.zeros_like(init_latents),
                score=0.0,
            )
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _step_one(p: _Path, t_flat, t_4d, dt, step_idx: int) -> _Path:
        if not use_euler:
            noise = p.latents if step_idx == 0 else torch.randn_like(p.latents)
            latents_in = (1.0 - t_4d) * p.dx + t_4d * noise
        else:
            latents_in = p.latents
        flow = su.transformer_step(args, ctx, latents_in, emb, variant_idx, t_flat, float(cfg_scale))
        new_latents, new_dx = su._apply_step(latents_in, flow, p.dx, t_4d, dt, use_euler, x0_sampler)
        return _Path(latents=new_latents, dx=new_dx, score=p.score)

    def _perturb(p: _Path, t_4d) -> _Path:
        # Inject fresh noise at the current intermediate state.
        # SiD: re-noise current x0_pred with a fresh ε.
        # Euler: stochastic linear blend toward fresh ε scaled by t.
        new_noise = torch.randn_like(p.latents)
        if not use_euler:
            new_latents = (1.0 - t_4d) * p.dx + t_4d * new_noise
        else:
            new_latents = (1.0 - t_4d) * p.latents + t_4d * new_noise
        return _Path(latents=new_latents, dx=p.dx.clone(), score=p.score)

    def _decode_x0_score(p: _Path) -> float:
        # Score from x0_pred (SiD) or running latent (Euler).
        tensor = p.latents if use_euler else p.dx
        img = su.decode_to_pil(ctx, tensor)
        return float(su.score_image(reward_model, prompt, img))

    print(
        f"  sop: N={n_init} M={branch_M} K={keep_K} every={branch_every} "
        f"window=[{start_frac:.2f},{end_frac:.2f}] score={'x0' if score_x0 else 'final'} "
        f"steps={n_steps} cfg={cfg_scale:.2f} variant={variant_idx}"
    )

    branch_step_log: list[dict] = []

    # ── Main loop ────────────────────────────────────────────────────────────

    for step_idx, (t_flat, t_4d, dt) in enumerate(sched):
        sigma_i = float(t_flat.item())
        progress = max(0.0, min(1.0, 1.0 - (sigma_i - sigma_min) / span))
        in_window = (progress >= start_frac - 1e-8) and (progress <= end_frac + 1e-8)
        do_branch = in_window and ((step_idx % branch_every) == 0)

        if not do_branch:
            paths = [_step_one(p, t_flat, t_4d, dt, step_idx) for p in paths]
            continue

        children: list[_Path] = []
        for p in paths:
            for _m in range(branch_M):
                p_branched = _perturb(p, t_4d)
                p_advanced = _step_one(p_branched, t_flat, t_4d, dt, step_idx)
                if score_x0:
                    p_advanced.score = _decode_x0_score(p_advanced)
                children.append(p_advanced)

        if score_x0:
            children.sort(key=lambda x: x.score, reverse=True)
        paths = children[: max(1, keep_K)]

        top = paths[0].score if paths else float("nan")
        print(
            f"    step {step_idx + 1}/{n_steps} progress={progress:.2f} "
            f"branched K*M={len(children)} kept={len(paths)} top_score={top:.4f}"
        )
        branch_step_log.append({
            "step": int(step_idx),
            "progress": float(progress),
            "n_children": int(len(children)),
            "n_kept": int(len(paths)),
            "top_score": float(top) if score_x0 else None,
        })

    # ── Final decode + score ────────────────────────────────────────────────

    best_score = -float("inf")
    best_img: Image.Image | None = None
    final_scores: list[float] = []
    for p in paths:
        tensor = p.latents if use_euler else p.dx
        img = su.decode_to_pil(ctx, tensor)
        s = float(su.score_image(reward_model, prompt, img))
        final_scores.append(s)
        if s > best_score:
            best_score = s
            best_img = img
    assert best_img is not None

    actions = [(variant_idx, float(cfg_scale), 0.0) for _ in range(n_steps)]
    diagnostics = {
        "n_init": n_init,
        "branch_factor": branch_M,
        "keep_top": keep_K,
        "branch_every": branch_every,
        "start_frac": start_frac,
        "end_frac": end_frac,
        "score_decode": "x0_pred" if score_x0 else "final",
        "n_branch_steps": len(branch_step_log),
        "final_scores": [float(x) for x in final_scores],
        "branch_log": branch_step_log,
    }
    return su.SearchResult(image=best_img, score=best_score, actions=actions, diagnostics=diagnostics)
