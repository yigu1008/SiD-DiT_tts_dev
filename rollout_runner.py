from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

import sampling_unified as su
from blend_ops import BlendResult, blend_prompt_embeddings
from prompt_basis import PromptBasisEmbeddings
from weight_policy import WeightParams, progress_from_sigmas, weights_for_progress


@dataclass
class StepTrace:
    step: int
    sigma: float
    progress: float
    blend_family: str
    selected_indices: list[int]
    selected_labels: list[str]
    selected_weights: list[float]
    weights_by_label: dict[str, float]
    preview_reward: float
    delta_reward: float


@dataclass
class RolloutResult:
    final_score: float
    final_image: Image.Image
    step_traces: list[StepTrace]
    preview_rewards: list[float]


def resolve_resolution(
    args: Any,
    ctx: su.PipelineContext,
) -> tuple[int, int, int, int]:
    orig_h = int(args.height)
    orig_w = int(args.width)
    h, w = su.maybe_resize_to_bin(ctx, orig_h, orig_w, bool(args.resolution_binning))
    return orig_h, orig_w, int(h), int(w)


def _should_preview(step_idx: int, steps: int, every: int) -> bool:
    if every < 0:
        return False
    if every <= 0:
        return step_idx == steps - 1
    return ((step_idx + 1) % every == 0) or (step_idx == steps - 1)


def _prepare_schedule(ctx: su.PipelineContext, steps: int, dtype: torch.dtype) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[float], list[float]]:
    sched = su._step_tensors(ctx, int(steps), dtype)
    sigmas = [float(t_flat[0].item()) for t_flat, _ in sched]
    progress = progress_from_sigmas(sigmas)
    return sched, sigmas, progress


@torch.no_grad()
def run_dynamic_rollout(
    args: Any,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    basis_emb: PromptBasisEmbeddings,
    basis_indices: list[int],
    blend_family: str,
    weight_params: WeightParams,
    preview_every: int = 1,
    save_path: str | None = None,
    tag: str = "dynamic",
) -> RolloutResult:
    if len(basis_indices) <= 0:
        raise ValueError("basis_indices cannot be empty.")
    idxs = [int(i) for i in basis_indices]
    if min(idxs) < 0 or max(idxs) >= len(basis_emb.pe_list):
        raise ValueError(f"basis_indices out of range: {idxs}")
    k = len(idxs)
    if int(weight_params.a.shape[0]) != k or int(weight_params.b.shape[0]) != k:
        raise ValueError(f"Weight params dimension mismatch: k={k} a={weight_params.a.shape} b={weight_params.b.shape}")

    latents = su.make_latents(ctx, int(seed), int(h), int(w), basis_emb.orig_pe.dtype)
    sched, sigmas, progress = _prepare_schedule(ctx, int(args.steps), latents.dtype)
    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(int(seed) + 2048)

    step_traces: list[StepTrace] = []
    preview_rewards: list[float] = []
    prev_reward = 0.0

    for step_idx, ((t_flat, t_4d), sigma_t, u_t) in enumerate(zip(sched, sigmas, progress)):
        noise = latents if step_idx == 0 else torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=rng,
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise

        weights_np = weights_for_progress(weight_params, float(u_t))
        weights_t = torch.tensor(weights_np, dtype=basis_emb.pe_list[0][0].dtype, device=ctx.device)

        pe_bank = [basis_emb.pe_list[i][0] for i in idxs]
        pm_bank = [basis_emb.pe_list[i][1] for i in idxs]
        blend: BlendResult = blend_prompt_embeddings(pe_bank, pm_bank, weights_t, blend_family)

        velocity = su.transformer_step(
            args,
            ctx,
            latents,
            blend.prompt_embed,
            blend.prompt_mask,
            basis_emb.ue,
            basis_emb.um,
            t_flat,
            float(args.guidance_scale),
        )
        dx = latents - t_4d * velocity

        cur_reward = prev_reward
        delta = 0.0
        if _should_preview(step_idx, int(args.steps), int(preview_every)):
            preview = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_preview")
            cur_reward = float(reward_ctx.score_images(prompt, [preview], metadata)[0])
            delta = cur_reward - prev_reward
            preview_rewards.append(cur_reward)

        weights_by_label = {
            basis_emb.labels[idxs[i]]: float(weights_np[i])
            for i in range(k)
        }
        selected_labels = [basis_emb.labels[idxs[i]] for i in blend.selected_indices]
        selected_weights = [float(blend.selected_weights[i]) for i in range(len(blend.selected_weights))]
        step_traces.append(
            StepTrace(
                step=int(step_idx),
                sigma=float(sigma_t),
                progress=float(u_t),
                blend_family=str(blend_family),
                selected_indices=[int(idxs[i]) for i in blend.selected_indices],
                selected_labels=selected_labels,
                selected_weights=selected_weights,
                weights_by_label=weights_by_label,
                preview_reward=float(cur_reward),
                delta_reward=float(delta),
            )
        )
        prev_reward = cur_reward

    final_image = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    if save_path:
        final_image.save(save_path)
    return RolloutResult(
        final_score=float(final_score),
        final_image=final_image,
        step_traces=step_traces,
        preview_rewards=[float(x) for x in preview_rewards],
    )


@torch.no_grad()
def run_single_prompt_rollout(
    args: Any,
    ctx: su.PipelineContext,
    reward_ctx: su.RewardContext,
    prompt: str,
    metadata: dict[str, Any] | None,
    seed: int,
    h: int,
    w: int,
    orig_h: int,
    orig_w: int,
    pe: torch.Tensor,
    pm: torch.Tensor,
    ue: torch.Tensor,
    um: torch.Tensor,
    preview_every: int = 1,
    save_path: str | None = None,
    tag: str = "single",
) -> RolloutResult:
    latents = su.make_latents(ctx, int(seed), int(h), int(w), pe.dtype)
    sched, sigmas, progress = _prepare_schedule(ctx, int(args.steps), latents.dtype)
    dx = torch.zeros_like(latents)
    rng = torch.Generator(device=ctx.device).manual_seed(int(seed) + 2048)

    step_traces: list[StepTrace] = []
    preview_rewards: list[float] = []
    prev_reward = 0.0

    for step_idx, ((t_flat, t_4d), sigma_t, u_t) in enumerate(zip(sched, sigmas, progress)):
        noise = latents if step_idx == 0 else torch.randn(
            latents.shape,
            device=latents.device,
            dtype=latents.dtype,
            generator=rng,
        )
        latents = (1.0 - t_4d) * dx + t_4d * noise
        velocity = su.transformer_step(args, ctx, latents, pe, pm, ue, um, t_flat, float(args.guidance_scale))
        dx = latents - t_4d * velocity

        cur_reward = prev_reward
        delta = 0.0
        if _should_preview(step_idx, int(args.steps), int(preview_every)):
            preview = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_preview")
            cur_reward = float(reward_ctx.score_images(prompt, [preview], metadata)[0])
            delta = cur_reward - prev_reward
            preview_rewards.append(cur_reward)

        step_traces.append(
            StepTrace(
                step=int(step_idx),
                sigma=float(sigma_t),
                progress=float(u_t),
                blend_family="single",
                selected_indices=[0],
                selected_labels=["single"],
                selected_weights=[1.0],
                weights_by_label={"single": 1.0},
                preview_reward=float(cur_reward),
                delta_reward=float(delta),
            )
        )
        prev_reward = cur_reward

    final_image = su.decode_to_pil(ctx, dx, orig_h, orig_w, tag=f"{tag}_final")
    final_score = float(reward_ctx.score_images(prompt, [final_image], metadata)[0])
    if save_path:
        final_image.save(save_path)
    return RolloutResult(
        final_score=float(final_score),
        final_image=final_image,
        step_traces=step_traces,
        preview_rewards=[float(x) for x in preview_rewards],
    )
