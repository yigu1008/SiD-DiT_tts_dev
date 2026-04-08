"""
Decoupled SD3.5 base sampler with integrated MCTS (lookahead + dynamic CFG).

Uses diffusers StableDiffusion3Pipeline directly — no SiD dependency.
Delegates shared utilities (transformer_step, encode_variants, greedy/GA/SMC/BoN/beam)
to sampling_unified_sd35, overriding only:
  - load_pipeline  (diffusers-native)
  - run_mcts       (unified lookahead + dynamic CFG, no reward correction)

SD3.5 base defaults: steps=28, cfg_scales=[3.5..7.0], baseline_cfg=4.5, bfloat16.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sampling_unified_sd35 as su


# ---------------------------------------------------------------------------
# SD3.5 base defaults
# ---------------------------------------------------------------------------

_SD35_BASE_CFG_SCALES = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0]
_SD35_BASE_BASELINE_CFG = 4.5
_SD35_BASE_STEPS = 28


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_extra_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse lookahead + dynamic-CFG MCTS args on top of the base sampler args."""
    extra = argparse.ArgumentParser(add_help=False)

    # Lookahead reweighting knobs
    extra.add_argument(
        "--lookahead_mode",
        choices=[
            "standard",
            "instrumentation",
            "rollout_prior",
            "tree_prior",
            "rollout_tree_prior",
            "rollout_tree_prior_adaptive_cfg",
            "adaptive_cfg_width",
        ],
        default="rollout_tree_prior_adaptive_cfg",
    )
    extra.add_argument("--lookahead_u_t_def", choices=["latent_delta_rms", "latent_rms", "dx_rms"], default="latent_delta_rms")
    extra.add_argument("--lookahead_tau", type=float, default=0.35)
    extra.add_argument("--lookahead_c_puct", type=float, default=1.20)
    extra.add_argument("--lookahead_u_ref", type=float, default=0.0)
    extra.add_argument("--lookahead_w_cfg", type=float, default=1.0)
    extra.add_argument("--lookahead_w_variant", type=float, default=0.25)
    extra.add_argument("--lookahead_w_q", type=float, default=0.20)
    extra.add_argument("--lookahead_w_explore", type=float, default=0.05)
    extra.add_argument("--lookahead_cfg_width_min", type=int, default=3)
    extra.add_argument("--lookahead_cfg_width_max", type=int, default=7)
    extra.add_argument("--lookahead_cfg_anchor_count", type=int, default=2)
    extra.add_argument("--lookahead_min_visits_for_center", type=int, default=3)
    extra.add_argument("--lookahead_log_action_topk", type=int, default=12)

    # Dynamic CFG knobs
    extra.add_argument("--mcts_cfg_mode", choices=["adaptive", "fixed"], default="adaptive")
    extra.add_argument("--mcts_cfg_root_bank", nargs="+", type=float, default=None)
    extra.add_argument("--mcts_cfg_anchors", nargs="+", type=float, default=None)
    extra.add_argument("--mcts_cfg_step_anchor_count", type=int, default=2)
    extra.add_argument("--mcts_cfg_min_parent_visits", type=int, default=3)
    extra.add_argument("--mcts_cfg_round_ndigits", type=int, default=6)
    extra.add_argument("--mcts_cfg_log_action_topk", type=int, default=12)

    parsed_extra, remaining = extra.parse_known_args(argv)
    args = su.parse_args(remaining)
    for key, value in vars(parsed_extra).items():
        setattr(args, key, value)
    return args


def _apply_sd35_base_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Override defaults for SD3.5 base when the user hasn't set them explicitly."""
    # Force sd35_base backend
    args.backend = "sd35_base"

    # Steps
    if args.steps == 4:  # parser default (SiD)
        args.steps = _SD35_BASE_STEPS

    # CFG scales — replace SiD defaults with SD3.5 base range
    sid_defaults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
    if list(args.cfg_scales) == sid_defaults:
        args.cfg_scales = list(_SD35_BASE_CFG_SCALES)

    # Baseline CFG
    if args.baseline_cfg == 1.0:  # SiD default
        args.baseline_cfg = _SD35_BASE_BASELINE_CFG

    # Dynamic CFG root bank / anchors: default to SD3.5 base range if not set
    if getattr(args, "mcts_cfg_root_bank", None) is None:
        args.mcts_cfg_root_bank = [3.5, 5.0, 7.0]
    if getattr(args, "mcts_cfg_anchors", None) is None:
        args.mcts_cfg_anchors = [3.5, 7.0]

    # Force no reward correction
    args.correction_strengths = [0.0]

    # Force x0_sampler off (SD3.5 base uses flow matching, not x0 prediction)
    args.x0_sampler = False

    return args


# ---------------------------------------------------------------------------
# Pipeline loading (diffusers-native, no SiD)
# ---------------------------------------------------------------------------

def load_pipeline_sd35base(args: argparse.Namespace) -> su.PipelineContext:
    """Load SD3.5 base using StableDiffusion3Pipeline from diffusers."""
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Compatibility shim for mixed huggingface_hub versions.
    try:
        import huggingface_hub.constants as hhc
        if not hasattr(hhc, "HF_HOME"):
            cache_root = getattr(hhc, "HUGGINGFACE_HUB_CACHE", None)
            if cache_root:
                hhc.HF_HOME = str(Path(cache_root).expanduser().parent)
            else:
                hhc.HF_HOME = str(Path.home() / ".cache" / "huggingface")
    except Exception:
        pass

    # Compat shims for transformers
    try:
        import transformers.utils as _tu
        if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
            _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        import transformers.modeling_utils as _tmu
        if not hasattr(_tmu, "apply_chunking_to_forward"):
            def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
                if chunk_size > 0:
                    tensor_shape = input_tensors[0].shape[chunk_dim]
                    if tensor_shape % chunk_size != 0:
                        raise ValueError(f"tensor shape {tensor_shape} not divisible by chunk_size {chunk_size}")
                    num_chunks = tensor_shape // chunk_size
                    return torch.cat(
                        [forward_fn(*[t.narrow(chunk_dim, c * chunk_size, chunk_size) for t in input_tensors])
                         for c in range(num_chunks)],
                        dim=chunk_dim,
                    )
                return forward_fn(*input_tensors)
            _tmu.apply_chunking_to_forward = _apply_chunking_to_forward
        if not hasattr(_tmu, "find_pruneable_heads_and_indices"):
            def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
                mask = torch.ones(n_heads, head_size)
                heads = set(heads) - already_pruned_heads
                for head in sorted(heads):
                    head -= sum(1 for h in sorted(already_pruned_heads) if h < head)
                    mask[head] = 0
                mask = mask.view(-1).eq(1)
                index = torch.arange(len(mask))[mask].long()
                return heads, index
            _tmu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
    except Exception:
        pass

    from diffusers import StableDiffusion3Pipeline

    cuda_available = torch.cuda.is_available()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if cuda_available:
        device = f"cuda:{local_rank}" if world_size > 1 else "cuda"
    else:
        device = "cpu"
        if world_size > 1:
            raise RuntimeError(
                "CUDA is unavailable under torchrun (WORLD_SIZE>1). "
                f"WORLD_SIZE={world_size} LOCAL_RANK={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )

    dtype_str = getattr(args, "dtype", None) or "bfloat16"
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    model_id = args.model_id or "stabilityai/stable-diffusion-3.5-large"
    dev_count = int(torch.cuda.device_count()) if cuda_available else 0
    print(
        f"Loading SD3.5 base pipeline: {model_id} "
        f"(device={device} dtype={dtype_str} device_count={dev_count} local_rank={local_rank})"
    )

    transformer_id = getattr(args, "transformer_id", None)
    transformer_subfolder = getattr(args, "transformer_subfolder", None)

    pretrained_kwargs: dict = {"torch_dtype": dtype}
    if transformer_id:
        from diffusers.models.transformers import SD3Transformer2DModel
        tf_kwargs: dict = {"torch_dtype": dtype}
        tf_path = transformer_id
        if transformer_subfolder and os.path.isdir(transformer_id):
            joined = os.path.join(transformer_id, transformer_subfolder)
            if os.path.isdir(joined):
                tf_path = joined
            else:
                tf_kwargs["subfolder"] = transformer_subfolder
        elif transformer_subfolder:
            tf_kwargs["subfolder"] = transformer_subfolder
        print(f"Loading transformer from {tf_path} subfolder={tf_kwargs.get('subfolder')}")
        pretrained_kwargs["transformer"] = SD3Transformer2DModel.from_pretrained(
            tf_path, **tf_kwargs
        ).to(device)

    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, **pretrained_kwargs).to(device)
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    # Normalize text-encoder dtypes.
    for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        enc = getattr(pipe, name, None)
        if enc is not None:
            try:
                enc.to(device=device, dtype=dtype)
            except Exception as exc:
                print(f"[warn] unable to cast {name} to {dtype}: {exc}")

    # Optional checkpoint override
    if args.ckpt:
        print(f"Loading transformer weights from {args.ckpt}")
        raw = torch.load(args.ckpt, map_location=device, weights_only=False)
        state_dict = su._unwrap_state_dict(raw)
        if any(str(k).startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        print(
            f"  loaded={len(state_dict) - len(unexpected)}/{len(state_dict)} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    pipe.transformer.eval()
    latent_c = pipe.transformer.config.in_channels
    return su.PipelineContext(pipe=pipe, device=device, dtype=dtype, latent_c=latent_c)


# ---------------------------------------------------------------------------
# MCTS Node (with u_t + PUCT from lookahead)
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = (
        "step", "dx", "latents", "parent", "incoming_action", "u_t",
        "children", "visits", "action_visits", "action_values",
    )

    def __init__(
        self,
        step: int,
        dx: torch.Tensor,
        latents: torch.Tensor | None,
        *,
        parent: "MCTSNode | None" = None,
        incoming_action: tuple[int, float] | None = None,
        u_t: float = 0.0,
    ) -> None:
        self.step = int(step)
        self.dx = dx
        self.latents = latents
        self.parent = parent
        self.incoming_action = incoming_action
        self.u_t = float(u_t)
        self.children: dict[tuple[int, float], MCTSNode] = {}
        self.visits = 0
        self.action_visits: dict[tuple[int, float], int] = {}
        self.action_values: dict[tuple[int, float], float] = {}

    def is_leaf(self, max_steps: int) -> bool:
        return self.step >= int(max_steps)

    def untried_actions(self, actions: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [a for a in actions if a not in self.action_visits]

    def ucb(self, action: tuple[int, float], c: float) -> float:
        n = int(self.action_visits.get(action, 0))
        if n <= 0:
            return float("inf")
        mean = float(self.action_values.get(action, 0.0)) / float(n)
        return mean + float(c) * math.sqrt(math.log(max(self.visits, 1)) / float(n))

    def best_ucb(self, actions: list[tuple[int, float]], c: float) -> tuple[int, float]:
        return max(actions, key=lambda a: self.ucb(a, c))

    def puct(self, action: tuple[int, float], prior: float, c_puct: float) -> float:
        n = int(self.action_visits.get(action, 0))
        q = float(self.action_values.get(action, 0.0)) / float(n) if n > 0 else 0.0
        bonus = float(c_puct) * float(prior) * math.sqrt(float(max(1, self.visits))) / float(1 + n)
        return float(q + bonus)

    def best_puct(
        self,
        actions: list[tuple[int, float]],
        prior_map: dict[tuple[int, float], float],
        c_puct: float,
    ) -> tuple[int, float]:
        return max(actions, key=lambda a: self.puct(a, float(prior_map.get(a, 0.0)), c_puct))

    def best_exploit(self, actions: list[tuple[int, float]]) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        best_v = -float("inf")
        for action in actions:
            n = int(self.action_visits.get(action, 0))
            if n <= 0:
                continue
            mean = float(self.action_values.get(action, 0.0)) / float(n)
            if mean > best_v:
                best_v = mean
                best = action
        return best


# ---------------------------------------------------------------------------
# Utility functions (from lookahead + dynamic_cfg, adapted for no correction)
# ---------------------------------------------------------------------------

def _rms_tensor(t: torch.Tensor) -> float:
    if t is None or t.numel() <= 0:
        return 0.0
    x = t.detach().float()
    return float(torch.sqrt(torch.mean(x * x)).item())


def _compute_u_t(
    u_def: str,
    parent_latents: torch.Tensor | None,
    child_latents: torch.Tensor | None,
    child_dx: torch.Tensor | None,
) -> float:
    key = str(u_def).strip().lower()
    if key == "latent_delta_rms":
        if parent_latents is None or child_latents is None:
            return 0.0
        return _rms_tensor(child_latents - parent_latents)
    if key == "latent_rms":
        return _rms_tensor(child_latents if child_latents is not None else parent_latents)
    if key == "dx_rms":
        return _rms_tensor(child_dx)
    return 0.0


def _softmax_prior(logits: np.ndarray, tau: float) -> np.ndarray:
    if logits.size <= 0:
        return np.zeros((0,), dtype=np.float64)
    t = max(1e-6, float(tau))
    shifted = (logits - float(np.max(logits))) / t
    shifted = np.clip(shifted, -50.0, 50.0)
    e = np.exp(shifted)
    s = float(np.sum(e))
    if not np.isfinite(s) or s <= 0.0:
        return np.full((logits.size,), 1.0 / float(max(1, logits.size)), dtype=np.float64)
    return (e / s).astype(np.float64)


def _zscore(x: float, values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std())
    if sd <= 1e-8:
        return 0.0
    return float((float(x) - float(arr.mean())) / sd)


def _dedup_float_list(values: list[float], ndigits: int = 6) -> list[float]:
    out: list[float] = []
    seen: set[float] = set()
    for val in values:
        v = float(round(float(val), int(ndigits)))
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _pick_anchor_cfgs(cfg_values: list[float], baseline_cfg: float, k: int) -> list[float]:
    if len(cfg_values) <= 0 or k <= 0:
        return []
    vals = sorted(float(x) for x in cfg_values)
    if len(vals) <= k:
        return _dedup_float_list(vals + [float(baseline_cfg)])
    idxs = np.linspace(0, len(vals) - 1, num=int(k), dtype=int).tolist()
    picked = [float(vals[int(i)]) for i in idxs]
    picked.append(float(baseline_cfg))
    return _dedup_float_list(picked)


def _cfg_delta_for_depth(depth: int) -> float:
    if int(depth) <= 0:
        return 0.50
    if int(depth) == 1:
        return 0.25
    return 0.125


def _mode_flags(mode: str) -> dict[str, bool]:
    m = str(mode).strip().lower()
    return {
        "use_rollout_prior": m in {"rollout_prior", "rollout_tree_prior", "rollout_tree_prior_adaptive_cfg"},
        "use_tree_prior": m in {"tree_prior", "rollout_tree_prior", "rollout_tree_prior_adaptive_cfg"},
        "adaptive_cfg_width": m in {"adaptive_cfg_width", "rollout_tree_prior_adaptive_cfg"},
        "instrumentation_only": m in {"instrumentation"},
    }


# ---------------------------------------------------------------------------
# Expand child (no reward correction)
# ---------------------------------------------------------------------------

def _expand_child(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    node: MCTSNode,
    action: tuple[int, float],
    sched: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    variant_idx, cfg = action
    t_flat, t_4d = sched[node.step]
    flow = su.transformer_step(args, ctx, node.latents, emb, variant_idx, t_flat, cfg)
    new_dx = su._pred_x0(node.latents, t_4d, flow, args.x0_sampler)
    next_step = node.step + 1
    if next_step < len(sched):
        _, next_t_4d = sched[next_step]
        noise = torch.randn_like(new_dx)
        new_latents = (1.0 - next_t_4d) * new_dx + next_t_4d * noise
    else:
        new_latents = None
    return new_dx, new_latents


# ---------------------------------------------------------------------------
# Integrated MCTS: lookahead priors + dynamic CFG, no reward correction
# ---------------------------------------------------------------------------

def run_mcts_sd35base(
    args: argparse.Namespace,
    ctx: su.PipelineContext,
    emb: su.EmbeddingContext,
    reward_model: su.UnifiedRewardScorer,
    prompt: str,
    variants: list[str],
    seed: int,
) -> su.SearchResult:
    del variants

    # Embedding interpolation
    family = getattr(args, "mcts_interp_family", "none")
    n_interp = int(getattr(args, "mcts_n_interp", 1))
    if family != "none":
        emb = su.expand_emb_with_interp(emb, family, n_interp)
        print(f"  mcts: interp={family} n_interp={n_interp} total_variants={len(emb.cond_text)}")

    # Mode flags (from lookahead)
    flags = _mode_flags(str(getattr(args, "lookahead_mode", "rollout_tree_prior_adaptive_cfg")))
    use_rollout_prior = bool(flags["use_rollout_prior"])
    use_tree_prior = bool(flags["use_tree_prior"])
    adaptive_cfg_width = bool(flags["adaptive_cfg_width"])

    # CFG values and action space — no correction_strengths
    cfg_values = _dedup_float_list([float(x) for x in getattr(args, "cfg_scales", list(_SD35_BASE_CFG_SCALES))])
    if len(cfg_values) <= 0:
        cfg_values = [float(getattr(args, "baseline_cfg", _SD35_BASE_BASELINE_CFG))]
    cfg_min = float(min(cfg_values))
    cfg_max = float(max(cfg_values))
    cfg_span = max(1e-6, float(cfg_max - cfg_min))
    n_variants = len(emb.cond_text)

    # Dynamic CFG settings
    cfg_mode = str(getattr(args, "mcts_cfg_mode", "adaptive")).strip().lower()
    cfg_root_bank = [float(x) for x in getattr(args, "mcts_cfg_root_bank", [3.5, 5.0, 7.0])]
    cfg_anchors = [float(x) for x in getattr(args, "mcts_cfg_anchors", [3.5, 7.0])]
    cfg_step_anchor_count = max(0, int(getattr(args, "mcts_cfg_step_anchor_count", 2)))
    cfg_min_parent_visits = max(1, int(getattr(args, "mcts_cfg_min_parent_visits", 3)))
    cfg_round_ndigits = max(1, int(getattr(args, "mcts_cfg_round_ndigits", 6)))

    def clamp_cfg(cfg: float) -> float:
        return float(round(float(np.clip(float(cfg), cfg_min, cfg_max)), cfg_round_ndigits))

    def dedup_cfg(values: list[float]) -> list[float]:
        out: list[float] = []
        seen: set[float] = set()
        for val in values:
            v = clamp_cfg(float(val))
            if v not in seen:
                out.append(float(v))
                seen.add(float(v))
        if len(out) <= 0:
            out = [clamp_cfg(float(getattr(args, "baseline_cfg", cfg_values[0])))]
        return out

    root_cfg_values_step0 = dedup_cfg(cfg_root_bank + cfg_anchors)
    fixed_cfg_values = dedup_cfg(cfg_values + cfg_anchors)
    default_cfg = clamp_cfg(float(getattr(args, "baseline_cfg", cfg_values[0])))
    cfg_root_center = float(np.mean(root_cfg_values_step0))

    # Lookahead adaptive-width settings
    root_cfg_anchors_la = _pick_anchor_cfgs(
        cfg_values,
        baseline_cfg=float(getattr(args, "baseline_cfg", cfg_values[0])),
        k=max(0, int(getattr(args, "lookahead_cfg_anchor_count", 2))),
    )
    min_visits_for_center = max(1, int(getattr(args, "lookahead_min_visits_for_center", 3)))
    cfg_w_min = max(1, int(getattr(args, "lookahead_cfg_width_min", 3)))
    cfg_w_max = max(cfg_w_min, int(getattr(args, "lookahead_cfg_width_max", 7)))
    cfg_w_max = min(cfg_w_max, len(cfg_values))
    if cfg_w_min > len(cfg_values):
        cfg_w_min = len(cfg_values)

    # MCTS tuning
    rng = np.random.default_rng(int(seed) + 4046)
    n_sims = max(1, int(getattr(args, "n_sims", 50)))
    ucb_c = float(getattr(args, "ucb_c", 1.41))
    c_puct = float(getattr(args, "lookahead_c_puct", 1.20))
    tau = float(getattr(args, "lookahead_tau", 0.35))
    topk = int(getattr(args, "lookahead_log_action_topk", 12))

    # u_t tracking
    u_values_seen: list[float] = []

    def current_u_ref() -> float:
        fixed_ref = float(getattr(args, "lookahead_u_ref", 0.0))
        if fixed_ref > 0.0:
            return max(1e-6, fixed_ref)
        if len(u_values_seen) <= 0:
            return 1.0
        arr = np.asarray(u_values_seen, dtype=np.float64)
        q = float(np.percentile(arr, 75))
        if np.isfinite(q) and q > 1e-8:
            return q
        m = float(np.mean(np.abs(arr)))
        return max(1e-6, m if m > 1e-8 else 1.0)

    # --- Dynamic CFG helpers ---

    def timestep_root_bank(step: int) -> dict[str, Any]:
        t = max(0, int(step))
        if t <= 0:
            bank_vals = dedup_cfg(cfg_root_bank + cfg_anchors)
            source = "explicit_root_bank_t0"
        else:
            dt = float(_cfg_delta_for_depth(t))
            bank_vals = dedup_cfg([float(cfg_root_center - dt), float(cfg_root_center), float(cfg_root_center + dt)] + cfg_anchors)
            source = "timestep_root_bank"
        return {"step": int(t), "values": [float(x) for x in bank_vals], "source": str(source)}

    def pick_step_anchors(step_root: list[float], extra_anchors: list[float], k: int) -> list[float]:
        merged = dedup_cfg([float(x) for x in step_root] + [float(x) for x in extra_anchors])
        if k <= 0 or len(merged) <= 0:
            return []
        if len(merged) <= k:
            return [float(x) for x in merged]
        idx = np.linspace(0, len(merged) - 1, num=k, dtype=int).tolist()
        return [float(merged[i]) for i in sorted(set(int(i) for i in idx))]

    def visit_weighted_center(node: MCTSNode) -> tuple[float | None, int]:
        total = 0
        wsum = 0.0
        for action, visits in node.action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            total += int(v)
            wsum += float(action[1]) * float(v)
        if total <= 0:
            return None, 0
        return float(wsum / float(total)), int(total)

    def parent_best_cfg(node: MCTSNode) -> float | None:
        parent = node.parent
        if parent is None:
            return None
        best_cfg = None
        best_mean = -float("inf")
        for action, visits in parent.action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            mean = float(parent.action_values.get(action, 0.0)) / float(v)
            if mean > best_mean:
                best_mean = mean
                best_cfg = float(action[1])
        return None if best_cfg is None else float(best_cfg)

    def cfg_width_from_u(u_t: float) -> int:
        if not adaptive_cfg_width:
            return int(len(cfg_values))
        u_ref = current_u_ref()
        ratio = float(np.clip(float(u_t) / max(1e-6, u_ref), 0.0, 2.0))
        frac = float(np.clip(ratio / 2.0, 0.0, 1.0))
        raw = float(cfg_w_min) + (float(cfg_w_max - cfg_w_min) * frac)
        width = int(round(raw))
        width = max(cfg_w_min, min(cfg_w_max, width))
        if width % 2 == 0 and width < cfg_w_max:
            width += 1
        elif width % 2 == 0 and width > cfg_w_min:
            width -= 1
        return int(max(1, width))

    # --- Unified node_candidates: dynamic CFG + adaptive width ---

    def node_candidates(node: MCTSNode) -> tuple[dict[str, Any], list[tuple[int, float]]]:
        # Step 1: determine the CFG bank for this node
        if cfg_mode == "fixed":
            cfg_bank = [float(x) for x in fixed_cfg_values]
            center = float(np.mean(cfg_bank))
            center_source = "fixed_global"
            width = int(len(cfg_bank))
        elif int(node.step) <= 0:
            step_root = timestep_root_bank(0)
            cfg_bank = [float(x) for x in step_root["values"]]
            center = float(np.mean(cfg_bank))
            center_source = "root_timestep_bank"
            width = int(len(cfg_bank))
        else:
            # Adaptive CFG center (from dynamic_cfg)
            center = None
            center_source = "fallback"

            local_center, local_visits = visit_weighted_center(node)
            if local_center is not None and local_visits >= cfg_min_parent_visits:
                center = float(local_center)
                center_source = "node_visit_weighted"

            if center is None:
                parent = node.parent
                if parent is not None:
                    parent_center, parent_visits = visit_weighted_center(parent)
                    if parent_center is not None and parent_visits >= cfg_min_parent_visits:
                        center = float(parent_center)
                        center_source = "parent_visit_weighted"

            if center is None:
                best_cfg = parent_best_cfg(node)
                if best_cfg is not None:
                    center = float(best_cfg)
                    center_source = "parent_best_sparse"

            if center is None and node.incoming_action is not None:
                center = float(node.incoming_action[1])
                center_source = "incoming_cfg_fallback"

            if center is None:
                center = float(default_cfg)
                center_source = "baseline_cfg_fallback"

            # Adaptive width from u_t (from lookahead)
            width = cfg_width_from_u(node.u_t)

            # Dynamic CFG delta (from dynamic_cfg)
            delta = float(_cfg_delta_for_depth(int(node.step)))
            step_root = timestep_root_bank(int(node.step))
            step_root_anchors = pick_step_anchors(step_root["values"], cfg_anchors, cfg_step_anchor_count)

            # Build CFG bank: center +/- delta, plus anchors, narrowed by width
            base_cfgs = dedup_cfg(
                [float(center - delta), float(center), float(center + delta)]
                + [float(x) for x in step_root_anchors]
            )
            # If adaptive_cfg_width: narrow to width closest to center
            if adaptive_cfg_width and len(base_cfgs) > width:
                nearest = sorted(base_cfgs, key=lambda c: (abs(float(c) - float(center)), float(c)))
                cfg_bank = sorted(_dedup_float_list(
                    [float(x) for x in nearest[:width]] + [float(x) for x in root_cfg_anchors_la]
                ))
            else:
                cfg_bank = sorted(base_cfgs)

        # Step 2: build actions = variant x cfg
        actions: list[tuple[int, float]] = []
        for vi in range(n_variants):
            for cfg in cfg_bank:
                actions.append((int(vi), clamp_cfg(float(cfg))))
        if len(actions) <= 0:
            actions = [(0, default_cfg)]

        meta = {
            "step_idx": int(node.step),
            "u_t": float(node.u_t),
            "cfg_bank": [float(x) for x in cfg_bank],
            "cfg_bank_width": int(width),
            "cfg_center": float(center),
            "cfg_center_source": str(center_source),
            "u_ref": float(current_u_ref()),
            "candidate_count": int(len(actions)),
            "adaptive_cfg_width": bool(adaptive_cfg_width),
            "cfg_mode": str(cfg_mode),
        }
        return meta, actions

    # --- Action logits (from lookahead, no w_cs) ---

    def compute_action_logits(
        node: MCTSNode,
        candidates: list[tuple[int, float]],
    ) -> np.ndarray:
        if len(candidates) <= 0:
            return np.zeros((0,), dtype=np.float64)

        w_cfg = float(getattr(args, "lookahead_w_cfg", 1.0))
        w_variant = float(getattr(args, "lookahead_w_variant", 0.25))
        w_q = float(getattr(args, "lookahead_w_q", 0.20))
        w_explore = float(getattr(args, "lookahead_w_explore", 0.05))

        u_ratio = float(np.clip(float(node.u_t) / max(1e-6, current_u_ref()), 0.0, 2.0))
        u01 = float(np.clip(u_ratio, 0.0, 1.0))
        cfg_target = float(cfg_min + (cfg_span * u01))

        by_variant: dict[int, list[float]] = {}
        for action, visits in node.action_visits.items():
            v = int(visits)
            if v <= 0:
                continue
            mean = float(node.action_values.get(action, 0.0)) / float(v)
            by_variant.setdefault(int(action[0]), []).append(float(mean))
        variant_means = {k: float(np.mean(vals)) for k, vals in by_variant.items() if len(vals) > 0}
        variant_pool = [float(x) for x in variant_means.values()]

        action_q_means: dict[tuple[int, float], float] = {}
        for action in candidates:
            n = int(node.action_visits.get(action, 0))
            if n > 0:
                action_q_means[action] = float(node.action_values.get(action, 0.0)) / float(n)
            else:
                action_q_means[action] = 0.0
        q_pool = [float(x) for x in action_q_means.values()]

        out = np.zeros((len(candidates),), dtype=np.float64)
        for i, action in enumerate(candidates):
            vi, cfg = int(action[0]), float(action[1])
            cfg_score = -abs(cfg - cfg_target) / cfg_span
            variant_score = _zscore(float(variant_means.get(vi, 0.0)), variant_pool)
            q_score = _zscore(float(action_q_means.get(action, 0.0)), q_pool)
            n = int(node.action_visits.get(action, 0))
            explore_score = 1.0 / math.sqrt(float(1 + n))
            out[i] = (
                (w_cfg * cfg_score)
                + (w_variant * variant_score)
                + (w_q * q_score)
                + (w_explore * explore_score)
            )
        return out

    # --- Selection helpers ---

    def sample_with_prior(
        candidates: list[tuple[int, float]],
        prior: np.ndarray,
    ) -> tuple[int, float]:
        if len(candidates) <= 0:
            raise RuntimeError("Cannot sample from empty candidates.")
        if prior.size != len(candidates):
            return candidates[int(rng.integers(0, len(candidates)))]
        if not np.all(np.isfinite(prior)) or float(np.sum(prior)) <= 0.0:
            return candidates[int(rng.integers(0, len(candidates)))]
        idx = int(rng.choice(len(candidates), p=prior))
        return candidates[idx]

    def select_untried_with_optional_prior(
        node: MCTSNode,
        candidates: list[tuple[int, float]],
        prior: np.ndarray,
    ) -> tuple[int, float]:
        untried_idx = [i for i, action in enumerate(candidates) if action not in node.action_visits]
        if len(untried_idx) <= 0:
            return candidates[int(rng.integers(0, len(candidates)))]
        if not (use_rollout_prior or use_tree_prior):
            return candidates[int(untried_idx[int(rng.integers(0, len(untried_idx)))])]
        p = np.asarray([float(prior[i]) for i in untried_idx], dtype=np.float64)
        s = float(np.sum(p))
        if (not np.isfinite(s)) or s <= 0.0:
            return candidates[int(untried_idx[int(rng.integers(0, len(untried_idx)))])]
        p = p / s
        picked_local = int(rng.choice(len(untried_idx), p=p))
        return candidates[int(untried_idx[picked_local])]

    # --- Initialize tree ---

    latents0 = su.make_latents(ctx, seed, args.height, args.width, emb.cond_text[0].dtype)
    dx0 = torch.zeros_like(latents0)
    sched = su.step_schedule(ctx.device, latents0.dtype, args.steps, getattr(args, "sigmas", None))
    _, t0_4d = sched[0]
    start_latents = (1.0 - t0_4d) * dx0 + t0_4d * latents0
    root = MCTSNode(step=0, dx=dx0, latents=start_latents)

    best_global_score = -float("inf")
    best_global_dx = None
    best_global_path: list[tuple[int, float]] = []
    history: list[dict[str, Any]] = []
    node_logs: list[dict[str, Any]] = []
    log_every = 10

    print(
        f"  mcts(sd35base): sims={n_sims} variants={n_variants} steps={int(args.steps)} "
        f"cfg_mode={cfg_mode} cfg_range=[{cfg_min:.1f},{cfg_max:.1f}] "
        f"lookahead_mode={getattr(args, 'lookahead_mode', 'rollout_tree_prior_adaptive_cfg')}"
    )

    # --- Main MCTS loop ---

    for sim in range(n_sims):
        node = root
        path: list[tuple[MCTSNode, tuple[int, float]]] = []

        action: tuple[int, float] | None = None
        candidate_meta: dict[str, Any] | None = None
        candidates: list[tuple[int, float]] = []
        logits = np.zeros((0,), dtype=np.float64)
        prior = np.zeros((0,), dtype=np.float64)

        # SELECT
        while not node.is_leaf(args.steps):
            candidate_meta, candidates = node_candidates(node)
            logits = compute_action_logits(node, candidates)
            prior = _softmax_prior(logits, tau=tau)
            untried = node.untried_actions(candidates)
            if len(untried) > 0:
                action = select_untried_with_optional_prior(node, candidates, prior)
                break

            if use_tree_prior:
                prior_map = {a: float(prior[i]) for i, a in enumerate(candidates)}
                action = node.best_puct(candidates, prior_map, c_puct)
            else:
                action = node.best_ucb(candidates, ucb_c)
            path.append((node, action))
            if action in node.children:
                node = node.children[action]
            else:
                break

        if action is None:
            raise RuntimeError("Tree search failed to pick an action.")

        # EXPAND
        if not node.is_leaf(args.steps):
            if action not in node.children:
                child_dx, child_lat = _expand_child(args, ctx, emb, node, action, sched)
                child_u = _compute_u_t(
                    str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
                    parent_latents=node.latents,
                    child_latents=child_lat,
                    child_dx=child_dx,
                )
                if np.isfinite(child_u) and child_u > 0.0:
                    u_values_seen.append(float(child_u))
                node.children[action] = MCTSNode(
                    step=node.step + 1,
                    dx=child_dx,
                    latents=child_lat,
                    parent=node,
                    incoming_action=action,
                    u_t=float(child_u),
                )
            path.append((node, action))
            node = node.children[action]

        # ROLLOUT
        rollout_dx = node.dx
        rollout_latents = node.latents
        rollout_step = node.step
        rollout_node = node
        while rollout_step < int(args.steps):
            r_meta, r_candidates = node_candidates(rollout_node)
            r_logits = compute_action_logits(rollout_node, r_candidates)
            r_prior = _softmax_prior(r_logits, tau=tau)
            if use_rollout_prior:
                variant_idx, cfg = sample_with_prior(r_candidates, r_prior)
            else:
                variant_idx, cfg = r_candidates[int(rng.integers(0, len(r_candidates)))]

            t_flat, t_4d = sched[rollout_step]
            flow = su.transformer_step(args, ctx, rollout_latents, emb, variant_idx, t_flat, cfg)
            rollout_dx = su._pred_x0(rollout_latents, t_4d, flow, args.x0_sampler)
            rollout_step += 1
            if rollout_step < int(args.steps):
                _, next_t_4d = sched[rollout_step]
                noise = torch.randn_like(rollout_dx)
                next_latents = (1.0 - next_t_4d) * rollout_dx + next_t_4d * noise
                child_u = _compute_u_t(
                    str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
                    parent_latents=rollout_node.latents,
                    child_latents=next_latents,
                    child_dx=rollout_dx,
                )
                if np.isfinite(child_u) and child_u > 0.0:
                    u_values_seen.append(float(child_u))
                rollout_node = MCTSNode(
                    step=rollout_step,
                    dx=rollout_dx,
                    latents=next_latents,
                    parent=rollout_node,
                    incoming_action=(int(variant_idx), float(cfg)),
                    u_t=float(child_u),
                )
                rollout_latents = next_latents

        # Score
        rollout_img = su.decode_to_pil(ctx, rollout_dx)
        rollout_score = float(su.score_image(reward_model, prompt, rollout_img))
        if rollout_score > best_global_score:
            best_global_score = float(rollout_score)
            best_global_dx = rollout_dx.clone()
            best_global_path = [a for _, a in path]

        # BACKPROP
        for pnode, paction in path:
            pnode.visits += 1
            pnode.action_visits[paction] = int(pnode.action_visits.get(paction, 0) + 1)
            pnode.action_values[paction] = float(pnode.action_values.get(paction, 0.0) + float(rollout_score))

        if (sim + 1) % log_every == 0 or sim == 0:
            history.append({
                "sim": int(sim + 1),
                "best_score": float(best_global_score),
                "root_visits": int(root.visits),
                "u_values_count": int(len(u_values_seen)),
                "u_ref": float(current_u_ref()),
            })
            print(f"    sim {sim + 1:3d}/{n_sims} best={best_global_score:.4f}")

    # --- Exploit: replay best path ---

    exploit_path: list[tuple[int, float]] = []
    node = root
    for _ in range(int(args.steps)):
        _, candidates = node_candidates(node)
        action = node.best_exploit(candidates)
        if action is None:
            break
        exploit_path.append(action)
        if action in node.children:
            node = node.children[action]
        else:
            break

    replay_dx = dx0
    replay_lat = start_latents
    for step_idx, (variant_idx, cfg) in enumerate(exploit_path):
        t_flat, t_4d = sched[step_idx]
        flow = su.transformer_step(args, ctx, replay_lat, emb, variant_idx, t_flat, cfg)
        replay_dx = su._pred_x0(replay_lat, t_4d, flow, args.x0_sampler)
        if step_idx + 1 < int(args.steps):
            _, next_t_4d = sched[step_idx + 1]
            noise = torch.randn_like(replay_dx)
            replay_lat = (1.0 - next_t_4d) * replay_dx + next_t_4d * noise

    exploit_img = su.decode_to_pil(ctx, replay_dx)
    exploit_score = float(su.score_image(reward_model, prompt, exploit_img))

    out_img = exploit_img
    out_score = float(exploit_score)
    out_actions = exploit_path
    if exploit_score < best_global_score and best_global_dx is not None:
        out_img = su.decode_to_pil(ctx, best_global_dx)
        out_score = float(best_global_score)
        out_actions = list(best_global_path)

    # Convert 2-tuple actions to 3-tuple for SearchResult compatibility
    actions_3t: list[tuple[int, float, float]] = [(int(v), float(c), 0.0) for v, c in out_actions]
    exploit_3t = [(int(v), float(c), 0.0) for v, c in exploit_path]
    best_global_3t = [(int(v), float(c), 0.0) for v, c in best_global_path]

    u_arr = np.asarray(u_values_seen, dtype=np.float64) if len(u_values_seen) > 0 else np.asarray([], dtype=np.float64)
    diagnostics = {
        "backend": "sd35_base",
        "lookahead_mode": str(getattr(args, "lookahead_mode", "rollout_tree_prior_adaptive_cfg")),
        "lookahead_flags": {
            "use_rollout_prior": bool(use_rollout_prior),
            "use_tree_prior": bool(use_tree_prior),
            "adaptive_cfg_width": bool(adaptive_cfg_width),
        },
        "mcts_cfg_mode": str(cfg_mode),
        "cfg_range": [float(cfg_min), float(cfg_max)],
        "cfg_root_bank": [float(x) for x in root_cfg_values_step0],
        "u_t_def": str(getattr(args, "lookahead_u_t_def", "latent_delta_rms")),
        "u_t_stats": {
            "count": int(u_arr.size),
            "mean": float(u_arr.mean()) if u_arr.size > 0 else 0.0,
            "std": float(u_arr.std()) if u_arr.size > 0 else 0.0,
            "min": float(u_arr.min()) if u_arr.size > 0 else 0.0,
            "max": float(u_arr.max()) if u_arr.size > 0 else 0.0,
            "u_ref_final": float(current_u_ref()),
        },
        "history": history,
        "final_cfg_trajectory": [float(a[1]) for a in out_actions],
        "exploit_cfg_trajectory": [float(a[1]) for a in exploit_path],
        "best_global_cfg_trajectory": [float(a[1]) for a in best_global_path],
    }
    return su.SearchResult(
        image=out_img,
        score=float(out_score),
        actions=actions_3t,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Entry point: run / main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Run the sampler with sd35base pipeline and integrated MCTS."""
    os.makedirs(args.out_dir, exist_ok=True)
    prompts = su.load_prompts(args)
    ctx = load_pipeline_sd35base(args)
    reward_model = su.load_reward_model(args, ctx.device)

    rewrite_cache: dict[str, list[str]] = {}
    if args.rewrites_file and os.path.exists(args.rewrites_file):
        import json
        rewrite_cache = json.load(open(args.rewrites_file))
        print(f"Loaded rewrite cache for {len(rewrite_cache)} prompts.")

    summary: list[dict[str, Any]] = []
    for prompt_idx, prompt in enumerate(prompts):
        slug = f"p{prompt_idx:02d}"
        print(f"\n{'='*72}\n[{slug}] {prompt}\n{'='*72}")
        ctx.nfe = 0
        ctx.correction_nfe = 0
        variants = su.generate_variants(args, prompt, rewrite_cache)
        with open(os.path.join(args.out_dir, f"{slug}_variants.txt"), "w") as f:
            for vi, text in enumerate(variants):
                f.write(f"v{vi}: {text}\n")
        emb = su.encode_variants(ctx, variants)

        base_img, base_score = su.run_baseline(
            args, ctx, emb, reward_model, prompt, args.seed,
            cfg_scale=float(args.baseline_cfg),
        )

        method = args.search_method
        if method == "mcts":
            search = run_mcts_sd35base(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif method == "greedy":
            search = su.run_greedy(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif method == "ga":
            search = su.run_ga(args, ctx, emb, reward_model, prompt, variants, args.seed,
                               log_dir=args.out_dir, log_prefix=slug)
        elif method == "smc":
            search = su.run_smc(args, ctx, emb, reward_model, prompt, variants, args.seed)
        elif method == "bon":
            search = su.run_bon(args, ctx, emb, reward_model, prompt, args.seed)
        elif method == "beam":
            search = su.run_beam(args, ctx, emb, reward_model, prompt, variants, args.seed)
        else:
            raise RuntimeError(f"Unsupported search_method: {method}")

        base_path = os.path.join(args.out_dir, f"{slug}_baseline.png")
        search_path = os.path.join(args.out_dir, f"{slug}_{method}.png")
        comp_path = os.path.join(args.out_dir, f"{slug}_comparison.png")
        base_img.save(base_path)
        search.image.save(search_path)
        su.save_comparison(comp_path, base_img, search.image, base_score, search.score, search.actions)

        total_nfe = ctx.nfe + ctx.correction_nfe
        print(
            f"baseline={base_score:.4f} {method}={search.score:.4f} "
            f"delta={search.score - base_score:+.4f}  nfe={total_nfe} (T:{ctx.nfe})"
        )
        summary.append({
            "slug": slug,
            "prompt": prompt,
            "variants": variants,
            "baseline_cfg": float(args.baseline_cfg),
            "baseline_IR": base_score,
            f"{method}_IR": search.score,
            "delta_IR": search.score - base_score,
            "actions": [[int(v), float(c), float(cs)] for v, c, cs in search.actions],
            "search_diagnostics": search.diagnostics,
            "nfe_transformer": ctx.nfe,
            "nfe_total": total_nfe,
        })

    import json
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for row in summary:
        print(f"{row['slug']} delta_IR={row['delta_IR']:+.4f}")
    if summary:
        print(f"mean delta={float(np.mean([r['delta_IR'] for r in summary])):+.4f}")
    print(f"summary json: {summary_path}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_extra_args(argv)
    args = _apply_sd35_base_defaults(args)
    args = su.normalize_paths(args)
    run(args)


if __name__ == "__main__":
    main()
