"""Unit + smoke tests for dynamic_cfg_x0.

Run with:  python -m pytest test_dynamic_cfg_x0.py -v
       or: python test_dynamic_cfg_x0.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

import dynamic_cfg_x0 as dcx


# ── Progress ─────────────────────────────────────────────────────────────


def test_progress_from_sigma_endpoints():
    assert abs(dcx.progress_from_sigma(1.0, 0.0, 1.0) - 0.0) < 1e-6
    assert abs(dcx.progress_from_sigma(0.0, 0.0, 1.0) - 1.0) < 1e-6
    assert abs(dcx.progress_from_sigma(0.5, 0.0, 1.0) - 0.5) < 1e-6


def test_progress_from_sigma_zero_span():
    # Degenerate schedule should not crash
    assert dcx.progress_from_sigma(0.5, 0.5, 0.5) == 0.0


def test_progress_from_step():
    assert abs(dcx.progress_from_step(0, 4) - 0.0) < 1e-6
    assert abs(dcx.progress_from_step(3, 4) - 1.0) < 1e-6


# ── Should-score gating ──────────────────────────────────────────────────


def test_should_score_step_gating():
    cfg = dcx.DynamicCfgX0Config(
        enabled=True, score_start_frac=0.25, score_end_frac=0.95, score_every=2,
    )
    # Disabled
    cfg.enabled = False
    assert not dcx.should_score_step(0, 0.5, cfg)
    cfg.enabled = True
    # Before window
    assert not dcx.should_score_step(0, 0.10, cfg)
    # In window, on stride
    assert dcx.should_score_step(2, 0.50, cfg)
    # In window, off stride
    assert not dcx.should_score_step(3, 0.50, cfg)
    # After window
    assert not dcx.should_score_step(8, 0.99, cfg)


# ── Candidate generation ─────────────────────────────────────────────────


def test_generate_cfg_candidates_dedup_clamp():
    cfg = dcx.DynamicCfgX0Config(
        cfg_grid=[1.0, 2.0, 2.0, 5.0, 9.5],
        cfg_min=0.0, cfg_max=8.0,
        add_local_neighborhood=False,
    )
    out = dcx.generate_cfg_candidates(cfg, w_prev=None)
    assert out == [1.0, 2.0, 5.0, 8.0]


def test_generate_cfg_candidates_neighborhood():
    cfg = dcx.DynamicCfgX0Config(
        cfg_grid=[2.0, 5.0],
        cfg_min=0.0, cfg_max=10.0,
        add_local_neighborhood=True,
        neighborhood_deltas=[-0.5, 0.5],
    )
    out = dcx.generate_cfg_candidates(cfg, w_prev=4.0)
    # Includes 3.5, 4.5 around w_prev=4.0
    assert 3.5 in out and 4.5 in out
    # Sorted
    assert out == sorted(out)


# ── Evaluator weights ────────────────────────────────────────────────────


def test_evaluator_weights_renormalize_after_filter():
    cfg = dcx.DynamicCfgX0Config(
        evaluators=["imagereward", "hpsv3"],  # spec also has "special" but not enabled
        prompt_type="text",
        weight_schedule="piecewise",
    )
    w = dcx.evaluator_weights(progress=0.5, cfg=cfg)
    assert set(w.keys()) == {"imagereward", "hpsv3"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    # Late progress: hpsv3 should outweigh imagereward in 'general'/'text'
    cfg.prompt_type = "general"
    w_late = dcx.evaluator_weights(progress=0.9, cfg=cfg)
    assert w_late["hpsv3"] > w_late["imagereward"]


def test_evaluator_weights_smooth_interpolates():
    cfg = dcx.DynamicCfgX0Config(
        evaluators=["imagereward", "hpsv3"],
        prompt_type="general",
        weight_schedule="smooth",
    )
    w0 = dcx.evaluator_weights(progress=0.0, cfg=cfg)
    w1 = dcx.evaluator_weights(progress=1.0, cfg=cfg)
    assert w0["imagereward"] > w0["hpsv3"]
    assert w1["hpsv3"] > w1["imagereward"]
    # midpoint between 0.15 and 0.50 should produce a value between band 0 and band 1
    w_mid = dcx.evaluator_weights(progress=0.325, cfg=cfg)
    assert 0.30 < w_mid["imagereward"] < 0.75


# ── Z-score normalization ────────────────────────────────────────────────


def test_zscore_normalize_mean_zero_unit_std():
    raw = {1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 4.0: 4.0}
    out = dcx.zscore_normalize(raw)
    arr = np.array(list(out.values()))
    assert abs(arr.mean()) < 1e-3
    assert abs(arr.std() - 1.0) < 1e-2


def test_zscore_normalize_zero_when_constant():
    raw = {1.0: 0.5, 2.0: 0.5, 3.0: 0.5}
    out = dcx.zscore_normalize(raw)
    assert all(abs(v) < 1e-6 for v in out.values())


# ── Confidence gating ────────────────────────────────────────────────────


def test_confidence_gating_drops_constant_evaluator():
    raw = {
        "imagereward": {1.0: 0.5, 2.0: 0.5, 3.0: 0.5},  # zero std → dropped
        "hpsv3":       {1.0: 0.1, 2.0: 0.5, 3.0: 0.9},  # nonzero std → kept
    }
    base = {"imagereward": 0.5, "hpsv3": 0.5}
    gated = dcx.confidence_gate(raw, base)
    assert gated["hpsv3"] > 0.99
    assert gated["imagereward"] < 0.01


def test_confidence_gating_falls_back_when_all_constant():
    raw = {
        "imagereward": {1.0: 0.5, 2.0: 0.5},
        "hpsv3":       {1.0: 0.7, 2.0: 0.7},
    }
    base = {"imagereward": 0.4, "hpsv3": 0.6}
    gated = dcx.confidence_gate(raw, base)
    # Falls back to base
    assert abs(gated["imagereward"] - 0.4) < 1e-6
    assert abs(gated["hpsv3"] - 0.6) < 1e-6


# ── Combine + select ─────────────────────────────────────────────────────


def test_combine_and_select_simple():
    cands = [2.0, 5.0, 8.0]
    norm = {
        "imagereward": {2.0: -1.0, 5.0: 1.0, 8.0: 0.0},
        "hpsv3":       {2.0:  0.0, 5.0: 0.5, 8.0: -0.5},
    }
    weights = {"imagereward": 0.5, "hpsv3": 0.5}
    cfg = dcx.DynamicCfgX0Config(cfg_smooth_weight=0.0, high_cfg_penalty=0.0)
    totals = dcx.combine_scores(norm, weights, cands, w_prev=None, cfg=cfg)
    best = dcx.select_best_cfg(totals)
    assert best == 5.0


def test_combine_high_cfg_penalty():
    cands = [5.0, 9.0]
    norm = {"imagereward": {5.0: 0.0, 9.0: 0.5}}
    weights = {"imagereward": 1.0}
    cfg = dcx.DynamicCfgX0Config(cfg_smooth_weight=0.0, high_cfg_penalty=10.0, cfg_soft_max=7.5)
    totals = dcx.combine_scores(norm, weights, cands, w_prev=None, cfg=cfg)
    # 9.0 should be penalized below 5.0
    assert totals[5.0] > totals[9.0]


def test_combine_smoothness_bias():
    cands = [3.0, 5.0]
    norm = {"imagereward": {3.0: 0.0, 5.0: 0.0}}  # tied
    weights = {"imagereward": 1.0}
    cfg = dcx.DynamicCfgX0Config(cfg_smooth_weight=0.5, high_cfg_penalty=0.0)
    totals = dcx.combine_scores(norm, weights, cands, w_prev=3.0, cfg=cfg)
    # When tied on raw, smoothness should pick the nearer to w_prev
    assert dcx.select_best_cfg(totals) == 3.0


# ── x0_pred ──────────────────────────────────────────────────────────────


def test_x0_pred_from_flow_flow_model():
    latents = torch.ones(1, 2, 2, 2)
    sigma = torch.full((1, 1, 1, 1), 0.5)
    flow = torch.full_like(latents, 2.0)
    out = dcx.x0_pred_from_flow(latents, sigma, flow, x0_sampler=False)
    # x0 = latents - sigma * flow = 1 - 0.5*2 = 0
    assert torch.allclose(out, torch.zeros_like(latents))


def test_x0_pred_from_flow_x0_model():
    latents = torch.ones(1, 2, 2, 2)
    sigma = torch.full((1, 1, 1, 1), 0.5)
    flow = torch.full_like(latents, 0.7)
    out = dcx.x0_pred_from_flow(latents, sigma, flow, x0_sampler=True)
    # In x0-prediction mode, flow IS x0
    assert torch.allclose(out, flow)


# ── End-to-end smoke (dummy reward) ──────────────────────────────────────


def test_smoke_dummy_reward_picks_target_cfg():
    """Rig eval_fn to return r = -|w - 5.0|. The selector should choose 5.0."""
    cfg = dcx.DynamicCfgX0Config(
        enabled=True,
        cfg_grid=[2.0, 3.5, 5.0, 6.5, 8.0],
        evaluators=["imagereward"],
        weight_schedule="piecewise",
        prompt_type="general",
        confidence_gating=False,
        cfg_smooth_weight=0.0, high_cfg_penalty=0.0,
        cfg_min=0.0, cfg_max=10.0,
    )

    # Build a tiny synthetic problem.
    latents = torch.zeros(1, 1, 2, 2)
    flow_u = torch.zeros_like(latents)
    flow_c = torch.ones_like(latents)
    sigma = torch.full((1, 1, 1, 1), 1.0)

    # Stash the recombined CFG in image dims (PIL trick: white pixel count proxy).
    # We use a closure that reads back the chosen w from the candidate index.
    seen_w_in_decode: list[float] = []

    def decode_fn(x0: torch.Tensor) -> Image.Image:
        # x0 = latents - sigma*flow_w. With latents=0, sigma=1, flow_u=0, flow_c=1:
        #   x0 = -w
        seen_w_in_decode.append(float(-x0.mean().item()))
        return Image.new("RGB", (4, 4), color=(0, 0, 0))

    def eval_fn(evaluator: str, prompt: str, image: Image.Image) -> float:
        # Use the most recent w pushed into the decode trace
        w = seen_w_in_decode[-1]
        return -abs(w - 5.0)

    result = dcx.select_cfg_for_step(
        candidates=cfg.cfg_grid,
        flow_u=flow_u,
        flow_c=flow_c,
        latents=latents,
        sigma_4d=sigma,
        x0_sampler=False,
        decode_fn=decode_fn,
        eval_fn=eval_fn,
        prompt="test prompt",
        progress=0.5,
        w_prev=None,
        cfg=cfg,
    )
    assert result["chosen_cfg"] == 5.0


# ── Logger ───────────────────────────────────────────────────────────────


def test_logger_appends_jsonl():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sub", "log.jsonl")
        lg = dcx.DynamicCfgLogger(path)
        lg.log({"step": 0, "chosen_cfg": 5.0})
        lg.log({"step": 1, "chosen_cfg": 3.5})
        lg.close()
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        assert len(lines) == 2
        import json
        parsed = [json.loads(ln) for ln in lines]
        assert parsed[0]["step"] == 0
        assert parsed[1]["chosen_cfg"] == 3.5


# ── Manual runner ────────────────────────────────────────────────────────


if __name__ == "__main__":
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    if failed:
        print(f"\n{failed}/{len(funcs)} tests failed.")
        sys.exit(1)
    print(f"\nAll {len(funcs)} tests passed.")
