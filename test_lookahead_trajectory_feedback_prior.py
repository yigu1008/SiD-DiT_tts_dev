from __future__ import annotations

import numpy as np

import sampling_unified_sd35_lookahead_reweighting as lr


class _DummyChild:
    def __init__(self, u_t: float, d_t: float) -> None:
        self.u_t = float(u_t)
        self.d_t = float(d_t)


class _DummyNode:
    def __init__(self) -> None:
        self.children: dict[tuple, _DummyChild] = {}


def test_unexpanded_actions_preserve_base_ratio():
    node = _DummyNode()
    a0, a1, a2 = ("a0",), ("a1",), ("a2",)
    candidates = [a0, a1, a2]
    node.children[a0] = _DummyChild(u_t=3.0, d_t=0.1)

    base_prior = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    prior, _meta = lr.trajectory_feedback_prior(
        node,
        candidates,
        base_prior=base_prior,
        u_ref=1.0,
        d_ref=1.0,
        w_update=1.0,
        w_cond=1.0,
        tau=0.35,
    )

    # a1/a2 are both unexpanded => their relative ratio should remain unchanged.
    old_ratio = float(base_prior[1] / base_prior[2])
    new_ratio = float(prior[1] / prior[2])
    assert abs(new_ratio - old_ratio) < 1e-9

    # Expanded action should be the only one affected by feedback.
    assert abs(float(prior[0]) - float(base_prior[0])) > 1e-6


def test_all_unexpanded_matches_base_prior():
    node = _DummyNode()
    candidates = [("a0",), ("a1",), ("a2",)]
    base_prior = np.asarray([0.1, 0.7, 0.2], dtype=np.float64)
    prior, _meta = lr.trajectory_feedback_prior(
        node,
        candidates,
        base_prior=base_prior,
        u_ref=2.0,
        d_ref=2.0,
        w_update=1.2,
        w_cond=0.7,
        tau=0.5,
    )
    assert np.allclose(prior, base_prior / np.sum(base_prior), atol=1e-9, rtol=0.0)
