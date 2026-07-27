from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.summarize_inference_costs import _candidate_account, summarize


class InferenceCostSummaryTest(unittest.TestCase):
    def test_sampled_mcts_history_is_not_treated_as_complete(self) -> None:
        account = _candidate_account(
            {
                "mode": "mcts",
                "search_diagnostics": {
                    "history": [{"sim": 1}, {"sim": 10}, {"sim": 20}],
                },
            },
            "mcts",
        )
        self.assertIsNone(account["terminal_candidates_total"])
        self.assertFalse(account["accounting_exact"])

    def test_sparse_refinement_is_left_unknown(self) -> None:
        account = _candidate_account(
            {
                "mode": "mcts",
                "search_diagnostics": {
                    "sparse_noise_refine": {
                        "enabled": True,
                        "rollouts_evaluated": 3,
                    },
                    "bon_mcts": {
                        "prescreen_ranked": [{"seed": i} for i in range(8)],
                        "tree_refine": [
                            {"n_sims_used": 13},
                            {"n_sims_used": 12},
                        ],
                    },
                },
            },
            "bon_mcts",
        )
        self.assertIsNone(account["objective_evaluations"])
        self.assertFalse(account["accounting_exact"])

    def test_sid_three_reward_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "study_manifest_v1.json").write_text(
                json.dumps(
                    {
                        "prompt_variant_control": {
                            "use_qwen": False,
                            "shared_rewrites_file": str(root / "rewrites.json"),
                        }
                    }
                ),
                encoding="utf-8",
            )
            for arm in ("imagereward", "vqascore", "ir_vqa_equal"):
                method = root / arm / "run_v1" / "bon_mcts"
                logs = method / "logs"
                logs.mkdir(parents=True)
                rows = []
                for prompt_index in range(2):
                    rows.append(
                        {
                            "prompt_index": prompt_index,
                            "prompt": f"prompt {prompt_index}",
                            "mode": "mcts",
                            "score": 1.0,
                            "nfe": 100,
                            "search_diagnostics": {
                                "integrated_noise_actions": True,
                                "sparse_noise_refine": None,
                                "bon_mcts": {
                                    "prescreen_ranked": [{"seed": i} for i in range(8)],
                                    "tree_refine": [
                                        {"n_sims_used": 13},
                                        {"n_sims_used": 12},
                                    ],
                                },
                            },
                        }
                    )
                (logs / "rank_000.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                (method / "aggregate_ddp.json").write_text(
                    json.dumps({"elapsed_sec": 3600}),
                    encoding="utf-8",
                )

            summary = summarize(
                root,
                root / "summary",
                generation_gpus=3,
                reward_gpus=1,
                generation_gpu_hour_price=None,
                reward_gpu_hour_price=None,
                memory_summary=None,
            )
            aggregate = summary["aggregate"]
            self.assertEqual(aggregate["terminal_candidates_total"], 216)
            self.assertEqual(aggregate["objective_evaluations"], 216)
            self.assertEqual(
                aggregate["reward_backend_calls"],
                {"imagereward": 144, "vqascore": 144},
            )
            self.assertEqual(aggregate["generator_nfe_logged"], 600)
            self.assertEqual(aggregate["total_gpu_hours"], 12.0)
            self.assertEqual(summary["prompt_rewriter"]["calls"], 0)


if __name__ == "__main__":
    unittest.main()
