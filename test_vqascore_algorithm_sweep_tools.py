from __future__ import annotations

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from evaluate_best_images_multi_reward import _sd35_mode_key_and_suffix
from dynamic_cfg_x0 import DynamicCfgX0Config, evaluator_weights
from tools.merge_posthoc_reward_evals import _merge_method
from tools.check_posthoc_eval_complete import check_complete
from tools.prepare_human_eval_prompts import prepare_prompts
from tools.prepare_genai_sweep_subset import prepare


class PrepareSubsetTest(unittest.TestCase):
    def test_genai200_has_fifty_prompts_per_length_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "metadata.csv"
            fields = [
                "Index",
                "Prompt",
                "basic_skills",
                "advanced_skills",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for difficulty in ("basic", "advanced"):
                    for index in range(120):
                        writer.writerow(
                            {
                                "Index": f"{difficulty}-{index}",
                                "Prompt": (
                                    f"{difficulty} prompt {index} "
                                    + " ".join(["detail"] * (index + 1))
                                ),
                                "basic_skills": (
                                    '["scene"]' if difficulty == "basic" else "[]"
                                ),
                                "advanced_skills": (
                                    '["counting"]'
                                    if difficulty == "advanced"
                                    else "[]"
                                ),
                            }
                        )
            summary = prepare_prompts(
                source,
                root / "prompts_genai200.csv",
                num_prompts=200,
                num_reserve=16,
                seed=20260728,
            )
            self.assertEqual(summary.selected_count, 200)
            cell_counts = summary.counts_by_difficulty_and_length_quartile
            self.assertEqual(set(cell_counts.values()), {25})
            quartile_totals = {
                quartile: sum(
                    cell_counts[f"{difficulty}/q{quartile}"]
                    for difficulty in ("basic", "advanced")
                )
                for quartile in range(1, 5)
            }
            self.assertEqual(set(quartile_totals.values()), {50})
            self.assertTrue((root / "prompts_genai200_reserve.csv").is_file())

    def test_balanced_subset_and_shared_seed_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "prompts.csv"
            fields = [
                "prompt_id",
                "prompt",
                "difficulty",
                "length_quartile",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                index = 0
                for difficulty in ("basic", "advanced"):
                    for quartile in range(1, 5):
                        for _ in range(5):
                            writer.writerow(
                                {
                                    "prompt_id": f"p{index:03d}",
                                    "prompt": f"prompt {index}",
                                    "difficulty": difficulty,
                                    "length_quartile": quartile,
                                }
                            )
                            index += 1

            manifest = prepare(
                source=source,
                output_csv=root / "subset.csv",
                output_txt=root / "subset.txt",
                seed_map=root / "seeds.json",
                manifest_path=root / "manifest.json",
                subset_size=16,
                subset_seed=20260728,
                generation_base_seed=12345,
                study_id="test-study",
            )
            counts = Counter(
                (row["difficulty"], row["length_quartile"])
                for row in manifest["prompts"]
            )
            self.assertEqual(set(counts.values()), {2})
            self.assertEqual(len(manifest["exclusions"]), 24)
            seed_payload = json.loads((root / "seeds.json").read_text())
            self.assertEqual(len(seed_payload["seeds"]), 16)
            self.assertTrue(seed_payload["shared_base_seed_across_methods"])


class MergeEvaluationsTest(unittest.TestCase):
    def test_merge_distinguishes_rank_local_duplicate_slugs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_dir = root / "flux_schnell" / "run_v1" / "dynamic_cfg_x0"
            method_dir.mkdir(parents=True)
            (method_dir / "aggregate_ddp.json").write_text(
                json.dumps({"num_samples": 2, "mean_search_score": 0.8}),
                encoding="utf-8",
            )
            backends = ["imagereward", "hpsv3", "pickscore"]
            for backend_index, backend in enumerate(backends):
                rows = []
                for rank in (0, 1):
                    rows.append(
                        {
                            "prompt_index": 0,
                            "slug": "p0000",
                            "sample_index": 0,
                            "prompt": f"rank {rank} original c0",
                            "image_path": f"/tmp/rank_{rank}/p0000.png",
                            "scores": {backend: float(backend_index + rank)},
                        }
                    )
                (method_dir / f"best_images_{backend}.json").write_text(
                    json.dumps({"rows": rows}), encoding="utf-8"
                )
            summary = _merge_method(
                method_dir,
                root,
                backends,
                strict=True,
                expected_count=2,
            )
            assert summary is not None
            self.assertEqual(summary["prompt_count"], 2)
            merged = json.loads(
                (method_dir / "best_images_multi_reward.json").read_text()
            )
            self.assertEqual(len(merged["rows"]), 2)

    def test_posthoc_resume_rejects_stale_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method_dir = Path(temporary) / "dynamic_cfg_x0"
            rank_dir = method_dir / "rank_0"
            rank_dir.mkdir(parents=True)
            image_path = rank_dir / "p0000_s0_dynamic_cfg_x0.png"
            Image.new("RGB", (8, 8), "white").save(image_path)
            (rank_dir / "summary.json").write_text(
                json.dumps(
                    [
                        {
                            "slug": "p0000",
                            "prompt": "current original c0",
                            "search_method": "dynamic_cfg_x0",
                            "samples": [{"search_score": 0.8}],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            eval_path = method_dir / "best_images_imagereward.json"
            row = {
                "prompt_index": 0,
                "slug": "p0000",
                "sample_index": 0,
                "prompt": "current original c0",
                "image_path": str(image_path),
                "scores": {"imagereward": 1.0},
            }
            eval_path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
            complete, _ = check_complete(
                layout="flux",
                method_out=method_dir,
                method="dynamic_cfg_x0",
                backend="imagereward",
                eval_json=eval_path,
                expected_count=1,
            )
            self.assertTrue(complete)

            row["prompt"] = "stale prompt from a different subset"
            eval_path.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
            complete, reason = check_complete(
                layout="flux",
                method_out=method_dir,
                method="dynamic_cfg_x0",
                backend="imagereward",
                eval_json=eval_path,
                expected_count=1,
            )
            self.assertFalse(complete)
            self.assertIn("prompt mismatch", reason)

    def test_aliases_and_four_backend_merge(self) -> None:
        self.assertEqual(_sd35_mode_key_and_suffix("das"), ("bon", "bon"))
        self.assertEqual(
            _sd35_mode_key_and_suffix("fksteering"), ("smc", "smc")
        )
        self.assertEqual(_sd35_mode_key_and_suffix("dts_star"), ("mcts", "mcts"))
        dynamic_cfg = DynamicCfgX0Config(evaluators=["vqascore"])
        self.assertEqual(evaluator_weights(0.5, dynamic_cfg), {"vqascore": 1.0})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_dir = root / "sid" / "run_v1" / "das"
            (method_dir / "logs").mkdir(parents=True)
            (method_dir / "aggregate_ddp.json").write_text(
                json.dumps(
                    {
                        "num_samples": 1,
                        "mean_search_score": 0.8,
                    }
                )
            )
            (method_dir / "logs" / "rank_0.jsonl").write_text(
                json.dumps({"prompt_index": 0, "nfe": 64}) + "\n"
            )
            backends = ["imagereward", "hpsv3", "pickscore", "vqascore"]
            for index, backend in enumerate(backends):
                row = {
                    "prompt_index": 0,
                    "slug": "p00000",
                    "sample_index": 0,
                    "prompt": "original c0",
                    "image_path": "/tmp/image.png",
                    "scores": {backend: float(index + 1)},
                }
                (method_dir / f"best_images_{backend}.json").write_text(
                    json.dumps({"rows": [row]})
                )

            summary = _merge_method(
                method_dir, root, backends, strict=True, expected_count=1
            )
            assert summary is not None
            self.assertEqual(summary["method_label"], "DAS")
            self.assertEqual(summary["mean_nfe"], 64.0)
            merged = json.loads(
                (method_dir / "best_images_multi_reward.json").read_text()
            )
            self.assertEqual(
                set(merged["rows"][0]["scores"]),
                set(backends),
            )


if __name__ == "__main__":
    unittest.main()
