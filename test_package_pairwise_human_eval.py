from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from tools.package_pairwise_human_eval import package_pairwise


class PairwiseHumanEvalPackageTest(unittest.TestCase):
    def _fixture(self, root: Path, *, omit: tuple[str, str, str] | None = None) -> Path:
        manifest = root / "legacy_summary" / "legacy_manifest.csv"
        manifest.parent.mkdir(parents=True)
        fields = [
            "image_id",
            "group_id",
            "model_id",
            "model_name",
            "algorithm_id",
            "prompt_id",
            "source_id",
            "difficulty",
            "original_prompt_c0",
            "logged_root_seed",
            "selected_candidate_seed",
            "planned_method_seed",
            "total_nfe",
            "reward_scores_json",
            "actions_json",
            "summary_image_path",
            "source_image_path",
            "status",
        ]
        rows = []
        algorithms = ("baseline", "bon", "das", "bon_mcts")
        for model_number, model in enumerate(("sid", "flux_schnell")):
            for prompt_number in range(3):
                prompt_id = f"p{prompt_number:03d}"
                group_id = f"{model}__{prompt_id}"
                for algorithm_number, algorithm in enumerate(algorithms):
                    if omit == (model, prompt_id, algorithm):
                        continue
                    image = root / "images" / f"{model}_{prompt_id}_{algorithm}.png"
                    image.parent.mkdir(exist_ok=True)
                    color = (
                        20 + model_number * 80,
                        20 + prompt_number * 60,
                        20 + algorithm_number * 50,
                    )
                    Image.new("RGB", (24, 24), color).save(image)
                    rows.append(
                        {
                            "image_id": f"{group_id}__{algorithm}",
                            "group_id": group_id,
                            "model_id": model,
                            "model_name": model,
                            "algorithm_id": algorithm,
                            "prompt_id": prompt_id,
                            "source_id": f"source-{prompt_id}",
                            "difficulty": "test",
                            "original_prompt_c0": f"prompt {prompt_number}",
                            "logged_root_seed": 100 + prompt_number,
                            "selected_candidate_seed": 200 + algorithm_number,
                            "planned_method_seed": 100 + prompt_number,
                            "total_nfe": 4 + algorithm_number,
                            "reward_scores_json": '{"imagereward":0.5}',
                            "actions_json": "[]",
                            "summary_image_path": str(image),
                            "source_image_path": str(image),
                            "status": "complete",
                        }
                    )
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def test_packages_blinded_balanced_reproducible_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root)
            first = root / "package_one"
            second = root / "package_two"
            result = package_pairwise(manifest, first, seed=42)
            package_pairwise(manifest, second, seed=42)

            self.assertEqual(result["trial_count"], 18)
            self.assertEqual(result["image_count"], 36)
            public_one = (first / "public" / "pairwise_trials.csv").read_text()
            public_two = (second / "public" / "pairwise_trials.csv").read_text()
            private_one = (first / "private" / "pairwise_answer_key.csv").read_text()
            private_two = (second / "private" / "pairwise_answer_key.csv").read_text()
            self.assertEqual(public_one, public_two)
            self.assertEqual(private_one, private_two)
            self.assertNotIn("algorithm", public_one.splitlines()[0])
            self.assertNotIn("reward", public_one.splitlines()[0])

            with (first / "private" / "pairwise_answer_key.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                keys = list(csv.DictReader(handle))
            for row in keys:
                self.assertEqual(
                    row["image_id_a"].rsplit("__", 1)[0],
                    row["image_id_b"].rsplit("__", 1)[0],
                )
                self.assertEqual(row["same_logged_root_seed"], "true")

            counts = Counter(
                (row["model_id"], row["comparison_id"], row["algorithm_a"])
                for row in keys
            )
            for model in ("sid", "flux_schnell"):
                for opponent in ("baseline", "bon", "das"):
                    comparison = f"bon_mcts_vs_{opponent}"
                    actdiff_a = counts[(model, comparison, "bon_mcts")]
                    opponent_a = counts[(model, comparison, opponent)]
                    self.assertLessEqual(abs(actdiff_a - opponent_a), 1)
            self.assertEqual(len(list((first / "public" / "images").iterdir())), 36)

    def test_incomplete_is_strict_by_default_and_logged_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._fixture(root, omit=("sid", "p000", "das"))
            with self.assertRaises(RuntimeError):
                package_pairwise(manifest, root / "strict", seed=1)
            result = package_pairwise(
                manifest,
                root / "relaxed",
                seed=1,
                allow_incomplete=True,
            )
            self.assertEqual(result["trial_count"], 17)
            self.assertEqual(result["excluded_trial_count"], 1)
            exclusions = (
                root / "relaxed" / "private" / "excluded_trials.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("sid__p000", exclusions)


if __name__ == "__main__":
    unittest.main()
