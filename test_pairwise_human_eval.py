from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.pairwise_human_eval import (
    ANNOTATION_HTML,
    build_tasks,
    compute_winrates,
    import_legacy,
    validate_inputs,
)


class PairwiseHumanEvalTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        prompts = root / "prompts.csv"
        with prompts.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["prompt_id", "prompt"])
            writer.writeheader()
            for index in range(40):
                writer.writerow(
                    {"prompt_id": f"p{index:03d}", "prompt": f"Prompt {index}"}
                )
        models = [f"model_{index}" for index in range(5)]
        for model_index, model in enumerate(models):
            for prompt_index in range(40):
                folder = root / "images" / model / f"p{prompt_index:03d}"
                folder.mkdir(parents=True)
                for method_index, method in enumerate(
                    ("baseline", "bon", "das", "actdiff")
                ):
                    Image.new(
                        "RGB",
                        (16, 16),
                        (
                            model_index * 30,
                            prompt_index * 5,
                            method_index * 60,
                        ),
                    ).save(folder / f"{method}.png")
        config = root / "config.yaml"
        config.write_text(
            "\n".join(
                [
                    f"study_root: {root}",
                    "random_seed: 7",
                    "annotator_id: tester",
                    "expected_model_count: 5",
                    "expected_prompt_count: 40",
                    "models:",
                    *[
                        f"  - {{id: {model}, display_name: Model {index}}}"
                        for index, model in enumerate(models)
                    ],
                    "method_files:",
                    "  baseline: baseline.png",
                    "  bon: bon.png",
                    "  das: das.png",
                    "  actdiff: actdiff.png",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    def test_builds_exactly_600_balanced_blinded_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            self.assertTrue(validate_inputs(config)["valid"])
            report = build_tasks(config)
            self.assertEqual(report["task_count"], 600)
            tasks_path = root / "tasks" / "pairwise_tasks.jsonl"
            first = tasks_path.read_text(encoding="utf-8")
            tasks = [json.loads(line) for line in first.splitlines()]
            self.assertEqual(
                {task["competitor"] for task in tasks},
                {"baseline", "bon", "das"},
            )
            for counts in report["side_counts"].values():
                self.assertEqual(counts, {"left": 20, "right": 20})
            build_tasks(config, overwrite=True)
            self.assertEqual(first, tasks_path.read_text(encoding="utf-8"))
            self.assertNotIn("actdiff", ANNOTATION_HTML.lower())
            self.assertNotIn("bon_mcts", ANNOTATION_HTML.lower())

    def test_winrate_uses_ties_and_excludes_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            build_tasks(config)
            tasks = [
                json.loads(line)
                for line in (root / "tasks" / "pairwise_tasks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            responses = root / "responses" / "responses.csv"
            responses.parent.mkdir()
            with responses.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["annotator_id", "task_id", "choice", "timestamp"],
                )
                writer.writeheader()
                for index, task in enumerate(tasks):
                    if index % 4 == 0:
                        choice = (
                            "left"
                            if task["left_method"] == "actdiff"
                            else "right"
                        )
                    elif index % 4 == 1:
                        choice = (
                            "right"
                            if task["left_method"] == "actdiff"
                            else "left"
                        )
                    elif index % 4 == 2:
                        choice = "tie"
                    else:
                        choice = "skip"
                    writer.writerow(
                        {
                            "annotator_id": "tester",
                            "task_id": task["task_id"],
                            "choice": choice,
                            "timestamp": str(index),
                        }
                    )
            result = compute_winrates(config)
            self.assertEqual(result["answered_tasks"], 600)
            self.assertEqual(result["valid_responses"], 450)
            self.assertEqual(result["skips"], 150)
            with (root / "results" / "winrates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            aggregate = next(
                row
                for row in rows
                if row["model"] == "Aggregate" and row["comparison"] == "overall"
            )
            self.assertEqual(aggregate["win_rate"], "0.500000")
            markdown = (root / "results" / "winrates.md").read_text()
            self.assertIn("| Model | vs. baseline | vs. BoN | vs. DAS | Overall |", markdown)

    def test_imports_legacy_bon_mcts_as_actdiff_without_renaming_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            source_root = root / "legacy_sources"
            (root / "images").rename(source_root)
            manifest = root / "legacy_manifest.csv"
            fields = [
                "image_id",
                "model_id",
                "prompt_id",
                "algorithm_id",
                "original_prompt_c0",
                "source_image_path",
                "summary_image_path",
                "status",
            ]
            rows = []
            for model_index in range(5):
                model = f"model_{model_index}"
                for prompt_index in range(40):
                    prompt_id = f"p{prompt_index:03d}"
                    for method in ("baseline", "bon", "das", "actdiff"):
                        legacy_method = "bon_mcts" if method == "actdiff" else method
                        source = source_root / model / prompt_id / f"{method}.png"
                        rows.append(
                            {
                                "image_id": f"{model}__{prompt_id}__{legacy_method}",
                                "model_id": model,
                                "prompt_id": prompt_id,
                                "algorithm_id": legacy_method,
                                "original_prompt_c0": f"Prompt {prompt_index}",
                                "source_image_path": str(source),
                                "summary_image_path": "",
                                "status": "complete",
                            }
                        )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            source_actdiff = source_root / "model_0" / "p000" / "actdiff.png"
            original_digest = source_actdiff.read_bytes()
            report = import_legacy(config, manifest)
            self.assertTrue(report["valid"])
            self.assertEqual(report["imported_images"], 800)
            self.assertTrue(root.joinpath("images/model_0/p000/actdiff.png").is_file())
            self.assertFalse(root.joinpath("images/model_0/p000/bon_mcts.png").exists())
            self.assertEqual(source_actdiff.read_bytes(), original_digest)
            self.assertTrue(validate_inputs(config)["valid"])


if __name__ == "__main__":
    unittest.main()
