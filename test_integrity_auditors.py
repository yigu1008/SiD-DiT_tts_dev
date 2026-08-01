from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parent
HUMAN_AUDITOR = REPO / "tools" / "audit_human_eval_integrity.py"
BON_AUDITOR = REPO / "tools" / "audit_hpsv2_bon8_results.py"


class IntegrityAuditorTest(unittest.TestCase):
    def test_human_eval_package_and_reference_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "human_eval"
            root.mkdir()
            with (root / "prompts.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt_id", "prompt"])
                writer.writeheader()
                writer.writerows([
                    {"prompt_id": "p000", "prompt": "A red cube"},
                    {"prompt_id": "p001", "prompt": "A blue sphere"},
                ])
            for prompt_index in range(2):
                folder = root / "images" / "sid" / f"p{prompt_index:03d}"
                folder.mkdir(parents=True)
                for method_index, method in enumerate(("baseline", "bon", "das", "actdiff")):
                    Image.new(
                        "RGB", (8, 8), (prompt_index * 50, method_index * 50, 20)
                    ).save(folder / f"{method}.png")
            config = root / "config.yaml"
            config.write_text(
                "\n".join([
                    "expected_model_count: 1",
                    "expected_prompt_count: 2",
                    "models:",
                    "  - {id: sid, display_name: SiD-SD3.5}",
                    "method_files:",
                    "  baseline: baseline.png",
                    "  bon: bon.png",
                    "  das: das.png",
                    "  actdiff: actdiff.png",
                    "paths:",
                    "  tasks: tasks/pairwise_tasks.jsonl",
                    "  responses: responses/responses.csv",
                ]) + "\n",
                encoding="utf-8",
            )
            first = subprocess.run(
                [sys.executable, str(HUMAN_AUDITOR), "--root", str(root), "--config", str(config)],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("images=8/8", first.stdout)
            inventory = root / "integrity_reports/human_eval_base/image_inventory.csv"
            second = subprocess.run(
                [
                    sys.executable, str(HUMAN_AUDITOR), "--root", str(root),
                    "--config", str(config), "--reference-inventory", str(inventory),
                ],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("[human-eval] OK", second.stdout)

    def test_human_eval_detects_stale_relocated_task_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "prompts.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt_id", "prompt"])
                writer.writeheader()
                writer.writerow({"prompt_id": "p000", "prompt": "Prompt"})
            folder = root / "images/sid/p000"
            folder.mkdir(parents=True)
            for index, method in enumerate(("baseline", "bon", "das", "actdiff")):
                Image.new("RGB", (8, 8), (index * 60, 0, 0)).save(folder / f"{method}.png")
            config = root / "config.yaml"
            config.write_text(
                "expected_model_count: 1\nexpected_prompt_count: 1\n"
                "models:\n  - {id: sid}\nmethod_files:\n"
                "  baseline: baseline.png\n  bon: bon.png\n  das: das.png\n  actdiff: actdiff.png\n"
                "paths:\n  tasks: tasks/pairwise_tasks.jsonl\n",
                encoding="utf-8",
            )
            tasks = root / "tasks/pairwise_tasks.jsonl"
            tasks.parent.mkdir()
            rows = []
            for competitor in ("baseline", "bon", "das"):
                rows.append({
                    "task_id": f"task_{competitor}", "model_id": "sid",
                    "prompt_id": "p000", "prompt": "Prompt",
                    "left_image": f"image_l_{competitor}",
                    "right_image": f"image_r_{competitor}",
                    "left_method": "actdiff", "right_method": competitor,
                    "competitor": competitor,
                    "left_image_path": "/old/server/images/sid/p000/actdiff.png",
                    "right_image_path": f"/old/server/images/sid/p000/{competitor}.png",
                })
            tasks.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = subprocess.run(
                [sys.executable, str(HUMAN_AUDITOR), "--root", str(root), "--config", str(config)],
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            report = json.loads(
                (root / "integrity_reports/human_eval_base/integrity_report.json").read_text()
            )
            self.assertIn("stale_task_image_path", {row["kind"] for row in report["errors"]})

    def test_complete_bon8_cell(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompt = "A red cube"
            (root / "study_manifest.json").write_text(json.dumps({
                "run_id": "v1", "prompt_count": 1, "candidate_count": 8,
                "rewrite_count_per_prompt": 1,
                "models": [{"model_id": "sid"}],
                "reward_arms": {"imagereward": {}},
            }))
            with (root / "prompts.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["prompt_id", "prompt"])
                writer.writeheader()
                writer.writerow({"prompt_id": "hpsv2_0000", "prompt": prompt})
            (root / "fixed_rewrite_cache.json").write_text(
                json.dumps({prompt: ["A carefully lit red cube"]})
            )
            seeds = root / "seed_maps"
            seeds.mkdir()
            (seeds / "sid.json").write_text(json.dumps({"seeds": {"0": 123}}))
            method = root / "sid/imagereward/run_v1/bon_fixed_rewrite"
            logs = method / "logs"
            rank = method / "rank_0"
            logs.mkdir(parents=True)
            rank.mkdir(parents=True)
            (method / "aggregate_ddp.json").write_text(json.dumps({
                "num_samples": 1, "search_reward": "imagereward",
            }))
            (logs / "rank_0.jsonl").write_text(json.dumps({
                "prompt_index": 0, "prompt": prompt, "mode": "bon",
                "search_diagnostics": {"bon_n": 8},
            }) + "\n")
            (rank / "p0000_variants.txt").write_text(
                "fixed_rewrite: A carefully lit red cube\n"
            )
            image = method / "winner.png"
            Image.new("RGB", (8, 8), (255, 0, 0)).save(image)
            row = {
                "prompt_index": 0, "slug": "p0000", "sample_index": 0,
                "prompt": prompt, "image_path": str(image),
            }
            for backend in ("imagereward", "hpsv3", "pickscore", "hpsv2"):
                backend_row = {**row, "scores": {backend: 1.0}}
                (method / f"best_images_{backend}.json").write_text(
                    json.dumps({"rows": [backend_row]})
                )
            result = subprocess.run(
                [sys.executable, str(BON_AUDITOR), "--root", str(root)],
                text=True, capture_output=True, check=True,
            )
            self.assertIn("cells=1/1", result.stdout)
            report = json.loads(
                (root / "integrity_reports/bon8/bon8_integrity_report.json").read_text()
            )
            self.assertTrue(report["complete"])


if __name__ == "__main__":
    unittest.main()
