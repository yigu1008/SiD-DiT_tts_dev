from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.summarize_human_eval_legacy import summarize


class LegacySummaryTest(unittest.TestCase):
    def test_collects_sd35_and_flux_with_older_run_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "prompts.csv"
            with prompts.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["prompt_id", "source_id", "prompt", "difficulty"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "prompt_id": "p000",
                        "source_id": "g0",
                        "prompt": "a red circle",
                        "difficulty": "basic",
                    }
                )
                writer.writerow(
                    {
                        "prompt_id": "p001",
                        "source_id": "g1",
                        "prompt": "a blue square",
                        "difficulty": "advanced",
                    }
                )
            config = root / "config.yaml"
            config.write_text(
                """
models:
  - model_id: sid
  - model_id: flux_schnell
algorithms:
  - baseline
""".lstrip(),
                encoding="utf-8",
            )

            seed_dir = root / "legacy_runs" / "_seed_maps"
            for model in ("sid", "flux_schnell"):
                path = seed_dir / model / "baseline.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"seeds": {"0": 10, "1": 11}}),
                    encoding="utf-8",
                )

            old_sid = (
                root / "legacy_runs" / "sid" / "baseline" / "run_001" / "baseline"
            )
            (old_sid / "images").mkdir(parents=True)
            (old_sid / "logs").mkdir()
            Image.new("RGB", (32, 32), "red").save(old_sid / "images" / "p00000_base.png")
            Image.new("RGB", (32, 32), "blue").save(old_sid / "images" / "p00001_base.png")
            with (old_sid / "logs" / "rank_000.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "prompt_index": 0,
                            "prompt": "a red circle",
                            "seed": 10,
                            "mode": "base",
                            "score": 0.1,
                            "nfe": 4,
                        }
                    )
                    + "\n"
                )
                handle.write(
                    json.dumps(
                        {
                            "prompt_index": 1,
                            "prompt": "a blue square",
                            "seed": 11,
                            "mode": "base",
                            "score": 0.2,
                            "nfe": 4,
                        }
                    )
                    + "\n"
                )
            (old_sid / "best_images_multi_reward.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "prompt_index": 0,
                                "prompt": "rewritten red circle",
                                "scores": {"imagereward": 9.0},
                            },
                            {
                                "prompt_index": 0,
                                "prompt": "a red circle",
                                "scores": {"imagereward": 0.5},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            # Newer run has only p000; p001 must deterministically fall back.
            new_sid = (
                root / "legacy_runs" / "sid" / "baseline" / "run_002" / "baseline"
            )
            (new_sid / "images").mkdir(parents=True)
            (new_sid / "logs").mkdir()
            Image.new("RGB", (32, 32), "orange").save(
                new_sid / "images" / "p00000_base.png"
            )
            (new_sid / "logs" / "rank_000.jsonl").write_text(
                json.dumps(
                    {
                        "prompt_index": 0,
                        "prompt": "a red circle",
                        "seed": 10,
                        "mode": "base",
                        "score": 0.3,
                        "nfe": 4,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (new_sid / "best_images_multi_reward.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "prompt_index": 0,
                                "prompt": "rewritten red circle",
                                "scores": {"imagereward": 9.0},
                            },
                            {
                                "prompt_index": 0,
                                "prompt": "a red circle",
                                "scores": {"imagereward": 0.5},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            flux = (
                root
                / "legacy_runs"
                / "flux_schnell"
                / "baseline"
                / "run_001"
                / "baseline"
                / "rank_0"
            )
            flux.mkdir(parents=True)
            entries = []
            for index, (prompt, color) in enumerate(
                (("a red circle", "red"), ("a blue square", "blue"))
            ):
                slug = f"p{index:04d}"
                Image.new("RGB", (32, 32), color).save(
                    flux / f"{slug}_s0_baseline.png"
                )
                entries.append(
                    {
                        "slug": slug,
                        "prompt": prompt,
                        "search_method": "smc",
                        "samples": [
                            {
                                "seed": 10 + index,
                                "baseline_score": 0.1 + index,
                                "search_score": 0.1 + index,
                                "delta_score": 0.0,
                            }
                        ],
                    }
                )
            (flux / "summary.json").write_text(json.dumps(entries), encoding="utf-8")

            result = summarize(
                root,
                root / "summary",
                config,
                prompts,
                materialize="symlink",
            )
            self.assertEqual(result["status_counts"], {"complete": 4})
            payload = json.loads(
                (root / "summary" / "legacy_manifest.json").read_text(encoding="utf-8")
            )
            records = {row["image_id"]: row for row in payload["records"]}
            self.assertTrue(
                records["sid__p000__baseline"]["source_run_path"].endswith("run_002")
            )
            self.assertTrue(
                records["sid__p001__baseline"]["source_run_path"].endswith("run_001")
            )
            self.assertEqual(
                records["sid__p000__baseline"]["reward_scores"]["imagereward"], 0.5
            )
            self.assertTrue(
                payload["prompt_audit"]["all_models_cover_expected_prompt_set"]
            )
            self.assertTrue(
                Path(records["sid__p000__baseline"]["summary_image_path"]).is_file()
            )
            with (root / "summary" / "legacy_groups.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                groups = list(csv.DictReader(handle))
            self.assertEqual(len(groups), 4)
            self.assertTrue(all(row["complete"] == "True" for row in groups))


if __name__ == "__main__":
    unittest.main()
