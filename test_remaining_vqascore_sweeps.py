from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent
RUNNER = REPO / "tools" / "run_remaining_vqascore_algorithm_sweeps.sh"
AUDITOR = REPO / "tools" / "audit_vqascore_sweep_coverage.py"


class RemainingVqaSweepsTest(unittest.TestCase):
    def test_dry_run_reuses_sid_subset_and_excludes_sid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sid_source"
            source.mkdir()
            (source / "prompts_subset.txt").write_text("red cube\nblue sphere\n")
            (source / "shared_root_seed_map.json").write_text(
                json.dumps({"seeds": {"0": 1, "1": 2}})
            )
            (source / "subset_manifest.json").write_text(
                json.dumps({"subset_size": 2})
            )
            (source / "study_manifest.json").write_text(json.dumps({"prompt_count": 2}))
            (source / "shared_rewrites_cache.json").write_text(
                json.dumps({"red cube": ["red cube"], "blue sphere": ["blue sphere"]})
            )
            study = root / "remaining"
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO,
                env={
                    **os.environ,
                    "PYTHON_BIN": sys.executable,
                    "SOURCE_SID_RUN_ROOT": str(source),
                    "STUDY_ROOT": str(study),
                    "RUN_ID": "dry",
                    "DRY_RUN": "1",
                },
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("[dry-run] prompts=2", result.stdout)
            manifest = json.loads(
                (study / "dry" / "study_manifest.json").read_text()
            )
            model_ids = {row["model_id"] for row in manifest["models"]}
            self.assertEqual(
                model_ids,
                {"sd35_base", "senseflow_large", "senseflow_medium", "flux_schnell"},
            )
            self.assertNotIn("sid", model_ids)

    def test_coverage_auditor_checks_every_eval_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method = root / "flux_schnell" / "run_v1" / "baseline"
            method.mkdir(parents=True)
            (method / "aggregate_ddp.json").write_text(
                json.dumps({"num_samples": 2, "search_reward": "vqascore"})
            )
            for backend in ("imagereward", "hpsv3", "pickscore", "vqascore"):
                (method / f"best_images_{backend}.json").write_text(
                    json.dumps({"rows": [{}, {}]})
                )
            result = subprocess.run(
                [
                    sys.executable, str(AUDITOR), "--root", str(root),
                    "--models", "flux_schnell", "--methods", "baseline",
                    "--expected-prompts", "2", "--run-id", "v1",
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("failures=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
