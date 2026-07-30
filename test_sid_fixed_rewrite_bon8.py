from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent
RUNNER = REPO / "tools" / "run_sid_fixed_rewrite_bon8.sh"


class FixedRewriteBon8Test(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path]:
        prompts = root / "prompts.csv"
        with prompts.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["prompt_id", "prompt"])
            writer.writeheader()
            writer.writerow({"prompt_id": "p000", "prompt": "A red cube"})
            writer.writerow({"prompt_id": "p001", "prompt": "A blue sphere"})
        seed_map = root / "shared_seed_map.json"
        seed_map.write_text(
            json.dumps({"seeds": {"0": 101, "1": 202}}),
            encoding="utf-8",
        )
        return prompts, seed_map

    def test_dry_run_records_fixed_rewrite_bon8_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts, seed_map = self._inputs(root)
            study = root / "study"
            env = {
                **os.environ,
                "PYTHON_BIN": sys.executable,
                "HUMAN_EVAL_ROOT": str(root),
                "PROMPTS_FILE": str(prompts),
                "STUDY_ROOT": str(study),
                "RUN_ID": "dry",
                "SOURCE_SEED_MAP_FILE": str(seed_map),
                "DRY_RUN": "1",
            }
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("[dry-run] fixed-rewrite BoN-8", result.stdout)
            manifest = json.loads(
                (study / "dry" / "study_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["algorithm_id"], "bon_fixed_rewrite")
            self.assertEqual(manifest["candidate_count"], 8)
            self.assertEqual(manifest["rewrite_count_per_prompt"], 1)
            self.assertEqual(manifest["fixed_cfg"], 1.0)
            self.assertEqual(manifest["prompt_count"], 2)
            self.assertEqual(manifest["seed_source"], str(seed_map.resolve()))
            self.assertIn("original c0", manifest["reward_prompt_invariant"])
            written_seeds = json.loads(
                (study / "dry" / "root_seed_map.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written_seeds["seeds"], {"0": 101, "1": 202})

    def test_runner_refuses_non_eight_candidate_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts, _ = self._inputs(root)
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO,
                env={
                    **os.environ,
                    "PYTHON_BIN": sys.executable,
                    "PROMPTS_FILE": str(prompts),
                    "STUDY_ROOT": str(root / "study"),
                    "BON_N": "7",
                    "DRY_RUN": "1",
                },
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fixed to BON_N=8", result.stderr)

    def test_suite_mode_fixes_one_rewrite_and_c0_is_reward_prompt(self) -> None:
        suite = (REPO / "hpsv2_sd35_sid_ddp_suite.sh").read_text(encoding="utf-8")
        start = suite.index("    bon_fixed_rewrite)")
        end = suite.index("      ;;", start)
        mode = suite[start:end]
        self.assertIn("BON_ACTION_DIVERSE=0", mode)
        self.assertIn("N_VARIANTS=1", mode)
        self.assertIn('CFG_SCALES="${BASELINE_CFG}"', mode)

        sampler = (REPO / "sampling_unified_sd35.py").read_text(encoding="utf-8")
        self.assertIn("score_image(reward_model, prompt, img)", sampler)
        self.assertIn(
            "sample_vars = [sample_variant_idx] * int(args.steps)", sampler
        )

    def test_system_prompt_can_be_overridden_for_precompute(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import precompute_sd35_rewrites as p; print(p.REWRITE_SYSTEM)",
            ],
            cwd=REPO,
            env={**os.environ, "REWRITE_SYSTEM_OVERRIDE": "fixed rewrite test"},
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(result.stdout.splitlines()[-1], "fixed rewrite test")


if __name__ == "__main__":
    unittest.main()
