from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parent
RUNNER = REPO / "tools" / "run_hpsv2_fixed_rewrite_bon8_reward_sweep.sh"


class Hpsv2FixedRewriteBon8RewardSweepTest(unittest.TestCase):
    def test_dry_run_builds_twelve_matched_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "hpsv2_subset.txt"
            prompts.write_text("A red cube\nA blue sphere\n", encoding="utf-8")
            study = root / "study"
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO,
                env={
                    **os.environ,
                    "PYTHON_BIN": sys.executable,
                    "PROMPTS_FILE": str(prompts),
                    "STUDY_ROOT": str(study),
                    "RUN_ID": "dry",
                    "EXPECTED_PROMPT_COUNT": "2",
                    "DRY_RUN": "1",
                },
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn(
                "[dry-run] HPSv2 fixed-rewrite BoN-8 reward sweep",
                result.stdout,
            )
            manifest = json.loads(
                (study / "dry" / "study_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["prompt_count"], 2)
            self.assertEqual(manifest["candidate_count"], 8)
            self.assertEqual(len(manifest["models"]), 4)
            self.assertEqual(len(manifest["cells"]), 12)
            self.assertEqual(
                set(manifest["reward_arms"]),
                {"imagereward", "hpsv3", "multi_reward"},
            )
            self.assertEqual(
                manifest["reward_arms"]["multi_reward"]["reward_backend"],
                "composite_hpsv3_ir",
            )
            self.assertIn(
                "0.5 * minmax(ImageReward",
                manifest["reward_arms"]["multi_reward"]["definition"],
            )
            self.assertIn("original c0", manifest["reward_prompt_invariant"])
            self.assertEqual(len(manifest["seed_map_files"]), 4)
            for path in manifest["seed_map_files"].values():
                seed_map = json.loads(Path(path).read_text(encoding="utf-8"))
                self.assertEqual(seed_map["prompt_count"], 2)
                self.assertEqual(set(seed_map["seeds"]), {"0", "1"})

    def test_unknown_reward_arm_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "hpsv2_subset.txt"
            prompts.write_text("A red cube\n", encoding="utf-8")
            result = subprocess.run(
                ["bash", str(RUNNER)],
                cwd=REPO,
                env={
                    **os.environ,
                    "PYTHON_BIN": sys.executable,
                    "PROMPTS_FILE": str(prompts),
                    "STUDY_ROOT": str(root / "study"),
                    "REWARD_ARMS": "imagereward unknown",
                    "EXPECTED_PROMPT_COUNT": "1",
                    "DRY_RUN": "1",
                },
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown reward arm", result.stderr)

    def test_flux_mode_uses_one_cached_rewrite_and_scores_c0(self) -> None:
        suite = (REPO / "hpsv2_flux_schnell_ddp_suite.sh").read_text(
            encoding="utf-8"
        )
        start = suite.index("    bon_fixed_rewrite)")
        end = suite.index("      ;;", start)
        mode = suite[start:end]
        self.assertIn("--bon_action_diverse 0", mode)
        self.assertIn("--n_variants 1", mode)
        self.assertIn("--fixed_rewrite_only", mode)
        self.assertIn('--rewrites_file "${REWRITES_FILE}"', mode)

        sampler = (REPO / "sampling_flux_unified.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "select_prompt_bank_with_rewrites(",
            sampler,
        )
        self.assertIn("score_image(reward_model, prompt, img)", sampler)
        self.assertIn('return [("fixed_rewrite", values[0])]', sampler)


if __name__ == "__main__":
    unittest.main()
