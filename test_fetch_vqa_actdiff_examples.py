from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.fetch_vqa_actdiff_examples import select_model


class FetchVqaActdiffExamplesTest(unittest.TestCase):
    def _method(
        self,
        run_root: Path,
        method: str,
        scores: list[float],
        prompts: list[str],
    ) -> None:
        method_dir = run_root / method
        (method_dir / "logs").mkdir(parents=True)
        (method_dir / "aggregate_ddp.json").write_text(
            json.dumps({"search_reward": "vqascore", "num_samples": len(scores)}),
            encoding="utf-8",
        )
        rows = []
        log_rows = []
        for index, (score, prompt) in enumerate(zip(scores, prompts)):
            image_path = method_dir / f"p{index:04d}.png"
            Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(image_path)
            rows.append({
                "prompt_index": index,
                "prompt": prompt,
                "image_path": str(image_path),
                "scores": {"vqascore": score},
            })
            log_rows.append(json.dumps({
                "prompt_index": index,
                "prompt": prompt,
                "score": score,
            }))
        (method_dir / "best_images_vqascore.json").write_text(
            json.dumps({"rows": rows}), encoding="utf-8"
        )
        (method_dir / "logs" / "rank_0.jsonl").write_text(
            "\n".join(log_rows) + "\n", encoding="utf-8"
        )

    def test_selects_only_positive_margin_and_copies_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            prompts = ["two red cubes", "a blue dog", "three birds"]
            self._method(run_root, "baseline", [0.2, 0.7, 0.3], prompts)
            self._method(run_root, "bon", [0.4, 0.8, 0.5], prompts)
            self._method(run_root, "bon_mcts", [0.9, 0.75, 0.8], prompts)

            selected, report = select_model(
                model_id="sid",
                run_root=run_root,
                output_dir=root / "bundle",
                top_k=3,
            )
            self.assertEqual([row["prompt_index"] for row in selected], [0, 2])
            self.assertEqual(selected[0]["strongest_control"], "bon")
            self.assertAlmostEqual(selected[0]["delta_vs_strongest"], 0.5)
            self.assertTrue(Path(selected[0]["actdiff_bundle_image"]).is_file())
            self.assertTrue(Path(selected[0]["control_bundle_image"]).is_file())
            self.assertTrue(Path(selected[0]["baseline_bundle_image"]).is_file())
            self.assertEqual(report["positive_margin_rows"], 2)
            self.assertEqual(
                report["exclusions"]["actdiff_not_above_strongest_control"], 1
            )

    def test_rank_local_prompt_indices_do_not_overwrite_distinct_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "run"
            prompts = ["first prompt", "second prompt"]
            self._method(run_root, "baseline", [0.1, 0.2], prompts)
            self._method(run_root, "bon_mcts", [0.8, 0.9], prompts)
            for method in ("baseline", "bon_mcts"):
                path = run_root / method / "best_images_vqascore.json"
                payload = json.loads(path.read_text())
                payload["rows"][1]["prompt_index"] = 0
                path.write_text(json.dumps(payload))

            selected, report = select_model(
                model_id="flux_schnell",
                run_root=run_root,
                output_dir=root / "bundle",
                top_k=3,
            )
            self.assertEqual(len(selected), 2)
            self.assertEqual({row["prompt"] for row in selected}, set(prompts))
            self.assertEqual(report["actdiff_rows"], 2)


if __name__ == "__main__":
    unittest.main()
