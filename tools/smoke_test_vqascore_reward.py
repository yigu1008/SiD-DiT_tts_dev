#!/usr/bin/env python3
"""Load CLIP-FlanT5 through reward_server and score one synthetic image."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import reward_server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="clip-flant5-xxl")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    reward_server._inject_wandb_stub()
    scorer = reward_server._load_vqascore(args.device, args.model)

    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 48, 176, 176), fill="red")
    score = float(scorer("a red circle on a white background", image))
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"invalid VQAScore result: {score!r}")
    print(
        f"[vqa-smoke] model={args.model} device={args.device} "
        f"score={score:.6f} OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
