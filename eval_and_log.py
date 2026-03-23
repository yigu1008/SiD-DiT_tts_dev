from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _font(size: int = 18) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_score_panel(
    rows: list[tuple[str, Image.Image, float]],
    out_path: str,
) -> None:
    if len(rows) == 0:
        raise ValueError("rows cannot be empty")
    w, h = rows[0][1].size
    label_h = 52
    panel = Image.new("RGB", (w * len(rows), h + label_h), (20, 20, 20))
    draw = ImageDraw.Draw(panel)
    f_title = _font(18)
    for i, (name, img, score) in enumerate(rows):
        x0 = i * w
        panel.paste(img, (x0, label_h))
        draw.text((x0 + 8, 8), f"{name} score={score:.4f}", fill=(220, 220, 220), font=f_title)
    panel.save(out_path)
