#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


def _check_wrapper_searchresult(root: Path) -> list[str]:
    errors: list[str] = []
    pattern = re.compile(r"\b\w+\.SearchResult\(")
    for path in sorted(root.glob("sd35_ddp_experiment*.py")):
        text = path.read_text(encoding="utf-8")
        for idx, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                errors.append(
                    f"{path}:{idx}: avoid module-qualified SearchResult constructors in wrappers "
                    "(return backend result object or import SearchResult directly)."
                )
    return errors


def _check_senseflow_offline_preload(root: Path) -> list[str]:
    errors: list[str] = []
    amlt_dir = root / "amlt"
    if not amlt_dir.exists():
        return errors

    for path in sorted(amlt_dir.glob("senseflow*.yaml")):
        text = path.read_text(encoding="utf-8")
        has_offline = ("HF_HUB_OFFLINE=1" in text) or ("TRANSFORMERS_OFFLINE=1" in text)
        if not has_offline:
            continue

        marker = "snapshot_download('domiso/SenseFlow'"
        pos_snap = text.find(marker)
        pos_offline = text.find("HF_HUB_OFFLINE=1")
        if pos_snap < 0:
            errors.append(
                f"{path}: offline mode enabled but SenseFlow preload is missing "
                f"({marker})."
            )
            continue
        if pos_offline >= 0 and pos_snap > pos_offline:
            errors.append(
                f"{path}: SenseFlow preload appears after HF_HUB_OFFLINE=1; "
                "preload must run before offline mode."
            )
    return errors


def _check_sd35_offline_loader(root: Path) -> list[str]:
    errors: list[str] = []
    path = root / "sampling_unified_sd35.py"
    if not path.exists():
        return errors
    text = path.read_text(encoding="utf-8")
    required_snippets = (
        "offline = str(os.environ.get(\"HF_HUB_OFFLINE\", \"\")).strip().lower()",
        "local_files_only=True",
        "Resolved offline transformer snapshot:",
    )
    for snippet in required_snippets:
        if snippet not in text:
            errors.append(f"{path}: missing offline-safe loader snippet: {snippet!r}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parent
    errors: list[str] = []
    errors.extend(_check_wrapper_searchresult(root))
    errors.extend(_check_senseflow_offline_preload(root))
    errors.extend(_check_sd35_offline_loader(root))

    if errors:
        print("Runtime guard checks FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Runtime guard checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
