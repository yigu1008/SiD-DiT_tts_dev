"""Compatibility entrypoint for prompt-variant debug tests."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    debug_script = (
        Path(__file__).resolve().parent / "debug" / "prompt" / "test_prompt_variants_debug.py"
    )
    runpy.run_path(str(debug_script), run_name="__main__")
