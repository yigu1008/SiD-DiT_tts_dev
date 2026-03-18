"""Compatibility entrypoint for CFG debug sweep."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    debug_script = Path(__file__).resolve().parent / "debug" / "cfg" / "test_sid_cfg_debug.py"
    runpy.run_path(str(debug_script), run_name="__main__")
