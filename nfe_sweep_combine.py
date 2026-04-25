"""Combine NFE-sweep results from one or more backends and emit:

    - combined.csv  : all (backend, method, target_nfe) rows in a single file
                      with per-eval-backend reward columns; easy to re-plot.
    - <backend>_<eval>_vs_nfe.png : one curve per method per eval backend.
    - summary_<eval>_vs_nfe.png   : one figure per eval backend with one
                                    subplot per backend, all methods overlaid.

Usage:
    python nfe_sweep_combine.py \
        --inputs <sweep_root_or_tsv> [<more> ...] \
        --out_dir <merged_dir>

Each input may be either a sweep_summary.tsv file (preferred) or a sweep root
directory containing sweep_summary.tsv. The backend label is taken from the
parent directory by default; override with --label <input>=<label>.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path


def _resolve_input(path: str) -> tuple[Path, str]:
    p = Path(path).expanduser().resolve()
    if p.is_file() and p.name.endswith(".tsv"):
        tsv = p
    elif p.is_dir():
        cand = p / "sweep_summary.tsv"
        if not cand.exists():
            matches = sorted(p.glob("**/sweep_summary.tsv"))
            if not matches:
                raise FileNotFoundError(f"no sweep_summary.tsv under {p}")
            cand = matches[0]
        tsv = cand
    else:
        raise FileNotFoundError(f"input not found: {p}")

    # Default label = name of dir 2 levels up from sweep_summary.tsv.
    # Layout: <out_root_base>/<backend>/sweep_<TS>/sweep_summary.tsv
    parts = tsv.parts
    label = "unknown"
    for cand_label in (tsv.parent.parent.name, tsv.parent.name):
        if cand_label and not cand_label.startswith("sweep_"):
            label = cand_label
            break
    return tsv, label


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _to_float(v: object) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: object) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def combine(inputs: list[tuple[Path, str]]) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    eval_cols: list[str] = []
    for tsv_path, label in inputs:
        for row in _read_tsv(tsv_path):
            out: dict[str, object] = {
                "backend": label,
                "method": row.get("method", ""),
                "target_nfe": _to_int(row.get("target_nfe")),
                "status": row.get("status", "ok"),
                "num_samples": _to_int(row.get("num_samples")),
                "elapsed_sec": _to_int(row.get("elapsed_sec")),
                "mean_baseline": _to_float(row.get("mean_baseline")),
                "mean_search": _to_float(row.get("mean_search")),
                "mean_delta": _to_float(row.get("mean_delta")),
            }
            for k, v in row.items():
                if k.startswith("eval_"):
                    out[k] = _to_float(v)
                    if k not in eval_cols:
                        eval_cols.append(k)
            rows.append(out)
    return rows, eval_cols


def write_combined_csv(rows: list[dict[str, object]], eval_cols: list[str], out_path: Path) -> None:
    base_cols = [
        "backend", "method", "target_nfe", "status", "num_samples",
        "elapsed_sec", "mean_baseline", "mean_search", "mean_delta",
    ]
    cols = base_cols + sorted(eval_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in rows:
            writer.writerow([row.get(c, "") for c in cols])


_METHOD_STYLE = {
    "bon":      {"color": "#1f77b4", "marker": "o"},
    "beam":     {"color": "#ff7f0e", "marker": "s"},
    "smc":      {"color": "#2ca02c", "marker": "^"},
    "bon_mcts": {"color": "#d62728", "marker": "D"},
    "baseline": {"color": "#7f7f7f", "marker": "x"},
}


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def plot_per_backend(rows: list[dict[str, object]], eval_cols: list[str], out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[combine] matplotlib unavailable, skipping plots: {exc}", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    backends = sorted({str(r["backend"]) for r in rows})
    score_cols = sorted(eval_cols) or ["mean_search"]

    # Per-backend plot, one figure per eval column.
    for backend in backends:
        sub = [r for r in rows if r["backend"] == backend]
        for col in score_cols:
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            methods = sorted({str(r["method"]) for r in sub})
            for method in methods:
                pts = sorted(
                    [(int(r["target_nfe"]), float(r[col]))
                     for r in sub
                     if r["method"] == method
                     and r.get(col) is not None
                     and r.get("target_nfe") is not None],
                    key=lambda p: p[0],
                )
                if not pts:
                    continue
                style = _METHOD_STYLE.get(method, {"color": None, "marker": "o"})
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker=style["marker"], color=style["color"], label=method)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Target NFE")
            ax.set_ylabel(col.replace("eval_", ""))
            ax.set_title(f"{backend}: {col} vs NFE")
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            path = out_dir / f"{_safe_filename(backend)}_{_safe_filename(col)}_vs_nfe.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    # Combined figure: one panel per backend, per eval column.
    if len(backends) > 1:
        for col in score_cols:
            fig, axes = plt.subplots(1, len(backends), figsize=(6.0 * len(backends), 4.5), sharey=True)
            if len(backends) == 1:
                axes = [axes]
            for ax, backend in zip(axes, backends):
                sub = [r for r in rows if r["backend"] == backend]
                methods = sorted({str(r["method"]) for r in sub})
                for method in methods:
                    pts = sorted(
                        [(int(r["target_nfe"]), float(r[col]))
                         for r in sub
                         if r["method"] == method
                         and r.get(col) is not None
                         and r.get("target_nfe") is not None],
                        key=lambda p: p[0],
                    )
                    if not pts:
                        continue
                    style = _METHOD_STYLE.get(method, {"color": None, "marker": "o"})
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker=style["marker"], color=style["color"], label=method)
                ax.set_xscale("log", base=2)
                ax.set_xlabel("Target NFE")
                ax.set_title(backend)
                ax.grid(True, which="both", linestyle=":", alpha=0.4)
            axes[0].set_ylabel(col.replace("eval_", ""))
            axes[-1].legend(loc="best", fontsize=9)
            fig.suptitle(f"{col} vs NFE")
            fig.tight_layout()
            path = out_dir / f"summary_{_safe_filename(col)}_vs_nfe.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Paths to sweep_summary.tsv files or sweep root dirs.")
    ap.add_argument("--label", action="append", default=[],
                    help="Override label for an input as 'path=label' (may repeat).")
    ap.add_argument("--out_dir", required=True,
                    help="Directory for combined.csv and plot PNGs.")
    args = ap.parse_args()

    overrides = {}
    for entry in args.label:
        if "=" not in entry:
            print(f"[combine] ignoring malformed --label: {entry}", file=sys.stderr)
            continue
        k, v = entry.split("=", 1)
        overrides[str(Path(k).expanduser().resolve())] = v

    inputs: list[tuple[Path, str]] = []
    for raw in args.inputs:
        tsv_path, default_label = _resolve_input(raw)
        label = overrides.get(str(Path(raw).expanduser().resolve()), default_label)
        inputs.append((tsv_path, label))
        print(f"[combine] input: {tsv_path}  label={label}")

    rows, eval_cols = combine(inputs)
    print(f"[combine] {len(rows)} rows, {len(eval_cols)} eval columns")

    out_dir = Path(args.out_dir).expanduser().resolve()
    csv_path = out_dir / "combined.csv"
    write_combined_csv(rows, eval_cols, csv_path)
    print(f"[combine] wrote {csv_path}")

    plots = plot_per_backend(rows, eval_cols, out_dir)
    for p in plots:
        print(f"[combine] wrote {p}")


if __name__ == "__main__":
    main()
