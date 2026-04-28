#!/usr/bin/env python3
"""Plot NFE vs reward curves from a combined.csv table.

Expected columns (minimum):
  - backend
  - method
  - target_nfe
  - status (optional; defaults to 'ok' filtering)
  - reward columns such as eval_imagereward / eval_hpsv2 / eval_pickscore
    (or mean_search / mean_delta)
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


_METHOD_STYLE = {
    "baseline":     {"color": "#7f7f7f", "marker": "x"},
    "greedy":       {"color": "#17becf", "marker": "o"},
    "mcts":         {"color": "#d62728", "marker": "D"},
    "bon_mcts":     {"color": "#9467bd", "marker": "P"},
    "beam":         {"color": "#ff7f0e", "marker": "s"},
    "smc":          {"color": "#2ca02c", "marker": "^"},
    "fksteering":   {"color": "#006400", "marker": "<"},
    "ga":           {"color": "#1f77b4", "marker": "v"},
    "bon":          {"color": "#8c564b", "marker": "h"},
    "dts":          {"color": "#e377c2", "marker": "*"},
    "dts_star":     {"color": "#bcbd22", "marker": "X"},
    "noise_inject": {"color": "#ff1493", "marker": ">"},
}


def _to_float(v: str | None) -> float | None:
    if v is None:
        return None
    vv = str(v).strip()
    if vv == "":
        return None
    try:
        return float(vv)
    except ValueError:
        return None


def _to_int(v: str | None) -> int | None:
    f = _to_float(v)
    if f is None:
        return None
    return int(round(float(f)))


def _safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _choose_y_cols(header: list[str], requested: list[str] | None) -> list[str]:
    if requested:
        cols = [c for c in requested if c in header]
        if not cols:
            raise RuntimeError(
                f"None of requested y columns found. requested={requested} available={header}"
            )
        return cols

    eval_cols = [c for c in header if c.startswith("eval_")]
    aux = [c for c in ("mean_search", "mean_delta") if c in header]
    cols = eval_cols + aux
    if not cols:
        raise RuntimeError("No reward-like columns found (expected eval_* or mean_search/mean_delta).")
    return cols


def _aggregate_points(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    by_x: dict[int, list[float]] = defaultdict(list)
    for x, y in points:
        by_x[int(x)].append(float(y))
    out = []
    for x in sorted(by_x):
        vals = by_x[x]
        out.append((x, sum(vals) / max(1, len(vals))))
    return out


def _parse_multi(value: str | None) -> set[str]:
    if not value:
        return set()
    out = set()
    for token in str(value).replace(",", " ").split():
        tok = token.strip()
        if tok:
            out.add(tok)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default="/Users/guyi/Downloads/combined.csv",
        help="Path to combined.csv.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for plots (default: <csv_dir>/plots_nfe_vs_reward).",
    )
    p.add_argument("--x_col", default="target_nfe")
    p.add_argument("--y_cols", nargs="+", default=None)
    p.add_argument(
        "--status",
        default="ok",
        help="Filter status column value (set to 'all' to disable).",
    )
    p.add_argument(
        "--backends",
        default="",
        help="Optional filter list, comma/space separated.",
    )
    p.add_argument(
        "--methods",
        default="",
        help="Optional filter list, comma/space separated.",
    )
    p.add_argument(
        "--no_log2_x",
        action="store_true",
        help="Disable log2 x-axis for NFE.",
    )
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f"No rows in CSV: {path}")
    return rows, header


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (csv_path.parent / "plots_nfe_vs_reward").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, header = load_rows(csv_path)
    y_cols = _choose_y_cols(header, args.y_cols)
    backend_filter = _parse_multi(args.backends)
    method_filter = _parse_multi(args.methods)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to plot: {exc}") from exc

    print(f"[plot] csv={csv_path}")
    print(f"[plot] y_cols={y_cols}")
    print(f"[plot] out_dir={out_dir}")

    # Keep only rows that can contribute to at least one target curve.
    filtered_rows: list[dict[str, str]] = []
    for r in rows:
        backend = str(r.get("backend", "")).strip()
        method = str(r.get("method", "")).strip()
        status = str(r.get("status", "")).strip().lower()
        if backend_filter and backend not in backend_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        if str(args.status).lower() != "all" and status != str(args.status).lower():
            continue
        filtered_rows.append(r)

    if not filtered_rows:
        raise RuntimeError("No rows left after filters.")

    backends = sorted({str(r.get("backend", "")).strip() for r in filtered_rows if r.get("backend")})

    for y_col in y_cols:
        # Series by (backend, method)
        series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        for r in filtered_rows:
            x = _to_int(r.get(args.x_col))
            y = _to_float(r.get(y_col))
            if x is None or y is None:
                continue
            key = (str(r.get("backend", "")).strip(), str(r.get("method", "")).strip())
            series[key].append((int(x), float(y)))

        # Figure 1: all backend/method curves overlaid.
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plotted = 0
        for (backend, method), pts in sorted(series.items()):
            agg = _aggregate_points(pts)
            if not agg:
                continue
            style = _METHOD_STYLE.get(method, {"color": None, "marker": "o"})
            ax.plot(
                [p[0] for p in agg],
                [p[1] for p in agg],
                marker=style["marker"],
                color=style["color"],
                linewidth=1.8,
                label=f"{backend}/{method}",
            )
            plotted += 1
        if args.no_log2_x is False:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(args.x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {args.x_col} (all)")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        if plotted > 0:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        all_out = out_dir / f"all_{_safe_name(y_col)}_vs_{_safe_name(args.x_col)}.png"
        fig.savefig(all_out, dpi=int(args.dpi))
        plt.close(fig)
        print(f"[plot] wrote {all_out}")

        # Figure 2: one panel per backend, methods overlaid.
        n_b = max(1, len(backends))
        fig2, axes = plt.subplots(1, n_b, figsize=(6.2 * n_b, 4.8), sharey=True)
        if n_b == 1:
            axes = [axes]
        for ax, backend in zip(axes, backends):
            methods = sorted({m for (b, m) in series if b == backend})
            for method in methods:
                agg = _aggregate_points(series.get((backend, method), []))
                if not agg:
                    continue
                style = _METHOD_STYLE.get(method, {"color": None, "marker": "o"})
                ax.plot(
                    [p[0] for p in agg],
                    [p[1] for p in agg],
                    marker=style["marker"],
                    color=style["color"],
                    linewidth=1.8,
                    label=method,
                )
            if args.no_log2_x is False:
                ax.set_xscale("log", base=2)
            ax.set_title(backend)
            ax.set_xlabel(args.x_col)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
        axes[0].set_ylabel(y_col)
        if len(backends) > 0:
            axes[-1].legend(loc="best", fontsize=8)
        fig2.suptitle(f"{y_col} vs {args.x_col} (by backend)")
        fig2.tight_layout()
        byb_out = out_dir / f"by_backend_{_safe_name(y_col)}_vs_{_safe_name(args.x_col)}.png"
        fig2.savefig(byb_out, dpi=int(args.dpi))
        plt.close(fig2)
        print(f"[plot] wrote {byb_out}")


if __name__ == "__main__":
    main()
