"""benchmark_suite/compare.py — diff two result dirs into delta tables.

Loads summary.json (preferred) or summary.csv fallback, builds a flat
metric→value map per dir, and emits delta.csv + delta.md with per-metric
delta, pct_change, direction (higher_better / lower_better / neutral), and
regression flags.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from benchmark_suite.report import SUMMARY_CSV_COLUMNS

HIGHER_BETTER_METRICS: frozenset[str] = frozenset(
    {"output_tok_s", "total_tok_s", "judge_score", "successful", "agentic_accuracy"}
)
LOWER_BETTER_METRICS: frozenset[str] = frozenset({"duration_s", "kl_divergence"})
LOWER_BETTER_PREFIXES: tuple[str, ...] = ("ttft_", "tpot_", "perplexity_")

DELTA_CSV_COLUMNS: tuple[str, ...] = (
    "metric", "value_a", "value_b", "delta", "pct_change", "direction", "regression",
)


def _metric_direction(metric: str) -> str:
    """Classify a metric: "higher_better" / "lower_better" / "neutral"."""
    if metric in HIGHER_BETTER_METRICS:
        return "higher_better"
    if metric in LOWER_BETTER_METRICS or metric.startswith(LOWER_BETTER_PREFIXES):
        return "lower_better"
    return "neutral"


def _is_regression(metric: str, delta: float) -> bool:
    """True when delta moves the metric in its bad direction."""
    direction = _metric_direction(metric)
    if direction == "higher_better":
        return delta < 0
    if direction == "lower_better":
        return delta > 0
    return False


def _parse_number(text: str) -> float | int | None:
    """Parse a CSV cell into int (when exact) or float; None when not numeric."""
    try:
        return int(text)
    except ValueError:
        return _parse_float(text)


def _parse_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _metrics_from_json(path: Path) -> dict[str, float | int]:
    """Flat metric→value map from summary.json (last score wins on conflicts)."""
    data: Any = json.loads(path.read_text())
    metrics: dict[str, float | int] = {}
    for score in data.get("scores", []):
        for key, value in score.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
    return metrics


def _metrics_from_csv(path: Path) -> dict[str, float | int]:
    """Flat metric→value map from summary.csv (numeric columns only)."""
    metrics: dict[str, float | int] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            for key, cell in row.items():
                if key is None or cell in (None, ""):
                    continue
                parsed = _parse_number(cell)
                if parsed is not None:
                    metrics[key] = parsed
    return metrics


def _load_metrics(result_dir: Path) -> dict[str, float | int]:
    """Flat metric→value map from a result dir (summary.json, else summary.csv)."""
    json_path = result_dir / "summary.json"
    if json_path.exists():
        return _metrics_from_json(json_path)
    csv_path = result_dir / "summary.csv"
    if csv_path.exists():
        return _metrics_from_csv(csv_path)
    raise FileNotFoundError(
        f"no summary.json or summary.csv in {result_dir}; run `bs report` first"
    )


def _ordered_metrics(a: dict[str, float | int], b: dict[str, float | int]) -> list[str]:
    """Legacy column order first, then any extra metric names sorted."""
    common = set(a) & set(b)
    known = [col for col in SUMMARY_CSV_COLUMNS if col in common]
    extras = sorted(common - set(SUMMARY_CSV_COLUMNS))
    return [*known, *extras]


def _delta_row(metric: str, value_a: float | int, value_b: float | int) -> dict[str, Any]:
    """One row of the delta table."""
    delta = value_b - value_a
    pct_change: float | str = (
        "inf" if value_a == 0 else round((value_b - value_a) / value_a * 100, 4)
    )
    return {
        "metric": metric,
        "value_a": value_a,
        "value_b": value_b,
        "delta": delta,
        "pct_change": pct_change,
        "direction": _metric_direction(metric),
        "regression": "yes" if _is_regression(metric, delta) else "no",
    }


def _write_delta_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(DELTA_CSV_COLUMNS)
        for row in rows:
            writer.writerow([row[col] for col in DELTA_CSV_COLUMNS])


def _render_delta_md(dir_a: Path, dir_b: Path, rows: list[dict[str, Any]]) -> str:
    lines = [f"# Comparison: {dir_a} vs {dir_b}", ""]
    if not rows:
        lines.append("No comparable numeric metrics found in both result dirs.")
        return "\n".join(lines) + "\n"
    header = "| " + " | ".join(DELTA_CSV_COLUMNS) + " |"
    separator = "|" + "|".join(["---"] * len(DELTA_CSV_COLUMNS)) + "|"
    lines.append(header)
    lines.append(separator)
    for row in rows:
        cells = [str(row[col]) for col in DELTA_CSV_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def compare_results(
    dir_a: Path,
    dir_b: Path,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """Compare two result dirs; write delta.csv + delta.md; return their paths."""
    metrics_a = _load_metrics(dir_a)
    metrics_b = _load_metrics(dir_b)
    out_dir = output_dir if output_dir is not None else dir_b
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _delta_row(metric, metrics_a[metric], metrics_b[metric])
        for metric in _ordered_metrics(metrics_a, metrics_b)
    ]
    csv_path = out_dir / "delta.csv"
    _write_delta_csv(rows, csv_path)
    md_path = out_dir / "delta.md"
    md_path.write_text(_render_delta_md(dir_a, dir_b, rows))
    return {"csv": csv_path, "md": md_path}