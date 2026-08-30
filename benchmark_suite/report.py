"""benchmark_suite/report.py — summary.csv/json + README.md writers.

summary.csv reuses the parent's legacy column names (parse_matrix_results.py)
verbatim so existing reports in parent /bench_results/ keep working.
README.md follows the parent bench_results/ house style:
Date / Goal / Configuration / Results / Interpretation / Verdict / Files.
"""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from benchmark_suite.paths import readme_path, summary_csv_path, summary_json_path
from benchmark_suite.recipe import Recipe
from benchmark_suite.scoring.base import ScoreRecord, ScoreStatus

# Legacy column names from parent's parse_matrix_results.py. Reused verbatim
# so existing reports in parent /bench_results/ keep working.
SUMMARY_CSV_COLUMNS: tuple[str, ...] = (
    "cell_id", "scorer_kind", "status",
    "output_tok_s", "peak_output_tok_s", "total_tok_s",
    "ttft_mean_ms", "ttft_median_ms", "ttft_p99_ms",
    "tpot_mean_ms", "tpot_median_ms",
    "duration_s", "successful", "failed",
    "concurrency",
    "kl_divergence",
    "judge_score",
    "perplexity_wikitext",
    "agentic_accuracy",
    "started_at", "finished_at",
)

# Columns sourced from ScoreRecord identity fields instead of its metrics dict.
_IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {"cell_id", "scorer_kind", "status", "started_at", "finished_at"}
)


def _cell_value(score: ScoreRecord, column: str) -> float | int | str:
    """One summary column's value for one score; "" when absent (never 0)."""
    if column == "cell_id":
        return score.cell_id
    if column == "scorer_kind":
        return score.kind
    if column == "status":
        return score.status
    if column == "started_at":
        return score.started_at.isoformat()
    if column == "finished_at":
        return score.finished_at.isoformat()
    return score.metrics.get(column, "")


def write_summary_csv(scores: list[ScoreRecord], path: Path) -> None:
    """Write the flat summary: one row per ScoreRecord, legacy columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(SUMMARY_CSV_COLUMNS)
        for score in scores:
            writer.writerow([_cell_value(score, col) for col in SUMMARY_CSV_COLUMNS])


def write_summary_json(recipe: Recipe, scores: list[ScoreRecord], path: Path) -> None:
    """Write the structured summary: recipe dump + one to_dict per score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "recipe": recipe.model_dump(mode="json"),
        "scores": [score.to_dict() for score in scores],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _verdict(scores: list[ScoreRecord]) -> str:
    """PASS when every score succeeded, FAIL when every score failed, else PARTIAL."""
    if not scores:
        return "FAIL"
    statuses = {score.status for score in scores}
    if statuses == {ScoreStatus.SUCCESS}:
        return "PASS"
    if statuses == {ScoreStatus.FAILURE}:
        return "FAIL"
    return "PARTIAL"


def _run_date(scores: list[ScoreRecord]) -> str:
    """ISO date of the run: latest score start, else today."""
    if scores:
        return max(score.started_at for score in scores).date().isoformat()
    return datetime.now(UTC).date().isoformat()


def _results_columns(scores: list[ScoreRecord]) -> list[str]:
    """Identity columns + every metric column holding at least one value.

    All-null metric columns are omitted; extra (non-legacy) metric names are
    appended sorted. Timestamps stay out of the table (Date section covers them).
    """
    present: set[str] = set()
    for score in scores:
        present.update(score.metrics)
    known = [col for col in SUMMARY_CSV_COLUMNS if col in present]
    extras = sorted(present - set(SUMMARY_CSV_COLUMNS))
    return ["cell_id", "scorer_kind", "status", *known, *extras]


def _results_table(scores: list[ScoreRecord]) -> list[str]:
    """Markdown table lines for the Results section."""
    if not scores:
        return ["No scores recorded."]
    columns = _results_columns(scores)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(_cell_value(score, col)) for col in columns) + " |"
        for score in scores
    ]
    return [header, separator, *rows]


def _interpretation(scores: list[ScoreRecord]) -> list[str]:
    """One bullet per score: kind, cell, status, and error when present."""
    if not scores:
        return ["No scorers ran; nothing to interpret."]
    lines: list[str] = []
    for score in scores:
        line = f"- {score.kind} ({score.cell_id}): {score.status}"
        if score.error:
            line += f" — error: {score.error}"
        lines.append(line)
    return lines


def _files(scores: list[ScoreRecord]) -> list[str]:
    """Bullet list of the report artifacts plus every score's artifacts."""
    lines = [
        "- `summary.csv` — flat per-scorer metrics (legacy columns)",
        "- `summary.json` — structured recipe + score records",
        "- `README.md` — this report",
    ]
    for score in scores:
        for rel in sorted(score.artifacts.values()):
            lines.append(f"- `{rel}` — {score.kind} artifact")
    return lines


def _render_readme(recipe: Recipe, scores: list[ScoreRecord]) -> str:
    """Assemble the full README.md text (parent bench_results style)."""
    title = recipe.meta.name
    if recipe.meta.description:
        title = f"{title} — {recipe.meta.description}"
    config_yaml = yaml.safe_dump(
        recipe.model_dump(mode="json", exclude_none=True), sort_keys=False
    )
    lines: list[str] = [
        f"# {title}",
        "",
        "## Date",
        _run_date(scores),
        "",
        "## Goal",
        recipe.meta.description or "(no description recorded)",
        "",
        "## Configuration",
        "```yaml",
        config_yaml.rstrip("\n"),
        "```",
        "",
        "## Results",
        *_results_table(scores),
        "",
        "## Interpretation",
        *_interpretation(scores),
        "",
        "## Verdict",
        _verdict(scores),
        "",
        "## Files",
        *_files(scores),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_readme(recipe: Recipe, scores: list[ScoreRecord], result_dir: Path) -> Path:
    """Write README.md in parent bench_results/ style; return its path."""
    result_dir.mkdir(parents=True, exist_ok=True)
    path = readme_path(result_dir)
    path.write_text(_render_readme(recipe, scores))
    return path


def write_summary(
    recipe: Recipe,
    scores: list[ScoreRecord],
    result_dir: Path,
) -> dict[str, Path]:
    """Write summary.csv + summary.json + README.md into result_dir."""
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_csv_path(result_dir)
    json_path = summary_json_path(result_dir)
    write_summary_csv(scores, csv_path)
    write_summary_json(recipe, scores, json_path)
    readme = write_readme(recipe, scores, result_dir)
    return {"summary_csv": csv_path, "summary_json": json_path, "readme": readme}