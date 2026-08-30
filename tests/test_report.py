"""Tests for benchmark_suite.report — summary.csv/json + README.md writers."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_suite.recipe import Recipe
from benchmark_suite.report import (
    SUMMARY_CSV_COLUMNS,
    write_readme,
    write_summary,
    write_summary_csv,
    write_summary_json,
)
from benchmark_suite.scoring.base import ScoreRecord, ScoreStatus

T0 = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 8, 30, 12, 5, 0, tzinfo=UTC)
CELL = "dense_triton_triton_cg0_mtp0"


def make_recipe() -> Recipe:
    """Minimal valid recipe with a known name/description."""
    return Recipe.model_validate(
        {"meta": {"name": "test-recipe", "description": "A test recipe for report tests"}}
    )


def make_score(
    kind: str = "throughput",
    status: str = ScoreStatus.SUCCESS,
    metrics: dict[str, float | int | str] | None = None,
    cell_id: str = CELL,
) -> ScoreRecord:
    """Deterministic ScoreRecord with fixed timestamps."""
    return ScoreRecord(
        kind=kind,
        cell_id=cell_id,
        status=status,
        started_at=T0,
        finished_at=T1,
        metrics=metrics if metrics is not None else {},
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def verdict_of(content: str) -> str:
    """Extract the verdict line that follows the '## Verdict' header."""
    return content.split("## Verdict", 1)[1].strip().splitlines()[0]


class TestWriteSummaryCsv:
    def test_write_summary_csv_columns_match_legacy(self, tmp_path: Path) -> None:
        """Header must equal the legacy column tuple byte-for-byte (parent parser)."""
        path = tmp_path / "summary.csv"
        write_summary_csv([make_score()], path)
        with path.open(newline="") as f:
            header = next(csv.reader(f))
        assert tuple(header) == SUMMARY_CSV_COLUMNS
        # Lock the legacy contract from parent parse_matrix_results.py verbatim.
        assert SUMMARY_CSV_COLUMNS == (
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
        # Unix line endings, no \r anywhere.
        assert "\r" not in path.read_text()

    def test_write_summary_csv_one_row_per_score(self, tmp_path: Path) -> None:
        """One ScoreRecord = exactly one CSV row, in input order."""
        scores = [
            make_score(kind="throughput"),
            make_score(kind="kld"),
            make_score(kind="llm_judge"),
        ]
        path = tmp_path / "summary.csv"
        write_summary_csv(scores, path)
        rows = read_csv_rows(path)
        assert len(rows) == 3
        assert [r["scorer_kind"] for r in rows] == ["throughput", "kld", "llm_judge"]

    def test_write_summary_csv_metrics_populated(self, tmp_path: Path) -> None:
        """Metric columns come from ScoreRecord.metrics; identity columns from fields."""
        score = make_score(
            metrics={"output_tok_s": 76.5, "ttft_mean_ms": 120.0, "successful": 32}
        )
        path = tmp_path / "summary.csv"
        write_summary_csv([score], path)
        row = read_csv_rows(path)[0]
        assert row["output_tok_s"] == "76.5"
        assert row["ttft_mean_ms"] == "120.0"
        assert row["successful"] == "32"
        assert row["cell_id"] == CELL
        assert row["scorer_kind"] == "throughput"
        assert row["status"] == "success"
        assert row["started_at"] == "2026-08-30T12:00:00+00:00"
        assert row["finished_at"] == "2026-08-30T12:05:00+00:00"

    def test_write_summary_csv_missing_metric_blank(self, tmp_path: Path) -> None:
        """Metrics absent from the metrics dict render as empty string, never 0."""
        score = make_score(metrics={"output_tok_s": 76.5})
        path = tmp_path / "summary.csv"
        write_summary_csv([score], path)
        row = read_csv_rows(path)[0]
        assert row["kl_divergence"] == ""
        assert row["judge_score"] == ""
        assert row["perplexity_wikitext"] == ""
        assert row["concurrency"] == ""
        assert row["agentic_accuracy"] == ""


class TestWriteSummaryJson:
    def test_write_summary_json_round_trip(self, tmp_path: Path) -> None:
        """summary.json holds the recipe dump plus one to_dict per score."""
        recipe = make_recipe()
        score = make_score(
            metrics={"output_tok_s": 76.5},
        )
        score.artifacts["llm_perf_c1.json"] = "artifacts/llm_perf_c1.json"
        path = tmp_path / "summary.json"
        write_summary_json(recipe, [score], path)
        data = json.loads(path.read_text())
        assert data["recipe"]["meta"]["name"] == "test-recipe"
        assert data["recipe"]["meta"]["description"] == "A test recipe for report tests"
        assert data["recipe"]["endpoint"]["url"] == recipe.endpoint.url
        assert len(data["scores"]) == 1
        s = data["scores"][0]
        assert s["kind"] == "throughput"
        assert s["cell_id"] == CELL
        assert s["status"] == "success"
        assert s["started_at"] == "2026-08-30T12:00:00+00:00"
        assert s["finished_at"] == "2026-08-30T12:05:00+00:00"
        assert s["metrics"] == {"output_tok_s": 76.5}
        assert s["artifacts"] == {"llm_perf_c1.json": "artifacts/llm_perf_c1.json"}
        assert s["error"] is None


class TestWriteReadme:
    def test_write_readme_includes_all_sections(self, tmp_path: Path) -> None:
        """README has the parent bench_results section set + recipe YAML config."""
        readme = write_readme(make_recipe(), [make_score()], tmp_path)
        content = readme.read_text()
        assert content.startswith("# test-recipe — A test recipe for report tests")
        for section in (
            "## Date",
            "## Goal",
            "## Configuration",
            "## Results",
            "## Interpretation",
            "## Verdict",
            "## Files",
        ):
            assert section in content, f"missing section: {section}"
        # Configuration embeds the recipe as a YAML code block.
        assert "```yaml" in content
        assert "name: test-recipe" in content
        # Date section carries the run date derived from the scores.
        assert "2026-08-30" in content.split("## Date", 1)[1].split("##", 1)[0]

    def test_write_readme_verdict_pass_when_all_success(self, tmp_path: Path) -> None:
        scores = [make_score(kind="throughput"), make_score(kind="kld")]
        content = write_readme(make_recipe(), scores, tmp_path).read_text()
        assert verdict_of(content) == "PASS"

    def test_write_readme_verdict_partial_when_some_fail(self, tmp_path: Path) -> None:
        scores = [
            make_score(kind="throughput", status=ScoreStatus.SUCCESS),
            make_score(kind="kld", status=ScoreStatus.FAILURE),
        ]
        content = write_readme(make_recipe(), scores, tmp_path).read_text()
        assert verdict_of(content) == "PARTIAL"

    def test_write_readme_verdict_fail_when_all_fail(self, tmp_path: Path) -> None:
        scores = [
            make_score(kind="throughput", status=ScoreStatus.FAILURE),
            make_score(kind="kld", status=ScoreStatus.FAILURE),
        ]
        content = write_readme(make_recipe(), scores, tmp_path).read_text()
        assert verdict_of(content) == "FAIL"

    def test_write_readme_results_table_omits_null_columns(self, tmp_path: Path) -> None:
        """Results table keeps columns with >=1 value, drops all-null columns."""
        scores = [
            make_score(kind="throughput", metrics={"output_tok_s": 76.5}),
            make_score(kind="llm_judge", metrics={"judge_score": 8.5}),
        ]
        content = write_readme(make_recipe(), scores, tmp_path).read_text()
        header_line = next(
            line for line in content.splitlines() if line.startswith("| cell_id")
        )
        assert "output_tok_s" in header_line
        assert "judge_score" in header_line
        assert "kl_divergence" not in header_line
        assert "perplexity_wikitext" not in header_line
        assert "agentic_accuracy" not in header_line


class TestWriteSummary:
    def test_write_summary_creates_files(self, tmp_path: Path) -> None:
        """write_summary orchestrates all three artifacts and returns their paths."""
        result_dir = tmp_path / "results" / "cell"
        paths = write_summary(make_recipe(), [make_score()], result_dir)
        assert set(paths) == {"summary_csv", "summary_json", "readme"}
        for p in paths.values():
            assert p.exists()
        assert (result_dir / "summary.csv").exists()
        assert (result_dir / "summary.json").exists()
        assert (result_dir / "README.md").exists()
        # The CSV is loadable and has the legacy header.
        with (result_dir / "summary.csv").open(newline="") as f:
            assert next(csv.reader(f))[0] == "cell_id"