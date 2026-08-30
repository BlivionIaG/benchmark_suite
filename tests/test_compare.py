"""Tests for benchmark_suite.compare — result-dir delta tables."""
from __future__ import annotations

import csv
from pathlib import Path

from benchmark_suite.compare import (
    _is_regression,  # pyright: ignore[reportPrivateUsage]
    _metric_direction,  # pyright: ignore[reportPrivateUsage]
    compare_results,
)
from benchmark_suite.recipe import Recipe
from benchmark_suite.report import write_summary_csv, write_summary_json
from benchmark_suite.scoring.base import ScoreRecord, ScoreStatus


def make_result_dir(tmp_path: Path, name: str, metrics: dict[str, float | int | str]) -> Path:
    """Result dir containing one summary.json with a single scored metric set."""
    d = tmp_path / name
    d.mkdir(parents=True)
    recipe = Recipe.model_validate(
        {"meta": {"name": f"recipe-{name}", "description": "compare test"}}
    )
    score = ScoreRecord(
        kind="throughput",
        cell_id="cell_a",
        status=ScoreStatus.SUCCESS,
        metrics=metrics,
    )
    write_summary_json(recipe, [score], d / "summary.json")
    return d


def read_delta_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["metric"]: row for row in csv.DictReader(f)}


class TestMetricDirection:
    def test_metric_direction_higher_better(self) -> None:
        assert _metric_direction("output_tok_s") == "higher_better"
        assert _metric_direction("total_tok_s") == "higher_better"
        assert _metric_direction("judge_score") == "higher_better"
        assert _metric_direction("successful") == "higher_better"
        assert _metric_direction("agentic_accuracy") == "higher_better"

    def test_metric_direction_lower_better(self) -> None:
        assert _metric_direction("ttft_mean_ms") == "lower_better"
        assert _metric_direction("ttft_p99_ms") == "lower_better"
        assert _metric_direction("tpot_median_ms") == "lower_better"
        assert _metric_direction("duration_s") == "lower_better"
        assert _metric_direction("kl_divergence") == "lower_better"
        assert _metric_direction("perplexity_wikitext") == "lower_better"

    def test_metric_direction_neutral(self) -> None:
        assert _metric_direction("failed") == "neutral"
        assert _metric_direction("cell_id") == "neutral"
        assert _metric_direction("scorer_kind") == "neutral"
        assert _metric_direction("status") == "neutral"
        assert _metric_direction("concurrency") == "neutral"
        assert _metric_direction("started_at") == "neutral"
        assert _metric_direction("finished_at") == "neutral"
        assert _metric_direction("peak_output_tok_s") == "neutral"

    def test_is_regression_directional(self) -> None:
        """_is_regression flags only movement in the metric's bad direction."""
        assert _is_regression("output_tok_s", -10.0) is True
        assert _is_regression("output_tok_s", 10.0) is False
        assert _is_regression("ttft_mean_ms", 10.0) is True
        assert _is_regression("ttft_mean_ms", -10.0) is False
        assert _is_regression("failed", 10.0) is False
        assert _is_regression("peak_output_tok_s", -10.0) is False


class TestCompareResults:
    def test_compare_two_results_basic_delta(self, tmp_path: Path) -> None:
        """delta = b - a and pct_change = (b - a) / a * 100 for shared metrics."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 100.0, "ttft_mean_ms": 100.0})
        dir_b = make_result_dir(tmp_path, "b", {"output_tok_s": 110.0, "ttft_mean_ms": 90.0})
        paths = compare_results(dir_a, dir_b)
        rows = read_delta_csv(paths["csv"])
        assert rows["output_tok_s"]["value_a"] == "100.0"
        assert rows["output_tok_s"]["value_b"] == "110.0"
        assert float(rows["output_tok_s"]["delta"]) == 10.0
        assert float(rows["output_tok_s"]["pct_change"]) == 10.0
        assert float(rows["ttft_mean_ms"]["delta"]) == -10.0
        assert float(rows["ttft_mean_ms"]["pct_change"]) == -10.0

    def test_compare_regression_higher_better(self, tmp_path: Path) -> None:
        """A drop in a higher_better metric is flagged as a regression."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 100.0})
        dir_b = make_result_dir(tmp_path, "b", {"output_tok_s": 90.0})
        paths = compare_results(dir_a, dir_b)
        rows = read_delta_csv(paths["csv"])
        assert rows["output_tok_s"]["direction"] == "higher_better"
        assert rows["output_tok_s"]["regression"] == "yes"

    def test_compare_regression_lower_better(self, tmp_path: Path) -> None:
        """A rise in a lower_better metric is a regression; a drop is not."""
        dir_a = make_result_dir(tmp_path, "a", {"ttft_mean_ms": 100.0})
        dir_b = make_result_dir(tmp_path, "b", {"ttft_mean_ms": 120.0})
        rows = read_delta_csv(compare_results(dir_a, dir_b)["csv"])
        assert rows["ttft_mean_ms"]["direction"] == "lower_better"
        assert rows["ttft_mean_ms"]["regression"] == "yes"

        dir_c = make_result_dir(tmp_path, "c", {"ttft_mean_ms": 80.0})
        rows_improved = read_delta_csv(compare_results(dir_a, dir_c)["csv"])
        assert rows_improved["ttft_mean_ms"]["regression"] == "no"

    def test_compare_handles_zero_baseline(self, tmp_path: Path) -> None:
        """a == 0 → pct_change is the literal string "inf", delta still numeric."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 0.0})
        dir_b = make_result_dir(tmp_path, "b", {"output_tok_s": 5.0})
        paths = compare_results(dir_a, dir_b)
        rows = read_delta_csv(paths["csv"])
        assert rows["output_tok_s"]["pct_change"] == "inf"
        assert float(rows["output_tok_s"]["delta"]) == 5.0
        assert rows["output_tok_s"]["regression"] == "no"

    def test_compare_markdown_format(self, tmp_path: Path) -> None:
        """delta.md is a markdown table with the delta.csv column set."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 100.0})
        dir_b = make_result_dir(tmp_path, "b", {"output_tok_s": 110.0})
        paths = compare_results(dir_a, dir_b)
        content = paths["md"].read_text()
        assert "| metric |" in content
        assert "|---" in content
        assert "output_tok_s" in content
        assert "higher_better" in content
        assert "yes" in content or "no" in content

    def test_compare_returns_paths(self, tmp_path: Path) -> None:
        """Returns {"csv", "md"} paths; default output dir is dir_b."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 100.0})
        dir_b = make_result_dir(tmp_path, "b", {"output_tok_s": 110.0})
        paths = compare_results(dir_a, dir_b)
        assert set(paths) == {"csv", "md"}
        assert paths["csv"].name == "delta.csv"
        assert paths["md"].name == "delta.md"
        assert paths["csv"].parent == dir_b
        assert paths["md"].parent == dir_b
        assert paths["csv"].exists()
        assert paths["md"].exists()

    def test_compare_respects_explicit_output_dir(self, tmp_path: Path) -> None:
        """output_dir overrides the default dir_b destination."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 100.0})
        dir_b = make_result_dir(tmp_path, "b", {"output_tok_s": 110.0})
        out = tmp_path / "delta_out"
        paths = compare_results(dir_a, dir_b, output_dir=out)
        assert paths["csv"].parent == out
        assert paths["csv"].exists()

    def test_compare_csv_fallback_when_no_json(self, tmp_path: Path) -> None:
        """Dirs with only summary.csv still compare (numeric columns only)."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        write_summary_csv(
            [ScoreRecord(kind="throughput", cell_id="c", status=ScoreStatus.SUCCESS,
                          metrics={"output_tok_s": 100.0})],
            dir_a / "summary.csv",
        )
        write_summary_csv(
            [ScoreRecord(kind="throughput", cell_id="c", status=ScoreStatus.SUCCESS,
                          metrics={"output_tok_s": 120.0})],
            dir_b / "summary.csv",
        )
        paths = compare_results(dir_a, dir_b)
        rows = read_delta_csv(paths["csv"])
        assert float(rows["output_tok_s"]["delta"]) == 20.0
        assert float(rows["output_tok_s"]["pct_change"]) == 20.0

    def test_compare_missing_summary_raises(self, tmp_path: Path) -> None:
        """A dir with neither summary.json nor summary.csv fails loudly."""
        dir_a = make_result_dir(tmp_path, "a", {"output_tok_s": 100.0})
        empty = tmp_path / "empty"
        empty.mkdir()
        try:
            compare_results(dir_a, empty)
        except FileNotFoundError as exc:
            assert "summary" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError for missing summaries")