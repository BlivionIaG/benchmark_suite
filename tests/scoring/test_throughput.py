"""Tests for benchmark_suite.scoring.throughput."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark_suite.recipe import (
    BenchSection,
    Recipe,
    ThroughputScorer,
)
from benchmark_suite.runner.llm_perf import LLMPerfResult
from benchmark_suite.scoring.base import (
    Scorer,
    ScoreRecord,
    ScorerRegistry,
    ScoreStatus,
)
from benchmark_suite.scoring.throughput import ThroughputScorerImpl


class TestScoreRecord:
    def test_to_dict_roundtrip(self) -> None:
        """ScoreRecord.to_dict must preserve all fields including datetimes."""
        started = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
        finished = datetime(2026, 8, 30, 12, 1, 0, tzinfo=UTC)
        record = ScoreRecord(
            kind="throughput",
            cell_id="dense_fardna2_rdna2_cg1_mtp0",
            status=ScoreStatus.SUCCESS,
            started_at=started,
            finished_at=finished,
            metrics={"output_tok_s": 76.5, "ttft_mean_ms": 120.0},
            artifacts={"artifacts/llm_perf_c1.json": "artifacts/llm_perf_c1.json"},
            notes={"concurrencies_tested": 1},
        )
        d = record.to_dict()
        assert d["kind"] == "throughput"
        assert d["cell_id"] == "dense_fardna2_rdna2_cg1_mtp0"
        assert d["status"] == ScoreStatus.SUCCESS
        assert d["started_at"] == "2026-08-30T12:00:00+00:00"
        assert d["finished_at"] == "2026-08-30T12:01:00+00:00"
        assert d["metrics"]["output_tok_s"] == 76.5
        assert d["metrics"]["ttft_mean_ms"] == 120.0
        assert d["artifacts"] == {
            "artifacts/llm_perf_c1.json": "artifacts/llm_perf_c1.json"
        }
        assert d["notes"]["concurrencies_tested"] == 1
        assert d["error"] is None

    def test_to_dict_with_error(self) -> None:
        """ScoreRecord.to_dict must include error field when set."""
        started = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)
        finished = datetime(2026, 8, 30, 12, 0, 5, tzinfo=UTC)
        record = ScoreRecord(
            kind="throughput",
            cell_id="dense_fardna2_rdna2_cg1_mtp0",
            status=ScoreStatus.FAILURE,
            started_at=started,
            finished_at=finished,
            metrics={},
            error="FileNotFoundError: llm-perf binary not found",
        )
        d = record.to_dict()
        assert d["status"] == ScoreStatus.FAILURE
        assert "FileNotFoundError" in d["error"]


class TestScorerRegistry:
    def test_register_and_get(self) -> None:
        """@scorer decorator registers the class; get returns it."""

        @ScorerRegistry.register
        class DummyScorer(Scorer):
            kind = "dummy"
            pass

        assert ScorerRegistry.get("dummy") is DummyScorer

    def test_register_validates_kind(self) -> None:
        """register must reject a class with no kind."""

        class NoKindScorer(Scorer):
            pass

        with pytest.raises(ValueError, match="must have a non-empty kind"):
            ScorerRegistry.register(NoKindScorer)

    def test_unknown_kind_returns_none(self) -> None:
        """get() returns None for an unregistered kind."""
        assert ScorerRegistry.get("nonexistent") is None

    def test_build_scorers_empty(self) -> None:
        """build_scorers returns [] when bench.scoring is empty."""
        bench = BenchSection(scoring=[])
        assert ScorerRegistry.build_scorers(bench) == []

    def test_build_scorers_returns_one_throughput(self) -> None:
        """build_scorers returns one ThroughputScorerImpl for a ThroughputScorer config."""
        scorer_cfg = ThroughputScorer(tool="llm-perf")
        bench = BenchSection(scoring=[scorer_cfg])
        scorers = ScorerRegistry.build_scorers(bench)
        assert len(scorers) == 1
        assert isinstance(scorers[0], ThroughputScorerImpl)


class TestThroughputScorerImpl:
    def test_maps_llm_perf_fields_correctly(
        self,
        simple_recipe: Recipe,
        mock_llm_perf_result: LLMPerfResult,
        tmp_result_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ThroughputScorer maps LLMPerfResult to legacy metrics with best-of aggregation."""
        import benchmark_suite.scoring.throughput as tp_module

        def fake_run(recipe: Recipe, *, concurrency: int, output_json_path: Path) -> LLMPerfResult:
            return mock_llm_perf_result

        monkeypatch.setattr(tp_module, "run_llm_perf_bench", fake_run)

        impl = ThroughputScorerImpl(ThroughputScorer(tool="llm-perf"))
        record = impl.score(simple_recipe, result_dir=tmp_result_dir)

        assert record.kind == "throughput"
        assert record.status == ScoreStatus.SUCCESS
        assert record.cell_id == "dense_triton_triton_cg0_mtp0"
        # Check best-of aggregation
        assert record.metrics["best_output_tok_s"] == 76.5
        assert record.metrics["best_total_tok_s"] == 200.0
        assert record.metrics["min_ttft_mean_ms"] == 120.0
        assert record.metrics["min_tpot_mean_ms"] == 15.0
        assert record.metrics["concurrencies_tested"] == 3
        # Per-concurrency rows stored in notes
        assert len(record.notes["per_concurrency"]) == 3
        assert record.notes["per_concurrency"][0]["concurrency"] == 1
        assert record.notes["tool"] == "llm-perf"

    def test_handles_vllm_bench_tool(
        self,
        simple_recipe: Recipe,
        tmp_result_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ThroughputScorer uses run_vllm_bench when tool="vllm-bench"."""
        import benchmark_suite.scoring.throughput as tp_module
        from benchmark_suite.runner.vllm_bench import VLLMBenchResult

        fake_result = VLLMBenchResult(
            raw_stdout="",
            output_tok_s=50.0,
            peak_output_tok_s=60.0,
            total_tok_s=120.0,
            ttft_mean_ms=200.0,
            ttft_median_ms=190.0,
            ttft_p99_ms=300.0,
            tpot_mean_ms=20.0,
            tpot_median_ms=18.0,
            duration_s=10.0,
            successful=32,
            failed=0,
        )

        def fake_run(recipe: Recipe, *, endpoint_url: str, concurrency: int) -> VLLMBenchResult:
            return fake_result

        monkeypatch.setattr(tp_module, "run_vllm_bench", fake_run)

        impl = ThroughputScorerImpl(ThroughputScorer(tool="vllm-bench"))
        record = impl.score(simple_recipe, result_dir=tmp_result_dir)

        assert record.status == ScoreStatus.SUCCESS
        assert record.metrics["best_output_tok_s"] == 50.0
        assert record.metrics["min_ttft_mean_ms"] == 200.0
        assert record.notes["tool"] == "vllm-bench"

    def test_handles_binary_missing_gracefully(
        self,
        simple_recipe: Recipe,
        tmp_result_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """FileNotFoundError from runner produces FAILURE status with informative error."""
        import benchmark_suite.scoring.throughput as tp_module

        def fake_missing(recipe: Recipe, *, concurrency: int,
            output_json_path: Path) -> LLMPerfResult:
            raise FileNotFoundError("llm-perf binary not found: llm-perf")

        monkeypatch.setattr(tp_module, "run_llm_perf_bench", fake_missing)

        impl = ThroughputScorerImpl(ThroughputScorer(tool="llm-perf"))
        record = impl.score(simple_recipe, result_dir=tmp_result_dir)

        assert record.status == ScoreStatus.FAILURE
        assert record.error is not None and "FileNotFoundError" in record.error
        assert record.metrics == {}

    def test_one_row_per_concurrency(
        self,
        simple_recipe: Recipe,
        mock_llm_perf_result: LLMPerfResult,
        tmp_result_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scorer calls runner once per concurrency value."""
        import benchmark_suite.scoring.throughput as tp_module

        calls: list[int] = []

        def fake_run(recipe: Recipe, *, concurrency: int, output_json_path: Path) -> LLMPerfResult:
            calls.append(concurrency)
            return mock_llm_perf_result

        monkeypatch.setattr(tp_module, "run_llm_perf_bench", fake_run)

        impl = ThroughputScorerImpl(ThroughputScorer(tool="llm-perf"))
        impl.score(simple_recipe, result_dir=tmp_result_dir)

        assert sorted(calls) == [1, 4, 8]
        # Also verify per_concurrency has 3 rows
        impl2 = ThroughputScorerImpl(ThroughputScorer(tool="llm-perf"))
        record = impl2.score(simple_recipe, result_dir=tmp_result_dir)
        assert len(record.notes["per_concurrency"]) == 3

    def test_min_ttft_aggregation(
        self,
        simple_recipe: Recipe,
        tmp_result_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """min_ttft_mean_ms selects the minimum across concurrencies."""
        import benchmark_suite.scoring.throughput as tp_module

        # Three different TTFT values
        results = [
            LLMPerfResult(
                raw={},
                output_tok_s=50.0,
                total_tok_s=100.0,
                peak_output_tok_s=50.0,
                ttft_mean_ms=200.0,
                ttft_median_ms=200.0,
                ttft_p99_ms=200.0,
                tpot_mean_ms=20.0,
                tpot_median_ms=20.0,
                duration_s=10.0,
                successful=10,
                failed=0,
            ),
            LLMPerfResult(
                raw={},
                output_tok_s=50.0,
                total_tok_s=100.0,
                peak_output_tok_s=50.0,
                ttft_mean_ms=100.0,
                ttft_median_ms=100.0,
                ttft_p99_ms=100.0,
                tpot_mean_ms=20.0,
                tpot_median_ms=20.0,
                duration_s=10.0,
                successful=10,
                failed=0,
            ),
            LLMPerfResult(
                raw={},
                output_tok_s=50.0,
                total_tok_s=100.0,
                peak_output_tok_s=50.0,
                ttft_mean_ms=300.0,
                ttft_median_ms=300.0,
                ttft_p99_ms=300.0,
                tpot_mean_ms=20.0,
                tpot_median_ms=20.0,
                duration_s=10.0,
                successful=10,
                failed=0,
            ),
        ]
        idx = [0]

        def fake_run(recipe: Recipe, *, concurrency: int, output_json_path: Path) -> LLMPerfResult:
            nonlocal idx
            res = results[idx[0] % len(results)]
            idx[0] += 1
            return res

        monkeypatch.setattr(tp_module, "run_llm_perf_bench", fake_run)

        impl = ThroughputScorerImpl(ThroughputScorer(tool="llm-perf"))
        record = impl.score(simple_recipe, result_dir=tmp_result_dir)

        assert record.metrics["min_ttft_mean_ms"] == 100.0
