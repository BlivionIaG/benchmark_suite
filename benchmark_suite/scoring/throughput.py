"""benchmark_suite/scoring/throughput.py — throughput scorer via llm-perf or vllm-bench.

Maps LLMPerfResult / VLLMBenchResult fields to ScoreRecord.metrics using the
parent's legacy column names byte-for-byte so summary.csv stays compatible.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_suite.recipe import Recipe, ThroughputScorer
from benchmark_suite.runner.llm_perf import LLMPerfResult, run_llm_perf_bench
from benchmark_suite.runner.vllm_bench import VLLMBenchResult, run_vllm_bench
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer


@scorer
class ThroughputScorerImpl(Scorer):
    """Runs the throughput scorer: per-concurrency bench via llm-perf or vllm-bench."""

    kind = "throughput"

    def __init__(self, config: ThroughputScorer) -> None:
        self.config = config

    def score(
        self,
        recipe: Recipe,
        *,
        result_dir: Path,
        endpoint_url: str | None = None,
    ) -> ScoreRecord:
        started = datetime.now(UTC)
        cell_id = recipe.cell.render()
        endpoint = endpoint_url or recipe.endpoint.url

        try:
            per_concurrency: list[dict[str, Any]] = []
            artifacts: dict[str, str] = {}

            for c in recipe.bench.load.concurrencies:
                if self.config.tool == "llm-perf":
                    out_json = result_dir / "artifacts" / f"llm_perf_c{c}.json"
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    lr_res: LLMPerfResult = run_llm_perf_bench(
                        recipe,
                        concurrency=c,
                        output_json_path=out_json,
                    )
                    row: dict[str, Any] = {
                        "concurrency": c,
                        "output_tok_s": lr_res.output_tok_s,
                        "peak_output_tok_s": lr_res.peak_output_tok_s,
                        "total_tok_s": lr_res.total_tok_s,
                        "ttft_mean_ms": lr_res.ttft_mean_ms,
                        "ttft_median_ms": lr_res.ttft_median_ms,
                        "ttft_p99_ms": lr_res.ttft_p99_ms,
                        "tpot_mean_ms": lr_res.tpot_mean_ms,
                        "tpot_median_ms": lr_res.tpot_median_ms,
                        "duration_s": lr_res.duration_s,
                        "successful": lr_res.successful,
                        "failed": lr_res.failed,
                    }
                    artifacts[f"llm_perf_c{c}.json"] = f"artifacts/llm_perf_c{c}.json"
                else:
                    # vllm-bench
                    vl_res: VLLMBenchResult = run_vllm_bench(
                        recipe, endpoint_url=endpoint, concurrency=c
                    )
                    row = {
                        "concurrency": c,
                        "output_tok_s": vl_res.output_tok_s,
                        "peak_output_tok_s": vl_res.peak_output_tok_s,
                        "total_tok_s": vl_res.total_tok_s,
                        "ttft_mean_ms": vl_res.ttft_mean_ms,
                        "ttft_median_ms": vl_res.ttft_median_ms,
                        "ttft_p99_ms": vl_res.ttft_p99_ms,
                        "tpot_mean_ms": vl_res.tpot_mean_ms,
                        "tpot_median_ms": vl_res.tpot_median_ms,
                        "duration_s": vl_res.duration_s,
                        "successful": vl_res.successful,
                        "failed": vl_res.failed,
                    }
                per_concurrency.append(row)

            # Aggregate: best (peak) per metric across concurrencies, plus per-c rows.
            metrics: dict[str, float | int | str] = {
                "concurrencies_tested": len(per_concurrency),
                "best_output_tok_s": max(r["output_tok_s"] for r in per_concurrency),
                "best_total_tok_s": max(r["total_tok_s"] for r in per_concurrency),
                "min_ttft_mean_ms": min(r["ttft_mean_ms"] for r in per_concurrency),
                "min_tpot_mean_ms": min(r["tpot_mean_ms"] for r in per_concurrency),
            }

            return ScoreRecord(
                kind=self.kind,
                cell_id=cell_id,
                status=ScoreStatus.SUCCESS,
                started_at=started,
                finished_at=datetime.now(UTC),
                metrics=metrics,
                artifacts=artifacts,
                notes={"per_concurrency": per_concurrency, "tool": self.config.tool},
            )

        except Exception as exc:
            return ScoreRecord(
                kind=self.kind,
                cell_id=cell_id,
                status=ScoreStatus.FAILURE,
                started_at=started,
                finished_at=datetime.now(UTC),
                metrics={},
                artifacts={},
                error=f"{type(exc).__name__}: {exc}",
            )
