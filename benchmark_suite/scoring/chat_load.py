"""sauce chat phase — concurrent diverse chat completions on a 16/8/4/2/1 ladder."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

from benchmark_suite.recipe import ChatLoadSuite, Recipe, SauceChatSection
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus
from benchmark_suite.scoring.chat_http import (
    ChatCompletionResult,
    auth_headers,
    chat_completions_url,
    run_concurrent_wave,
)
from benchmark_suite.workloads.chat_catalog import (
    assemble_chat_messages,
    prompts_for_concurrency,
    validate_chat_catalog,
)
from benchmark_suite.workloads.tokens import content_tokens


def _p99(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, math.ceil(0.99 * len(ys)) - 1))
    return ys[idx]


def _wave_metrics(
    results: list[ChatCompletionResult], wall_s: float, concurrency: int
) -> dict[str, float | int]:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    out_tok = sum(r.completion_tokens for r in ok)
    ttfts = [r.ttft_ms for r in ok]
    latencies = [r.latency_ms for r in ok]
    tpot: list[float] = []
    for r in ok:
        if r.completion_tokens > 1:
            tpot.append((r.latency_ms - r.ttft_ms) / (r.completion_tokens - 1))
        elif r.completion_tokens == 1:
            tpot.append(0.0)
    wall = wall_s if wall_s > 0 else 1e-9
    return {
        "concurrency": concurrency,
        "output_tok_s": out_tok / wall,
        "ttft_mean_ms": float(statistics.fmean(ttfts)) if ttfts else 0.0,
        "ttft_median_ms": float(statistics.median(ttfts)) if ttfts else 0.0,
        "ttft_p99_ms": _p99(ttfts),
        "tpot_mean_ms": float(statistics.fmean(tpot)) if tpot else 0.0,
        "duration_s": wall_s,
        "successful": len(ok),
        "failed": failed,
        "input_tokens": sum(r.prompt_tokens for r in ok),
        "output_tokens": out_tok,
        "mean_latency_ms": float(statistics.fmean(latencies)) if latencies else 0.0,
    }


class ChatLoadScorerImpl(Scorer):
    """Concurrent chat/completions using the shipped diverse prompt catalog."""

    kind = "chat_load"

    def __init__(self, config: SauceChatSection) -> None:
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
        try:
            validate_chat_catalog()
            url = chat_completions_url(endpoint_url or recipe.endpoint.url)
            api_key = os.environ.get(recipe.endpoint.api_key_env, "")
            headers = auth_headers(api_key)
            artifacts_dir = result_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            per_suite: list[dict[str, Any]] = []
            artifacts: dict[str, str] = {}
            skipped: list[str] = []

            timeout = recipe.endpoint.timeout_s
            with httpx.Client(timeout=timeout) as client:
                for suite in self.config.suites:
                    suite_row = self._run_suite(
                        recipe,
                        client=client,
                        url=url,
                        headers=headers,
                        suite=suite,
                        artifacts_dir=artifacts_dir,
                        artifacts=artifacts,
                        skipped=skipped,
                    )
                    if suite_row is not None:
                        per_suite.append(suite_row)

            if not per_suite:
                return ScoreRecord(
                    kind=self.kind,
                    cell_id=cell_id,
                    status=ScoreStatus.FAILURE,
                    started_at=started,
                    finished_at=datetime.now(UTC),
                    metrics={},
                    artifacts=artifacts,
                    error="no chat_load suite ran (all skipped or empty)",
                    notes={"skipped_suites": skipped},
                )

            # Aggregate: best output tok/s across waves; fill legacy throughput columns.
            all_waves: list[dict[str, Any]] = []
            for row in per_suite:
                waves_obj = row.get("waves", [])
                if isinstance(waves_obj, list):
                    for wave in cast(list[object], waves_obj):
                        if isinstance(wave, dict):
                            all_waves.append(cast(dict[str, Any], wave))
            successful = sum(int(w["successful"]) for w in all_waves)
            failed = sum(int(w["failed"]) for w in all_waves)
            tpot_vals = [
                float(w["tpot_mean_ms"])
                for w in all_waves
                if float(w["tpot_mean_ms"]) > 0
            ]
            metrics: dict[str, float | int | str] = {
                "output_tok_s": max(float(w["output_tok_s"]) for w in all_waves),
                "ttft_mean_ms": min(float(w["ttft_mean_ms"]) for w in all_waves),
                "ttft_median_ms": min(float(w["ttft_median_ms"]) for w in all_waves),
                "ttft_p99_ms": min(float(w["ttft_p99_ms"]) for w in all_waves),
                "tpot_mean_ms": min(tpot_vals) if tpot_vals else 0.0,
                "duration_s": sum(float(w["duration_s"]) for w in all_waves),
                "successful": successful,
                "failed": failed,
                "concurrency": max(int(w["concurrency"]) for w in all_waves),
            }
            status = ScoreStatus.SUCCESS if successful > 0 else ScoreStatus.FAILURE
            return ScoreRecord(
                kind=self.kind,
                cell_id=cell_id,
                status=status,
                started_at=started,
                finished_at=datetime.now(UTC),
                metrics=metrics,
                artifacts=artifacts,
                error=None if successful else "all chat_load requests failed",
                notes={"suites": per_suite, "skipped_suites": skipped},
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

    def _run_suite(
        self,
        recipe: Recipe,
        *,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        suite: ChatLoadSuite,
        artifacts_dir: Path,
        artifacts: dict[str, str],
        skipped: list[str],
    ) -> dict[str, Any] | None:
        need = suite.input_tokens + suite.output_tokens
        if recipe.resources.max_model_len < need:
            skipped.append(
                f"{suite.name}: needs {need} tokens, max_model_len="
                f"{recipe.resources.max_model_len}"
            )
            return None
        waves: list[dict[str, Any]] = []
        for conc in self.config.ladder:
            specs = prompts_for_concurrency(conc)
            items: list[tuple[str, list[dict[str, str]]]] = []
            for spec in specs:
                messages = assemble_chat_messages(spec, input_tokens=suite.input_tokens)
                items.append((spec.prompt_id, messages))
            t0 = time.perf_counter()
            results = run_concurrent_wave(
                client,
                url=url,
                model=recipe.endpoint.model_name,
                items=items,
                max_tokens=suite.output_tokens,
                temperature=self.config.temperature,
                stream=self.config.stream,
                headers=headers,
            )
            wall = time.perf_counter() - t0
            row = _wave_metrics(results, wall, conc)
            row["input_tokens_est"] = sum(content_tokens(m) for _, m in items)
            waves.append(row)
            rel = f"artifacts/chat_load_{suite.name}_c{conc}.json"
            payload = {
                "suite": suite.name,
                "concurrency": conc,
                "input_tokens": suite.input_tokens,
                "output_tokens": suite.output_tokens,
                "results": [
                    {
                        "prompt_id": r.prompt_id,
                        "ok": r.ok,
                        "error": r.error,
                        "ttft_ms": r.ttft_ms,
                        "latency_ms": r.latency_ms,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                    }
                    for r in results
                ],
            }
            (artifacts_dir / f"chat_load_{suite.name}_c{conc}.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            artifacts[f"chat_load_{suite.name}_c{conc}.json"] = rel
        return {"name": suite.name, "waves": waves}
