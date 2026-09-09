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
    request_row,
    run_concurrent_wave,
)
from benchmark_suite.scoring.kv_metrics import fetch_kv_cache_perc, metrics_url
from benchmark_suite.scoring.perf import compact, latency_rates
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


def _fmean(xs: list[float]) -> float | None:
    return float(statistics.fmean(xs)) if xs else None


def _max_num(waves: list[dict[str, Any]], key: str) -> float | None:
    vals: list[float] = []
    for wave in waves:
        value = wave.get(key)
        if isinstance(value, (int, float)):
            vals.append(float(value))
    return max(vals) if vals else None


def _sum_int(waves: list[dict[str, Any]], key: str) -> int:
    total = 0
    for wave in waves:
        value = wave.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def _wave_metrics(
    results: list[ChatCompletionResult],
    wall_s: float,
    concurrency: int,
    *,
    kv_cache_perc: float | None = None,
) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    out_tok = sum(r.completion_tokens for r in ok)
    in_tok = sum(r.prompt_tokens for r in ok)
    ttfts = [r.ttft_ms for r in ok]
    latencies = [r.latency_ms for r in ok]
    prefills: list[float] = []
    decodes: list[float] = []
    tpots: list[float] = []
    cached_sum = 0
    cached_any = False
    for r in ok:
        rates = latency_rates(
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            ttft_ms=r.ttft_ms,
            latency_ms=r.latency_ms,
        )
        if rates.prefill_tok_s is not None:
            prefills.append(rates.prefill_tok_s)
        if rates.decode_tok_s is not None:
            decodes.append(rates.decode_tok_s)
        if rates.tpot_ms is not None:
            tpots.append(rates.tpot_ms)
        if r.cached_tokens is not None:
            cached_sum += r.cached_tokens
            cached_any = True
    wall = wall_s if wall_s > 0 else 1e-9
    row: dict[str, Any] = {
        "concurrency": concurrency,
        "output_tok_s": out_tok / wall,
        "input_tok_s": in_tok / wall,
        "ttft_mean_ms": float(statistics.fmean(ttfts)) if ttfts else 0.0,
        "ttft_median_ms": float(statistics.median(ttfts)) if ttfts else 0.0,
        "ttft_p99_ms": _p99(ttfts),
        "tpot_mean_ms": float(statistics.fmean(tpots)) if tpots else 0.0,
        "tpot_median_ms": float(statistics.median(tpots)) if tpots else 0.0,
        "duration_s": wall_s,
        "successful": len(ok),
        "failed": failed,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "mean_latency_ms": float(statistics.fmean(latencies)) if latencies else 0.0,
    }
    prefill_mean = _fmean(prefills)
    decode_mean = _fmean(decodes)
    if prefill_mean is not None:
        row["prefill_tok_s"] = prefill_mean
    if decode_mean is not None:
        row["decode_tok_s"] = decode_mean
    if cached_any:
        row["cached_tokens"] = cached_sum
    if kv_cache_perc is not None:
        row["kv_cache_perc"] = kv_cache_perc
    return row


def _set_if(
    metrics: dict[str, float | int | str], key: str, value: float | int | None
) -> None:
    if value is not None:
        metrics[key] = value


class ChatLoadScorerImpl(Scorer):
    """Concurrent chat/completions using the shipped diverse prompt catalog."""

    kind = "chat_load"

    def __init__(
        self,
        config: SauceChatSection,
        *,
        kv_metrics: bool = True,
        kv_metrics_path: str = "/metrics",
    ) -> None:
        self.config = config
        self.kv_metrics = kv_metrics
        self.kv_metrics_path = kv_metrics_path

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
            base = endpoint_url or recipe.endpoint.url
            url = chat_completions_url(base)
            kv_url = metrics_url(base, self.kv_metrics_path)
            api_key = os.environ.get(recipe.endpoint.api_key_env, "")
            headers = auth_headers(api_key)
            artifacts_dir = result_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            per_suite: list[dict[str, Any]] = []
            artifacts: dict[str, str] = {}
            skipped: list[str] = []
            events: list[dict[str, Any]] = []
            samples: list[dict[str, Any]] = []

            timeout = recipe.endpoint.timeout_s
            with httpx.Client(timeout=timeout) as client:
                for suite in self.config.suites:
                    suite_row = self._run_suite(
                        recipe,
                        client=client,
                        url=url,
                        kv_url=kv_url,
                        headers=headers,
                        suite=suite,
                        artifacts_dir=artifacts_dir,
                        artifacts=artifacts,
                        skipped=skipped,
                        events=events,
                        samples=samples,
                    )
                    if suite_row is not None:
                        per_suite.append(suite_row)

            if events or samples:
                ts_name = "chat_timeseries.json"
                (artifacts_dir / ts_name).write_text(
                    json.dumps({"kind": "chat", "events": events, "samples": samples}, indent=2)
                    + "\n"
                )
                artifacts[ts_name] = f"artifacts/{ts_name}"

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
            tpot_medians = [
                float(w["tpot_median_ms"])
                for w in all_waves
                if float(w.get("tpot_median_ms") or 0) > 0
            ]
            metrics: dict[str, float | int | str] = {
                "output_tok_s": max(float(w["output_tok_s"]) for w in all_waves),
                "ttft_mean_ms": min(float(w["ttft_mean_ms"]) for w in all_waves),
                "ttft_median_ms": min(float(w["ttft_median_ms"]) for w in all_waves),
                "ttft_p99_ms": min(float(w["ttft_p99_ms"]) for w in all_waves),
                "tpot_mean_ms": min(tpot_vals) if tpot_vals else 0.0,
                "tpot_median_ms": min(tpot_medians) if tpot_medians else 0.0,
                "duration_s": sum(float(w["duration_s"]) for w in all_waves),
                "successful": successful,
                "failed": failed,
                "concurrency": max(int(w["concurrency"]) for w in all_waves),
            }
            _set_if(metrics, "input_tok_s", _max_num(all_waves, "input_tok_s"))
            _set_if(metrics, "prefill_tok_s", _max_num(all_waves, "prefill_tok_s"))
            _set_if(metrics, "decode_tok_s", _max_num(all_waves, "decode_tok_s"))
            cached_total = _sum_int(all_waves, "cached_tokens")
            if any("cached_tokens" in w for w in all_waves):
                metrics["cached_tokens"] = cached_total
            _set_if(metrics, "kv_cache_perc", _max_num(all_waves, "kv_cache_perc"))
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
        kv_url: str,
        headers: dict[str, str],
        suite: ChatLoadSuite,
        artifacts_dir: Path,
        artifacts: dict[str, str],
        skipped: list[str],
        events: list[dict[str, Any]],
        samples: list[dict[str, Any]],
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
            kv: float | None = None
            if self.kv_metrics:
                kv = fetch_kv_cache_perc(client, kv_url)
            row = _wave_metrics(results, wall, conc, kv_cache_perc=kv)
            row["input_tokens_est"] = sum(content_tokens(m) for _, m in items)
            waves.append(row)
            if kv is not None:
                samples.append(
                    compact(
                        {
                            "phase": "chat",
                            "suite": suite.name,
                            "concurrency": conc,
                            "t_unix_ms": int(time.time() * 1000),
                            "kv_cache_perc": kv,
                        }
                    )
                )
            for r in results:
                events.append(
                    compact(
                        {
                            "phase": "chat",
                            "suite": suite.name,
                            "concurrency": conc,
                            **request_row(r, kv_cache_perc=kv),
                        }
                    )
                )
            rel = f"artifacts/chat_load_{suite.name}_c{conc}.json"
            payload = compact(
                {
                    "suite": suite.name,
                    "concurrency": conc,
                    "input_tokens": suite.input_tokens,
                    "output_tokens": suite.output_tokens,
                    "kv_cache_perc": kv,
                    "metrics": row,
                    "results": [request_row(r, kv_cache_perc=kv) for r in results],
                }
            )
            (artifacts_dir / f"chat_load_{suite.name}_c{conc}.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            artifacts[f"chat_load_{suite.name}_c{conc}.json"] = rel
        return {"name": suite.name, "waves": waves}
