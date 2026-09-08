"""sauce session phase — growing multi-turn coding-agent conversations."""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from benchmark_suite.recipe import Recipe, SauceSessionSection
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus
from benchmark_suite.scoring.chat_http import (
    auth_headers,
    chat_completions_url,
    post_chat_completion,
    request_row,
)
from benchmark_suite.scoring.kv_metrics import fetch_kv_cache_perc, metrics_url
from benchmark_suite.scoring.perf import compact
from benchmark_suite.workloads.session_catalog import (
    SESSION_SPECS,
    SESSION_SYSTEM_PROMPT,
    plan_turn_input_targets,
    session_followups,
    workspace_blob,
)
from benchmark_suite.workloads.tokens import content_tokens, pad_last_user_to_total


def _fmean(xs: list[float]) -> float | None:
    return float(statistics.fmean(xs)) if xs else None


def _set_if(
    metrics: dict[str, float | int | str], key: str, value: float | int | None
) -> None:
    if value is not None:
        metrics[key] = value


class SessionScorerImpl(Scorer):
    """Pi-style coding sessions whose context climbs toward ``max_context_tokens``."""

    kind = "session"

    def __init__(
        self,
        config: SauceSessionSection,
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
            reserve = max(self.config.output_reserve_tokens, self.config.output_tokens)
            cap = min(self.config.max_context_tokens, recipe.resources.max_model_len)
            budget = cap - reserve
            if budget < 256:
                return ScoreRecord(
                    kind=self.kind,
                    cell_id=cell_id,
                    status=ScoreStatus.FAILURE,
                    started_at=started,
                    finished_at=datetime.now(UTC),
                    metrics={},
                    error=(
                        f"session budget {budget} is too small "
                        f"(max_model_len={recipe.resources.max_model_len}, "
                        f"reserve={reserve})"
                    ),
                )
            max_turns = self.config.max_turns or 13
            targets = plan_turn_input_targets(budget=budget, max_turns=max_turns)
            specs = SESSION_SPECS[: self.config.n_sessions]
            base = endpoint_url or recipe.endpoint.url
            url = chat_completions_url(base)
            kv_url = metrics_url(base, self.kv_metrics_path)
            headers = auth_headers(os.environ.get(recipe.endpoint.api_key_env, ""))
            artifacts_dir = result_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            session_rows: list[dict[str, Any]] = []
            events: list[dict[str, Any]] = []
            samples: list[dict[str, Any]] = []
            t0 = time.perf_counter()
            with httpx.Client(timeout=recipe.endpoint.timeout_s) as client:
                for spec in specs:
                    session_rows.append(
                        self._run_session(
                            recipe,
                            client=client,
                            url=url,
                            kv_url=kv_url,
                            headers=headers,
                            spec_id=spec.session_id,
                            title=spec.title,
                            initial=spec.initial,
                            followups=session_followups(spec),
                            targets=targets,
                            cap=cap,
                            events=events,
                            samples=samples,
                        )
                    )
            wall = time.perf_counter() - t0
            ok_sessions = [s for s in session_rows if s["ok"]]
            total_turns = sum(int(s["turns"]) for s in session_rows)
            total_out = sum(int(s["output_tokens"]) for s in session_rows)
            max_in = max((int(s["max_input_tokens"]) for s in session_rows), default=0)
            rel = "artifacts/session_sessions.json"
            (artifacts_dir / "session_sessions.json").write_text(
                json.dumps(
                    {
                        "budget": budget,
                        "targets": targets,
                        "sessions": session_rows,
                    },
                    indent=2,
                )
                + "\n"
            )
            artifacts: dict[str, str] = {"session_sessions.json": rel}
            if events or samples:
                ts_name = "session_timeseries.json"
                (artifacts_dir / ts_name).write_text(
                    json.dumps(
                        {"kind": "session", "events": events, "samples": samples},
                        indent=2,
                    )
                    + "\n"
                )
                artifacts[ts_name] = f"artifacts/{ts_name}"

            turn_ttft: list[float] = []
            turn_tpot: list[float] = []
            turn_prefill: list[float] = []
            turn_decode: list[float] = []
            turn_cached: list[int] = []
            turn_kv: list[float] = []
            for session in session_rows:
                turns_obj = session.get("turns_detail", [])
                if not isinstance(turns_obj, list):
                    continue
                for turn in turns_obj:
                    if not isinstance(turn, dict):
                        continue
                    ttft = turn.get("ttft_ms")
                    if isinstance(ttft, (int, float)):
                        turn_ttft.append(float(ttft))
                    tpot = turn.get("tpot_ms")
                    if isinstance(tpot, (int, float)):
                        turn_tpot.append(float(tpot))
                    prefill = turn.get("prefill_tok_s")
                    if isinstance(prefill, (int, float)):
                        turn_prefill.append(float(prefill))
                    decode = turn.get("decode_tok_s")
                    if isinstance(decode, (int, float)):
                        turn_decode.append(float(decode))
                    cached = turn.get("cached_tokens")
                    if isinstance(cached, (int, float)):
                        turn_cached.append(int(cached))
                    kv = turn.get("kv_cache_perc")
                    if isinstance(kv, (int, float)):
                        turn_kv.append(float(kv))

            metrics: dict[str, float | int | str] = {
                "session_turns": total_turns,
                "session_success_rate": (
                    len(ok_sessions) / len(session_rows) if session_rows else 0.0
                ),
                "session_max_input_tokens": max_in,
                "successful": len(ok_sessions),
                "failed": len(session_rows) - len(ok_sessions),
                "duration_s": wall,
                "output_tok_s": total_out / wall if wall > 0 else 0.0,
            }
            ttft_mean = _fmean(turn_ttft)
            tpot_mean = _fmean(turn_tpot)
            prefill_mean = _fmean(turn_prefill)
            decode_mean = _fmean(turn_decode)
            _set_if(metrics, "session_ttft_mean_ms", ttft_mean)
            _set_if(metrics, "session_tpot_mean_ms", tpot_mean)
            _set_if(metrics, "session_prefill_tok_s", prefill_mean)
            _set_if(metrics, "session_decode_tok_s", decode_mean)
            if turn_cached:
                metrics["session_cached_tokens"] = sum(turn_cached)
            kv_peak = max(turn_kv) if turn_kv else None
            _set_if(metrics, "session_kv_cache_perc", kv_peak)
            # Session-only runs fill the unprefixed columns so summary.csv has TTFT/KV.
            _set_if(metrics, "ttft_mean_ms", ttft_mean)
            _set_if(metrics, "tpot_mean_ms", tpot_mean)
            _set_if(metrics, "prefill_tok_s", prefill_mean)
            _set_if(metrics, "decode_tok_s", decode_mean)
            _set_if(metrics, "kv_cache_perc", kv_peak)
            status = (
                ScoreStatus.SUCCESS if ok_sessions else ScoreStatus.FAILURE
            )
            return ScoreRecord(
                kind=self.kind,
                cell_id=cell_id,
                status=status,
                started_at=started,
                finished_at=datetime.now(UTC),
                metrics=metrics,
                artifacts=artifacts,
                error=None if ok_sessions else "all sessions failed",
                notes={
                    "sessions": session_rows,
                    "budget": budget,
                    "turn_targets": targets,
                },
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

    def _run_session(
        self,
        recipe: Recipe,
        *,
        client: httpx.Client,
        url: str,
        kv_url: str,
        headers: dict[str, str],
        spec_id: str,
        title: str,
        initial: str,
        followups: tuple[str, ...],
        targets: list[int],
        cap: int,
        events: list[dict[str, Any]],
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        user_turns = (initial, *followups)
        n_plan = min(len(targets), len(user_turns))
        history: list[dict[str, str]] = [
            {"role": "system", "content": SESSION_SYSTEM_PROMPT}
        ]
        turns_ok = 0
        max_input = 0
        output_tokens = 0
        error: str | None = None
        turns_detail: list[dict[str, Any]] = []
        for turn_i in range(n_plan):
            blob = workspace_blob(spec_id, turn_i, extra_tokens=0)
            user = f"{user_turns[turn_i]}\n\n{blob}"
            history.append({"role": "user", "content": user})
            history = pad_last_user_to_total(
                history, targets[turn_i], seed=f"{spec_id}:{turn_i}"
            )
            used = content_tokens(history)
            if used + self.config.output_tokens > cap:
                history.pop()
                break
            max_input = max(max_input, used)
            result = post_chat_completion(
                client,
                url=url,
                model=recipe.endpoint.model_name,
                messages=history,
                max_tokens=self.config.output_tokens,
                temperature=self.config.temperature,
                stream=self.config.stream,
                headers=headers,
                prompt_id=f"{spec_id}:t{turn_i}",
            )
            kv: float | None = None
            if self.kv_metrics:
                kv = fetch_kv_cache_perc(client, kv_url)
            turn_row = compact(
                {
                    "turn": turn_i,
                    "session_id": spec_id,
                    "input_tokens": used,
                    **request_row(result, kv_cache_perc=kv),
                }
            )
            turns_detail.append(turn_row)
            events.append(
                compact(
                    {
                        "phase": "session",
                        "session_id": spec_id,
                        "turn": turn_i,
                        **request_row(result, kv_cache_perc=kv),
                        "input_tokens": used,
                    }
                )
            )
            if kv is not None:
                samples.append(
                    compact(
                        {
                            "phase": "session",
                            "session_id": spec_id,
                            "turn": turn_i,
                            "t_unix_ms": result.finished_unix_ms or int(time.time() * 1000),
                            "kv_cache_perc": kv,
                        }
                    )
                )
            if not result.ok:
                error = result.error
                history.pop()
                break
            turns_ok += 1
            output_tokens += result.completion_tokens
            history.append(
                {"role": "assistant", "content": result.text or "(empty)"}
            )
            if content_tokens(history) >= targets[-1]:
                break
        return {
            "session_id": spec_id,
            "title": title,
            "ok": error is None and turns_ok > 0,
            "turns": turns_ok,
            "planned_turns": n_plan,
            "max_input_tokens": max_input,
            "output_tokens": output_tokens,
            "error": error,
            "turns_detail": turns_detail,
        }
