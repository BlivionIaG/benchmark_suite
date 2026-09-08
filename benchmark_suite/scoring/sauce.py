"""sauce scorer — house-blend mixed workload original to this suite.

Not a wrapper around llm-perf, inspect-ai, or promptfoo. Two phases share
``/v1/chat/completions``:

1. **chat** — 31 unique (system, task) pairs on a 16/8/4/2/1 concurrency
   ladder, at 1k/512 and 16k/1k.
2. **session** — Pi-style coding-agent sessions that grow toward 200k tokens.

Both phases record TTFT, TPOT, prefill tok/s, decode tok/s, cached tokens,
and KV-cache % (when the server exposes Prometheus ``/metrics``), per request
and over time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from benchmark_suite.recipe import Recipe, SauceScorer
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer
from benchmark_suite.scoring.chat_load import ChatLoadScorerImpl
from benchmark_suite.scoring.session import SessionScorerImpl

_SESSION_COPY_KEYS = (
    "session_turns",
    "session_success_rate",
    "session_max_input_tokens",
    "session_ttft_mean_ms",
    "session_tpot_mean_ms",
    "session_prefill_tok_s",
    "session_decode_tok_s",
    "session_cached_tokens",
    "session_kv_cache_perc",
)


def _as_number(value: float | int | str | None, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _numeric(metrics: dict[str, float | int | str], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _event_time(row: dict[str, Any]) -> int:
    value = row.get("t_unix_ms")
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _merge_kv_perc(
    metrics: dict[str, float | int | str],
    chat: ScoreRecord | None,
    session: ScoreRecord | None,
) -> None:
    vals: list[float] = []
    if chat is not None:
        kv = _numeric(chat.metrics, "kv_cache_perc")
        if kv is not None:
            vals.append(kv)
    if session is not None:
        for key in ("kv_cache_perc", "session_kv_cache_perc"):
            kv = _numeric(session.metrics, key)
            if kv is not None:
                vals.append(kv)
    if vals:
        metrics["kv_cache_perc"] = max(vals)


def write_sauce_timeseries(result_dir: Path, artifacts: dict[str, str]) -> None:
    """Merge chat + session timeseries into one artifact, sorted by t_unix_ms."""
    events: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    artifacts_dir = result_dir / "artifacts"
    for fname in ("chat_timeseries.json", "session_timeseries.json"):
        path = artifacts_dir / fname
        if not path.is_file():
            continue
        loaded: object = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            continue
        data = cast(dict[str, Any], loaded)
        raw_events: object = data.get("events")
        if isinstance(raw_events, list):
            for item in cast(list[object], raw_events):
                if isinstance(item, dict):
                    events.append(cast(dict[str, Any], item))
        raw_samples: object = data.get("samples")
        if isinstance(raw_samples, list):
            for item in cast(list[object], raw_samples):
                if isinstance(item, dict):
                    samples.append(cast(dict[str, Any], item))
    if not events and not samples:
        return
    events.sort(key=_event_time)
    samples.sort(key=_event_time)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    name = "sauce_timeseries.json"
    (artifacts_dir / name).write_text(
        json.dumps({"kind": "sauce", "events": events, "samples": samples}, indent=2)
        + "\n"
    )
    artifacts[name] = f"artifacts/{name}"


def merge_sauce_records(
    *,
    cell_id: str,
    started: datetime,
    chat: ScoreRecord | None,
    session: ScoreRecord | None,
) -> ScoreRecord:
    """Combine chat + session helper records into one ``kind=sauce`` row."""
    metrics: dict[str, float | int | str] = {}
    artifacts: dict[str, str] = {}
    notes: dict[str, Any] = {}
    errors: list[str] = []
    phase_ok = 0

    if chat is not None:
        metrics.update(chat.metrics)
        artifacts.update(chat.artifacts)
        notes["chat"] = chat.notes
        if chat.status == ScoreStatus.SUCCESS:
            phase_ok += 1
        elif chat.error:
            errors.append(f"chat: {chat.error}")

    if session is not None:
        artifacts.update(session.artifacts)
        notes["session"] = session.notes
        if chat is None:
            metrics.update(session.metrics)
        else:
            for key in _SESSION_COPY_KEYS:
                if key in session.metrics:
                    metrics[key] = session.metrics[key]
            metrics["duration_s"] = _as_number(metrics.get("duration_s")) + _as_number(
                session.metrics.get("duration_s")
            )
            metrics["successful"] = int(_as_number(metrics.get("successful"))) + int(
                _as_number(session.metrics.get("successful"))
            )
            metrics["failed"] = int(_as_number(metrics.get("failed"))) + int(
                _as_number(session.metrics.get("failed"))
            )
        if session.status == ScoreStatus.SUCCESS:
            phase_ok += 1
        elif session.error:
            errors.append(f"session: {session.error}")

    _merge_kv_perc(metrics, chat, session)

    if phase_ok == 0:
        status = ScoreStatus.FAILURE
        error = "; ".join(errors) if errors else "sauce produced no successful phase"
    else:
        status = ScoreStatus.SUCCESS
        error = "; ".join(errors) if errors else None

    return ScoreRecord(
        kind="sauce",
        cell_id=cell_id,
        status=status,
        started_at=started,
        finished_at=datetime.now(UTC),
        metrics=metrics,
        artifacts=artifacts,
        error=error,
        notes=notes,
    )


@scorer
class SauceScorerImpl(Scorer):
    """Original mixed workload: diverse concurrent chat + growing coding sessions."""

    kind = "sauce"

    def __init__(self, config: SauceScorer) -> None:
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
        chat_rec: ScoreRecord | None = None
        session_rec: ScoreRecord | None = None
        kv_metrics = self.config.kv_metrics
        kv_path = self.config.kv_metrics_path
        if self.config.chat.enabled:
            chat_rec = ChatLoadScorerImpl(
                self.config.chat,
                kv_metrics=kv_metrics,
                kv_metrics_path=kv_path,
            ).score(recipe, result_dir=result_dir, endpoint_url=endpoint_url)
        if self.config.session.enabled:
            session_rec = SessionScorerImpl(
                self.config.session,
                kv_metrics=kv_metrics,
                kv_metrics_path=kv_path,
            ).score(recipe, result_dir=result_dir, endpoint_url=endpoint_url)
        merged = merge_sauce_records(
            cell_id=cell_id, started=started, chat=chat_rec, session=session_rec
        )
        write_sauce_timeseries(result_dir, merged.artifacts)
        return merged
