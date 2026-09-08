"""sauce scorer — house-blend mixed workload original to this suite.

Not a wrapper around llm-perf, inspect-ai, or promptfoo. Two phases share
``/v1/chat/completions``:

1. **chat** — 31 unique (system, task) pairs on a 16/8/4/2/1 concurrency
   ladder, at 1k/512 and 16k/1k.
2. **session** — Pi-style coding-agent sessions that grow toward 200k tokens.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark_suite.recipe import Recipe, SauceScorer
from benchmark_suite.scoring.base import Scorer, ScoreRecord, ScoreStatus, scorer
from benchmark_suite.scoring.chat_load import ChatLoadScorerImpl
from benchmark_suite.scoring.session import SessionScorerImpl


def _as_number(value: float | int | str | None, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


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
            for key in (
                "session_turns",
                "session_success_rate",
                "session_max_input_tokens",
            ):
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
        if self.config.chat.enabled:
            chat_rec = ChatLoadScorerImpl(self.config.chat).score(
                recipe, result_dir=result_dir, endpoint_url=endpoint_url
            )
        if self.config.session.enabled:
            session_rec = SessionScorerImpl(self.config.session).score(
                recipe, result_dir=result_dir, endpoint_url=endpoint_url
            )
        return merge_sauce_records(
            cell_id=cell_id, started=started, chat=chat_rec, session=session_rec
        )
