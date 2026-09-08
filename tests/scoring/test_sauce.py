"""Tests for kind=sauce — the house-blend mixed workload."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import respx

from benchmark_suite.recipe import BenchSection, Recipe, SauceScorer
from benchmark_suite.scoring.base import ScoreRecord, ScorerRegistry, ScoreStatus
from benchmark_suite.scoring.sauce import SauceScorerImpl, merge_sauce_records

ENDPOINT = "http://127.0.0.1:8000"
T0 = datetime(2026, 9, 8, 12, 0, 0, tzinfo=UTC)


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 4},
        },
    )


def test_merge_sauce_records_combines_chat_and_session_metrics() -> None:
    chat = ScoreRecord(
        kind="chat_load",
        cell_id="cell",
        status=ScoreStatus.SUCCESS,
        metrics={"output_tok_s": 10.0, "successful": 16, "failed": 0, "duration_s": 2.0},
        artifacts={"chat_load_short_c16.json": "artifacts/chat_load_short_c16.json"},
        notes={"suites": []},
    )
    session = ScoreRecord(
        kind="session",
        cell_id="cell",
        status=ScoreStatus.SUCCESS,
        metrics={
            "session_turns": 12,
            "session_success_rate": 1.0,
            "session_max_input_tokens": 180000,
            "successful": 16,
            "failed": 0,
            "duration_s": 5.0,
            "output_tok_s": 1.0,
        },
        artifacts={"session_sessions.json": "artifacts/session_sessions.json"},
        notes={"budget": 197952},
    )
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=chat, session=session)
    assert merged.kind == "sauce"
    assert merged.status == ScoreStatus.SUCCESS
    assert merged.metrics["output_tok_s"] == 10.0
    assert merged.metrics["session_turns"] == 12
    assert merged.metrics["session_success_rate"] == 1.0
    assert merged.metrics["duration_s"] == 7.0
    assert merged.metrics["successful"] == 32
    assert "chat_load_short_c16.json" in merged.artifacts
    assert "session_sessions.json" in merged.artifacts


def _record(
    *,
    kind: str,
    status: str,
    metrics: dict[str, float | int | str] | None = None,
    error: str | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        kind=kind,
        cell_id="cell",
        status=status,
        metrics=metrics or {},
        error=error,
    )


def test_merge_session_only_keeps_session_throughput() -> None:
    session = ScoreRecord(
        kind="session",
        cell_id="cell",
        status=ScoreStatus.SUCCESS,
        metrics={
            "session_turns": 3,
            "session_success_rate": 1.0,
            "output_tok_s": 4.5,
            "successful": 1,
            "failed": 0,
            "duration_s": 2.0,
        },
    )
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=None, session=session)
    assert merged.kind == "sauce"
    assert merged.metrics["output_tok_s"] == 4.5
    assert merged.metrics["successful"] == 1
    assert merged.metrics["session_turns"] == 3


def test_merge_chat_only_keeps_chat_metrics() -> None:
    chat = _record(
        kind="chat_load",
        status=ScoreStatus.SUCCESS,
        metrics={"output_tok_s": 9.0, "successful": 4, "failed": 0, "duration_s": 1.5},
    )
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=chat, session=None)
    assert merged.status == ScoreStatus.SUCCESS
    assert merged.metrics["output_tok_s"] == 9.0
    assert "session_turns" not in merged.metrics


def test_merge_empty_phases_is_failure() -> None:
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=None, session=None)
    assert merged.status == ScoreStatus.FAILURE
    assert merged.error == "sauce produced no successful phase"


def test_merge_chat_failure_session_success_is_overall_success() -> None:
    chat = _record(
        kind="chat_load",
        status=ScoreStatus.FAILURE,
        error="all chat_load requests failed",
        metrics={"successful": 0, "failed": 1, "duration_s": 0.2},
    )
    session = _record(
        kind="session",
        status=ScoreStatus.SUCCESS,
        metrics={"session_turns": 2, "successful": 1, "failed": 0, "duration_s": 1.0},
    )
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=chat, session=session)
    assert merged.status == ScoreStatus.SUCCESS
    assert merged.error is not None
    assert "chat:" in merged.error
    assert merged.metrics["session_turns"] == 2


def test_merge_chat_success_session_failure_is_overall_success() -> None:
    chat = _record(
        kind="chat_load",
        status=ScoreStatus.SUCCESS,
        metrics={"output_tok_s": 3.0, "successful": 1, "failed": 0, "duration_s": 0.4},
    )
    session = _record(
        kind="session",
        status=ScoreStatus.FAILURE,
        error="all sessions failed",
        metrics={"successful": 0, "failed": 1, "duration_s": 0.1},
    )
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=chat, session=session)
    assert merged.status == ScoreStatus.SUCCESS
    assert merged.error is not None
    assert "session:" in merged.error
    assert merged.metrics["output_tok_s"] == 3.0


def test_merge_both_phases_failed_is_failure() -> None:
    chat = _record(kind="chat_load", status=ScoreStatus.FAILURE, error="chat boom")
    session = _record(kind="session", status=ScoreStatus.FAILURE, error="session boom")
    merged = merge_sauce_records(cell_id="cell", started=T0, chat=chat, session=session)
    assert merged.status == ScoreStatus.FAILURE
    assert merged.error is not None
    assert "chat:" in merged.error
    assert "session:" in merged.error


def test_sauce_is_registered_and_legacy_kinds_are_not() -> None:
    assert ScorerRegistry.get("sauce") is SauceScorerImpl
    assert ScorerRegistry.get("chat_load") is None
    assert ScorerRegistry.get("session") is None


def test_build_scorers_returns_sauce_impl() -> None:
    bench = BenchSection(scoring=[SauceScorer()])
    scorers = ScorerRegistry.build_scorers(bench)
    assert len(scorers) == 1
    assert isinstance(scorers[0], SauceScorerImpl)


def test_sauce_runs_chat_then_session(
    respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    respx_mock.post(f"{ENDPOINT}/v1/chat/completions").mock(return_value=_ok_response())
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "sauce-both"},
            "backend": {"type": "external"},
            "endpoint": {"url": ENDPOINT, "model_name": "candidate"},
            "resources": {"max_model_len": 8192},
            "bench": {
                "scoring": [
                    {
                        "kind": "sauce",
                        "chat": {
                            "stream": False,
                            "ladder": [1],
                            "suites": [
                                {"name": "short", "input_tokens": 64, "output_tokens": 8}
                            ],
                        },
                        "session": {
                            "stream": False,
                            "n_sessions": 1,
                            "max_turns": 2,
                            "output_tokens": 8,
                            "output_reserve_tokens": 256,
                        },
                    }
                ]
            },
        }
    )
    cfg = recipe.bench.scoring[0]
    assert isinstance(cfg, SauceScorer)
    rec = SauceScorerImpl(cfg).score(recipe, result_dir=tmp_path)
    assert rec.kind == "sauce"
    assert rec.status == ScoreStatus.SUCCESS
    assert rec.metrics["successful"] == 2  # 1 chat request + 1 completed session
    assert rec.metrics["session_turns"] == 2
    assert "chat" in rec.notes
    assert "session" in rec.notes
