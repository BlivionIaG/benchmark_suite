"""Tests for benchmark_suite.scoring.llm_judge — native + promptfoo drivers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml

from benchmark_suite.recipe import LLMJudgeScorer, Recipe
from benchmark_suite.scoring.base import ScoreStatus
from benchmark_suite.scoring.llm_judge import (
    LLMJudgeScorerImpl,
    build_promptfoo_config,
    native_judge,
    parse_judge_response,
    run_promptfoo,
)

JUDGE_URL = "http://127.0.0.1:9000"
JUDGE_MODEL = "judge-model"


def _judge_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
    )


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "judge-recipe"},
            "backend": {"type": "external"},
            "endpoint": {"url": "http://127.0.0.1:8000", "model_name": "candidate"},
        }
    )


def _judge_config(**overrides: object) -> LLMJudgeScorer:
    base: dict[str, object] = {
        "kind": "llm_judge",
        "judge_url": JUDGE_URL,
        "judge_model": JUDGE_MODEL,
    }
    base.update(overrides)
    return LLMJudgeScorer.model_validate(base)


def _write_prompts_file(tmp_path: Path) -> Path:
    """Write a JSONL prompts file and return its path."""
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"prompt": "What is 2+2?", "candidate_answer": "4"}) + "\n"
    )
    return path


# --- native_judge ---


def test_native_judge_success(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{JUDGE_URL}/chat/completions").mock(
        return_value=_judge_response('{"score": 8, "rationale": "good"}')
    )
    result = native_judge(
        JUDGE_URL,
        JUDGE_MODEL,
        "",
        prompt="What is 2+2?",
        candidate_answer="4",
        rubric="Score correctness.",
    )
    assert result["score"] == 8
    assert result["rationale"] == "good"


def test_native_judge_with_markdown_fence(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{JUDGE_URL}/chat/completions").mock(
        return_value=_judge_response('```json\n{"score": 6, "rationale": "ok"}\n```')
    )
    result = native_judge(
        JUDGE_URL,
        JUDGE_MODEL,
        "",
        prompt="p",
        candidate_answer="a",
        rubric="r",
    )
    assert result["score"] == 6
    assert result["rationale"] == "ok"


def test_native_judge_with_surrounding_text(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(f"{JUDGE_URL}/chat/completions").mock(
        return_value=_judge_response('Sure! The score is: ```json {"score": 7} ```')
    )
    result = native_judge(
        JUDGE_URL,
        JUDGE_MODEL,
        "",
        prompt="p",
        candidate_answer="a",
        rubric="r",
    )
    assert result["score"] == 7


# --- parse_judge_response ---


def test_parse_judge_response_direct_json() -> None:
    parsed = parse_judge_response('{"score": 8, "rationale": "good"}')
    assert parsed == {"score": 8, "rationale": "good"}


def test_parse_judge_response_regex_fallback() -> None:
    parsed = parse_judge_response("The answer is decent. score: 9, rationale: solid work")
    assert parsed["score"] == 9


def test_parse_judge_response_no_score_raises() -> None:
    with pytest.raises(ValueError):
        parse_judge_response("no numeric score anywhere in this text")


# --- build_promptfoo_config ---


def test_build_promptfoo_config_writes_yaml(tmp_path: Path) -> None:
    recipe = _recipe()
    config = _judge_config()
    prompts = [
        {"prompt": "What is 2+2?", "candidate_answer": "4"},
        {"prompt": "Capital of France?", "candidate_answer": "Paris"},
    ]
    out = build_promptfoo_config(recipe, config, prompts, tmp_path)
    assert out == tmp_path / "promptfooconfig.yaml"
    data = yaml.safe_load(out.read_text())
    assert "providers" in data
    assert "prompts" in data
    assert "tests" in data
    assert data["providers"][0]["id"] == f"openai:chat:{JUDGE_MODEL}"


# --- run_promptfoo ---


def test_run_promptfoo_invokes_npx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    npx = fake_bin / "npx"
    canned: dict[str, Any] = {"results": {"version": 3, "results": []}}
    npx.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{json.dumps(canned)}'\n"
    )
    npx.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    config_path = tmp_path / "promptfooconfig.yaml"
    config_path.write_text("providers: []\n")
    result = run_promptfoo(config_path)
    assert result == canned


def test_run_promptfoo_missing_npx_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    config_path = tmp_path / "promptfooconfig.yaml"
    config_path.write_text("providers: []\n")
    with pytest.raises(FileNotFoundError):
        run_promptfoo(config_path)


# --- scorer end-to-end ---


def test_llm_judge_scorer_native_driver_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    prompts_file = _write_prompts_file(tmp_path)
    config = _judge_config(prompts_file=prompts_file)

    def fake_native_judge(*args: object, **kwargs: object) -> dict[str, Any]:
        return {"score": 8, "rationale": "good", "raw": "{}"}

    monkeypatch.setattr(
        "benchmark_suite.scoring.llm_judge.native_judge", fake_native_judge
    )
    impl = LLMJudgeScorerImpl(config)
    record = impl.score(recipe, result_dir=tmp_path)
    assert record.metrics["judge_score"] == 8
    assert record.status == ScoreStatus.SUCCESS


def test_llm_judge_scorer_promptfoo_driver_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    prompts_file = _write_prompts_file(tmp_path)
    config = _judge_config(driver="promptfoo", prompts_file=prompts_file)

    def fake_run_promptfoo(*args: object, **kwargs: object) -> dict[str, Any]:
        return {"results": {"results": [{"score": 7.0}]}}

    monkeypatch.setattr(
        "benchmark_suite.scoring.llm_judge.run_promptfoo", fake_run_promptfoo
    )
    impl = LLMJudgeScorerImpl(config)
    record = impl.score(recipe, result_dir=tmp_path)
    assert record.metrics["judge_score"] == 7.0
    assert record.status == ScoreStatus.SUCCESS


def test_llm_judge_scorer_handles_judge_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = _recipe()
    prompts_file = _write_prompts_file(tmp_path)
    config = _judge_config(prompts_file=prompts_file)

    def fake_native_judge(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("judge down")

    monkeypatch.setattr(
        "benchmark_suite.scoring.llm_judge.native_judge", fake_native_judge
    )
    impl = LLMJudgeScorerImpl(config)
    record = impl.score(recipe, result_dir=tmp_path)
    assert record.status == ScoreStatus.FAILURE
    assert record.error is not None