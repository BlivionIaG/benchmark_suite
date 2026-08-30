"""Tests for benchmark_suite.scoring.kl_divergence — top-k KL + vocab check + scorer.

Synthetic logits fixtures only (small vocab, a few prompts). No torch, no GPU,
no network — the safetensors cache is generated directly with
``safetensors.numpy.save_file`` and the endpoint is respx-mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from safetensors.numpy import (
    save_file as np_save_safetensors,  # type: ignore[reportUnknownVariableType]
)

from benchmark_suite.recipe import KLDScorer, Recipe
from benchmark_suite.scoring.base import ScorerRegistry, ScoreStatus
from benchmark_suite.scoring.kl_divergence import (
    KLDScorerImpl,
    _load_manifest,  # type: ignore[reportPrivateUsage]
    _top_k_kl,  # type: ignore[reportPrivateUsage]
    _vocab_check,  # type: ignore[reportPrivateUsage]
)
from benchmark_suite.tools.convert_kl_logits import KLCacheManifest

VOCAB = 8
PROMPT_LEN = 4


def _manifest_dict(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model_name": "ref-model",
        "tokenizer_id": "ref-tokenizer",
        "vocab_size": VOCAB,
        "prompt_count": 3,
        "max_prompt_tokens": PROMPT_LEN,
        "shape_per_prompt": [[PROMPT_LEN, VOCAB]] * 3,
        "dtype": "float32",
        "created_at": "2026-08-30T00:00:00+00:00",
        "source_files": ["f_0.fp16", "f_1.fp16", "f_2.fp16"],
        "cache_format": "safetensors+manifest",
        "schema_version": "1.0",
    }
    base.update(overrides)
    return base


def _write_cache(logits_dir: Path, logits: list[np.ndarray]) -> None:
    """Write manifest.json + one safetensors shard per prompt."""
    logits_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_dict(
        prompt_count=len(logits),
        shape_per_prompt=[[arr.shape[0], arr.shape[1]] for arr in logits],
        max_prompt_tokens=max(arr.shape[0] for arr in logits),
    )
    (logits_dir / "manifest.json").write_text(json.dumps(manifest))
    for idx, arr in enumerate(logits):
        np_save_safetensors({"logits": arr}, logits_dir / f"prompt_{idx:06d}.safetensors")


# --- manifest roundtrip ---


def test_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = KLCacheManifest(
        model_name="m",
        tokenizer_id="t",
        vocab_size=32000,
        prompt_count=2,
        max_prompt_tokens=16,
        shape_per_prompt=[(16, 32000), (8, 32000)],
        dtype="float16",
        created_at="2026-08-30T00:00:00+00:00",
        source_files=["f_0.fp16", "f_1.fp16"],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest.to_dict()))
    loaded = json.loads(path.read_text())
    assert loaded == manifest.to_dict()
    assert loaded["vocab_size"] == 32000
    assert loaded["shape_per_prompt"] == [[16, 32000], [8, 32000]]


# --- _load_manifest ---


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError) as exc_info:
        _load_manifest(empty)
    msg = str(exc_info.value)
    assert "manifest.json" in msg
    assert "convert-logits" in msg


def test_load_manifest_malformed_json_raises(tmp_path: Path) -> None:
    d = tmp_path / "cache"
    d.mkdir()
    (d / "manifest.json").write_text("{ not valid json")
    with pytest.raises(json.JSONDecodeError):
        _load_manifest(d)


# --- _top_k_kl ---


def test_top_k_kl_identical_distributions_is_zero() -> None:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((PROMPT_LEN, VOCAB)).astype(np.float32)
    result = _top_k_kl(logits, logits.copy(), top_k=4)
    assert abs(result["mean_kl"]) < 1e-5
    assert abs(result["max_kl"]) < 1e-5
    assert abs(result["p99_kl"]) < 1e-5


def test_top_k_kl_different_distributions_is_positive() -> None:
    ref = np.zeros((1, VOCAB), dtype=np.float32)  # uniform
    cand = np.zeros((1, VOCAB), dtype=np.float32)
    cand[0, 0] = 10.0  # peaked at index 0
    result = _top_k_kl(ref, cand, top_k=VOCAB)
    assert result["mean_kl"] > 0.0


def test_top_k_kl_symmetry_violation() -> None:
    ref = np.zeros((1, VOCAB), dtype=np.float32)  # uniform
    cand = np.zeros((1, VOCAB), dtype=np.float32)
    cand[0, 0] = 10.0  # peaked
    kl_ref_cand = _top_k_kl(ref, cand, top_k=VOCAB)["mean_kl"]
    kl_cand_ref = _top_k_kl(cand, ref, top_k=VOCAB)["mean_kl"]
    assert kl_ref_cand != pytest.approx(kl_cand_ref)


def test_top_k_kl_top_k_restriction() -> None:
    # Large vocab; only top-2 indices of the reference contribute.
    big_vocab = 1000
    ref = np.zeros((1, big_vocab), dtype=np.float32)
    ref[0, 0] = 5.0
    ref[0, 1] = 4.0
    cand = np.zeros((1, big_vocab), dtype=np.float32)
    cand[0, 0] = 5.0
    cand[0, 1] = 4.0
    # Perturb a non-top-k index heavily; with top_k=2 it must not matter.
    cand[0, 999] = 100.0
    result = _top_k_kl(ref, cand, top_k=2)
    assert abs(result["mean_kl"]) < 1e-5


# --- _vocab_check ---


def test_vocab_check_matches() -> None:
    manifest = _manifest_dict(vocab_size=32000)
    assert _vocab_check(manifest, 32000) is True


def test_vocab_check_mismatch_raises() -> None:
    manifest = _manifest_dict(vocab_size=32000)
    with pytest.raises(ValueError) as exc_info:
        _vocab_check(manifest, 128000)
    msg = str(exc_info.value)
    assert "32000" in msg
    assert "128000" in msg


# --- scorer registration ---


def test_kld_scorer_kind_registered() -> None:
    assert ScorerRegistry.get("kld") is KLDScorerImpl


# --- scorer behavior ---


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "kld-recipe"},
            "backend": {"type": "external"},
            "endpoint": {"url": "http://127.0.0.1:8000", "model_name": "candidate"},
        }
    )


def test_kld_scorer_handles_missing_manifest_gracefully(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    config = KLDScorer.model_validate(
        {"kind": "kld", "source": "logits_dir", "reference_logits_dir": empty}
    )
    impl = KLDScorerImpl(config)
    record = impl.score(_recipe(), result_dir=tmp_path)
    assert record.status == ScoreStatus.FAILURE
    assert record.error is not None
    assert "convert-logits" in record.error


class _FakeTokenizer:
    """Minimal tokenizer stand-in: vocab_size + token→id mapping."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def convert_tokens_to_ids(self, token: str) -> int | None:
        try:
            return int(token)
        except ValueError:
            return None


def test_kld_scorer_logits_dir_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    # Reference cache: uniform logits (so a candidate matching them → KL ≈ 0).
    ref_logits = [np.zeros((PROMPT_LEN, VOCAB), dtype=np.float32) for _ in range(3)]
    logits_dir = tmp_path / "cache"
    _write_cache(logits_dir, ref_logits)

    prompts_file = tmp_path / "prompts.jsonl"
    prompts_file.write_text(
        "".join(json.dumps({"prompt": f"p{i}"}) + "\n" for i in range(3))
    )

    def _fake_load_tokenizer(tokenizer_id: str) -> _FakeTokenizer:
        return _FakeTokenizer(VOCAB)

    monkeypatch.setattr(
        "benchmark_suite.scoring.kl_divergence._load_tokenizer",
        _fake_load_tokenizer,
    )

    # Candidate endpoint returns uniform top_logprobs over the vocab (matching
    # the uniform reference), so KL ≈ 0.
    top_logprobs = [
        {"token": str(i), "logprob": -np.log(VOCAB)} for i in range(VOCAB)
    ]
    respx_mock.post("http://127.0.0.1:8000/v1/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"text": "x", "logprobs": {"top_logprobs": top_logprobs}}
                ]
            },
        )
    )

    config = KLDScorer.model_validate(
        {
            "kind": "kld",
            "source": "logits_dir",
            "reference_logits_dir": logits_dir,
            "prompts_file": prompts_file,
            "top_k": VOCAB,
        }
    )
    impl = KLDScorerImpl(config)
    record = impl.score(_recipe(), result_dir=tmp_path)
    assert record.status == ScoreStatus.SUCCESS
    assert record.metrics["mean_kl"] == pytest.approx(0.0, abs=1e-5)


def test_kld_scorer_llm_perf_source_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = KLDScorer.model_validate(
        {"kind": "kld", "source": "llm-perf", "reference_endpoint": "http://ref:8000"}
    )

    def fake_run_logprobs(*args: object, **kwargs: object) -> Path:
        return tmp_path / "logprobs.jsonl"

    def fake_run_kl(*args: object, **kwargs: object) -> dict[str, Any]:
        return {"kl_divergence": 0.42}

    monkeypatch.setattr(
        "benchmark_suite.scoring.kl_divergence.run_llm_perf_logprobs", fake_run_logprobs
    )
    monkeypatch.setattr(
        "benchmark_suite.scoring.kl_divergence.run_llm_perf_kl_divergence", fake_run_kl
    )
    impl = KLDScorerImpl(config)
    record = impl.score(_recipe(), result_dir=tmp_path)
    assert record.status == ScoreStatus.SUCCESS
    assert record.metrics["kl_divergence"] == pytest.approx(0.42)