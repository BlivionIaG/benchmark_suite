"""Fixtures for scoring tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import (
    save_file as np_save_safetensors,  # type: ignore[reportUnknownVariableType]
)

from benchmark_suite.recipe import (
    BenchSection,
    LoadSection,
    MetaSection,
    Recipe,
    ThroughputScorer,
)
from benchmark_suite.runner.llm_perf import LLMPerfResult


@pytest.fixture
def simple_recipe() -> Recipe:
    """Minimal valid recipe with one ThroughputScorer (no other scoring fields)."""
    return Recipe(
        meta=MetaSection(name="simple", description="test recipe"),
        bench=BenchSection(
            load=LoadSection(concurrencies=[1, 4, 8]),
            scoring=[ThroughputScorer(tool="llm-perf")],
        ),
    )


@pytest.fixture
def mock_llm_perf_result() -> LLMPerfResult:
    """Fake LLMPerfResult with known values for deterministic assertions."""
    return LLMPerfResult(
        raw={},
        output_tok_s=76.5,
        total_tok_s=200.0,
        peak_output_tok_s=80.0,
        ttft_mean_ms=120.0,
        ttft_median_ms=110.0,
        ttft_p99_ms=200.0,
        tpot_mean_ms=15.0,
        tpot_median_ms=14.0,
        duration_s=30.0,
        successful=32,
        failed=0,
    )


@pytest.fixture
def tmp_result_dir(tmp_path: Path) -> Path:
    """Provides a Path for result_dir; creates artifacts/ subdir."""
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- KLD (T6) fixtures ---

KLD_VOCAB = 8
KLD_PROMPT_LEN = 4


class _FakeTokenizer:
    """Minimal tokenizer stand-in: vocab_size + token→id mapping."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def convert_tokens_to_ids(self, token: str) -> int | None:
        try:
            return int(token)
        except ValueError:
            return None


@pytest.fixture
def mock_tokenizer() -> _FakeTokenizer:
    """Fake tokenizer exposing ``vocab_size`` + ``convert_tokens_to_ids``."""
    return _FakeTokenizer(32000)


@pytest.fixture
def synthetic_logits_dir(tmp_path: Path) -> Path:
    """A converted logit cache: manifest.json + 3 safetensors shards of known logits."""
    logits_dir = tmp_path / "kl_cache"
    logits_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "model_name": "ref-model",
        "tokenizer_id": "ref-tokenizer",
        "vocab_size": KLD_VOCAB,
        "prompt_count": 3,
        "max_prompt_tokens": KLD_PROMPT_LEN,
        "shape_per_prompt": [[KLD_PROMPT_LEN, KLD_VOCAB]] * 3,
        "dtype": "float32",
        "created_at": "2026-08-30T00:00:00+00:00",
        "source_files": ["f_0.fp16", "f_1.fp16", "f_2.fp16"],
        "cache_format": "safetensors+manifest",
        "schema_version": "1.0",
    }
    (logits_dir / "manifest.json").write_text(json.dumps(manifest))
    rng = np.random.default_rng(0)
    for idx in range(3):
        arr = rng.standard_normal((KLD_PROMPT_LEN, KLD_VOCAB)).astype(np.float32)
        np_save_safetensors({"logits": arr}, logits_dir / f"prompt_{idx:06d}.safetensors")
    return logits_dir
