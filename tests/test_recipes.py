"""Tests for shipped recipes — every recipes/*.yaml must load via load_recipe.

These tests are the contract for the four reference recipes shipped in the
repo: they must be valid YAML, load into a Recipe, stay uniquely named, and
carry the sections the runner relies on (meta, backend, bench.load, scoring).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_suite.recipe import (
    KLDScorer,
    PerplexityScorer,
    Recipe,
    ThroughputScorer,
    load_recipe,
)

RECIPES_DIR = Path(__file__).resolve().parent.parent / "recipes"
RECIPE_FILES: list[Path] = sorted(RECIPES_DIR.glob("*.yaml"))


def _recipe(name: str) -> Recipe:
    """Load recipes/<name>.yaml from the repo."""
    return load_recipe(RECIPES_DIR / f"{name}.yaml")


def test_recipes_dir_exists() -> None:
    assert RECIPES_DIR.is_dir(), f"missing recipes dir: {RECIPES_DIR}"
    assert len(RECIPE_FILES) >= 4, "expected at least 4 shipped recipes"


@pytest.mark.parametrize("path", RECIPE_FILES, ids=lambda p: p.name)
def test_recipes_all_load(path: Path) -> None:
    recipe = load_recipe(path)
    assert isinstance(recipe, Recipe)


def test_recipes_have_unique_meta_name() -> None:
    names = [load_recipe(p).meta.name for p in RECIPE_FILES]
    assert len(names) == len(set(names)), f"duplicate meta.name values: {names}"


@pytest.mark.parametrize("path", RECIPE_FILES, ids=lambda p: p.name)
def test_recipes_have_required_sections(path: Path) -> None:
    recipe = load_recipe(path)
    assert recipe.meta.name, "meta.name must be non-empty"
    assert recipe.meta.description, "meta.description must be non-empty"
    assert recipe.backend.type in ("vllm", "llamacpp", "tgi", "external")
    assert recipe.bench.load.concurrencies, "bench.load.concurrencies must be non-empty"
    assert recipe.bench.scoring, "bench.scoring must be a non-empty list"


def test_qwen36_27b_gptq_recipe_specifics() -> None:
    r = _recipe("qwen36-27b-gptq-tp4")
    assert r.backend.type == "vllm"
    assert r.resources.tensor_parallel_size == 4
    assert r.cell.linear == "rdna2"
    throughput = [s for s in r.bench.scoring if isinstance(s, ThroughputScorer)]
    assert any(s.tool == "llm-perf" for s in throughput)


def test_qwen36_35b_a3b_recipe_specifics() -> None:
    r = _recipe("qwen36-35b-a3b-fp16-tp4")
    assert r.backend.type == "vllm"
    assert "A3B" in r.backend.model_path
    kinds = [s.kind for s in r.bench.scoring]
    assert "throughput" in kinds


def test_perplexity_compare_recipe_specifics() -> None:
    r = _recipe("perplexity-compare")
    # Single endpoint by design: the description documents that the recipe is
    # pointed at any externally-running OpenAI-compatible endpoint by editing
    # endpoint.url — no server is spawned (backend.type=external).
    assert r.backend.type == "external"
    assert r.endpoint.url
    perplexity = [s for s in r.bench.scoring if isinstance(s, PerplexityScorer)]
    assert perplexity, "perplexity-compare must include a perplexity scorer"


def test_kld_recipe_specifics() -> None:
    r = _recipe("kld-vs-fp16-reference")
    kld = [s for s in r.bench.scoring if isinstance(s, KLDScorer)]
    assert kld, "kld-vs-fp16-reference must include a kld scorer"
    assert kld[0].source == "logits_dir"
    assert kld[0].vocab_check is True
