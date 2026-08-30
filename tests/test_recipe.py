"""Tests for benchmark_suite.recipe — schema validation, union discrimination, env merge."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from benchmark_suite.recipe import (
    CANONICAL_ENV,
    KLDScorer,
    Recipe,
    ThroughputScorer,
    load_recipe,
)


def test_load_minimal_recipe(tmp_recipe_path: Path) -> None:
    r = load_recipe(tmp_recipe_path)
    assert r.meta.name == "test-recipe"
    assert r.meta.description == "minimal valid recipe"
    assert r.backend.type == "external"
    assert r.cell.render() == "dense_triton_triton_cg0_mtp0"
    assert r.merged_env() == CANONICAL_ENV


def test_load_full_recipe(tmp_path: Path) -> None:
    full = {
        "meta": {
            "name": "full-recipe",
            "description": "full",
            "version": "2.0.0",
            "author": "me",
            "tags": ["a", "b"],
        },
        "backend": {
            "type": "vllm",
            "model_path": "/models/Test-Model",
            "served_model_name": "served",
            "vllm": {"language-model-only": True},
        },
        "endpoint": {"url": "http://127.0.0.1:8000", "model_name": "custom"},
        "resources": {"tensor_parallel_size": 4, "dtype": "float16", "devices": "0,1,2,3"},
        "runtime": {"env": {"VLLM_X": "1"}, "startup_wait_s": 1200},
        "bench": {
            "load": {"concurrencies": [1, 4, 8], "num_prompts": 32},
            "scoring": [
                {"kind": "throughput", "tool": "llm-perf"},
                {"kind": "kld", "source": "logits_dir"},
                {"kind": "perplexity", "tasks": ["wikitext"]},
                {"kind": "llm_judge", "driver": "native"},
                {"kind": "agentic", "harness": "inspect"},
            ],
            "stop_conditions": {"max_duration_s": 3600.0},
        },
        "report": {"output_format": "all", "output_dir": "results"},
        "cell": {"family": "dense", "attn": "fardna2", "linear": "rdna2", "cg": 1, "mtp": 2},
    }
    p = tmp_path / "full.yaml"
    p.write_text(yaml.safe_dump(full))
    r = load_recipe(p)
    assert r.meta.version == "2.0.0"
    assert r.meta.tags == ["a", "b"]
    assert r.resources.tensor_parallel_size == 4
    assert r.resources.dtype == "float16"
    assert len(r.bench.scoring) == 5
    assert r.bench.stop_conditions.max_duration_s == 3600.0
    assert r.cell.render() == "dense_fardna2_rdna2_cg1_mtp2"


def test_invalid_meta_name_slug() -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate({"meta": {"name": "Bad Name!", "description": "x"}})


def test_scorer_discrimination() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "bench": {
                "scoring": [
                    {"kind": "throughput", "tool": "llm-perf"},
                    {"kind": "kld", "source": "logits_dir"},
                ]
            },
        }
    )
    assert isinstance(r.bench.scoring[0], ThroughputScorer)
    assert isinstance(r.bench.scoring[1], KLDScorer)


def test_unknown_scorer_kind() -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate(
            {
                "meta": {"name": "x", "description": "y"},
                "bench": {"scoring": [{"kind": "bogus"}]},
            }
        )


def test_merged_env_precedence() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "runtime": {"env": {"VLLM_ROCM_USE_AITER": "1"}},
        }
    )
    assert r.merged_env()["VLLM_ROCM_USE_AITER"] == "1"
    assert CANONICAL_ENV["VLLM_ROCM_USE_AITER"] == "0"


def test_merged_env_canonical_default() -> None:
    r = Recipe.model_validate({"meta": {"name": "x", "description": "y"}})
    assert r.merged_env() == CANONICAL_ENV


def test_model_path_required_when_not_external() -> None:
    with pytest.raises(ValidationError):
        Recipe.model_validate(
            {
                "meta": {"name": "x", "description": "y"},
                "backend": {"type": "vllm", "model_path": ""},
            }
        )


def test_endpoint_model_name_default() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/Test-Model"},
        }
    )
    assert r.endpoint.model_name == "Test-Model"


def test_endpoint_model_name_overrides() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/Test-Model"},
            "endpoint": {"model_name": "custom-name"},
        }
    )
    assert r.endpoint.model_name == "custom-name"


def test_hardware_section_defaults() -> None:
    from benchmark_suite.recipe import HardwareSection

    h = HardwareSection()
    assert h.vendor == "amd"
    assert h.count == 1
    assert h.vram_gb == 0
    assert h.model == ""
    assert not h.is_complete()


def test_hardware_section_is_complete() -> None:
    from benchmark_suite.recipe import HardwareSection

    h = HardwareSection(model="Radeon PRO V620", vram_gb=32, count=4)
    assert h.is_complete()

    h_no_model = HardwareSection(vram_gb=32, count=4)
    assert not h_no_model.is_complete()

    h_no_vram = HardwareSection(model="X", count=4)
    assert not h_no_vram.is_complete()


def test_hardware_vendor_enum() -> None:
    from typing import Any, cast

    from pydantic import ValidationError

    from benchmark_suite.recipe import HardwareSection

    h = HardwareSection(vendor="nvidia")
    assert h.vendor == "nvidia"
    with pytest.raises(ValidationError):
        HardwareSection(vendor=cast(Any, "bogus-vendor"))


def test_recipe_quantization_defaults_to_fp16_for_float16_dtype() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/M"},
            "resources": {"dtype": "float16"},
        }
    )
    assert r.quantization == "FP16"


def test_recipe_quantization_explicit_overrides_default() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/M"},
            "resources": {"dtype": "float16"},
            "quantization": "W4A16-G32",
        }
    )
    assert r.quantization == "W4A16-G32"


def test_recipe_quantization_stays_empty_for_non_float16_dtype() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/M"},
            "resources": {"dtype": "bfloat16"},
        }
    )
    assert r.quantization == ""


def test_recipe_hardware_default_is_empty_section() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/M"},
        }
    )
    assert r.hardware.vendor == "amd"
    assert r.hardware.count == 1
    assert r.hardware.vram_gb == 0
    assert not r.hardware.is_complete()


def test_recipe_hardware_loads_from_dict() -> None:
    r = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/M"},
            "hardware": {
                "vendor": "amd",
                "model": "Radeon PRO V620",
                "count": 4,
                "vram_gb": 32,
                "cpu": "EPYC 7452",
                "ram_gb": 256,
                "os": "Ubuntu 22.04",
                "power_watts": 200,
            },
        }
    )
    assert r.hardware.is_complete()
    assert r.hardware.model == "Radeon PRO V620"
    assert r.hardware.count == 4
    assert r.hardware.vram_gb == 32
    assert r.hardware.power_watts == 200