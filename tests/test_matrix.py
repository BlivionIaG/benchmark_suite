"""Tests for benchmark_suite._matrix — cartesian axis expansion."""
from __future__ import annotations

from benchmark_suite._matrix import MatrixAxis, expand_matrix
from benchmark_suite.recipe import Recipe


def _recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/Test-Model"},
        }
    )


def test_expand_no_axes() -> None:
    r = _recipe()
    result = expand_matrix(r, [])
    assert len(result) == 1
    assert result[0] == r


def test_expand_single_axis_two_values() -> None:
    r = _recipe()
    axes = [MatrixAxis(("cell", "attn"), ["triton", "fardna2"])]
    result = expand_matrix(r, axes)
    assert len(result) == 2
    assert result[0].cell.attn == "triton"
    assert result[1].cell.attn == "fardna2"


def test_expand_two_axes_cartesian() -> None:
    r = _recipe()
    axes = [
        MatrixAxis(("cell", "attn"), ["triton", "fardna2"]),
        MatrixAxis(("cell", "linear"), ["rdna2", "exllama"]),
    ]
    result = expand_matrix(r, axes)
    assert len(result) == 4
    combos = {(v.cell.attn, v.cell.linear) for v in result}
    assert combos == {
        ("triton", "rdna2"),
        ("triton", "exllama"),
        ("fardna2", "rdna2"),
        ("fardna2", "exllama"),
    }


def test_expand_does_not_mutate_input() -> None:
    r = _recipe()
    axes = [MatrixAxis(("cell", "attn"), ["triton", "fardna2"])]
    expand_matrix(r, axes)
    assert r.cell.attn == "triton"  # default unchanged


def test_expand_runtime_env_axis() -> None:
    r = _recipe()
    axes = [MatrixAxis(("runtime", "env", "VLLM_USE_RDNA2_FA"), ["0", "1"])]
    result = expand_matrix(r, axes)
    assert len(result) == 2
    assert result[0].runtime.env["VLLM_USE_RDNA2_FA"] == "0"
    assert result[1].runtime.env["VLLM_USE_RDNA2_FA"] == "1"
    assert "VLLM_USE_RDNA2_FA" not in r.runtime.env