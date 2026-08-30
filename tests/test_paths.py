"""Tests for benchmark_suite.paths — CellId render + result-dir layout."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from benchmark_suite.paths import (
    artifacts_dir,
    logs_dir,
    readme_path,
    result_dir,
    summary_csv_path,
    summary_json_path,
    ts_slug,
)
from benchmark_suite.recipe import CellId


def test_cell_id_render_dense_fardna2_cg1_mtp2() -> None:
    cell = CellId(family="dense", attn="fardna2", linear="rdna2", cg=1, mtp=2)
    assert cell.render() == "dense_fardna2_rdna2_cg1_mtp2"


def test_cell_id_render_with_extra() -> None:
    cell = CellId(extra={"x": "y"})
    assert cell.render() == "dense_triton_triton_cg0_mtp0_xy"


def test_cell_id_render_defaults() -> None:
    assert CellId().render() == "dense_triton_triton_cg0_mtp0"


def test_result_dir_layout() -> None:
    ts = datetime(2026, 8, 30, 14, 23, 17)
    cell = CellId(family="dense", attn="fardna2", linear="rdna2", cg=1, mtp=2)
    d = result_dir("qwen36", ts, cell)
    assert d == Path("results") / "qwen36_2026-08-30T14-23-17" / "dense_fardna2_rdna2_cg1_mtp2"


def test_ts_slug_filesafe() -> None:
    assert ts_slug(datetime(2026, 8, 30, 14, 23, 17)) == "2026-08-30T14-23-17"


def test_path_helpers() -> None:
    d = Path("results") / "foo_2026-08-30T14-23-17" / "dense_triton_triton_cg0_mtp0"
    assert summary_csv_path(d) == d / "summary.csv"
    assert summary_json_path(d) == d / "summary.json"
    assert readme_path(d) == d / "README.md"
    assert logs_dir(d) == d / "logs"
    assert artifacts_dir(d) == d / "artifacts"