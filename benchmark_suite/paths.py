"""benchmark_suite/paths.py — result-dir layout helpers (pure path construction)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from benchmark_suite.recipe import CellId


def ts_slug(dt: datetime) -> str:
    """Filesystem-safe timestamp slug: <YYYY-MM-DDTHH-MM-SS> (no colons)."""
    return dt.strftime("%Y-%m-%dT%H-%M-%S")


def result_dir(recipe_name: str, ts: datetime, cell: CellId) -> Path:
    """results/<recipe_name>_<ts_slug>/<cell.render()>/"""
    return Path("results") / f"{recipe_name}_{ts_slug(ts)}" / cell.render()


def summary_csv_path(result_dir: Path) -> Path:
    return result_dir / "summary.csv"


def summary_json_path(result_dir: Path) -> Path:
    return result_dir / "summary.json"


def readme_path(result_dir: Path) -> Path:
    return result_dir / "README.md"


def logs_dir(result_dir: Path) -> Path:
    return result_dir / "logs"


def artifacts_dir(result_dir: Path) -> Path:
    return result_dir / "artifacts"