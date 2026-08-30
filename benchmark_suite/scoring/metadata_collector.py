"""benchmark_suite/scoring/metadata_collector.py — hardware/software/model context.

Gathers the metadata that accompanies a leaderboard submission, matching the
`benchmark_suite_leaderboard` repo's `schema/metadata.schema.json`. Every
collector is best-effort: it never raises, and missing keys are simply absent
from the returned dict (the schema treats absent keys as "unknown", which is
valid — just unhelpful).
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, cast

_ROCMSMI = "rocm-smi"
_ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")
_DRIVER_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


class _TorchCuda(Protocol):
    def is_available(self) -> bool: ...
    def get_device_properties(self, idx: int) -> object: ...
    def device_count(self) -> int: ...


class _TorchModule(Protocol):
    cuda: _TorchCuda


class _HfModelInfo(Protocol):
    sha: str | None


class _HfApi(Protocol):
    def model_info(self, repo_id: str) -> _HfModelInfo: ...


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command best-effort; never raises."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(cmd, 1, "", "")


def _rocm_smi_csv(*args: str) -> list[list[str]]:
    """Run `rocm-smi <args> --csv` and parse the CSV rows (header + data)."""
    proc = _run([_ROCMSMI, *args, "--csv"])
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows: list[list[str]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([cell.strip() for cell in line.split(",")])
    return rows


def _torch_gpu_props() -> object | None:
    """Return torch.cuda.get_device_properties(0) or None if torch is absent."""
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    torch_mod = cast("_TorchModule", torch)
    try:
        if not torch_mod.cuda.is_available():
            return None
        return torch_mod.cuda.get_device_properties(0)
    except Exception:
        return None


def collect_hardware() -> dict[str, Any]:
    """Detect hardware: gpu, gpu_count, vram_gb, cpu, ram_gb.

    Prefers rocm-smi (gfx1030/RDNA environments); falls back to torch.cuda;
    returns an empty dict if neither is available. Missing keys are absent.
    """
    result: dict[str, Any] = {}

    # GPU name + count via rocm-smi --showproductname --csv.
    product_rows = _rocm_smi_csv("--showproductname")
    if product_rows:
        header = product_rows[0]
        data = product_rows[1:]
        name_col = next((i for i, h in enumerate(header) if "name" in h.lower()), None)
        if name_col is not None and data:
            names = [row[name_col] for row in data if len(row) > name_col and row[name_col]]
            if names:
                result["gpu"] = names[0]
                result["gpu_count"] = len(names)

    # VRAM via rocm-smi --showmeminfo vram --csv.
    mem_rows = _rocm_smi_csv("--showmeminfo", "vram")
    if mem_rows:
        header = mem_rows[0]
        data = mem_rows[1:]
        vram_col = next(
            (i for i, h in enumerate(header) if "vram" in h.lower() or "total" in h.lower()),
            None,
        )
        if vram_col is not None and data:
            total_mb = 0
            for row in data:
                if len(row) > vram_col and row[vram_col].isdigit():
                    total_mb += int(row[vram_col])
            if total_mb > 0:
                result["vram_gb"] = round(total_mb / 1024)

    # Fallback to torch when rocm-smi gave us nothing.
    if "gpu" not in result:
        props = _torch_gpu_props()
        if props is not None:
            result["gpu"] = getattr(props, "name", "")
            total_memory = getattr(props, "total_memory", 0)
            if total_memory:
                result["vram_gb"] = round(total_memory / 1e9)
            if "gpu_count" not in result:
                try:
                    import torch  # type: ignore[import-not-found]

                    torch_mod = cast("_TorchModule", torch)
                    result["gpu_count"] = torch_mod.cuda.device_count()
                except Exception:
                    pass

    # CPU + RAM from platform (always available, best-effort).
    with suppress(Exception):
        result["cpu"] = platform.processor() or platform.machine()
    with suppress(Exception):
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        result["ram_gb"] = round(ram_bytes / 1e9)

    return result


def _rocm_version() -> str:
    """ROCm version from /opt/rocm/.info/version, else 'n/a'."""
    try:
        if _ROCM_VERSION_FILE.exists():
            text = _ROCM_VERSION_FILE.read_text().strip()
            if text:
                return text.splitlines()[0].strip()
    except OSError:
        pass
    return "n/a"


def _pip_show_version(package: str) -> str:
    """Version field from `pip show <package>`, else ''."""
    proc = _run(["pip", "show", package])
    if proc.returncode != 0:
        return ""
    for line in proc.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return ""


def _driver_version() -> str:
    """Driver version from `rocm-smi --showdriverversion`, else ''."""
    proc = _run([_ROCMSMI, "--showdriverversion"])
    if proc.returncode != 0:
        return ""
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = _DRIVER_VERSION_RE.search(combined)
    return match.group(1) if match else ""


def collect_software() -> dict[str, Any]:
    """Detect software: os, kernel, rocm, vllm, python, driver."""
    result: dict[str, Any] = {}
    try:
        uname = platform.uname()
        result["os"] = uname.system
        result["kernel"] = uname.release
    except Exception:
        pass
    result["rocm"] = _rocm_version()
    vllm = _pip_show_version("vllm")
    if vllm:
        result["vllm"] = vllm
    result["python"] = platform.python_version()
    driver = _driver_version()
    if driver:
        result["driver"] = driver
    return result


def _looks_like_hf_repo(model_path: str) -> bool:
    """Heuristic: 'org/name' (no leading slash, no filesystem path) is an HF repo id."""
    if model_path.startswith(("/", "./", "../", "~")):
        return False
    parts = model_path.split("/")
    return len(parts) == 2 and all(parts) and not Path(model_path).exists()


def collect_model_info(model_path: str) -> dict[str, str]:
    """Resolve model_path to (hf_repo, hf_commit). Never guesses.

    - HF repo id → hf_repo = id, hf_commit = sha via huggingface_hub (or "unknown").
    - Local dir with .git → ("local", <git rev-parse HEAD>).
    - Otherwise → (model_path, "local").
    """
    if not model_path:
        return {"hf_repo": "", "hf_commit": "unknown"}

    if _looks_like_hf_repo(model_path):
        commit = "unknown"
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            api = cast("_HfApi", HfApi())
            info = api.model_info(model_path)
            commit = info.sha or "unknown"
        except Exception:
            pass
        return {"hf_repo": model_path, "hf_commit": commit}

    path = Path(model_path)
    if path.is_dir() and (path / ".git").exists():
        proc = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
        commit = proc.stdout.strip() if proc.returncode == 0 else "local"
        return {"hf_repo": "local", "hf_commit": commit}

    return {"hf_repo": model_path, "hf_commit": "local"}


def build_metadata(
    *,
    submitter: str,
    date_str: str,
    hardware: dict[str, Any] | None = None,
    software: dict[str, Any] | None = None,
    model: dict[str, str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Assemble a metadata dict matching schema/metadata.schema.json.

    Auto-fills hardware/software/model from the collectors when not provided.
    Raises ValueError if a required field (submitter, date) is empty.
    """
    if not submitter:
        raise ValueError("submitter is required and must be non-empty")
    if not date_str:
        raise ValueError("date_str is required and must be non-empty")

    return {
        "submitter": submitter,
        "date": date_str,
        "hardware": hardware if hardware is not None else collect_hardware(),
        "software": software if software is not None else collect_software(),
        "model": model if model is not None else collect_model_info(""),
        "notes": notes,
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata as pretty JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def read_metadata(path: Path) -> dict[str, Any]:
    """Read and validate metadata. Raises FileNotFoundError, JSONDecodeError, KeyError."""
    data = json.loads(path.read_text())
    # Validate required keys are present (KeyError if missing).
    _ = data["submitter"]
    _ = data["date"]
    _ = data["hardware"]
    _ = data["software"]
    _ = data["model"]
    return data