"""benchmark_suite/runner/serve.py — server lifecycle: cmd synthesis, GPU lock, spawn/teardown."""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import subprocess
import time
import warnings
from collections.abc import Generator
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, TextIO, assert_never
from urllib.parse import urlparse

import httpx

from benchmark_suite.recipe import Recipe, ResourcesSection

logger = logging.getLogger(__name__)

_DEFAULT_PORT = "8000"


@dataclass
class ServerHandle:
    """Handle to a running server subprocess."""

    pid: int
    process: subprocess.Popen[bytes]
    log_path: Path
    started_at: datetime
    backend: str
    teardown_timeout_s: float = 30.0

    def terminate(self) -> None:
        _terminate_process_group(self)

    def __enter__(self) -> ServerHandle:
        return self

    def __exit__(self, *args: object) -> None:
        self.terminate()


class GPUFileLock:
    """Process-level file lock around GPU-owning operations."""

    def __init__(self, lock_path: Path | None = None) -> None:
        self.lock_path = lock_path or Path.home() / ".cache" / "benchmark_suite" / "gpu.lock"
        self._file: TextIO | None = None

    def __enter__(self) -> Self:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.lock_path.open("w")
        try:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            warnings.warn(
                "fcntl unavailable; GPU lock is a no-op (no mutual exclusion)",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is None:
            return
        try:
            import fcntl

            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        finally:
            self._file.close()
            self._file = None


def _flag_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _flatten_flags(flags: dict[str, object]) -> list[str]:
    out: list[str] = []
    for key, value in flags.items():
        out.append(f"--{key}")
        out.append(_flag_value(value))
    return out


def _resources_flags(resources: ResourcesSection) -> list[str]:
    flags: list[str] = [
        "--tensor-parallel-size",
        str(resources.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(resources.pipeline_parallel_size),
        "--gpu-memory-utilization",
        str(resources.gpu_memory_utilization),
        "--max-model-len",
        str(resources.max_model_len),
        "--max-num-seqs",
        str(resources.max_num_seqs),
        "--dtype",
        resources.dtype,
    ]
    if resources.enforce_eager:
        flags.append("--enforce-eager")
    flags.extend(_flatten_flags(resources.extra_args))
    return flags


def _port_from_url(url: str) -> str:
    parsed = urlparse(url)
    return str(parsed.port) if parsed.port is not None else _DEFAULT_PORT


def _synthesize_vllm(recipe: Recipe) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        recipe.backend.model_path,
        "--port",
        _port_from_url(recipe.endpoint.url),
        "--host",
        "127.0.0.1",
    ]
    cmd.extend(_resources_flags(recipe.resources))
    cmd.extend(_flatten_flags(recipe.backend.vllm))
    return cmd


def _synthesize_llamacpp(recipe: Recipe) -> list[str]:
    cmd = ["llama-server", "-m", recipe.backend.model_path]
    cmd.extend(_flatten_flags(recipe.backend.llamacpp))
    return cmd


def _synthesize_tgi(recipe: Recipe) -> list[str]:
    cmd = ["text-generation-launcher", "--model-id", recipe.backend.model_path]
    cmd.extend(_flatten_flags(recipe.backend.tgi))
    return cmd


def synthesize_server_cmd(recipe: Recipe) -> list[str]:
    """Build the shell argv to launch a vllm / llamacpp / tgi server from a Recipe."""
    if recipe.runtime.server_cmd:
        return shlex.split(recipe.runtime.server_cmd)

    backend_type = recipe.backend.type
    if backend_type == "external":
        return []
    if backend_type == "vllm":
        return _synthesize_vllm(recipe)
    if backend_type == "llamacpp":
        return _synthesize_llamacpp(recipe)
    if backend_type == "tgi":
        return _synthesize_tgi(recipe)
    assert_never(backend_type)


async def wait_for_health(
    url: str,
    health_path: str = "/health",
    *,
    timeout_s: float = 900.0,
    poll_interval_s: float = 2.0,
) -> None:
    """Poll {url}{health_path} until 200 or timeout. Raises TimeoutError on timeout."""
    deadline = time.monotonic() + timeout_s
    health_url = url.rstrip("/") + health_path
    while True:
        try:
            resp = httpx.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"server at {url} did not become healthy within {timeout_s}s")
        await asyncio.sleep(poll_interval_s)


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(proc.pid), sig)


def _force_kill(handle: ServerHandle) -> None:
    proc = handle.process
    if proc.poll() is not None:
        return
    _signal_group(proc, signal.SIGKILL)
    proc.wait()


def _terminate_process_group(handle: ServerHandle) -> None:
    proc = handle.process
    if proc.poll() is not None:
        return
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=handle.teardown_timeout_s)
    except subprocess.TimeoutExpired:
        _force_kill(handle)


def spawn_server(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_dir: Path,
    backend: str = "vllm",
    teardown_timeout_s: float = 30.0,
) -> ServerHandle:
    """Spawn cmd in a subprocess with merged env, redirecting output to log_dir/<backend>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{backend}.log"
    full_env = {**os.environ, **env}
    log_file = log_path.open("wb")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=full_env,
            start_new_session=True,
        )
    finally:
        log_file.close()
    return ServerHandle(
        pid=process.pid,
        process=process,
        log_path=log_path,
        started_at=datetime.now(),
        backend=backend,
        teardown_timeout_s=teardown_timeout_s,
    )


def _build_env(recipe: Recipe) -> dict[str, str]:
    env = recipe.merged_env()
    devices = recipe.resources.devices
    if devices:
        env["HIP_VISIBLE_DEVICES"] = devices
        env["CUDA_VISIBLE_DEVICES"] = devices
    return env


@contextmanager
def managed_server(
    recipe: Recipe,
    *,
    log_dir: Path,
    gpu_lock: bool = True,
) -> Generator[ServerHandle | None, None, None]:
    """Context manager tying GPU lock + spawn + wait-for-health + teardown together."""
    if recipe.backend.type == "external":
        yield None
        return

    lock: Any = GPUFileLock() if gpu_lock else nullcontext()
    with lock:
        handle = spawn_server(
            synthesize_server_cmd(recipe),
            env=_build_env(recipe),
            log_dir=log_dir,
            backend=recipe.backend.type,
            teardown_timeout_s=recipe.runtime.teardown_timeout_s,
        )
        try:
            asyncio.run(
                wait_for_health(
                    recipe.endpoint.url,
                    recipe.runtime.health_path,
                    timeout_s=recipe.runtime.startup_wait_s,
                )
            )
            yield handle
        except Exception:
            logger.exception("server lifecycle failed; force-killing pid %d", handle.pid)
            _force_kill(handle)
            raise
        finally:
            _terminate_process_group(handle)