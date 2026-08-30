"""Tests for benchmark_suite.runner.serve — cmd synthesis, GPU lock, lifecycle."""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from benchmark_suite.recipe import Recipe
from benchmark_suite.runner.serve import (
    GPUFileLock,
    managed_server,
    spawn_server,
    synthesize_server_cmd,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fake_serve_recipe(port: int) -> Recipe:
    script = Path(__file__).parent / "_fake_serve.sh"
    return Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/Test-Model"},
            "endpoint": {"url": f"http://127.0.0.1:{port}"},
            "runtime": {
                "server_cmd": f"bash {script}",
                "env": {"FAKE_SERVE_PORT": str(port)},
                "startup_wait_s": 30.0,
            },
        }
    )


def test_synthesize_vllm_cmd() -> None:
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {"type": "vllm", "model_path": "/models/Test-Model"},
            "endpoint": {"url": "http://127.0.0.1:8000"},
            "resources": {"tensor_parallel_size": 4, "devices": "0,1,2,3"},
        }
    )
    assert synthesize_server_cmd(recipe) == [
        "vllm",
        "serve",
        "/models/Test-Model",
        "--port",
        "8000",
        "--host",
        "127.0.0.1",
        "--tensor-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--dtype",
        "float16",
    ]


def test_synthesize_vllm_cmd_translates_backend_dict() -> None:
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {
                "type": "vllm",
                "model_path": "/models/Test-Model",
                "vllm": {
                    "compilation-config": "{...}",
                    "language-model-only": True,
                },
            },
        }
    )
    cmd = synthesize_server_cmd(recipe)
    assert cmd[-4:] == [
        "--compilation-config",
        "{...}",
        "--language-model-only",
        "true",
    ]


def test_synthesize_llamacpp_cmd() -> None:
    recipe = Recipe.model_validate(
        {
            "meta": {"name": "x", "description": "y"},
            "backend": {
                "type": "llamacpp",
                "model_path": "/models/Test-Model.gguf",
                "llamacpp": {"n-gpu-layers": 99, "flash-attn": True},
            },
        }
    )
    assert synthesize_server_cmd(recipe) == [
        "llama-server",
        "-m",
        "/models/Test-Model.gguf",
        "--n-gpu-layers",
        "99",
        "--flash-attn",
        "true",
    ]


def test_synthesize_external_cmd_returns_empty() -> None:
    recipe = Recipe.model_validate({"meta": {"name": "x", "description": "y"}})
    assert synthesize_server_cmd(recipe) == []


def test_gpu_file_lock_serializes_two_holders(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    acquired = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def holder_a() -> None:
        with GPUFileLock(lock_path):
            order.append("a-enter")
            acquired.set()
            release.wait(timeout=2.0)
            order.append("a-exit")

    def holder_b() -> None:
        acquired.wait(timeout=2.0)
        with GPUFileLock(lock_path):
            order.append("b-enter")
            order.append("b-exit")

    ta = threading.Thread(target=holder_a)
    tb = threading.Thread(target=holder_b)
    ta.start()
    tb.start()
    time.sleep(0.1)
    assert order == ["a-enter"]  # b is blocked on the lock
    release.set()
    ta.join()
    tb.join()
    assert order == ["a-enter", "a-exit", "b-enter", "b-exit"]


def test_gpu_file_lock_released_on_exception(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    with pytest.raises(RuntimeError), GPUFileLock(lock_path):
        raise RuntimeError("boom")
    acquired = threading.Event()

    def try_acquire() -> None:
        with GPUFileLock(lock_path):
            acquired.set()

    t = threading.Thread(target=try_acquire)
    t.start()
    assert acquired.wait(timeout=2.0)
    t.join()


def test_spawn_server_creates_log_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    handle = spawn_server(["sleep", "1"], env={}, log_dir=log_dir, backend="vllm")
    try:
        assert handle.log_path.exists()
        assert handle.process.poll() is None  # still alive
    finally:
        handle.terminate()


def test_managed_server_external_yields_none(tmp_path: Path) -> None:
    recipe = Recipe.model_validate({"meta": {"name": "x", "description": "y"}})
    with managed_server(recipe, log_dir=tmp_path) as handle:
        assert handle is None


def test_managed_server_teardown_terminates(tmp_path: Path) -> None:
    recipe = _fake_serve_recipe(_free_port())
    with managed_server(recipe, log_dir=tmp_path, gpu_lock=False) as handle:
        assert handle is not None
        assert handle.process.poll() is None
    assert handle.process.poll() is not None  # terminated after context exit


def test_managed_server_gpu_lock_held_during_lifecycle(tmp_path: Path) -> None:
    recipe = _fake_serve_recipe(_free_port())
    acquired = threading.Event()

    def try_acquire() -> None:
        with GPUFileLock():
            acquired.set()

    with managed_server(recipe, log_dir=tmp_path, gpu_lock=True) as handle:
        assert handle is not None
        t = threading.Thread(target=try_acquire)
        t.start()
        time.sleep(0.2)
        assert not acquired.is_set()  # second holder blocked while server runs
    t.join()
    assert acquired.is_set()