"""benchmark_suite/submission.py — bundle a result dir into a submission tarball
and POST it to a leaderboard endpoint.

This module is shared by `bs export` (writes the tarball to disk) and `bs submit`
(wraps it in multipart/form-data and POSTs to LEADERBOARD_URL). The tarball
contents are the same in both cases:

    recipe.yaml + summary.csv + summary.json + README.md + metadata.json + artifacts/

`bs submit` returns a SubmitResult with the public_url of the new run. If the
endpoint is unreachable, it falls back to writing the bundle to
`~/.cache/bs/submissions/` so the user can retry later without losing data.
"""
from __future__ import annotations

import io
import json
import os
import socket
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import httpx
import yaml

from benchmark_suite.scoring.metadata_collector import (
    build_metadata,
    collect_hardware,
    collect_model_info,
    collect_software,
)

DEFAULT_LEADERBOARD_URL = "https://bench.uncool.red"
SUBMISSION_TIMEOUT_S = 30.0
ARTIFACT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per-file cap
REQUIRED_FILES = ("summary.csv", "summary.json", "README.md")
CACHE_DIR = Path(os.environ.get("BS_CACHE_DIR", "~/.cache/bs/submissions")).expanduser()


class SubmitResult(TypedDict, total=False):
    """Result of `bs submit`. Exactly one of `submission_id` or `local_path` is set.

    - submission_id + public_url: posted successfully to the leaderboard.
    - local_path: endpoint unreachable; bundle cached for retry.
    - error + details: validation failed before any network call.
    """

    submission_id: str
    public_url: str
    endpoint: str
    local_path: str
    error: str
    details: str
    http_status: int


def _merge_metadata(
    overrides: dict[str, object] | None,
    *,
    auto_detect: bool,
    model_path: str,
) -> dict[str, Any]:
    """Assemble the metadata dict: auto-detect (optional) merged with overrides."""
    hardware: dict[str, Any] = collect_hardware() if auto_detect else {}
    software: dict[str, Any] = collect_software() if auto_detect else {}
    model: dict[str, str] = collect_model_info(model_path) if auto_detect else {}

    if overrides:
        for section in ("hardware", "software", "model"):
            extra = overrides.get(section)
            if isinstance(extra, dict):
                target = {"hardware": hardware, "software": software, "model": model}[section]
                target.update(extra)  # type: ignore[arg-type]

    return {"hardware": hardware, "software": software, "model": model}


def _read_submission_inputs(
    result_dir: Path,
    *,
    submitter: str,
    metadata_overrides: dict[str, object] | None,
    auto_detect: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Build the submission tarball in memory. Returns (tarball_bytes, metadata_dict).

    Raises:
        FileNotFoundError: if result_dir lacks summary.json.
    """
    summary_json = result_dir / "summary.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"no summary.json in {result_dir}")

    data: Any = json.loads(summary_json.read_text())
    recipe_dict: dict[str, Any] = data["recipe"]
    recipe_yaml = yaml.safe_dump(recipe_dict, sort_keys=False)

    model_path = str(recipe_dict.get("backend", {}).get("model_path", ""))
    sections = _merge_metadata(metadata_overrides, auto_detect=auto_detect, model_path=model_path)
    metadata = build_metadata(
        submitter=submitter,
        date_str=datetime.now(UTC).date().isoformat(),
        hardware=sections["hardware"],
        software=sections["software"],
        model=sections["model"],
        notes=str(metadata_overrides.get("notes", "")) if metadata_overrides else "",
    )

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        recipe_info = tarfile.TarInfo("recipe.yaml")
        recipe_bytes = recipe_yaml.encode()
        recipe_info.size = len(recipe_bytes)
        tf.addfile(recipe_info, io.BytesIO(recipe_bytes))

        for name in REQUIRED_FILES:
            src = result_dir / name
            if src.exists():
                tf.add(src, arcname=name)

        metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode()
        metadata_info = tarfile.TarInfo("metadata.json")
        metadata_info.size = len(metadata_bytes)
        tf.addfile(metadata_info, io.BytesIO(metadata_bytes))

        artifacts_dir = result_dir / "artifacts"
        if artifacts_dir.is_dir():
            for artifact in sorted(artifacts_dir.rglob("*")):
                if not artifact.is_file():
                    continue
                if artifact.stat().st_size > ARTIFACT_MAX_BYTES:
                    continue
                tf.add(artifact, arcname=str(artifact.relative_to(result_dir)))

    return buf.getvalue(), metadata


def export_submission(
    result_dir: Path,
    *,
    submitter: str,
    output: Path,
    metadata_overrides: dict[str, object] | None = None,
    auto_detect: bool = False,
) -> Path:
    """Bundle result_dir + recipe + metadata into a gzipped tar at `output`."""
    if not submitter:
        raise ValueError("submitter is required and must be non-empty")

    tar_bytes, _ = _read_submission_inputs(
        result_dir,
        submitter=submitter,
        metadata_overrides=metadata_overrides,
        auto_detect=auto_detect,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tar_bytes)
    return output


def resolve_leaderboard_url(explicit: str | None) -> str:
    """Resolve the leaderboard URL: explicit > LEADERBOARD_URL env > default."""
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("LEADERBOARD_URL")
    if env:
        return env.rstrip("/")
    return DEFAULT_LEADERBOARD_URL


def _post_multipart(
    endpoint: str,
    *,
    tar_bytes: bytes,
    handle: str,
    date: str,
    metadata: dict[str, Any],
) -> httpx.Response:
    """POST the tarball as multipart/form-data to the leaderboard API."""
    metadata_str = json.dumps(metadata)
    files = {"file": ("submission.tar.gz", io.BytesIO(tar_bytes), "application/gzip")}
    data = {
        "handle": handle,
        "date": date,
        "metadata": metadata_str,
    }
    client = httpx.Client(timeout=SUBMISSION_TIMEOUT_S)
    try:
        return client.post(
            f"{endpoint}/api/submissions",
            files=files,
            data=data,
        )
    finally:
        client.close()


def _cache_offline(tar_bytes: bytes, *, handle: str) -> Path:
    """Write the tarball to ~/.cache/bs/submissions for later retry."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = CACHE_DIR / f"{handle}-{ts}.tar.gz"
    out.write_bytes(tar_bytes)
    return out


def submit_submission(
    result_dir: Path,
    *,
    handle: str,
    endpoint: str | None = None,
    metadata_overrides: dict[str, object] | None = None,
    auto_detect: bool = False,
    allow_offline_fallback: bool = True,
) -> SubmitResult:
    """Build the submission tarball and POST it to the leaderboard.

    Network errors are caught and, when `allow_offline_fallback=True`, the
    tarball is written to `~/.cache/bs/submissions/` for retry. Validation
    errors raise (no network call happens until the tarball is built).
    """
    if not handle:
        raise ValueError("handle is required and must be non-empty")

    tar_bytes, metadata = _read_submission_inputs(
        result_dir,
        submitter=handle,
        metadata_overrides=metadata_overrides,
        auto_detect=auto_detect,
    )
    date = str(metadata.get("date") or datetime.now(UTC).date().isoformat())
    url = resolve_leaderboard_url(endpoint)

    try:
        resp = _post_multipart(
            url,
            tar_bytes=tar_bytes,
            handle=handle,
            date=date,
            metadata=metadata,
        )
    except (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        socket.gaierror,
        OSError,
    ) as exc:
        if allow_offline_fallback:
            local = _cache_offline(tar_bytes, handle=handle)
            return SubmitResult(
                endpoint=url,
                local_path=str(local),
                error="offline",
                details=f"{type(exc).__name__}: {exc}",
            )
        return SubmitResult(
            endpoint=url,
            error="offline",
            details=f"{type(exc).__name__}: {exc}",
        )

    if resp.status_code in (200, 201):
        body: dict[str, object] = resp.json()
        sub_id_raw = body.get("submission_id", "")
        public_raw = body.get("public_url", f"{url}/runs/{sub_id_raw}")
        sub_id = str(sub_id_raw) if sub_id_raw else ""
        public = str(public_raw) if public_raw else f"{url}/runs/{sub_id}"
        return SubmitResult(
            submission_id=sub_id,
            public_url=public,
            endpoint=url,
            http_status=resp.status_code,
        )

    # 4xx/5xx — return structured error from the server
    error_body: dict[str, object]
    try:
        error_body = resp.json()
    except (json.JSONDecodeError, ValueError):
        error_body = {}
    return SubmitResult(
        endpoint=url,
        error=str(error_body.get("error", f"http_{resp.status_code}")),
        details=str(error_body.get("details", resp.text[:500])),
        http_status=resp.status_code,
    )