"""Tests for `bs setup` — onboarding wizard for localmaxxing.com + lmx."""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmark_suite.setup import (
    LMX_AUTH_HINT,
    LMX_INSTALL_HINT,
    SetupResult,
    detect_hardware,
    find_api_key,
    lmx_version,
    redact_key,
    run_setup,
)


def _which_returns(fake: Path) -> Callable[[str], str | None]:
    def _w(name: str) -> str | None:
        return str(fake) if name == "lmx" else None

    return _w


def _which_never() -> Callable[[str], str | None]:
    def _w(name: str) -> str | None:
        return None

    return _w


def _detect_hardware_stub(out: Path) -> dict[str, object]:
    return {
        "hwClass": "DISCRETE_GPU",
        "gpuName": "X",
        "gpuCount": 1,
        "vramGb": 8,
    }


def _detect_hardware_v620(out: Path) -> dict[str, object]:
    return {
        "hwClass": "DISCRETE_GPU",
        "gpuName": "Radeon PRO V620",
        "gpuCount": 4,
        "vramGb": 32,
    }


# ----- redact_key -----


def test_redact_key_short_key() -> None:
    assert redact_key("bhk_ab") == "bhk_****"


def test_redact_key_long_key() -> None:
    out = redact_key("bhk_1234567890abcdef")
    assert out.startswith("bhk_1234")
    assert "****" in out
    assert "1234567890abcdef" not in out


def test_redact_key_empty() -> None:
    assert redact_key("") == ""


# ----- find_api_key -----


def test_find_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMX_API_KEY", "bhk_env_key")
    assert find_api_key() == "bhk_env_key"


def test_find_api_key_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMX_API_KEY", raising=False)
    monkeypatch.setattr(
        "benchmark_suite.setup.LMX_CONFIG_PATH",
        Path("/tmp/fake-lmx-config.json"),
    )
    Path("/tmp/fake-lmx-config.json").write_text(json.dumps({"apiKey": "bhk_cfg_key"}))
    try:
        assert find_api_key() == "bhk_cfg_key"
    finally:
        Path("/tmp/fake-lmx-config.json").unlink(missing_ok=True)


def test_find_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMX_API_KEY", raising=False)
    monkeypatch.setattr("benchmark_suite.setup.LMX_CONFIG_PATH", Path("/tmp/does-not-exist.json"))
    assert find_api_key() is None


def test_find_api_key_config_with_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMX_API_KEY", raising=False)
    p = Path("/tmp/empty-config.json")
    p.write_text(json.dumps({"otherField": "x"}))
    monkeypatch.setattr("benchmark_suite.setup.LMX_CONFIG_PATH", p)
    try:
        assert find_api_key() is None
    finally:
        p.unlink(missing_ok=True)


# ----- lmx_version -----


def test_lmx_version_parses_standard(tmp_path: Path) -> None:
    fake = tmp_path / "fake-lmx"
    fake.write_text("#!/usr/bin/env bash\necho 'lmx version 1.2.3'\n")
    fake.chmod(0o755)
    assert lmx_version(str(fake)) == "1.2.3"


def test_lmx_version_parses_stderr(tmp_path: Path) -> None:
    fake = tmp_path / "fake-lmx"
    fake.write_text("#!/usr/bin/env bash\necho 'v4.5.6' >&2\nexit 1\n")
    fake.chmod(0o755)
    assert lmx_version(str(fake)) == "4.5.6"


def test_lmx_version_no_version_string(tmp_path: Path) -> None:
    fake = tmp_path / "fake-lmx"
    fake.write_text("#!/usr/bin/env bash\necho 'no version here'\n")
    fake.chmod(0o755)
    assert lmx_version(str(fake)) == "no version here"


def test_lmx_version_missing_binary(tmp_path: Path) -> None:
    assert lmx_version(str(tmp_path / "nope")) is None


# ----- detect_hardware -----


def test_detect_hardware_uses_lmx_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = tmp_path / "fake-lmx"
    payload_path = tmp_path / "hardware.json"
    payload_str = str(payload_path)
    fake.write_text(
        "#!/usr/bin/env bash\n"
        "python3 - <<'PY'\n"
        "import json\n"
        "out = {'hwClass': 'DISCRETE_GPU', 'gpuName': 'TEST GPU', "
        "'gpuCount': 2, 'vramGb': 16}\n"
        f"open('{payload_str}', 'w').write(json.dumps(out, indent=2))\n"
        "PY\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr("shutil.which", _which_returns(fake))

    result = detect_hardware(payload_path)
    assert isinstance(result, dict)
    assert result["gpuName"] == "TEST GPU"
    assert result["gpuCount"] == 2
    assert payload_path.exists()


def test_detect_hardware_falls_back_to_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shutil.which", _which_never()
    )

    class _Props:
        name = "Radeon PRO V620"
        total_memory = 32 * 1024**3

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(_idx: int) -> _Props:
            return _Props()

        @staticmethod
        def device_count() -> int:
            return 4

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setitem(sys.modules, "torch", _Torch())

    payload_path = tmp_path / "hardware.json"
    result = detect_hardware(payload_path)
    assert isinstance(result, dict)
    assert result["gpuName"] == "Radeon PRO V620"
    assert result["gpuCount"] == 4
    assert result["vramGb"] == 32
    assert payload_path.exists()


def test_detect_hardware_no_lmx_no_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", _which_never())
    monkeypatch.setitem(sys.modules, "torch", None)

    assert detect_hardware(tmp_path / "hardware.json") is None


# ----- run_setup -----


def test_run_setup_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_lmx = tmp_path / "fake-lmx"
    fake_lmx.write_text("#!/usr/bin/env bash\necho 'lmx version 1.0.0'\n")
    fake_lmx.chmod(0o755)
    monkeypatch.setattr("shutil.which", _which_returns(fake_lmx))
    monkeypatch.setenv("LMX_API_KEY", "bhk_abcdefghij")
    monkeypatch.setattr(
        "benchmark_suite.setup.detect_hardware",
        _detect_hardware_v620,
    )

    result = run_setup(skip_auth=False, run_auth_if_missing=False)

    assert isinstance(result, SetupResult)
    assert result.lmx_path == str(fake_lmx)
    assert result.lmx_version == "1.0.0"
    assert result.api_key_prefix == "bhk_abcd****"
    assert result.hardware is not None
    assert result.hardware["gpuName"] == "Radeon PRO V620"
    assert all(step.ok for step in result.steps)


def test_run_setup_missing_lmx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", _which_never())
    monkeypatch.setenv("LMX_API_KEY", "bhk_test")
    monkeypatch.setattr(
        "benchmark_suite.setup.detect_hardware",
        _detect_hardware_stub,
    )

    result = run_setup(skip_auth=True)

    assert result.lmx_path is None
    assert any(not step.ok and step.name == "lmx installed" for step in result.steps)


def test_run_setup_missing_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_lmx = tmp_path / "fake-lmx"
    fake_lmx.write_text("#!/usr/bin/env bash\necho 'lmx 1.0'\n")
    fake_lmx.chmod(0o755)
    monkeypatch.setattr("shutil.which", _which_returns(fake_lmx))
    monkeypatch.delenv("LMX_API_KEY", raising=False)
    monkeypatch.setattr(
        "benchmark_suite.setup.LMX_CONFIG_PATH", Path("/tmp/does-not-exist-config.json")
    )
    monkeypatch.setattr(
        "benchmark_suite.setup.detect_hardware",
        _detect_hardware_stub,
    )

    result = run_setup(skip_auth=False, run_auth_if_missing=False)

    assert result.api_key_prefix is None
    assert any(not step.ok and step.name == "API key" for step in result.steps)


def test_run_setup_skips_auth_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_lmx = tmp_path / "fake-lmx"
    fake_lmx.write_text("#!/usr/bin/env bash\necho 'lmx 1.0'\n")
    fake_lmx.chmod(0o755)
    monkeypatch.setattr("shutil.which", _which_returns(fake_lmx))
    monkeypatch.setattr(
        "benchmark_suite.setup.detect_hardware",
        _detect_hardware_stub,
    )

    result = run_setup(skip_auth=True)

    api_step = next(s for s in result.steps if s.name == "API key")
    assert not api_step.ok
    assert "skipped" in api_step.summary.lower()


# ----- constants exposed -----


def test_lmx_install_hint_mentions_github_url() -> None:
    assert "LottoLottoLotto" in LMX_INSTALL_HINT
    assert "releases/latest" in LMX_INSTALL_HINT


def test_lmx_auth_hint_mentions_all_three_methods() -> None:
    assert "LMX_API_KEY" in LMX_AUTH_HINT
    assert "auth login" in LMX_AUTH_HINT
    assert "key-stdin" in LMX_AUTH_HINT