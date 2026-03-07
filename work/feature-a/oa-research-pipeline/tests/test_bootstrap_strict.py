from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from oarp.models import RunConfig
from oarp.workflow import _resolve_tgi_platform


def test_bootstrap_script_help() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_strict.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    assert "Bootstrap strict OARP runtime" in proc.stdout


def test_tgi_platform_auto_falls_back_to_amd64_on_arm(monkeypatch, tmp_path: Path) -> None:
    cfg = RunConfig(run_dir=tmp_path / "run", tgi_platform="auto", tgi_mode="docker")
    monkeypatch.setattr("oarp.workflow._host_arch", lambda: "arm64")
    monkeypatch.setattr("oarp.workflow._docker_manifest_supports", lambda image, platform_token: False)
    selected, emulated, reason = _resolve_tgi_platform(cfg)
    assert selected == "linux/amd64"
    assert emulated is True
    assert "fallback" in reason
