from __future__ import annotations

from pathlib import Path

from oarp.models import RunConfig
from oarp.workflow import preflight_strict


def test_preflight_uses_target_python(monkeypatch, tmp_path: Path) -> None:
    local_repo = tmp_path / "pdf_repo"
    local_repo.mkdir(parents=True, exist_ok=True)
    (local_repo / "sample.pdf").write_bytes(b"%PDF-1.4 test")

    cfg = RunConfig(
        run_dir=tmp_path / "run",
        local_repo_paths=[str(local_repo)],
        local_file_glob="*.pdf",
        local_repo_recursive=True,
        tgi_mode="external",
        tgi_endpoint="http://127.0.0.1:8080/generate",
    )

    monkeypatch.setattr("oarp.workflow._python_version_from_exec", lambda py: (True, "3.11.15"))
    monkeypatch.setattr("oarp.workflow._module_probe", lambda py, mods: (True, [], ""))
    monkeypatch.setattr(
        "oarp.workflow.tgi_status",
        lambda local_cfg, check_generate=False: {
            "healthy": True,
            "generate_ok": bool(check_generate),
            "selected_tgi_platform": "",
            "tgi_emulation_used": False,
        },
    )

    report = preflight_strict(cfg, python_exec="/opt/custom/python3.11", check_tgi_generate=True)
    assert report.ok is True
    assert report.target_python_executable == "/opt/custom/python3.11"
    names = {str(item.get("name")) for item in report.checks}
    assert "tgi_generate_ok" in names
    assert any(name.startswith("local_repo_file_count:") for name in names)
