#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        capture_output=True,
    )


def _find_python311() -> str:
    if sys.version_info >= (3, 11):
        return sys.executable
    candidate = shutil.which("python3.11")
    if candidate:
        return candidate
    return ""


def _append_repeated_flag(base: list[str], flag: str, values: list[str]) -> list[str]:
    out = list(base)
    for value in values:
        token = str(value).strip()
        if token:
            out.extend([flag, token])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap strict OARP runtime from a near-clean Python 3.11 host.")
    parser.add_argument("--run", required=True, help="Run directory where .venv and artifacts will be created.")
    parser.add_argument("--profile", default="strict_full")
    parser.add_argument("--local-repo-path", action="append", default=[])
    parser.add_argument("--tgi-platform", default="auto")
    parser.add_argument("--tgi-mode", default="docker")
    parser.add_argument("--tgi-model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--python-exec", default="")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--run-full", action="store_true")
    parser.add_argument("--spec", default="")
    parser.add_argument("--query", default="")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).expanduser().resolve().parents[1]
    run_dir = Path(args.run).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = run_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"

    python_exec = str(args.python_exec or "").strip() or _find_python311()
    if not python_exec:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "python3.11_not_found",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if not venv_python.exists():
        created = _run([python_exec, "-m", "venv", str(venv_dir)])
        if created.returncode != 0:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "venv_create_failed",
                        "stderr": created.stderr,
                        "stdout": created.stdout,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return created.returncode

    pip_up = _run([str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    if pip_up.returncode != 0:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "pip_upgrade_failed",
                    "stderr": pip_up.stderr,
                    "stdout": pip_up.stdout,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return pip_up.returncode

    install = _run([str(venv_python), "-m", "pip", "install", "-e", ".[ml,pdf,html]"], cwd=repo_root)
    if install.returncode != 0:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "oarp_install_failed",
                    "stderr": install.stderr,
                    "stdout": install.stdout,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return install.returncode

    bootstrap_cmd = [
        str(venv_python),
        "-m",
        "oarp",
        "bootstrap-strict",
        "--out",
        str(run_dir),
        "--workflow-profile",
        str(args.profile),
        "--python-exec",
        str(venv_python),
        "--tgi-platform",
        str(args.tgi_platform),
        "--tgi-mode",
        str(args.tgi_mode),
        "--tgi-model-id",
        str(args.tgi_model_id),
        "--already-bootstrapped",
    ]
    bootstrap_cmd = _append_repeated_flag(bootstrap_cmd, "--local-repo-path", list(args.local_repo_path or []))
    boot = _run(bootstrap_cmd, cwd=repo_root)
    if boot.returncode != 0:
        print(boot.stdout)
        print(boot.stderr, file=sys.stderr)
        return boot.returncode

    preflight_cmd = [
        str(venv_python),
        "-m",
        "oarp",
        "preflight",
        "--run",
        str(run_dir),
        "--python-exec",
        str(venv_python),
        "--tgi-platform",
        str(args.tgi_platform),
        "--tgi-mode",
        str(args.tgi_mode),
        "--tgi-model-id",
        str(args.tgi_model_id),
    ]
    preflight_cmd = _append_repeated_flag(preflight_cmd, "--local-repo-path", list(args.local_repo_path or []))
    if args.skip_preflight:
        preflight_result = None
    else:
        preflight_result = _run(preflight_cmd, cwd=repo_root)
        if preflight_result.returncode != 0:
            print(preflight_result.stdout)
            print(preflight_result.stderr, file=sys.stderr)
            return preflight_result.returncode

    if args.run_full:
        if not args.spec or not args.query:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "reason": "run_full_requires_spec_and_query",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        run_full_cmd = [
            str(venv_python),
            "-m",
            "oarp",
            "run-full",
            "--spec",
            str(args.spec),
            "--query",
            str(args.query),
            "--out",
            str(run_dir),
            "--python-exec",
            str(venv_python),
            "--already-bootstrapped",
            "--tgi-platform",
            str(args.tgi_platform),
            "--tgi-mode",
            str(args.tgi_mode),
            "--tgi-model-id",
            str(args.tgi_model_id),
        ]
        run_full_cmd = _append_repeated_flag(run_full_cmd, "--local-repo-path", list(args.local_repo_path or []))
        full = _run(run_full_cmd, cwd=repo_root)
        if full.returncode != 0:
            print(full.stdout)
            print(full.stderr, file=sys.stderr)
            return full.returncode

    print(
        json.dumps(
            {
                "status": "ok",
                "repo_root": str(repo_root),
                "run_dir": str(run_dir),
                "venv_python": str(venv_python),
                "bootstrap_cmd": " ".join(bootstrap_cmd),
                "preflight_ran": bool(not args.skip_preflight),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
