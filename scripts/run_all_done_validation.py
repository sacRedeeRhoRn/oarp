#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _append_repeated_flag(base: list[str], flag: str, values: list[str]) -> list[str]:
    out = list(base)
    for value in values:
        token = str(value).strip()
        if token:
            out.extend([flag, token])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run strict all-done validation via bootstrapped OARP venv.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--shadow-gold", default="")
    parser.add_argument("--gold-context", default="")
    parser.add_argument("--precision-gate", type=float, default=0.80)
    parser.add_argument("--context-completeness-gate", type=float, default=0.70)
    parser.add_argument("--context-precision-gate", type=float, default=0.80)
    parser.add_argument("--local-repo-path", action="append", default=[])
    args = parser.parse_args(argv)

    repo_root = Path(__file__).expanduser().resolve().parents[1]
    run_dir = Path(args.run).expanduser().resolve()
    venv_python = run_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print(f"missing bootstrapped interpreter: {venv_python}", file=sys.stderr)
        return 2

    cmd = [
        str(venv_python),
        "-m",
        "oarp",
        "validate-all-done",
        "--spec",
        str(Path(args.spec).expanduser().resolve()),
        "--query",
        str(args.query),
        "--out",
        str(run_dir),
        "--gold",
        str(Path(args.gold).expanduser().resolve()),
        "--precision-gate",
        str(float(args.precision_gate)),
        "--context-completeness-gate",
        str(float(args.context_completeness_gate)),
        "--context-precision-gate",
        str(float(args.context_precision_gate)),
        "--already-bootstrapped",
    ]
    if str(args.shadow_gold or "").strip():
        cmd.extend(["--shadow-gold", str(Path(args.shadow_gold).expanduser().resolve())])
    if str(args.gold_context or "").strip():
        cmd.extend(["--gold-context", str(Path(args.gold_context).expanduser().resolve())])
    cmd = _append_repeated_flag(cmd, "--local-repo-path", list(args.local_repo_path or []))

    completed = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
