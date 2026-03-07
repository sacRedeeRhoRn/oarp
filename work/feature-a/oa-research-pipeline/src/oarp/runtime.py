from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_run_layout(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir).expanduser().resolve()
    artifacts = root / "artifacts"
    raw = root / "raw"
    outputs = root / "outputs"
    plots = outputs / "plots"
    state = root / "state"
    for path in (root, artifacts, raw, outputs, plots, state):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": root,
        "artifacts": artifacts,
        "raw": raw,
        "outputs": outputs,
        "plots": plots,
        "state": state,
    }


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_run_state(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    layout = ensure_run_layout(run_dir)
    target = layout["state"] / "run_state.json"
    write_json(target, payload)
    return target


def load_run_state(run_dir: str | Path) -> dict[str, Any]:
    layout = ensure_run_layout(run_dir)
    target = layout["state"] / "run_state.json"
    if not target.exists():
        raise FileNotFoundError(f"run state not found: {target}")
    payload = read_json(target)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid run state payload: {target}")
    return payload


def init_index_db(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                query TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lineage (
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                source_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_artifact(
    *,
    db_path: str | Path,
    run_id: str,
    name: str,
    path: str | Path,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO artifacts(run_id, name, path, created_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(run_id, name)
            DO UPDATE SET path = excluded.path, created_at = excluded.created_at
            """,
            (run_id, name, str(path), now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def append_lineage(
    *,
    db_path: str | Path,
    run_id: str,
    stage: str,
    source_name: str,
    target_name: str,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO lineage(run_id, stage, source_name, target_name, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (run_id, stage, source_name, target_name, now_iso()),
        )
        conn.commit()
    finally:
        conn.close()
