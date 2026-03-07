from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from oarp.runtime import now_iso
from oarp.service_models import JobStatus, JobStatusPayload, JobType


def init_jobs_db(path: str | Path) -> Path:
    db_path = Path(path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                result_json TEXT NOT NULL DEFAULT '{}',
                run_dir TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                retries INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 0,
                parent_job_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                finished_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS job_artifacts (
                job_id TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (job_id, name, path)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _as_payload(row: sqlite3.Row) -> JobStatusPayload:
    return JobStatusPayload(
        job_id=str(row["job_id"]),
        job_type=JobType(str(row["job_type"])),
        status=JobStatus(str(row["status"])),
        run_dir=str(row["run_dir"] or ""),
        retries=int(row["retries"] or 0),
        max_retries=int(row["max_retries"] or 0),
        created_at=str(row["created_at"] or ""),
        started_at=str(row["started_at"] or ""),
        finished_at=str(row["finished_at"] or ""),
        error=str(row["error"] or ""),
    )


def create_job(
    *,
    db_path: str | Path,
    job_type: JobType,
    payload: dict[str, Any],
    run_dir: str,
    retries: int = 0,
    max_retries: int = 0,
    parent_job_id: str = "",
) -> JobStatusPayload:
    init_jobs_db(db_path)
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    now = now_iso()
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    try:
        conn.execute(
            """
            INSERT INTO jobs(
                job_id, job_type, status, payload_json, run_dir, retries, max_retries,
                parent_job_id, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type.value,
                JobStatus.PENDING.value,
                json.dumps(payload, sort_keys=True),
                str(run_dir),
                int(retries),
                int(max_retries),
                str(parent_job_id or ""),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    row = get_job(db_path=db_path, job_id=job_id)
    if row is None:
        raise RuntimeError(f"failed to create job: {job_id}")
    return row


def get_job(db_path: str | Path, job_id: str) -> JobStatusPayload | None:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (str(job_id),),
        ).fetchone()
        if row is None:
            return None
        return _as_payload(row)
    finally:
        conn.close()


def get_job_payload(db_path: str | Path, job_id: str) -> dict[str, Any] | None:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT payload_json FROM jobs WHERE job_id = ?",
            (str(job_id),),
        ).fetchone()
        if row is None:
            return None
        try:
            out = json.loads(str(row["payload_json"] or "{}"))
        except Exception:
            out = {}
        return out if isinstance(out, dict) else {}
    finally:
        conn.close()


def get_job_result(db_path: str | Path, job_id: str) -> dict[str, Any] | None:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT result_json FROM jobs WHERE job_id = ?",
            (str(job_id),),
        ).fetchone()
        if row is None:
            return None
        try:
            out = json.loads(str(row["result_json"] or "{}"))
        except Exception:
            out = {}
        return out if isinstance(out, dict) else {}
    finally:
        conn.close()


def claim_next_pending_job(db_path: str | Path) -> dict[str, Any] | None:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = ?
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (JobStatus.PENDING.value,),
        ).fetchone()
        if row is None:
            conn.commit()
            return None
        now = now_iso()
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, started_at = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (JobStatus.RUNNING.value, now, now, str(row["job_id"])),
        )
        conn.commit()
        return {
            "job_id": str(row["job_id"]),
            "job_type": str(row["job_type"]),
            "run_dir": str(row["run_dir"]),
            "retries": int(row["retries"] or 0),
            "max_retries": int(row["max_retries"] or 0),
            "payload": json.loads(str(row["payload_json"] or "{}")),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def complete_job(
    *,
    db_path: str | Path,
    job_id: str,
    result: dict[str, Any],
) -> None:
    init_jobs_db(db_path)
    now = now_iso()
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    try:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, result_json = ?, finished_at = ?, updated_at = ?, error = ''
            WHERE job_id = ?
            """,
            (
                JobStatus.SUCCEEDED.value,
                json.dumps(result, sort_keys=True),
                now,
                now,
                str(job_id),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fail_job(
    *,
    db_path: str | Path,
    job_id: str,
    error: str,
) -> None:
    init_jobs_db(db_path)
    now = now_iso()
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    try:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?, error = ?, finished_at = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (
                JobStatus.FAILED.value,
                str(error or ""),
                now,
                now,
                str(job_id),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def append_job_artifact(
    *,
    db_path: str | Path,
    job_id: str,
    name: str,
    path: str | Path,
) -> None:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    try:
        conn.execute(
            """
            INSERT OR IGNORE INTO job_artifacts(job_id, name, path, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (str(job_id), str(name), str(path), now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def list_job_artifacts(db_path: str | Path, job_id: str) -> list[dict[str, str]]:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT name, path, created_at FROM job_artifacts WHERE job_id = ? ORDER BY created_at ASC",
            (str(job_id),),
        ).fetchall()
        return [
            {
                "name": str(row["name"]),
                "path": str(row["path"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def create_retry_job(
    *,
    db_path: str | Path,
    job_id: str,
) -> JobStatusPayload:
    init_jobs_db(db_path)
    conn = sqlite3.connect(Path(db_path).expanduser().resolve())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
        if row is None:
            raise FileNotFoundError(f"job not found: {job_id}")
        retries = int(row["retries"] or 0)
        max_retries = int(row["max_retries"] or 0)
        if retries >= max_retries:
            raise ValueError(f"retry limit reached ({retries}/{max_retries}) for {job_id}")
        payload = json.loads(str(row["payload_json"] or "{}"))
        return create_job(
            db_path=db_path,
            job_type=JobType(str(row["job_type"])),
            payload=payload if isinstance(payload, dict) else {},
            run_dir=str(row["run_dir"] or ""),
            retries=retries + 1,
            max_retries=max_retries,
            parent_job_id=str(row["job_id"] or ""),
        )
    finally:
        conn.close()
