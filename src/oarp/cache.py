from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.models import RunConfig
from oarp.runtime import now_iso


def _iso_to_dt(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _dt_to_iso(value: datetime | None) -> str:
    if value is None:
        return ""
    dt = value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(microsecond=0).isoformat()


class CacheManager:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.mode = str(cfg.cache_mode or "run_local").strip().lower()
        self.read_only = bool(cfg.cache_read_only)
        self.default_ttl_hours = max(1, int(cfg.cache_ttl_hours or 168))

        run_root = cfg.as_path() / "artifacts" / "cache_store"
        shared_root = Path(str(cfg.shared_cache_root or "")).expanduser().resolve() if str(cfg.shared_cache_root).strip() else None
        if self.mode not in {"run_local", "hybrid_local_shared"}:
            self.mode = "run_local"
        if self.mode == "hybrid_local_shared" and shared_root is None:
            self.mode = "run_local"

        self.run_root = run_root
        self.shared_root = shared_root if self.mode == "hybrid_local_shared" else None
        self.read_roots: list[Path] = []
        self.write_roots: list[Path] = []
        if self.shared_root is not None:
            self.read_roots.append(self.shared_root)
            if not self.read_only:
                self.write_roots.append(self.shared_root)
        self.read_roots.append(self.run_root)
        if not self.read_only:
            self.write_roots.append(self.run_root)

        self.audit_rows: list[dict[str, Any]] = []
        for root in self.read_roots:
            self._init_root(root)

    def make_key(self, namespace: str, *parts: Any) -> str:
        payload = {
            "namespace": str(namespace or ""),
            "parts": [str(item) for item in parts],
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8", errors="replace")).hexdigest()
        return digest

    def _init_root(self, root: Path) -> None:
        (root / "files").mkdir(parents=True, exist_ok=True)
        db_path = root / "cache_index.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    key_text TEXT NOT NULL,
                    ext TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    PRIMARY KEY(namespace, key_hash)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _entry(self, root: Path, namespace: str, key_hash: str) -> dict[str, Any] | None:
        db_path = root / "cache_index.sqlite"
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                """
                SELECT namespace, key_hash, key_text, ext, relative_path, created_at, expires_at, size_bytes
                FROM cache_entries
                WHERE namespace = ? AND key_hash = ?
                """,
                (namespace, key_hash),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {
            "namespace": str(row[0] or ""),
            "key_hash": str(row[1] or ""),
            "key_text": str(row[2] or ""),
            "ext": str(row[3] or ""),
            "relative_path": str(row[4] or ""),
            "created_at": str(row[5] or ""),
            "expires_at": str(row[6] or ""),
            "size_bytes": int(row[7] or 0),
        }

    def _expired(self, entry: dict[str, Any], ttl_hours: int | None = None) -> bool:
        now = datetime.now(timezone.utc)
        expires_at = _iso_to_dt(str(entry.get("expires_at") or ""))
        if expires_at is not None and now > expires_at:
            return True
        if ttl_hours is None:
            return False
        created = _iso_to_dt(str(entry.get("created_at") or ""))
        if created is None:
            return False
        return now > (created + timedelta(hours=max(1, int(ttl_hours))))

    def _record_audit(self, row: dict[str, Any]) -> None:
        payload = dict(row)
        payload["created_at"] = now_iso()
        self.audit_rows.append(payload)

    def get_bytes(self, namespace: str, key_hash: str, *, ttl_hours: int | None = None) -> bytes | None:
        ns = str(namespace or "").strip()
        key = str(key_hash or "").strip()
        if not ns or not key:
            return None
        for root in self.read_roots:
            entry = self._entry(root, ns, key)
            if entry is None:
                continue
            if self._expired(entry, ttl_hours):
                self._record_audit(
                    {
                        "namespace": ns,
                        "key_hash": key,
                        "event": "expired",
                        "cache_root": str(root),
                    }
                )
                continue
            target = root / str(entry.get("relative_path") or "")
            if not target.exists():
                self._record_audit(
                    {
                        "namespace": ns,
                        "key_hash": key,
                        "event": "missing_file",
                        "cache_root": str(root),
                    }
                )
                continue
            data = target.read_bytes()
            self._record_audit(
                {
                    "namespace": ns,
                    "key_hash": key,
                    "event": "hit",
                    "cache_root": str(root),
                    "size_bytes": len(data),
                }
            )
            return data
        self._record_audit(
            {
                "namespace": ns,
                "key_hash": key,
                "event": "miss",
            }
        )
        return None

    def put_bytes(
        self,
        namespace: str,
        key_hash: str,
        data: bytes,
        *,
        key_text: str = "",
        ext: str = "blob",
        ttl_hours: int | None = None,
    ) -> bool:
        ns = str(namespace or "").strip()
        key = str(key_hash or "").strip()
        if not ns or not key:
            return False
        if self.read_only:
            self._record_audit(
                {
                    "namespace": ns,
                    "key_hash": key,
                    "event": "skip_read_only",
                }
            )
            return False
        expires_dt = datetime.now(timezone.utc) + timedelta(hours=max(1, int(ttl_hours or self.default_ttl_hours)))
        wrote = False
        for root in self.write_roots:
            rel = Path("files") / ns / f"{key}.{ext}"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
            db_path = root / "cache_index.sqlite"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO cache_entries(namespace, key_hash, key_text, ext, relative_path, created_at, expires_at, size_bytes)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, key_hash) DO UPDATE SET
                        key_text = excluded.key_text,
                        ext = excluded.ext,
                        relative_path = excluded.relative_path,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        size_bytes = excluded.size_bytes
                    """,
                    (
                        ns,
                        key,
                        str(key_text or ""),
                        str(ext or "blob"),
                        str(rel),
                        now_iso(),
                        _dt_to_iso(expires_dt),
                        int(len(data)),
                    ),
                )
                conn.commit()
                wrote = True
            finally:
                conn.close()
            self._record_audit(
                {
                    "namespace": ns,
                    "key_hash": key,
                    "event": "write",
                    "cache_root": str(root),
                    "size_bytes": len(data),
                }
            )
        return wrote

    def get_json(self, namespace: str, key_hash: str, *, ttl_hours: int | None = None) -> dict[str, Any] | list[Any] | None:
        raw = self.get_bytes(namespace, key_hash, ttl_hours=ttl_hours)
        if raw is None:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        if isinstance(payload, (dict, list)):
            return payload
        return None

    def put_json(
        self,
        namespace: str,
        key_hash: str,
        payload: dict[str, Any] | list[Any],
        *,
        key_text: str = "",
        ttl_hours: int | None = None,
    ) -> bool:
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        return self.put_bytes(
            namespace,
            key_hash,
            data,
            key_text=key_text,
            ext="json",
            ttl_hours=ttl_hours,
        )

    def write_audit(self, path: Path | None = None) -> Path:
        out_path = path or (self.cfg.as_path() / "artifacts" / "cache_audit.parquet")
        rows = self.audit_rows if self.audit_rows else []
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=["namespace", "key_hash", "event", "cache_root", "size_bytes", "created_at"])
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                frame = pd.concat([existing, frame], ignore_index=True)
            except Exception:
                pass
        if not frame.empty:
            frame = frame.drop_duplicates(
                subset=["namespace", "key_hash", "event", "cache_root", "created_at"],
                keep="last",
            ).reset_index(drop=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_path, index=False)
        self.audit_rows = []
        return out_path


def prepare_feature_cache(run_cfg: RunConfig) -> CacheManager:
    return CacheManager(run_cfg)
