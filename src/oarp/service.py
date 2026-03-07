from __future__ import annotations

import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from oarp.jobs import create_job, create_retry_job
from oarp.service_api import (
    _jobs_db,
    dump_job_payload,
    get_job_artifacts,
    get_job_status,
    run_one_pending_job,
)
from oarp.service_models import (
    JobRef,
    JobType,
    RecipeGenerateRequest,
    ResearchIndexRequest,
    ServiceConfig,
)
from oarp.topic_spec import ensure_query, load_topic_spec


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _create_research_job(payload: dict[str, Any], cfg: ServiceConfig) -> JobRef:
    spec_path = str(payload.get("spec_path") or "").strip()
    if not spec_path:
        raise ValueError("spec_path is required")
    query = str(payload.get("query") or "").strip()
    run_dir = str(payload.get("run_dir") or "").strip()
    cfg_overrides = payload.get("cfg_overrides")
    if not isinstance(cfg_overrides, dict):
        cfg_overrides = {}

    spec = load_topic_spec(spec_path)
    if not query:
        query = ensure_query(spec, None)
    if not run_dir:
        run_dir = str(cfg.jobs_runs_root() / f"research_{uuid.uuid4().hex[:10]}")

    req = ResearchIndexRequest(
        spec_path=str(Path(spec_path).expanduser().resolve()),
        query=query,
        run_dir=run_dir,
        cfg_overrides=cfg_overrides,
    )
    row = create_job(
        db_path=_jobs_db(cfg),
        job_type=JobType.RESEARCH_INDEX,
        payload=req.model_dump(mode="python"),
        run_dir=run_dir,
    )
    return JobRef(job_id=row.job_id, status=row.status)


def _create_recipe_job(payload: dict[str, Any], cfg: ServiceConfig) -> JobRef:
    req = RecipeGenerateRequest.model_validate(payload)
    run_dir = req.knowledge_run_dir.strip() or str(cfg.jobs_runs_root() / f"recipe_{uuid.uuid4().hex[:10]}")
    data = req.model_dump(mode="python")
    data["knowledge_run_dir"] = run_dir
    row = create_job(
        db_path=_jobs_db(cfg),
        job_type=JobType.RECIPE_GENERATE,
        payload=data,
        run_dir=run_dir,
        max_retries=int(req.max_retry_loops),
    )
    return JobRef(job_id=row.job_id, status=row.status)


class _Worker:
    def __init__(self, cfg: ServiceConfig):
        self.cfg = cfg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            did = run_one_pending_job(self.cfg)
            if not did:
                time.sleep(float(self.cfg.worker_poll_sec))


def start_service(cfg: ServiceConfig) -> None:
    cfg.data_root().mkdir(parents=True, exist_ok=True)
    cfg.jobs_runs_root().mkdir(parents=True, exist_ok=True)
    _jobs_db(cfg)
    worker = _Worker(cfg)
    worker.start()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/v1/jobs/research-index":
                    req = _read_json(self)
                    job = _create_research_job(req, cfg)
                    _json_response(
                        self,
                        HTTPStatus.ACCEPTED,
                        {"job_id": job.job_id, "status": job.status.value},
                    )
                    return
                if self.path == "/v1/jobs/recipe-generate":
                    req = _read_json(self)
                    job = _create_recipe_job(req, cfg)
                    _json_response(
                        self,
                        HTTPStatus.ACCEPTED,
                        {"job_id": job.job_id, "status": job.status.value},
                    )
                    return
                if self.path.startswith("/v1/jobs/") and self.path.endswith("/retry"):
                    parts = self.path.strip("/").split("/")
                    if len(parts) != 4:
                        raise ValueError("invalid retry path")
                    job_id = parts[2]
                    retry_job = create_retry_job(db_path=_jobs_db(cfg), job_id=job_id)
                    _json_response(
                        self,
                        HTTPStatus.ACCEPTED,
                        {"job_id": retry_job.job_id, "status": retry_job.status.value},
                    )
                    return
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint: {self.path}"})
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path.startswith("/v1/jobs/") and self.path.endswith("/artifacts"):
                    parts = self.path.strip("/").split("/")
                    if len(parts) != 4:
                        raise ValueError("invalid artifacts path")
                    job_id = parts[2]
                    payload = get_job_artifacts(job_id, cfg)
                    _json_response(self, HTTPStatus.OK, {"job_id": job_id, "artifacts": payload})
                    return
                if self.path.startswith("/v1/jobs/"):
                    parts = self.path.strip("/").split("/")
                    if len(parts) != 3:
                        raise ValueError("invalid job path")
                    job_id = parts[2]
                    status = get_job_status(job_id, cfg)
                    payload = dump_job_payload(status, cfg)
                    _json_response(self, HTTPStatus.OK, payload)
                    return
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": f"unknown endpoint: {self.path}"})
            except FileNotFoundError as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((cfg.host, int(cfg.port)), Handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        worker.stop()
