from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from oarp.agents import JobEnvelope, ManagerAgent
from oarp.jobs import (
    append_job_artifact,
    claim_next_pending_job,
    complete_job,
    create_job,
    fail_job,
    get_job,
    get_job_result,
    init_jobs_db,
    list_job_artifacts,
)
from oarp.service_models import (
    JobRef,
    JobStatusPayload,
    JobType,
    RecipeGenerateRequest,
    RecipeResult,
    ResearchIndexRequest,
    ServiceConfig,
)
from oarp.topic_spec import ensure_query, load_topic_spec


def _jobs_db(cfg: ServiceConfig) -> Path:
    path = cfg.jobs_db()
    init_jobs_db(path)
    return path


def submit_research_index_job(
    spec_path: str,
    cfg: ServiceConfig,
    query: str | None = None,
    run_dir: str | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> JobRef:
    spec = load_topic_spec(spec_path)
    resolved_query = query.strip() if isinstance(query, str) and query.strip() else ensure_query(spec, None)
    resolved_run_dir = run_dir.strip() if isinstance(run_dir, str) and run_dir.strip() else str(
        cfg.jobs_runs_root() / f"research_{uuid.uuid4().hex[:10]}"
    )
    req = ResearchIndexRequest(
        spec_path=str(Path(spec_path).expanduser().resolve()),
        query=resolved_query,
        run_dir=str(resolved_run_dir),
        cfg_overrides=dict(cfg_overrides or {}),
    )
    job = create_job(
        db_path=_jobs_db(cfg),
        job_type=JobType.RESEARCH_INDEX,
        payload=req.model_dump(mode="python"),
        run_dir=str(resolved_run_dir),
        max_retries=0,
    )
    return JobRef(job_id=job.job_id, status=job.status)


def submit_recipe_generation_job(request: RecipeGenerateRequest, cfg: ServiceConfig) -> JobRef:
    run_dir = request.knowledge_run_dir.strip() or str(cfg.jobs_runs_root() / f"recipe_{uuid.uuid4().hex[:10]}")
    payload = request.model_dump(mode="python")
    payload["knowledge_run_dir"] = run_dir
    job = create_job(
        db_path=_jobs_db(cfg),
        job_type=JobType.RECIPE_GENERATE,
        payload=payload,
        run_dir=run_dir,
        max_retries=int(request.max_retry_loops),
    )
    return JobRef(job_id=job.job_id, status=job.status)


def get_job_status(job_id: str, cfg: ServiceConfig) -> JobStatusPayload:
    row = get_job(db_path=_jobs_db(cfg), job_id=job_id)
    if row is None:
        raise FileNotFoundError(f"job not found: {job_id}")
    return row


def load_recipe_result(job_id: str, cfg: ServiceConfig) -> RecipeResult:
    status = get_job_status(job_id, cfg)
    if status.job_type != JobType.RECIPE_GENERATE:
        raise ValueError(f"job is not recipe-generate: {job_id}")
    result = get_job_result(db_path=_jobs_db(cfg), job_id=job_id) or {}
    if not result:
        raise ValueError(f"recipe result not ready: {job_id} ({status.status.value})")
    return RecipeResult.model_validate(result)


def get_job_artifacts(job_id: str, cfg: ServiceConfig) -> list[dict[str, str]]:
    return list_job_artifacts(db_path=_jobs_db(cfg), job_id=job_id)


def run_one_pending_job(cfg: ServiceConfig) -> bool:
    db_path = _jobs_db(cfg)
    claimed = claim_next_pending_job(db_path=db_path)
    if claimed is None:
        return False
    job_id = str(claimed["job_id"])
    try:
        manager = ManagerAgent(
            materials_project_enabled=cfg.materials_project_enabled,
            materials_project_api_key=cfg.materials_project_api_key,
            materials_project_endpoint=cfg.materials_project_endpoint,
            materials_project_scope=cfg.materials_project_scope,
        )
        result = manager.run(
            JobEnvelope(
                job_id=job_id,
                job_type=str(claimed["job_type"]),
                run_dir=str(claimed["run_dir"]),
                payload=dict(claimed.get("payload") or {}),
            )
        )
        complete_job(db_path=db_path, job_id=job_id, result=result.result)
        for item in result.artifacts:
            name = str(item.get("name") or "").strip()
            path = str(item.get("path") or "").strip()
            if name and path:
                append_job_artifact(db_path=db_path, job_id=job_id, name=name, path=path)
        return True
    except Exception as exc:
        fail_job(db_path=db_path, job_id=job_id, error=f"{type(exc).__name__}: {exc}")
        return True


def run_pending_jobs(cfg: ServiceConfig, max_jobs: int = 100) -> int:
    ran = 0
    while ran < int(max_jobs):
        did = run_one_pending_job(cfg)
        if not did:
            break
        ran += 1
    return ran


def dump_job_payload(status: JobStatusPayload, cfg: ServiceConfig) -> dict[str, Any]:
    result = get_job_result(db_path=_jobs_db(cfg), job_id=status.job_id) or {}
    artifacts = get_job_artifacts(status.job_id, cfg)
    payload = {
        "job_id": status.job_id,
        "job_type": status.job_type.value,
        "status": status.status.value,
        "run_dir": status.run_dir,
        "retries": status.retries,
        "max_retries": status.max_retries,
        "created_at": status.created_at,
        "started_at": status.started_at,
        "finished_at": status.finished_at,
        "error": status.error,
        "artifacts": artifacts,
        "result": result,
    }
    return json.loads(json.dumps(payload, sort_keys=True))
