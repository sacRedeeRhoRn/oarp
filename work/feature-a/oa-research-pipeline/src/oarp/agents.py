from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.knowledge import build_knowledge
from oarp.models import RunConfig
from oarp.pipeline import run_pipeline
from oarp.recipe import generate_recipes
from oarp.service_models import RecipeGenerateRequest, ResearchIndexRequest, SafetyPolicy


@dataclass
class JobEnvelope:
    job_id: str
    job_type: str
    run_dir: str
    payload: dict[str, Any]


@dataclass
class JobResult:
    status: str
    result: dict[str, Any]
    artifacts: list[dict[str, str]]


@dataclass
class KnowledgeBundleLite:
    run_dir: str
    phase_events_path: str
    condition_graph_path: str
    quality_outcomes_path: str


class ResearchAgent:
    def build_knowledge(self, req: ResearchIndexRequest) -> KnowledgeBundleLite:
        cfg = RunConfig(run_dir=req.run_dir)
        for key, value in req.cfg_overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        _ = run_pipeline(spec_path=req.spec_path, query=req.query, cfg=cfg)
        knowledge = build_knowledge(req.run_dir)
        return KnowledgeBundleLite(
            run_dir=str(Path(req.run_dir).expanduser().resolve()),
            phase_events_path=str(knowledge.phase_events_path),
            condition_graph_path=str(knowledge.condition_graph_path),
            quality_outcomes_path=str(knowledge.quality_outcomes_path),
        )


class RecipePlannerAgent:
    def propose(self, req: RecipeGenerateRequest, kb: KnowledgeBundleLite) -> list[dict[str, Any]]:
        phase_path = Path(kb.phase_events_path)
        if not phase_path.exists():
            return []
        frame = pd.read_parquet(phase_path)
        if frame.empty:
            return []
        out: list[dict[str, Any]] = []
        target = req.target_film_material.strip().lower()
        for row in frame.to_dict(orient="records"):
            phase = str(row.get("phase_label") or "").strip().lower()
            if target and target not in phase and target not in str(row.get("snippet") or "").lower():
                continue
            out.append(row)
            if len(out) >= req.max_candidates:
                break
        return out or frame.head(req.max_candidates).to_dict(orient="records")


class EvidenceVerifierAgent:
    def score(self, candidates: list[dict[str, Any]], kb: KnowledgeBundleLite) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in candidates:
            score = float(row.get("film_quality_score_numeric") or 0.0)
            confidence = max(0.0, min(1.0, 0.45 + 0.50 * score))
            item = dict(row)
            item["evidence_support_hint"] = confidence
            out.append(item)
        return out


class SafetyGateAgent:
    def filter(self, scored: list[dict[str, Any]], policy: SafetyPolicy) -> list[dict[str, Any]]:
        blocked_methods = {item.strip().upper() for item in policy.forbidden_methods if item.strip()}
        blocked_precursors = {item.strip().lower() for item in policy.forbidden_precursors if item.strip()}
        out: list[dict[str, Any]] = []
        for row in scored:
            method = str(row.get("method_family") or "").upper()
            precursor = str(row.get("precursor") or "").lower()
            if method and method in blocked_methods:
                continue
            if precursor and any(token in precursor for token in blocked_precursors):
                continue
            out.append(dict(row))
        return out


class ManagerAgent:
    def __init__(
        self,
        *,
        materials_project_enabled: bool = True,
        materials_project_api_key: str = "",
        materials_project_endpoint: str = "https://api.materialsproject.org",
        materials_project_scope: str = "summary_thermo",
    ) -> None:
        self.research_agent = ResearchAgent()
        self.recipe_planner = RecipePlannerAgent()
        self.evidence_verifier = EvidenceVerifierAgent()
        self.safety_gate = SafetyGateAgent()
        self.materials_project_enabled = bool(materials_project_enabled)
        self.materials_project_api_key = str(materials_project_api_key or "")
        self.materials_project_endpoint = str(materials_project_endpoint or "https://api.materialsproject.org")
        self.materials_project_scope = str(materials_project_scope or "summary_thermo")

    def run(self, job: JobEnvelope) -> JobResult:
        if job.job_type == "research-index":
            req = ResearchIndexRequest.model_validate(job.payload)
            kb = self.research_agent.build_knowledge(req)
            artifacts = [
                {"name": "phase_events", "path": kb.phase_events_path},
                {"name": "condition_graph", "path": kb.condition_graph_path},
                {"name": "quality_outcomes", "path": kb.quality_outcomes_path},
            ]
            return JobResult(
                status="SUCCEEDED",
                result={
                    "job_id": job.job_id,
                    "status": "SUCCEEDED",
                    "run_dir": kb.run_dir,
                    "knowledge": {
                        "phase_events_path": kb.phase_events_path,
                        "condition_graph_path": kb.condition_graph_path,
                        "quality_outcomes_path": kb.quality_outcomes_path,
                    },
                },
                artifacts=artifacts,
            )

        if job.job_type == "recipe-generate":
            req = RecipeGenerateRequest.model_validate(job.payload)
            run_dir = req.knowledge_run_dir.strip() or job.run_dir
            result, artifact_info = generate_recipes(
                run_dir=run_dir,
                request=req,
                materials_project_enabled=self.materials_project_enabled,
                materials_project_api_key=self.materials_project_api_key,
                materials_project_endpoint=self.materials_project_endpoint,
                mp_scope=self.materials_project_scope,
            )
            payload = result.model_dump(mode="python")
            payload["job_id"] = job.job_id
            artifacts = [
                {"name": "recipe_candidates", "path": str(artifact_info.candidates_path)},
                {"name": "recipe_ranked", "path": str(artifact_info.ranked_path)},
                {"name": "recipe_cards", "path": str(artifact_info.cards_path)},
                {"name": "gate_audit", "path": str(artifact_info.gate_audit_path)},
                {"name": "safety_audit", "path": str(artifact_info.safety_audit_path)},
                {
                    "name": "materials_project_refs",
                    "path": str(artifact_info.materials_project_refs_path),
                },
                {"name": "recipe_result", "path": str(artifact_info.result_path)},
            ]
            return JobResult(status="SUCCEEDED", result=payload, artifacts=artifacts)

        raise ValueError(f"unsupported job_type: {job.job_type}")
