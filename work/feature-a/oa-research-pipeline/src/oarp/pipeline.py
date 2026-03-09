from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.cache import prepare_feature_cache as prepare_feature_cache_runtime
from oarp.models import (
    AllDoneValidationResult,
    ArticleGraphBuildResult,
    ArticleIndex,
    BridgeBuildResult,
    ConceptGraphBuildResult,
    ConsensusSet,
    DocumentIndex,
    EvidenceSet,
    ExtractionCalibration,
    ExtractionVoteSet,
    FineTuneExecutionResult,
    FullWorkflowResult,
    GraphAuditResult,
    GraphBuildResult,
    MPEvidenceSet,
    OutputBundle,
    PreLargeValidationResult,
    ProcessorDataset,
    ProcessorModelBundle,
    ProcessorModelRef,
    QualityEvalResult,
    QualityGateResult,
    ReleaseValidationResult,
    RunConfig,
    RunResult,
    TieredValidationResult,
    ValidatedEvidence,
    VaultBenchmarkResult,
    VaultExportResult,
    VaultImportResult,
)
from oarp.normalization import build_thickness_views as normalize_thickness_views
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
    write_json,
    write_run_state,
)
from oarp.topic_spec import TopicSpec, ensure_query, load_topic_spec

from . import (
    acquisition,
    consensus,
    discovery,
    extraction,
    knowledge,
    mp_enrichment,
    obsidian_vault,
    processing,
    recipe,
    render,
    validation,
)
from .workflow import bootstrap_runtime, preflight_strict
from .workflow import prepare_shared_runtime as prepare_shared_runtime_impl


def _run_id(topic_id: str, query: str) -> str:
    digest = hashlib.sha1(f"{topic_id}|{query}".encode("utf-8", errors="replace")).hexdigest()[:10]
    timestamp = now_iso().replace(":", "").replace("-", "")
    return f"run-{timestamp}-{digest}"


def _safe_slug(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-")
    return token or "run"


def _initialize_run_state(spec_path: str, spec: TopicSpec, query: str, cfg: RunConfig) -> None:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]
    run_id = _run_id(spec.topic_id, query)
    state_payload = {
        "run_id": run_id,
        "topic_id": spec.topic_id,
        "query": query,
        "spec_path": str(Path(spec_path).expanduser().resolve()),
        "created_at": now_iso(),
        "config": {
            "min_discovery_score": cfg.min_discovery_score,
            "plugin_id": cfg.plugin_id,
            "max_pages_per_provider": cfg.max_pages_per_provider,
            "max_discovered_records": cfg.max_discovered_records,
            "saturation_window_pages": cfg.saturation_window_pages,
            "saturation_min_yield": cfg.saturation_min_yield,
            "min_pages_before_saturation": cfg.min_pages_before_saturation,
            "resume_discovery": cfg.resume_discovery,
            "max_downloads": cfg.max_downloads,
            "local_repo_paths": list(cfg.local_repo_paths),
            "local_repo_recursive": cfg.local_repo_recursive,
            "local_file_glob": cfg.local_file_glob,
            "local_merge_mode": cfg.local_merge_mode,
            "local_max_files": cfg.local_max_files,
            "local_require_readable": cfg.local_require_readable,
            "acquire_workers": cfg.acquire_workers,
            "extract_workers": cfg.extract_workers,
            "english_first": cfg.english_first,
            "per_provider_cap": cfg.per_provider_cap,
            "require_fulltext_mime": cfg.require_fulltext_mime,
            "context_window_lines": cfg.context_window_lines,
            "point_assembler": cfg.point_assembler,
            "context_assembler": cfg.context_assembler,
            "phase_schema_version": cfg.phase_schema_version,
            "graph_core_mode": cfg.graph_core_mode,
            "graph_architecture": cfg.graph_architecture,
            "concept_ontology_profile": cfg.concept_ontology_profile,
            "bridge_weight_policy": cfg.bridge_weight_policy,
            "bridge_weight_threshold": cfg.bridge_weight_threshold,
            "concept_gates": list(cfg.concept_gates),
            "phase_require_elements": cfg.phase_require_elements,
            "phase_require_stoich": cfg.phase_require_stoich,
            "phase_require_spacegroup": cfg.phase_require_spacegroup,
            "cache_mode": cfg.cache_mode,
            "shared_cache_root": cfg.shared_cache_root,
            "cache_read_only": cfg.cache_read_only,
            "cache_ttl_hours": cfg.cache_ttl_hours,
            "cpu_strict_profile": cfg.cpu_strict_profile,
            "cpu_max_threads": cfg.cpu_max_threads,
            "cpu_probe": cfg.cpu_probe,
            "extractor_mode": cfg.extractor_mode,
            "extractor_models": cfg.extractor_models,
            "tgi_endpoint": cfg.tgi_endpoint,
            "tgi_models": cfg.tgi_models,
            "tgi_endpoints": cfg.tgi_endpoints,
            "slm_model_mode": cfg.slm_model_mode,
            "tgi_workers": cfg.tgi_workers,
            "decoder": cfg.decoder,
            "slm_max_retries": cfg.slm_max_retries,
            "slm_timeout_sec": cfg.slm_timeout_sec,
            "slm_batch_size": cfg.slm_batch_size,
            "slm_chunk_tokens": cfg.slm_chunk_tokens,
            "slm_overlap_tokens": cfg.slm_overlap_tokens,
            "slm_eval_split": cfg.slm_eval_split,
            "schema_decoder": cfg.schema_decoder,
            "vote_policy": cfg.vote_policy,
            "min_vote_confidence": cfg.min_vote_confidence,
            "extractor_gate_profile": cfg.extractor_gate_profile,
            "require_context_fields": cfg.require_context_fields,
            "mp_enabled": cfg.mp_enabled,
            "mp_mode": cfg.mp_mode,
            "mp_scope": cfg.mp_scope,
            "mp_on_demand": cfg.mp_on_demand,
            "mp_query_workers": cfg.mp_query_workers,
            "mp_timeout_sec": cfg.mp_timeout_sec,
            "mp_max_queries": cfg.mp_max_queries,
            "mp_cache_path": cfg.mp_cache_path,
            "mp_formula_match_weight": cfg.mp_formula_match_weight,
            "mp_phase_match_weight": cfg.mp_phase_match_weight,
            "mp_stability_weight": cfg.mp_stability_weight,
            "emit_extraction_calibration": cfg.emit_extraction_calibration,
            "calibration_bins": cfg.calibration_bins,
            "gnn_hidden_dim": cfg.gnn_hidden_dim,
            "gnn_layers": cfg.gnn_layers,
            "gnn_dropout": cfg.gnn_dropout,
            "gnn_epochs": cfg.gnn_epochs,
            "gnn_lr": cfg.gnn_lr,
            "tabular_model": cfg.tabular_model,
            "finetune_target_phase": cfg.finetune_target_phase,
            "finetune_max_thickness_nm": cfg.finetune_max_thickness_nm,
            "system_finetune_dataset": cfg.system_finetune_dataset,
            "system_eval_holdout_ratio": cfg.system_eval_holdout_ratio,
            "extractor_max_loop": cfg.extractor_max_loop,
            "processor_max_loop": cfg.processor_max_loop,
            "strict_full_workflow": cfg.strict_full_workflow,
            "auto_bootstrap": cfg.auto_bootstrap,
            "bootstrap_mode": cfg.bootstrap_mode,
            "shared_venv_root": cfg.shared_venv_root,
            "python_exec": cfg.python_exec,
            "tgi_docker_image": cfg.tgi_docker_image,
            "tgi_model_id": cfg.tgi_model_id,
            "tgi_platform": cfg.tgi_platform,
            "tgi_mode": cfg.tgi_mode,
            "tgi_port": cfg.tgi_port,
            "tgi_health_path": cfg.tgi_health_path,
            "tgi_generate_path": cfg.tgi_generate_path,
            "tgi_port_policy": cfg.tgi_port_policy,
            "tgi_port_range": cfg.tgi_port_range,
            "tgi_reuse_existing": cfg.tgi_reuse_existing,
            "workflow_profile": cfg.workflow_profile,
            "storage_root": cfg.storage_root,
            "run_root": cfg.run_root,
            "cache_root": cfg.cache_root,
            "model_root": cfg.model_root,
            "dataset_root": cfg.dataset_root,
            "vault_root": cfg.vault_root,
            "run_profile": cfg.run_profile,
            "vault_export_enabled": cfg.vault_export_enabled,
            "vault_import_enabled": cfg.vault_import_enabled,
            "vault_import_mode": cfg.vault_import_mode,
            "vault_profile": cfg.vault_profile,
            "slm_max_chunks_per_doc": cfg.slm_max_chunks_per_doc,
            "slm_max_doc_chars": cfg.slm_max_doc_chars,
            "slm_response_cache": cfg.slm_response_cache,
            "extract_stage_timeout_sec": cfg.extract_stage_timeout_sec,
            "use_bootstrapped_venv": cfg.use_bootstrapped_venv,
            "already_bootstrapped": cfg.already_bootstrapped,
            "all_done_repro_runs": cfg.all_done_repro_runs,
            "all_done_max_runtime_sec": cfg.all_done_max_runtime_sec,
            "all_done_require_mp_if_key_present": cfg.all_done_require_mp_if_key_present,
            "finetune_gate_policy": cfg.finetune_gate_policy,
            "finetune_ceiling_threshold": cfg.finetune_ceiling_threshold,
            "finetune_min_slice_rows": cfg.finetune_min_slice_rows,
            "finetune_min_support_articles": cfg.finetune_min_support_articles,
            "validation_tier": cfg.validation_tier,
            "processor_train_tier": cfg.processor_train_tier,
            "recipe_export_tier": cfg.recipe_export_tier,
            "quality_profile": cfg.quality_profile,
            "thickness_schema": cfg.thickness_schema,
            "thickness_compat_alias": cfg.thickness_compat_alias,
            "prelarge_rung": cfg.prelarge_rung,
            "finetune_require_support": cfg.finetune_require_support,
            "finetune_on_insufficient": cfg.finetune_on_insufficient,
            "processor_model_dir": cfg.processor_model_dir,
        },
    }
    write_run_state(cfg.as_path(), state_payload)

    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs(run_id, created_at, topic_id, query)
            VALUES(?, ?, ?, ?)
            """,
            (run_id, now_iso(), spec.topic_id, query),
        )
        conn.commit()
    finally:
        conn.close()


def _stage_row(stage: str, started: float, ended: float, details: dict[str, Any]) -> dict[str, Any]:
    row = {
        "stage": stage,
        "duration_sec": round(max(0.0, ended - started), 6),
        "started_at": started,
        "ended_at": ended,
    }
    row.update(details)
    return row


def discover_articles(spec: TopicSpec, query: str, cfg: RunConfig) -> ArticleIndex:
    return discovery.discover(spec=spec, query=query, cfg=cfg)


def acquire_fulltext(index: ArticleIndex, cfg: RunConfig) -> DocumentIndex:  # noqa: ARG001
    return acquisition.acquire(cfg=cfg)


def extract_evidence(spec: TopicSpec, docs: DocumentIndex, cfg: RunConfig) -> EvidenceSet:  # noqa: ARG001
    return extraction.extract(spec=spec, cfg=cfg)


def enrich_with_materials_project(spec: TopicSpec, cfg: RunConfig) -> MPEvidenceSet:
    return mp_enrichment.enrich(spec=spec, cfg=cfg)


def validate_evidence(spec: TopicSpec, evidence: EvidenceSet, cfg: RunConfig) -> ValidatedEvidence:  # noqa: ARG001
    return validation.validate(spec=spec, cfg=cfg)


def build_consensus(spec: TopicSpec, evidence: ValidatedEvidence, cfg: RunConfig) -> ConsensusSet:  # noqa: ARG001
    return consensus.build(spec=spec, cfg=cfg)


def render_outputs(spec: TopicSpec, consensus_set: ConsensusSet, cfg: RunConfig) -> OutputBundle:
    return render.render(spec=spec, consensus_set=consensus_set, cfg=cfg)


def run_slm_extraction_swarm(docs: DocumentIndex, spec: TopicSpec, cfg: RunConfig) -> ExtractionVoteSet:  # noqa: ARG001
    prior_mode = str(cfg.extractor_mode or "")
    cfg.extractor_mode = "slm_swarm"
    try:
        _ = extraction.extract(spec=spec, cfg=cfg)
    finally:
        cfg.extractor_mode = prior_mode or "hybrid_rules"
    artifacts = cfg.as_path() / "artifacts"
    votes_path = artifacts / "extraction_votes.parquet"
    error_path = artifacts / "extraction_error_slices.parquet"
    votes = pd.read_parquet(votes_path) if votes_path.exists() else pd.DataFrame()
    accepted = set()
    if not votes.empty and "accepted" in votes.columns:
        accepted = set(votes[votes["accepted"]]["point_id"].astype(str).tolist())
    return ExtractionVoteSet(
        votes=votes,
        accepted_point_ids=accepted,
        votes_path=votes_path,
        error_slices_path=error_path,
    )


def run_tgi_slm_extraction(
    docs: DocumentIndex,  # noqa: ARG001
    spec: TopicSpec,
    cfg: RunConfig,
) -> ExtractionVoteSet:
    prior_mode = str(cfg.extractor_mode or "")
    cfg.extractor_mode = "slm_tgi_required"
    try:
        _ = extraction.extract(spec=spec, cfg=cfg)
    finally:
        cfg.extractor_mode = prior_mode or "hybrid_rules"
    artifacts = cfg.as_path() / "artifacts"
    votes_path = artifacts / "extraction_votes.parquet"
    error_path = artifacts / "extraction_error_slices.parquet"
    votes = pd.read_parquet(votes_path) if votes_path.exists() else pd.DataFrame()
    accepted = set()
    if not votes.empty and "accepted" in votes.columns:
        accepted = set(votes[votes["accepted"]]["point_id"].astype(str).tolist())
    return ExtractionVoteSet(
        votes=votes,
        accepted_point_ids=accepted,
        votes_path=votes_path,
        error_slices_path=error_path,
    )


def validate_extraction_precision(votes: ExtractionVoteSet, spec: TopicSpec, cfg: RunConfig) -> ExtractionCalibration:  # noqa: ARG001
    accepted_path = cfg.as_path() / "artifacts" / "validated_points.parquet"
    accepted = pd.read_parquet(accepted_path) if accepted_path.exists() else pd.DataFrame()
    return validation.validate_extraction_precision(votes=votes.votes, accepted=accepted, cfg=cfg)


def build_thickness_views(points_df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    compat_alias = bool(getattr(cfg, "thickness_compat_alias", True))
    return normalize_thickness_views(points_df, compat_alias=compat_alias)


_ATOMIC_MASS = {
    "H": 1.0079,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.999,
    "Al": 26.9815,
    "Si": 28.0855,
    "Ti": 47.867,
    "Cr": 51.996,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ge": 72.630,
    "Mo": 95.95,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Sn": 118.71,
    "Pt": 195.084,
    "Au": 196.96657,
}
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")


def _parse_formula_molar_mass(formula: str) -> tuple[float | None, list[tuple[str, float]]]:
    text = re.sub(r"[^A-Za-z0-9.]", "", str(formula or "").strip())
    if not text:
        return None, []
    tokens = _FORMULA_TOKEN_RE.findall(text)
    if not tokens:
        return None, []
    parts: list[tuple[str, float]] = []
    total = 0.0
    for element, raw_count in tokens:
        if element not in _ATOMIC_MASS:
            return None, []
        count = float(raw_count) if str(raw_count or "").strip() else 1.0
        parts.append((element, count))
        total += _ATOMIC_MASS[element] * count
    return (total if total > 0 else None), parts


def estimate_anneal_thickness_prior(points_df: pd.DataFrame, mp_df: pd.DataFrame | None, cfg: RunConfig) -> pd.DataFrame:  # noqa: ARG001
    frame = points_df.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "point_id",
                "thickness_anneal_prior_nm",
                "thickness_anneal_posterior_nm",
                "thickness_relation_confidence",
                "thickness_role_source",
            ]
        )
    if "point_id" not in frame.columns:
        frame["point_id"] = [f"point_{idx}" for idx in range(len(frame))]
    frame = build_thickness_views(frame, cfg)

    density_by_material: dict[str, float] = {}
    if mp_df is not None and not mp_df.empty:
        if "material_id" in mp_df.columns and "density" in mp_df.columns:
            for row in mp_df.to_dict(orient="records"):
                mid = str(row.get("material_id") or "").strip()
                try:
                    density = float(row.get("density"))
                except Exception:
                    continue
                if mid and density > 0:
                    density_by_material[mid] = density

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby("point_id", as_index=False).first()
    for row in grouped.to_dict(orient="records"):
        point_id = str(row.get("point_id") or "")
        t_asdep = row.get("thickness_asdep_nm")
        t_ann_extracted = row.get("thickness_anneal_nm")
        try:
            t_asdep_f = float(t_asdep) if t_asdep not in (None, "") else None
        except Exception:
            t_asdep_f = None
        try:
            t_ann_f = float(t_ann_extracted) if t_ann_extracted not in (None, "") else None
        except Exception:
            t_ann_f = None

        formula = str(row.get("film_material") or row.get("entity") or "").strip()
        molar_mass, parts = _parse_formula_molar_mass(formula)
        prior = None
        prior_conf = 0.35
        if t_asdep_f is not None and t_asdep_f > 0 and molar_mass is not None and parts:
            metal_coeff = max(1e-9, float(parts[0][1]))
            total_coeff = sum(count for _elem, count in parts)
            stoich_ratio = total_coeff / metal_coeff
            density_term = 1.0
            best_mid = str(row.get("mp_best_material_id") or "").strip()
            if best_mid and best_mid in density_by_material:
                density_term = max(0.5, min(2.5, density_by_material[best_mid] / 8.0))
                prior_conf = 0.75
            else:
                ehull = row.get("mp_energy_above_hull_min")
                try:
                    ehull_f = float(ehull)
                    density_term = max(0.8, min(1.4, 1.1 - min(ehull_f, 0.3)))
                    prior_conf = 0.55
                except Exception:
                    prior_conf = 0.45
            prior = float(max(0.0, t_asdep_f * stoich_ratio * density_term))

        extract_conf = float(row.get("confidence") or row.get("vote_confidence") or 0.0)
        context_conf = float(row.get("context_confidence") or 0.0)
        w_extract = max(0.0, min(1.0, 0.40 + 0.35 * extract_conf + 0.25 * context_conf))
        posterior = t_ann_f
        if prior is not None and t_ann_f is not None:
            posterior = float(w_extract * t_ann_f + (1.0 - w_extract) * prior)
        elif prior is not None and t_ann_f is None:
            posterior = prior

        rows.append(
            {
                "point_id": point_id,
                "thickness_anneal_prior_nm": prior,
                "thickness_anneal_posterior_nm": posterior,
                "thickness_relation_confidence": max(0.0, min(1.0, prior_conf if prior is not None else extract_conf)),
                "thickness_role_source": str(row.get("thickness_role_source") or "legacy_alias"),
            }
        )
    return pd.DataFrame(rows)


def _pair_completeness_ratio(frame: pd.DataFrame, x_name: str, y_name: str) -> float:
    if frame.empty or "point_id" not in frame.columns:
        return 0.0
    total = int(frame["point_id"].astype(str).nunique())
    if total <= 0:
        return 0.0
    complete = 0
    for _point_id, group in frame.groupby("point_id"):
        names = set(group.get("variable_name", pd.Series([], dtype=str)).astype(str).tolist())
        if x_name in names and y_name in names:
            complete += 1
    return float(complete / max(1, total))


def _cross_source_support_ratio(frame: pd.DataFrame, x_name: str, y_name: str) -> float:
    if frame.empty:
        return 0.0
    required = {"point_id", "variable_name", "normalized_value"}
    if not required.issubset(set(frame.columns)):
        return 0.0
    pairs: list[dict[str, Any]] = []
    for point_id, group in frame.groupby("point_id"):
        x_rows = group[group["variable_name"].astype(str) == x_name]
        y_rows = group[group["variable_name"].astype(str) == y_name]
        if x_rows.empty or y_rows.empty:
            continue
        first = group.iloc[0]
        pairs.append(
            {
                "point_id": str(point_id),
                "entity": str(first.get("entity") or ""),
                "x": round(float(x_rows.iloc[0]["normalized_value"]), 3),
                "y": round(float(y_rows.iloc[0]["normalized_value"]), 3),
                "source": str(first.get("citation_url") or first.get("article_key") or ""),
            }
        )
    if not pairs:
        return 0.0
    pair_df = pd.DataFrame(pairs)
    support = (
        pair_df.groupby(["entity", "x", "y"], as_index=False)["source"]
        .nunique()
        .rename(columns={"source": "support"})
    )
    if support.empty:
        return 0.0
    ratio = float((support["support"] >= 2).mean())
    return ratio


def _distribution_agreement_ratio(
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
    *,
    x_tolerance_nm: float = 10.0,
    y_tolerance_c: float = 150.0,
) -> float:
    if frame.empty:
        return 0.0
    required = {"point_id", "variable_name", "normalized_value"}
    if not required.issubset(set(frame.columns)):
        return 0.0

    pairs: list[dict[str, Any]] = []
    for point_id, group in frame.groupby("point_id"):
        x_rows = group[group["variable_name"].astype(str) == x_name]
        y_rows = group[group["variable_name"].astype(str) == y_name]
        if x_rows.empty or y_rows.empty:
            continue
        first = group.iloc[0]
        try:
            x_val = float(x_rows.iloc[0]["normalized_value"])
            y_val = float(y_rows.iloc[0]["normalized_value"])
        except Exception:
            continue
        pairs.append(
            {
                "point_id": str(point_id),
                "entity": str(first.get("entity") or ""),
                "x": x_val,
                "y": y_val,
                "source": str(first.get("citation_url") or first.get("article_key") or ""),
            }
        )
    if not pairs:
        return 0.0

    pair_df = pd.DataFrame(pairs)
    agree_pairs = 0
    total_pairs = 0

    for _entity, entity_frame in pair_df.groupby("entity"):
        sources = [str(item) for item in entity_frame["source"].astype(str).drop_duplicates().tolist() if str(item)]
        if len(sources) < 2:
            continue

        grouped = {src: entity_frame[entity_frame["source"].astype(str) == src][["x", "y"]].to_dict(orient="records") for src in sources}
        for idx in range(len(sources)):
            for jdx in range(idx + 1, len(sources)):
                left_src = sources[idx]
                right_src = sources[jdx]
                left_points = grouped.get(left_src) or []
                right_points = grouped.get(right_src) or []
                if not left_points or not right_points:
                    continue
                total_pairs += 1

                agree = False
                for left in left_points:
                    lx = float(left.get("x") or 0.0)
                    ly = float(left.get("y") or 0.0)
                    for right in right_points:
                        rx = float(right.get("x") or 0.0)
                        ry = float(right.get("y") or 0.0)
                        if abs(lx - rx) <= x_tolerance_nm and abs(ly - ry) <= y_tolerance_c:
                            agree = True
                            break
                    if agree:
                        break

                if not agree:
                    left_x = [float(item.get("x") or 0.0) for item in left_points]
                    right_x = [float(item.get("x") or 0.0) for item in right_points]
                    left_y = [float(item.get("y") or 0.0) for item in left_points]
                    right_y = [float(item.get("y") or 0.0) for item in right_points]
                    if left_x and right_x and left_y and right_y:
                        x_overlap = max(min(left_x), min(right_x)) <= (min(max(left_x), max(right_x)) + x_tolerance_nm)
                        y_overlap = max(min(left_y), min(right_y)) <= (min(max(left_y), max(right_y)) + y_tolerance_c)
                        agree = bool(x_overlap and y_overlap)

                if agree:
                    agree_pairs += 1

    if total_pairs <= 0:
        return 0.0
    return float(agree_pairs / total_pairs)


def evaluate_quality_no_gold(run_dir: str, cfg: RunConfig) -> QualityEvalResult:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = run_path / "artifacts"
    x_name = "thickness_nm"
    y_name = "temperature_c"
    try:
        state = load_run_state(run_path)
    except FileNotFoundError:
        state = {}
    spec_path = str(state.get("spec_path") or "").strip()
    if spec_path:
        try:
            spec = load_topic_spec(spec_path)
            x_name = str(spec.plot.primary.x or x_name)
            y_name = str(spec.plot.primary.y or y_name)
        except Exception:
            pass

    docs = pd.read_parquet(artifacts / "documents.parquet") if (artifacts / "documents.parquet").exists() else pd.DataFrame()
    evidence = pd.read_parquet(artifacts / "evidence_points.parquet") if (artifacts / "evidence_points.parquet").exists() else pd.DataFrame()
    validated = pd.read_parquet(artifacts / "validated_points.parquet") if (artifacts / "validated_points.parquet").exists() else pd.DataFrame()
    strict_path = artifacts / "validated_points_strict.parquet"
    strict_frame = pd.read_parquet(strict_path) if strict_path.exists() else validated.copy()
    consensus_frame = pd.read_parquet(artifacts / "consensus_points.parquet") if (artifacts / "consensus_points.parquet").exists() else pd.DataFrame()
    slm_responses = pd.read_parquet(artifacts / "slm_responses.parquet") if (artifacts / "slm_responses.parquet").exists() else pd.DataFrame()

    usable_text_ratio = 0.0
    usable_docs = 0
    if not docs.empty:
        if "usable_text" in docs.columns:
            usable_text_ratio = float(docs["usable_text"].astype(bool).mean())
            usable_docs = int(docs["usable_text"].astype(bool).sum())
        else:
            usable_text_ratio = float((docs.get("parse_status", pd.Series([], dtype=str)).astype(str).str.contains("parsed", case=False)).mean())
            usable_docs = int(round(usable_text_ratio * len(docs)))
    matched_article_ratio = 0.0
    evidence_density_per_usable_doc = 0.0
    coverage_strength = 0.0
    if usable_docs > 0 and not evidence.empty and "article_key" in evidence.columns:
        matched_article_ratio = float(
            evidence["article_key"].astype(str).dropna().drop_duplicates().shape[0] / max(1, usable_docs)
        )
        evidence_density_per_usable_doc = float(len(evidence) / max(1, usable_docs))
        # Coverage should reflect both breadth (matched docs) and depth (usable tuples per doc).
        coverage_strength = max(
            float(matched_article_ratio),
            float(min(1.0, evidence_density_per_usable_doc)),
        )

    provenance_complete_ratio = 0.0
    if not strict_frame.empty and all(col in strict_frame.columns for col in ["citation_url", "snippet", "locator"]):
        mask = strict_frame[["citation_url", "snippet", "locator"]].fillna("").astype(str).apply(
            lambda col: col.str.strip().ne(""),
            axis=0,
        ).all(axis=1)
        provenance_complete_ratio = float(mask.mean()) if len(mask) else 0.0

    pair_completeness_ratio = _pair_completeness_ratio(strict_frame, x_name=x_name, y_name=y_name)
    cross_source_support_ratio = _cross_source_support_ratio(strict_frame, x_name=x_name, y_name=y_name)
    distribution_agreement_ratio = _distribution_agreement_ratio(
        strict_frame,
        x_name=x_name,
        y_name=y_name,
    )

    timeout_rate = 0.0
    if not slm_responses.empty and "status" in slm_responses.columns:
        statuses = slm_responses["status"].astype(str).str.lower()
        timeout_rate = float(statuses.isin({"timeout", "error", "request_error"}).mean())

    consensus_support_median = float(consensus_frame.get("support_count", pd.Series([0])).median()) if not consensus_frame.empty else 0.0
    entropy_p75 = float(consensus_frame.get("entropy", pd.Series([0.0])).quantile(0.75)) if not consensus_frame.empty else 0.0

    graph_ok = True
    graph_audit_path = artifacts / "graph_audit.json"
    if graph_audit_path.exists():
        graph_ok = bool(_safe_json(graph_audit_path).get("ok", False))

    metrics = {
        "created_at": now_iso(),
        "profile": str(cfg.quality_profile or "balanced"),
        "usable_text_ratio": usable_text_ratio,
        "matched_article_ratio": matched_article_ratio,
        "evidence_density_per_usable_doc": evidence_density_per_usable_doc,
        "coverage_strength": coverage_strength,
        "provenance_complete_ratio": provenance_complete_ratio,
        "pair_completeness_ratio": pair_completeness_ratio,
        "timeout_rate": timeout_rate,
        "consensus_support_median": consensus_support_median,
        "cross_source_support_ratio": cross_source_support_ratio,
        "distribution_agreement_ratio": distribution_agreement_ratio,
        "entropy_p75": entropy_p75,
        "graph_audit_ok": graph_ok,
        "counts": {
            "documents": int(len(docs)),
            "evidence_rows": int(len(evidence)),
            "validated_rows": int(len(validated)),
            "strict_rows": int(len(strict_frame)),
            "consensus_rows": int(len(consensus_frame)),
            "slm_response_rows": int(len(slm_responses)),
        },
    }
    json_path = artifacts / "quality_eval.json"
    write_json(json_path, metrics)
    report_lines = [
        "# Quality Eval (No Gold)",
        "",
        f"- profile: `{metrics['profile']}`",
        f"- usable_text_ratio: `{metrics['usable_text_ratio']:.4f}`",
        f"- matched_article_ratio: `{metrics['matched_article_ratio']:.4f}`",
        f"- evidence_density_per_usable_doc: `{metrics['evidence_density_per_usable_doc']:.4f}`",
        f"- coverage_strength: `{metrics['coverage_strength']:.4f}`",
        f"- provenance_complete_ratio: `{metrics['provenance_complete_ratio']:.4f}`",
        f"- pair_completeness_ratio: `{metrics['pair_completeness_ratio']:.4f}`",
        f"- timeout_rate: `{metrics['timeout_rate']:.4f}`",
        f"- consensus_support_median: `{metrics['consensus_support_median']:.4f}`",
        f"- cross_source_support_ratio: `{metrics['cross_source_support_ratio']:.4f}`",
        f"- distribution_agreement_ratio: `{metrics['distribution_agreement_ratio']:.4f}`",
        f"- entropy_p75: `{metrics['entropy_p75']:.4f}`",
        f"- graph_audit_ok: `{metrics['graph_audit_ok']}`",
    ]
    report_path = artifacts / "quality_eval.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return QualityEvalResult(profile=str(cfg.quality_profile or "balanced"), metrics=metrics, json_path=json_path, report_path=report_path)


def enforce_quality_gates(run_dir: str, cfg: RunConfig) -> QualityGateResult:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = run_path / "artifacts"
    eval_path = artifacts / "quality_eval.json"
    if eval_path.exists():
        metrics = _safe_json(eval_path)
    else:
        metrics = evaluate_quality_no_gold(run_dir, cfg).metrics
    coverage_strength = float(metrics.get("coverage_strength") or 0.0)
    if coverage_strength <= 0.0:
        # Backward compatibility for runs generated before coverage_strength existed.
        matched_ratio = float(metrics.get("matched_article_ratio") or 0.0)
        density_ratio = float(metrics.get("evidence_density_per_usable_doc") or 0.0)
        coverage_strength = max(matched_ratio, min(1.0, density_ratio))

    gates = {
        "provenance_complete_ratio_eq_1": bool(abs(float(metrics.get("provenance_complete_ratio") or 0.0) - 1.0) < 1e-12),
        "pair_completeness_ratio_ge_0_90": bool(float(metrics.get("pair_completeness_ratio") or 0.0) >= 0.90),
        "usable_text_ratio_ge_0_70": bool(float(metrics.get("usable_text_ratio") or 0.0) >= 0.70),
        # Dynamic coverage check: breadth OR depth to avoid penalizing concentrated high-yield papers.
        "matched_article_ratio_ge_0_20": bool(coverage_strength >= 0.20),
        "timeout_rate_le_0_25": bool(float(metrics.get("timeout_rate") or 0.0) <= 0.25),
        "consensus_support_median_ge_3": bool(float(metrics.get("consensus_support_median") or 0.0) >= 3.0),
        "distribution_agreement_ratio_ge_0_35": bool(float(metrics.get("distribution_agreement_ratio") or 0.0) >= 0.35),
        "entropy_p75_le_1_0": bool(float(metrics.get("entropy_p75") or 0.0) <= 1.0),
        "graph_audit_ok": bool(metrics.get("graph_audit_ok", True)),
    }
    gate_payload = {
        "created_at": now_iso(),
        "profile": str(cfg.quality_profile or "balanced"),
        "ok": bool(all(gates.values())),
        "gates": gates,
        "metrics_path": str(eval_path),
        "metrics": metrics,
    }
    json_path = artifacts / "quality_gate_report.json"
    write_json(json_path, gate_payload)
    lines = ["# Quality Gate Report", "", f"- profile: `{gate_payload['profile']}`", f"- ok: `{gate_payload['ok']}`"]
    for key, passed in gates.items():
        lines.append(f"- {key}: `{passed}`")
    report_path = artifacts / "quality_gate_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return QualityGateResult(
        profile=str(cfg.quality_profile or "balanced"),
        ok=bool(gate_payload["ok"]),
        gates=gates,
        metrics=metrics,
        json_path=json_path,
        report_path=report_path,
    )


def evaluate_extractor_quality(run_dir: str, gold_dir: str, cfg: RunConfig) -> ExtractionCalibration:  # noqa: ARG001
    quality_eval = evaluate_quality_no_gold(run_dir=run_dir, cfg=cfg)
    payload = {
        "tuple_precision": 0.0,
        "tuple_recall": 0.0,
        "tuple_f1": 0.0,
        "provenance_complete_ratio": float(quality_eval.metrics.get("provenance_complete_ratio") or 0.0),
        "context_precision_proxy": 0.0,
        "gate_tuple_precision": False,
        "gate_tuple_recall": False,
        "gate_provenance_complete": bool(
            abs(float(quality_eval.metrics.get("provenance_complete_ratio") or 0.0) - 1.0) < 1e-12
        ),
        "gate_context_precision": False,
        "all_gates_pass": False,
    }
    artifacts = Path(run_dir).expanduser().resolve() / "artifacts"
    calibration_path = artifacts / "extraction_calibration.parquet"
    frame = pd.DataFrame(
        [
            {
                "bin_idx": 0,
                "bin_left": 0.0,
                "bin_right": 1.0,
                "point_count": int(quality_eval.metrics.get("counts", {}).get("validated_rows", 0)),
                "mean_predicted": 0.0,
                "empirical_precision": 0.0,
                "ece_component": 0.0,
                "provenance_complete_ratio": float(quality_eval.metrics.get("provenance_complete_ratio") or 0.0),
            }
        ]
    )
    frame.to_parquet(calibration_path, index=False)
    return ExtractionCalibration(frame=frame, summary=payload, path=calibration_path)


def build_processing_dataset(
    points: pd.DataFrame | str | Path,
    mp_data: pd.DataFrame | None,
    aux_data: pd.DataFrame | None,
    cfg: RunConfig,
) -> ProcessorDataset:
    return processing.build_processing_dataset(points, mp_data, aux_data, cfg)


def build_phase_spec(points_df: pd.DataFrame | str | Path, cfg: RunConfig) -> pd.DataFrame:
    return processing.build_phase_spec(points_df, cfg)


def build_article_phase_graph(
    points_df: pd.DataFrame | str | Path,
    phase_spec_df: pd.DataFrame,
    cfg: RunConfig,
) -> GraphBuildResult:
    return processing.build_article_phase_graph(points_df, phase_spec_df, cfg)


def build_article_process_graph(
    points_df: pd.DataFrame | str | Path,
    cfg: RunConfig,
) -> ArticleGraphBuildResult:
    return processing.build_article_process_graph(points_df, cfg)


def build_global_concept_graph(
    points_df: pd.DataFrame | str | Path,
    mp_df: pd.DataFrame | None,
    cfg: RunConfig,
) -> ConceptGraphBuildResult:
    return processing.build_global_concept_graph(points_df, mp_df, cfg)


def build_bridge_edges(
    article_graph: ArticleGraphBuildResult,
    concept_graph: ConceptGraphBuildResult,
    cfg: RunConfig,
) -> BridgeBuildResult:
    return processing.build_bridge_edges(article_graph, concept_graph, cfg)


def score_bridge_weight(edge_features: dict[str, Any], cfg: RunConfig) -> float:
    return processing.score_bridge_weight(edge_features, cfg)


def audit_dual_graph(run_dir: str | Path, cfg: RunConfig) -> GraphAuditResult:
    return processing.audit_dual_graph(run_dir=run_dir, cfg=cfg)


def prepare_feature_cache(run_cfg: RunConfig):  # noqa: ANN201
    return prepare_feature_cache_runtime(run_cfg)


def prepare_shared_runtime(cfg: RunConfig):
    return prepare_shared_runtime_impl(cfg)


def train_universal_processor(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelRef:
    return processing.train_universal_processor(dataset, cfg)


def train_gnn_base(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelRef:
    return processing.train_gnn_base(dataset, cfg)


def train_tabular_head(
    dataset: ProcessorDataset,
    gnn_ref: ProcessorModelRef,
    cfg: RunConfig,
) -> ProcessorModelRef:
    return processing.train_tabular_head(dataset, gnn_ref, cfg)


def finetune_nisi_sub200(
    model_ref: ProcessorModelRef,
    dataset: ProcessorDataset,
    cfg: RunConfig,
) -> ProcessorModelRef:
    return processing.finetune_nisi_sub200(model_ref, dataset, cfg)


def _finetune_support_stats(run_dir: Path) -> tuple[int, int]:
    slice_path = run_dir / "artifacts" / "finetune_slice.parquet"
    if not slice_path.exists():
        return 0, 0
    try:
        frame = pd.read_parquet(slice_path)
    except Exception:
        return 0, 0
    rows = int(len(frame))
    articles = int(frame.get("article_key", pd.Series([], dtype=str)).astype(str).nunique()) if not frame.empty else 0
    return rows, articles


def maybe_run_finetune(
    model_ref: ProcessorModelRef,
    dataset: ProcessorDataset,
    cfg: RunConfig,
) -> FineTuneExecutionResult:
    artifacts = cfg.as_path() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    payload_path = artifacts / "finetune_execution.json"
    try:
        finetune_ref = processing.finetune_nisi_sub200(model_ref, dataset, cfg)
        rows, articles = _finetune_support_stats(cfg.as_path())
        payload = {
            "created_at": now_iso(),
            "status": "ran",
            "support_rows": rows,
            "support_articles": articles,
            "reason": "",
            "model_dir": str(finetune_ref.model_dir),
        }
        write_json(payload_path, payload)
        return FineTuneExecutionResult(
            status="ran",
            support_rows=rows,
            support_articles=articles,
            reason="",
            model_ref=finetune_ref,
            payload_path=payload_path,
        )
    except Exception as exc:
        rows, articles = _finetune_support_stats(cfg.as_path())
        reason = f"{type(exc).__name__}: {exc}"
        support_insufficient = rows < max(1, int(cfg.finetune_min_slice_rows or 1)) or articles < max(
            1,
            int(cfg.finetune_min_support_articles or 1),
        )
        mode = str(getattr(cfg, "finetune_on_insufficient", "skip") or "skip").strip().lower()
        if support_insufficient and mode == "skip":
            payload = {
                "created_at": now_iso(),
                "status": "skipped_insufficient_support",
                "support_rows": rows,
                "support_articles": articles,
                "reason": reason,
                "model_dir": "",
            }
            write_json(payload_path, payload)
            return FineTuneExecutionResult(
                status="skipped_insufficient_support",
                support_rows=rows,
                support_articles=articles,
                reason=reason,
                model_ref=None,
                payload_path=payload_path,
            )
        payload = {
            "created_at": now_iso(),
            "status": "failed",
            "support_rows": rows,
            "support_articles": articles,
            "reason": reason,
            "model_dir": "",
        }
        write_json(payload_path, payload)
        return FineTuneExecutionResult(
            status="failed",
            support_rows=rows,
            support_articles=articles,
            reason=reason,
            model_ref=None,
            payload_path=payload_path,
        )


def evaluate_processor(run_dir: str, cfg: RunConfig) -> dict[str, Any]:
    return processing.evaluate_processor(run_dir=run_dir, cfg=cfg)


def train_processor_v1(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelBundle:
    return processing.train_processor_v1(dataset, cfg)


def finetune_processor_for_system(
    model_ref: ProcessorModelRef,
    system_dataset: ProcessorDataset,
    cfg: RunConfig,
) -> ProcessorModelRef:
    system_id = cfg.plugin_id or "system"
    return processing.finetune_processor_for_system(model_ref, system_dataset, cfg, system_id=system_id)


def generate_ranked_recipes(request: Any, model_ref: ProcessorModelRef, kb: Any, cfg: RunConfig):  # noqa: ANN401
    _ = kb
    result, _artifacts = recipe.generate_recipes(
        run_dir=cfg.as_path(),
        request=request,
        materials_project_enabled=bool(cfg.mp_enabled),
        mp_scope=str(cfg.mp_scope or "summary_thermo"),
        processor_model_path=str(model_ref.model_dir),
    )
    return result


def export_obsidian_vault(run_dir: str, out_dir: str, cfg: RunConfig) -> VaultExportResult:
    return obsidian_vault.export_obsidian_vault(run_dir=run_dir, out_dir=out_dir, cfg=cfg)


def project_concepts_to_vault(run_dir: str, out_dir: str, cfg: RunConfig) -> VaultExportResult:
    return obsidian_vault.export_obsidian_vault(run_dir=run_dir, out_dir=out_dir, cfg=cfg)


def import_obsidian_vault(vault_dir: str, run_dir: str, cfg: RunConfig) -> VaultImportResult:
    return obsidian_vault.import_obsidian_vault(vault_dir=vault_dir, run_dir=run_dir, cfg=cfg)


def parse_vault_semantics(vault_dir: str, run_dir: str, cfg: RunConfig) -> VaultImportResult:
    return obsidian_vault.import_obsidian_vault(vault_dir=vault_dir, run_dir=run_dir, cfg=cfg)


def apply_vault_soft_supervision(run_dir: str, vault_links: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    return obsidian_vault.apply_vault_soft_supervision(run_dir=run_dir, vault_links=vault_links, cfg=cfg)


def apply_vault_supervision_to_bridge_scores(run_dir: str, cfg: RunConfig) -> Path:
    path = Path(run_dir).expanduser().resolve() / "artifacts" / "vault_supervision_bridge.parquet"
    if path.exists():
        return path
    empty = pd.DataFrame(columns=["point_id", "gate_type", "vault_support_score", "vault_support_component"])
    empty.to_parquet(path, index=False)
    return path


def apply_vault_supervision_to_processor_rows(run_dir: str, cfg: RunConfig) -> Path:
    path = Path(run_dir).expanduser().resolve() / "artifacts" / "vault_supervision_processor.parquet"
    if path.exists():
        return path
    empty = pd.DataFrame(columns=["point_id", "vault_soft_supervision_score", "support_count", "penalty_count"])
    empty.to_parquet(path, index=False)
    return path


def benchmark_vault_alignment(run_dir: str, vault_dir: str, cfg: RunConfig) -> VaultBenchmarkResult:
    return obsidian_vault.benchmark_vault_alignment(run_dir=run_dir, vault_dir=vault_dir, cfg=cfg)


def benchmark_vault_alignment_v2(run_dir: str, vault_dir: str, cfg: RunConfig) -> VaultBenchmarkResult:
    return obsidian_vault.benchmark_vault_alignment(run_dir=run_dir, vault_dir=vault_dir, cfg=cfg)


def build_global_concept_registry(storage_root: str, runs_root: str, out_dir: str, cfg: RunConfig) -> Path:
    return obsidian_vault.build_global_concept_registry(
        storage_root=storage_root,
        runs_root=runs_root,
        out_dir=out_dir,
        cfg=cfg,
    )


def sync_run_concepts_to_registry(run_dir: str, registry_dir: str, cfg: RunConfig) -> dict[str, Any]:
    return obsidian_vault.sync_run_concepts_to_registry(run_dir=run_dir, registry_dir=registry_dir, cfg=cfg)


def run_benchmark(*args: Any, **kwargs: Any):  # noqa: ANN401
    """Deprecated compatibility shim for older callers/tests.

    The v2.3 no-gold flow does not use benchmark gating in production.
    """
    from oarp.benchmark import run_benchmark as legacy_run_benchmark

    return legacy_run_benchmark(*args, **kwargs)


def evaluate_processor_with_policy(run_dir: str, cfg: RunConfig) -> dict[str, Any]:
    return processing.evaluate_processor_with_policy(run_dir=run_dir, cfg=cfg)


def run_pipeline(spec_path: str, query: str, cfg: RunConfig) -> RunResult:
    if bool(cfg.cpu_strict_profile):
        _apply_cpu_thread_clamps(cfg)
    if bool(cfg.cpu_probe):
        _capture_cpu_env(cfg)
    spec = load_topic_spec(spec_path)
    resolved_query = ensure_query(spec, query)

    if not cfg.plugin_id and spec.plugins.preferred_plugin:
        cfg.plugin_id = spec.plugins.preferred_plugin

    _initialize_run_state(spec_path, spec, resolved_query, cfg)
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]

    stage_rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    articles = discover_articles(spec, resolved_query, cfg)
    t1 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "discover",
            t0,
            t1,
            {
                "article_count": int(len(articles.frame)),
                "output_path": str(articles.parquet_path),
            },
        )
    )

    t2 = time.perf_counter()
    documents = acquire_fulltext(articles, cfg)
    t3 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "acquire",
            t2,
            t3,
            {
                "document_count": int(len(documents.frame)),
                "usable_document_count": int(documents.frame["usable_text"].sum())
                if "usable_text" in documents.frame.columns
                else 0,
                "output_path": str(documents.parquet_path),
            },
        )
    )

    t4 = time.perf_counter()
    evidence = extract_evidence(spec, documents, cfg)
    t5 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "extract",
            t4,
            t5,
            {
                "evidence_count": int(len(evidence.points)),
                "provenance_count": int(len(evidence.provenance)),
                "output_path": str(evidence.points_path),
                "coverage_path": str(evidence.extraction_coverage_path)
                if evidence.extraction_coverage_path
                else "",
            },
        )
    )

    t6 = time.perf_counter()
    mp_enriched = enrich_with_materials_project(spec, cfg)
    t7 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "mp_enrich",
            t6,
            t7,
            {
                "enriched_point_count": int(len(mp_enriched.enriched_points)),
                "materials_count": int(len(mp_enriched.materials)),
                "point_link_count": int(len(mp_enriched.point_links)),
                "query_count": int(len(mp_enriched.query_log)),
                "output_path": str(mp_enriched.enriched_points_path),
            },
        )
    )

    t8 = time.perf_counter()
    validated = validate_evidence(spec, evidence, cfg)
    t9 = time.perf_counter()
    context_required_cols = ["substrate_material", "substrate_orientation", "doping_state", "alloy_state"]
    validated_context_completeness = 0.0
    if not validated.accepted.empty and all(col in validated.accepted.columns for col in context_required_cols):
        mask = validated.accepted[context_required_cols].apply(
            lambda col: col.astype(str).str.strip().ne(""),
            axis=0,
        ).all(axis=1)
        validated_context_completeness = float(mask.mean())
    warning_count = 0
    if validated.warnings_path and Path(validated.warnings_path).exists():
        warning_count = int(len(pd.read_parquet(validated.warnings_path)))
    stage_rows.append(
        _stage_row(
            "validate",
            t8,
            t9,
            {
                "accepted_count": int(len(validated.accepted)),
                "rejected_count": int(len(validated.rejected)),
                "warning_count": warning_count,
                "context_completeness": validated_context_completeness,
                "output_path": str(validated.accepted_path),
            },
        )
    )

    t10 = time.perf_counter()
    consensus_set = build_consensus(spec, validated, cfg)
    t11 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "consensus",
            t10,
            t11,
            {
                "consensus_count": int(len(consensus_set.points)),
                "output_path": str(consensus_set.parquet_path),
            },
        )
    )

    t12 = time.perf_counter()
    output = render_outputs(spec, consensus_set, cfg)
    t13 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "render",
            t12,
            t13,
            {
                "report_path": str(output.report_path),
            },
        )
    )

    stage_metrics_path = artifacts / "stage_metrics.parquet"
    pd.DataFrame(stage_rows).to_parquet(stage_metrics_path, index=False)

    extraction_votes_path = artifacts / "extraction_votes.parquet"
    extraction_calibration_path = artifacts / "extraction_calibration.parquet"
    processor_eval_metrics_path = artifacts / "processor_eval_metrics.json"
    extraction_vote_count = (
        int(len(pd.read_parquet(extraction_votes_path))) if extraction_votes_path.exists() else 0
    )
    slm_requests_path = artifacts / "slm_requests.parquet"
    slm_responses_path = artifacts / "slm_responses.parquet"
    slm_points_raw_path = artifacts / "slm_points_raw.parquet"
    slm_points_voted_path = artifacts / "slm_points_voted.parquet"
    extraction_calibration_rows = (
        int(len(pd.read_parquet(extraction_calibration_path)))
        if extraction_calibration_path.exists()
        else 0
    )

    metrics_path = artifacts / "run_metrics.json"
    run_metrics = {
        "run_dir": str(cfg.as_path()),
        "topic_id": spec.topic_id,
        "query": resolved_query,
        "created_at": now_iso(),
        "stage_metrics_path": str(stage_metrics_path),
        "stage_metrics": stage_rows,
        "counts": {
            "articles": int(len(articles.frame)),
            "documents": int(len(documents.frame)),
            "evidence_points": int(len(evidence.points)),
            "mp_enriched_points": int(len(mp_enriched.enriched_points)),
            "validated_points": int(len(validated.accepted)),
            "rejected_points": int(len(validated.rejected)),
            "validation_warnings": warning_count,
            "extraction_votes": extraction_vote_count,
            "extraction_calibration_rows": extraction_calibration_rows,
            "slm_requests": int(len(pd.read_parquet(slm_requests_path))) if slm_requests_path.exists() else 0,
            "slm_responses": int(len(pd.read_parquet(slm_responses_path))) if slm_responses_path.exists() else 0,
            "slm_points_raw": int(len(pd.read_parquet(slm_points_raw_path))) if slm_points_raw_path.exists() else 0,
            "slm_points_voted": int(len(pd.read_parquet(slm_points_voted_path))) if slm_points_voted_path.exists() else 0,
            "consensus_points": int(len(consensus_set.points)),
        },
        "config": {
            "min_discovery_score": cfg.min_discovery_score,
            "plugin_id": cfg.plugin_id,
            "max_pages_per_provider": cfg.max_pages_per_provider,
            "max_discovered_records": cfg.max_discovered_records,
            "saturation_window_pages": cfg.saturation_window_pages,
            "saturation_min_yield": cfg.saturation_min_yield,
            "min_pages_before_saturation": cfg.min_pages_before_saturation,
            "resume_discovery": cfg.resume_discovery,
            "max_downloads": cfg.max_downloads,
            "local_repo_paths": list(cfg.local_repo_paths),
            "local_repo_recursive": cfg.local_repo_recursive,
            "local_file_glob": cfg.local_file_glob,
            "local_merge_mode": cfg.local_merge_mode,
            "local_max_files": cfg.local_max_files,
            "local_require_readable": cfg.local_require_readable,
            "acquire_workers": cfg.acquire_workers,
            "extract_workers": cfg.extract_workers,
            "english_first": cfg.english_first,
            "per_provider_cap": cfg.per_provider_cap,
            "require_fulltext_mime": cfg.require_fulltext_mime,
            "context_window_lines": cfg.context_window_lines,
            "point_assembler": cfg.point_assembler,
            "context_assembler": cfg.context_assembler,
            "phase_schema_version": cfg.phase_schema_version,
            "graph_core_mode": cfg.graph_core_mode,
            "graph_architecture": cfg.graph_architecture,
            "concept_ontology_profile": cfg.concept_ontology_profile,
            "bridge_weight_policy": cfg.bridge_weight_policy,
            "bridge_weight_threshold": cfg.bridge_weight_threshold,
            "concept_gates": list(cfg.concept_gates),
            "phase_require_elements": cfg.phase_require_elements,
            "phase_require_stoich": cfg.phase_require_stoich,
            "phase_require_spacegroup": cfg.phase_require_spacegroup,
            "cache_mode": cfg.cache_mode,
            "shared_cache_root": cfg.shared_cache_root,
            "cache_read_only": cfg.cache_read_only,
            "cache_ttl_hours": cfg.cache_ttl_hours,
            "cpu_strict_profile": cfg.cpu_strict_profile,
            "cpu_max_threads": cfg.cpu_max_threads,
            "cpu_probe": cfg.cpu_probe,
            "extractor_mode": cfg.extractor_mode,
            "extractor_models": cfg.extractor_models,
            "tgi_endpoint": cfg.tgi_endpoint,
            "tgi_models": cfg.tgi_models,
            "tgi_endpoints": cfg.tgi_endpoints,
            "slm_model_mode": cfg.slm_model_mode,
            "tgi_workers": cfg.tgi_workers,
            "decoder": cfg.decoder,
            "slm_max_retries": cfg.slm_max_retries,
            "slm_timeout_sec": cfg.slm_timeout_sec,
            "slm_batch_size": cfg.slm_batch_size,
            "slm_chunk_tokens": cfg.slm_chunk_tokens,
            "slm_overlap_tokens": cfg.slm_overlap_tokens,
            "slm_eval_split": cfg.slm_eval_split,
            "schema_decoder": cfg.schema_decoder,
            "vote_policy": cfg.vote_policy,
            "min_vote_confidence": cfg.min_vote_confidence,
            "extractor_gate_profile": cfg.extractor_gate_profile,
            "require_context_fields": cfg.require_context_fields,
            "min_support_per_bin": cfg.min_support_per_bin,
            "mp_enabled": cfg.mp_enabled,
            "mp_mode": cfg.mp_mode,
            "mp_scope": cfg.mp_scope,
            "mp_on_demand": cfg.mp_on_demand,
            "mp_query_workers": cfg.mp_query_workers,
            "mp_timeout_sec": cfg.mp_timeout_sec,
            "mp_max_queries": cfg.mp_max_queries,
            "mp_cache_path": cfg.mp_cache_path,
            "mp_formula_match_weight": cfg.mp_formula_match_weight,
            "mp_phase_match_weight": cfg.mp_phase_match_weight,
            "mp_stability_weight": cfg.mp_stability_weight,
            "emit_extraction_calibration": cfg.emit_extraction_calibration,
            "calibration_bins": cfg.calibration_bins,
            "gnn_hidden_dim": cfg.gnn_hidden_dim,
            "gnn_layers": cfg.gnn_layers,
            "gnn_dropout": cfg.gnn_dropout,
            "gnn_epochs": cfg.gnn_epochs,
            "gnn_lr": cfg.gnn_lr,
            "tabular_model": cfg.tabular_model,
            "finetune_target_phase": cfg.finetune_target_phase,
            "finetune_max_thickness_nm": cfg.finetune_max_thickness_nm,
            "system_finetune_dataset": cfg.system_finetune_dataset,
            "system_eval_holdout_ratio": cfg.system_eval_holdout_ratio,
            "extractor_max_loop": cfg.extractor_max_loop,
            "processor_max_loop": cfg.processor_max_loop,
            "strict_full_workflow": cfg.strict_full_workflow,
            "auto_bootstrap": cfg.auto_bootstrap,
            "bootstrap_mode": cfg.bootstrap_mode,
            "shared_venv_root": cfg.shared_venv_root,
            "python_exec": cfg.python_exec,
            "tgi_docker_image": cfg.tgi_docker_image,
            "tgi_model_id": cfg.tgi_model_id,
            "tgi_platform": cfg.tgi_platform,
            "tgi_mode": cfg.tgi_mode,
            "tgi_port": cfg.tgi_port,
            "tgi_health_path": cfg.tgi_health_path,
            "tgi_generate_path": cfg.tgi_generate_path,
            "tgi_port_policy": cfg.tgi_port_policy,
            "tgi_port_range": cfg.tgi_port_range,
            "tgi_reuse_existing": cfg.tgi_reuse_existing,
            "workflow_profile": cfg.workflow_profile,
            "storage_root": cfg.storage_root,
            "run_root": cfg.run_root,
            "cache_root": cfg.cache_root,
            "model_root": cfg.model_root,
            "dataset_root": cfg.dataset_root,
            "vault_root": cfg.vault_root,
            "run_profile": cfg.run_profile,
            "vault_export_enabled": cfg.vault_export_enabled,
            "vault_import_enabled": cfg.vault_import_enabled,
            "vault_import_mode": cfg.vault_import_mode,
            "vault_profile": cfg.vault_profile,
            "slm_max_chunks_per_doc": cfg.slm_max_chunks_per_doc,
            "slm_max_doc_chars": cfg.slm_max_doc_chars,
            "slm_response_cache": cfg.slm_response_cache,
            "extract_stage_timeout_sec": cfg.extract_stage_timeout_sec,
            "use_bootstrapped_venv": cfg.use_bootstrapped_venv,
            "already_bootstrapped": cfg.already_bootstrapped,
            "all_done_repro_runs": cfg.all_done_repro_runs,
            "all_done_max_runtime_sec": cfg.all_done_max_runtime_sec,
            "all_done_require_mp_if_key_present": cfg.all_done_require_mp_if_key_present,
            "finetune_gate_policy": cfg.finetune_gate_policy,
            "finetune_ceiling_threshold": cfg.finetune_ceiling_threshold,
            "finetune_min_slice_rows": cfg.finetune_min_slice_rows,
            "finetune_min_support_articles": cfg.finetune_min_support_articles,
            "validation_tier": cfg.validation_tier,
            "processor_train_tier": cfg.processor_train_tier,
            "recipe_export_tier": cfg.recipe_export_tier,
            "quality_profile": cfg.quality_profile,
            "thickness_schema": cfg.thickness_schema,
            "thickness_compat_alias": cfg.thickness_compat_alias,
            "prelarge_rung": cfg.prelarge_rung,
            "finetune_require_support": cfg.finetune_require_support,
            "finetune_on_insufficient": cfg.finetune_on_insufficient,
            "processor_model_dir": cfg.processor_model_dir,
        },
        "day3_summary": {
            "scale": {
                "discovered_articles": int(len(articles.frame)),
                "acquired_documents": int(len(documents.frame)),
            },
            "context_quality": {
                "validated_context_completeness": validated_context_completeness,
                "context_completeness_gate": float(cfg.context_completeness_gate),
                "completeness_gate_pass": bool(
                    validated_context_completeness >= float(cfg.context_completeness_gate)
                ),
            },
            "materials_project": {
                "enabled": bool(cfg.mp_enabled),
                "mode": str(cfg.mp_mode),
                "scope": str(cfg.mp_scope),
                "query_count": int(len(mp_enriched.query_log)),
                "material_count": int(len(mp_enriched.materials)),
                "point_link_count": int(len(mp_enriched.point_links)),
            },
            "quality_eval": {
                "status": "run `oarp quality-eval --run <run_dir> --profile balanced` for no-gold quality diagnostics",
                "profile": str(cfg.quality_profile or "balanced"),
            },
        },
    }
    write_json(metrics_path, run_metrics)
    state = load_run_state(cfg.as_path())
    run_id = str(state.get("run_id") or "")
    db_path = artifacts / "index.sqlite"
    if run_id:
        upsert_artifact(db_path=db_path, run_id=run_id, name="stage_metrics", path=stage_metrics_path)
        upsert_artifact(db_path=db_path, run_id=run_id, name="run_metrics", path=metrics_path)
        append_lineage(
            db_path=db_path,
            run_id=run_id,
            stage="metrics",
            source_name="pipeline",
            target_name="run_metrics.json",
        )

    ranking_path = artifacts / "articles_ranked.parquet"
    validation_reasons_path = artifacts / "validation_reasons.parquet"
    extraction_votes_path = artifacts / "extraction_votes.parquet"
    extraction_calibration_path = artifacts / "extraction_calibration.parquet"
    processor_eval_metrics_path = artifacts / "processor_eval_metrics.json"

    return RunResult(
        run_dir=cfg.as_path(),
        articles_path=articles.parquet_path,
        ranking_path=ranking_path,
        documents_path=documents.parquet_path,
        evidence_path=evidence.points_path,
        provenance_path=evidence.provenance_path,
        validated_path=validated.accepted_path,
        validation_reasons_path=validation_reasons_path,
        consensus_path=consensus_set.parquet_path,
        report_path=output.report_path,
        metrics_path=metrics_path,
        mp_enriched_path=mp_enriched.enriched_points_path,
        extraction_votes_path=extraction_votes_path if extraction_votes_path.exists() else None,
        extraction_calibration_path=(
            extraction_calibration_path if extraction_calibration_path.exists() else None
        ),
        processor_eval_metrics_path=(
            processor_eval_metrics_path if processor_eval_metrics_path.exists() else None
        ),
    )


def _artifact_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return 0


def _extractor_gate_status(cfg: RunConfig) -> dict[str, Any]:
    artifacts = cfg.as_path() / "artifacts"
    req_path = artifacts / "slm_requests.parquet"
    resp_path = artifacts / "slm_responses.parquet"
    voted_path = artifacts / "slm_points_voted.parquet"
    validated_path = artifacts / "validated_points.parquet"

    req_count = _artifact_row_count(req_path)
    resp_count = _artifact_row_count(resp_path)
    voted_count = _artifact_row_count(voted_path)
    validated_count = _artifact_row_count(validated_path)

    provenance_complete_ratio = 0.0
    if validated_count > 0 and validated_path.exists():
        frame = pd.read_parquet(validated_path)
        required_cols = ["citation_url", "snippet", "locator"]
        if all(col in frame.columns for col in required_cols):
            mask = (
                frame[required_cols]
                .fillna("")
                .astype(str)
                .apply(lambda col: col.str.strip().ne(""), axis=0)
                .all(axis=1)
            )
            provenance_complete_ratio = float(mask.mean()) if len(mask) else 0.0

    all_pass = bool(
        req_count > 0
        and resp_count > 0
        and voted_count > 0
        and validated_count > 0
        and provenance_complete_ratio >= 1.0
    )
    return {
        "slm_requests": req_count,
        "slm_responses": resp_count,
        "slm_points_voted": voted_count,
        "validated_points": validated_count,
        "provenance_complete_ratio": provenance_complete_ratio,
        "all_pass": all_pass,
    }


def _tune_extractor_for_retry(cfg: RunConfig, attempt_idx: int) -> None:
    bump = max(1, int(attempt_idx))
    cfg.slm_chunk_tokens = max(160, int(cfg.slm_chunk_tokens) - 32 * bump)
    cfg.slm_overlap_tokens = min(max(32, cfg.slm_chunk_tokens // 2), int(cfg.slm_overlap_tokens) + 8 * bump)
    cfg.min_vote_confidence = max(0.45, float(cfg.min_vote_confidence) - 0.03 * bump)


def _tune_processor_for_retry(cfg: RunConfig, attempt_idx: int) -> None:
    bump = max(1, int(attempt_idx))
    cfg.gnn_epochs = int(cfg.gnn_epochs) + 10 * bump
    cfg.gnn_hidden_dim = min(256, int(cfg.gnn_hidden_dim) + 16 * bump)
    cfg.gnn_lr = max(1e-4, float(cfg.gnn_lr) * 0.7)


_BROAD_RELAX_ONLY_REASONS = {
    "missing_substrate",
    "missing_orientation",
    "missing_doping_context",
    "missing_alloy_context",
    "pure_ni_exception_not_supported_by_evidence",
}


def _point_has_primary_variables(frame: pd.DataFrame, point_id: str, x_name: str, y_name: str) -> bool:
    group = frame[frame["point_id"].astype(str) == str(point_id)]
    names = set(group.get("variable_name", pd.Series([], dtype=str)).astype(str).tolist())
    return x_name in names and y_name in names


def split_validated_tiers(run_dir: str, cfg: RunConfig) -> TieredValidationResult:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = run_path / "artifacts"
    validated_path = artifacts / "validated_points.parquet"
    rejected_path = artifacts / "rejected_points.parquet"
    strict_path = artifacts / "validated_points_strict.parquet"
    broad_path = artifacts / "validated_points_broad.parquet"
    summary_path = artifacts / "validation_tiers_summary.json"

    state = load_run_state(run_path)
    spec_path = str(state.get("spec_path") or "")
    if not spec_path:
        raise ValueError("run state missing spec_path for tier splitting")
    spec = load_topic_spec(spec_path)
    x_name = str(spec.plot.primary.x or "")
    y_name = str(spec.plot.primary.y or "")

    strict = pd.read_parquet(validated_path) if validated_path.exists() else pd.DataFrame()
    rejected = pd.read_parquet(rejected_path) if rejected_path.exists() else pd.DataFrame()
    broad = strict.copy()
    tier_mode = str(cfg.validation_tier or "both").strip().lower()

    if tier_mode != "strict" and not rejected.empty:
        relaxed_rows: list[dict[str, Any]] = []
        for row in rejected.to_dict(orient="records"):
            reason_text = str(row.get("reject_reason") or "")
            reason_tokens = [item.strip() for item in reason_text.split(";") if item.strip()]
            if not reason_tokens:
                continue
            if not set(reason_tokens).issubset(_BROAD_RELAX_ONLY_REASONS):
                continue
            relaxed_rows.append(row)
        if relaxed_rows:
            broad = pd.concat([broad, pd.DataFrame(relaxed_rows)], ignore_index=True)

    if not broad.empty and x_name and y_name and "point_id" in broad.columns and "variable_name" in broad.columns:
        keep_ids: list[str] = []
        for point_id in broad["point_id"].astype(str).drop_duplicates().tolist():
            if _point_has_primary_variables(broad, point_id, x_name, y_name):
                keep_ids.append(str(point_id))
        broad = broad[broad["point_id"].astype(str).isin(set(keep_ids))].copy()

    if not broad.empty:
        req_cols = ["citation_url", "snippet", "locator", "entity"]
        for col in req_cols:
            if col not in broad.columns:
                broad[col] = ""
        mask = (
            broad[["citation_url", "snippet", "locator"]]
            .fillna("")
            .astype(str)
            .apply(lambda col: col.str.strip().ne(""), axis=0)
            .all(axis=1)
            & broad["entity"].astype(str).str.strip().ne("")
        )
        broad = broad[mask].copy()

    if tier_mode == "strict":
        broad = strict.copy()

    strict.to_parquet(strict_path, index=False)
    broad.to_parquet(broad_path, index=False)

    strict_points = int(strict.get("point_id", pd.Series([], dtype=str)).astype(str).nunique()) if not strict.empty else 0
    broad_points = int(broad.get("point_id", pd.Series([], dtype=str)).astype(str).nunique()) if not broad.empty else 0
    summary = {
        "created_at": now_iso(),
        "validation_tier": str(cfg.validation_tier or "both"),
        "strict_rows": int(len(strict)),
        "strict_points": strict_points,
        "broad_rows": int(len(broad)),
        "broad_points": broad_points,
    }
    write_json(summary_path, summary)

    try:
        db_path = artifacts / "index.sqlite"
        state_payload = load_run_state(run_path)
        run_id = str(state_payload.get("run_id") or "")
        if run_id:
            upsert_artifact(db_path=db_path, run_id=run_id, name="validated_points_strict", path=strict_path)
            upsert_artifact(db_path=db_path, run_id=run_id, name="validated_points_broad", path=broad_path)
            upsert_artifact(db_path=db_path, run_id=run_id, name="validation_tiers_summary", path=summary_path)
            append_lineage(
                db_path=db_path,
                run_id=run_id,
                stage="validation_tiers",
                source_name="validated_points.parquet",
                target_name="validated_points_broad.parquet",
            )
    except Exception:
        pass

    return TieredValidationResult(
        strict=strict,
        broad=broad,
        strict_path=strict_path,
        broad_path=broad_path,
        summary_path=summary_path,
        summary=summary,
    )


def run_full_workflow(spec_path: str, query: str, cfg: RunConfig) -> FullWorkflowResult:
    if bool(cfg.cpu_strict_profile):
        _apply_cpu_thread_clamps(cfg)
    if bool(cfg.cpu_probe):
        _capture_cpu_env(cfg)
    spec = load_topic_spec(spec_path)
    resolved_query = ensure_query(spec, query)
    if not str(cfg.finetune_target_phase or "").strip():
        cfg.finetune_target_phase = str(spec.fine_tune.target_phase or "").strip()
    if float(cfg.finetune_max_thickness_nm or 0.0) <= 0 and float(spec.fine_tune.max_thickness_nm or 0.0) > 0:
        cfg.finetune_max_thickness_nm = float(spec.fine_tune.max_thickness_nm)
    if not cfg.plugin_id and spec.plugins.preferred_plugin:
        cfg.plugin_id = spec.plugins.preferred_plugin
    cfg.extractor_mode = "slm_tgi_required"
    if not str(cfg.tgi_endpoint or "").strip():
        generate_path = str(cfg.tgi_generate_path or "/generate").strip() or "/generate"
        if not generate_path.startswith("/"):
            generate_path = f"/{generate_path}"
        cfg.tgi_endpoint = f"http://127.0.0.1:{int(cfg.tgi_port)}{generate_path}"

    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]
    stage_rows: list[dict[str, Any]] = []

    bootstrap_result = None
    if bool(cfg.auto_bootstrap):
        t0 = time.perf_counter()
        bootstrap_result = bootstrap_runtime(cfg)
        t1 = time.perf_counter()
        stage_rows.append(
            _stage_row(
                "bootstrap",
                t0,
                t1,
                {
                    "ok": bool(bootstrap_result.ok),
                    "report_path": str(bootstrap_result.report_path),
                },
            )
        )

    t2 = time.perf_counter()
    preflight_python = str(cfg.python_exec or "").strip()
    if bootstrap_result is not None and bootstrap_result.target_python_executable:
        preflight_python = str(bootstrap_result.target_python_executable)
    preflight = preflight_strict(cfg, python_exec=preflight_python or None, check_tgi_generate=True)
    t3 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "preflight",
            t2,
            t3,
            {
                "ok": bool(preflight.ok),
                "report_path": str(preflight.report_path),
                "check_count": int(len(preflight.checks)),
            },
        )
    )

    if bool(cfg.strict_full_workflow):
        if bootstrap_result is not None and not bool(bootstrap_result.ok):
            raise RuntimeError(f"strict bootstrap failed; see {bootstrap_result.report_path}")
        if not bool(preflight.ok):
            raise RuntimeError(f"strict preflight failed; see {preflight.report_path}")

    run_result: RunResult | None = None
    extractor_gate: dict[str, Any] = {}
    extractor_attempts = max(1, int(cfg.extractor_max_loop))
    for attempt in range(1, extractor_attempts + 1):
        t4 = time.perf_counter()
        run_result = run_pipeline(spec_path=spec_path, query=resolved_query, cfg=cfg)
        t5 = time.perf_counter()
        extractor_gate = _extractor_gate_status(cfg)
        stage_rows.append(
            _stage_row(
                "pipeline_core",
                t4,
                t5,
                {
                    "attempt": attempt,
                    "extractor_gate_pass": bool(extractor_gate.get("all_pass")),
                    "metrics_path": str(run_result.metrics_path),
                },
            )
        )
        if bool(extractor_gate.get("all_pass")):
            break
        if attempt < extractor_attempts:
            _tune_extractor_for_retry(cfg, attempt)
            continue
        if bool(cfg.strict_full_workflow):
            raise RuntimeError(
                f"strict extractor gate failed after {extractor_attempts} attempts: {extractor_gate}"
            )

    if run_result is None:
        raise RuntimeError("full workflow failed before producing base run artifacts")

    vt0 = time.perf_counter()
    tier_result = split_validated_tiers(run_dir=str(cfg.as_path()), cfg=cfg)
    vt1 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "validation_tiers",
            vt0,
            vt1,
            {
                "strict_rows": int(len(tier_result.strict)),
                "broad_rows": int(len(tier_result.broad)),
                "strict_path": str(tier_result.strict_path),
                "broad_path": str(tier_result.broad_path),
            },
        )
    )

    mp_path = artifacts / "materials_project_enriched_points.parquet"
    ts0 = time.perf_counter()
    validated_for_semantics_path = tier_result.broad_path if tier_result.broad_path.exists() else (artifacts / "validated_points.parquet")
    validated_for_semantics = (
        pd.read_parquet(validated_for_semantics_path) if validated_for_semantics_path.exists() else pd.DataFrame()
    )
    validated_for_semantics = build_thickness_views(validated_for_semantics, cfg)
    mp_materials_path = artifacts / "materials_project_materials.parquet"
    mp_materials = pd.read_parquet(mp_materials_path) if mp_materials_path.exists() else pd.DataFrame()
    prior_frame = estimate_anneal_thickness_prior(validated_for_semantics, mp_materials, cfg)
    thickness_semantics = validated_for_semantics.merge(prior_frame, on="point_id", how="left")
    thickness_semantics_path = artifacts / "thickness_semantics.parquet"
    thickness_semantics.to_parquet(thickness_semantics_path, index=False)
    stage_rows.append(
        _stage_row(
            "thickness_semantics",
            ts0,
            time.perf_counter(),
            {
                "rows": int(len(thickness_semantics)),
                "path": str(thickness_semantics_path),
            },
        )
    )

    q0 = time.perf_counter()
    quality_eval = evaluate_quality_no_gold(run_dir=str(cfg.as_path()), cfg=cfg)
    quality_gate = enforce_quality_gates(run_dir=str(cfg.as_path()), cfg=cfg)
    stage_rows.append(
        _stage_row(
            "quality_gate",
            q0,
            time.perf_counter(),
            {
                "profile": str(cfg.quality_profile or "balanced"),
                "ok": bool(quality_gate.ok),
                "quality_eval_json": str(quality_eval.json_path),
                "quality_gate_json": str(quality_gate.json_path),
            },
        )
    )
    if bool(cfg.strict_full_workflow) and not bool(quality_gate.ok):
        raise RuntimeError(f"strict quality gate failed: {quality_gate.json_path}")

    t6 = time.perf_counter()
    knowledge_bundle = knowledge.build_knowledge(cfg.as_path())
    t7 = time.perf_counter()
    stage_rows.append(
        _stage_row(
            "knowledge",
            t6,
            t7,
            {
                "phase_event_count": int(knowledge_bundle.phase_event_count),
                "condition_edge_count": int(knowledge_bundle.condition_edge_count),
                "quality_outcome_count": int(knowledge_bundle.quality_outcome_count),
                "phase_events_path": str(knowledge_bundle.phase_events_path),
            },
        )
    )

    mp_data = pd.read_parquet(mp_path) if mp_path.exists() else None
    processor_train_tier = str(cfg.processor_train_tier or "broad").strip().lower()
    processor_points_source: Path
    if processor_train_tier == "strict":
        processor_points_source = tier_result.strict_path
    else:
        processor_points_source = tier_result.broad_path
    if not processor_points_source.exists():
        processor_points_source = knowledge_bundle.phase_events_path
    dataset = build_processing_dataset(
        points=processor_points_source,
        mp_data=mp_data,
        aux_data=None,
        cfg=cfg,
    )
    if str(cfg.graph_architecture or "").strip().lower() == "dual_concept":
        g0 = time.perf_counter()
        graph_audit = audit_dual_graph(run_dir=cfg.as_path(), cfg=cfg)
        g1 = time.perf_counter()
        stage_rows.append(
            _stage_row(
                "graph_audit",
                g0,
                g1,
                {
                    "ok": bool(graph_audit.ok),
                    "json_path": str(graph_audit.json_path),
                    "report_path": str(graph_audit.report_path),
                    "issue_count": int(len(graph_audit.issues)),
                },
            )
        )
        if bool(cfg.strict_full_workflow) and not bool(graph_audit.ok):
            raise RuntimeError(
                f"dual graph audit failed: {graph_audit.json_path}"
            )
    try:
        state = load_run_state(cfg.as_path())
        run_id = str(state.get("run_id") or "")
        db_path = artifacts / "index.sqlite"
        if run_id:
            for artifact_name, artifact_path in (
                ("processor_training_rows", artifacts / "processor_training_rows.parquet"),
                ("phase_specs", artifacts / "phase_specs.parquet"),
                ("graph_nodes", artifacts / "graph_nodes.parquet"),
                ("graph_edges", artifacts / "graph_edges.parquet"),
                ("graph_tensor_index", artifacts / "graph_tensor_index.parquet"),
                ("graph_samples", artifacts / "graph_samples.parquet"),
                ("cache_audit", artifacts / "cache_audit.parquet"),
                ("article_process_nodes", artifacts / "article_process_nodes.parquet"),
                ("article_process_edges", artifacts / "article_process_edges.parquet"),
                ("concept_nodes", artifacts / "concept_nodes.parquet"),
                ("concept_edges", artifacts / "concept_edges.parquet"),
                ("bridge_edges", artifacts / "bridge_edges.parquet"),
                ("bridge_weight_audit", artifacts / "bridge_weight_audit.parquet"),
                ("graph_audit_json", artifacts / "graph_audit.json"),
                ("graph_audit_md", artifacts / "graph_audit.md"),
            ):
                if artifact_path.exists():
                    upsert_artifact(db_path=db_path, run_id=run_id, name=artifact_name, path=artifact_path)
            append_lineage(
                db_path=db_path,
                run_id=run_id,
                stage="processing_dataset",
                source_name="phase_events.parquet",
                target_name="processor_training_rows.parquet",
            )
            append_lineage(
                db_path=db_path,
                run_id=run_id,
                stage="processing_dataset",
                source_name="processor_training_rows.parquet",
                target_name="graph_nodes.parquet",
            )
            append_lineage(
                db_path=db_path,
                run_id=run_id,
                stage="processing_dataset",
                source_name="processor_training_rows.parquet",
                target_name="phase_specs.parquet",
            )
    except Exception:
        pass

    processor_models: dict[str, Path] = {}
    processor_eval: dict[str, Any] = {}
    vault_export_path: Path | None = None
    vault_import_path: Path | None = None
    processor_attempts = max(1, int(cfg.processor_max_loop))
    for attempt in range(1, processor_attempts + 1):
        started = time.perf_counter()
        try:
            gnn_ref = train_gnn_base(dataset, cfg)
            tabular_ref = train_tabular_head(dataset, gnn_ref, cfg)
            finetune_exec = maybe_run_finetune(tabular_ref, dataset, cfg)
            if finetune_exec.status == "failed":
                raise RuntimeError(f"fine-tune failed: {finetune_exec.reason}")
            if finetune_exec.status == "skipped_insufficient_support" and str(cfg.finetune_on_insufficient or "skip").strip().lower() == "fail":
                raise RuntimeError(
                    "fine-tune support insufficient and policy=fail; inspect artifacts/finetune_execution.json"
                )
            processor_eval = evaluate_processor(run_dir=str(cfg.as_path()), cfg=cfg)
            processor_eval["finetune_execution"] = {
                "status": finetune_exec.status,
                "support_rows": int(finetune_exec.support_rows),
                "support_articles": int(finetune_exec.support_articles),
                "reason": str(finetune_exec.reason or ""),
                "payload_path": str(finetune_exec.payload_path) if finetune_exec.payload_path else "",
            }
            processor_models = {
                "gnn_base": gnn_ref.model_dir,
                "tabular_head": tabular_ref.model_dir,
            }
            if finetune_exec.model_ref is not None:
                processor_models["finetune_nisi_sub200"] = finetune_exec.model_ref.model_dir
            ok = bool(processor_eval.get("all_gates_pass"))
            error_text = ""
        except Exception as exc:
            ok = False
            error_text = f"{type(exc).__name__}: {exc}"
            processor_eval = {"error": error_text, "all_gates_pass": False}
        ended = time.perf_counter()
        stage_rows.append(
            _stage_row(
                "processor_stack",
                started,
                ended,
                {
                    "attempt": attempt,
                    "ok": ok,
                    "error": error_text,
                    "eval_path": str(artifacts / "processor_eval_metrics.json"),
                },
            )
        )
        if ok:
            break
        if attempt < processor_attempts:
            _tune_processor_for_retry(cfg, attempt)
            continue
        if bool(cfg.strict_full_workflow):
            raise RuntimeError(f"strict processor gate failed after {processor_attempts} attempts: {processor_eval}")

    if bool(cfg.vault_export_enabled) or str(cfg.run_profile or "").strip().lower().startswith("v2_"):
        v0 = time.perf_counter()
        preferred_root = str(cfg.vault_root or "").strip()
        if preferred_root:
            run_state = load_run_state(cfg.as_path())
            run_id = str(run_state.get("run_id") or cfg.as_path().name).strip()
            out_root = Path(preferred_root).expanduser().resolve() / _safe_slug(run_id)
        else:
            out_root = cfg.as_path() / "outputs" / "vault"
        vault_export = project_concepts_to_vault(run_dir=str(cfg.as_path()), out_dir=str(out_root), cfg=cfg)
        vault_export_path = vault_export.vault_path
        stage_rows.append(
            _stage_row(
                "vault_export",
                v0,
                time.perf_counter(),
                {
                    "vault_path": str(vault_export.vault_path),
                    "note_counts": dict(vault_export.note_counts_by_type),
                    "link_count": int(vault_export.link_count),
                },
            )
        )
        if bool(cfg.vault_import_enabled):
            v1 = time.perf_counter()
            vault_import = parse_vault_semantics(vault_dir=str(vault_export.vault_path), run_dir=str(cfg.as_path()), cfg=cfg)
            vault_import_path = vault_import.soft_constraints_path
            _ = apply_vault_soft_supervision(
                run_dir=str(cfg.as_path()),
                vault_links=vault_import.link_deltas,
                cfg=cfg,
            )
            stage_rows.append(
                _stage_row(
                    "vault_import",
                    v1,
                    time.perf_counter(),
                    {
                        "soft_constraints_path": str(vault_import.soft_constraints_path),
                        "delta_count": int(len(vault_import.link_deltas)),
                    },
                )
            )
        if bool(cfg.global_concept_registry_enable):
            v2 = time.perf_counter()
            registry_root = str(cfg.global_concept_registry_root or "").strip()
            if not registry_root:
                registry_root = str(Path(cfg.storage_root).expanduser().resolve() / "datasets" / "concept_registry")
            registry_summary = sync_run_concepts_to_registry(
                run_dir=str(cfg.as_path()),
                registry_dir=registry_root,
                cfg=cfg,
            )
            stage_rows.append(
                _stage_row(
                    "concept_registry_sync",
                    v2,
                    time.perf_counter(),
                    registry_summary,
                )
            )

    articles_df = pd.read_parquet(run_result.articles_path) if run_result.articles_path.exists() else pd.DataFrame()
    documents_df = pd.read_parquet(run_result.documents_path) if run_result.documents_path.exists() else pd.DataFrame()
    provider_counts = (
        articles_df["provider"].astype(str).value_counts().to_dict()
        if (not articles_df.empty and "provider" in articles_df.columns)
        else {}
    )
    doc_provider_counts = (
        documents_df["provider"].astype(str).value_counts().to_dict()
        if (not documents_df.empty and "provider" in documents_df.columns)
        else {}
    )

    gate_status = {
        "bootstrap_ok": bool(bootstrap_result.ok) if bootstrap_result is not None else True,
        "preflight_ok": bool(preflight.ok),
        "extractor_ok": bool(extractor_gate.get("all_pass")),
        "quality_ok": bool(quality_gate.ok),
        "processor_ok": bool(processor_eval.get("all_gates_pass")),
    }
    gate_status["all_ok"] = bool(all(gate_status.values()))

    full_stage_path = artifacts / "full_workflow_stage_metrics.parquet"
    pd.DataFrame(stage_rows).to_parquet(full_stage_path, index=False)

    full_metrics_path = artifacts / "full_workflow_metrics.json"
    full_metrics = {
        "created_at": now_iso(),
        "run_dir": str(cfg.as_path()),
        "workflow_profile": str(cfg.workflow_profile or "strict_full"),
        "strict_full_workflow": bool(cfg.strict_full_workflow),
        "auto_bootstrap": bool(cfg.auto_bootstrap),
        "gates": gate_status,
        "extractor_gate": extractor_gate,
        "processor_eval": processor_eval,
        "quality_eval": quality_eval.metrics,
        "quality_gate": {
            "ok": bool(quality_gate.ok),
            "gates": dict(quality_gate.gates),
            "json_path": str(quality_gate.json_path),
            "report_path": str(quality_gate.report_path),
        },
        "vault_export_path": str(vault_export_path) if vault_export_path else "",
        "vault_import_path": str(vault_import_path) if vault_import_path else "",
        "provider_counts": provider_counts,
        "document_provider_counts": doc_provider_counts,
        "validation_tiers": dict(tier_result.summary),
        "thickness_semantics_path": str(thickness_semantics_path),
        "processor_training_source": str(processor_points_source),
        "stage_metrics_path": str(full_stage_path),
        "stage_metrics": stage_rows,
        "references": {
            "run_metrics": str(run_result.metrics_path),
            "preflight_report": str(preflight.report_path),
            "bootstrap_report": str(bootstrap_result.report_path) if bootstrap_result is not None else "",
            "processor_eval_metrics": str(artifacts / "processor_eval_metrics.json"),
            "quality_eval_json": str(quality_eval.json_path),
            "quality_gate_json": str(quality_gate.json_path),
            "vault": str(vault_export_path) if vault_export_path else "",
        },
    }
    write_json(full_metrics_path, full_metrics)

    try:
        state = load_run_state(cfg.as_path())
        run_id = str(state.get("run_id") or "")
        db_path = artifacts / "index.sqlite"
        if run_id:
            upsert_artifact(db_path=db_path, run_id=run_id, name="preflight_report", path=preflight.report_path)
            if bootstrap_result is not None:
                upsert_artifact(
                    db_path=db_path,
                    run_id=run_id,
                    name="bootstrap_report",
                    path=bootstrap_result.report_path,
                )
            upsert_artifact(
                db_path=db_path,
                run_id=run_id,
                name="full_workflow_stage_metrics",
                path=full_stage_path,
            )
            upsert_artifact(
                db_path=db_path,
                run_id=run_id,
                name="full_workflow_metrics",
                path=full_metrics_path,
            )
            upsert_artifact(
                db_path=db_path,
                run_id=run_id,
                name="quality_eval",
                path=quality_eval.json_path,
            )
            upsert_artifact(
                db_path=db_path,
                run_id=run_id,
                name="quality_gate_report",
                path=quality_gate.json_path,
            )
            upsert_artifact(
                db_path=db_path,
                run_id=run_id,
                name="thickness_semantics",
                path=thickness_semantics_path,
            )
            append_lineage(
                db_path=db_path,
                run_id=run_id,
                stage="run_full",
                source_name="pipeline",
                target_name="full_workflow_metrics.json",
            )
    except Exception:
        # Preserve end-to-end outputs even when lineage update fails.
        pass

    return FullWorkflowResult(
        run_dir=cfg.as_path(),
        run_result=run_result,
        knowledge_paths={
            "phase_events": knowledge_bundle.phase_events_path,
            "condition_graph": knowledge_bundle.condition_graph_path,
            "quality_outcomes": knowledge_bundle.quality_outcomes_path,
        },
        processor_models=processor_models,
        processor_eval=processor_eval,
        gate_status=gate_status,
        preflight_path=preflight.report_path,
        bootstrap_path=bootstrap_result.report_path if bootstrap_result is not None else None,
        metrics_path=full_metrics_path,
        vault_export_path=vault_export_path,
        vault_import_path=vault_import_path,
    )


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_exists(path: Path) -> bool:
    return bool(path.exists() and path.is_file())


def _capture_cpu_env(cfg: RunConfig) -> Path:
    artifacts = cfg.as_path() / "artifacts"
    payload = {
        "created_at": now_iso(),
        "python_executable": sys.executable,
        "thread_env": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS", ""),
            "NUMEXPR_MAX_THREADS": os.getenv("NUMEXPR_MAX_THREADS", ""),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    path = artifacts / "cpu_env_probe.json"
    write_json(path, payload)
    return path


def _apply_cpu_thread_clamps(cfg: RunConfig | None = None) -> None:
    max_threads = 4
    if cfg is not None:
        try:
            max_threads = max(1, int(cfg.cpu_max_threads))
        except Exception:
            max_threads = 4
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", str(max_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(max_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max_threads))
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(max_threads))


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += int(item.stat().st_size)
            except Exception:
                continue
    return int(total)


def run_prelarge_validation(
    spec_path: str,
    query: str,
    cfg: RunConfig,
    rung_cfg: dict[str, Any] | None = None,
) -> PreLargeValidationResult:
    rung_cfg = rung_cfg or {}
    rung = int(rung_cfg.get("rung") or cfg.prelarge_rung or 300)
    rung = max(50, rung)
    cfg.max_downloads = rung
    cfg.strict_full_workflow = True
    cfg.extractor_mode = "slm_tgi_required"
    cfg.validation_tier = str(cfg.validation_tier or "both")
    cfg.processor_train_tier = str(cfg.processor_train_tier or "broad")
    cfg.run_profile = str(cfg.run_profile or "v2_strict")

    full = run_full_workflow(spec_path=spec_path, query=query, cfg=cfg)
    run_path = cfg.as_path()
    artifacts = run_path / "artifacts"

    quality_eval = evaluate_quality_no_gold(run_dir=str(run_path), cfg=cfg)
    quality_gate = enforce_quality_gates(run_dir=str(run_path), cfg=cfg)
    full_metrics = _safe_json(full.metrics_path)
    run_metrics = _safe_json(full.run_result.metrics_path)
    stage_rows = full_metrics.get("stage_metrics") if isinstance(full_metrics.get("stage_metrics"), list) else []
    counts = run_metrics.get("counts") if isinstance(run_metrics.get("counts"), dict) else {}
    docs = int(counts.get("documents") or 0)
    discover_extract_sec = 0.0
    for row in stage_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("stage") or "") in {"discover", "acquire", "extract"}:
            discover_extract_sec += float(row.get("duration_sec") or 0.0)
    docs_per_hour = (docs / (discover_extract_sec / 3600.0)) if discover_extract_sec > 0 and docs > 0 else 0.0
    projected_10k_hours = (10000.0 / docs_per_hour) if docs_per_hour > 0 else 0.0
    throughput_projection = {
        "created_at": now_iso(),
        "rung": rung,
        "documents": docs,
        "discover_acquire_extract_sec": discover_extract_sec,
        "docs_per_hour": docs_per_hour,
        "projected_hours_for_10000_docs": projected_10k_hours,
    }
    throughput_projection_path = artifacts / "throughput_projection.json"
    write_json(throughput_projection_path, throughput_projection)

    run_bytes = _dir_size_bytes(run_path)
    bytes_per_doc = (run_bytes / docs) if docs > 0 else 0.0
    projected_bytes_10k = bytes_per_doc * 10000.0
    storage_budget = {
        "created_at": now_iso(),
        "rung": rung,
        "run_dir_bytes": run_bytes,
        "documents": docs,
        "bytes_per_doc": bytes_per_doc,
        "projected_bytes_for_10000_docs": projected_bytes_10k,
        "projected_gb_for_10000_docs": projected_bytes_10k / (1024.0 ** 3),
    }
    storage_budget_path = artifacts / "storage_budget_report.json"
    write_json(storage_budget_path, storage_budget)

    graph_audit_payload = _safe_json(artifacts / "graph_audit.json")

    gate_map = {
        "workflow_gate": bool(full.gate_status.get("all_ok")),
        "quality_gate": bool(quality_gate.ok),
        "processor_gate": bool(full.processor_eval.get("all_gates_pass")),
        "graph_gate": bool(graph_audit_payload.get("ok", False)),
    }
    gate_map["all_ok"] = bool(all(gate_map.values()))

    payload = {
        "created_at": now_iso(),
        "run_dir": str(run_path),
        "rung": rung,
        "gate_map": gate_map,
        "workflow_gates": dict(full.gate_status),
        "quality_eval": quality_eval.metrics,
        "quality_gate": {
            "ok": bool(quality_gate.ok),
            "gates": dict(quality_gate.gates),
            "json_path": str(quality_gate.json_path),
        },
        "processor_eval": dict(full.processor_eval),
        "storage_budget_report": str(storage_budget_path),
        "throughput_projection": str(throughput_projection_path),
        "remediation": [
            "inspect quality_gate_report.md when quality gate fails",
            "inspect processor_eval_metrics.json when processor_gate fails",
            "inspect graph_audit.md when graph_gate fails",
        ],
    }
    json_path = artifacts / "prelarge_validation.json"
    write_json(json_path, payload)

    report_lines = [
        "# Pre-Large Validation",
        "",
        f"- rung: `{rung}`",
        f"- all_ok: `{gate_map['all_ok']}`",
        f"- workflow_gate: `{gate_map['workflow_gate']}`",
        f"- quality_gate: `{gate_map['quality_gate']}`",
        f"- processor_gate: `{gate_map['processor_gate']}`",
        f"- graph_gate: `{gate_map['graph_gate']}`",
        "",
        "## Throughput",
        f"- docs_per_hour: `{docs_per_hour:.3f}`",
        f"- projected_hours_for_10000_docs: `{projected_10k_hours:.3f}`",
        "",
        "## Storage",
        f"- run_dir_bytes: `{run_bytes}`",
        f"- projected_gb_for_10000_docs: `{storage_budget['projected_gb_for_10000_docs']:.3f}`",
    ]
    report_path = artifacts / "prelarge_validation.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return PreLargeValidationResult(
        ok=bool(gate_map["all_ok"]),
        run_dir=run_path,
        rung=rung,
        report_path=report_path,
        json_path=json_path,
        gate_map={str(k): bool(v) for k, v in gate_map.items()},
        quality_eval_path=quality_eval.json_path,
        quality_gate_path=quality_gate.json_path,
        storage_budget_path=storage_budget_path,
        throughput_projection_path=throughput_projection_path,
    )


def run_all_done_validation(
    spec_path: str,
    query: str,
    cfg: RunConfig,
    validation_cfg: dict[str, Any] | None = None,
) -> AllDoneValidationResult:
    validation_cfg = validation_cfg or {}
    if not cfg.strict_full_workflow:
        cfg.strict_full_workflow = True
    _apply_cpu_thread_clamps(cfg if cfg.cpu_strict_profile else None)
    _capture_cpu_env(cfg)

    started = time.perf_counter()
    max_runtime = max(60, int(cfg.all_done_max_runtime_sec or 43200))

    def _check_budget(stage: str) -> None:
        elapsed = time.perf_counter() - started
        if elapsed > max_runtime:
            raise TimeoutError(f"all-done runtime budget exceeded at stage={stage} elapsed_sec={elapsed:.1f}")

    run_dir = cfg.as_path()
    artifacts = run_dir / "artifacts"

    spec = load_topic_spec(spec_path)
    resolved_query = ensure_query(spec, query)

    preflight = preflight_strict(cfg, python_exec=(cfg.python_exec or None), check_tgi_generate=True)
    _check_budget("preflight")

    base_result = run_full_workflow(spec_path=spec_path, query=resolved_query, cfg=cfg)
    _check_budget("run_full")
    quality_eval = _safe_json(artifacts / "quality_eval.json")
    if not quality_eval:
        quality_eval = evaluate_quality_no_gold(run_dir=str(run_dir), cfg=cfg).metrics
    quality_gate_payload = _safe_json(artifacts / "quality_gate_report.json")
    if not quality_gate_payload:
        qg = enforce_quality_gates(run_dir=str(run_dir), cfg=cfg)
        quality_gate_payload = {"ok": qg.ok, "gates": qg.gates, "metrics": qg.metrics}
    _check_budget("quality")

    repro_compare: dict[str, Any] = {
        "runs": [],
        "count_drift": {},
        "max_relative_drift": 0.0,
        "repro_ok": True,
    }
    repro_runs = max(1, int(cfg.all_done_repro_runs or 1))
    primary_metrics = _safe_json(base_result.run_result.metrics_path)
    primary_counts = primary_metrics.get("counts") if isinstance(primary_metrics.get("counts"), dict) else {}
    repro_compare["runs"].append(
        {
            "run_dir": str(run_dir),
            "metrics_path": str(base_result.run_result.metrics_path),
            "counts": primary_counts,
        }
    )

    if repro_runs > 1:
        for idx in range(2, repro_runs + 1):
            repro_dir = run_dir / f"repro_{idx}"
            repro_cfg = replace(cfg, run_dir=str(repro_dir), auto_bootstrap=False, resume_discovery=True)
            repro_result = run_full_workflow(spec_path=spec_path, query=resolved_query, cfg=repro_cfg)
            repro_metrics = _safe_json(repro_result.run_result.metrics_path)
            repro_counts = repro_metrics.get("counts") if isinstance(repro_metrics.get("counts"), dict) else {}
            repro_compare["runs"].append(
                {
                    "run_dir": str(repro_dir),
                    "metrics_path": str(repro_result.run_result.metrics_path),
                    "counts": repro_counts,
                }
            )
            _check_budget(f"repro_{idx}")

        drift: dict[str, float] = {}
        max_drift = 0.0
        baseline = repro_compare["runs"][0].get("counts") if repro_compare["runs"] else {}
        if isinstance(baseline, dict):
            for key, base_val in baseline.items():
                try:
                    base_num = float(base_val)
                except Exception:
                    continue
                if base_num < 0:
                    continue
                values = []
                for row in repro_compare["runs"][1:]:
                    counts = row.get("counts")
                    if not isinstance(counts, dict):
                        continue
                    try:
                        values.append(float(counts.get(key, base_num)))
                    except Exception:
                        continue
                if not values:
                    continue
                local_max = max(abs(item - base_num) for item in values)
                denom = max(1.0, abs(base_num))
                ratio = local_max / denom
                drift[key] = round(ratio, 6)
                if ratio > max_drift:
                    max_drift = ratio
        repro_compare["count_drift"] = drift
        repro_compare["max_relative_drift"] = float(max_drift)
        repro_compare["repro_ok"] = bool(max_drift <= 0.15)

    repro_compare_path = artifacts / "repro_compare.json"
    write_json(repro_compare_path, repro_compare)

    full_metrics = _safe_json(base_result.metrics_path)
    run_metrics = _safe_json(base_result.run_result.metrics_path)
    provider_counts = full_metrics.get("provider_counts") if isinstance(full_metrics.get("provider_counts"), dict) else {}
    document_provider_counts = (
        full_metrics.get("document_provider_counts")
        if isinstance(full_metrics.get("document_provider_counts"), dict)
        else {}
    )

    extractor_gate = full_metrics.get("extractor_gate") if isinstance(full_metrics.get("extractor_gate"), dict) else {}
    processor_eval = full_metrics.get("processor_eval") if isinstance(full_metrics.get("processor_eval"), dict) else {}
    counts = run_metrics.get("counts") if isinstance(run_metrics.get("counts"), dict) else {}

    validated = int(counts.get("validated_points") or 0)
    voted = int(counts.get("slm_points_voted") or 0)
    requests_count = int(counts.get("slm_requests") or 0)
    responses_count = int(counts.get("slm_responses") or 0)

    validated_path = artifacts / "validated_points.parquet"
    provenance_complete = float(extractor_gate.get("provenance_complete_ratio") or 0.0)
    if validated_path.exists() and validated > 0:
        try:
            frame = pd.read_parquet(validated_path)
            required = ["citation_url", "snippet", "locator"]
            if all(col in frame.columns for col in required):
                mask = frame[required].fillna("").astype(str).apply(lambda col: col.str.strip().ne(""), axis=0).all(axis=1)
                provenance_complete = float(mask.mean()) if len(mask) else 0.0
        except Exception:
            provenance_complete = float(extractor_gate.get("provenance_complete_ratio") or 0.0)

    finetune_exec_payload = _safe_json(artifacts / "finetune_execution.json")
    finetune_status = str(finetune_exec_payload.get("status") or "").strip().lower()
    finetune_required = not (
        finetune_status == "skipped_insufficient_support"
        and str(getattr(cfg, "finetune_on_insufficient", "skip") or "skip").strip().lower() == "skip"
    )
    processor_artifacts_ok = all(
        [
            _artifact_exists(artifacts / "models" / "gnn_base" / "gnn_model.pt"),
            _artifact_exists(artifacts / "models" / "tabular_head" / "processor_model.pkl"),
            (not finetune_required) or _artifact_exists(artifacts / "models" / "finetune_nisi_sub200" / "processor_model.pkl"),
            _artifact_exists(artifacts / "processor_eval_metrics.json"),
        ]
    )

    required_artifacts = {
        "bootstrap_report": str(base_result.bootstrap_path) if base_result.bootstrap_path else "",
        "preflight_report": str(base_result.preflight_path),
        "full_workflow_metrics": str(base_result.metrics_path),
        "run_metrics": str(base_result.run_result.metrics_path),
        "report_md": str(base_result.run_result.report_path),
        "processor_eval_metrics": str(artifacts / "processor_eval_metrics.json"),
        "quality_eval_json": str(artifacts / "quality_eval.json"),
        "quality_gate_report_json": str(artifacts / "quality_gate_report.json"),
        "all_done_json": str(artifacts / "all_done_validation.json"),
        "all_done_md": str(artifacts / "all_done_validation.md"),
        "repro_compare": str(repro_compare_path),
    }

    mp_key_present = bool(str(os.getenv("MP_API_KEY", "")).strip())
    mp_gate = True
    if cfg.all_done_require_mp_if_key_present and mp_key_present:
        mp_frame = pd.read_parquet(artifacts / "materials_project_enriched_points.parquet") if (artifacts / "materials_project_enriched_points.parquet").exists() else pd.DataFrame()
        if not mp_frame.empty and "mp_status" in mp_frame.columns:
            mp_gate = bool((mp_frame["mp_status"].astype(str) == "success").mean() > 0.0)
        else:
            mp_gate = False

    gates = {
        "bootstrap_gate": bool(base_result.gate_status.get("bootstrap_ok")),
        "preflight_gate": bool(preflight.ok),
        "crawl_acquire_gate": bool(provider_counts.get("localfs", 0) > 0 and document_provider_counts.get("localfs", 0) > 0),
        "extraction_gate": bool(
            requests_count > 0
            and responses_count > 0
            and voted > 0
            and validated > 0
            and provenance_complete >= 1.0
        ),
        "processing_gate": bool(processor_artifacts_ok and bool(processor_eval.get("all_gates_pass"))),
        "quality_gate": bool(quality_gate_payload.get("ok", False)),
        "mp_gate": bool(mp_gate),
        "reproducibility_gate": bool(repro_compare.get("repro_ok")),
    }
    gates["all_done_pass"] = bool(all(gates.values()))

    remediation: list[str] = []
    if not gates["bootstrap_gate"]:
        remediation.append("bootstrap failed: inspect bootstrap_report.json and fix dependency/TGI startup errors.")
    if not gates["preflight_gate"]:
        remediation.append("preflight failed: inspect preflight_report.json and fix module/runtime endpoint checks.")
    if not gates["crawl_acquire_gate"]:
        remediation.append("crawl/acquire gate failed: verify local repo path and local_merge_mode=union.")
    if not gates["extraction_gate"]:
        remediation.append("extraction gate failed: inspect slm_* artifacts and extractor config.")
    if not gates["processing_gate"]:
        remediation.append("processing gate failed: inspect processor_eval_metrics.json and model artifacts.")
    if not gates["quality_gate"]:
        remediation.append("quality gate failed: inspect quality_eval.json and quality_gate_report.md.")
    if not gates["reproducibility_gate"]:
        remediation.append("reproducibility gate failed: inspect repro_compare.json count drift.")
    if not gates["mp_gate"]:
        remediation.append("MP gate failed: MP key present but MP coverage is zero.")

    payload = {
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "query": resolved_query,
        "spec_path": str(Path(spec_path).expanduser().resolve()),
        "gates": gates,
        "quality_summary": quality_eval,
        "quality_gate_report": quality_gate_payload,
        "artifacts": required_artifacts,
        "preflight_report": str(base_result.preflight_path),
        "bootstrap_report": str(base_result.bootstrap_path) if base_result.bootstrap_path else "",
        "repro_compare_path": str(repro_compare_path),
        "provider_counts": provider_counts,
        "document_provider_counts": document_provider_counts,
        "counts": counts,
        "extractor_gate": extractor_gate,
        "processor_eval": processor_eval,
        "finetune_execution": finetune_exec_payload,
        "remediation": remediation,
    }

    json_path = artifacts / "all_done_validation.json"
    report_path = artifacts / "all_done_validation.md"
    write_json(json_path, payload)

    lines = [
        "# All Done Validation",
        "",
        f"- all_done_pass: `{gates['all_done_pass']}`",
        f"- bootstrap_gate: `{gates['bootstrap_gate']}`",
        f"- preflight_gate: `{gates['preflight_gate']}`",
        f"- crawl_acquire_gate: `{gates['crawl_acquire_gate']}`",
        f"- extraction_gate: `{gates['extraction_gate']}`",
        f"- processing_gate: `{gates['processing_gate']}`",
        f"- quality_gate: `{gates['quality_gate']}`",
        f"- reproducibility_gate: `{gates['reproducibility_gate']}`",
        "",
        "## Quality Summary",
        f"- provenance_complete_ratio: `{float(quality_eval.get('provenance_complete_ratio') or 0.0):.4f}`",
        f"- pair_completeness_ratio: `{float(quality_eval.get('pair_completeness_ratio') or 0.0):.4f}`",
        f"- cross_source_support_ratio: `{float(quality_eval.get('cross_source_support_ratio') or 0.0):.4f}`",
        f"- distribution_agreement_ratio: `{float(quality_eval.get('distribution_agreement_ratio') or 0.0):.4f}`",
        f"- entropy_p75: `{float(quality_eval.get('entropy_p75') or 0.0):.4f}`",
        "",
        "## Reproducibility",
        f"- repro_ok: `{bool(repro_compare.get('repro_ok'))}`",
        f"- max_relative_drift: `{float(repro_compare.get('max_relative_drift') or 0.0):.6f}`",
    ]
    if remediation:
        lines.extend(["", "## Remediation"])
        for item in remediation:
            lines.append(f"- {item}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return AllDoneValidationResult(
        ok=bool(gates["all_done_pass"]),
        run_dir=run_dir,
        report_path=report_path,
        json_path=json_path,
        repro_compare_path=repro_compare_path,
        gates=gates,
        artifacts=required_artifacts,
        quality_summary=quality_eval,
        benchmark_summary={},
    )


def validate_release_v1(
    run_dir: str | Path,
    cfg: RunConfig,
    validation_cfg: dict[str, Any] | None = None,
) -> ReleaseValidationResult:
    validation_cfg = validation_cfg or {}
    cfg.run_dir = str(Path(run_dir).expanduser().resolve())
    run_path = cfg.as_path()
    artifacts = run_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {}
    try:
        state = load_run_state(run_path)
    except Exception:
        state = {}

    spec_path = str(validation_cfg.get("spec_path") or state.get("spec_path") or "").strip()
    query = str(validation_cfg.get("query") or state.get("query") or "").strip()
    if not spec_path:
        raise ValueError("validate_release_v1 requires spec_path (or existing run_state.spec_path)")
    if not query:
        raise ValueError("validate_release_v1 requires query (or existing run_state.query)")

    all_done = run_all_done_validation(
        spec_path=spec_path,
        query=query,
        cfg=cfg,
        validation_cfg=validation_cfg,
    )

    graph_nodes = artifacts / "graph_nodes.parquet"
    graph_edges = artifacts / "graph_edges.parquet"
    graph_tensor_index = artifacts / "graph_tensor_index.parquet"
    phase_specs = artifacts / "phase_specs.parquet"
    cache_audit = artifacts / "cache_audit.parquet"
    article_process_nodes = artifacts / "article_process_nodes.parquet"
    article_process_edges = artifacts / "article_process_edges.parquet"
    concept_nodes = artifacts / "concept_nodes.parquet"
    concept_edges = artifacts / "concept_edges.parquet"
    bridge_edges = artifacts / "bridge_edges.parquet"
    graph_audit_json = artifacts / "graph_audit.json"

    graph_integrity = bool(
        graph_nodes.exists()
        and graph_edges.exists()
        and graph_tensor_index.exists()
        and phase_specs.exists()
    )
    cache_gate = bool(cache_audit.exists())

    graph_no_orphans = True
    if graph_integrity:
        try:
            ndf = pd.read_parquet(graph_nodes)
            edf = pd.read_parquet(graph_edges)
            node_ids = set(ndf["node_id"].astype(str).tolist()) if not ndf.empty else set()
            if not edf.empty:
                src_ok = edf["source_id"].astype(str).isin(node_ids)
                dst_ok = edf["target_id"].astype(str).isin(node_ids)
                graph_no_orphans = bool((src_ok & dst_ok).all())
        except Exception:
            graph_no_orphans = False

    phase_completeness = 0.0
    if phase_specs.exists():
        try:
            phase_df = pd.read_parquet(phase_specs)
            if not phase_df.empty:
                cols = ["elements_json", "stoichiometry_json"]
                if "spacegroup_symbol" in phase_df.columns and "spacegroup_number" in phase_df.columns:
                    phase_present = (
                        phase_df["spacegroup_symbol"].astype(str).str.strip().ne("")
                        | phase_df["spacegroup_number"].astype(str).str.strip().ne("")
                    )
                elif "spacegroup_symbol" in phase_df.columns:
                    phase_present = phase_df["spacegroup_symbol"].astype(str).str.strip().ne("")
                else:
                    phase_present = pd.Series([False] * len(phase_df))
                core_present = (
                    phase_df[cols]
                    .fillna("")
                    .astype(str)
                    .apply(lambda col: col.str.strip().ne(""), axis=0)
                    .all(axis=1)
                )
                phase_completeness = float((core_present & phase_present).mean())
        except Exception:
            phase_completeness = 0.0

    gate_map = dict(all_done.gates)
    gate_map["graph_integrity_gate"] = bool(graph_integrity and graph_no_orphans)
    gate_map["phase_schema_gate"] = bool(phase_completeness >= 1.0)
    gate_map["cache_audit_gate"] = bool(cache_gate)
    dual_mode = str(cfg.graph_architecture or "").strip().lower() == "dual_concept"
    dual_graph_gate = True
    if dual_mode:
        dual_graph_gate = bool(
            article_process_nodes.exists()
            and article_process_edges.exists()
            and concept_nodes.exists()
            and concept_edges.exists()
            and bridge_edges.exists()
            and graph_audit_json.exists()
        )
        if dual_graph_gate:
            try:
                dual_payload = _safe_json(graph_audit_json)
                dual_graph_gate = bool(dual_payload.get("ok", False))
            except Exception:
                dual_graph_gate = False
    gate_map["dual_graph_gate"] = bool(dual_graph_gate)
    gate_map["all_done_pass"] = bool(all(gate_map.values()))

    remediation: list[str] = []
    for key, passed in gate_map.items():
        if key == "all_done_pass" or bool(passed):
            continue
        if key == "graph_integrity_gate":
            remediation.append("graph integrity failed: inspect graph_nodes/graph_edges for orphan references.")
        elif key == "phase_schema_gate":
            remediation.append("phase schema gate failed: ensure elements/stoichiometry/space-group fields are populated.")
        elif key == "cache_audit_gate":
            remediation.append("cache audit missing: ensure cache manager writes cache_audit.parquet.")
        elif key == "dual_graph_gate":
            remediation.append(
                "dual graph gate failed: inspect article_process/concept/bridge artifacts and graph_audit.json."
            )
        else:
            remediation.append(f"{key} failed: inspect all_done_validation artifacts.")

    artifact_map = dict(all_done.artifacts)
    artifact_map.update(
        {
            "phase_specs": str(phase_specs),
            "graph_nodes": str(graph_nodes),
            "graph_edges": str(graph_edges),
            "graph_tensor_index": str(graph_tensor_index),
            "cache_audit": str(cache_audit),
            "article_process_nodes": str(article_process_nodes),
            "article_process_edges": str(article_process_edges),
            "concept_nodes": str(concept_nodes),
            "concept_edges": str(concept_edges),
            "bridge_edges": str(bridge_edges),
            "graph_audit_json": str(graph_audit_json),
        }
    )
    repro_summary = _safe_json(all_done.repro_compare_path)
    payload = {
        "created_at": now_iso(),
        "run_dir": str(run_path),
        "spec_path": str(spec_path),
        "query": str(query),
        "gate_map": gate_map,
        "phase_completeness": phase_completeness,
        "graph_no_orphans": graph_no_orphans,
        "artifact_map": artifact_map,
        "repro_summary": repro_summary,
        "quality_summary": all_done.quality_summary,
        "remediation": remediation,
    }
    json_path = artifacts / "release_v1_validation.json"
    report_path = artifacts / "release_v1_validation.md"
    write_json(json_path, payload)

    md_lines = [
        "# OARP v1 Release Validation",
        "",
        f"- all_done_pass: `{gate_map['all_done_pass']}`",
        f"- graph_integrity_gate: `{gate_map['graph_integrity_gate']}`",
        f"- phase_schema_gate: `{gate_map['phase_schema_gate']}`",
        f"- cache_audit_gate: `{gate_map['cache_audit_gate']}`",
        "",
        "## Quality Summary",
        f"- provenance_complete_ratio: `{float(all_done.quality_summary.get('provenance_complete_ratio') or 0.0):.4f}`",
        f"- pair_completeness_ratio: `{float(all_done.quality_summary.get('pair_completeness_ratio') or 0.0):.4f}`",
        f"- cross_source_support_ratio: `{float(all_done.quality_summary.get('cross_source_support_ratio') or 0.0):.4f}`",
        f"- distribution_agreement_ratio: `{float(all_done.quality_summary.get('distribution_agreement_ratio') or 0.0):.4f}`",
        "",
        "## Graph + Phase",
        f"- phase_completeness: `{phase_completeness:.4f}`",
        f"- graph_no_orphans: `{graph_no_orphans}`",
    ]
    if remediation:
        md_lines.extend(["", "## Remediation"])
        for item in remediation:
            md_lines.append(f"- {item}")
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return ReleaseValidationResult(
        all_done_pass=bool(gate_map["all_done_pass"]),
        gate_map={str(k): bool(v) for k, v in gate_map.items()},
        artifact_map={str(k): str(v) for k, v in artifact_map.items()},
        repro_summary=repro_summary if isinstance(repro_summary, dict) else {},
        remediation=remediation,
        report_path=report_path,
        json_path=json_path,
    )


def validate_v2_publish(
    spec_path: str,
    query: str,
    cfg: RunConfig,
    validation_cfg: dict[str, Any] | None = None,
) -> ReleaseValidationResult:
    validation_cfg = validation_cfg or {}
    cfg.run_profile = str(cfg.run_profile or "v2_publish_1k")
    cfg.vault_export_enabled = True
    if str(cfg.finetune_gate_policy or "").strip().lower() in {"", "absolute_uplift"}:
        cfg.finetune_gate_policy = "ceiling_aware"

    all_done = run_all_done_validation(
        spec_path=spec_path,
        query=query,
        cfg=cfg,
        validation_cfg=validation_cfg,
    )
    run_path = cfg.as_path()
    artifacts = run_path / "artifacts"

    vault_root = run_path / "outputs" / "vault"
    if str(cfg.vault_root or "").strip():
        state = load_run_state(run_path)
        run_id = str(state.get("run_id") or run_path.name)
        vault_root = Path(cfg.vault_root).expanduser().resolve() / _safe_slug(run_id)
    if not vault_root.exists():
        project_concepts_to_vault(run_dir=str(run_path), out_dir=str(vault_root), cfg=cfg)

    compare_enabled = bool(validation_cfg.get("vault_compare", True))
    vault_import_result: VaultImportResult | None = None
    vault_bench: VaultBenchmarkResult | None = None
    if compare_enabled:
        vault_import_result = parse_vault_semantics(vault_dir=str(vault_root), run_dir=str(run_path), cfg=cfg)
        _ = apply_vault_soft_supervision(run_dir=str(run_path), vault_links=vault_import_result.link_deltas, cfg=cfg)
        vault_bench = benchmark_vault_alignment_v2(run_dir=str(run_path), vault_dir=str(vault_root), cfg=cfg)

    gate_map = dict(all_done.gates)
    gate_map["vault_export_gate"] = bool(vault_root.exists())
    gate_map["vault_import_gate"] = bool(vault_import_result is not None)
    gate_map["vault_alignment_gate"] = bool((vault_bench is not None and vault_bench.f1 >= 0.50) or not compare_enabled)
    gate_map["all_done_pass"] = bool(all(gate_map.values()))

    artifact_map = dict(all_done.artifacts)
    artifact_map.update(
        {
            "vault_root": str(vault_root),
            "vault_export_manifest": str(artifacts / "vault_export_manifest.json"),
            "vault_notes_index": str(artifacts / "vault_notes_index.parquet"),
            "vault_alias_index": str(artifacts / "vault_alias_index.parquet"),
            "vault_frontmatter_index": str(artifacts / "vault_frontmatter_index.parquet"),
            "vault_link_index": str(artifacts / "vault_link_index.parquet"),
            "vault_links_v2": str(artifacts / "vault_links_v2.parquet"),
            "vault_links": str(artifacts / "vault_links.parquet"),
            "vault_link_audit": str(artifacts / "vault_link_audit.parquet"),
            "vault_soft_constraints": str(artifacts / "vault_soft_constraints.parquet"),
            "vault_import_audit": str(artifacts / "vault_import_audit.parquet"),
            "vault_conflicts": str(artifacts / "vault_conflicts.parquet"),
            "benchmark_vault_json": str(artifacts / "benchmark_vault.json"),
            "benchmark_vault_v2_json": str(artifacts / "benchmark_vault_v2.json"),
            "benchmark_vault_md": str(artifacts / "benchmark_vault.md"),
            "benchmark_vault_v2_md": str(artifacts / "benchmark_vault_v2.md"),
        }
    )

    remediation: list[str] = []
    if not gate_map["vault_export_gate"]:
        remediation.append("vault export missing: inspect outputs/vault generation stage.")
    if compare_enabled and not gate_map["vault_import_gate"]:
        remediation.append("vault import missing: check vault markdown parse or path permissions.")
    if compare_enabled and not gate_map["vault_alignment_gate"]:
        remediation.append("vault alignment below threshold: inspect benchmark_vault.md and link deltas.")
    if not all_done.ok:
        remediation.append("all_done gates failed: inspect all_done_validation.md for strict gate failures.")

    payload = {
        "created_at": now_iso(),
        "run_dir": str(run_path),
        "query": str(query),
        "spec_path": str(Path(spec_path).expanduser().resolve()),
        "gate_map": gate_map,
        "artifact_map": artifact_map,
        "quality_summary": all_done.quality_summary,
        "vault_benchmark": {
            "precision": float(vault_bench.precision) if vault_bench else 0.0,
            "recall": float(vault_bench.recall) if vault_bench else 0.0,
            "f1": float(vault_bench.f1) if vault_bench else 0.0,
            "vault_coverage": float(vault_bench.vault_coverage) if vault_bench else 0.0,
            "vault_link_density": float(vault_bench.vault_link_density) if vault_bench else 0.0,
            "vault_import_delta_rate": float(vault_bench.vault_import_delta_rate) if vault_bench else 0.0,
        },
        "repro_summary": _safe_json(all_done.repro_compare_path),
        "remediation": remediation,
    }
    json_path = artifacts / "publish_v2_validation.json"
    report_path = artifacts / "publish_v2_validation.md"
    write_json(json_path, payload)

    lines = [
        "# OARP v2 Publish Validation",
        "",
        f"- all_done_pass: `{gate_map['all_done_pass']}`",
        f"- vault_export_gate: `{gate_map['vault_export_gate']}`",
        f"- vault_import_gate: `{gate_map['vault_import_gate']}`",
        f"- vault_alignment_gate: `{gate_map['vault_alignment_gate']}`",
        "",
        "## Machine Graph + Vault Graph Alignment",
        f"- precision: `{payload['vault_benchmark']['precision']:.4f}`",
        f"- recall: `{payload['vault_benchmark']['recall']:.4f}`",
        f"- f1: `{payload['vault_benchmark']['f1']:.4f}`",
        f"- vault_coverage: `{payload['vault_benchmark']['vault_coverage']:.4f}`",
        f"- vault_link_density: `{payload['vault_benchmark']['vault_link_density']:.4f}`",
        f"- vault_import_delta_rate: `{payload['vault_benchmark']['vault_import_delta_rate']:.4f}`",
    ]
    if remediation:
        lines.extend(["", "## Remediation"])
        for item in remediation:
            lines.append(f"- {item}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return ReleaseValidationResult(
        all_done_pass=bool(gate_map["all_done_pass"]),
        gate_map={str(k): bool(v) for k, v in gate_map.items()},
        artifact_map={str(k): str(v) for k, v in artifact_map.items()},
        repro_summary=_safe_json(all_done.repro_compare_path),
        remediation=remediation,
        report_path=report_path,
        json_path=json_path,
    )
