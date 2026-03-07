from __future__ import annotations

import hashlib
import json
import os
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
    FullWorkflowResult,
    GraphAuditResult,
    GraphBuildResult,
    MPEvidenceSet,
    OutputBundle,
    ProcessorDataset,
    ProcessorModelBundle,
    ProcessorModelRef,
    ReleaseValidationResult,
    RunConfig,
    RunResult,
    ValidatedEvidence,
)
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
    processing,
    recipe,
    render,
    validation,
)
from .benchmark import run_benchmark
from .workflow import bootstrap_runtime, preflight_strict


def _run_id(topic_id: str, query: str) -> str:
    digest = hashlib.sha1(f"{topic_id}|{query}".encode("utf-8", errors="replace")).hexdigest()[:10]
    timestamp = now_iso().replace(":", "").replace("-", "")
    return f"run-{timestamp}-{digest}"


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
            "python_exec": cfg.python_exec,
            "tgi_docker_image": cfg.tgi_docker_image,
            "tgi_model_id": cfg.tgi_model_id,
            "tgi_platform": cfg.tgi_platform,
            "tgi_mode": cfg.tgi_mode,
            "tgi_port": cfg.tgi_port,
            "tgi_health_path": cfg.tgi_health_path,
            "tgi_generate_path": cfg.tgi_generate_path,
            "workflow_profile": cfg.workflow_profile,
            "use_bootstrapped_venv": cfg.use_bootstrapped_venv,
            "already_bootstrapped": cfg.already_bootstrapped,
            "all_done_repro_runs": cfg.all_done_repro_runs,
            "all_done_max_runtime_sec": cfg.all_done_max_runtime_sec,
            "all_done_require_mp_if_key_present": cfg.all_done_require_mp_if_key_present,
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


def evaluate_extractor_quality(run_dir: str, gold_dir: str, cfg: RunConfig) -> ExtractionCalibration:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = run_path / "artifacts"
    state = load_run_state(run_path)
    spec = load_topic_spec(str(state.get("spec_path") or ""))
    eval_out = artifacts / "extractor_eval"
    eval_out.mkdir(parents=True, exist_ok=True)

    bench = run_benchmark(
        spec=spec,
        gold_dir=gold_dir,
        out_dir=eval_out,
        run_dir=run_path,
        strict_gold=True,
    )

    validated_path = artifacts / "validated_points.parquet"
    validated = pd.read_parquet(validated_path) if validated_path.exists() else pd.DataFrame()
    if validated.empty:
        provenance_complete_ratio = 0.0
        context_precision_proxy = 0.0
    else:
        required = ["citation_url", "snippet", "locator"]
        complete_mask = validated[required].fillna("").astype(str).apply(lambda col: col.str.strip().ne(""), axis=0).all(axis=1)
        provenance_complete_ratio = float(complete_mask.mean())
        context_cols = ["substrate_material", "substrate_orientation", "doping_state", "alloy_state"]
        if all(col in validated.columns for col in context_cols):
            context_mask = validated[context_cols].fillna("").astype(str).apply(lambda col: col.str.strip().ne(""), axis=0).all(axis=1)
            context_precision_proxy = float(context_mask.mean())
        else:
            context_precision_proxy = 0.0

    payload = {
        "tuple_precision": float(bench.precision),
        "tuple_recall": float(bench.recall),
        "tuple_f1": float(bench.f1),
        "provenance_complete_ratio": float(provenance_complete_ratio),
        "context_precision_proxy": float(context_precision_proxy),
        "gate_tuple_precision": bool(float(bench.precision) >= 0.85),
        "gate_tuple_recall": bool(float(bench.recall) >= 0.65),
        "gate_provenance_complete": bool(float(provenance_complete_ratio) >= 1.0),
        "gate_context_precision": bool(float(context_precision_proxy) >= 0.80),
    }
    payload["all_gates_pass"] = bool(
        payload["gate_tuple_precision"]
        and payload["gate_tuple_recall"]
        and payload["gate_provenance_complete"]
        and payload["gate_context_precision"]
    )

    slm_eval_path = artifacts / "slm_eval_metrics.json"
    write_json(slm_eval_path, payload)

    calibration_frame = pd.DataFrame(
        [
            {
                "bin_idx": 0,
                "bin_left": 0.0,
                "bin_right": 1.0,
                "point_count": int(len(validated)),
                "mean_predicted": float(bench.precision),
                "empirical_precision": float(bench.precision),
                "ece_component": 0.0,
                "tuple_recall": float(bench.recall),
                "tuple_f1": float(bench.f1),
                "provenance_complete_ratio": float(provenance_complete_ratio),
                "context_precision_proxy": float(context_precision_proxy),
                "all_gates_pass": bool(payload["all_gates_pass"]),
            }
        ]
    )
    calibration_path = artifacts / "extraction_calibration.parquet"
    calibration_frame.to_parquet(calibration_path, index=False)
    return ExtractionCalibration(frame=calibration_frame, summary=payload, path=calibration_path)


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
            "python_exec": cfg.python_exec,
            "tgi_docker_image": cfg.tgi_docker_image,
            "tgi_model_id": cfg.tgi_model_id,
            "tgi_platform": cfg.tgi_platform,
            "tgi_mode": cfg.tgi_mode,
            "tgi_port": cfg.tgi_port,
            "tgi_health_path": cfg.tgi_health_path,
            "tgi_generate_path": cfg.tgi_generate_path,
            "workflow_profile": cfg.workflow_profile,
            "use_bootstrapped_venv": cfg.use_bootstrapped_venv,
            "already_bootstrapped": cfg.already_bootstrapped,
            "all_done_repro_runs": cfg.all_done_repro_runs,
            "all_done_max_runtime_sec": cfg.all_done_max_runtime_sec,
            "all_done_require_mp_if_key_present": cfg.all_done_require_mp_if_key_present,
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
            "benchmark": {
                "status": "run `oarp benchmark --gold-context <dir>` to append context benchmark artifacts",
                "context_precision_gate": float(cfg.context_precision_gate),
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


def run_full_workflow(spec_path: str, query: str, cfg: RunConfig) -> FullWorkflowResult:
    if bool(cfg.cpu_strict_profile):
        _apply_cpu_thread_clamps(cfg)
    if bool(cfg.cpu_probe):
        _capture_cpu_env(cfg)
    spec = load_topic_spec(spec_path)
    resolved_query = ensure_query(spec, query)
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

    mp_path = artifacts / "materials_project_enriched_points.parquet"
    mp_data = pd.read_parquet(mp_path) if mp_path.exists() else None
    dataset = build_processing_dataset(
        points=knowledge_bundle.phase_events_path,
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
    processor_attempts = max(1, int(cfg.processor_max_loop))
    for attempt in range(1, processor_attempts + 1):
        started = time.perf_counter()
        try:
            gnn_ref = train_gnn_base(dataset, cfg)
            tabular_ref = train_tabular_head(dataset, gnn_ref, cfg)
            finetune_ref = finetune_nisi_sub200(tabular_ref, dataset, cfg)
            processor_eval = evaluate_processor(run_dir=str(cfg.as_path()), cfg=cfg)
            processor_models = {
                "gnn_base": gnn_ref.model_dir,
                "tabular_head": tabular_ref.model_dir,
                "finetune_nisi_sub200": finetune_ref.model_dir,
            }
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
        "provider_counts": provider_counts,
        "document_provider_counts": doc_provider_counts,
        "stage_metrics_path": str(full_stage_path),
        "stage_metrics": stage_rows,
        "references": {
            "run_metrics": str(run_result.metrics_path),
            "preflight_report": str(preflight.report_path),
            "bootstrap_report": str(bootstrap_result.report_path) if bootstrap_result is not None else "",
            "processor_eval_metrics": str(artifacts / "processor_eval_metrics.json"),
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


def _benchmark_summary_from_files(bench_out: Path) -> dict[str, Any]:
    base_json = _safe_json(bench_out / "benchmark.json")
    context_json = _safe_json(bench_out / "benchmark_context.json")
    shadow_json = _safe_json(bench_out / "benchmark_shadow.json")
    mp_json = _safe_json(bench_out / "benchmark_mp.json")
    return {
        "precision": float(base_json.get("precision") or 0.0),
        "recall": float(base_json.get("recall") or 0.0),
        "f1": float(base_json.get("f1") or 0.0),
        "threshold_met": bool(base_json.get("threshold_met")),
        "context_threshold_met": bool(context_json.get("context_threshold_met"))
        if context_json
        else True,
        "context_completeness": float(context_json.get("condition_completeness") or 0.0)
        if context_json
        else 0.0,
        "context_precision": float(context_json.get("condition_precision") or 0.0)
        if context_json
        else 0.0,
        "shadow_precision": float(shadow_json.get("precision") or 0.0) if shadow_json else 0.0,
        "mp_coverage": float(mp_json.get("mp_coverage") or 0.0) if mp_json else 0.0,
    }


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
    bench_out = artifacts / "all_done_benchmark"
    bench_out.mkdir(parents=True, exist_ok=True)

    spec = load_topic_spec(spec_path)
    resolved_query = ensure_query(spec, query)

    preflight = preflight_strict(cfg, python_exec=(cfg.python_exec or None), check_tgi_generate=True)
    _check_budget("preflight")

    base_result = run_full_workflow(spec_path=spec_path, query=resolved_query, cfg=cfg)
    _check_budget("run_full")

    gold_dir = str(validation_cfg.get("gold_dir") or "").strip()
    shadow_gold_dir = str(validation_cfg.get("shadow_gold_dir") or "").strip()
    gold_context_dir = str(validation_cfg.get("gold_context_dir") or "").strip()
    precision_gate = float(validation_cfg.get("precision_gate") or 0.80)
    context_completeness_gate = float(
        validation_cfg.get("context_completeness_gate") or cfg.context_completeness_gate
    )
    context_precision_gate = float(
        validation_cfg.get("context_precision_gate") or cfg.context_precision_gate
    )

    benchmark_summary: dict[str, Any] = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "threshold_met": False,
        "context_threshold_met": True,
        "context_completeness": 0.0,
        "context_precision": 0.0,
        "shadow_precision": 0.0,
        "mp_coverage": 0.0,
    }
    if gold_dir:
        _ = run_benchmark(
            spec=spec,
            gold_dir=gold_dir,
            out_dir=bench_out,
            run_dir=run_dir,
            precision_gate=precision_gate,
            shadow_gold_dir=shadow_gold_dir or None,
            strict_gold=True,
            gold_context_dir=gold_context_dir or None,
            context_completeness_gate=context_completeness_gate,
            context_precision_gate=context_precision_gate,
        )
        benchmark_summary = _benchmark_summary_from_files(bench_out)
    _check_budget("benchmark")

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

    processor_artifacts_ok = all(
        [
            _artifact_exists(artifacts / "models" / "gnn_base" / "gnn_model.pt"),
            _artifact_exists(artifacts / "models" / "tabular_head" / "processor_model.pkl"),
            _artifact_exists(artifacts / "models" / "finetune_nisi_sub200" / "processor_model.pkl"),
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
        "all_done_json": str(artifacts / "all_done_validation.json"),
        "all_done_md": str(artifacts / "all_done_validation.md"),
        "repro_compare": str(repro_compare_path),
    }

    benchmark_gate = bool(benchmark_summary.get("threshold_met"))
    if gold_context_dir:
        benchmark_gate = benchmark_gate and bool(benchmark_summary.get("context_threshold_met"))

    mp_key_present = bool(str(os.getenv("MP_API_KEY", "")).strip())
    mp_gate = True
    if cfg.all_done_require_mp_if_key_present and mp_key_present:
        mp_gate = float(benchmark_summary.get("mp_coverage") or 0.0) > 0.0

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
        "benchmark_gate": bool(benchmark_gate),
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
    if not gates["benchmark_gate"]:
        remediation.append("benchmark gate failed: inspect benchmark.json/context report and tighten extraction quality.")
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
        "benchmark_summary": benchmark_summary,
        "artifacts": required_artifacts,
        "preflight_report": str(base_result.preflight_path),
        "bootstrap_report": str(base_result.bootstrap_path) if base_result.bootstrap_path else "",
        "repro_compare_path": str(repro_compare_path),
        "provider_counts": provider_counts,
        "document_provider_counts": document_provider_counts,
        "counts": counts,
        "extractor_gate": extractor_gate,
        "processor_eval": processor_eval,
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
        f"- benchmark_gate: `{gates['benchmark_gate']}`",
        f"- reproducibility_gate: `{gates['reproducibility_gate']}`",
        "",
        "## Benchmark Summary",
        f"- precision: `{benchmark_summary.get('precision', 0.0):.4f}`",
        f"- recall: `{benchmark_summary.get('recall', 0.0):.4f}`",
        f"- f1: `{benchmark_summary.get('f1', 0.0):.4f}`",
        f"- context_threshold_met: `{bool(benchmark_summary.get('context_threshold_met'))}`",
        f"- mp_coverage: `{benchmark_summary.get('mp_coverage', 0.0):.4f}`",
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
        benchmark_summary=benchmark_summary,
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
        "benchmark_summary": all_done.benchmark_summary,
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
        "## Benchmark Summary",
        f"- precision: `{float(all_done.benchmark_summary.get('precision') or 0.0):.4f}`",
        f"- recall: `{float(all_done.benchmark_summary.get('recall') or 0.0):.4f}`",
        f"- f1: `{float(all_done.benchmark_summary.get('f1') or 0.0):.4f}`",
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
