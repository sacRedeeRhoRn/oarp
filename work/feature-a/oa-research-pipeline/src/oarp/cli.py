from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from oarp import consensus as consensus_stage
from oarp import extraction as extraction_stage
from oarp import mp_enrichment as mp_enrichment_stage
from oarp import processing as processing_stage
from oarp import render as render_stage
from oarp import validation as validation_stage
from oarp.acquisition import acquire
from oarp.adapters import build_agents_inc_handoff
from oarp.benchmark import run_benchmark, run_extraction_benchmark, run_processing_benchmark
from oarp.discovery import discover
from oarp.jobs import create_retry_job
from oarp.knowledge import build_knowledge
from oarp.models import ConsensusSet, ProcessorModelRef, RunConfig
from oarp.pipeline import (
    apply_vault_soft_supervision,
    run_all_done_validation,
    run_full_workflow,
    run_pipeline,
    validate_release_v1,
    validate_v2_publish,
)
from oarp.pipeline import (
    audit_dual_graph as audit_dual_graph_api,
)
from oarp.pipeline import (
    benchmark_vault_alignment as benchmark_vault_alignment_api,
)
from oarp.pipeline import (
    build_article_process_graph as build_article_process_graph_api,
)
from oarp.pipeline import (
    build_bridge_edges as build_bridge_edges_api,
)
from oarp.pipeline import (
    build_global_concept_graph as build_global_concept_graph_api,
)
from oarp.pipeline import (
    build_processing_dataset as build_processing_dataset_api,
)
from oarp.pipeline import evaluate_processor as evaluate_processor_api
from oarp.pipeline import export_obsidian_vault as export_obsidian_vault_api
from oarp.pipeline import finetune_nisi_sub200 as finetune_nisi_sub200_api
from oarp.pipeline import (
    finetune_processor_for_system as finetune_processor_for_system_api,
)
from oarp.pipeline import import_obsidian_vault as import_obsidian_vault_api
from oarp.pipeline import train_gnn_base as train_gnn_base_api
from oarp.pipeline import train_tabular_head as train_tabular_head_api
from oarp.pipeline import (
    train_universal_processor as train_universal_processor_api,
)
from oarp.recipe import generate_recipes
from oarp.runtime import load_run_state
from oarp.service import start_service
from oarp.service_api import (
    dump_job_payload,
    get_job_artifacts,
    get_job_status,
    run_pending_jobs,
    submit_recipe_generation_job,
    submit_research_index_job,
)
from oarp.service_models import RecipeGenerateRequest, ServiceConfig
from oarp.topic_spec import ensure_query, load_topic_spec
from oarp.workflow import bootstrap_runtime, preflight_strict, tgi_start, tgi_status, tgi_stop


def _print_payload(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_provider_caps(raw: str) -> dict[str, int]:
    text = str(raw or "").strip()
    if not text:
        return {}
    out: dict[str, int] = {}
    for item in text.split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        provider, cap = part.split(":", 1)
        key = provider.strip().lower()
        try:
            value = int(cap.strip())
        except Exception:
            continue
        if key and value > 0:
            out[key] = value
    return out


def _add_mp_flags(parser: argparse.ArgumentParser, *, include_enabled: bool = True) -> None:
    if include_enabled:
        parser.add_argument("--mp-enabled", dest="mp_enabled", action="store_true", default=True)
        parser.add_argument("--no-mp-enabled", dest="mp_enabled", action="store_false")
    parser.add_argument("--mp-mode", choices=["interpreter", "hybrid", "hard"], default="interpreter")
    parser.add_argument("--mp-scope", choices=["summary_thermo", "summary_only", "broad"], default="summary_thermo")
    parser.add_argument("--mp-on-demand", dest="mp_on_demand", action="store_true", default=True)
    parser.add_argument("--disable-mp-on-demand", dest="mp_on_demand", action="store_false")
    parser.add_argument("--mp-query-workers", type=int, default=4)
    parser.add_argument("--mp-timeout-sec", type=float, default=20.0)
    parser.add_argument("--mp-max-queries", type=int, default=20000)
    parser.add_argument("--mp-cache-path", default="")


def _add_local_repo_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--local-repo-path",
        dest="local_repo_paths",
        action="append",
        default=[],
        help="Repeatable local repository root path for PDF ingestion.",
    )
    parser.add_argument("--local-repo-recursive", dest="local_repo_recursive", action="store_true", default=True)
    parser.add_argument("--no-local-repo-recursive", dest="local_repo_recursive", action="store_false")
    parser.add_argument("--local-file-glob", default="*.pdf")
    parser.add_argument(
        "--local-merge-mode",
        choices=["union", "local_only", "online_only"],
        default="union",
    )
    parser.add_argument("--local-max-files", type=int, default=0)
    parser.add_argument("--local-require-readable", dest="local_require_readable", action="store_true", default=True)
    parser.add_argument("--no-local-require-readable", dest="local_require_readable", action="store_false")


def _add_strict_workflow_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strict-full-workflow", dest="strict_full_workflow", action="store_true", default=True)
    parser.add_argument("--no-strict-full-workflow", dest="strict_full_workflow", action="store_false")
    parser.add_argument("--auto-bootstrap", dest="auto_bootstrap", action="store_true", default=True)
    parser.add_argument("--no-auto-bootstrap", dest="auto_bootstrap", action="store_false")
    parser.add_argument("--python-exec", default="")
    parser.add_argument(
        "--tgi-docker-image",
        default="ghcr.io/huggingface/text-generation-inference:2.3.1",
    )
    parser.add_argument("--tgi-model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--tgi-platform", choices=["auto", "linux/amd64", "linux/arm64"], default="auto")
    parser.add_argument("--tgi-mode", choices=["docker", "external"], default="docker")
    parser.add_argument("--tgi-port", type=int, default=8080)
    parser.add_argument("--tgi-health-path", default="/health")
    parser.add_argument("--tgi-generate-path", default="/generate")
    parser.add_argument("--use-bootstrapped-venv", dest="use_bootstrapped_venv", action="store_true", default=True)
    parser.add_argument("--no-use-bootstrapped-venv", dest="use_bootstrapped_venv", action="store_false")
    parser.add_argument("--already-bootstrapped", action="store_true", default=False)
    parser.add_argument("--all-done-repro-runs", type=int, default=2)
    parser.add_argument("--all-done-max-runtime-sec", type=int, default=43200)
    parser.add_argument(
        "--all-done-require-mp-if-key-present",
        dest="all_done_require_mp_if_key_present",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-all-done-require-mp-if-key-present",
        dest="all_done_require_mp_if_key_present",
        action="store_false",
    )
    parser.add_argument("--workflow-profile", default="strict_full")


def _add_v2_storage_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--storage-root", default="/Volumes/Moon Seo/oarp_v2")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--vault-root", default="")
    parser.add_argument("--run-profile", choices=["", "v2_fast", "v2_strict", "v2_publish_1k"], default="")


def _add_v2_runtime_hardening_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tgi-port-policy", choices=["reuse_or_allocate", "fixed"], default="reuse_or_allocate")
    parser.add_argument("--tgi-port-range", default="8080-8090")
    parser.add_argument("--tgi-reuse-existing", dest="tgi_reuse_existing", action="store_true", default=True)
    parser.add_argument("--no-tgi-reuse-existing", dest="tgi_reuse_existing", action="store_false")
    parser.add_argument("--slm-max-chunks-per-doc", type=int, default=0)
    parser.add_argument("--slm-max-doc-chars", type=int, default=0)
    parser.add_argument("--slm-response-cache", choices=["on", "off"], default="on")
    parser.add_argument("--extract-stage-timeout-sec", type=int, default=0)


def _add_v2_finetune_policy_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--finetune-gate-policy",
        choices=["absolute_uplift", "ceiling_aware", "non_degradation"],
        default="absolute_uplift",
    )
    parser.add_argument("--finetune-ceiling-threshold", type=float, default=0.95)
    parser.add_argument("--finetune-min-slice-rows", type=int, default=20)
    parser.add_argument("--finetune-min-support-articles", type=int, default=5)


def _add_v1_release_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phase-schema-version", default="v1")
    parser.add_argument("--graph-core-mode", default="per_article_typed")
    parser.add_argument("--phase-require-elements", dest="phase_require_elements", action="store_true", default=True)
    parser.add_argument("--no-phase-require-elements", dest="phase_require_elements", action="store_false")
    parser.add_argument("--phase-require-stoich", dest="phase_require_stoich", action="store_true", default=True)
    parser.add_argument("--no-phase-require-stoich", dest="phase_require_stoich", action="store_false")
    parser.add_argument("--phase-require-spacegroup", dest="phase_require_spacegroup", action="store_true", default=True)
    parser.add_argument("--no-phase-require-spacegroup", dest="phase_require_spacegroup", action="store_false")
    parser.add_argument("--cache-mode", choices=["run_local", "hybrid_local_shared"], default="run_local")
    parser.add_argument("--shared-cache-root", default="")
    parser.add_argument("--cache-read-only", action="store_true", default=False)
    parser.add_argument("--cache-ttl-hours", type=int, default=168)
    parser.add_argument("--cpu-strict-profile", action="store_true", default=False)
    parser.add_argument("--cpu-max-threads", type=int, default=4)
    parser.add_argument("--cpu-probe", action="store_true", default=False)


def _add_dual_graph_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph-architecture", choices=["legacy_v1", "dual_concept"], default="legacy_v1")
    parser.add_argument("--concept-ontology-profile", choices=["maximal"], default="maximal")
    parser.add_argument("--bridge-weight-policy", choices=["deterministic_v1"], default="deterministic_v1")
    parser.add_argument("--bridge-weight-threshold", type=float, default=0.15)
    parser.add_argument("--concept-gates", default="film,substrate,dopant,precursor,method,thermo,electronic")


def _add_system_finetune_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--system-finetune-dataset",
        default="",
        help=(
            "Optional system-specific dataset path for fine-tuning/eval. "
            "Accepts a parquet/csv file, or a run directory containing artifacts/processor_training_rows.parquet."
        ),
    )
    parser.add_argument("--system-eval-holdout-ratio", type=float, default=0.30)


def _load_spec_from_run(run_dir: str | Path):
    state = load_run_state(run_dir)
    spec_path = state.get("spec_path")
    if not spec_path:
        raise ValueError("run state missing spec_path")
    spec = load_topic_spec(spec_path)
    return spec, state


def _apply_run_profile_defaults(cfg: RunConfig) -> None:
    profile = str(cfg.run_profile or "").strip().lower()
    if not profile:
        return
    if profile in {"v2_fast", "v2_strict", "v2_publish_1k"}:
        cfg.graph_architecture = "dual_concept"
        cfg.extractor_mode = "slm_tgi_required"
        cfg.vault_export_enabled = True
        cfg.finetune_gate_policy = "ceiling_aware"
    if profile == "v2_fast":
        cfg.max_downloads = max(120, int(cfg.max_downloads))
        cfg.slm_max_chunks_per_doc = max(0, int(cfg.slm_max_chunks_per_doc or 8))
    if profile == "v2_strict":
        cfg.strict_full_workflow = True
        cfg.cpu_strict_profile = True
    if profile == "v2_publish_1k":
        cfg.strict_full_workflow = True
        cfg.cpu_strict_profile = True
        cfg.max_downloads = max(1000, int(cfg.max_downloads))
        cfg.extract_workers = max(4, int(cfg.extract_workers))
        cfg.acquire_workers = max(8, int(cfg.acquire_workers))
        cfg.vault_import_enabled = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oarp", description="OA research extraction pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run full pipeline")
    p_run.add_argument("--spec", required=True)
    p_run.add_argument("--query", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--min-discovery-score", type=float, default=0.0)
    p_run.add_argument("--plugin", default="")
    p_run.add_argument("--max-pages-per-provider", type=int, default=60)
    p_run.add_argument("--max-discovered-records", type=int, default=100000)
    p_run.add_argument("--saturation-window-pages", type=int, default=8)
    p_run.add_argument("--saturation-min-yield", type=float, default=0.03)
    p_run.add_argument("--min-pages-before-saturation", type=int, default=20)
    p_run.add_argument("--resume-discovery", action="store_true")
    p_run.add_argument("--max-downloads", type=int, default=200)
    p_run.add_argument("--acquire-workers", type=int, default=8)
    p_run.add_argument("--extract-workers", type=int, default=4)
    p_run.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_run.add_argument("--disable-english-first", dest="english_first", action="store_false")
    p_run.add_argument("--per-provider-cap", default="")
    _add_local_repo_flags(p_run)
    p_run.add_argument("--require-fulltext-mime", action="store_true")
    p_run.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_run.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_run.add_argument("--context-window-lines", type=int, default=2)
    p_run.add_argument("--point-assembler", default="window-merge")
    p_run.add_argument("--context-assembler", default="sentence-window")
    p_run.add_argument(
        "--extractor-mode",
        choices=["hybrid_rules", "slm_swarm", "slm_tgi_required"],
        default="hybrid_rules",
    )
    p_run.add_argument("--extractor-models", default="")
    p_run.add_argument("--tgi-endpoint", default="")
    p_run.add_argument("--tgi-models", default="")
    p_run.add_argument("--tgi-workers", type=int, default=2)
    p_run.add_argument("--decoder", choices=["jsonschema"], default="jsonschema")
    p_run.add_argument("--slm-max-retries", type=int, default=2)
    p_run.add_argument("--slm-timeout-sec", type=float, default=20.0)
    p_run.add_argument("--slm-batch-size", type=int, default=8)
    p_run.add_argument("--slm-chunk-tokens", type=int, default=384)
    p_run.add_argument("--slm-overlap-tokens", type=int, default=96)
    p_run.add_argument("--slm-eval-split", choices=["train", "val", "test", "all"], default="all")
    p_run.add_argument("--schema-decoder", choices=["llguidance", "outlines"], default="llguidance")
    p_run.add_argument("--vote-policy", choices=["majority", "weighted"], default="weighted")
    p_run.add_argument("--min-vote-confidence", type=float, default=0.60)
    p_run.add_argument("--emit-debug-ranking", action="store_true")
    p_run.add_argument("--emit-validation-metrics", action="store_true")
    p_run.add_argument("--emit-extraction-calibration", action="store_true")
    p_run.add_argument("--calibration-bins", type=int, default=10)
    _add_system_finetune_flags(p_run)
    _add_strict_workflow_flags(p_run)
    _add_v2_storage_flags(p_run)
    _add_v2_runtime_hardening_flags(p_run)
    _add_v2_finetune_policy_flags(p_run)
    _add_mp_flags(p_run, include_enabled=True)
    _add_v1_release_flags(p_run)
    _add_dual_graph_flags(p_run)

    p_run_full = sub.add_parser("run-full", help="Run strict full workflow including processor stack")
    p_run_full.add_argument("--spec", required=True)
    p_run_full.add_argument("--query", required=True)
    p_run_full.add_argument("--out", required=True)
    p_run_full.add_argument("--min-discovery-score", type=float, default=0.0)
    p_run_full.add_argument("--plugin", default="")
    p_run_full.add_argument("--max-pages-per-provider", type=int, default=60)
    p_run_full.add_argument("--max-discovered-records", type=int, default=100000)
    p_run_full.add_argument("--saturation-window-pages", type=int, default=8)
    p_run_full.add_argument("--saturation-min-yield", type=float, default=0.03)
    p_run_full.add_argument("--min-pages-before-saturation", type=int, default=20)
    p_run_full.add_argument("--resume-discovery", action="store_true")
    p_run_full.add_argument("--max-downloads", type=int, default=200)
    p_run_full.add_argument("--acquire-workers", type=int, default=8)
    p_run_full.add_argument("--extract-workers", type=int, default=4)
    p_run_full.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_run_full.add_argument("--disable-english-first", dest="english_first", action="store_false")
    p_run_full.add_argument("--per-provider-cap", default="")
    _add_local_repo_flags(p_run_full)
    p_run_full.add_argument("--require-fulltext-mime", action="store_true")
    p_run_full.add_argument(
        "--extractor-mode",
        choices=["hybrid_rules", "slm_swarm", "slm_tgi_required"],
        default="slm_tgi_required",
    )
    p_run_full.add_argument("--extractor-models", default="")
    p_run_full.add_argument("--tgi-endpoint", default="")
    p_run_full.add_argument("--tgi-models", default="")
    p_run_full.add_argument("--tgi-workers", type=int, default=2)
    p_run_full.add_argument("--decoder", choices=["jsonschema"], default="jsonschema")
    p_run_full.add_argument("--slm-max-retries", type=int, default=2)
    p_run_full.add_argument("--slm-timeout-sec", type=float, default=20.0)
    p_run_full.add_argument("--slm-batch-size", type=int, default=8)
    p_run_full.add_argument("--slm-chunk-tokens", type=int, default=384)
    p_run_full.add_argument("--slm-overlap-tokens", type=int, default=96)
    p_run_full.add_argument("--schema-decoder", choices=["llguidance", "outlines"], default="llguidance")
    p_run_full.add_argument("--vote-policy", choices=["majority", "weighted"], default="weighted")
    p_run_full.add_argument("--min-vote-confidence", type=float, default=0.60)
    p_run_full.add_argument("--extractor-max-loop", type=int, default=2)
    p_run_full.add_argument("--processor-max-loop", type=int, default=2)
    p_run_full.add_argument("--gnn-hidden-dim", type=int, default=64)
    p_run_full.add_argument("--gnn-layers", type=int, default=2)
    p_run_full.add_argument("--gnn-dropout", type=float, default=0.1)
    p_run_full.add_argument("--gnn-epochs", type=int, default=80)
    p_run_full.add_argument("--gnn-lr", type=float, default=1e-3)
    p_run_full.add_argument("--tabular-model", default="xgboost")
    p_run_full.add_argument("--finetune-target-phase", default="NiSi")
    p_run_full.add_argument("--finetune-max-thickness-nm", type=float, default=200.0)
    _add_system_finetune_flags(p_run_full)
    p_run_full.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_run_full.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_run_full.add_argument("--context-window-lines", type=int, default=2)
    p_run_full.add_argument("--point-assembler", default="window-merge")
    p_run_full.add_argument("--context-assembler", default="sentence-window")
    p_run_full.add_argument("--emit-debug-ranking", action="store_true")
    p_run_full.add_argument("--emit-validation-metrics", action="store_true")
    p_run_full.add_argument("--emit-extraction-calibration", action="store_true")
    p_run_full.add_argument("--calibration-bins", type=int, default=10)
    _add_strict_workflow_flags(p_run_full)
    _add_v2_storage_flags(p_run_full)
    _add_v2_runtime_hardening_flags(p_run_full)
    _add_v2_finetune_policy_flags(p_run_full)
    _add_mp_flags(p_run_full, include_enabled=True)
    _add_v1_release_flags(p_run_full)
    _add_dual_graph_flags(p_run_full)

    p_validate_all = sub.add_parser("validate-all-done", help="Run strict full all-done validation pack")
    p_validate_all.add_argument("--spec", required=True)
    p_validate_all.add_argument("--query", required=True)
    p_validate_all.add_argument("--out", required=True)
    p_validate_all.add_argument("--gold", required=True)
    p_validate_all.add_argument("--shadow-gold", default="")
    p_validate_all.add_argument("--gold-context", default="")
    p_validate_all.add_argument("--precision-gate", type=float, default=0.80)
    p_validate_all.add_argument("--context-completeness-gate", type=float, default=0.70)
    p_validate_all.add_argument("--context-precision-gate", type=float, default=0.80)
    p_validate_all.add_argument("--min-discovery-score", type=float, default=0.0)
    p_validate_all.add_argument("--plugin", default="")
    p_validate_all.add_argument("--max-pages-per-provider", type=int, default=60)
    p_validate_all.add_argument("--max-discovered-records", type=int, default=100000)
    p_validate_all.add_argument("--saturation-window-pages", type=int, default=8)
    p_validate_all.add_argument("--saturation-min-yield", type=float, default=0.03)
    p_validate_all.add_argument("--min-pages-before-saturation", type=int, default=20)
    p_validate_all.add_argument("--resume-discovery", action="store_true")
    p_validate_all.add_argument("--max-downloads", type=int, default=200)
    p_validate_all.add_argument("--acquire-workers", type=int, default=8)
    p_validate_all.add_argument("--extract-workers", type=int, default=4)
    p_validate_all.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_validate_all.add_argument("--disable-english-first", dest="english_first", action="store_false")
    p_validate_all.add_argument("--per-provider-cap", default="")
    _add_local_repo_flags(p_validate_all)
    p_validate_all.add_argument("--require-fulltext-mime", action="store_true")
    p_validate_all.add_argument(
        "--extractor-mode",
        choices=["hybrid_rules", "slm_swarm", "slm_tgi_required"],
        default="slm_tgi_required",
    )
    p_validate_all.add_argument("--extractor-models", default="")
    p_validate_all.add_argument("--tgi-endpoint", default="")
    p_validate_all.add_argument("--tgi-models", default="")
    p_validate_all.add_argument("--tgi-workers", type=int, default=2)
    p_validate_all.add_argument("--decoder", choices=["jsonschema"], default="jsonschema")
    p_validate_all.add_argument("--slm-max-retries", type=int, default=2)
    p_validate_all.add_argument("--slm-timeout-sec", type=float, default=20.0)
    p_validate_all.add_argument("--slm-batch-size", type=int, default=8)
    p_validate_all.add_argument("--slm-chunk-tokens", type=int, default=384)
    p_validate_all.add_argument("--slm-overlap-tokens", type=int, default=96)
    p_validate_all.add_argument("--schema-decoder", choices=["llguidance", "outlines"], default="llguidance")
    p_validate_all.add_argument("--vote-policy", choices=["majority", "weighted"], default="weighted")
    p_validate_all.add_argument("--min-vote-confidence", type=float, default=0.60)
    p_validate_all.add_argument("--extractor-max-loop", type=int, default=2)
    p_validate_all.add_argument("--processor-max-loop", type=int, default=2)
    p_validate_all.add_argument("--gnn-hidden-dim", type=int, default=64)
    p_validate_all.add_argument("--gnn-layers", type=int, default=2)
    p_validate_all.add_argument("--gnn-dropout", type=float, default=0.1)
    p_validate_all.add_argument("--gnn-epochs", type=int, default=80)
    p_validate_all.add_argument("--gnn-lr", type=float, default=1e-3)
    p_validate_all.add_argument("--tabular-model", default="xgboost")
    p_validate_all.add_argument("--finetune-target-phase", default="NiSi")
    p_validate_all.add_argument("--finetune-max-thickness-nm", type=float, default=200.0)
    _add_system_finetune_flags(p_validate_all)
    p_validate_all.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_validate_all.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_validate_all.add_argument("--context-window-lines", type=int, default=2)
    p_validate_all.add_argument("--point-assembler", default="window-merge")
    p_validate_all.add_argument("--context-assembler", default="sentence-window")
    p_validate_all.add_argument("--emit-debug-ranking", action="store_true")
    p_validate_all.add_argument("--emit-validation-metrics", action="store_true")
    p_validate_all.add_argument("--emit-extraction-calibration", action="store_true")
    p_validate_all.add_argument("--calibration-bins", type=int, default=10)
    _add_strict_workflow_flags(p_validate_all)
    _add_v2_storage_flags(p_validate_all)
    _add_v2_runtime_hardening_flags(p_validate_all)
    _add_v2_finetune_policy_flags(p_validate_all)
    _add_mp_flags(p_validate_all, include_enabled=True)
    _add_v1_release_flags(p_validate_all)
    _add_dual_graph_flags(p_validate_all)

    p_release_v1 = sub.add_parser("release-validate-v1", help="Run OARP v1 release validation pack")
    p_release_v1.add_argument("--spec", required=True)
    p_release_v1.add_argument("--query", required=True)
    p_release_v1.add_argument("--out", required=True)
    p_release_v1.add_argument("--gold", required=True)
    p_release_v1.add_argument("--shadow-gold", default="")
    p_release_v1.add_argument("--gold-context", default="")
    p_release_v1.add_argument("--precision-gate", type=float, default=0.80)
    p_release_v1.add_argument("--context-completeness-gate", type=float, default=0.70)
    p_release_v1.add_argument("--context-precision-gate", type=float, default=0.80)
    p_release_v1.add_argument("--min-discovery-score", type=float, default=0.0)
    p_release_v1.add_argument("--plugin", default="")
    p_release_v1.add_argument("--max-pages-per-provider", type=int, default=60)
    p_release_v1.add_argument("--max-discovered-records", type=int, default=100000)
    p_release_v1.add_argument("--saturation-window-pages", type=int, default=8)
    p_release_v1.add_argument("--saturation-min-yield", type=float, default=0.03)
    p_release_v1.add_argument("--min-pages-before-saturation", type=int, default=20)
    p_release_v1.add_argument("--resume-discovery", action="store_true")
    p_release_v1.add_argument("--max-downloads", type=int, default=200)
    p_release_v1.add_argument("--acquire-workers", type=int, default=8)
    p_release_v1.add_argument("--extract-workers", type=int, default=4)
    p_release_v1.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_release_v1.add_argument("--disable-english-first", dest="english_first", action="store_false")
    p_release_v1.add_argument("--per-provider-cap", default="")
    _add_local_repo_flags(p_release_v1)
    p_release_v1.add_argument("--require-fulltext-mime", action="store_true")
    p_release_v1.add_argument(
        "--extractor-mode",
        choices=["hybrid_rules", "slm_swarm", "slm_tgi_required"],
        default="slm_tgi_required",
    )
    p_release_v1.add_argument("--extractor-models", default="")
    p_release_v1.add_argument("--tgi-endpoint", default="")
    p_release_v1.add_argument("--tgi-models", default="")
    p_release_v1.add_argument("--tgi-workers", type=int, default=2)
    p_release_v1.add_argument("--decoder", choices=["jsonschema"], default="jsonschema")
    p_release_v1.add_argument("--slm-max-retries", type=int, default=2)
    p_release_v1.add_argument("--slm-timeout-sec", type=float, default=20.0)
    p_release_v1.add_argument("--slm-batch-size", type=int, default=8)
    p_release_v1.add_argument("--slm-chunk-tokens", type=int, default=384)
    p_release_v1.add_argument("--slm-overlap-tokens", type=int, default=96)
    p_release_v1.add_argument("--schema-decoder", choices=["llguidance", "outlines"], default="llguidance")
    p_release_v1.add_argument("--vote-policy", choices=["majority", "weighted"], default="weighted")
    p_release_v1.add_argument("--min-vote-confidence", type=float, default=0.60)
    p_release_v1.add_argument("--extractor-max-loop", type=int, default=2)
    p_release_v1.add_argument("--processor-max-loop", type=int, default=2)
    p_release_v1.add_argument("--gnn-hidden-dim", type=int, default=64)
    p_release_v1.add_argument("--gnn-layers", type=int, default=2)
    p_release_v1.add_argument("--gnn-dropout", type=float, default=0.1)
    p_release_v1.add_argument("--gnn-epochs", type=int, default=80)
    p_release_v1.add_argument("--gnn-lr", type=float, default=1e-3)
    p_release_v1.add_argument("--tabular-model", default="xgboost")
    p_release_v1.add_argument("--finetune-target-phase", default="NiSi")
    p_release_v1.add_argument("--finetune-max-thickness-nm", type=float, default=200.0)
    _add_system_finetune_flags(p_release_v1)
    p_release_v1.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_release_v1.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_release_v1.add_argument("--context-window-lines", type=int, default=2)
    p_release_v1.add_argument("--point-assembler", default="window-merge")
    p_release_v1.add_argument("--context-assembler", default="sentence-window")
    p_release_v1.add_argument("--emit-debug-ranking", action="store_true")
    p_release_v1.add_argument("--emit-validation-metrics", action="store_true")
    p_release_v1.add_argument("--emit-extraction-calibration", action="store_true")
    p_release_v1.add_argument("--calibration-bins", type=int, default=10)
    _add_strict_workflow_flags(p_release_v1)
    _add_v2_storage_flags(p_release_v1)
    _add_v2_runtime_hardening_flags(p_release_v1)
    _add_v2_finetune_policy_flags(p_release_v1)
    _add_mp_flags(p_release_v1, include_enabled=True)
    _add_v1_release_flags(p_release_v1)
    _add_dual_graph_flags(p_release_v1)

    p_release_v2 = sub.add_parser("validate-v2-publish", help="Run OARP v2 publish validation pack")
    p_release_v2.add_argument("--spec", required=True)
    p_release_v2.add_argument("--query", required=True)
    p_release_v2.add_argument("--out", required=True)
    p_release_v2.add_argument("--gold", required=True)
    p_release_v2.add_argument("--shadow-gold", default="")
    p_release_v2.add_argument("--gold-context", default="")
    p_release_v2.add_argument("--vault-compare", dest="vault_compare", action="store_true", default=True)
    p_release_v2.add_argument("--no-vault-compare", dest="vault_compare", action="store_false")
    p_release_v2.add_argument("--precision-gate", type=float, default=0.80)
    p_release_v2.add_argument("--context-completeness-gate", type=float, default=0.70)
    p_release_v2.add_argument("--context-precision-gate", type=float, default=0.80)
    p_release_v2.add_argument("--min-discovery-score", type=float, default=0.0)
    p_release_v2.add_argument("--plugin", default="")
    p_release_v2.add_argument("--max-pages-per-provider", type=int, default=60)
    p_release_v2.add_argument("--max-discovered-records", type=int, default=100000)
    p_release_v2.add_argument("--saturation-window-pages", type=int, default=8)
    p_release_v2.add_argument("--saturation-min-yield", type=float, default=0.03)
    p_release_v2.add_argument("--min-pages-before-saturation", type=int, default=20)
    p_release_v2.add_argument("--resume-discovery", action="store_true")
    p_release_v2.add_argument("--max-downloads", type=int, default=200)
    p_release_v2.add_argument("--acquire-workers", type=int, default=8)
    p_release_v2.add_argument("--extract-workers", type=int, default=4)
    p_release_v2.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_release_v2.add_argument("--disable-english-first", dest="english_first", action="store_false")
    p_release_v2.add_argument("--per-provider-cap", default="")
    _add_local_repo_flags(p_release_v2)
    p_release_v2.add_argument("--require-fulltext-mime", action="store_true")
    p_release_v2.add_argument(
        "--extractor-mode",
        choices=["hybrid_rules", "slm_swarm", "slm_tgi_required"],
        default="slm_tgi_required",
    )
    p_release_v2.add_argument("--extractor-models", default="")
    p_release_v2.add_argument("--tgi-endpoint", default="")
    p_release_v2.add_argument("--tgi-models", default="")
    p_release_v2.add_argument("--tgi-workers", type=int, default=2)
    p_release_v2.add_argument("--decoder", choices=["jsonschema"], default="jsonschema")
    p_release_v2.add_argument("--slm-max-retries", type=int, default=2)
    p_release_v2.add_argument("--slm-timeout-sec", type=float, default=20.0)
    p_release_v2.add_argument("--slm-batch-size", type=int, default=8)
    p_release_v2.add_argument("--slm-chunk-tokens", type=int, default=384)
    p_release_v2.add_argument("--slm-overlap-tokens", type=int, default=96)
    p_release_v2.add_argument("--schema-decoder", choices=["llguidance", "outlines"], default="llguidance")
    p_release_v2.add_argument("--vote-policy", choices=["majority", "weighted"], default="weighted")
    p_release_v2.add_argument("--min-vote-confidence", type=float, default=0.60)
    p_release_v2.add_argument("--extractor-max-loop", type=int, default=2)
    p_release_v2.add_argument("--processor-max-loop", type=int, default=2)
    p_release_v2.add_argument("--gnn-hidden-dim", type=int, default=64)
    p_release_v2.add_argument("--gnn-layers", type=int, default=2)
    p_release_v2.add_argument("--gnn-dropout", type=float, default=0.1)
    p_release_v2.add_argument("--gnn-epochs", type=int, default=80)
    p_release_v2.add_argument("--gnn-lr", type=float, default=1e-3)
    p_release_v2.add_argument("--tabular-model", default="xgboost")
    p_release_v2.add_argument("--finetune-target-phase", default="NiSi")
    p_release_v2.add_argument("--finetune-max-thickness-nm", type=float, default=200.0)
    _add_system_finetune_flags(p_release_v2)
    p_release_v2.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_release_v2.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_release_v2.add_argument("--context-window-lines", type=int, default=2)
    p_release_v2.add_argument("--point-assembler", default="window-merge")
    p_release_v2.add_argument("--context-assembler", default="sentence-window")
    p_release_v2.add_argument("--emit-debug-ranking", action="store_true")
    p_release_v2.add_argument("--emit-validation-metrics", action="store_true")
    p_release_v2.add_argument("--emit-extraction-calibration", action="store_true")
    p_release_v2.add_argument("--calibration-bins", type=int, default=10)
    _add_strict_workflow_flags(p_release_v2)
    _add_v2_storage_flags(p_release_v2)
    _add_v2_runtime_hardening_flags(p_release_v2)
    _add_v2_finetune_policy_flags(p_release_v2)
    _add_mp_flags(p_release_v2, include_enabled=True)
    _add_v1_release_flags(p_release_v2)
    _add_dual_graph_flags(p_release_v2)

    p_discover = sub.add_parser("discover", help="Discover OA candidate articles")
    p_discover.add_argument("--spec", required=True)
    p_discover.add_argument("--query", required=True)
    p_discover.add_argument("--out", required=True)
    p_discover.add_argument("--min-discovery-score", type=float, default=0.0)
    p_discover.add_argument("--max-pages-per-provider", type=int, default=60)
    p_discover.add_argument("--max-discovered-records", type=int, default=100000)
    p_discover.add_argument("--saturation-window-pages", type=int, default=8)
    p_discover.add_argument("--saturation-min-yield", type=float, default=0.03)
    p_discover.add_argument("--min-pages-before-saturation", type=int, default=20)
    p_discover.add_argument("--resume-discovery", action="store_true")
    p_discover.add_argument("--max-downloads", type=int, default=200)
    p_discover.add_argument("--per-provider-cap", default="")
    p_discover.add_argument("--plugin", default="")
    p_discover.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_discover.add_argument("--disable-english-first", dest="english_first", action="store_false")
    _add_local_repo_flags(p_discover)
    p_discover.add_argument("--emit-debug-ranking", action="store_true")

    p_acquire = sub.add_parser("acquire", help="Acquire article full text")
    p_acquire.add_argument("--run", required=True)
    p_acquire.add_argument("--max-downloads", type=int, default=200)
    p_acquire.add_argument("--acquire-workers", type=int, default=8)
    p_acquire.add_argument("--english-first", dest="english_first", action="store_true", default=True)
    p_acquire.add_argument("--disable-english-first", dest="english_first", action="store_false")
    p_acquire.add_argument("--resume-discovery", action="store_true")
    p_acquire.add_argument("--require-fulltext-mime", action="store_true")
    _add_local_repo_flags(p_acquire)

    p_extract = sub.add_parser("extract", help="Extract evidence points")
    p_extract.add_argument("--run", required=True)
    p_extract.add_argument("--engines", default="text,table,figure")
    p_extract.add_argument("--context-window-lines", type=int, default=2)
    p_extract.add_argument("--point-assembler", default="window-merge")
    p_extract.add_argument("--extract-workers", type=int, default=4)
    p_extract.add_argument("--context-assembler", default="sentence-window")
    p_extract.add_argument(
        "--extractor-mode",
        choices=["hybrid_rules", "slm_swarm", "slm_tgi_required"],
        default="hybrid_rules",
    )
    p_extract.add_argument("--extractor-models", default="")
    p_extract.add_argument("--tgi-endpoint", default="")
    p_extract.add_argument("--tgi-models", default="")
    p_extract.add_argument("--tgi-workers", type=int, default=2)
    p_extract.add_argument("--decoder", choices=["jsonschema"], default="jsonschema")
    p_extract.add_argument("--slm-max-retries", type=int, default=2)
    p_extract.add_argument("--slm-timeout-sec", type=float, default=20.0)
    p_extract.add_argument("--slm-batch-size", type=int, default=8)
    p_extract.add_argument("--slm-chunk-tokens", type=int, default=384)
    p_extract.add_argument("--slm-overlap-tokens", type=int, default=96)
    p_extract.add_argument("--slm-eval-split", choices=["train", "val", "test", "all"], default="all")
    p_extract.add_argument("--schema-decoder", choices=["llguidance", "outlines"], default="llguidance")
    p_extract.add_argument("--vote-policy", choices=["majority", "weighted"], default="weighted")
    p_extract.add_argument("--min-vote-confidence", type=float, default=0.60)
    _add_v2_runtime_hardening_flags(p_extract)
    p_extract.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_extract.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_extract.add_argument("--plugin", default="")
    _add_mp_flags(p_extract, include_enabled=True)
    _add_dual_graph_flags(p_extract)

    p_validate = sub.add_parser("validate", help="Validate extracted evidence")
    p_validate.add_argument("--run", required=True)
    p_validate.add_argument("--require-context-fields", dest="require_context_fields", action="store_true", default=True)
    p_validate.add_argument("--no-require-context-fields", dest="require_context_fields", action="store_false")
    p_validate.add_argument("--emit-validation-metrics", action="store_true")
    p_validate.add_argument("--emit-extraction-calibration", action="store_true")
    p_validate.add_argument("--calibration-bins", type=int, default=10)
    p_validate.add_argument("--extractor-gate-profile", default="default")
    _add_mp_flags(p_validate, include_enabled=True)
    _add_dual_graph_flags(p_validate)

    p_consensus = sub.add_parser("consensus", help="Build consensus points")
    p_consensus.add_argument("--run", required=True)
    _add_mp_flags(p_consensus, include_enabled=True)
    _add_dual_graph_flags(p_consensus)

    p_plot = sub.add_parser("plot", help="Render plot output")
    p_plot.add_argument("--run", required=True)

    p_report = sub.add_parser("report", help="Render report output")
    p_report.add_argument("--run", required=True)

    p_bootstrap = sub.add_parser("bootstrap", help="Auto-bootstrap strict runtime stack")
    p_bootstrap.add_argument("--out", required=True)
    _add_local_repo_flags(p_bootstrap)
    _add_strict_workflow_flags(p_bootstrap)
    p_bootstrap.add_argument("--tgi-endpoint", default="")
    _add_mp_flags(p_bootstrap, include_enabled=True)

    p_bootstrap_strict = sub.add_parser("bootstrap-strict", help="Self-hosted bootstrap for strict workflow")
    p_bootstrap_strict.add_argument("--out", required=True)
    _add_local_repo_flags(p_bootstrap_strict)
    _add_strict_workflow_flags(p_bootstrap_strict)
    p_bootstrap_strict.add_argument("--tgi-endpoint", default="")
    _add_mp_flags(p_bootstrap_strict, include_enabled=True)

    p_preflight = sub.add_parser("preflight", help="Run strict preflight checks")
    p_preflight.add_argument("--run", required=True)
    _add_local_repo_flags(p_preflight)
    _add_strict_workflow_flags(p_preflight)
    p_preflight.add_argument("--tgi-endpoint", default="")
    p_preflight.add_argument("--check-tgi-generate", dest="check_tgi_generate", action="store_true", default=True)
    p_preflight.add_argument("--no-check-tgi-generate", dest="check_tgi_generate", action="store_false")
    _add_mp_flags(p_preflight, include_enabled=True)

    p_tgi = sub.add_parser("tgi", help="Manage local Docker TGI runtime")
    tgi_sub = p_tgi.add_subparsers(dest="tgi_cmd", required=True)
    p_tgi_status = tgi_sub.add_parser("status", help="Show TGI runtime status")
    p_tgi_status.add_argument("--run", required=True)
    _add_strict_workflow_flags(p_tgi_status)
    p_tgi_status.add_argument("--tgi-endpoint", default="")
    p_tgi_start = tgi_sub.add_parser("start", help="Start Docker TGI runtime")
    p_tgi_start.add_argument("--run", required=True)
    _add_strict_workflow_flags(p_tgi_start)
    p_tgi_start.add_argument("--tgi-endpoint", default="")
    p_tgi_stop = tgi_sub.add_parser("stop", help="Stop Docker TGI runtime")
    p_tgi_stop.add_argument("--run", required=True)
    _add_strict_workflow_flags(p_tgi_stop)
    p_tgi_stop.add_argument("--tgi-endpoint", default="")

    p_bench = sub.add_parser("benchmark", help="Evaluate precision-first benchmark")
    p_bench.add_argument("--spec", required=True)
    p_bench.add_argument("--gold", required=True)
    p_bench.add_argument("--out", required=True)
    p_bench.add_argument("--run", default="")
    p_bench.add_argument("--shadow-gold", default="")
    p_bench.add_argument("--gold-context", default="")
    p_bench.add_argument("--context-completeness-gate", type=float, default=0.70)
    p_bench.add_argument("--context-precision-gate", type=float, default=0.80)
    p_bench.add_argument("--strict-gold", dest="strict_gold", action="store_true", default=True)
    p_bench.add_argument("--no-strict-gold", dest="strict_gold", action="store_false")

    p_knowledge = sub.add_parser("knowledge", help="Build knowledge-plane artifacts from validated points")
    p_knowledge.add_argument("--run", required=True)

    p_graph_build = sub.add_parser("graph-build", help="Build dual concept graph artifacts from run evidence")
    p_graph_build.add_argument("--run", required=True)
    p_graph_build.add_argument(
        "--source",
        default="phase_events",
        help="phase_events|validated_points|processor_training_rows or explicit parquet/csv path",
    )
    _add_dual_graph_flags(p_graph_build)

    p_graph_audit = sub.add_parser("graph-audit", help="Audit dual concept graph artifacts")
    p_graph_audit.add_argument("--run", required=True)
    _add_dual_graph_flags(p_graph_audit)

    p_vault_export = sub.add_parser("vault-export", help="Export Obsidian vault from run artifacts")
    p_vault_export.add_argument("--run", required=True)
    p_vault_export.add_argument("--out", default="")
    p_vault_export.add_argument("--profile", default="per_run_v1")
    _add_v2_storage_flags(p_vault_export)

    p_vault_import = sub.add_parser("vault-import", help="Import Obsidian vault links into run as soft supervision")
    p_vault_import.add_argument("--vault", required=True)
    p_vault_import.add_argument("--run", required=True)
    p_vault_import.add_argument("--mode", choices=["soft_supervision"], default="soft_supervision")
    _add_v2_storage_flags(p_vault_import)

    p_vault_bench = sub.add_parser("vault-benchmark", help="Benchmark vault links against machine graph projection")
    p_vault_bench.add_argument("--run", required=True)
    p_vault_bench.add_argument("--vault", required=True)
    _add_v2_storage_flags(p_vault_bench)

    p_recipe = sub.add_parser("recipe-generate", help="Generate recipe artifacts directly from a run")
    p_recipe.add_argument("--run", required=True)
    p_recipe.add_argument("--request-file", required=True)
    p_recipe.add_argument("--processor-model", default="")
    p_recipe.add_argument("--gate-profile", default="")
    p_recipe.add_argument("--mp-enabled", dest="mp_enabled", action="store_true", default=True)
    p_recipe.add_argument("--no-mp-enabled", dest="mp_enabled", action="store_false")
    p_recipe.add_argument("--mp-api-key", default="")
    p_recipe.add_argument("--mp-endpoint", default="https://api.materialsproject.org")
    _add_mp_flags(p_recipe, include_enabled=False)
    _add_dual_graph_flags(p_recipe)

    p_mp_enrich = sub.add_parser("mp-enrich", help="Enrich extracted evidence with Materials Project data")
    p_mp_enrich.add_argument("--run", required=True)
    _add_mp_flags(p_mp_enrich, include_enabled=True)

    p_proc_train = sub.add_parser("process-train-universal", help="Train universal graph+tabular processing model")
    p_proc_train.add_argument("--run", required=True)
    p_proc_train.add_argument("--datasets", default="phase_events")
    p_proc_train.add_argument("--out", required=True)
    _add_dual_graph_flags(p_proc_train)

    p_proc_ft = sub.add_parser("process-finetune-system", help="Fine-tune processor model for a specific system")
    p_proc_ft.add_argument("--run", required=True)
    p_proc_ft.add_argument("--system-id", required=True)
    p_proc_ft.add_argument("--base-model", required=True)
    p_proc_ft.add_argument("--out", required=True)
    _add_system_finetune_flags(p_proc_ft)
    _add_dual_graph_flags(p_proc_ft)

    p_gnn = sub.add_parser("processor-train-gnn", help="Train PyG hetero-GNN base processor")
    p_gnn.add_argument("--run", required=True)
    p_gnn.add_argument("--epochs", type=int, default=80)
    p_gnn.add_argument("--batch-size", type=int, default=32)
    p_gnn.add_argument("--lr", type=float, default=1e-3)
    p_gnn.add_argument("--hidden-dim", type=int, default=64)
    p_gnn.add_argument("--layers", type=int, default=2)
    p_gnn.add_argument("--dropout", type=float, default=0.1)
    _add_dual_graph_flags(p_gnn)

    p_tab = sub.add_parser("processor-train-tabular", help="Train tabular head from GNN embeddings")
    p_tab.add_argument("--run", required=True)
    p_tab.add_argument("--base-gnn", required=True)
    p_tab.add_argument("--model", default="xgboost")
    _add_dual_graph_flags(p_tab)

    p_ft2 = sub.add_parser("processor-finetune", help="Fine-tune processor on phase/thickness slice")
    p_ft2.add_argument("--run", required=True)
    p_ft2.add_argument("--base-model", required=True)
    p_ft2.add_argument("--target", default="NiSi")
    p_ft2.add_argument("--max-thickness-nm", type=float, default=200.0)
    p_ft2.add_argument("--epochs", type=int, default=40)
    _add_system_finetune_flags(p_ft2)
    _add_v2_finetune_policy_flags(p_ft2)
    _add_dual_graph_flags(p_ft2)

    p_eval = sub.add_parser("processor-eval", help="Evaluate base + fine-tuned processor models")
    p_eval.add_argument("--run", required=True)
    p_eval.add_argument("--suite", default="base_and_finetune")
    _add_system_finetune_flags(p_eval)
    _add_v2_finetune_policy_flags(p_eval)
    _add_dual_graph_flags(p_eval)

    p_bench_extract = sub.add_parser("benchmark-extraction", help="Run extraction benchmark suites")
    p_bench_extract.add_argument("--suite", required=True, choices=["matsci_nlp", "supermat", "measeval", "thinfilm_gold"])
    p_bench_extract.add_argument("--spec", required=True)
    p_bench_extract.add_argument("--run", required=True)
    p_bench_extract.add_argument("--out", required=True)
    p_bench_extract.add_argument("--gold", default="")

    p_bench_process = sub.add_parser("benchmark-processing", help="Run processing benchmark suites")
    p_bench_process.add_argument("--suite", required=True, choices=["phase_transition", "recipe_rank", "context_quality"])
    p_bench_process.add_argument("--run", required=True)
    p_bench_process.add_argument("--out", required=True)

    p_service = sub.add_parser("service", help="Async job service")
    service_sub = p_service.add_subparsers(dest="service_cmd", required=True)
    p_service_start = service_sub.add_parser("start", help="Start async HTTP service")
    p_service_start.add_argument("--data-dir", default="~/.oarp_service")
    p_service_start.add_argument("--jobs-db", default="")
    p_service_start.add_argument("--host", default="127.0.0.1")
    p_service_start.add_argument("--port", type=int, default=8787)
    p_service_start.add_argument("--worker-poll-sec", type=float, default=0.5)
    p_service_start.add_argument("--mp-enabled", dest="mp_enabled", action="store_true", default=True)
    p_service_start.add_argument("--no-mp-enabled", dest="mp_enabled", action="store_false")
    p_service_start.add_argument("--mp-api-key", default="")
    p_service_start.add_argument("--mp-endpoint", default="https://api.materialsproject.org")
    p_service_start.add_argument("--mp-scope", choices=["summary_thermo", "summary_only", "broad"], default="summary_thermo")

    p_job = sub.add_parser("job", help="Manage async jobs")
    job_sub = p_job.add_subparsers(dest="job_cmd", required=True)

    p_job_submit_research = job_sub.add_parser("submit-research", help="Submit research-index job")
    p_job_submit_research.add_argument("--spec", required=True)
    p_job_submit_research.add_argument("--query", default="")
    p_job_submit_research.add_argument("--run-dir", default="")
    p_job_submit_research.add_argument("--cfg-overrides-file", default="")
    p_job_submit_research.add_argument("--data-dir", default="~/.oarp_service")
    p_job_submit_research.add_argument("--jobs-db", default="")

    p_job_submit_recipe = job_sub.add_parser("submit-recipe", help="Submit recipe-generation job")
    p_job_submit_recipe.add_argument("--request-file", required=True)
    p_job_submit_recipe.add_argument("--data-dir", default="~/.oarp_service")
    p_job_submit_recipe.add_argument("--jobs-db", default="")

    p_job_status = job_sub.add_parser("status", help="Get job status/result")
    p_job_status.add_argument("--job-id", required=True)
    p_job_status.add_argument("--data-dir", default="~/.oarp_service")
    p_job_status.add_argument("--jobs-db", default="")

    p_job_artifacts = job_sub.add_parser("artifacts", help="List job artifacts")
    p_job_artifacts.add_argument("--job-id", required=True)
    p_job_artifacts.add_argument("--data-dir", default="~/.oarp_service")
    p_job_artifacts.add_argument("--jobs-db", default="")

    p_job_retry = job_sub.add_parser("retry", help="Retry a failed job")
    p_job_retry.add_argument("--job-id", required=True)
    p_job_retry.add_argument("--data-dir", default="~/.oarp_service")
    p_job_retry.add_argument("--jobs-db", default="")

    p_job_worker = job_sub.add_parser("run-worker", help="Run queued jobs once/batch without HTTP server")
    p_job_worker.add_argument("--max-jobs", type=int, default=50)
    p_job_worker.add_argument("--data-dir", default="~/.oarp_service")
    p_job_worker.add_argument("--jobs-db", default="")
    p_job_worker.add_argument("--mp-enabled", dest="mp_enabled", action="store_true", default=True)
    p_job_worker.add_argument("--no-mp-enabled", dest="mp_enabled", action="store_false")
    p_job_worker.add_argument("--mp-api-key", default="")
    p_job_worker.add_argument("--mp-endpoint", default="https://api.materialsproject.org")
    p_job_worker.add_argument("--mp-scope", choices=["summary_thermo", "summary_only", "broad"], default="summary_thermo")

    p_adapter = sub.add_parser("agents-inc-handoff", help="Build agents-inc compatible handoff JSON")
    p_adapter.add_argument("--run", required=True)
    p_adapter.add_argument("--out", default="")

    return parser


def _cfg_from_args(args: argparse.Namespace, run_path: str | Path) -> RunConfig:
    cfg = RunConfig(run_dir=str(Path(run_path).expanduser().resolve()))
    if hasattr(args, "storage_root"):
        cfg.storage_root = str(args.storage_root or cfg.storage_root).strip() or cfg.storage_root
    if hasattr(args, "run_root"):
        cfg.run_root = str(args.run_root or "").strip()
    if hasattr(args, "cache_root"):
        cfg.cache_root = str(args.cache_root or "").strip()
    if hasattr(args, "model_root"):
        cfg.model_root = str(args.model_root or "").strip()
    if hasattr(args, "dataset_root"):
        cfg.dataset_root = str(args.dataset_root or "").strip()
    if hasattr(args, "vault_root"):
        cfg.vault_root = str(args.vault_root or "").strip()
    if hasattr(args, "run_profile"):
        cfg.run_profile = str(args.run_profile or "").strip()
    if hasattr(args, "min_discovery_score"):
        cfg.min_discovery_score = float(args.min_discovery_score)
    if hasattr(args, "plugin"):
        cfg.plugin_id = str(args.plugin or "").strip()
    if hasattr(args, "max_pages_per_provider"):
        cfg.max_pages_per_provider = max(1, int(args.max_pages_per_provider))
    if hasattr(args, "max_discovered_records"):
        cfg.max_discovered_records = max(1, int(args.max_discovered_records))
    if hasattr(args, "saturation_window_pages"):
        cfg.saturation_window_pages = max(1, int(args.saturation_window_pages))
    if hasattr(args, "saturation_min_yield"):
        cfg.saturation_min_yield = float(args.saturation_min_yield)
    if hasattr(args, "min_pages_before_saturation"):
        cfg.min_pages_before_saturation = max(1, int(args.min_pages_before_saturation))
    if hasattr(args, "resume_discovery"):
        cfg.resume_discovery = bool(args.resume_discovery)
    if hasattr(args, "max_downloads"):
        cfg.max_downloads = max(1, int(args.max_downloads))
    if hasattr(args, "local_repo_paths"):
        local_repo_paths = [str(Path(item).expanduser().resolve()) for item in list(args.local_repo_paths or []) if str(item).strip()]
        if local_repo_paths:
            cfg.local_repo_paths = local_repo_paths
    if hasattr(args, "local_repo_recursive"):
        cfg.local_repo_recursive = bool(args.local_repo_recursive)
    if hasattr(args, "local_file_glob"):
        cfg.local_file_glob = str(args.local_file_glob or "*.pdf").strip() or "*.pdf"
    if hasattr(args, "local_merge_mode"):
        cfg.local_merge_mode = str(args.local_merge_mode or "union").strip().lower() or "union"
    if hasattr(args, "local_max_files"):
        cfg.local_max_files = max(0, int(args.local_max_files))
    if hasattr(args, "local_require_readable"):
        cfg.local_require_readable = bool(args.local_require_readable)
    if hasattr(args, "acquire_workers"):
        cfg.acquire_workers = max(1, int(args.acquire_workers))
    if hasattr(args, "extract_workers"):
        cfg.extract_workers = max(1, int(args.extract_workers))
    if hasattr(args, "english_first"):
        cfg.english_first = bool(args.english_first)
    if hasattr(args, "per_provider_cap"):
        cfg.per_provider_cap = _parse_provider_caps(args.per_provider_cap)
    if hasattr(args, "require_fulltext_mime"):
        cfg.require_fulltext_mime = bool(args.require_fulltext_mime)
    if hasattr(args, "require_context_fields"):
        cfg.require_context_fields = bool(args.require_context_fields)
    if hasattr(args, "context_window_lines"):
        cfg.context_window_lines = int(args.context_window_lines)
    if hasattr(args, "point_assembler"):
        cfg.point_assembler = str(args.point_assembler or "window-merge")
    if hasattr(args, "context_assembler"):
        cfg.context_assembler = str(args.context_assembler or "sentence-window")
    if hasattr(args, "phase_schema_version"):
        cfg.phase_schema_version = str(args.phase_schema_version or "v1").strip() or "v1"
    if hasattr(args, "graph_core_mode"):
        cfg.graph_core_mode = str(args.graph_core_mode or "per_article_typed").strip() or "per_article_typed"
    if hasattr(args, "graph_architecture"):
        cfg.graph_architecture = str(args.graph_architecture or "legacy_v1").strip().lower() or "legacy_v1"
    if hasattr(args, "concept_ontology_profile"):
        cfg.concept_ontology_profile = str(args.concept_ontology_profile or "maximal").strip().lower() or "maximal"
    if hasattr(args, "bridge_weight_policy"):
        cfg.bridge_weight_policy = str(args.bridge_weight_policy or "deterministic_v1").strip().lower() or "deterministic_v1"
    if hasattr(args, "bridge_weight_threshold"):
        cfg.bridge_weight_threshold = max(0.0, min(1.0, float(args.bridge_weight_threshold)))
    if hasattr(args, "concept_gates"):
        gates = [item.strip().lower() for item in str(args.concept_gates or "").split(",") if item.strip()]
        if gates:
            cfg.concept_gates = gates
    if hasattr(args, "phase_require_elements"):
        cfg.phase_require_elements = bool(args.phase_require_elements)
    if hasattr(args, "phase_require_stoich"):
        cfg.phase_require_stoich = bool(args.phase_require_stoich)
    if hasattr(args, "phase_require_spacegroup"):
        cfg.phase_require_spacegroup = bool(args.phase_require_spacegroup)
    if hasattr(args, "cache_mode"):
        cfg.cache_mode = str(args.cache_mode or "run_local").strip().lower() or "run_local"
    if hasattr(args, "shared_cache_root"):
        cfg.shared_cache_root = str(args.shared_cache_root or "").strip()
    if hasattr(args, "cache_read_only"):
        cfg.cache_read_only = bool(args.cache_read_only)
    if hasattr(args, "cache_ttl_hours"):
        cfg.cache_ttl_hours = max(1, int(args.cache_ttl_hours))
    if hasattr(args, "cpu_strict_profile"):
        cfg.cpu_strict_profile = bool(args.cpu_strict_profile)
    if hasattr(args, "cpu_max_threads"):
        cfg.cpu_max_threads = max(1, int(args.cpu_max_threads))
    if hasattr(args, "cpu_probe"):
        cfg.cpu_probe = bool(args.cpu_probe)
    if hasattr(args, "extractor_mode"):
        cfg.extractor_mode = str(args.extractor_mode or "hybrid_rules")
    if hasattr(args, "extractor_models"):
        raw_models = str(args.extractor_models or "").strip()
        if raw_models:
            cfg.extractor_models = [item.strip() for item in raw_models.split(",") if item.strip()]
    if hasattr(args, "tgi_endpoint"):
        cfg.tgi_endpoint = str(args.tgi_endpoint or "").strip()
    if hasattr(args, "tgi_models"):
        raw_tgi_models = str(args.tgi_models or "").strip()
        if raw_tgi_models:
            cfg.tgi_models = [item.strip() for item in raw_tgi_models.split(",") if item.strip()]
    if hasattr(args, "tgi_workers"):
        cfg.tgi_workers = max(1, int(args.tgi_workers))
    if hasattr(args, "decoder"):
        cfg.decoder = str(args.decoder or "jsonschema")
    if hasattr(args, "slm_max_retries"):
        cfg.slm_max_retries = max(0, int(args.slm_max_retries))
    if hasattr(args, "slm_timeout_sec"):
        cfg.slm_timeout_sec = max(1.0, float(args.slm_timeout_sec))
    if hasattr(args, "slm_batch_size"):
        cfg.slm_batch_size = max(1, int(args.slm_batch_size))
    if hasattr(args, "slm_chunk_tokens"):
        cfg.slm_chunk_tokens = max(64, int(args.slm_chunk_tokens))
    if hasattr(args, "slm_overlap_tokens"):
        cfg.slm_overlap_tokens = max(0, int(args.slm_overlap_tokens))
    if hasattr(args, "slm_max_chunks_per_doc"):
        cfg.slm_max_chunks_per_doc = max(0, int(args.slm_max_chunks_per_doc))
    if hasattr(args, "slm_max_doc_chars"):
        cfg.slm_max_doc_chars = max(0, int(args.slm_max_doc_chars))
    if hasattr(args, "slm_response_cache"):
        cfg.slm_response_cache = str(args.slm_response_cache or "on").strip().lower()
    if hasattr(args, "extract_stage_timeout_sec"):
        cfg.extract_stage_timeout_sec = max(0, int(args.extract_stage_timeout_sec))
    if hasattr(args, "slm_eval_split"):
        cfg.slm_eval_split = str(args.slm_eval_split or "all")
    if hasattr(args, "schema_decoder"):
        cfg.schema_decoder = str(args.schema_decoder or "llguidance")
    if hasattr(args, "vote_policy"):
        cfg.vote_policy = str(args.vote_policy or "weighted")
    if hasattr(args, "min_vote_confidence"):
        cfg.min_vote_confidence = max(0.0, min(1.0, float(args.min_vote_confidence)))
    if hasattr(args, "emit_debug_ranking"):
        cfg.emit_debug_ranking = bool(args.emit_debug_ranking)
    if hasattr(args, "emit_validation_metrics"):
        cfg.emit_validation_metrics = bool(args.emit_validation_metrics)
    if hasattr(args, "emit_extraction_calibration"):
        cfg.emit_extraction_calibration = bool(args.emit_extraction_calibration)
    if hasattr(args, "calibration_bins"):
        cfg.calibration_bins = max(2, int(args.calibration_bins))
    if hasattr(args, "extractor_gate_profile"):
        cfg.extractor_gate_profile = str(args.extractor_gate_profile or "default")
    if hasattr(args, "extractor_max_loop"):
        cfg.extractor_max_loop = max(1, int(args.extractor_max_loop))
    if hasattr(args, "processor_max_loop"):
        cfg.processor_max_loop = max(1, int(args.processor_max_loop))
    if hasattr(args, "hidden_dim"):
        cfg.gnn_hidden_dim = max(8, int(args.hidden_dim))
    if hasattr(args, "gnn_hidden_dim"):
        cfg.gnn_hidden_dim = max(8, int(args.gnn_hidden_dim))
    if hasattr(args, "layers"):
        cfg.gnn_layers = max(1, int(args.layers))
    if hasattr(args, "gnn_layers"):
        cfg.gnn_layers = max(1, int(args.gnn_layers))
    if hasattr(args, "dropout"):
        cfg.gnn_dropout = max(0.0, min(0.9, float(args.dropout)))
    if hasattr(args, "gnn_dropout"):
        cfg.gnn_dropout = max(0.0, min(0.9, float(args.gnn_dropout)))
    if hasattr(args, "epochs"):
        cfg.gnn_epochs = max(1, int(args.epochs))
    if hasattr(args, "gnn_epochs"):
        cfg.gnn_epochs = max(1, int(args.gnn_epochs))
    if hasattr(args, "lr"):
        cfg.gnn_lr = max(1e-6, float(args.lr))
    if hasattr(args, "gnn_lr"):
        cfg.gnn_lr = max(1e-6, float(args.gnn_lr))
    if hasattr(args, "model"):
        cfg.tabular_model = str(args.model or "xgboost").strip().lower()
    if hasattr(args, "tabular_model"):
        cfg.tabular_model = str(args.tabular_model or "xgboost").strip().lower()
    if hasattr(args, "target"):
        cfg.finetune_target_phase = str(args.target or "NiSi")
    if hasattr(args, "finetune_target_phase"):
        cfg.finetune_target_phase = str(args.finetune_target_phase or "NiSi")
    if hasattr(args, "max_thickness_nm"):
        cfg.finetune_max_thickness_nm = float(args.max_thickness_nm)
    if hasattr(args, "finetune_max_thickness_nm"):
        cfg.finetune_max_thickness_nm = float(args.finetune_max_thickness_nm)
    if hasattr(args, "system_finetune_dataset"):
        cfg.system_finetune_dataset = str(args.system_finetune_dataset or "").strip()
    if hasattr(args, "system_eval_holdout_ratio"):
        cfg.system_eval_holdout_ratio = max(0.0, min(0.95, float(args.system_eval_holdout_ratio)))
    if hasattr(args, "mp_enabled"):
        cfg.mp_enabled = bool(args.mp_enabled)
    if hasattr(args, "mp_mode"):
        cfg.mp_mode = str(args.mp_mode or "interpreter")
    if hasattr(args, "mp_scope"):
        cfg.mp_scope = str(args.mp_scope or "summary_thermo")
    if hasattr(args, "mp_on_demand"):
        cfg.mp_on_demand = bool(args.mp_on_demand)
    if hasattr(args, "mp_query_workers"):
        cfg.mp_query_workers = max(1, int(args.mp_query_workers))
    if hasattr(args, "mp_timeout_sec"):
        cfg.mp_timeout_sec = max(1.0, float(args.mp_timeout_sec))
    if hasattr(args, "mp_max_queries"):
        cfg.mp_max_queries = max(1, int(args.mp_max_queries))
    if hasattr(args, "mp_cache_path"):
        cfg.mp_cache_path = str(args.mp_cache_path or "").strip()
    if hasattr(args, "strict_full_workflow"):
        cfg.strict_full_workflow = bool(args.strict_full_workflow)
    if hasattr(args, "auto_bootstrap"):
        cfg.auto_bootstrap = bool(args.auto_bootstrap)
    if hasattr(args, "python_exec"):
        cfg.python_exec = str(args.python_exec or "").strip()
    if hasattr(args, "tgi_docker_image"):
        cfg.tgi_docker_image = str(args.tgi_docker_image or cfg.tgi_docker_image).strip()
    if hasattr(args, "tgi_model_id"):
        cfg.tgi_model_id = str(args.tgi_model_id or cfg.tgi_model_id).strip()
    if hasattr(args, "tgi_platform"):
        cfg.tgi_platform = str(args.tgi_platform or "auto").strip().lower()
    if hasattr(args, "tgi_mode"):
        cfg.tgi_mode = str(args.tgi_mode or "docker").strip().lower()
    if hasattr(args, "tgi_port"):
        cfg.tgi_port = max(1, int(args.tgi_port))
    if hasattr(args, "tgi_port_policy"):
        cfg.tgi_port_policy = str(args.tgi_port_policy or "reuse_or_allocate").strip().lower()
    if hasattr(args, "tgi_port_range"):
        cfg.tgi_port_range = str(args.tgi_port_range or "8080-8090").strip() or "8080-8090"
    if hasattr(args, "tgi_reuse_existing"):
        cfg.tgi_reuse_existing = bool(args.tgi_reuse_existing)
    if hasattr(args, "tgi_health_path"):
        health = str(args.tgi_health_path or "/health").strip() or "/health"
        cfg.tgi_health_path = health if health.startswith("/") else f"/{health}"
    if hasattr(args, "tgi_generate_path"):
        generate_path = str(args.tgi_generate_path or "/generate").strip() or "/generate"
        cfg.tgi_generate_path = generate_path if generate_path.startswith("/") else f"/{generate_path}"
    if hasattr(args, "use_bootstrapped_venv"):
        cfg.use_bootstrapped_venv = bool(args.use_bootstrapped_venv)
    if hasattr(args, "already_bootstrapped"):
        cfg.already_bootstrapped = bool(args.already_bootstrapped)
    if hasattr(args, "all_done_repro_runs"):
        cfg.all_done_repro_runs = max(1, int(args.all_done_repro_runs))
    if hasattr(args, "all_done_max_runtime_sec"):
        cfg.all_done_max_runtime_sec = max(60, int(args.all_done_max_runtime_sec))
    if hasattr(args, "all_done_require_mp_if_key_present"):
        cfg.all_done_require_mp_if_key_present = bool(args.all_done_require_mp_if_key_present)
    if hasattr(args, "workflow_profile"):
        cfg.workflow_profile = str(args.workflow_profile or "strict_full").strip() or "strict_full"
    if hasattr(args, "finetune_gate_policy"):
        cfg.finetune_gate_policy = str(args.finetune_gate_policy or "absolute_uplift").strip().lower()
    if hasattr(args, "finetune_ceiling_threshold"):
        cfg.finetune_ceiling_threshold = max(0.0, min(1.0, float(args.finetune_ceiling_threshold)))
    if hasattr(args, "finetune_min_slice_rows"):
        cfg.finetune_min_slice_rows = max(1, int(args.finetune_min_slice_rows))
    if hasattr(args, "finetune_min_support_articles"):
        cfg.finetune_min_support_articles = max(1, int(args.finetune_min_support_articles))
    if hasattr(args, "vault_compare"):
        cfg.vault_import_enabled = bool(args.vault_compare)
    if hasattr(args, "out") and str(getattr(args, "command", "")).strip() in {
        "process-train-universal",
        "process-finetune-system",
    }:
        out_path = str(getattr(args, "out", "") or "").strip()
        if out_path:
            cfg.processor_model_dir = str(Path(out_path).expanduser().resolve())
    _apply_run_profile_defaults(cfg)
    return cfg


def _service_cfg_from_args(args: argparse.Namespace) -> ServiceConfig:
    kwargs = {
        "data_dir": str(getattr(args, "data_dir", "~/.oarp_service") or "~/.oarp_service"),
        "jobs_db_path": str(getattr(args, "jobs_db", "") or ""),
        "host": str(getattr(args, "host", "127.0.0.1") or "127.0.0.1"),
        "port": int(getattr(args, "port", 8787) or 8787),
        "worker_poll_sec": float(getattr(args, "worker_poll_sec", 0.5) or 0.5),
        "materials_project_enabled": bool(getattr(args, "mp_enabled", True)),
        "materials_project_endpoint": str(
            getattr(args, "mp_endpoint", "https://api.materialsproject.org")
            or "https://api.materialsproject.org"
        ),
        "materials_project_scope": str(getattr(args, "mp_scope", "summary_thermo") or "summary_thermo"),
    }
    mp_api_key = str(getattr(args, "mp_api_key", "") or "").strip()
    if mp_api_key:
        kwargs["materials_project_api_key"] = mp_api_key
    return ServiceConfig(
        **kwargs,
    )


def _load_json_file(path: str | Path) -> dict:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json payload must be an object: {path}")
    return payload


def _summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "point_count": 0,
            "consensus_entity_count": 0,
            "outlier_count": 0,
            "mean_entropy": 0.0,
        }
    return {
        "point_count": int(len(frame)),
        "consensus_entity_count": int(frame["consensus_entity"].nunique()),
        "outlier_count": int(frame["is_outlier"].sum()),
        "mean_entropy": float(frame["entropy"].mean()),
    }


def _maybe_reexec_into_bootstrap_venv(raw_argv: list[str], cfg: RunConfig) -> int | None:
    if not bool(cfg.use_bootstrapped_venv):
        return None
    if bool(cfg.already_bootstrapped):
        return None
    venv_python = cfg.as_path() / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return None
    try:
        current = Path(sys.executable).expanduser().resolve()
        target = venv_python.resolve()
    except Exception:
        return None
    if current == target:
        return None
    cmd = [str(target), "-m", "oarp", *raw_argv]
    if "--already-bootstrapped" not in raw_argv:
        cmd.append("--already-bootstrapped")
    completed = subprocess.run(cmd)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)

    if args.command == "run":
        cfg = _cfg_from_args(args, args.out)
        result = run_pipeline(
            spec_path=args.spec,
            query=args.query,
            cfg=cfg,
        )
        _print_payload(
            {
                "status": "ok",
                "run_dir": str(result.run_dir),
                "report_path": str(result.report_path),
                "consensus_path": str(result.consensus_path),
                "ranking_path": str(result.ranking_path),
                "validation_reasons_path": str(result.validation_reasons_path),
                "metrics_path": str(result.metrics_path),
                "extraction_votes_path": str(result.extraction_votes_path) if result.extraction_votes_path else "",
                "extraction_calibration_path": str(result.extraction_calibration_path)
                if result.extraction_calibration_path
                else "",
                "processor_eval_metrics_path": str(result.processor_eval_metrics_path)
                if result.processor_eval_metrics_path
                else "",
            }
        )
        return 0

    if args.command == "run-full":
        cfg = _cfg_from_args(args, args.out)
        reexec_code = _maybe_reexec_into_bootstrap_venv(raw_argv, cfg)
        if reexec_code is not None:
            return int(reexec_code)
        result = run_full_workflow(
            spec_path=args.spec,
            query=args.query,
            cfg=cfg,
        )
        _print_payload(
            {
                "status": "ok",
                "run_dir": str(result.run_dir),
                "preflight_path": str(result.preflight_path),
                "bootstrap_path": str(result.bootstrap_path) if result.bootstrap_path else "",
                "full_metrics_path": str(result.metrics_path),
                "run_metrics_path": str(result.run_result.metrics_path),
                "report_path": str(result.run_result.report_path),
                "processor_eval": result.processor_eval,
                "gate_status": result.gate_status,
                "processor_models": {key: str(path) for key, path in result.processor_models.items()},
                "knowledge_paths": {key: str(path) for key, path in result.knowledge_paths.items()},
            }
        )
        return 0

    if args.command == "validate-all-done":
        cfg = _cfg_from_args(args, args.out)
        reexec_code = _maybe_reexec_into_bootstrap_venv(raw_argv, cfg)
        if reexec_code is not None:
            return int(reexec_code)
        validation_cfg = {
            "gold_dir": str(args.gold),
            "shadow_gold_dir": str(args.shadow_gold or "").strip(),
            "gold_context_dir": str(args.gold_context or "").strip(),
            "precision_gate": float(args.precision_gate),
            "context_completeness_gate": float(args.context_completeness_gate),
            "context_precision_gate": float(args.context_precision_gate),
        }
        result = run_all_done_validation(
            spec_path=args.spec,
            query=args.query,
            cfg=cfg,
            validation_cfg=validation_cfg,
        )
        _print_payload(
            {
                "status": "ok" if result.ok else "failed",
                "ok": bool(result.ok),
                "run_dir": str(result.run_dir),
                "report_path": str(result.report_path),
                "json_path": str(result.json_path),
                "repro_compare_path": str(result.repro_compare_path),
                "gates": result.gates,
                "benchmark_summary": result.benchmark_summary,
                "artifacts": result.artifacts,
            }
        )
        return 0 if result.ok else 1

    if args.command == "release-validate-v1":
        cfg = _cfg_from_args(args, args.out)
        cfg.cpu_strict_profile = True if not bool(cfg.cpu_strict_profile) else cfg.cpu_strict_profile
        cfg.cpu_probe = True if not bool(cfg.cpu_probe) else cfg.cpu_probe
        reexec_code = _maybe_reexec_into_bootstrap_venv(raw_argv, cfg)
        if reexec_code is not None:
            return int(reexec_code)
        validation_cfg = {
            "spec_path": str(args.spec),
            "query": str(args.query),
            "gold_dir": str(args.gold),
            "shadow_gold_dir": str(args.shadow_gold or "").strip(),
            "gold_context_dir": str(args.gold_context or "").strip(),
            "precision_gate": float(args.precision_gate),
            "context_completeness_gate": float(args.context_completeness_gate),
            "context_precision_gate": float(args.context_precision_gate),
        }
        result = validate_release_v1(
            run_dir=args.out,
            cfg=cfg,
            validation_cfg=validation_cfg,
        )
        _print_payload(
            {
                "status": "ok" if result.all_done_pass else "failed",
                "all_done_pass": bool(result.all_done_pass),
                "json_path": str(result.json_path),
                "report_path": str(result.report_path),
                "gate_map": result.gate_map,
                "artifacts": result.artifact_map,
                "repro_summary": result.repro_summary,
                "remediation": result.remediation,
            }
        )
        return 0 if result.all_done_pass else 1

    if args.command == "validate-v2-publish":
        cfg = _cfg_from_args(args, args.out)
        cfg.run_profile = str(cfg.run_profile or "v2_publish_1k")
        cfg.vault_export_enabled = True
        cfg.vault_import_enabled = bool(args.vault_compare)
        cfg.finetune_gate_policy = str(cfg.finetune_gate_policy or "ceiling_aware")
        cfg.cpu_strict_profile = True if not bool(cfg.cpu_strict_profile) else cfg.cpu_strict_profile
        cfg.cpu_probe = True if not bool(cfg.cpu_probe) else cfg.cpu_probe
        reexec_code = _maybe_reexec_into_bootstrap_venv(raw_argv, cfg)
        if reexec_code is not None:
            return int(reexec_code)
        validation_cfg = {
            "gold_dir": str(args.gold),
            "shadow_gold_dir": str(args.shadow_gold or "").strip(),
            "gold_context_dir": str(args.gold_context or "").strip(),
            "precision_gate": float(args.precision_gate),
            "context_completeness_gate": float(args.context_completeness_gate),
            "context_precision_gate": float(args.context_precision_gate),
            "vault_compare": bool(args.vault_compare),
        }
        result = validate_v2_publish(
            spec_path=args.spec,
            query=args.query,
            cfg=cfg,
            validation_cfg=validation_cfg,
        )
        _print_payload(
            {
                "status": "ok" if result.all_done_pass else "failed",
                "all_done_pass": bool(result.all_done_pass),
                "json_path": str(result.json_path),
                "report_path": str(result.report_path),
                "gate_map": result.gate_map,
                "artifacts": result.artifact_map,
                "repro_summary": result.repro_summary,
                "remediation": result.remediation,
            }
        )
        return 0 if result.all_done_pass else 1

    if args.command == "discover":
        spec = load_topic_spec(args.spec)
        query = ensure_query(spec, args.query)
        cfg = _cfg_from_args(args, args.out)

        from oarp.pipeline import _initialize_run_state  # local import to keep public API clean

        _initialize_run_state(args.spec, spec, query, cfg)
        index = discover(spec=spec, query=query, cfg=cfg)

        payload = {
            "status": "ok",
            "articles": len(index.frame),
            "path": str(index.parquet_path),
            "ranking_path": str(cfg.as_path() / "artifacts" / "articles_ranked.parquet"),
        }
        if args.emit_debug_ranking:
            ranked_path = cfg.as_path() / "artifacts" / "articles_ranked.parquet"
            if ranked_path.exists():
                ranked = pd.read_parquet(ranked_path)
                payload["top_ranked"] = ranked.head(10)[
                    ["title", "provider", "discovery_score"]
                ].to_dict(orient="records")
        _print_payload(payload)
        return 0

    if args.command == "acquire":
        cfg = _cfg_from_args(args, args.run)
        index = acquire(cfg=cfg)
        _print_payload({"status": "ok", "documents": len(index.frame), "path": str(index.parquet_path)})
        return 0

    if args.command == "extract":
        cfg = _cfg_from_args(args, args.run)
        spec, _ = _load_spec_from_run(cfg.as_path())
        engines = [item.strip() for item in args.engines.split(",") if item.strip()]
        result = extraction_stage.extract(spec=spec, cfg=cfg, engines=extraction_stage.default_engines(engines))
        _print_payload(
            {
                "status": "ok",
                "points": len(result.points),
                "provenance": len(result.provenance),
                "points_path": str(result.points_path),
                "assembled_points_path": str(result.assembled_points_path),
                "extractor_mode": cfg.extractor_mode,
                "extraction_votes_path": str(cfg.as_path() / "artifacts" / "extraction_votes.parquet")
                if (cfg.as_path() / "artifacts" / "extraction_votes.parquet").exists()
                else "",
            }
        )
        return 0

    if args.command == "validate":
        cfg = _cfg_from_args(args, args.run)
        spec, _ = _load_spec_from_run(cfg.as_path())
        result = validation_stage.validate(spec=spec, cfg=cfg)
        _print_payload(
            {
                "status": "ok",
                "accepted": len(result.accepted),
                "rejected": len(result.rejected),
                "accepted_path": str(result.accepted_path),
                "validation_reasons_path": str(result.validation_reasons_path),
                "extraction_calibration_path": str(result.extraction_calibration_path)
                if result.extraction_calibration_path
                else "",
            }
        )
        return 0

    if args.command == "consensus":
        cfg = _cfg_from_args(args, args.run)
        spec, _ = _load_spec_from_run(cfg.as_path())
        result = consensus_stage.build(spec=spec, cfg=cfg)
        _print_payload(
            {
                "status": "ok",
                "points": len(result.points),
                "summary": result.summary,
                "path": str(result.parquet_path),
            }
        )
        return 0

    if args.command in {"plot", "report"}:
        cfg = _cfg_from_args(args, args.run)
        spec, _ = _load_spec_from_run(cfg.as_path())
        consensus_path = cfg.as_path() / "artifacts" / "consensus_points.parquet"
        if not consensus_path.exists():
            raise FileNotFoundError(
                f"missing consensus artifact: {consensus_path}. run `oarp consensus --run {cfg.as_path()}` first."
            )
        frame = pd.read_parquet(consensus_path)
        cset = ConsensusSet(points=frame, summary=_summary(frame), parquet_path=consensus_path)
        output = render_stage.render(spec=spec, consensus_set=cset, cfg=cfg)
        _print_payload(
            {
                "status": "ok",
                "report_path": str(output.report_path),
                "plot_paths": [str(path) for path in output.plot_paths],
                "citation_table_path": str(output.citation_table_path),
            }
        )
        return 0

    if args.command in {"bootstrap", "bootstrap-strict"}:
        cfg = _cfg_from_args(args, args.out)
        result = bootstrap_runtime(cfg)
        _print_payload(
            {
                "status": "ok" if result.ok else "failed",
                "ok": bool(result.ok),
                "python_executable": result.python_executable,
                "target_python_executable": result.target_python_executable,
                "venv_path": str(result.venv_path),
                "tgi_status": result.tgi_status,
                "selected_tgi_platform": result.selected_tgi_platform,
                "tgi_emulation_used": result.tgi_emulation_used,
                "reexec_required": result.reexec_required,
                "step_count": len(result.steps),
                "report_path": str(result.report_path),
            }
        )
        return 0

    if args.command == "preflight":
        cfg = _cfg_from_args(args, args.run)
        result = preflight_strict(
            cfg,
            python_exec=(cfg.python_exec or None),
            check_tgi_generate=bool(args.check_tgi_generate),
        )
        _print_payload(
            {
                "status": "ok" if result.ok else "failed",
                "ok": bool(result.ok),
                "check_count": len(result.checks),
                "target_python_executable": result.target_python_executable,
                "report_path": str(result.report_path),
                "checks": result.checks,
            }
        )
        return 0

    if args.command == "tgi":
        cfg = _cfg_from_args(args, args.run)
        if args.tgi_cmd == "status":
            _print_payload({"status": "ok", **tgi_status(cfg)})
            return 0
        if args.tgi_cmd == "start":
            result = tgi_start(cfg)
            _print_payload({"status": "ok" if result.get("ok") else "failed", **result})
            return 0
        if args.tgi_cmd == "stop":
            result = tgi_stop(cfg)
            _print_payload({"status": "ok" if result.get("ok") else "failed", **result})
            return 0
        parser.error(f"unknown tgi subcommand: {args.tgi_cmd}")
        return 2

    if args.command == "benchmark":
        spec = load_topic_spec(args.spec)
        run_dir = args.run or args.out
        result = run_benchmark(
            spec=spec,
            gold_dir=args.gold,
            out_dir=args.out,
            run_dir=run_dir,
            shadow_gold_dir=args.shadow_gold or None,
            strict_gold=bool(args.strict_gold),
            gold_context_dir=args.gold_context or None,
            context_completeness_gate=float(args.context_completeness_gate),
            context_precision_gate=float(args.context_precision_gate),
        )
        _print_payload(
            {
                "status": "ok",
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "threshold_met": result.threshold_met,
                "report_path": str(result.report_path),
                "shadow_metrics_path": str(result.shadow_metrics_path) if result.shadow_metrics_path else "",
                "context_report_path": str(result.context_report_path) if result.context_report_path else "",
                "mp_json_path": str(result.mp_json_path) if result.mp_json_path else "",
                "mp_report_path": str(result.mp_report_path) if result.mp_report_path else "",
            }
        )
        return 0

    if args.command == "benchmark-extraction":
        spec = load_topic_spec(args.spec)
        payload = run_extraction_benchmark(
            suite=args.suite,
            spec=spec,
            run_dir=args.run,
            out_dir=args.out,
            gold_dir=args.gold or None,
        )
        _print_payload({"status": "ok", **payload})
        return 0

    if args.command == "benchmark-processing":
        payload = run_processing_benchmark(
            suite=args.suite,
            run_dir=args.run,
            out_dir=args.out,
        )
        _print_payload({"status": "ok", **payload})
        return 0

    if args.command in {"process-train-universal", "processor-train-gnn", "processor-train-tabular", "processor-finetune", "process-finetune-system", "processor-eval"}:
        cfg = _cfg_from_args(args, args.run)
        artifacts = cfg.as_path() / "artifacts"

        def _load_processing_dataset(dataset_hint: str = "phase_events"):
            dataset_name = str(dataset_hint or "phase_events").strip().lower()
            source_path = artifacts / "phase_events.parquet"
            if dataset_name not in {"phase_events", "validated_points"}:
                source_path = Path(dataset_hint).expanduser().resolve()
            elif dataset_name == "validated_points":
                source_path = artifacts / "validated_points.parquet"
            if not source_path.exists():
                raise FileNotFoundError(f"processing dataset source not found: {source_path}")
            mp_path = artifacts / "materials_project_enriched_points.parquet"
            mp_data = pd.read_parquet(mp_path) if mp_path.exists() else None
            return build_processing_dataset_api(points=source_path, mp_data=mp_data, aux_data=None, cfg=cfg)

        if args.command == "processor-train-gnn":
            cfg.gnn_epochs = max(1, int(args.epochs))
            cfg.gnn_lr = max(1e-6, float(args.lr))
            cfg.gnn_hidden_dim = max(8, int(args.hidden_dim))
            cfg.gnn_layers = max(1, int(args.layers))
            cfg.gnn_dropout = max(0.0, min(0.9, float(args.dropout)))
            dataset = _load_processing_dataset("phase_events")
            model_ref = train_gnn_base_api(dataset, cfg)
            _print_payload(
                {
                    "status": "ok",
                    "dataset_path": str(dataset.path),
                    "model_type": model_ref.model_type,
                    "model_dir": str(model_ref.model_dir),
                    "model_path": str(model_ref.model_path),
                    "metadata_path": str(model_ref.metadata_path),
                    "metrics_path": str(model_ref.metrics_path),
                }
            )
            return 0

        if args.command == "processor-train-tabular":
            dataset = _load_processing_dataset("phase_events")
            gnn_dir = Path(args.base_gnn).expanduser().resolve()
            if gnn_dir.is_file():
                gnn_dir = gnn_dir.parent
            gnn_ref = ProcessorModelRef(
                model_dir=gnn_dir,
                model_path=gnn_dir / "gnn_model.pt",
                metadata_path=gnn_dir / "gnn_model_meta.json",
                metrics_path=cfg.as_path() / "artifacts" / "gnn_train_metrics.json",
                model_type="pyg_hetero_gnn",
            )
            cfg.tabular_model = str(args.model or "xgboost").strip().lower()
            model_ref = train_tabular_head_api(dataset, gnn_ref, cfg)
            _print_payload(
                {
                    "status": "ok",
                    "dataset_path": str(dataset.path),
                    "model_type": model_ref.model_type,
                    "model_dir": str(model_ref.model_dir),
                    "model_path": str(model_ref.model_path),
                    "metadata_path": str(model_ref.metadata_path),
                    "metrics_path": str(model_ref.metrics_path),
                }
            )
            return 0

        if args.command == "processor-finetune":
            dataset = _load_processing_dataset("phase_events")
            cfg.finetune_target_phase = str(args.target or "NiSi")
            cfg.finetune_max_thickness_nm = float(args.max_thickness_nm)
            cfg.gnn_epochs = max(1, int(args.epochs))
            base_dir = Path(args.base_model).expanduser().resolve()
            if base_dir.is_file():
                base_dir = base_dir.parent
            base_ref = ProcessorModelRef(
                model_dir=base_dir,
                model_path=base_dir / "processor_model.pkl",
                metadata_path=base_dir / "processor_model_meta.json",
                metrics_path=cfg.as_path() / "artifacts" / "tabular_train_metrics.json",
                model_type="graph_tabular_fusion",
            )
            model_ref = finetune_nisi_sub200_api(base_ref, dataset, cfg)
            _print_payload(
                {
                    "status": "ok",
                    "dataset_path": str(dataset.path),
                    "target_phase": cfg.finetune_target_phase,
                    "max_thickness_nm": cfg.finetune_max_thickness_nm,
                    "model_dir": str(model_ref.model_dir),
                    "model_path": str(model_ref.model_path),
                    "metadata_path": str(model_ref.metadata_path),
                    "metrics_path": str(model_ref.metrics_path),
                    "finetune_slice_path": str(cfg.as_path() / "artifacts" / "finetune_slice.parquet"),
                }
            )
            return 0

        if args.command == "processor-eval":
            payload = evaluate_processor_api(run_dir=args.run, cfg=cfg)
            _print_payload({"status": "ok", "suite": str(args.suite), **payload})
            return 0

        if args.command == "process-train-universal":
            dataset = _load_processing_dataset(args.datasets)
            cfg.processor_model_dir = str(Path(args.out).expanduser().resolve())
            model_ref = train_universal_processor_api(dataset, cfg)
            _print_payload(
                {
                    "status": "ok",
                    "dataset_path": str(dataset.path),
                    "model_type": model_ref.model_type,
                    "model_dir": str(model_ref.model_dir),
                    "model_path": str(model_ref.model_path),
                    "metadata_path": str(model_ref.metadata_path),
                    "metrics_path": str(model_ref.metrics_path),
                }
            )
            return 0

        if args.command == "process-finetune-system":
            dataset = _load_processing_dataset("phase_events")
            cfg.processor_model_dir = str(Path(args.out).expanduser().resolve())
            cfg.plugin_id = str(args.system_id or "").strip()
            _ = processing_stage.load_processor_model(args.base_model)
            base_dir = Path(args.base_model).expanduser().resolve()
            if base_dir.is_file():
                base_dir = base_dir.parent
            base_ref = ProcessorModelRef(
                model_dir=base_dir,
                model_path=base_dir / "processor_model.pkl",
                metadata_path=base_dir / "processor_model_meta.json",
                metrics_path=cfg.as_path() / "artifacts" / "processor_eval_metrics.json",
            )
            model_ref = finetune_processor_for_system_api(base_ref, dataset, cfg)
            _print_payload(
                {
                    "status": "ok",
                    "system_id": args.system_id,
                    "dataset_path": str(dataset.path),
                    "model_type": model_ref.model_type,
                    "model_dir": str(model_ref.model_dir),
                    "model_path": str(model_ref.model_path),
                    "metadata_path": str(model_ref.metadata_path),
                    "metrics_path": str(model_ref.metrics_path),
                }
            )
            return 0

    if args.command == "knowledge":
        bundle = build_knowledge(args.run)
        _print_payload(
            {
                "status": "ok",
                "phase_events_path": str(bundle.phase_events_path),
                "condition_graph_path": str(bundle.condition_graph_path),
                "quality_outcomes_path": str(bundle.quality_outcomes_path),
                "phase_event_count": bundle.phase_event_count,
                "condition_edge_count": bundle.condition_edge_count,
                "quality_outcome_count": bundle.quality_outcome_count,
            }
        )
        return 0

    if args.command == "graph-build":
        cfg = _cfg_from_args(args, args.run)
        artifacts = cfg.as_path() / "artifacts"
        source_token = str(args.source or "phase_events").strip().lower()
        if source_token == "phase_events":
            source_path = artifacts / "phase_events.parquet"
        elif source_token == "validated_points":
            source_path = artifacts / "validated_points.parquet"
        elif source_token == "processor_training_rows":
            source_path = artifacts / "processor_training_rows.parquet"
        else:
            source_path = Path(args.source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"graph-build source not found: {source_path}")

        mp_path = artifacts / "materials_project_enriched_points.parquet"
        mp_data = pd.read_parquet(mp_path) if mp_path.exists() else None
        dataset = build_processing_dataset_api(points=source_path, mp_data=mp_data, aux_data=None, cfg=cfg)

        article_graph = build_article_process_graph_api(dataset.path, cfg)
        concept_graph = build_global_concept_graph_api(dataset.path, mp_data, cfg)
        bridge_graph = build_bridge_edges_api(article_graph, concept_graph, cfg)
        graph_audit = audit_dual_graph_api(run_dir=cfg.as_path(), cfg=cfg)

        _print_payload(
            {
                "status": "ok" if graph_audit.ok else "failed",
                "graph_architecture": cfg.graph_architecture,
                "dataset_path": str(dataset.path),
                "article_process_nodes_path": str(article_graph.nodes_path),
                "article_process_edges_path": str(article_graph.edges_path),
                "concept_nodes_path": str(concept_graph.nodes_path),
                "concept_edges_path": str(concept_graph.edges_path),
                "bridge_edges_path": str(bridge_graph.edges_path),
                "bridge_weight_audit_path": str(bridge_graph.audit_path),
                "graph_audit_json": str(graph_audit.json_path),
                "graph_audit_md": str(graph_audit.report_path),
                "graph_audit_ok": bool(graph_audit.ok),
                "graph_audit_issues": graph_audit.issues,
            }
        )
        return 0 if graph_audit.ok else 1

    if args.command == "graph-audit":
        cfg = _cfg_from_args(args, args.run)
        result = audit_dual_graph_api(run_dir=cfg.as_path(), cfg=cfg)
        _print_payload(
            {
                "status": "ok" if result.ok else "failed",
                "ok": bool(result.ok),
                "issues": result.issues,
                "summary": result.summary,
                "json_path": str(result.json_path),
                "report_path": str(result.report_path),
            }
        )
        return 0 if result.ok else 1

    if args.command == "vault-export":
        cfg = _cfg_from_args(args, args.run)
        cfg.vault_export_enabled = True
        cfg.vault_profile = str(args.profile or "per_run_v1")
        out_dir = str(args.out or "").strip()
        if not out_dir:
            out_dir = str(cfg.as_path() / "outputs" / "vault")
        result = export_obsidian_vault_api(run_dir=str(cfg.as_path()), out_dir=out_dir, cfg=cfg)
        _print_payload(
            {
                "status": "ok",
                "vault_path": str(result.vault_path),
                "index_path": str(result.index_path),
                "note_counts_by_type": result.note_counts_by_type,
                "link_count": int(result.link_count),
                "warnings": result.warnings,
            }
        )
        return 0

    if args.command == "vault-import":
        cfg = _cfg_from_args(args, args.run)
        cfg.vault_import_enabled = True
        cfg.vault_import_mode = str(args.mode or "soft_supervision")
        result = import_obsidian_vault_api(vault_dir=args.vault, run_dir=str(cfg.as_path()), cfg=cfg)
        supervision = apply_vault_soft_supervision(
            run_dir=str(cfg.as_path()),
            vault_links=result.link_deltas,
            cfg=cfg,
        )
        _print_payload(
            {
                "status": "ok",
                "parsed_links": int(len(result.parsed_links)),
                "delta_links": int(len(result.link_deltas)),
                "soft_constraints_path": str(result.soft_constraints_path),
                "parsed_links_path": str(result.parsed_links_path),
                "delta_path": str(result.delta_path),
                "audit_path": str(result.audit_path),
                "point_supervision_rows": int(len(supervision)),
                "conflicts": result.conflicts,
            }
        )
        return 0

    if args.command == "vault-benchmark":
        cfg = _cfg_from_args(args, args.run)
        result = benchmark_vault_alignment_api(run_dir=str(cfg.as_path()), vault_dir=args.vault, cfg=cfg)
        _print_payload(
            {
                "status": "ok",
                "precision": result.precision,
                "recall": result.recall,
                "f1": result.f1,
                "vault_coverage": result.vault_coverage,
                "vault_link_density": result.vault_link_density,
                "vault_import_delta_rate": result.vault_import_delta_rate,
                "json_path": str(result.json_path),
                "report_path": str(result.report_path),
            }
        )
        return 0

    if args.command == "mp-enrich":
        cfg = _cfg_from_args(args, args.run)
        spec, _ = _load_spec_from_run(cfg.as_path())
        result = mp_enrichment_stage.enrich(spec=spec, cfg=cfg)
        _print_payload(
            {
                "status": "ok",
                "enriched_points": len(result.enriched_points),
                "materials": len(result.materials),
                "point_links": len(result.point_links),
                "queries": len(result.query_log),
                "enriched_points_path": str(result.enriched_points_path),
                "materials_path": str(result.materials_path),
                "point_links_path": str(result.point_links_path),
                "query_log_path": str(result.query_log_path),
            }
        )
        return 0

    if args.command == "recipe-generate":
        req_payload = _load_json_file(args.request_file)
        if str(args.gate_profile or "").strip():
            req_payload["gate_profile"] = str(args.gate_profile).strip()
        request = RecipeGenerateRequest.model_validate(req_payload)
        mp_api_key = str(args.mp_api_key or "").strip() or str(os.getenv("MP_API_KEY", ""))
        result, artifacts = generate_recipes(
            run_dir=args.run,
            request=request,
            materials_project_enabled=bool(args.mp_enabled),
            materials_project_api_key=mp_api_key,
            materials_project_endpoint=str(args.mp_endpoint or "https://api.materialsproject.org"),
            mp_scope=str(args.mp_scope or "summary_thermo"),
            processor_model_path=str(args.processor_model or ""),
        )
        payload = result.model_dump(mode="python")
        payload["artifacts"] = {
            "recipe_candidates": str(artifacts.candidates_path),
            "recipe_ranked": str(artifacts.ranked_path),
            "recipe_cards": str(artifacts.cards_path),
            "gate_audit": str(artifacts.gate_audit_path),
            "safety_audit": str(artifacts.safety_audit_path),
            "materials_project_refs": str(artifacts.materials_project_refs_path),
            "recipe_result": str(artifacts.result_path),
        }
        _print_payload(payload)
        return 0

    if args.command == "service":
        if args.service_cmd == "start":
            cfg = _service_cfg_from_args(args)
            start_service(cfg)
            return 0
        parser.error(f"unknown service subcommand: {args.service_cmd}")
        return 2

    if args.command == "job":
        cfg = _service_cfg_from_args(args)
        if args.job_cmd == "submit-research":
            cfg_overrides = {}
            if str(args.cfg_overrides_file or "").strip():
                cfg_overrides = _load_json_file(args.cfg_overrides_file)
            ref = submit_research_index_job(
                spec_path=args.spec,
                cfg=cfg,
                query=str(args.query or ""),
                run_dir=str(args.run_dir or ""),
                cfg_overrides=cfg_overrides,
            )
            _print_payload({"status": "ok", "job_id": ref.job_id, "job_status": ref.status.value})
            return 0
        if args.job_cmd == "submit-recipe":
            req_payload = _load_json_file(args.request_file)
            request = RecipeGenerateRequest.model_validate(req_payload)
            ref = submit_recipe_generation_job(request, cfg)
            _print_payload({"status": "ok", "job_id": ref.job_id, "job_status": ref.status.value})
            return 0
        if args.job_cmd == "status":
            status = get_job_status(args.job_id, cfg)
            _print_payload(dump_job_payload(status, cfg))
            return 0
        if args.job_cmd == "artifacts":
            artifacts = get_job_artifacts(args.job_id, cfg)
            _print_payload({"status": "ok", "job_id": args.job_id, "artifacts": artifacts})
            return 0
        if args.job_cmd == "retry":
            row = create_retry_job(db_path=cfg.jobs_db(), job_id=args.job_id)
            _print_payload({"status": "ok", "job_id": row.job_id, "job_status": row.status.value})
            return 0
        if args.job_cmd == "run-worker":
            ran = run_pending_jobs(cfg, max_jobs=int(args.max_jobs))
            _print_payload({"status": "ok", "processed_jobs": ran})
            return 0
        parser.error(f"unknown job subcommand: {args.job_cmd}")
        return 2

    if args.command == "agents-inc-handoff":
        target = build_agents_inc_handoff(run_dir=args.run, out_path=args.out or None)
        _print_payload({"status": "ok", "handoff_path": str(target)})
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
