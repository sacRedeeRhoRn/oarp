from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RunConfig:
    run_dir: str | Path
    python_exec: str = ""
    timeout_sec: int = 20
    max_per_provider: int = 25
    max_pages_per_provider: int = 60
    max_discovered_records: int = 100000
    saturation_window_pages: int = 8
    saturation_min_yield: float = 0.03
    min_pages_before_saturation: int = 20
    resume_discovery: bool = False
    max_downloads: int = 200
    local_repo_paths: list[str] = field(
        default_factory=lambda: ["/Users/moon.s.june/matGen/MODELING/collect/pdf_repo"]
    )
    local_repo_recursive: bool = True
    local_file_glob: str = "*.pdf"
    local_merge_mode: str = "union"
    local_max_files: int = 0
    local_require_readable: bool = True
    per_provider_cap: dict[str, int] = field(default_factory=dict)
    retries: int = 2
    backoff_sec: float = 1.5
    user_agent: str = "oarp/0.1.0 (+https://github.com/sacRedeeRhoRn/agents-inc)"
    allow_direct_crawl: bool = True
    require_fulltext_mime: bool = False
    min_text_length: int = 500
    acquire_workers: int = 8
    extract_workers: int = 4
    english_first: bool = True
    requests_per_second: float = 1.0
    random_seed: int = 17
    min_discovery_score: float = 0.0
    plugin_id: str = ""
    context_window_lines: int = 2
    point_assembler: str = "window-merge"
    context_assembler: str = "sentence-window"
    phase_schema_version: str = "v1"
    graph_core_mode: str = "per_article_typed"
    graph_architecture: str = "legacy_v1"
    concept_ontology_profile: str = "maximal"
    bridge_weight_policy: str = "deterministic_v1"
    bridge_weight_threshold: float = 0.15
    concept_gates: list[str] = field(
        default_factory=lambda: ["film", "substrate", "dopant", "precursor", "method", "thermo", "electronic"]
    )
    phase_require_elements: bool = True
    phase_require_stoich: bool = True
    phase_require_spacegroup: bool = True
    cache_mode: str = "run_local"
    shared_cache_root: str = ""
    cache_read_only: bool = False
    cache_ttl_hours: int = 168
    cpu_strict_profile: bool = False
    cpu_max_threads: int = 4
    cpu_probe: bool = False
    extractor_mode: str = "hybrid_rules"
    extractor_models: list[str] = field(
        default_factory=lambda: ["llama-3.1-8b-instruct", "gemma-2-9b-it", "phi-3-mini-4k-instruct"]
    )
    tgi_endpoint: str = ""
    tgi_models: list[str] = field(default_factory=list)
    tgi_workers: int = 2
    decoder: str = "jsonschema"
    slm_max_retries: int = 2
    slm_timeout_sec: float = 20.0
    slm_batch_size: int = 8
    slm_chunk_tokens: int = 384
    slm_overlap_tokens: int = 96
    slm_eval_split: str = "all"
    schema_decoder: str = "llguidance"
    vote_policy: str = "weighted"
    min_vote_confidence: float = 0.60
    extractor_gate_profile: str = "default"
    require_context_fields: bool = True
    emit_debug_ranking: bool = False
    emit_validation_metrics: bool = False
    emit_extraction_calibration: bool = False
    calibration_bins: int = 10
    min_support_per_bin: int = 1
    low_n_confidence_penalty: float = 0.25
    assembled_confidence_floor: float = 0.78
    context_completeness_gate: float = 0.70
    context_precision_gate: float = 0.80
    entity_alias_overrides: dict[str, str] = field(default_factory=dict)
    mp_enabled: bool = True
    mp_mode: str = "interpreter"
    mp_scope: str = "summary_thermo"
    mp_on_demand: bool = True
    mp_query_workers: int = 4
    mp_timeout_sec: float = 20.0
    mp_max_queries: int = 20000
    mp_cache_path: str = ""
    mp_formula_match_weight: float = 0.50
    mp_phase_match_weight: float = 0.30
    mp_stability_weight: float = 0.20
    gnn_hidden_dim: int = 64
    gnn_layers: int = 2
    gnn_dropout: float = 0.1
    gnn_epochs: int = 80
    gnn_lr: float = 1e-3
    tabular_model: str = "xgboost"
    finetune_target_phase: str = "NiSi"
    finetune_max_thickness_nm: float = 200.0
    system_finetune_dataset: str = ""
    system_eval_holdout_ratio: float = 0.30
    extractor_max_loop: int = 2
    processor_max_loop: int = 2
    strict_full_workflow: bool = True
    auto_bootstrap: bool = True
    tgi_docker_image: str = "ghcr.io/huggingface/text-generation-inference:2.3.1"
    tgi_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tgi_platform: str = "auto"
    tgi_mode: str = "docker"
    tgi_port: int = 8080
    tgi_health_path: str = "/health"
    tgi_generate_path: str = "/generate"
    workflow_profile: str = "strict_full"
    use_bootstrapped_venv: bool = True
    already_bootstrapped: bool = False
    all_done_repro_runs: int = 2
    all_done_max_runtime_sec: int = 43200
    all_done_require_mp_if_key_present: bool = True
    processor_model_dir: str = ""

    def as_path(self) -> Path:
        return Path(self.run_dir).expanduser().resolve()


@dataclass
class ArticleCandidate:
    provider: str
    source_id: str
    doi: str
    title: str
    abstract: str
    year: int | None
    venue: str
    article_type: str
    is_oa: bool
    oa_url: str
    source_url: str
    license: str
    language: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullTextRecord:
    article_key: str
    provider: str
    source_url: str
    local_path: str
    mime: str
    content_hash: str
    parse_status: str
    text_content: str


@dataclass
class ArticleIndex:
    frame: pd.DataFrame
    parquet_path: Path


@dataclass
class DocumentIndex:
    frame: pd.DataFrame
    parquet_path: Path


@dataclass
class EvidenceSet:
    points: pd.DataFrame
    provenance: pd.DataFrame
    points_path: Path
    provenance_path: Path
    assembled_points_path: Path | None = None
    extraction_coverage_path: Path | None = None
    context_dimensions_path: Path | None = None


@dataclass
class ValidatedEvidence:
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    accepted_path: Path
    rejected_path: Path
    validation_reasons_path: Path | None = None
    context_validation_path: Path | None = None
    warnings_path: Path | None = None
    extraction_calibration_path: Path | None = None


@dataclass
class ConsensusSet:
    points: pd.DataFrame
    summary: dict[str, Any]
    parquet_path: Path


@dataclass
class OutputBundle:
    report_path: Path
    plot_paths: list[Path]
    citation_table_path: Path


@dataclass
class RunResult:
    run_dir: Path
    articles_path: Path
    ranking_path: Path
    documents_path: Path
    evidence_path: Path
    provenance_path: Path
    validated_path: Path
    validation_reasons_path: Path
    consensus_path: Path
    report_path: Path
    metrics_path: Path
    mp_enriched_path: Path | None = None
    extraction_votes_path: Path | None = None
    extraction_calibration_path: Path | None = None
    processor_eval_metrics_path: Path | None = None


@dataclass
class PreflightReport:
    ok: bool
    checks: list[dict[str, Any]]
    report_path: Path
    target_python_executable: str = ""


@dataclass
class BootstrapResult:
    ok: bool
    python_executable: str
    venv_path: Path
    tgi_status: dict[str, Any]
    steps: list[dict[str, Any]]
    report_path: Path
    target_python_executable: str = ""
    selected_tgi_platform: str = ""
    tgi_emulation_used: bool = False
    reexec_required: bool = False


@dataclass
class FullWorkflowResult:
    run_dir: Path
    run_result: RunResult
    knowledge_paths: dict[str, Path]
    processor_models: dict[str, Path]
    processor_eval: dict[str, Any]
    gate_status: dict[str, bool]
    preflight_path: Path
    bootstrap_path: Path | None
    metrics_path: Path


@dataclass
class AllDoneValidationResult:
    ok: bool
    run_dir: Path
    report_path: Path
    json_path: Path
    repro_compare_path: Path
    gates: dict[str, bool]
    artifacts: dict[str, str]
    benchmark_summary: dict[str, Any]


@dataclass
class MPEvidenceSet:
    enriched_points: pd.DataFrame
    materials: pd.DataFrame
    point_links: pd.DataFrame
    query_log: pd.DataFrame
    enriched_points_path: Path
    materials_path: Path
    point_links_path: Path
    query_log_path: Path


@dataclass
class MaterialContext:
    substrate_material: str
    substrate_orientation: str
    orientation_family: str
    doping_state: str
    doping_elements: list[str]
    doping_composition: list[dict[str, Any]]
    alloy_state: str
    alloy_elements: list[str]
    alloy_composition: list[dict[str, Any]]
    context_confidence: float
    pure_ni_evidence: bool = False


@dataclass
class ExtractionVoteSet:
    votes: pd.DataFrame
    accepted_point_ids: set[str]
    votes_path: Path
    error_slices_path: Path


@dataclass
class ExtractionCalibration:
    frame: pd.DataFrame
    summary: dict[str, Any]
    path: Path


@dataclass
class ProcessorDataset:
    frame: pd.DataFrame
    path: Path


@dataclass
class ProcessorModelRef:
    model_dir: Path
    model_path: Path
    metadata_path: Path
    metrics_path: Path
    model_type: str = "graph_tabular_fusion"


@dataclass
class SLMExtractionRecord:
    request_id: str
    doc_id: str
    chunk_id: str
    model_id: str
    prompt_hash: str
    response_json: str
    latency_ms: float
    status: str
    retry_idx: int


@dataclass
class SLMPoint:
    point_id: str
    doc_id: str
    chunk_id: str
    model_id: str
    entity: str
    thickness_nm: float
    temperature_c: float
    vote_confidence: float
    citation_url: str
    locator: str
    snippet: str


@dataclass
class HeteroGraphSample:
    graph_sample_id: str
    article_key: str
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    tensor_path: str


@dataclass
class ProcessorPrediction:
    phase_logits: list[float]
    phase_prob: float
    quality_score: float
    recipe_rank_score: float
    uncertainty: float


@dataclass
class FineTuneSliceSpec:
    target_phase: str = "NiSi"
    max_thickness_nm: float = 200.0
    substrate_filter: list[str] = field(default_factory=list)
    doping_filter: list[str] = field(default_factory=list)


@dataclass
class PhaseSpec:
    phase_id: str
    reduced_formula: str
    elements_json: str
    stoichiometry_json: str
    spacegroup_symbol: str
    spacegroup_number: str
    phase_signature: str


@dataclass
class GraphBuildResult:
    nodes_path: Path
    edges_path: Path
    tensor_index_path: Path
    graph_sample_count: int


@dataclass
class ConceptNode:
    concept_id: str
    concept_type: str
    canonical_label: str
    aliases_json: str
    attributes_json: str


@dataclass
class ArticleProcessNode:
    article_key: str
    stage: str
    state_id: str
    attributes_json: str


@dataclass
class BridgeEdge:
    source_id: str
    target_id: str
    gate_type: str
    weight: float
    weight_components_json: str
    provenance_ref: str
    source_point_id: str


@dataclass
class PhaseInstanceV2:
    reduced_formula: str
    elements_json: str
    stoichiometry_json: str
    spacegroup_symbol: str
    spacegroup_number: str
    energy_above_hull: float = 0.0
    formation_energy: float = 0.0
    band_gap: float = 0.0
    is_metal: str = ""
    magnetic_ordering: str = ""


@dataclass
class ArticleGraphBuildResult:
    nodes_path: Path
    edges_path: Path
    node_count: int
    edge_count: int


@dataclass
class ConceptGraphBuildResult:
    nodes_path: Path
    edges_path: Path
    node_count: int
    edge_count: int


@dataclass
class BridgeBuildResult:
    edges_path: Path
    audit_path: Path
    edge_count: int
    candidate_count: int


@dataclass
class GraphAuditResult:
    ok: bool
    summary: dict[str, Any]
    issues: list[str]
    json_path: Path
    report_path: Path


@dataclass
class ProcessorModelBundle:
    gnn_ref: ProcessorModelRef
    tabular_ref: ProcessorModelRef
    finetune_ref: ProcessorModelRef
    eval_metrics: dict[str, Any]


@dataclass
class ReleaseValidationResult:
    all_done_pass: bool
    gate_map: dict[str, bool]
    artifact_map: dict[str, str]
    repro_summary: dict[str, Any]
    remediation: list[str]
    report_path: Path
    json_path: Path
