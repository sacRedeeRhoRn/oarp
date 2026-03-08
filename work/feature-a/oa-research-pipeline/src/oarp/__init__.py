"""OA Research Pipeline package."""

from oarp.topic_spec import TopicSpec, load_topic_spec

__all__ = ["TopicSpec", "load_topic_spec"]

# Import runtime-heavy APIs lazily so lightweight schema usage does not require
# full data dependencies at import time.
try:  # pragma: no cover - import failure depends on environment setup.
    from oarp.models import (  # noqa: F401
        AllDoneValidationResult,
        ArticleGraphBuildResult,
        BootstrapResult,
        BridgeBuildResult,
        ConceptGraphBuildResult,
        FullWorkflowResult,
        GraphAuditResult,
        GraphBuildResult,
        MPEvidenceSet,
        PreflightReport,
        ProcessorDataset,
        ProcessorEvalResultV2,
        ProcessorModelBundle,
        ProcessorModelRef,
        ReleaseValidationResult,
        RunConfig,
        VaultBenchmarkResult,
        VaultExportResult,
        VaultImportResult,
    )
    from oarp.pipeline import (
        acquire_fulltext,  # noqa: F401
        apply_vault_soft_supervision,  # noqa: F401
        audit_dual_graph,  # noqa: F401
        benchmark_vault_alignment,  # noqa: F401
        build_article_phase_graph,  # noqa: F401
        build_article_process_graph,  # noqa: F401
        build_bridge_edges,  # noqa: F401
        build_consensus,  # noqa: F401
        build_global_concept_graph,  # noqa: F401
        build_phase_spec,  # noqa: F401
        build_processing_dataset,  # noqa: F401
        discover_articles,  # noqa: F401
        enrich_with_materials_project,  # noqa: F401
        evaluate_extractor_quality,  # noqa: F401
        evaluate_processor,  # noqa: F401
        evaluate_processor_with_policy,  # noqa: F401
        export_obsidian_vault,  # noqa: F401
        extract_evidence,  # noqa: F401
        finetune_nisi_sub200,  # noqa: F401
        finetune_processor_for_system,  # noqa: F401
        generate_ranked_recipes,  # noqa: F401
        import_obsidian_vault,  # noqa: F401
        prepare_feature_cache,  # noqa: F401
        render_outputs,  # noqa: F401
        run_all_done_validation,  # noqa: F401
        run_full_workflow,  # noqa: F401
        run_pipeline,  # noqa: F401
        run_slm_extraction_swarm,  # noqa: F401
        run_tgi_slm_extraction,  # noqa: F401
        score_bridge_weight,  # noqa: F401
        train_gnn_base,  # noqa: F401
        train_processor_v1,  # noqa: F401
        train_tabular_head,  # noqa: F401
        train_universal_processor,  # noqa: F401
        validate_evidence,  # noqa: F401
        validate_extraction_precision,  # noqa: F401
        validate_release_v1,  # noqa: F401
        validate_v2_publish,  # noqa: F401
    )
    from oarp.service_api import (  # noqa: F401
        get_job_status,
        load_recipe_result,
        submit_recipe_generation_job,
        submit_research_index_job,
    )
    from oarp.service_models import (  # noqa: F401
        JobRef,
        JobStatusPayload,
        RecipeGenerateRequest,
        ServiceConfig,
    )
    from oarp.workflow import (  # noqa: F401
        bootstrap_runtime,
        preflight_strict,
        tgi_start,
        tgi_status,
        tgi_stop,
    )

    __all__.extend(
        [
            "RunConfig",
            "run_pipeline",
            "run_full_workflow",
            "run_all_done_validation",
            "run_slm_extraction_swarm",
            "run_tgi_slm_extraction",
            "discover_articles",
            "acquire_fulltext",
            "extract_evidence",
            "export_obsidian_vault",
            "import_obsidian_vault",
            "apply_vault_soft_supervision",
            "benchmark_vault_alignment",
            "enrich_with_materials_project",
            "validate_evidence",
            "validate_extraction_precision",
            "evaluate_extractor_quality",
            "build_consensus",
            "render_outputs",
            "build_phase_spec",
            "build_article_phase_graph",
            "build_article_process_graph",
            "build_global_concept_graph",
            "build_bridge_edges",
            "score_bridge_weight",
            "audit_dual_graph",
            "build_processing_dataset",
            "train_gnn_base",
            "train_tabular_head",
            "train_universal_processor",
            "train_processor_v1",
            "finetune_nisi_sub200",
            "finetune_processor_for_system",
            "evaluate_processor",
            "evaluate_processor_with_policy",
            "generate_ranked_recipes",
            "MPEvidenceSet",
            "PreflightReport",
            "BootstrapResult",
            "FullWorkflowResult",
            "AllDoneValidationResult",
            "ReleaseValidationResult",
            "bootstrap_runtime",
            "preflight_strict",
            "tgi_start",
            "tgi_status",
            "tgi_stop",
            "ServiceConfig",
            "RecipeGenerateRequest",
            "JobRef",
            "JobStatusPayload",
            "submit_research_index_job",
            "submit_recipe_generation_job",
            "get_job_status",
            "load_recipe_result",
            "ProcessorDataset",
            "ProcessorModelRef",
            "ProcessorModelBundle",
            "ProcessorEvalResultV2",
            "GraphBuildResult",
            "ArticleGraphBuildResult",
            "ConceptGraphBuildResult",
            "BridgeBuildResult",
            "GraphAuditResult",
            "VaultExportResult",
            "VaultImportResult",
            "VaultBenchmarkResult",
            "prepare_feature_cache",
            "validate_release_v1",
            "validate_v2_publish",
        ]
    )
except Exception:
    pass
