from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from oarp.knowledge import KnowledgeBundle
from oarp.models import PreflightReport, ProcessorDataset, ProcessorModelRef, RunConfig, RunResult
from oarp.pipeline import _initialize_run_state, run_full_workflow
from oarp.runtime import ensure_run_layout, now_iso
from oarp.topic_spec import load_topic_spec


def _write_spec(path: Path) -> None:
    payload = {
        "topic_id": "full-workflow-test",
        "keywords": ["nickel silicide", "annealing"],
        "variables": [
            {"name": "thickness_nm", "aliases": ["thickness"], "unit": "nm", "datatype": "float"},
            {"name": "temperature_c", "aliases": ["temperature"], "unit": "c", "datatype": "float"},
        ],
        "entities": [
            {"name": "Ni2Si", "aliases": ["ni2si"]},
            {"name": "NiSi", "aliases": ["nisi"]},
            {"name": "NiSi2", "aliases": ["nisi2"]},
        ],
        "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
        "validation": {
            "min_confidence": 0.65,
            "required_provenance_fields": ["citation_url", "snippet", "locator"],
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_run_full_workflow_orchestrates_strict_stack(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "topic.yaml"
    _write_spec(spec_path)
    run_dir = tmp_path / "run"
    cfg = RunConfig(
        run_dir=run_dir,
        strict_full_workflow=True,
        auto_bootstrap=False,
        extractor_max_loop=1,
        processor_max_loop=1,
    )

    def _fake_preflight(local_cfg: RunConfig, python_exec: str | None = None, *, check_tgi_generate: bool = True) -> PreflightReport:  # noqa: ARG001
        layout = ensure_run_layout(local_cfg.as_path())
        report = layout["artifacts"] / "preflight_report.json"
        report.write_text(json.dumps({"ok": True, "checks": []}, indent=2) + "\n", encoding="utf-8")
        return PreflightReport(ok=True, checks=[], report_path=report, target_python_executable=str(python_exec or ""))

    def _fake_run_pipeline(spec_path: str, query: str, cfg: RunConfig) -> RunResult:
        spec = load_topic_spec(spec_path)
        _initialize_run_state(spec_path, spec, query, cfg)
        layout = ensure_run_layout(cfg.as_path())
        artifacts = layout["artifacts"]
        outputs = layout["outputs"]

        pd.DataFrame([{"request_id": "r1"}]).to_parquet(artifacts / "slm_requests.parquet", index=False)
        pd.DataFrame([{"request_id": "r1", "status": "ok"}]).to_parquet(
            artifacts / "slm_responses.parquet",
            index=False,
        )
        pd.DataFrame([{"point_id": "p1"}]).to_parquet(artifacts / "slm_points_voted.parquet", index=False)
        pd.DataFrame([{"point_id": "p1", "accepted": True}]).to_parquet(
            artifacts / "extraction_votes.parquet",
            index=False,
        )
        validated = pd.DataFrame(
            [
                {
                    "point_id": "p1",
                    "citation_url": "https://example.org/paper",
                    "snippet": "NiSi forms at 350 C for 30 nm films",
                    "locator": "line:10",
                }
            ]
        )
        validated.to_parquet(artifacts / "validated_points.parquet", index=False)
        pd.DataFrame(columns=["reason"]).to_parquet(artifacts / "validation_reasons.parquet", index=False)
        pd.DataFrame(
            [
                {"article_key": "a1", "provider": "localfs"},
                {"article_key": "a2", "provider": "openalex"},
            ]
        ).to_parquet(artifacts / "articles.parquet", index=False)
        pd.DataFrame([{"article_key": "a1", "provider": "localfs", "usable_text": True}]).to_parquet(
            artifacts / "documents.parquet",
            index=False,
        )
        pd.DataFrame([{"point_id": "p1"}]).to_parquet(artifacts / "evidence_points.parquet", index=False)
        pd.DataFrame([{"point_id": "p1"}]).to_parquet(artifacts / "provenance.parquet", index=False)
        pd.DataFrame([{"point_id": "c1"}]).to_parquet(artifacts / "consensus_points.parquet", index=False)
        (outputs / "report.md").write_text("# report\n", encoding="utf-8")
        (artifacts / "run_metrics.json").write_text(
            json.dumps({"created_at": now_iso(), "status": "ok"}, indent=2) + "\n",
            encoding="utf-8",
        )
        (artifacts / "stage_metrics.parquet").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"stage": "dummy"}]).to_parquet(artifacts / "stage_metrics.parquet", index=False)
        return RunResult(
            run_dir=cfg.as_path(),
            articles_path=artifacts / "articles.parquet",
            ranking_path=artifacts / "articles_ranked.parquet",
            documents_path=artifacts / "documents.parquet",
            evidence_path=artifacts / "evidence_points.parquet",
            provenance_path=artifacts / "provenance.parquet",
            validated_path=artifacts / "validated_points.parquet",
            validation_reasons_path=artifacts / "validation_reasons.parquet",
            consensus_path=artifacts / "consensus_points.parquet",
            report_path=outputs / "report.md",
            metrics_path=artifacts / "run_metrics.json",
            extraction_votes_path=artifacts / "extraction_votes.parquet",
        )

    def _fake_build_knowledge(_run_dir: str | Path) -> KnowledgeBundle:
        layout = ensure_run_layout(cfg.as_path())
        artifacts = layout["artifacts"]
        phase_events = pd.DataFrame(
            [
                {
                    "point_id": "p1",
                    "article_key": "a1",
                    "phase_label": "NiSi",
                    "film_material": "NiSi",
                    "substrate_material": "Si",
                    "substrate_orientation": "(111)",
                    "doping_state": "na_pure_ni",
                    "alloy_state": "na_pure_ni",
                    "method_family": "PVD",
                    "thickness_nm": 30.0,
                    "anneal_temperature_c": 350.0,
                    "film_quality_score_numeric": 0.9,
                }
            ]
        )
        phase_events_path = artifacts / "phase_events.parquet"
        condition_graph_path = artifacts / "condition_graph.parquet"
        quality_outcomes_path = artifacts / "quality_outcomes.parquet"
        phase_events.to_parquet(phase_events_path, index=False)
        pd.DataFrame([{"source_key": "a", "target_key": "b"}]).to_parquet(condition_graph_path, index=False)
        pd.DataFrame([{"event_id": "e1"}]).to_parquet(quality_outcomes_path, index=False)
        return KnowledgeBundle(
            phase_events_path=phase_events_path,
            condition_graph_path=condition_graph_path,
            quality_outcomes_path=quality_outcomes_path,
            phase_event_count=1,
            condition_edge_count=1,
            quality_outcome_count=1,
        )

    def _fake_build_processing_dataset(points, mp_data, aux_data, cfg: RunConfig):  # noqa: ANN001
        _ = (mp_data, aux_data)
        layout = ensure_run_layout(cfg.as_path())
        artifacts = layout["artifacts"]
        frame = pd.read_parquet(Path(points))
        out_path = artifacts / "processor_training_rows.parquet"
        frame.to_parquet(out_path, index=False)
        pd.DataFrame([{"graph_sample_id": "g1", "article_key": "a1"}]).to_parquet(
            artifacts / "graph_samples.parquet",
            index=False,
        )
        return ProcessorDataset(frame=frame, path=out_path)

    def _fake_model_ref(local_cfg: RunConfig, model_name: str, model_file: str, model_type: str) -> ProcessorModelRef:
        model_dir = local_cfg.as_path() / "artifacts" / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / model_file
        model_path.write_bytes(b"stub")
        meta = model_dir / "processor_model_meta.json"
        meta.write_text(json.dumps({"model": model_name}, indent=2) + "\n", encoding="utf-8")
        metrics = local_cfg.as_path() / "artifacts" / f"{model_name}_metrics.json"
        metrics.write_text(json.dumps({"ok": True}, indent=2) + "\n", encoding="utf-8")
        return ProcessorModelRef(
            model_dir=model_dir,
            model_path=model_path,
            metadata_path=meta,
            metrics_path=metrics,
            model_type=model_type,
        )

    def _fake_train_gnn(dataset: ProcessorDataset, local_cfg: RunConfig) -> ProcessorModelRef:
        _ = dataset
        return _fake_model_ref(local_cfg, "gnn_base", "gnn_model.pt", "pyg_hetero_gnn")

    def _fake_train_tab(dataset: ProcessorDataset, gnn_ref: ProcessorModelRef, local_cfg: RunConfig) -> ProcessorModelRef:
        _ = (dataset, gnn_ref)
        return _fake_model_ref(local_cfg, "tabular_head", "processor_model.pkl", "graph_tabular_fusion")

    def _fake_finetune(model_ref: ProcessorModelRef, dataset: ProcessorDataset, local_cfg: RunConfig) -> ProcessorModelRef:
        _ = (model_ref, dataset)
        slice_path = local_cfg.as_path() / "artifacts" / "finetune_slice.parquet"
        pd.DataFrame([{"phase_label": "NiSi", "thickness_nm": 120.0}]).to_parquet(slice_path, index=False)
        return _fake_model_ref(
            local_cfg,
            "finetune_nisi_sub200",
            "processor_model.pkl",
            "graph_tabular_fusion_finetune",
        )

    def _fake_eval(run_dir: str, cfg: RunConfig) -> dict:
        _ = run_dir
        payload = {
            "all_gates_pass": True,
            "gates": {
                "base_phase_macro_f1_ge_0_80": True,
                "base_ndcg_at_10_ge_0_75": True,
                "finetune_f1_uplift_ge_0_05": True,
                "finetune_ndcg_uplift_ge_0_05": True,
            },
        }
        metrics_path = cfg.as_path() / "artifacts" / "processor_eval_metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    monkeypatch.setattr("oarp.pipeline.preflight_strict", _fake_preflight)
    monkeypatch.setattr("oarp.pipeline.run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr("oarp.pipeline.knowledge.build_knowledge", _fake_build_knowledge)
    monkeypatch.setattr("oarp.pipeline.build_processing_dataset", _fake_build_processing_dataset)
    monkeypatch.setattr("oarp.pipeline.train_gnn_base", _fake_train_gnn)
    monkeypatch.setattr("oarp.pipeline.train_tabular_head", _fake_train_tab)
    monkeypatch.setattr("oarp.pipeline.finetune_nisi_sub200", _fake_finetune)
    monkeypatch.setattr("oarp.pipeline.evaluate_processor", _fake_eval)

    result = run_full_workflow(spec_path=str(spec_path), query="nickel silicide annealing", cfg=cfg)
    assert result.gate_status["all_ok"] is True
    assert result.metrics_path.exists()
    assert (run_dir / "artifacts" / "full_workflow_stage_metrics.parquet").exists()
    assert (run_dir / "artifacts" / "full_workflow_metrics.json").exists()
