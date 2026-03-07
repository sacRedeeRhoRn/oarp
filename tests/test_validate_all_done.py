from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from oarp.benchmark import BenchmarkResult
from oarp.models import FullWorkflowResult, PreflightReport, RunConfig, RunResult
from oarp.pipeline import run_all_done_validation
from oarp.runtime import ensure_run_layout


def _write_spec(path: Path) -> None:
    payload = {
        "topic_id": "all-done-test",
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


def test_validate_all_done_emits_gate_report(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    spec_path = tmp_path / "topic.yaml"
    _write_spec(spec_path)

    cfg = RunConfig(
        run_dir=run_dir,
        strict_full_workflow=True,
        auto_bootstrap=False,
        all_done_repro_runs=1,
        tgi_mode="external",
    )

    def _fake_preflight(local_cfg: RunConfig, python_exec: str | None = None, *, check_tgi_generate: bool = True) -> PreflightReport:  # noqa: ARG001
        layout = ensure_run_layout(local_cfg.as_path())
        report = layout["artifacts"] / "preflight_report.json"
        report.write_text(json.dumps({"ok": True, "checks": []}, indent=2) + "\n", encoding="utf-8")
        return PreflightReport(ok=True, checks=[], report_path=report, target_python_executable=str(python_exec or ""))

    def _fake_run_full(spec_path: str, query: str, cfg: RunConfig) -> FullWorkflowResult:  # noqa: ARG001
        local_cfg = cfg
        layout = ensure_run_layout(local_cfg.as_path())
        artifacts = layout["artifacts"]
        outputs = layout["outputs"]

        (artifacts / "models" / "gnn_base").mkdir(parents=True, exist_ok=True)
        (artifacts / "models" / "tabular_head").mkdir(parents=True, exist_ok=True)
        (artifacts / "models" / "finetune_nisi_sub200").mkdir(parents=True, exist_ok=True)
        (artifacts / "models" / "gnn_base" / "gnn_model.pt").write_bytes(b"x")
        (artifacts / "models" / "tabular_head" / "processor_model.pkl").write_bytes(b"x")
        (artifacts / "models" / "finetune_nisi_sub200" / "processor_model.pkl").write_bytes(b"x")
        (outputs / "report.md").write_text("# report\n", encoding="utf-8")
        (artifacts / "processor_eval_metrics.json").write_text(
            json.dumps({"all_gates_pass": True}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validated = pd.DataFrame(
            [
                {
                    "point_id": "p1",
                    "citation_url": "https://example.org/paper",
                    "snippet": "NiSi forms at 350 C for 30 nm films.",
                    "locator": "line:10",
                }
            ]
        )
        validated.to_parquet(artifacts / "validated_points.parquet", index=False)

        run_metrics = {
            "counts": {
                "slm_requests": 4,
                "slm_responses": 4,
                "slm_points_voted": 2,
                "validated_points": 1,
            }
        }
        run_metrics_path = artifacts / "run_metrics.json"
        run_metrics_path.write_text(json.dumps(run_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        full_metrics = {
            "provider_counts": {"localfs": 2, "openalex": 1},
            "document_provider_counts": {"localfs": 2},
            "extractor_gate": {"provenance_complete_ratio": 1.0},
            "processor_eval": {"all_gates_pass": True},
        }
        full_metrics_path = artifacts / "full_workflow_metrics.json"
        full_metrics_path.write_text(json.dumps(full_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (artifacts / "full_workflow_stage_metrics.parquet").write_bytes(b"PAR1")

        bootstrap_report = artifacts / "bootstrap_report.json"
        preflight_report = artifacts / "preflight_report.json"
        bootstrap_report.write_text(json.dumps({"ok": True}, indent=2) + "\n", encoding="utf-8")
        preflight_report.write_text(json.dumps({"ok": True}, indent=2) + "\n", encoding="utf-8")

        run_result = RunResult(
            run_dir=local_cfg.as_path(),
            articles_path=artifacts / "articles.parquet",
            ranking_path=artifacts / "articles_ranked.parquet",
            documents_path=artifacts / "documents.parquet",
            evidence_path=artifacts / "evidence_points.parquet",
            provenance_path=artifacts / "provenance.parquet",
            validated_path=artifacts / "validated_points.parquet",
            validation_reasons_path=artifacts / "validation_reasons.parquet",
            consensus_path=artifacts / "consensus_points.parquet",
            report_path=outputs / "report.md",
            metrics_path=run_metrics_path,
        )
        return FullWorkflowResult(
            run_dir=local_cfg.as_path(),
            run_result=run_result,
            knowledge_paths={},
            processor_models={},
            processor_eval={"all_gates_pass": True},
            gate_status={
                "bootstrap_ok": True,
                "preflight_ok": True,
                "extractor_ok": True,
                "processor_ok": True,
                "all_ok": True,
            },
            preflight_path=preflight_report,
            bootstrap_path=bootstrap_report,
            metrics_path=full_metrics_path,
        )

    def _fake_benchmark(**kwargs):  # noqa: ANN003
        out_dir = Path(kwargs["out_dir"]).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_json = out_dir / "benchmark.json"
        bench_json.write_text(
            json.dumps({"precision": 0.9, "recall": 0.8, "f1": 0.85, "threshold_met": True}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "benchmark.md").write_text("# benchmark\n", encoding="utf-8")
        (out_dir / "benchmark_context.json").write_text(
            json.dumps(
                {
                    "condition_completeness": 0.8,
                    "condition_precision": 0.85,
                    "context_threshold_met": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (out_dir / "benchmark_mp.json").write_text(
            json.dumps({"mp_coverage": 0.9}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "benchmark_mp.md").write_text("# benchmark mp\n", encoding="utf-8")
        return BenchmarkResult(
            precision=0.9,
            recall=0.8,
            f1=0.85,
            tp=9,
            fp=1,
            fn=2,
            threshold_met=True,
            report_path=out_dir / "benchmark.md",
            json_path=bench_json,
            context_report_path=out_dir / "benchmark_context.md",
            mp_json_path=out_dir / "benchmark_mp.json",
            mp_report_path=out_dir / "benchmark_mp.md",
        )

    monkeypatch.setattr("oarp.pipeline.preflight_strict", _fake_preflight)
    monkeypatch.setattr("oarp.pipeline.run_full_workflow", _fake_run_full)
    monkeypatch.setattr("oarp.pipeline.run_benchmark", _fake_benchmark)

    result = run_all_done_validation(
        spec_path=str(spec_path),
        query="nickel silicide annealing",
        cfg=cfg,
        validation_cfg={
            "gold_dir": str(tmp_path / "gold"),
            "gold_context_dir": str(tmp_path / "gold_context"),
            "precision_gate": 0.80,
            "context_completeness_gate": 0.70,
            "context_precision_gate": 0.80,
        },
    )

    assert result.ok is True
    assert result.report_path.exists()
    assert result.json_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["gates"]["all_done_pass"] is True
