from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.pipeline import enforce_quality_gates, evaluate_quality_no_gold
from oarp.runtime import ensure_run_layout


def test_quality_gate_uses_coverage_strength_not_only_unique_article_ratio(tmp_path: Path) -> None:
    run_dir = (tmp_path / "run_quality").resolve()
    artifacts = ensure_run_layout(run_dir)["artifacts"]

    docs = pd.DataFrame(
        [
            {"article_key": f"a{i}", "usable_text": True, "parse_status": "parsed_pdf"}
            for i in range(100)
        ]
    )
    docs.to_parquet(artifacts / "documents.parquet", index=False)

    evidence = pd.DataFrame(
        [
            {
                "point_id": f"p{i}",
                "article_key": "a1",
                "variable_name": "thickness_nm",
                "normalized_value": 10.0,
            }
            for i in range(30)
        ]
    )
    evidence.to_parquet(artifacts / "evidence_points.parquet", index=False)

    validated = pd.DataFrame(
        [
            {
                "point_id": "pt1",
                "article_key": "a1",
                "variable_name": "thickness_nm",
                "normalized_value": 10.0,
                "entity": "phaseA",
                "citation_url": "https://x/1",
                "snippet": "s",
                "locator": "L1",
            },
            {
                "point_id": "pt1",
                "article_key": "a1",
                "variable_name": "temperature_c",
                "normalized_value": 300.0,
                "entity": "phaseA",
                "citation_url": "https://x/1",
                "snippet": "s",
                "locator": "L1",
            },
            {
                "point_id": "pt2",
                "article_key": "a2",
                "variable_name": "thickness_nm",
                "normalized_value": 12.0,
                "entity": "phaseA",
                "citation_url": "https://x/2",
                "snippet": "s",
                "locator": "L2",
            },
            {
                "point_id": "pt2",
                "article_key": "a2",
                "variable_name": "temperature_c",
                "normalized_value": 320.0,
                "entity": "phaseA",
                "citation_url": "https://x/2",
                "snippet": "s",
                "locator": "L2",
            },
        ]
    )
    validated.to_parquet(artifacts / "validated_points.parquet", index=False)

    consensus = pd.DataFrame(
        [
            {"support_count": 3, "entropy": 0.5},
            {"support_count": 4, "entropy": 0.6},
        ]
    )
    consensus.to_parquet(artifacts / "consensus_points.parquet", index=False)

    slm_responses = pd.DataFrame([{"status": "ok"}, {"status": "ok"}])
    slm_responses.to_parquet(artifacts / "slm_responses.parquet", index=False)

    (artifacts / "graph_audit.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    cfg = RunConfig(run_dir=run_dir, quality_profile="balanced")
    eval_result = evaluate_quality_no_gold(run_dir=str(run_dir), cfg=cfg)
    assert abs(float(eval_result.metrics["matched_article_ratio"]) - 0.01) < 1e-12
    assert abs(float(eval_result.metrics["evidence_density_per_usable_doc"]) - 0.3) < 1e-12
    assert abs(float(eval_result.metrics["coverage_strength"]) - 0.3) < 1e-12

    gate_result = enforce_quality_gates(run_dir=str(run_dir), cfg=cfg)
    assert gate_result.gates["matched_article_ratio_ge_0_20"] is True
    assert gate_result.ok is True
