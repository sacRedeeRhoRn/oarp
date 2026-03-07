from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.models import AllDoneValidationResult, RunConfig
from oarp.pipeline import validate_release_v1


def test_validate_release_v1_emits_outputs(tmp_path: Path, monkeypatch) -> None:
    run_dir = (tmp_path / "run").resolve()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    (artifacts / "phase_specs.parquet").write_bytes(b"")
    phase_df = pd.DataFrame(
        [
            {
                "phase_id": "phase_1",
                "reduced_formula": "NiSi",
                "elements_json": '["Ni","Si"]',
                "stoichiometry_json": '[{"element":"Ni","ratio":1.0},{"element":"Si","ratio":1.0}]',
                "spacegroup_symbol": "Pnma",
                "spacegroup_number": "62",
                "phase_signature": "sig1",
            }
        ]
    )
    phase_df.to_parquet(artifacts / "phase_specs.parquet", index=False)
    pd.DataFrame(
        [
            {
                "node_id": "n1",
                "node_type": "phase_formula",
                "article_key": "a1",
                "label": "NiSi",
                "attributes_json": "{}",
            },
            {
                "node_id": "n2",
                "node_type": "element",
                "article_key": "a1",
                "label": "Ni",
                "attributes_json": "{}",
            },
        ]
    ).to_parquet(artifacts / "graph_nodes.parquet", index=False)
    pd.DataFrame(
        [
            {
                "edge_id": "e1",
                "article_key": "a1",
                "source_id": "n1",
                "target_id": "n2",
                "edge_type": "phase_formula_has_element",
                "evidence_weight": 0.9,
                "provenance_ref": "x",
                "source_point_id": "p1",
            }
        ]
    ).to_parquet(artifacts / "graph_edges.parquet", index=False)
    pd.DataFrame(
        [
            {
                "graph_sample_id": "g1",
                "article_key": "a1",
                "node_count": 2,
                "edge_count": 1,
                "tensor_path": str(artifacts / "graph_tensors_v1" / "g1.npz"),
            }
        ]
    ).to_parquet(artifacts / "graph_tensor_index.parquet", index=False)
    pd.DataFrame(
        [
            {
                "namespace": "x",
                "key_hash": "k",
                "event": "hit",
                "cache_root": str(artifacts),
                "size_bytes": 1,
                "created_at": "2026-03-07T00:00:00+00:00",
            }
        ]
    ).to_parquet(artifacts / "cache_audit.parquet", index=False)

    fake_all_done = AllDoneValidationResult(
        ok=True,
        run_dir=run_dir,
        report_path=artifacts / "all_done_validation.md",
        json_path=artifacts / "all_done_validation.json",
        repro_compare_path=artifacts / "repro_compare.json",
        gates={
            "bootstrap_gate": True,
            "preflight_gate": True,
            "crawl_acquire_gate": True,
            "extraction_gate": True,
            "processing_gate": True,
            "benchmark_gate": True,
            "mp_gate": True,
            "reproducibility_gate": True,
            "all_done_pass": True,
        },
        artifacts={"all_done_json": str(artifacts / "all_done_validation.json")},
        benchmark_summary={"precision": 0.9, "recall": 0.8, "f1": 0.85},
    )
    fake_all_done.report_path.write_text("# all done\n", encoding="utf-8")
    fake_all_done.json_path.write_text(json.dumps({"ok": True}, indent=2) + "\n", encoding="utf-8")
    fake_all_done.repro_compare_path.write_text(
        json.dumps({"repro_ok": True, "max_relative_drift": 0.0}, indent=2) + "\n",
        encoding="utf-8",
    )

    def _fake_run_all_done_validation(spec_path: str, query: str, cfg: RunConfig, validation_cfg: dict | None = None):  # noqa: ANN001
        _ = (spec_path, query, cfg, validation_cfg)
        return fake_all_done

    monkeypatch.setattr("oarp.pipeline.run_all_done_validation", _fake_run_all_done_validation)

    cfg = RunConfig(run_dir=run_dir)
    result = validate_release_v1(
        run_dir=run_dir,
        cfg=cfg,
        validation_cfg={
            "spec_path": str(tmp_path / "topic.yaml"),
            "query": "nickel silicide",
            "gold_dir": str(tmp_path / "gold"),
        },
    )

    assert result.all_done_pass is True
    assert result.json_path.exists()
    assert result.report_path.exists()
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["gate_map"]["graph_integrity_gate"] is True
    assert payload["gate_map"]["phase_schema_gate"] is True
    assert payload["gate_map"]["cache_audit_gate"] is True
