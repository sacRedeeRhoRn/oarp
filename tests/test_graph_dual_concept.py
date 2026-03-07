from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.processing import audit_dual_graph, build_processing_dataset, score_bridge_weight


def _points_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    payload = [
        ("p1", "a1", "NiSi", 10.0, 340.0, "(111)", "Pnma", "62", ["Pt"], "PVD"),
        ("p2", "a2", "Ni2Si", 18.0, 280.0, "(100)", "Fm-3m", "225", ["Au"], "ALD"),
    ]
    for point_id, article_key, phase, thick, temp, orient, sg_sym, sg_num, dopants, method in payload:
        rows.append(
            {
                "point_id": point_id,
                "article_key": article_key,
                "variable_name": "thickness_nm",
                "normalized_value": thick,
                "entity": phase,
                "phase_label": phase,
                "film_material": phase,
                "substrate_material": "Si",
                "substrate_orientation": orient,
                "doping_elements": str(dopants),
                "doping_state": "doped",
                "alloy_elements": "[]",
                "alloy_state": "na_pure_ni",
                "method_family": method,
                "confidence": 0.92,
                "citation_url": "https://example.org/paper",
                "locator": "line:10",
                "snippet": "phase evidence",
                "mp_spacegroup_symbol": sg_sym,
                "mp_spacegroup_number": sg_num,
                "mp_energy_above_hull_min": 0.04,
                "mp_interpretation_label": "supports",
            }
        )
        rows.append(
            {
                "point_id": point_id,
                "article_key": article_key,
                "variable_name": "temperature_c",
                "normalized_value": temp,
                "entity": phase,
                "phase_label": phase,
                "film_material": phase,
                "substrate_material": "Si",
                "substrate_orientation": orient,
                "doping_elements": str(dopants),
                "doping_state": "doped",
                "alloy_elements": "[]",
                "alloy_state": "na_pure_ni",
                "method_family": method,
                "confidence": 0.92,
                "citation_url": "https://example.org/paper",
                "locator": "line:11",
                "snippet": "anneal evidence",
                "mp_spacegroup_symbol": sg_sym,
                "mp_spacegroup_number": sg_num,
                "mp_energy_above_hull_min": 0.04,
                "mp_interpretation_label": "supports",
            }
        )
    return pd.DataFrame(rows)


def test_dual_graph_artifacts_and_concept_sharing(tmp_path: Path) -> None:
    cfg = RunConfig(
        run_dir=tmp_path / "run",
        graph_architecture="dual_concept",
        concept_ontology_profile="maximal",
        bridge_weight_policy="deterministic_v1",
        bridge_weight_threshold=0.15,
        concept_gates=["film", "substrate", "dopant", "precursor", "method", "thermo", "electronic"],
    )
    dataset = build_processing_dataset(points=_points_df(), mp_data=None, aux_data=None, cfg=cfg)
    artifacts = cfg.as_path() / "artifacts"

    assert dataset.path.exists()
    assert (artifacts / "article_process_nodes.parquet").exists()
    assert (artifacts / "article_process_edges.parquet").exists()
    assert (artifacts / "concept_nodes.parquet").exists()
    assert (artifacts / "concept_edges.parquet").exists()
    assert (artifacts / "bridge_edges.parquet").exists()
    assert (artifacts / "bridge_weight_audit.parquet").exists()
    assert (artifacts / "graph_audit.json").exists()
    assert (artifacts / "graph_audit.md").exists()
    # Backward compatibility artifacts must remain present.
    assert (artifacts / "graph_nodes.parquet").exists()
    assert (artifacts / "graph_edges.parquet").exists()
    assert (artifacts / "graph_tensor_index.parquet").exists()

    concept_nodes = pd.read_parquet(artifacts / "concept_nodes.parquet")
    ni_nodes = concept_nodes[
        (concept_nodes["concept_type"].astype(str) == "element")
        & (concept_nodes["canonical_label"].astype(str) == "Ni")
    ]
    assert len(ni_nodes) == 1
    element_nodes = concept_nodes[concept_nodes["concept_type"].astype(str) == "element"]
    assert len(element_nodes) >= 118
    xe_nodes = concept_nodes[
        (concept_nodes["concept_type"].astype(str) == "element")
        & (concept_nodes["canonical_label"].astype(str) == "Xe")
    ]
    assert len(xe_nodes) == 1

    bridge_edges = pd.read_parquet(artifacts / "bridge_edges.parquet")
    assert not bridge_edges.empty
    assert (bridge_edges["weight"].astype(float) >= cfg.bridge_weight_threshold).all()
    assert set(bridge_edges["gate_type"].astype(str)).intersection(
        {"film_gate", "substrate_gate", "dopant_gate", "precursor_gate", "method_gate", "thermo_gate", "electronic_gate"}
    )

    training = pd.read_parquet(dataset.path)
    assert "concept_bridge_count" in training.columns
    assert "concept_bridge_mean_weight" in training.columns
    assert "concept_bridge_max_weight" in training.columns
    assert "concept_gate_coverage" in training.columns
    assert "graph_schema_version" in training.columns
    assert (training["graph_schema_version"].astype(str) == "v2_dual_concept").all()

    audit = audit_dual_graph(cfg.as_path(), cfg)
    assert audit.ok is True


def test_bridge_weight_scoring_is_deterministic() -> None:
    cfg = RunConfig(run_dir="/tmp/oarp_dual_graph_score_test")
    features = {
        "extraction_conf": 0.9,
        "provenance_quality": 1.0,
        "context_proximity": 0.8,
        "cross_source_agreement": 0.5,
        "mp_alignment": 0.6,
    }
    score1 = score_bridge_weight(features, cfg)
    score2 = score_bridge_weight(features, cfg)
    assert score1 == score2
    assert 0.0 <= score1 <= 1.0
