from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.processing import build_processing_dataset


def _points_df() -> pd.DataFrame:
    rows = []
    base = [
        ("p1", "a1", "Ni2Si", 8.0, 250.0, "Fm-3m", "225"),
        ("p2", "a1", "NiSi", 12.0, 360.0, "Pnma", "62"),
        ("p3", "a2", "NiSi2", 20.0, 520.0, "Fm-3m", "225"),
    ]
    for pid, article_key, phase, x, y, sg_sym, sg_num in base:
        rows.append(
            {
                "point_id": pid,
                "article_key": article_key,
                "variable_name": "thickness_nm",
                "normalized_value": x,
                "entity": phase,
                "phase_label": phase,
                "film_material": phase,
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "alloy_state": "na_pure_ni",
                "method_family": "PVD",
                "mp_spacegroup_symbol": sg_sym,
                "mp_spacegroup_number": sg_num,
                "confidence": 0.9,
            }
        )
        rows.append(
            {
                "point_id": pid,
                "article_key": article_key,
                "variable_name": "temperature_c",
                "normalized_value": y,
                "entity": phase,
                "phase_label": phase,
                "film_material": phase,
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "alloy_state": "na_pure_ni",
                "method_family": "PVD",
                "mp_spacegroup_symbol": sg_sym,
                "mp_spacegroup_number": sg_num,
                "confidence": 0.9,
            }
        )
    return pd.DataFrame(rows)


def test_v1_phase_and_graph_artifacts_are_emitted(tmp_path: Path) -> None:
    cfg = RunConfig(
        run_dir=tmp_path / "run",
        cache_mode="run_local",
        phase_schema_version="v1",
        graph_core_mode="per_article_typed",
    )
    dataset = build_processing_dataset(points=_points_df(), mp_data=None, aux_data=None, cfg=cfg)
    assert dataset.path.exists()

    artifacts = cfg.as_path() / "artifacts"
    phase_specs = artifacts / "phase_specs.parquet"
    graph_nodes = artifacts / "graph_nodes.parquet"
    graph_edges = artifacts / "graph_edges.parquet"
    tensor_index = artifacts / "graph_tensor_index.parquet"
    cache_audit = artifacts / "cache_audit.parquet"
    assert phase_specs.exists()
    assert graph_nodes.exists()
    assert graph_edges.exists()
    assert tensor_index.exists()
    assert cache_audit.exists()

    nodes = pd.read_parquet(graph_nodes)
    edges = pd.read_parquet(graph_edges)
    assert not nodes.empty
    assert not edges.empty
    assert {"phase_formula", "element", "space_group", "condition_vector"}.issubset(
        set(nodes["node_type"].astype(str))
    )
    assert {
        "phase_formula_has_element",
        "phase_instance_has_space_group",
        "phase_instance_observed_under_condition",
    }.issubset(set(edges["edge_type"].astype(str)))
