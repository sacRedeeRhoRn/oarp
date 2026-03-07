from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.processing import build_processing_dataset, train_gnn_base


def _points_df() -> pd.DataFrame:
    rows = []
    base = [
        ("p1", "a1", "Ni2Si", 12.0, 260.0, 0.82),
        ("p2", "a1", "NiSi", 20.0, 340.0, 0.88),
        ("p3", "a2", "NiSi2", 28.0, 470.0, 0.81),
        ("p4", "a2", "NiSi", 16.0, 330.0, 0.86),
    ]
    for pid, article_key, ent, x, y, q in base:
        rows.append(
            {
                "point_id": pid,
                "article_key": article_key,
                "variable_name": "thickness_nm",
                "normalized_value": x,
                "entity": ent,
                "phase_label": ent,
                "film_material": ent,
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "alloy_state": "na_pure_ni",
                "film_quality_score_numeric": q,
                "method_family": "PVD",
            }
        )
        rows.append(
            {
                "point_id": pid,
                "article_key": article_key,
                "variable_name": "temperature_c",
                "normalized_value": y,
                "entity": ent,
                "phase_label": ent,
                "film_material": ent,
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "alloy_state": "na_pure_ni",
                "film_quality_score_numeric": q,
                "method_family": "PVD",
            }
        )
    return pd.DataFrame(rows)


def test_processing_dataset_emits_graph_samples(tmp_path: Path) -> None:
    cfg = RunConfig(run_dir=tmp_path / "run")
    dataset = build_processing_dataset(points=_points_df(), mp_data=None, aux_data=None, cfg=cfg)

    assert dataset.path.exists()
    graph_path = cfg.as_path() / "artifacts" / "graph_samples.parquet"
    assert graph_path.exists()

    graph_df = pd.read_parquet(graph_path)
    assert not graph_df.empty
    assert {"graph_sample_id", "article_key", "node_counts_json", "edge_counts_json"}.issubset(graph_df.columns)


def test_train_gnn_base_or_explicit_dependency_error(tmp_path: Path) -> None:
    cfg = RunConfig(run_dir=tmp_path / "run", gnn_epochs=2, gnn_hidden_dim=16, gnn_layers=1)
    dataset = build_processing_dataset(points=_points_df(), mp_data=None, aux_data=None, cfg=cfg)

    try:
        model_ref = train_gnn_base(dataset, cfg)
    except RuntimeError as exc:
        assert "torch + torch_geometric" in str(exc)
        return

    assert model_ref.model_path.exists()
    assert (model_ref.model_dir / "article_embeddings.parquet").exists()
    assert model_ref.metrics_path.exists()
