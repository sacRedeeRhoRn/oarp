from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.processing import (
    build_processing_dataset,
    finetune_processor_for_system,
    predict_with_processor,
    train_universal_processor,
)


def _points_df() -> pd.DataFrame:
    rows = []
    base = [
        ("p1", "Ni2Si", 12.0, 260.0, 0.82),
        ("p2", "NiSi", 20.0, 340.0, 0.88),
        ("p3", "NiSi2", 28.0, 470.0, 0.81),
        ("p4", "NiSi", 16.0, 330.0, 0.86),
    ]
    for pid, ent, x, y, q in base:
        rows.append(
            {
                "point_id": pid,
                "variable_name": "thickness_nm",
                "normalized_value": x,
                "entity": ent,
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
                "variable_name": "temperature_c",
                "normalized_value": y,
                "entity": ent,
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "alloy_state": "na_pure_ni",
                "film_quality_score_numeric": q,
                "method_family": "PVD",
            }
        )
    return pd.DataFrame(rows)


def test_processing_dataset_train_finetune_predict(tmp_path: Path) -> None:
    cfg = RunConfig(run_dir=tmp_path / "run")
    points = _points_df()

    dataset = build_processing_dataset(points=points, mp_data=None, aux_data=None, cfg=cfg)
    assert dataset.path.exists()
    assert "graph_support" in dataset.frame.columns

    cfg.processor_model_dir = str(tmp_path / "models" / "universal")
    base_model = train_universal_processor(dataset, cfg)
    assert base_model.model_path.exists()
    assert base_model.metadata_path.exists()

    cfg.processor_model_dir = str(tmp_path / "models" / "finetune")
    tuned_model = finetune_processor_for_system(base_model, dataset, cfg, system_id="nisi")
    assert tuned_model.model_path.exists()

    candidates = pd.DataFrame(
        [
            {
                "phase_label": "NiSi",
                "film_material": "NiSi",
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "method_family": "PVD",
                "thickness_nm": 18.0,
                "anneal_temperature_c": 335.0,
                "film_quality_score_numeric": 0.8,
            }
        ]
    )
    pred = predict_with_processor(candidates, tuned_model.model_dir, target_phase="NiSi")
    assert "processor_phase_prob" in pred.columns
    assert "processor_quality_score" in pred.columns
    assert "processor_uncertainty" in pred.columns
