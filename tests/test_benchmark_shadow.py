from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.benchmark import run_benchmark
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni",
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
        }
    )


def test_benchmark_writes_shadow_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)

    pred = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "x_value": 3.0,
                "y_value": 290.0,
                "entity": "Ni2Si",
            }
        ]
    )
    pred.to_parquet(artifacts / "consensus_points.parquet", index=False)

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "gold_points.csv").write_text("x_value,y_value,entity\n20,260,Ni2Si\n", encoding="utf-8")

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    (shadow_dir / "gold_points.csv").write_text("x_value,y_value,entity\n3,290,Ni2Si\n", encoding="utf-8")

    out_dir = tmp_path / "bench"
    result = run_benchmark(
        spec=_spec(),
        gold_dir=gold_dir,
        out_dir=out_dir,
        run_dir=run_dir,
        shadow_gold_dir=shadow_dir,
        strict_gold=True,
    )

    assert result.shadow_metrics_path is not None
    assert result.shadow_metrics_path.exists()
    payload = json.loads(result.shadow_metrics_path.read_text(encoding="utf-8"))
    assert float(payload["precision"]) == 1.0
