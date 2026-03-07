from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.benchmark import run_benchmark
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni-bench-mp",
            "keywords": ["nickel silicide"],
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "entities": [{"name": "NiSi", "aliases": ["nisi"]}],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
        }
    )


def test_benchmark_writes_mp_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    pred = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "x_value": 10.0,
                "y_value": 350.0,
                "entity": "NiSi",
                "mp_status": "success",
                "mp_interpretation_label": "supports",
            }
        ]
    )
    pred.to_parquet(artifacts / "consensus_points.parquet", index=False)

    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"x_value": 10.0, "y_value": 350.0, "entity": "NiSi"}]).to_csv(
        gold_dir / "gold_points.csv",
        index=False,
    )

    out_dir = tmp_path / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(spec=_spec(), gold_dir=gold_dir, out_dir=out_dir, run_dir=run_dir)
    assert result.mp_json_path is not None
    assert result.mp_report_path is not None
    assert result.mp_json_path.exists()
    assert result.mp_report_path.exists()
    payload = json.loads(result.mp_json_path.read_text(encoding="utf-8"))
    assert "mp_coverage" in payload
    assert "mp_support_precision" in payload
