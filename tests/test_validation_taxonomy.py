from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import spec_from_dict
from oarp.validation import validate


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni",
            "variables": [
                {"name": "thickness_nm", "aliases": ["thickness"], "unit": "nm", "datatype": "float", "normalization": "length", "min_value": 0},
                {"name": "temperature_c", "aliases": ["temperature"], "unit": "c", "datatype": "float", "normalization": "temperature", "min_value": 20, "max_value": 1400},
            ],
            "entities": [{"name": "NiSi", "aliases": ["nisi"]}],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c"}},
            "validation": {"required_provenance_fields": ["citation_url", "snippet", "locator"], "min_confidence": 0.7},
        }
    )


def test_validation_emits_reason_taxonomy(tmp_path: Path) -> None:
    spec = _spec()
    cfg = RunConfig(run_dir=tmp_path / "run", emit_validation_metrics=True)
    _initialize_run_state(str(tmp_path / "spec.yaml"), spec, "q", cfg)

    artifacts = cfg.as_path() / "artifacts"
    points = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "raw_value": "-10",
                "normalized_value": -10.0,
                "unit": "nm",
                "entity": "",
                "extraction_type": "text",
                "confidence": 0.6,
                "snippet": "thickness -10 nm",
                "locator": "line:1",
                "citation_url": "",
                "doi": "",
                "created_at": "now",
            }
        ]
    )
    points.to_parquet(artifacts / "evidence_points.parquet", index=False)

    prov = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "citation_url": "",
                "doi": "",
                "snippet": "",
                "locator": "",
                "extraction_type": "text",
                "parser_trace": "test",
                "confidence": 0.6,
                "provenance_mode": "paper-level",
                "created_at": "now",
            }
        ]
    )
    prov.to_parquet(artifacts / "provenance.parquet", index=False)

    result = validate(spec=spec, cfg=cfg)
    assert len(result.rejected) == 1
    assert result.validation_reasons_path is not None
    reasons = pd.read_parquet(result.validation_reasons_path)
    reason_names = set(reasons["reason"].tolist())
    assert "out_of_range" in reason_names
    assert "missing_primary_variables" in reason_names
