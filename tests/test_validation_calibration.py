from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import dump_topic_spec, spec_from_dict
from oarp.validation import validate


def _spec():
    return spec_from_dict(
        {
            "topic_id": "calib-demo",
            "keywords": ["nickel silicide"],
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "entities": [{"name": "NiSi", "aliases": ["nisi"]}],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
            "validation": {
                "min_confidence": 0.60,
                "required_provenance_fields": ["citation_url", "snippet", "locator"],
            },
        }
    )


def test_validation_emits_extraction_calibration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    spec = _spec()
    spec_path = tmp_path / "topic.yaml"
    dump_topic_spec(spec, spec_path)
    cfg = RunConfig(
        run_dir=run_dir,
        require_context_fields=False,
        emit_extraction_calibration=True,
        calibration_bins=6,
    )
    _initialize_run_state(str(spec_path), spec, "q", cfg)
    artifacts = run_dir / "artifacts"

    points = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "normalized_value": 20.0,
                "unit": "nm",
                "entity": "NiSi",
                "confidence": 0.92,
                "snippet": "NiSi at 20 nm and 340 C.",
                "locator": "sentence:1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
            },
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "temperature_c",
                "normalized_value": 340.0,
                "unit": "c",
                "entity": "NiSi",
                "confidence": 0.92,
                "snippet": "NiSi at 20 nm and 340 C.",
                "locator": "sentence:1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
            },
        ]
    )
    points.to_parquet(artifacts / "evidence_points.parquet", index=False)
    pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
                "snippet": "NiSi at 20 nm and 340 C.",
                "locator": "sentence:1",
                "extraction_type": "text",
                "parser_trace": "test",
                "confidence": 0.92,
                "provenance_mode": "point-level",
                "created_at": "now",
            }
        ]
    ).to_parquet(artifacts / "provenance.parquet", index=False)
    pd.DataFrame(
        [
            {
                "point_id": "p1",
                "model_id": "m1",
                "json_payload": "{}",
                "schema_valid": True,
                "rule_valid": True,
                "confidence": 0.91,
                "aggregated_support": 0.90,
                "accepted": True,
                "reason_codes": "",
            }
        ]
    ).to_parquet(artifacts / "extraction_votes.parquet", index=False)

    result = validate(spec=spec, cfg=cfg)
    assert result.extraction_calibration_path is not None
    assert result.extraction_calibration_path.exists()
    calib = pd.read_parquet(result.extraction_calibration_path)
    assert len(calib) == 6
