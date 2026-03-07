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
            "topic_id": "ni-mp-validate",
            "keywords": ["nickel silicide"],
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "entities": [{"name": "NiSi", "aliases": ["nisi"]}],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
            "validation": {
                "min_confidence": 0.65,
                "required_provenance_fields": ["citation_url", "snippet", "locator"],
            },
        }
    )


def _seed(run_dir: Path, mode: str) -> tuple[RunConfig, object]:
    spec = _spec()
    cfg = RunConfig(
        run_dir=run_dir,
        require_context_fields=False,
        mp_mode=mode,
    )
    spec_path = run_dir.parent / "spec.yaml"
    dump_topic_spec(spec, spec_path)
    _initialize_run_state(str(spec_path), spec, "q", cfg)
    artifacts = run_dir / "artifacts"

    rows = [
        {
            "point_id": "p1",
            "article_key": "a1",
            "provider": "mock",
            "variable_name": "thickness_nm",
            "normalized_value": 10.0,
            "unit": "nm",
            "entity": "NiSi",
            "confidence": 0.95,
            "snippet": "NiSi appears near 380 C.",
            "locator": "sentence:1",
            "citation_url": "https://example/a1",
            "doi": "10.1/a1",
            "mp_status": "success",
            "mp_interpretation_label": "conflicts",
            "mp_interpretation_score": 0.10,
            "mp_conflict_reason": "formula_mismatch",
        },
        {
            "point_id": "p1",
            "article_key": "a1",
            "provider": "mock",
            "variable_name": "temperature_c",
            "normalized_value": 380.0,
            "unit": "c",
            "entity": "NiSi",
            "confidence": 0.95,
            "snippet": "NiSi appears near 380 C.",
            "locator": "sentence:1",
            "citation_url": "https://example/a1",
            "doi": "10.1/a1",
            "mp_status": "success",
            "mp_interpretation_label": "conflicts",
            "mp_interpretation_score": 0.10,
            "mp_conflict_reason": "formula_mismatch",
        },
    ]
    pd.DataFrame(rows).to_parquet(artifacts / "materials_project_enriched_points.parquet", index=False)
    pd.DataFrame(rows).to_parquet(artifacts / "evidence_points.parquet", index=False)
    pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
                "snippet": "NiSi appears near 380 C.",
                "locator": "sentence:1",
                "extraction_type": "text",
                "parser_trace": "test",
                "confidence": 0.95,
                "provenance_mode": "point-level",
                "created_at": "now",
            }
        ]
    ).to_parquet(artifacts / "provenance.parquet", index=False)
    return cfg, spec


def test_validation_mp_mode_interpreter_non_gating(tmp_path: Path) -> None:
    cfg, spec = _seed(tmp_path / "run_i", "interpreter")
    result = validate(spec=spec, cfg=cfg)
    assert len(result.accepted) == 2
    warnings = pd.read_parquet(result.warnings_path)
    assert len(warnings) == 2


def test_validation_mp_mode_hybrid_rejects_strong_conflict(tmp_path: Path) -> None:
    cfg, spec = _seed(tmp_path / "run_h", "hybrid")
    result = validate(spec=spec, cfg=cfg)
    assert len(result.accepted) == 0
    assert len(result.rejected) == 2
    assert result.rejected["reject_reason"].astype(str).str.contains("mp_conflict_hybrid").all()


def test_validation_mp_mode_hard_rejects_conflict(tmp_path: Path) -> None:
    cfg, spec = _seed(tmp_path / "run_x", "hard")
    result = validate(spec=spec, cfg=cfg)
    assert len(result.accepted) == 0
    assert len(result.rejected) == 2
    assert result.rejected["reject_reason"].astype(str).str.contains("mp_conflict_hard").all()
