from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.mp_enrichment import enrich
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import dump_topic_spec, spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni-demo-mp",
            "keywords": ["nickel silicide"],
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "entities": [
                {"name": "NiSi", "aliases": ["nisi", "nickel monosilicide"]},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
        }
    )


def test_mp_enrichment_writes_artifacts_and_dedup_queries(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = RunConfig(run_dir=run_dir, require_context_fields=False, mp_enabled=True)
    spec = _spec()
    spec_path = tmp_path / "spec.yaml"
    dump_topic_spec(spec, spec_path)
    _initialize_run_state(str(spec_path), spec, "q", cfg)

    artifacts = run_dir / "artifacts"
    evidence = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "normalized_value": 8.0,
                "unit": "nm",
                "entity": "NiSi",
                "confidence": 0.9,
                "snippet": "NiSi appears at 350 C for 8 nm films.",
                "locator": "sentence:1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
            },
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "temperature_c",
                "normalized_value": 350.0,
                "unit": "c",
                "entity": "NiSi",
                "confidence": 0.9,
                "snippet": "NiSi appears at 350 C for 8 nm films.",
                "locator": "sentence:1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
            },
            {
                "point_id": "p2",
                "article_key": "a2",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "normalized_value": 12.0,
                "unit": "nm",
                "entity": "NiSi",
                "confidence": 0.92,
                "snippet": "NiSi forms near 420 C for 12 nm films.",
                "locator": "sentence:1",
                "citation_url": "https://example/a2",
                "doi": "10.1/a2",
            },
            {
                "point_id": "p2",
                "article_key": "a2",
                "provider": "mock",
                "variable_name": "temperature_c",
                "normalized_value": 420.0,
                "unit": "c",
                "entity": "NiSi",
                "confidence": 0.92,
                "snippet": "NiSi forms near 420 C for 12 nm films.",
                "locator": "sentence:1",
                "citation_url": "https://example/a2",
                "doi": "10.1/a2",
            },
        ]
    )
    evidence.to_parquet(artifacts / "evidence_points.parquet", index=False)

    result = enrich(spec=spec, cfg=cfg)
    assert result.enriched_points_path.exists()
    assert result.materials_path.exists()
    assert result.point_links_path.exists()
    assert result.query_log_path.exists()

    enriched = pd.read_parquet(result.enriched_points_path)
    assert "mp_status" in enriched.columns
    assert "mp_interpretation_label" in enriched.columns
    assert "mp_query_key" in enriched.columns
    assert len(enriched["mp_query_key"].dropna().unique()) == 1

    query_log = pd.read_parquet(result.query_log_path)
    assert len(query_log) == 1
    assert query_log.iloc[0]["status"] in {"no_api_key", "success", "fetch_error"}


def test_mp_enrichment_resolves_formula_from_snippet_when_entity_unknown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = RunConfig(run_dir=run_dir, require_context_fields=False, mp_enabled=False)
    spec = _spec()
    spec_path = tmp_path / "spec.yaml"
    dump_topic_spec(spec, spec_path)
    _initialize_run_state(str(spec_path), spec, "q", cfg)

    artifacts = run_dir / "artifacts"
    evidence = pd.DataFrame(
        [
            {
                "point_id": "p3",
                "article_key": "a3",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "normalized_value": 10.0,
                "unit": "nm",
                "entity": "unknown",
                "confidence": 0.91,
                "snippet": "beta-1 Ni 3 Si phase appears near 500 C for 10 nm film.",
                "locator": "sentence:1",
                "citation_url": "https://example/a3",
                "doi": "10.1/a3",
            },
            {
                "point_id": "p3",
                "article_key": "a3",
                "provider": "mock",
                "variable_name": "temperature_c",
                "normalized_value": 500.0,
                "unit": "c",
                "entity": "unknown",
                "confidence": 0.91,
                "snippet": "beta-1 Ni 3 Si phase appears near 500 C for 10 nm film.",
                "locator": "sentence:1",
                "citation_url": "https://example/a3",
                "doi": "10.1/a3",
            },
        ]
    )
    evidence.to_parquet(artifacts / "evidence_points.parquet", index=False)

    result = enrich(spec=spec, cfg=cfg)
    query_log = pd.read_parquet(result.query_log_path)
    assert len(query_log) == 1
    assert str(query_log.iloc[0]["formula"]) == "Ni3Si"
    assert str(query_log.iloc[0]["status"]) == "disabled"
