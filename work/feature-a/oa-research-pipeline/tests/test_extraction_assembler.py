from __future__ import annotations

import pandas as pd

from oarp.extraction import (
    WindowContextAssembler,
    _enrich_with_context,
    _extract_entity,
    _extract_transition_events,
    _extract_values_from_snippet,
)
from oarp.models import RunConfig
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni",
            "variables": [
                {
                    "name": "thickness_nm",
                    "aliases": ["thickness"],
                    "unit": "nm",
                    "datatype": "float",
                    "normalization": "length",
                    "min_value": 0,
                },
                {
                    "name": "temperature_c",
                    "aliases": ["temperature"],
                    "unit": "c",
                    "datatype": "float",
                    "normalization": "temperature",
                    "min_value": 20,
                    "max_value": 1400,
                },
            ],
            "entities": [{"name": "NiSi", "aliases": ["nisi"]}],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c"}},
        }
    )


def test_numeric_range_hyphen_not_parsed_as_negative() -> None:
    spec = _spec()
    vals = _extract_values_from_snippet("critical thickness range 15-20 nm thickness", spec)
    assert vals
    assert float(vals[0].normalized) >= 0


def test_numeric_identifier_noise_not_used_for_thickness() -> None:
    spec = _spec()
    vals = _extract_values_from_snippet(
        (
            "NiSi films were characterized. Film thickness was discussed for reproducibility. "
            "ISSN 1729-7648 and DOI 10.35596/1729-7648-2020-18-1-81-88 are metadata only."
        ),
        spec,
    )
    thickness = [float(item.normalized) for item in vals if item.variable == "thickness_nm"]
    assert 7648.0 not in thickness
    assert 1729.0 not in thickness


def test_temperature_fallback_rejects_resistivity_values_without_temp_unit() -> None:
    spec = _spec()
    vals = _extract_values_from_snippet(
        (
            "The annealing temperature dependence was evaluated and specific resistance "
            "increases to 26-30 μOhm×cm for this sample."
        ),
        spec,
    )
    temp_values = [float(item.normalized) for item in vals if item.variable == "temperature_c"]
    assert temp_values == []


def test_entity_alias_prefers_longest_match() -> None:
    spec = spec_from_dict(
        {
            "topic_id": "ni-entity",
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float"},
                {"name": "temperature_c", "unit": "c", "datatype": "float"},
            ],
            "entities": [
                {"name": "Ni2Si", "aliases": ["ni2si"]},
                {"name": "NiSi", "aliases": ["nisi"]},
                {"name": "NiSi2", "aliases": ["nisi2"]},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c"}},
        }
    )
    assert _extract_entity("The NiSi2 phase appears after annealing.", spec) == "NiSi2"
    assert _extract_entity("The NiSi 2 phase appears after annealing.", spec) == "NiSi2"
    assert _extract_entity("The Ni 2 Si phase appears first.", spec) == "Ni2Si"


def test_transition_event_extractor_builds_phase_points() -> None:
    spec = spec_from_dict(
        {
            "topic_id": "ni-transition",
            "variables": [
                {
                    "name": "thickness_nm",
                    "aliases": ["thickness"],
                    "unit": "nm",
                    "datatype": "float",
                    "normalization": "length",
                    "min_value": 0,
                },
                {
                    "name": "temperature_c",
                    "aliases": ["temperature"],
                    "unit": "c",
                    "datatype": "float",
                    "normalization": "temperature",
                    "min_value": 20,
                    "max_value": 1400,
                },
            ],
            "entities": [
                {"name": "Ni2Si", "aliases": ["ni2si"]},
                {"name": "NiSi", "aliases": ["nisi"]},
                {"name": "NiSi2", "aliases": ["nisi2"]},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c"}},
        }
    )
    text = (
        "For 20-30 nm Ni films, Ni2Si appears at 250-270 C, "
        "NiSi appears at 330-350 C, and NiSi2 appears near 510-530 C."
    )
    events = _extract_transition_events(text, spec)
    assert len(events) == 3
    payload = {item.entity: (round(item.thickness_nm, 2), round(item.temperature_c, 2)) for item in events}
    assert payload["Ni2Si"][0] == 25.0
    assert payload["NiSi"][0] == 25.0
    assert payload["NiSi2"][0] == 25.0
    assert payload["Ni2Si"][1] == 260.0
    assert payload["NiSi"][1] == 340.0
    assert payload["NiSi2"][1] == 520.0


def test_context_assembler_merges_neighbor_lines() -> None:
    spec = _spec()
    points = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "raw_value": "30",
                "normalized_value": 30.0,
                "unit": "nm",
                "entity": "NiSi",
                "extraction_type": "text",
                "confidence": 0.75,
                "snippet": "thickness 30 nm",
                "locator": "line:10",
                "line_no": 10,
                "citation_url": "https://example",
                "doi": "",
                "created_at": "now",
            },
            {
                "point_id": "p2",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "temperature_c",
                "raw_value": "350",
                "normalized_value": 350.0,
                "unit": "c",
                "entity": "NiSi",
                "extraction_type": "text",
                "confidence": 0.78,
                "snippet": "temperature 350 C",
                "locator": "line:11",
                "line_no": 11,
                "citation_url": "https://example",
                "doi": "",
                "created_at": "now",
            },
        ]
    )

    assembled = WindowContextAssembler(context_window_lines=1).assemble(points, pd.DataFrame(), spec)
    assert not assembled.empty
    vars_present = set(assembled["variable_name"].tolist())
    assert "thickness_nm" in vars_present and "temperature_c" in vars_present


def test_context_enrichment_uses_document_level_fallback() -> None:
    spec = _spec()
    cfg = RunConfig(run_dir="/tmp/oarp", require_context_fields=True)
    frame = pd.DataFrame(
        [
            {
                "point_id": "p-doc",
                "article_key": "a-doc",
                "provider": "mock",
                "variable_name": "temperature_c",
                "raw_value": "290",
                "normalized_value": 290.0,
                "unit": "c",
                "entity": "NiSi",
                "extraction_type": "text",
                "confidence": 0.85,
                "snippet": "NiSi appears around 290 C for 3 nm films.",
                "locator": "sentence:1",
                "line_no": 1,
                "citation_url": "https://example",
                "doi": "",
                "created_at": "now",
            }
        ]
    )
    documents = pd.DataFrame(
        [
            {
                "article_key": "a-doc",
                "provider": "mock",
                "source_url": "https://example",
                "text_content": (
                    "As-deposited Ni film on Si(100) substrate was annealed. "
                    "No dopant was intentionally added to the Ni layer."
                ),
            }
        ]
    )
    article_map = {
        "a-doc": {
            "title": "Nickel silicide phase evolution",
            "abstract": "Ni layers on silicon are annealed.",
        }
    }

    out = _enrich_with_context(
        frame,
        spec=spec,
        cfg=cfg,
        article_map=article_map,
        documents=documents,
    )
    assert not out.empty
    row = out.iloc[0]
    assert str(row.get("substrate_material") or "") == "Si"
    assert str(row.get("substrate_orientation") or "") in {"(100)", "(111)"}
    assert str(row.get("doping_state") or "") in {"undoped", "na_pure_ni"}
    assert str(row.get("alloy_state") or "") in {"pure", "na_pure_ni"}
