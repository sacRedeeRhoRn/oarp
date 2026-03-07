from __future__ import annotations

import pandas as pd

from oarp.discovery import WeightedCandidateScorer
from oarp.plugins.topic_plugins import load_topic_plugin
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni",
            "keywords": ["nickel silicide", "annealing", "thickness"],
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "entities": [
                {"name": "Ni2Si", "aliases": ["ni2si"]},
                {"name": "NiSi", "aliases": ["nisi"]},
                {"name": "NiSi2", "aliases": ["nisi2"]},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c"}},
        }
    )


def test_candidate_scorer_ranks_on_topic_higher() -> None:
    spec = _spec()
    plugin = load_topic_plugin("materials_ni_silicide")
    scorer = WeightedCandidateScorer(plugin=plugin)

    frame = pd.DataFrame(
        [
            {
                "title": "Nickel silicide phase sequence Ni2Si NiSi NiSi2 with annealing",
                "abstract": "Dependence of phase formation on nickel film thickness and temperature",
                "provider": "openalex",
                "oa_url": "https://example.org/paper.pdf",
                "year": 2024,
            },
            {
                "title": "Ferroelectric hafnia phase transitions in undoped HfO2 films",
                "abstract": "Thickness dependence in oxide systems",
                "provider": "arxiv",
                "oa_url": "https://example.org/hfo2",
                "year": 2024,
            },
        ]
    )

    ranked = scorer.score_candidates(frame, spec, "nickel silicide thickness annealing NiSi")
    assert float(ranked.iloc[0]["discovery_score"]) > float(ranked.iloc[1]["discovery_score"])
    top_gate = bool(ranked.iloc[0]["relevance_gate_ok"])
    bottom_gate = bool(ranked.iloc[1]["relevance_gate_ok"])
    assert top_gate is True
    assert bottom_gate is False
