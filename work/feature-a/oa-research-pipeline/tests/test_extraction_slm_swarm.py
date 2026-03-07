from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp import extraction as extraction_stage
from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "swarm-demo",
            "keywords": ["nickel silicide"],
            "variables": [
                {"name": "thickness_nm", "aliases": ["thickness"], "unit": "nm", "datatype": "float"},
                {"name": "temperature_c", "aliases": ["temperature"], "unit": "c", "datatype": "float"},
            ],
            "entities": [{"name": "NiSi", "aliases": ["nisi"]}],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
            "validation": {
                "min_confidence": 0.65,
                "required_provenance_fields": ["citation_url", "snippet", "locator"],
            },
        }
    )


def test_slm_swarm_extract_emits_votes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = RunConfig(
        run_dir=run_dir,
        require_context_fields=False,
        extractor_mode="slm_swarm",
        min_vote_confidence=0.20,
        extractor_models=["m1", "m2", "m3"],
        extract_workers=2,
    )
    spec = _spec()
    _initialize_run_state(str(tmp_path / "topic.yaml"), spec, "q", cfg)
    artifacts = run_dir / "artifacts"

    pd.DataFrame(
        [
            {
                "article_key": "a1",
                "provider": "mock",
                "title": "NiSi thin film",
                "abstract": "NiSi forms at 350 C for 30 nm films",
                "doi": "10.1/a1",
                "oa_url": "https://example/a1",
                "source_url": "https://example/a1",
            }
        ]
    ).to_parquet(artifacts / "articles.parquet", index=False)

    pd.DataFrame(
        [
            {
                "article_key": "a1",
                "provider": "mock",
                "source_url": "https://example/a1",
                "local_path": "/tmp/a1.txt",
                "mime": "text/plain",
                "content_hash": "x",
                "parse_status": "parsed_text",
                "text_content": "NiSi forms for thickness 30 nm at temperature 350 C.",
                "usable_text": True,
            }
        ]
    ).to_parquet(artifacts / "documents.parquet", index=False)

    result = extraction_stage.extract(spec=spec, cfg=cfg)
    assert not result.points.empty

    votes_path = artifacts / "extraction_votes.parquet"
    errors_path = artifacts / "extraction_error_slices.parquet"
    assert votes_path.exists()
    assert errors_path.exists()

    votes = pd.read_parquet(votes_path)
    assert not votes.empty
    assert "accepted" in votes.columns
