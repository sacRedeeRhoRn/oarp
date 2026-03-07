from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp import consensus as consensus_stage
from oarp import extraction as extraction_stage
from oarp import render as render_stage
from oarp import validation as validation_stage
from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "ni-demo",
            "keywords": ["nickel silicide"],
            "variables": [
                {
                    "name": "thickness_nm",
                    "aliases": ["thickness", "film thickness"],
                    "unit": "nm",
                    "datatype": "float",
                    "normalization": "length",
                },
                {
                    "name": "temperature_c",
                    "aliases": ["temperature", "anneal"],
                    "unit": "c",
                    "datatype": "float",
                    "normalization": "temperature",
                },
            ],
            "entities": [
                {"name": "Ni2Si", "aliases": ["ni2si"]},
                {"name": "NiSi", "aliases": ["nisi"]},
            ],
            "plot": {
                "primary": {
                    "x": "thickness_nm",
                    "y": "temperature_c",
                    "color_by": "entity",
                    "transparency_policy": "outlier-transparent",
                }
            },
            "validation": {
                "min_confidence": 0.65,
                "required_provenance_fields": ["citation_url", "snippet", "locator"],
            },
        }
    )


def test_mock_extract_validate_consensus_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = RunConfig(run_dir=run_dir, require_context_fields=False)
    spec = _spec()
    _initialize_run_state(str(tmp_path / "spec.yaml"), spec, "q", cfg)

    artifacts = run_dir / "artifacts"

    articles = pd.DataFrame(
        [
            {
                "article_key": "a1",
                "provider": "mock",
                "source_id": "a1",
                "doi": "10.1/a1",
                "title": "paper",
                "abstract": "NiSi at thickness 30 nm appears around temperature 350 C",
                "year": 2021,
                "venue": "",
                "article_type": "journal-article",
                "is_oa": True,
                "oa_url": "https://oa/paper",
                "source_url": "https://oa/paper",
                "license": "",
                "language": "en",
                "discovered_at": "now",
                "raw_json": "{}",
            }
        ]
    )
    articles.to_parquet(artifacts / "articles.parquet", index=False)

    text = "NiSi appears when thickness 30 nm and temperature 350 C.\nFigure: Ni2Si with thickness 20 nm temperature 270 C"
    docs = pd.DataFrame(
        [
            {
                "article_key": "a1",
                "provider": "mock",
                "source_url": "https://oa/paper",
                "local_path": "/tmp/demo.txt",
                "mime": "text/plain",
                "content_hash": "abc",
                "parse_status": "parsed_text",
                "text_content": text,
            }
        ]
    )
    docs.to_parquet(artifacts / "documents.parquet", index=False)

    evidence = extraction_stage.extract(spec=spec, cfg=cfg)
    assert not evidence.points.empty

    validated = validation_stage.validate(spec=spec, cfg=cfg)
    assert len(validated.accepted) > 0

    cons = consensus_stage.build(spec=spec, cfg=cfg)
    assert "display_alpha" in cons.points.columns

    output = render_stage.render(spec=spec, consensus_set=cons, cfg=cfg)
    assert output.report_path.exists()
    assert output.plot_paths[0].exists()
