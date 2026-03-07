from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from oarp import extraction as extraction_stage
from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "tgi-demo",
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


def _seed_run(run_dir: Path, cfg: RunConfig) -> None:
    spec = _spec()
    _initialize_run_state(str(run_dir / "topic.yaml"), spec, "q", cfg)
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


def test_tgi_required_mode_without_endpoint_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = RunConfig(
        run_dir=run_dir,
        extractor_mode="slm_tgi_required",
        tgi_models=["m1"],
        require_context_fields=False,
    )
    spec = _spec()
    _seed_run(run_dir, cfg)

    with pytest.raises(RuntimeError, match="requires --tgi-endpoint"):
        extraction_stage.extract(spec=spec, cfg=cfg)


def test_tgi_required_mode_emits_slm_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 200
        ok = True

        def raise_for_status(self) -> None:
            return None

        @property
        def text(self) -> str:
            return ""

        def json(self):
            payload = {
                "points": [
                    {
                        "entity": "NiSi",
                        "thickness_nm": 30,
                        "temperature_c": 350,
                        "substrate_material": "Si",
                        "substrate_orientation": "(111)",
                        "doping_state": "na_pure_ni",
                        "doping_elements": [],
                        "doping_composition": [],
                        "alloy_state": "na_pure_ni",
                        "alloy_elements": [],
                        "alloy_composition": [],
                        "snippet": "NiSi forms for thickness 30 nm at temperature 350 C.",
                        "locator": "chunk:1:point:1",
                        "confidence": 0.91,
                    }
                ]
            }
            return {"generated_text": json.dumps(payload)}

    def _fake_post(*args, **kwargs):  # noqa: ANN002, ANN003
        return _FakeResponse()

    monkeypatch.setattr(extraction_stage.requests, "post", _fake_post)

    run_dir = tmp_path / "run"
    cfg = RunConfig(
        run_dir=run_dir,
        extractor_mode="slm_tgi_required",
        tgi_endpoint="http://localhost:8080/generate",
        tgi_models=["m1", "m2"],
        min_vote_confidence=0.1,
        require_context_fields=False,
    )
    spec = _spec()
    _seed_run(run_dir, cfg)

    result = extraction_stage.extract(spec=spec, cfg=cfg)
    assert not result.points.empty

    artifacts = run_dir / "artifacts"
    assert (artifacts / "slm_requests.parquet").exists()
    assert (artifacts / "slm_responses.parquet").exists()
    assert (artifacts / "slm_points_raw.parquet").exists()
    assert (artifacts / "slm_points_voted.parquet").exists()
    assert (artifacts / "extraction_votes.parquet").exists()
