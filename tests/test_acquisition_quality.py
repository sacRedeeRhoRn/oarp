from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.acquisition import AcquisitionPlanner, DocumentQualityGate, acquire
from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import spec_from_dict


def test_acquisition_planner_skips_metadata_api_urls() -> None:
    cfg = RunConfig(run_dir="/tmp/oarp-test")
    planner = AcquisitionPlanner()

    row = {
        "provider": "openalex",
        "oa_url": "https://api.openalex.org/works/W123",
        "source_url": "https://api.openalex.org/works/W123",
        "raw_json": json.dumps(
            {
                "best_oa_location": {
                    "pdf_url": "https://example.org/paper.pdf",
                    "landing_page_url": "https://example.org/paper",
                },
                "open_access": {"oa_url": "https://example.org/open"},
            }
        ),
    }

    urls = planner.plan(row, cfg)
    assert urls
    assert "https://example.org/paper.pdf" in urls
    assert all("api.openalex.org/works" not in item for item in urls)


def test_document_quality_gate_enforces_text_and_parse_rules() -> None:
    cfg = RunConfig(run_dir="/tmp/oarp-test", min_text_length=100)
    gate = DocumentQualityGate(cfg)

    usable, reason = gate.assess(
        {
            "parse_status": "parsed_html",
            "mime": "text/html",
            "text_content": "hello" * 40,
        }
    )
    assert usable is True
    assert reason == "usable"

    usable2, reason2 = gate.assess(
        {
            "parse_status": "stored_binary",
            "mime": "application/pdf",
            "text_content": "",
        }
    )
    assert usable2 is False
    assert reason2 == "unsupported_parse_status"


def test_acquire_reads_local_file_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    local_file = tmp_path / "doc.txt"
    local_file.write_text("NiSi forms at 350 C for 30 nm film thickness.", encoding="utf-8")

    spec = spec_from_dict(
        {
            "topic_id": "local-acquire",
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

    cfg = RunConfig(
        run_dir=run_dir,
        max_downloads=5,
        min_text_length=1,
        require_fulltext_mime=True,
        local_merge_mode="local_only",
        local_repo_paths=[str(tmp_path)],
    )
    _initialize_run_state(str(tmp_path / "topic.yaml"), spec, "q", cfg)

    artifacts = run_dir / "artifacts"
    pd.DataFrame(
        [
            {
                "article_key": "a1",
                "provider": "localfs",
                "source_id": "a1",
                "doi": "",
                "title": "Local paper",
                "abstract": "",
                "year": 2020,
                "venue": "local_repository",
                "article_type": "journal-article",
                "is_oa": True,
                "oa_url": local_file.as_uri(),
                "source_url": local_file.as_uri(),
                "license": "localfs",
                "language": "en",
                "raw_json": json.dumps({"local_path": str(local_file)}),
            }
        ]
    ).to_parquet(artifacts / "articles.parquet", index=False)

    result = acquire(cfg=cfg)
    assert len(result.frame) == 1
    row = result.frame.iloc[0]
    assert str(row["provider"]) == "localfs"
    assert str(row["parse_status"]).startswith("parsed_")
    assert "NiSi forms" in str(row["text_content"])

    debug_path = artifacts / "acquisition_debug.parquet"
    debug = pd.read_parquet(debug_path)
    assert "local_selected" in set(debug["reason"].astype(str).tolist())
    assert "local_copy_ok" in set(debug["reason"].astype(str).tolist())
