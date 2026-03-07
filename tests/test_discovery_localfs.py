from __future__ import annotations

from pathlib import Path

from oarp.discovery import discover
from oarp.models import RunConfig
from oarp.pipeline import _initialize_run_state
from oarp.topic_spec import spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "local-discovery",
            "keywords": ["nickel silicide", "annealing"],
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


def test_localfs_discovery_writes_local_artifacts(tmp_path: Path) -> None:
    repo = tmp_path / "pdf_repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "(2019) Nickel silicide phase transition.pdf").write_bytes(b"%PDF-1.4\n%stub")
    (repo / "(2020) Annealing behavior of NiSi films.pdf").write_bytes(b"%PDF-1.4\n%stub2")
    (repo / "notes.txt").write_text("ignore", encoding="utf-8")

    run_dir = tmp_path / "run"
    cfg = RunConfig(
        run_dir=run_dir,
        local_repo_paths=[str(repo)],
        local_merge_mode="local_only",
        local_file_glob="*.pdf",
        local_repo_recursive=True,
        max_pages_per_provider=1,
        max_per_provider=50,
    )
    spec = _spec()
    _initialize_run_state(str(tmp_path / "topic.yaml"), spec, "nickel silicide annealing", cfg)

    index = discover(spec=spec, query="nickel silicide annealing", cfg=cfg)
    assert len(index.frame) >= 2
    assert set(index.frame["provider"].astype(str).str.lower().tolist()) == {"localfs"}

    artifacts = run_dir / "artifacts"
    assert (artifacts / "local_repository_index.parquet").exists()
    assert (artifacts / "local_repository_manifest.json").exists()
    assert (artifacts / "local_discovery_debug.parquet").exists()
