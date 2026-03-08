from __future__ import annotations

from pathlib import Path

import pandas as pd

from oarp.models import RunConfig
from oarp.obsidian_vault import (
    benchmark_vault_alignment,
    export_obsidian_vault,
    import_obsidian_vault,
)
from oarp.runtime import ensure_run_layout, write_run_state


def test_vault_export_import_benchmark_roundtrip(tmp_path: Path) -> None:
    run_dir = (tmp_path / "run").resolve()
    layout = ensure_run_layout(run_dir)
    artifacts = layout["artifacts"]

    write_run_state(
        run_dir,
        {
            "run_id": "run-test-vault",
            "topic_id": "vault-test",
            "query": "nickel silicide",
            "spec_path": str(tmp_path / "topic.yaml"),
        },
    )
    pd.DataFrame(
        [
            {
                "article_key": "a1",
                "title": "NiSi formation",
                "provider": "localfs",
                "source_url": "https://example.org/a1",
                "doi": "10.1000/a1",
            }
        ]
    ).to_parquet(artifacts / "articles.parquet", index=False)
    pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "entity": "NiSi",
                "snippet": "NiSi appears at 350 C for 30 nm film.",
                "locator": "line:10",
                "citation_url": "https://example.org/a1",
                "confidence": 0.92,
                "substrate_material": "Si",
                "method_family": "PVD",
            }
        ]
    ).to_parquet(artifacts / "validated_points.parquet", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "article_key": "a1",
                "phase_label": "NiSi",
                "point_id": "p1",
                "thickness_nm": 30.0,
                "anneal_temperature_c": 350.0,
            }
        ]
    ).to_parquet(artifacts / "phase_events.parquet", index=False)

    cfg = RunConfig(run_dir=run_dir)
    vault_dir = run_dir / "outputs" / "vault"
    exported = export_obsidian_vault(run_dir=run_dir, out_dir=vault_dir, cfg=cfg)
    assert exported.vault_path.exists()
    assert (vault_dir / "Articles").exists()
    assert (artifacts / "vault_link_index.parquet").exists()

    concept_note = vault_dir / "Concepts" / "film" / "NiSi.md"
    text = concept_note.read_text(encoding="utf-8")
    concept_note.write_text(text + "\n- [[Concepts/method/PVD]]\n", encoding="utf-8")

    imported = import_obsidian_vault(vault_dir=vault_dir, run_dir=run_dir, cfg=cfg)
    assert imported.parsed_links_path.exists()
    assert imported.soft_constraints_path.exists()
    assert len(imported.parsed_links) >= 1

    bench = benchmark_vault_alignment(run_dir=run_dir, vault_dir=vault_dir, cfg=cfg)
    assert bench.json_path.exists()
    assert bench.report_path.exists()
