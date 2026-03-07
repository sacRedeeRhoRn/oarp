from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.runtime import now_iso


def build_agents_inc_handoff(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Create a minimal `agents-inc` consumable handoff.json from OARP artifacts."""
    root = Path(run_dir).expanduser().resolve()
    artifacts = root / "artifacts"
    outputs = root / "outputs"

    consensus_path = artifacts / "consensus_points.parquet"
    recipe_ranked_path = artifacts / "recipe_ranked.parquet"
    recipe_cards_path = artifacts / "recipe_cards.json"
    report_path = outputs / "report.md"
    citation_path = outputs / "citation_points.csv"

    consensus = pd.read_parquet(consensus_path) if consensus_path.exists() else pd.DataFrame()
    recipe_ranked = pd.read_parquet(recipe_ranked_path) if recipe_ranked_path.exists() else pd.DataFrame()
    claims = []
    for _, row in consensus.head(20).iterrows():
        claim = {
            "claim": (
                f"At {row.get('x_name')}={row.get('x_value')}, entity {row.get('entity')} "
                f"observed near {row.get('y_name')}={row.get('y_value')} "
                f"(consensus {row.get('consensus_entity')}, entropy {row.get('entropy'):.3f})."
            ),
            "citation": str(row.get("citation_url") or ""),
        }
        claims.append(claim)
    if not claims and not recipe_ranked.empty:
        for _, row in recipe_ranked.head(20).iterrows():
            claims.append(
                {
                    "claim": (
                        "Recipe {0} targets {1} with {2} anneal at {3:.2f} C for {4:.2f} s "
                        "(score {5:.3f}).".format(
                            str(row.get("recipe_id") or ""),
                            str(row.get("target_film_material") or ""),
                            str(row.get("method_family") or ""),
                            float(row.get("anneal_temperature_c") or 0.0),
                            float(row.get("anneal_time_s") or 0.0),
                            float(row.get("weighted_score") or 0.0),
                        )
                    ),
                    "citation": str(row.get("citation_url") or ""),
                }
            )

    payload = {
        "schema_version": "3.0",
        "status": "VALID" if len(claims) > 0 else "PENDING",
        "generated_at": now_iso(),
        "artifacts": [
            str(report_path) if report_path.exists() else "",
            str(consensus_path) if consensus_path.exists() else "",
            str(citation_path) if citation_path.exists() else "",
            str(recipe_ranked_path) if recipe_ranked_path.exists() else "",
            str(recipe_cards_path) if recipe_cards_path.exists() else "",
        ],
        "claims_with_citations": claims,
        "repro_steps": [
            f"oarp run --spec <topic.yaml> --query \"<question>\" --out {root}",
            f"oarp report --run {root}",
        ],
        "risks": [],
    }

    target = Path(out_path).expanduser().resolve() if out_path else outputs / "agents_inc_handoff.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
