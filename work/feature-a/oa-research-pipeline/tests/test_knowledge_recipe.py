from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oarp.knowledge import build_knowledge
from oarp.pipeline import _initialize_run_state
from oarp.recipe import generate_recipes
from oarp.service_api import get_job_status, run_pending_jobs, submit_recipe_generation_job
from oarp.service_models import JobStatus, RecipeGenerateRequest, ServiceConfig
from oarp.topic_spec import dump_topic_spec, spec_from_dict


def _spec():
    return spec_from_dict(
        {
            "topic_id": "global-film",
            "keywords": ["thin film", "phase transition"],
            "variables": [
                {"name": "thickness_nm", "unit": "nm", "datatype": "float", "normalization": "length"},
                {"name": "temperature_c", "unit": "c", "datatype": "float", "normalization": "temperature"},
            ],
            "entities": [
                {"name": "Ni2Si", "aliases": ["ni2si"]},
                {"name": "NiSi", "aliases": ["nisi"]},
            ],
            "plot": {"primary": {"x": "thickness_nm", "y": "temperature_c", "color_by": "entity"}},
        }
    )


def _seed_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    spec_path = run_dir / "topic.yaml"
    dump_topic_spec(spec, spec_path)
    from oarp.models import RunConfig

    _initialize_run_state(str(spec_path), spec, "thin film phase", RunConfig(run_dir=run_dir, require_context_fields=False))
    artifacts = run_dir / "artifacts"
    validated = pd.DataFrame(
        [
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "normalized_value": 20.0,
                "unit": "nm",
                "entity": "Ni2Si",
                "confidence": 0.92,
                "snippet": "Ni2Si appears at 260 C on Si(111) after 120 s anneal.",
                "locator": "sentence:1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "doping_elements": "[]",
                "doping_composition": "[]",
                "alloy_state": "na_pure_ni",
                "alloy_elements": "[]",
                "alloy_composition": "[]",
                "pure_ni_evidence": True,
            },
            {
                "point_id": "p1",
                "article_key": "a1",
                "provider": "mock",
                "variable_name": "temperature_c",
                "normalized_value": 260.0,
                "unit": "c",
                "entity": "Ni2Si",
                "confidence": 0.92,
                "snippet": "Ni2Si appears at 260 C on Si(111) after 120 s anneal.",
                "locator": "sentence:1",
                "citation_url": "https://example/a1",
                "doi": "10.1/a1",
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "doping_elements": "[]",
                "doping_composition": "[]",
                "alloy_state": "na_pure_ni",
                "alloy_elements": "[]",
                "alloy_composition": "[]",
                "pure_ni_evidence": True,
            },
            {
                "point_id": "p2",
                "article_key": "a2",
                "provider": "mock",
                "variable_name": "thickness_nm",
                "normalized_value": 30.0,
                "unit": "nm",
                "entity": "NiSi",
                "confidence": 0.9,
                "snippet": "NiSi forms around 340 C on Si(111) with sputter deposition.",
                "locator": "sentence:1",
                "citation_url": "https://example/a2",
                "doi": "10.1/a2",
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "doping_elements": "[]",
                "doping_composition": "[]",
                "alloy_state": "na_pure_ni",
                "alloy_elements": "[]",
                "alloy_composition": "[]",
                "pure_ni_evidence": True,
            },
            {
                "point_id": "p2",
                "article_key": "a2",
                "provider": "mock",
                "variable_name": "temperature_c",
                "normalized_value": 340.0,
                "unit": "c",
                "entity": "NiSi",
                "confidence": 0.9,
                "snippet": "NiSi forms around 340 C on Si(111) with sputter deposition.",
                "locator": "sentence:1",
                "citation_url": "https://example/a2",
                "doi": "10.1/a2",
                "substrate_material": "Si",
                "substrate_orientation": "(111)",
                "doping_state": "na_pure_ni",
                "doping_elements": "[]",
                "doping_composition": "[]",
                "alloy_state": "na_pure_ni",
                "alloy_elements": "[]",
                "alloy_composition": "[]",
                "pure_ni_evidence": True,
            },
        ]
    )
    validated.to_parquet(artifacts / "validated_points.parquet", index=False)


def test_build_knowledge_and_generate_recipes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run(run_dir)

    kb = build_knowledge(run_dir)
    assert kb.phase_event_count > 0
    assert kb.phase_events_path.exists()
    assert kb.condition_graph_path.exists()
    assert kb.quality_outcomes_path.exists()

    req = RecipeGenerateRequest(
        target_film_material="NiSi",
        target_phase_target={"type": "phase_label", "value": "NiSi"},
        knowledge_run_dir=str(run_dir),
        text_constraints={"method_family": ["PVD"]},
        top_k=5,
        use_materials_project=False,
    )
    result, artifacts = generate_recipes(
        run_dir=run_dir,
        request=req,
        materials_project_enabled=False,
    )
    assert result.status == "SUCCEEDED"
    assert artifacts.recipe_count >= 1
    assert artifacts.cards_path.exists()
    assert artifacts.ranked_path.exists()
    assert artifacts.materials_project_refs_path.exists()
    mp_payload = json.loads(artifacts.materials_project_refs_path.read_text(encoding="utf-8"))
    assert mp_payload["status"] == "disabled"


def test_submit_recipe_job_and_run_worker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_run(run_dir)
    _ = build_knowledge(run_dir)

    service_cfg = ServiceConfig(
        data_dir=str(tmp_path / "service"),
        materials_project_enabled=False,
    )
    request = RecipeGenerateRequest(
        target_film_material="NiSi",
        target_phase_target={"type": "phase_label", "value": "NiSi"},
        knowledge_run_dir=str(run_dir),
        text_constraints={"method_family": ["PVD"]},
        top_k=3,
    )
    ref = submit_recipe_generation_job(request, service_cfg)
    assert ref.status == JobStatus.PENDING

    ran = run_pending_jobs(service_cfg, max_jobs=5)
    assert ran >= 1

    status = get_job_status(ref.job_id, service_cfg)
    assert status.status == JobStatus.SUCCEEDED
