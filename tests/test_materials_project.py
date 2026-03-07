from __future__ import annotations

from oarp.materials_project import fetch_materials_project_refs, score_materials_project_alignment


def test_fetch_materials_project_refs_no_key() -> None:
    result = fetch_materials_project_refs(
        film_material="NiSi",
        phase_target_type="phase_label",
        phase_target_value="NiSi",
        enabled=True,
        api_key="",
    )
    assert result.status == "no_api_key"
    assert result.references == []


def test_score_materials_project_alignment_matches() -> None:
    refs = [
        {
            "material_id": "mp-1",
            "formula_pretty": "NiSi",
            "spacegroup_symbol": "Pnma",
            "spacegroup_number": "62",
        }
    ]
    hit = score_materials_project_alignment(
        candidate={"film_material": "NiSi", "phase_label": "NiSi"},
        target_film_material="NiSi",
        phase_target_type="space_group",
        phase_target_value="Pnma",
        references=refs,
    )
    miss = score_materials_project_alignment(
        candidate={"film_material": "Ni2Si", "phase_label": "Ni2Si"},
        target_film_material="NiSi",
        phase_target_type="space_group",
        phase_target_value="Fm-3m",
        references=refs,
    )
    assert hit["mp_formula_match"] is True
    assert hit["mp_phase_match"] is True
    assert hit["mp_bonus"] > 0
    assert miss["mp_bonus"] < hit["mp_bonus"]
