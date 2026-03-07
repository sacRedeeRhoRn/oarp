from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class MaterialsProjectBundle:
    status: str
    query: dict[str, Any]
    references: list[dict[str, Any]]
    error: str = ""


def _canonical_formula(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def _spacegroup_tokens(value: Any) -> set[str]:
    text = str(value or "").strip().lower()
    if not text:
        return set()
    tokens = {text}
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        tokens.add(digits)
    for part in re.split(r"[\s,/;_-]+", text):
        clean = part.strip()
        if clean:
            tokens.add(clean)
            digit_part = "".join(ch for ch in clean if ch.isdigit())
            if digit_part:
                tokens.add(digit_part)
    return tokens


def _extract_documents(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    response = payload.get("response")
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        docs = response.get("docs")
        if isinstance(docs, list):
            return [item for item in docs if isinstance(item, dict)]
    return []


def _normalize_reference(doc: dict[str, Any]) -> dict[str, Any]:
    symmetry = doc.get("symmetry")
    if not isinstance(symmetry, dict):
        symmetry = {}
    return {
        "material_id": str(doc.get("material_id") or doc.get("material_ids") or ""),
        "formula_pretty": str(doc.get("formula_pretty") or doc.get("formula") or ""),
        "spacegroup_symbol": str(symmetry.get("symbol") or doc.get("spacegroup_symbol") or ""),
        "spacegroup_number": str(symmetry.get("number") or doc.get("spacegroup_number") or ""),
        "crystal_system": str(symmetry.get("crystal_system") or doc.get("crystal_system") or ""),
        "energy_above_hull": doc.get("energy_above_hull"),
        "is_theoretical": bool(doc.get("theoretical") or doc.get("is_theoretical") or False),
    }


def _thermo_energy_by_material(
    *,
    material_ids: list[str],
    api_key: str,
    endpoint: str,
    timeout_sec: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    base = str(endpoint or "").strip().rstrip("/")
    headers = {"X-API-KEY": api_key, "accept": "application/json"}
    for material_id in material_ids:
        mid = str(material_id or "").strip()
        if not mid:
            continue
        params = {
            "material_ids": mid,
            "_fields": "material_id,energy_above_hull",
            "_limit": 1,
        }
        try:
            response = requests.get(
                f"{base}/materials/thermo/",
                params=params,
                headers=headers,
                timeout=max(5.0, float(timeout_sec)),
            )
            if response.status_code >= 400:
                continue
            payload = response.json()
        except Exception:
            continue
        docs = _extract_documents(payload)
        for item in docs:
            item_mid = str(item.get("material_id") or "").strip()
            if not item_mid:
                continue
            try:
                out[item_mid] = float(item.get("energy_above_hull"))
            except Exception:
                continue
    return out


def _filter_by_phase_target(
    refs: list[dict[str, Any]],
    phase_target_type: str,
    phase_target_value: str,
) -> list[dict[str, Any]]:
    target = str(phase_target_value or "").strip()
    if not target:
        return refs

    if str(phase_target_type or "").strip().lower() == "space_group":
        target_tokens = _spacegroup_tokens(target)
        out: list[dict[str, Any]] = []
        for ref in refs:
            ref_tokens = _spacegroup_tokens(ref.get("spacegroup_symbol")) | _spacegroup_tokens(
                ref.get("spacegroup_number")
            )
            if ref_tokens and target_tokens.intersection(ref_tokens):
                out.append(ref)
        return out
    return refs


def fetch_materials_project_refs(
    *,
    film_material: str,
    phase_target_type: str,
    phase_target_value: str,
    enabled: bool,
    api_key: str,
    endpoint: str = "https://api.materialsproject.org",
    timeout_sec: float = 20.0,
    max_results: int = 100,
    scope: str = "summary_thermo",
) -> MaterialsProjectBundle:
    query = {
        "film_material": str(film_material or ""),
        "phase_target_type": str(phase_target_type or ""),
        "phase_target_value": str(phase_target_value or ""),
        "endpoint": str(endpoint or ""),
        "max_results": int(max_results),
        "scope": str(scope or ""),
    }
    if not enabled:
        return MaterialsProjectBundle(status="disabled", query=query, references=[])
    key = str(api_key or "").strip()
    if not key:
        return MaterialsProjectBundle(status="no_api_key", query=query, references=[])

    base = str(endpoint or "").strip().rstrip("/")
    if not base:
        return MaterialsProjectBundle(status="invalid_endpoint", query=query, references=[])

    params = {
        "formula": str(film_material or "").strip(),
        "_fields": "material_id,formula_pretty,symmetry,energy_above_hull,theoretical",
        "_limit": max(1, min(int(max_results), 500)),
    }
    headers = {"X-API-KEY": key, "accept": "application/json"}

    try:
        response = requests.get(
            f"{base}/materials/summary/",
            params=params,
            headers=headers,
            timeout=max(5.0, float(timeout_sec)),
        )
    except Exception as exc:
        return MaterialsProjectBundle(
            status="fetch_error",
            query=query,
            references=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    if response.status_code >= 400:
        detail = response.text[:200].strip()
        return MaterialsProjectBundle(
            status="fetch_error",
            query=query,
            references=[],
            error=f"http_{response.status_code}: {detail}",
        )

    try:
        payload = response.json()
    except Exception as exc:
        return MaterialsProjectBundle(
            status="parse_error",
            query=query,
            references=[],
            error=f"{type(exc).__name__}: {exc}",
        )

    docs = _extract_documents(payload)
    refs = [_normalize_reference(item) for item in docs]
    refs = _filter_by_phase_target(refs, phase_target_type, phase_target_value)
    refs = refs[: max(1, min(int(max_results), 500))]

    scope_name = str(scope or "").strip().lower()
    if scope_name in {"summary_thermo", "broad"} and refs and key:
        mids = [str(item.get("material_id") or "") for item in refs if str(item.get("material_id") or "")]
        thermo = _thermo_energy_by_material(
            material_ids=mids,
            api_key=key,
            endpoint=base,
            timeout_sec=timeout_sec,
        )
        if thermo:
            for item in refs:
                mid = str(item.get("material_id") or "")
                if mid in thermo:
                    item["energy_above_hull"] = thermo[mid]

    return MaterialsProjectBundle(status="success", query=query, references=refs)


def score_materials_project_alignment(
    *,
    candidate: dict[str, Any],
    target_film_material: str,
    phase_target_type: str,
    phase_target_value: str,
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    if not references:
        return {
            "mp_bonus": 0.0,
            "mp_formula_match": False,
            "mp_phase_match": False,
            "mp_support_count": 0,
        }

    target_formula = _canonical_formula(target_film_material).lower()
    candidate_formula = _canonical_formula(
        candidate.get("film_material") or candidate.get("phase_label") or target_film_material
    ).lower()
    target_phase_value = str(phase_target_value or "").strip().lower()
    target_phase_type = str(phase_target_type or "").strip().lower()

    formula_hits = 0
    phase_hits = 0
    for ref in references:
        ref_formula = _canonical_formula(ref.get("formula_pretty")).lower()
        formula_match = bool(ref_formula) and (
            ref_formula == target_formula
            or ref_formula == candidate_formula
            or target_formula in ref_formula
            or candidate_formula in ref_formula
        )
        if formula_match:
            formula_hits += 1

        phase_match = False
        if target_phase_type == "space_group":
            target_tokens = _spacegroup_tokens(target_phase_value)
            ref_tokens = _spacegroup_tokens(ref.get("spacegroup_symbol")) | _spacegroup_tokens(
                ref.get("spacegroup_number")
            )
            phase_match = bool(target_tokens and ref_tokens and target_tokens.intersection(ref_tokens))
        elif target_phase_type == "phase_label":
            phase_label = str(candidate.get("phase_label") or "").strip().lower()
            phase_match = bool(target_phase_value) and (
                target_phase_value == phase_label
                or target_phase_value == ref_formula
                or target_phase_value in phase_label
                or target_phase_value in ref_formula
            )
        if phase_match:
            phase_hits += 1

    formula_match = formula_hits > 0
    phase_match = phase_hits > 0

    bonus = 0.0
    bonus += 0.04 if formula_match else -0.04
    if target_phase_type == "space_group":
        bonus += 0.04 if phase_match else -0.05
    elif target_phase_type == "phase_label":
        bonus += 0.02 if phase_match else -0.01

    bonus = max(-0.12, min(0.12, bonus))
    return {
        "mp_bonus": bonus,
        "mp_formula_match": formula_match,
        "mp_phase_match": phase_match,
        "mp_support_count": max(formula_hits, phase_hits),
    }
