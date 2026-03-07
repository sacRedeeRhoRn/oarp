from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.knowledge import KnowledgeBundle, build_knowledge
from oarp.materials_project import fetch_materials_project_refs, score_materials_project_alignment
from oarp.processing import predict_with_processor
from oarp.runtime import now_iso
from oarp.service_models import (
    RecipeCard,
    RecipeCardStep,
    RecipeGenerateRequest,
    RecipeGenerateResult,
)


@dataclass
class RecipeArtifacts:
    candidates_path: Path
    ranked_path: Path
    cards_path: Path
    gate_audit_path: Path
    safety_audit_path: Path
    materials_project_refs_path: Path
    result_path: Path
    recipe_count: int
    phase: str


def _hash_recipe(parts: list[str]) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return f"rcp_{digest[:14]}"


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _range_ok(value: float | None, bounds: Any) -> bool:
    if value is None:
        return True
    if not hasattr(bounds, "min"):
        return True
    min_v = _to_float(getattr(bounds, "min", None))
    max_v = _to_float(getattr(bounds, "max", None))
    if min_v is not None and value < min_v:
        return False
    if max_v is not None and value > max_v:
        return False
    return True


def _method_candidates(req: RecipeGenerateRequest) -> list[str]:
    methods = [item.strip() for item in req.text_constraints.get("method_family", []) if item.strip()]
    if methods:
        return list(dict.fromkeys(methods))
    return ["PVD", "CVD", "ALD"]


def _default_precursor(method: str) -> str:
    clean = method.upper()
    if clean == "ALD":
        return "metal-organic precursor"
    if clean == "CVD":
        return "gas precursor"
    return "sputter target"


def _phase_match(row: dict[str, Any], req: RecipeGenerateRequest) -> bool:
    target_material = req.target_film_material.strip().lower()
    phase_label = str(row.get("phase_label") or "").strip().lower()
    film_material = str(row.get("film_material") or "").strip().lower()
    snippet = str(row.get("snippet") or "").strip().lower()

    base_match = target_material in phase_label or target_material in film_material or target_material in snippet
    if not base_match:
        return False

    phase_type = req.target_phase_target.type
    phase_value = req.target_phase_target.value.strip().lower()
    if phase_type == "phase_label":
        return phase_value in phase_label or phase_value in snippet
    if phase_type == "amorphous":
        return "amorph" in snippet or "amorph" in phase_label
    if phase_type == "space_group":
        return phase_value in snippet
    return base_match


def _apply_phase_event_mp_signal(
    *,
    evidence_support: float,
    reproducibility: float,
    row: dict[str, Any],
) -> tuple[float, float]:
    label = str(row.get("mp_interpretation_label") or "").strip().lower()
    score = _to_float(row.get("mp_interpretation_score"), None)
    if score is None:
        return evidence_support, reproducibility

    if label == "supports":
        evidence_support += 0.08 * score
        reproducibility += 0.06 * score
    elif label == "conflicts":
        evidence_support -= 0.14 * (1.0 - score)
        reproducibility -= 0.10 * (1.0 - score)
    elif label == "neutral":
        evidence_support += 0.02 * score
    return max(0.0, min(1.0, evidence_support)), max(0.0, min(1.0, reproducibility))


def _build_candidate_rows(
    *,
    phase_events: pd.DataFrame,
    req: RecipeGenerateRequest,
    attempt: int,
    rng: random.Random,
    materials_project_refs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if phase_events.empty:
        return []

    rows = phase_events.to_dict(orient="records")
    methods = _method_candidates(req)
    target_rows = [row for row in rows if _phase_match(row, req)]
    if not target_rows:
        target_rows = rows

    pool = target_rows if attempt == 0 else rows
    max_rows = min(len(pool), max(20, req.max_candidates // max(len(methods), 1)))
    pool = sorted(
        pool,
        key=lambda item: (
            -float(item.get("film_quality_score_numeric") or 0.0),
            str(item.get("phase_label") or ""),
        ),
    )[:max_rows]

    candidates: list[dict[str, Any]] = []
    for row in pool:
        for method in methods:
            thickness_nm = _to_float(row.get("thickness_nm"), 10.0) or 10.0
            temp_c = _to_float(row.get("anneal_temperature_c"), 400.0) or 400.0
            time_s = _to_float(row.get("anneal_time_s"), 120.0) or 120.0
            pressure_pa = _to_float(row.get("pressure_pa"), 101325.0) or 101325.0
            strain_pct = _to_float(row.get("strain_pct"), 0.0) or 0.0

            if attempt > 0:
                temp_c = max(20.0, temp_c + rng.uniform(-20.0, 25.0) * attempt)
                time_s = max(1.0, time_s * (1.0 + rng.uniform(-0.2, 0.4) * attempt))

            precursor_values = [item for item in req.text_constraints.get("precursor", []) if item.strip()]
            precursor = precursor_values[0] if precursor_values else _default_precursor(method)

            target_satisfaction = 1.0 if _phase_match(row, req) else 0.65
            numeric_checks = {
                "anneal_temperature_c": temp_c,
                "anneal_time_s": time_s,
                "pressure_pa": pressure_pa,
                "strain_pct": strain_pct,
                "thickness_nm": thickness_nm,
            }
            for key, value in numeric_checks.items():
                bounds = req.numeric_constraints.get(key)
                if bounds is not None and not _range_ok(value, bounds):
                    target_satisfaction -= 0.25

            substrate_constraints = [
                item.strip().lower() for item in req.text_constraints.get("substrate_material", []) if item.strip()
            ]
            substrate_material = str(row.get("substrate_material") or "")
            if substrate_constraints and substrate_material.strip().lower() not in substrate_constraints:
                target_satisfaction -= 0.15

            quality_score = max(0.0, min(1.0, float(row.get("film_quality_score_numeric") or 0.0)))
            support_hint = 1.0 if _phase_match(row, req) else 0.6
            evidence_support = max(0.0, min(1.0, 0.35 + 0.45 * quality_score + 0.20 * support_hint))
            reproducibility = max(0.0, min(1.0, 0.30 + 0.40 * quality_score + 0.30 * support_hint))
            evidence_support, reproducibility = _apply_phase_event_mp_signal(
                evidence_support=evidence_support,
                reproducibility=reproducibility,
                row=row,
            )

            mp_alignment = score_materials_project_alignment(
                candidate=row,
                target_film_material=req.target_film_material,
                phase_target_type=req.target_phase_target.type,
                phase_target_value=req.target_phase_target.value,
                references=materials_project_refs or [],
            )
            mp_bonus = float(mp_alignment["mp_bonus"])
            evidence_support = max(0.0, min(1.0, evidence_support + 0.5 * mp_bonus))
            reproducibility = max(0.0, min(1.0, reproducibility + 0.35 * mp_bonus))
            uncertainty = max(0.0, min(1.0, 1.0 - (0.6 * evidence_support + 0.4 * reproducibility)))

            novelty_seed = hashlib.sha1(
                f"{row.get('event_id')}|{method}|{attempt}".encode("utf-8", errors="replace")
            ).hexdigest()
            novelty = 0.15 + (int(novelty_seed[:4], 16) / 0xFFFF) * 0.75
            phase_prob = max(0.0, min(0.99, 0.55 * target_satisfaction + 0.45 * evidence_support))
            weighted_score = (
                0.45 * target_satisfaction
                + 0.30 * evidence_support
                + 0.20 * reproducibility
                + 0.05 * novelty
            )
            weighted_score = max(0.0, min(1.0, weighted_score + mp_bonus))

            recipe_id = _hash_recipe(
                [
                    str(row.get("event_id") or ""),
                    method.upper(),
                    f"{thickness_nm:.3f}",
                    f"{temp_c:.3f}",
                    f"{time_s:.3f}",
                    str(attempt),
                ]
            )

            candidates.append(
                {
                    "recipe_id": recipe_id,
                    "attempt": attempt,
                    "event_id": str(row.get("event_id") or ""),
                    "target_film_material": req.target_film_material,
                    "target_phase_type": req.target_phase_target.type,
                    "target_phase_value": req.target_phase_target.value,
                    "phase_label": str(row.get("phase_label") or ""),
                    "method_family": method.upper(),
                    "precursor": precursor,
                    "substrate_material": substrate_material,
                    "substrate_orientation": str(row.get("substrate_orientation") or ""),
                    "thickness_nm": thickness_nm,
                    "anneal_temperature_c": temp_c,
                    "anneal_time_s": time_s,
                    "pressure_pa": pressure_pa,
                    "strain_pct": strain_pct,
                    "composition_value": _to_float(row.get("composition_value"), None),
                    "vacancy_concentration_cm3": _to_float(row.get("vacancy_concentration_cm3"), None),
                    "film_quality_score_numeric": quality_score,
                    "film_quality_label": str(row.get("film_quality_label") or "unknown"),
                    "target_satisfaction": max(0.0, min(1.0, target_satisfaction)),
                    "evidence_support": evidence_support,
                    "reproducibility": reproducibility,
                    "uncertainty": uncertainty,
                    "novelty": novelty,
                    "phase_prob": phase_prob,
                    "weighted_score": weighted_score,
                    "mp_bonus": mp_bonus,
                    "mp_formula_match": bool(mp_alignment["mp_formula_match"]),
                    "mp_phase_match": bool(mp_alignment["mp_phase_match"]),
                    "mp_support_count": int(mp_alignment["mp_support_count"]),
                    "mp_interpretation_score": _to_float(row.get("mp_interpretation_score"), 0.0) or 0.0,
                    "mp_interpretation_label": str(row.get("mp_interpretation_label") or ""),
                    "mp_best_material_id": str(row.get("mp_best_material_id") or ""),
                    "citation_url": str(row.get("citation_url") or ""),
                    "doi": str(row.get("doi") or ""),
                    "locator": str(row.get("locator") or ""),
                    "snippet": str(row.get("snippet") or ""),
                    "created_at": now_iso(),
                }
            )
            if len(candidates) >= req.max_candidates:
                return candidates
    return candidates


def _apply_safety(
    candidates: pd.DataFrame,
    req: RecipeGenerateRequest,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return candidates, pd.DataFrame(columns=["recipe_id", "blocked", "reason", "created_at"])

    forbidden_precursors = {item.strip().lower() for item in req.safety_policy.forbidden_precursors if item.strip()}
    forbidden_methods = {item.strip().upper() for item in req.safety_policy.forbidden_methods if item.strip()}
    max_pressure = _to_float(req.safety_policy.max_pressure_pa, None)
    max_temp = _to_float(req.safety_policy.max_temperature_c, None)

    safety_rows: list[dict[str, Any]] = []
    keep_mask = []
    for _, row in candidates.iterrows():
        reasons: list[str] = []
        precursor = str(row.get("precursor") or "").lower()
        method = str(row.get("method_family") or "").upper()
        pressure = _to_float(row.get("pressure_pa"), None)
        temp = _to_float(row.get("anneal_temperature_c"), None)

        if forbidden_precursors and any(token in precursor for token in forbidden_precursors):
            reasons.append("forbidden_precursor")
        if forbidden_methods and method in forbidden_methods:
            reasons.append("forbidden_method")
        if max_pressure is not None and pressure is not None and pressure > max_pressure:
            reasons.append("pressure_exceeds_limit")
        if max_temp is not None and temp is not None and temp > max_temp:
            reasons.append("temperature_exceeds_limit")

        blocked = len(reasons) > 0
        keep_mask.append(not blocked)
        safety_rows.append(
            {
                "recipe_id": str(row.get("recipe_id")),
                "blocked": blocked,
                "reason": ";".join(reasons),
                "created_at": now_iso(),
            }
        )

    filtered = candidates[pd.Series(keep_mask, index=candidates.index)].copy()
    return filtered, pd.DataFrame(safety_rows)


def _progressive_gate(
    candidates: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    if candidates.empty:
        audit = pd.DataFrame(columns=["recipe_id", "stage", "passed", "reason", "created_at"])
        return candidates, audit, {"counts": {"loose": 0, "balanced": 0, "strict": 0, "best": 0}}, "none"

    thresholds = {
        "loose": {"evidence_support": 0.35, "uncertainty": 0.45},
        "balanced": {"evidence_support": 0.55, "reproducibility": 0.50, "uncertainty": 0.35},
        "strict": {"evidence_support": 0.70, "reproducibility": 0.65, "uncertainty": 0.25},
    }
    work = candidates.copy()
    audit_rows: list[dict[str, Any]] = []

    def _stage_pass(row: pd.Series, stage: str) -> tuple[bool, str]:
        t = thresholds[stage]
        reasons: list[str] = []
        if float(row.get("evidence_support") or 0.0) < float(t["evidence_support"]):
            reasons.append("low_evidence_support")
        if "reproducibility" in t and float(row.get("reproducibility") or 0.0) < float(t["reproducibility"]):
            reasons.append("low_reproducibility")
        if float(row.get("uncertainty") or 1.0) > float(t["uncertainty"]):
            reasons.append("high_uncertainty")
        return len(reasons) == 0, ";".join(reasons)

    stage_frames: dict[str, pd.DataFrame] = {}
    current = work
    for stage in ("loose", "balanced", "strict"):
        passed_mask = []
        for _, row in current.iterrows():
            passed, reason = _stage_pass(row, stage)
            passed_mask.append(passed)
            audit_rows.append(
                {
                    "recipe_id": str(row.get("recipe_id")),
                    "stage": stage,
                    "passed": passed,
                    "reason": reason,
                    "created_at": now_iso(),
                }
            )
        current = current[pd.Series(passed_mask, index=current.index)].copy()
        current["gate_stage_passed"] = stage
        stage_frames[stage] = current.copy()

    strict = stage_frames.get("strict", pd.DataFrame())
    if strict.empty:
        phase = "none"
        ranked = strict.copy()
    else:
        ranked = strict.sort_values(
            ["weighted_score", "target_satisfaction", "evidence_support", "reproducibility"],
            ascending=[False, False, False, False],
        ).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked = ranked.head(int(top_k)).copy()
        ranked["gate_stage_passed"] = "best"
        phase = "best"

    for _, row in ranked.iterrows():
        audit_rows.append(
            {
                "recipe_id": str(row.get("recipe_id")),
                "stage": "best",
                "passed": True,
                "reason": "",
                "created_at": now_iso(),
            }
        )

    gate_report = {
        "counts": {
            "loose": int(len(stage_frames.get("loose", pd.DataFrame()))),
            "balanced": int(len(stage_frames.get("balanced", pd.DataFrame()))),
            "strict": int(len(stage_frames.get("strict", pd.DataFrame()))),
            "best": int(len(ranked)),
        },
        "thresholds": thresholds,
    }
    return ranked, pd.DataFrame(audit_rows), gate_report, phase


def _apply_processor_scores(
    candidates: pd.DataFrame,
    req: RecipeGenerateRequest,
    processor_model_path: str,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    model_path = str(processor_model_path or "").strip()
    if not model_path:
        out = candidates.copy()
        out["processor_phase_prob"] = out.get("phase_prob", 0.0)
        out["processor_quality_score"] = out.get("film_quality_score_numeric", 0.0)
        out["processor_uncertainty"] = out.get("uncertainty", 1.0)
        return out
    try:
        scored = predict_with_processor(candidates, model_path, target_phase=str(req.target_phase_target.value))
    except Exception:
        scored = candidates.copy()
        scored["processor_phase_prob"] = scored.get("phase_prob", 0.0)
        scored["processor_quality_score"] = scored.get("film_quality_score_numeric", 0.0)
        scored["processor_uncertainty"] = scored.get("uncertainty", 1.0)

    # Blend processor outputs into ranking while preserving existing evidence-driven score.
    scored["weighted_score"] = (
        0.35 * scored["weighted_score"].astype(float)
        + 0.35 * scored["processor_phase_prob"].astype(float)
        + 0.20 * scored["processor_quality_score"].astype(float)
        + 0.10 * (1.0 - scored["processor_uncertainty"].astype(float))
    ).clip(lower=0.0, upper=1.0)
    scored["phase_prob"] = (
        0.55 * scored["phase_prob"].astype(float) + 0.45 * scored["processor_phase_prob"].astype(float)
    ).clip(lower=0.0, upper=1.0)
    scored["film_quality_score_numeric"] = (
        0.60 * scored["film_quality_score_numeric"].astype(float)
        + 0.40 * scored["processor_quality_score"].astype(float)
    ).clip(lower=0.0, upper=1.0)
    scored["uncertainty"] = (
        0.50 * scored["uncertainty"].astype(float)
        + 0.50 * scored["processor_uncertainty"].astype(float)
    ).clip(lower=0.0, upper=1.0)
    return scored


def _build_recipe_cards(rows: pd.DataFrame, req: RecipeGenerateRequest, mp_refs: list[dict[str, Any]]) -> list[RecipeCard]:
    cards: list[RecipeCard] = []
    if rows.empty:
        return cards
    for _, row in rows.iterrows():
        method = str(row.get("method_family") or "")
        precursor = str(row.get("precursor") or "")
        temp = _to_float(row.get("anneal_temperature_c"), 400.0) or 400.0
        time_s = _to_float(row.get("anneal_time_s"), 120.0) or 120.0
        pressure = _to_float(row.get("pressure_pa"), 101325.0) or 101325.0
        thickness = _to_float(row.get("thickness_nm"), 10.0) or 10.0
        phase_prob = _to_float(row.get("phase_prob"), 0.0) or 0.0
        q_score = _to_float(row.get("film_quality_score_numeric"), 0.0) or 0.0
        uncertainty = _to_float(row.get("uncertainty"), 1.0) or 1.0
        evidence_support = _to_float(row.get("evidence_support"), 0.0) or 0.0
        mp_interp_score = _to_float(row.get("mp_interpretation_score"), 0.0) or 0.0
        mp_interp_label = str(row.get("mp_interpretation_label") or "")
        mp_best_material_id = str(row.get("mp_best_material_id") or "")
        proc_phase_prob = _to_float(row.get("processor_phase_prob"), phase_prob) or phase_prob
        proc_quality = _to_float(row.get("processor_quality_score"), q_score) or q_score
        proc_uncertainty = _to_float(row.get("processor_uncertainty"), uncertainty) or uncertainty

        steps = [
            RecipeCardStep(
                step=1,
                action="deposit",
                method_family=method,
                inputs={
                    "precursor": precursor,
                    "substrate_material": str(row.get("substrate_material") or ""),
                    "substrate_orientation": str(row.get("substrate_orientation") or ""),
                    "thickness_nm": thickness,
                },
            ),
            RecipeCardStep(
                step=2,
                action="anneal",
                method_family="ANNEAL",
                inputs={
                    "temperature_c": temp,
                    "time_s": time_s,
                    "pressure_pa": pressure,
                    "ambient": "N2",
                },
            ),
        ]
        card = RecipeCard(
            recipe_id=str(row.get("recipe_id") or ""),
            target={
                "film_material": req.target_film_material,
                "phase_target": req.target_phase_target.model_dump(mode="python"),
            },
            method_flow=steps,
            condition_vector={
                "thickness_nm": thickness,
                "temperature_c": temp,
                "time_s": time_s,
                "pressure_pa": pressure,
                "strain_pct": _to_float(row.get("strain_pct"), 0.0),
            },
            predicted_outcomes={
                "phase_prob": phase_prob,
                "quality_score": q_score,
                "processor_phase_prob": proc_phase_prob,
                "processor_quality_score": proc_quality,
                "mp_interpretation_score": mp_interp_score,
                "mp_interpretation_label": mp_interp_label,
            },
            uncertainty={
                "epistemic": uncertainty,
                "processor_uncertainty": proc_uncertainty,
                "data_support": evidence_support,
            },
            evidence=[
                {
                    "doi": str(row.get("doi") or ""),
                    "locator": str(row.get("locator") or ""),
                    "citation_url": str(row.get("citation_url") or ""),
                    "materials_project_refs": mp_refs[:3],
                    "mp_best_material_id": mp_best_material_id,
                }
            ],
            safety_flags=[],
            gate_stage_passed=str(row.get("gate_stage_passed") or ""),
        )
        cards.append(card)
    return cards


def generate_recipes(
    *,
    run_dir: str | Path,
    request: RecipeGenerateRequest,
    materials_project_enabled: bool = True,
    materials_project_api_key: str = "",
    materials_project_endpoint: str = "https://api.materialsproject.org",
    mp_scope: str = "summary_thermo",
    processor_model_path: str = "",
) -> tuple[RecipeGenerateResult, RecipeArtifacts]:
    root = Path(run_dir).expanduser().resolve()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    phase_events_path = artifacts / "phase_events.parquet"
    if not phase_events_path.exists():
        kb: KnowledgeBundle = build_knowledge(root)
        phase_events_path = kb.phase_events_path
    phase_events = pd.read_parquet(phase_events_path) if phase_events_path.exists() else pd.DataFrame()

    mp_bundle = fetch_materials_project_refs(
        film_material=request.target_film_material,
        phase_target_type=request.target_phase_target.type,
        phase_target_value=request.target_phase_target.value,
        enabled=bool(materials_project_enabled and request.use_materials_project),
        api_key=str(materials_project_api_key or ""),
        endpoint=str(materials_project_endpoint or "https://api.materialsproject.org"),
        max_results=120,
        scope=str(mp_scope or "summary_thermo"),
    )
    mp_refs_path = artifacts / "materials_project_refs.json"
    mp_payload = {
        "status": mp_bundle.status,
        "error": mp_bundle.error,
        "query": mp_bundle.query,
        "reference_count": len(mp_bundle.references),
        "references": mp_bundle.references,
        "fetched_at": now_iso(),
    }
    mp_refs_path.write_text(json.dumps(mp_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rng = random.Random(17)
    all_candidates: list[pd.DataFrame] = []
    all_gate_audit: list[pd.DataFrame] = []
    all_safety: list[pd.DataFrame] = []
    best_rows = pd.DataFrame()
    best_phase = "none"
    gate_report: dict[str, Any] = {"counts": {"loose": 0, "balanced": 0, "strict": 0, "best": 0}}

    for attempt in range(int(request.max_retry_loops) + 1):
        generated = _build_candidate_rows(
            phase_events=phase_events,
            req=request,
            attempt=attempt,
            rng=rng,
            materials_project_refs=mp_bundle.references,
        )
        candidate_df = pd.DataFrame(generated)
        if candidate_df.empty:
            continue
        candidate_df = candidate_df.drop_duplicates(subset=["recipe_id"], keep="first").reset_index(drop=True)
        candidate_df = _apply_processor_scores(candidate_df, request, processor_model_path)
        filtered, safety_df = _apply_safety(candidate_df, request)
        ranked, gate_df, local_report, phase = _progressive_gate(filtered, top_k=request.top_k)
        all_candidates.append(candidate_df)
        all_gate_audit.append(gate_df)
        all_safety.append(safety_df)

        if int(local_report.get("counts", {}).get("best", 0)) > 0:
            best_rows = ranked.copy()
            best_phase = phase
            gate_report = local_report
            break

        gate_report = local_report
        if not ranked.empty:
            best_rows = ranked.copy()
            best_phase = phase

    candidates_df = (
        pd.concat(all_candidates, ignore_index=True).drop_duplicates(subset=["recipe_id"], keep="first")
        if all_candidates
        else pd.DataFrame()
    )
    gate_audit_df = pd.concat(all_gate_audit, ignore_index=True) if all_gate_audit else pd.DataFrame(
        columns=["recipe_id", "stage", "passed", "reason", "created_at"]
    )
    safety_df = pd.concat(all_safety, ignore_index=True) if all_safety else pd.DataFrame(
        columns=["recipe_id", "blocked", "reason", "created_at"]
    )
    ranked_df = best_rows.copy()
    if not ranked_df.empty and "rank" not in ranked_df.columns:
        ranked_df = ranked_df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
        ranked_df["rank"] = range(1, len(ranked_df) + 1)

    recipe_cards = _build_recipe_cards(ranked_df, request, mp_bundle.references)

    candidates_path = artifacts / "recipe_candidates.parquet"
    ranked_path = artifacts / "recipe_ranked.parquet"
    cards_path = artifacts / "recipe_cards.json"
    gate_audit_path = artifacts / "gate_audit.parquet"
    safety_audit_path = artifacts / "safety_audit.parquet"
    result_path = artifacts / "recipe_result.json"

    candidates_df.to_parquet(candidates_path, index=False)
    ranked_df.to_parquet(ranked_path, index=False)
    gate_audit_df.to_parquet(gate_audit_path, index=False)
    safety_df.to_parquet(safety_audit_path, index=False)
    cards_path.write_text(
        json.dumps([card.model_dump(mode="python") for card in recipe_cards], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    evidence_report = {
        "phase_event_count": int(len(phase_events)),
        "candidate_count": int(len(candidates_df)),
        "ranked_count": int(len(ranked_df)),
        "citations_covered": int(ranked_df["citation_url"].astype(str).str.strip().ne("").sum())
        if not ranked_df.empty and "citation_url" in ranked_df.columns
        else 0,
        "materials_project": {
            "status": mp_bundle.status,
            "reference_count": len(mp_bundle.references),
            "error": mp_bundle.error,
        },
        "processor_model_path": str(processor_model_path or ""),
    }
    result = RecipeGenerateResult(
        job_id="",
        status="SUCCEEDED",
        phase=best_phase,
        recipes=recipe_cards,
        gate_report=gate_report,
        evidence_report=evidence_report,
    )
    result_payload = result.model_dump(mode="python")
    result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts_info = RecipeArtifacts(
        candidates_path=candidates_path,
        ranked_path=ranked_path,
        cards_path=cards_path,
        gate_audit_path=gate_audit_path,
        safety_audit_path=safety_audit_path,
        materials_project_refs_path=mp_refs_path,
        result_path=result_path,
        recipe_count=int(len(recipe_cards)),
        phase=best_phase,
    )
    return result, artifacts_info
