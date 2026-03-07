from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.cache import prepare_feature_cache
from oarp.materials_project import fetch_materials_project_refs
from oarp.models import MPEvidenceSet, RunConfig
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
)
from oarp.topic_spec import TopicSpec

_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*){2,}$")
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA_TEXT_RE = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
_FORMULA_TEXT_LOOSE_RE = re.compile(r"\b(?:[A-Z][a-z]?\s*\d*\s*){2,4}\b")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_VALID_ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}


def _normalize_formula(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    text = text.translate(_SUBSCRIPTS)
    text = re.sub(r"[\s\-_/:]", "", text)
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    if not text:
        return ""
    if _FORMULA_RE.match(text):
        tokens = _FORMULA_TOKEN_RE.findall(text)
        if not tokens:
            return ""
        if any(element not in _VALID_ELEMENTS for element, _count in tokens):
            return ""
        return text
    return ""


def _extract_formula_tokens(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    raw = raw.translate(_SUBSCRIPTS)
    candidates: list[str] = []
    for pattern in (_FORMULA_TEXT_RE, _FORMULA_TEXT_LOOSE_RE):
        for match in pattern.finditer(raw):
            token = _normalize_formula(match.group(0))
            if token:
                candidates.append(token)
    ordered: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _formula_elements(formula: str) -> set[str]:
    out: set[str] = set()
    for element, _count in _FORMULA_TOKEN_RE.findall(formula):
        if element:
            out.add(element)
    return out


def _normalize_spacegroup(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]", "", text)


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


@dataclass
class MaterialIdentity:
    point_id: str
    entity: str
    formula: str
    phase_target_type: str
    phase_target_value: str
    doping_signature: str
    query_key: str


class MaterialIdentityResolver:
    def __init__(self, spec: TopicSpec, cfg: RunConfig):
        self.spec = spec
        self.alias_map = spec.entity_alias_map()
        for key, value in (cfg.entity_alias_overrides or {}).items():
            k = str(key or "").strip().lower()
            v = str(value or "").strip()
            if k and v:
                self.alias_map[k] = v

    def _resolve_formula_from_entity(self, entity: str) -> str:
        clean = _normalize_formula(entity)
        if clean:
            return clean
        low = str(entity or "").strip().lower()
        if low in {"", "unknown", "na", "n/a", "none", "null"}:
            return ""
        canonical = self.alias_map.get(low, "")
        if canonical:
            clean = _normalize_formula(canonical)
            if clean:
                return clean
        for alias, canonical_name in self.alias_map.items():
            if alias and alias in low:
                clean = _normalize_formula(canonical_name)
                if clean:
                    return clean
        return ""

    def _resolve_formula_from_row(self, first: dict[str, Any]) -> tuple[str, str]:
        # Prefer direct entity/label fields before scanning full snippets.
        for key in ("entity", "phase_label", "film_material", "target_film_material"):
            raw = str(first.get(key) or "").strip()
            formula = self._resolve_formula_from_entity(raw)
            if formula:
                return formula, raw
        snippet = str(first.get("snippet") or "")
        for token in _extract_formula_tokens(snippet):
            formula = self._resolve_formula_from_entity(token) or _normalize_formula(token)
            if formula:
                return formula, token
        return "", ""

    def resolve_point(self, point_id: str, group: pd.DataFrame, scope: str) -> MaterialIdentity:
        first = group.iloc[0].to_dict()
        entity = str(first.get("entity") or "").strip()
        formula, source = self._resolve_formula_from_row(first)
        phase_target_type = "phase_label"
        phase_target_value = (
            formula
            or entity
            or str(first.get("phase_label") or "").strip()
            or str(first.get("film_material") or "").strip()
            or source
        )

        doping_parts: list[str] = []
        for key in ("doping_state", "alloy_state"):
            val = str(first.get(key) or "").strip().lower()
            if val:
                doping_parts.append(f"{key}:{val}")
        for key in ("doping_elements", "alloy_elements"):
            elems = [str(item).strip() for item in _json_list(first.get(key)) if str(item).strip()]
            if elems:
                doping_parts.append(f"{key}:{','.join(sorted(set(elems)))}")

        doping_signature = "|".join(doping_parts)
        query_key = hashlib.sha1(
            f"{formula}|{phase_target_type}|{phase_target_value}|{doping_signature}|{scope}".encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()
        return MaterialIdentity(
            point_id=str(point_id),
            entity=entity,
            formula=formula,
            phase_target_type=phase_target_type,
            phase_target_value=phase_target_value,
            doping_signature=doping_signature,
            query_key=query_key,
        )


def _init_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mp_cache (
                query_key TEXT PRIMARY KEY,
                formula TEXT NOT NULL,
                phase_target_type TEXT NOT NULL,
                phase_target_value TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT NOT NULL,
                references_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _cache_get(path: Path, query_key: str) -> dict[str, Any] | None:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            """
            SELECT formula, phase_target_type, phase_target_value, status, error, references_json, updated_at
            FROM mp_cache WHERE query_key = ?
            """,
            (query_key,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    refs: list[dict[str, Any]] = []
    try:
        parsed = json.loads(str(row[5]))
        if isinstance(parsed, list):
            refs = [item for item in parsed if isinstance(item, dict)]
    except Exception:
        refs = []
    return {
        "formula": str(row[0] or ""),
        "phase_target_type": str(row[1] or ""),
        "phase_target_value": str(row[2] or ""),
        "status": str(row[3] or ""),
        "error": str(row[4] or ""),
        "references": refs,
        "updated_at": str(row[6] or ""),
    }


def _cache_put(path: Path, identity: MaterialIdentity, status: str, error: str, refs: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO mp_cache(
                query_key, formula, phase_target_type, phase_target_value,
                status, error, references_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query_key) DO UPDATE SET
                formula = excluded.formula,
                phase_target_type = excluded.phase_target_type,
                phase_target_value = excluded.phase_target_value,
                status = excluded.status,
                error = excluded.error,
                references_json = excluded.references_json,
                updated_at = excluded.updated_at
            """,
            (
                identity.query_key,
                identity.formula,
                identity.phase_target_type,
                identity.phase_target_value,
                status,
                error,
                json.dumps(refs, sort_keys=True),
                now_iso(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _score_formula_match(identity_formula: str, ref_formula: str) -> float:
    left = _normalize_formula(identity_formula)
    right = _normalize_formula(ref_formula)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_elements = _formula_elements(left)
    right_elements = _formula_elements(right)
    if left_elements and left_elements == right_elements:
        return 0.85
    if left_elements and right_elements and left_elements.intersection(right_elements):
        return 0.55
    return 0.0


def _score_phase_match(identity: MaterialIdentity, ref: dict[str, Any], formula_score: float) -> float:
    target_type = str(identity.phase_target_type or "").strip().lower()
    target_value = str(identity.phase_target_value or "").strip()
    if not target_type:
        return 0.0
    if target_type == "space_group":
        left = _normalize_spacegroup(target_value)
        right_symbol = _normalize_spacegroup(ref.get("spacegroup_symbol"))
        right_number = _normalize_spacegroup(ref.get("spacegroup_number"))
        if left and (left == right_symbol or left == right_number):
            return 1.0
        if left and (left in right_symbol or right_symbol in left):
            return 0.6
        return 0.0
    if formula_score >= 1.0:
        return 1.0
    if formula_score > 0.0:
        return 0.6
    return 0.0


def _score_stability(ref: dict[str, Any]) -> tuple[float, float | None]:
    ehull = _safe_float(ref.get("energy_above_hull"))
    if ehull is None:
        return 0.0, None
    if ehull <= 0.05:
        return 1.0, ehull
    if ehull <= 0.10:
        return 0.7, ehull
    return 0.4, ehull


def _interpretation_label(score: float, formula_score: float, has_refs: bool) -> str:
    if not has_refs:
        return "insufficient"
    if score >= 0.70:
        return "supports"
    if score <= 0.35 and formula_score < 0.40:
        return "conflicts"
    return "neutral"


def _conflict_reason(label: str, formula_score: float, stability_score: float) -> str:
    if label != "conflicts":
        return ""
    if formula_score < 0.40:
        return "formula_mismatch"
    if stability_score <= 0.40:
        return "low_stability"
    return "low_alignment"


def _fetch_with_retry(identity: MaterialIdentity, cfg: RunConfig, api_key: str) -> tuple[str, str, list[dict[str, Any]], int]:
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        bundle = fetch_materials_project_refs(
            film_material=identity.formula,
            phase_target_type=identity.phase_target_type,
            phase_target_value=identity.phase_target_value,
            enabled=bool(cfg.mp_enabled),
            api_key=api_key,
            endpoint=str(os.getenv("MP_ENDPOINT", "https://api.materialsproject.org")),
            timeout_sec=float(cfg.mp_timeout_sec),
            max_results=120,
            scope=cfg.mp_scope,
        )
        status = str(bundle.status or "")
        if status != "fetch_error":
            return status, str(bundle.error or ""), list(bundle.references or []), attempt
        if attempt < max_attempts:
            time.sleep((cfg.backoff_sec or 1.5) * (1.5 ** (attempt - 1)))
    return status, str(bundle.error or ""), list(bundle.references or []), max_attempts


def enrich(*, spec: TopicSpec, cfg: RunConfig) -> MPEvidenceSet:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]
    state = load_run_state(cfg.as_path())
    run_id = str(state.get("run_id") or "")
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)

    evidence_path = artifacts / "evidence_points.parquet"
    if not evidence_path.exists():
        raise FileNotFoundError(f"missing evidence points: {evidence_path}")
    evidence = pd.read_parquet(evidence_path)

    cache_path = Path(cfg.mp_cache_path).expanduser().resolve() if str(cfg.mp_cache_path).strip() else artifacts / "materials_project_cache.sqlite"
    _init_cache(cache_path)
    api_key = str(os.getenv("MP_API_KEY", "")).strip()
    feature_cache = prepare_feature_cache(cfg)

    resolver = MaterialIdentityResolver(spec, cfg)
    identities: dict[str, MaterialIdentity] = {}
    for point_id, group in evidence.groupby("point_id"):
        identity = resolver.resolve_point(str(point_id), group, str(cfg.mp_scope))
        identities[str(point_id)] = identity

    point_rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    query_results: dict[str, dict[str, Any]] = {}

    issued_queries = 0
    for identity in sorted(identities.values(), key=lambda item: item.query_key):
        if identity.query_key in query_results:
            continue
        feature_key = feature_cache.make_key("mp_query", identity.query_key, str(cfg.mp_scope))

        if not identity.formula:
            query_results[identity.query_key] = {
                "status": "no_match",
                "error": "formula_not_resolved",
                "references": [],
                "cache_hit": False,
                "attempts": 0,
                "latency_ms": 0.0,
            }
            query_rows.append(
                {
                    "query_key": identity.query_key,
                    "formula": identity.formula,
                    "phase_target_type": identity.phase_target_type,
                    "phase_target_value": identity.phase_target_value,
                    "doping_signature": identity.doping_signature,
                    "status": "no_match",
                    "error": "formula_not_resolved",
                    "cache_hit": False,
                    "latency_ms": 0.0,
                    "reference_count": 0,
                    "attempts": 0,
                    "created_at": now_iso(),
                }
            )
            continue

        start = time.perf_counter()
        feature_cached = feature_cache.get_json("mp_query", feature_key, ttl_hours=int(cfg.cache_ttl_hours))
        cached = None
        if isinstance(feature_cached, dict):
            refs_payload = feature_cached.get("references")
            cached = {
                "status": str(feature_cached.get("status") or ""),
                "error": str(feature_cached.get("error") or ""),
                "references": refs_payload if isinstance(refs_payload, list) else [],
            }
        if cached is None and cfg.mp_on_demand:
            cached = _cache_get(cache_path, identity.query_key)
        if cached is not None:
            status = str(cached["status"])
            error = str(cached["error"])
            refs = list(cached["references"])
            attempts = 0
            cache_hit = True
        else:
            cache_hit = False
            if issued_queries >= int(cfg.mp_max_queries):
                status, error, refs, attempts = "fetch_error", "max_queries_exceeded", [], 0
            elif not cfg.mp_enabled:
                status, error, refs, attempts = "disabled", "", [], 0
            elif not api_key:
                status, error, refs, attempts = "no_api_key", "", [], 0
            else:
                status, error, refs, attempts = _fetch_with_retry(identity, cfg, api_key)
            _cache_put(cache_path, identity, status, error, refs)
            feature_cache.put_json(
                "mp_query",
                feature_key,
                {
                    "status": status,
                    "error": error,
                    "references": refs,
                },
                key_text=identity.query_key,
                ttl_hours=int(cfg.cache_ttl_hours),
            )
            issued_queries += 1

        latency_ms = (time.perf_counter() - start) * 1000.0
        query_results[identity.query_key] = {
            "status": status,
            "error": error,
            "references": refs,
            "cache_hit": cache_hit,
            "attempts": attempts,
            "latency_ms": latency_ms,
        }
        query_rows.append(
            {
                "query_key": identity.query_key,
                "formula": identity.formula,
                "phase_target_type": identity.phase_target_type,
                "phase_target_value": identity.phase_target_value,
                "doping_signature": identity.doping_signature,
                "status": status,
                "error": error,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "reference_count": len(refs),
                "attempts": attempts,
                "created_at": now_iso(),
            }
        )

    for point_id, group in evidence.groupby("point_id"):
        identity = identities[str(point_id)]
        result = query_results.get(identity.query_key, {"status": "no_match", "error": "query_result_missing", "references": []})
        refs = list(result.get("references") or [])

        scored_refs: list[dict[str, Any]] = []
        for ref in refs:
            formula_score = _score_formula_match(identity.formula, str(ref.get("formula_pretty") or ""))
            phase_score = _score_phase_match(identity, ref, formula_score)
            stability_score, ehull = _score_stability(ref)
            interp_score = (
                float(cfg.mp_formula_match_weight) * formula_score
                + float(cfg.mp_phase_match_weight) * phase_score
                + float(cfg.mp_stability_weight) * stability_score
            )
            interp_score = max(0.0, min(1.0, interp_score))
            label = _interpretation_label(interp_score, formula_score, has_refs=True)
            scored_refs.append(
                {
                    "material_id": str(ref.get("material_id") or ""),
                    "formula_pretty": str(ref.get("formula_pretty") or ""),
                    "spacegroup_symbol": str(ref.get("spacegroup_symbol") or ""),
                    "spacegroup_number": str(ref.get("spacegroup_number") or ""),
                    "crystal_system": str(ref.get("crystal_system") or ""),
                    "energy_above_hull": _safe_float(ref.get("energy_above_hull")),
                    "is_theoretical": bool(ref.get("is_theoretical") or False),
                    "mp_formula_match_score": formula_score,
                    "mp_phase_match_score": phase_score,
                    "mp_stability_score": stability_score,
                    "mp_interpretation_score": interp_score,
                    "mp_interpretation_label": label,
                    "mp_conflict_reason": _conflict_reason(label, formula_score, stability_score),
                }
            )

        scored_refs = sorted(
            scored_refs,
            key=lambda item: (
                -float(item.get("mp_interpretation_score") or 0.0),
                -float(item.get("mp_formula_match_score") or 0.0),
                str(item.get("material_id") or ""),
            ),
        )
        best = scored_refs[0] if scored_refs else {}

        status = str(result.get("status") or "")
        if status == "success" and not scored_refs:
            status = "no_match"

        if not scored_refs:
            label = "insufficient"
            formula_score = 0.0
            phase_score = 0.0
            stability_score = 0.0
            interp_score = 0.0
            conflict_reason = ""
            best_mid = ""
            best_symbol = ""
            best_number = ""
            best_ehull = None
        else:
            label = str(best.get("mp_interpretation_label") or "insufficient")
            formula_score = float(best.get("mp_formula_match_score") or 0.0)
            phase_score = float(best.get("mp_phase_match_score") or 0.0)
            stability_score = float(best.get("mp_stability_score") or 0.0)
            interp_score = float(best.get("mp_interpretation_score") or 0.0)
            conflict_reason = str(best.get("mp_conflict_reason") or "")
            best_mid = str(best.get("material_id") or "")
            best_symbol = str(best.get("spacegroup_symbol") or "")
            best_number = str(best.get("spacegroup_number") or "")
            best_ehull = _safe_float(best.get("energy_above_hull"))

        mp_material_ids = [str(item.get("material_id") or "") for item in scored_refs if str(item.get("material_id") or "")]
        for row in group.to_dict(orient="records"):
            row_out = dict(row)
            row_out["mp_status"] = status
            row_out["mp_query_key"] = identity.query_key
            row_out["mp_material_ids"] = json.dumps(mp_material_ids, sort_keys=True)
            row_out["mp_best_material_id"] = best_mid
            row_out["mp_formula_match_score"] = formula_score
            row_out["mp_phase_match_score"] = phase_score
            row_out["mp_stability_score"] = stability_score
            row_out["mp_interpretation_score"] = interp_score
            row_out["mp_interpretation_label"] = label
            row_out["mp_energy_above_hull_min"] = best_ehull
            row_out["mp_spacegroup_symbol"] = best_symbol
            row_out["mp_spacegroup_number"] = best_number
            row_out["mp_conflict_reason"] = conflict_reason
            point_rows.append(row_out)

        for rank, item in enumerate(scored_refs, start=1):
            link_rows.append(
                {
                    "point_id": str(point_id),
                    "query_key": identity.query_key,
                    "material_id": str(item.get("material_id") or ""),
                    "rank": rank,
                    "formula_pretty": str(item.get("formula_pretty") or ""),
                    "spacegroup_symbol": str(item.get("spacegroup_symbol") or ""),
                    "spacegroup_number": str(item.get("spacegroup_number") or ""),
                    "energy_above_hull": _safe_float(item.get("energy_above_hull")),
                    "mp_formula_match_score": float(item.get("mp_formula_match_score") or 0.0),
                    "mp_phase_match_score": float(item.get("mp_phase_match_score") or 0.0),
                    "mp_stability_score": float(item.get("mp_stability_score") or 0.0),
                    "mp_interpretation_score": float(item.get("mp_interpretation_score") or 0.0),
                    "mp_interpretation_label": str(item.get("mp_interpretation_label") or ""),
                    "mp_conflict_reason": str(item.get("mp_conflict_reason") or ""),
                    "created_at": now_iso(),
                }
            )
            material_rows.append(
                {
                    "query_key": identity.query_key,
                    "material_id": str(item.get("material_id") or ""),
                    "formula_pretty": str(item.get("formula_pretty") or ""),
                    "spacegroup_symbol": str(item.get("spacegroup_symbol") or ""),
                    "spacegroup_number": str(item.get("spacegroup_number") or ""),
                    "crystal_system": str(item.get("crystal_system") or ""),
                    "energy_above_hull": _safe_float(item.get("energy_above_hull")),
                    "is_theoretical": bool(item.get("is_theoretical") or False),
                    "created_at": now_iso(),
                }
            )

        if label == "conflicts":
            conflict_rows.append(
                {
                    "point_id": str(point_id),
                    "query_key": identity.query_key,
                    "entity": identity.entity,
                    "mp_best_material_id": best_mid,
                    "mp_interpretation_score": interp_score,
                    "mp_conflict_reason": conflict_reason,
                    "created_at": now_iso(),
                }
            )

    enriched_df = pd.DataFrame(point_rows) if point_rows else evidence.copy()
    materials_df = pd.DataFrame(material_rows).drop_duplicates(subset=["query_key", "material_id"], keep="first") if material_rows else pd.DataFrame(
        columns=[
            "query_key",
            "material_id",
            "formula_pretty",
            "spacegroup_symbol",
            "spacegroup_number",
            "crystal_system",
            "energy_above_hull",
            "is_theoretical",
            "created_at",
        ]
    )
    links_df = pd.DataFrame(link_rows) if link_rows else pd.DataFrame(
        columns=[
            "point_id",
            "query_key",
            "material_id",
            "rank",
            "formula_pretty",
            "spacegroup_symbol",
            "spacegroup_number",
            "energy_above_hull",
            "mp_formula_match_score",
            "mp_phase_match_score",
            "mp_stability_score",
            "mp_interpretation_score",
            "mp_interpretation_label",
            "mp_conflict_reason",
            "created_at",
        ]
    )
    query_df = pd.DataFrame(query_rows) if query_rows else pd.DataFrame(
        columns=[
            "query_key",
            "formula",
            "phase_target_type",
            "phase_target_value",
            "doping_signature",
            "status",
            "error",
            "cache_hit",
            "latency_ms",
            "reference_count",
            "attempts",
            "created_at",
        ]
    )
    conflicts_df = pd.DataFrame(conflict_rows) if conflict_rows else pd.DataFrame(
        columns=[
            "point_id",
            "query_key",
            "entity",
            "mp_best_material_id",
            "mp_interpretation_score",
            "mp_conflict_reason",
            "created_at",
        ]
    )

    enriched_path = artifacts / "materials_project_enriched_points.parquet"
    materials_path = artifacts / "materials_project_materials.parquet"
    links_path = artifacts / "materials_project_point_links.parquet"
    query_log_path = artifacts / "materials_project_query_log.parquet"
    conflicts_path = artifacts / "materials_project_conflicts.parquet"

    enriched_df.to_parquet(enriched_path, index=False)
    materials_df.to_parquet(materials_path, index=False)
    links_df.to_parquet(links_path, index=False)
    query_df.to_parquet(query_log_path, index=False)
    conflicts_df.to_parquet(conflicts_path, index=False)
    feature_cache.write_audit(artifacts / "cache_audit.parquet")

    if run_id:
        upsert_artifact(db_path=db_path, run_id=run_id, name="materials_project_enriched_points", path=enriched_path)
        upsert_artifact(db_path=db_path, run_id=run_id, name="materials_project_materials", path=materials_path)
        upsert_artifact(db_path=db_path, run_id=run_id, name="materials_project_point_links", path=links_path)
        upsert_artifact(db_path=db_path, run_id=run_id, name="materials_project_query_log", path=query_log_path)
        upsert_artifact(db_path=db_path, run_id=run_id, name="materials_project_conflicts", path=conflicts_path)
        append_lineage(
            db_path=db_path,
            run_id=run_id,
            stage="mp-enrich",
            source_name="evidence_points.parquet",
            target_name="materials_project_enriched_points.parquet",
        )

    return MPEvidenceSet(
        enriched_points=enriched_df,
        materials=materials_df,
        point_links=links_df,
        query_log=query_df,
        enriched_points_path=enriched_path,
        materials_path=materials_path,
        point_links_path=links_path,
        query_log_path=query_log_path,
    )
