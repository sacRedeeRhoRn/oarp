from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
)
from oarp.topic_spec import load_topic_spec

_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(s|sec|second|seconds|min|minute|minutes|h|hr|hour|hours)\b", re.IGNORECASE)
_PRESSURE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(pa|kpa|mpa|bar|mbar|torr)\b", re.IGNORECASE)
_STRAIN_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
_VACANCY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:cm-?3|cm\^-3)", re.IGNORECASE)


@dataclass
class KnowledgeBundle:
    phase_events_path: Path
    condition_graph_path: Path
    quality_outcomes_path: Path
    phase_event_count: int
    condition_edge_count: int
    quality_outcome_count: int


def _safe_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _parse_time_seconds(snippet: str) -> float | None:
    match = _TIME_RE.search(str(snippet or ""))
    if not match:
        return None
    value = float(match.group(1))
    unit = str(match.group(2) or "").lower()
    if unit.startswith("h"):
        return value * 3600.0
    if unit.startswith("min"):
        return value * 60.0
    return value


def _parse_pressure_pa(snippet: str) -> float | None:
    match = _PRESSURE_RE.search(str(snippet or ""))
    if not match:
        return None
    value = float(match.group(1))
    unit = str(match.group(2) or "").lower()
    if unit == "kpa":
        return value * 1_000.0
    if unit == "mpa":
        return value * 1_000_000.0
    if unit == "bar":
        return value * 100_000.0
    if unit == "mbar":
        return value * 100.0
    if unit == "torr":
        return value * 133.322
    return value


def _parse_strain_pct(snippet: str) -> float | None:
    lower = str(snippet or "").lower()
    if "strain" not in lower and "stressed" not in lower:
        return None
    match = _STRAIN_RE.search(str(snippet or ""))
    if not match:
        return None
    return float(match.group(1))


def _parse_vacancy_cm3(snippet: str) -> float | None:
    lower = str(snippet or "").lower()
    if "vacanc" not in lower:
        return None
    match = _VACANCY_RE.search(str(snippet or ""))
    if not match:
        return None
    return float(match.group(1))


def _quality_label(snippet: str) -> str:
    text = str(snippet or "").lower()
    if any(token in text for token in ["epitax", "single crystal", "highly ordered"]):
        return "high_crystallinity"
    if any(token in text for token in ["agglomer", "rough", "void", "defect"]):
        return "degraded"
    if any(token in text for token in ["uniform", "stable", "smooth"]):
        return "uniform"
    return "unknown"


def _method_family(snippet: str) -> str:
    text = str(snippet or "").lower()
    if any(token in text for token in ["sputter", "evaporat", "e-beam", "physical vapor"]):
        return "PVD"
    if any(token in text for token in ["atomic layer deposition", "ald"]):
        return "ALD"
    if any(token in text for token in ["cvd", "chemical vapor"]):
        return "CVD"
    if any(token in text for token in ["anneal", "rapid thermal"]):
        return "ANNEAL"
    return ""


def _first_composition_value(row: dict[str, Any]) -> float | None:
    for key in ("doping_composition", "alloy_composition"):
        items = _safe_json_list(row.get(key))
        if not items:
            continue
        first = items[0]
        if isinstance(first, dict) and "value" in first:
            try:
                return float(first["value"])
            except Exception:
                continue
    return None


def _point_pairs(validated: pd.DataFrame, x_name: str, y_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if validated.empty:
        return rows

    for point_id, group in validated.groupby("point_id"):
        x_rows = group[group["variable_name"] == x_name]
        y_rows = group[group["variable_name"] == y_name]
        if x_rows.empty or y_rows.empty:
            continue
        x = float(x_rows.iloc[0]["normalized_value"])
        y = float(y_rows.iloc[0]["normalized_value"])
        first = group.iloc[0].to_dict()
        row = {
            "point_id": str(point_id),
            "article_key": str(first.get("article_key") or ""),
            "provider": str(first.get("provider") or ""),
            "phase_label": str(first.get("entity") or ""),
            "thickness_nm": x,
            "anneal_temperature_c": y,
            "anneal_time_s": _parse_time_seconds(str(first.get("snippet") or "")),
            "pressure_pa": _parse_pressure_pa(str(first.get("snippet") or "")),
            "strain_pct": _parse_strain_pct(str(first.get("snippet") or "")),
            "vacancy_concentration_cm3": _parse_vacancy_cm3(str(first.get("snippet") or "")),
            "composition_value": _first_composition_value(first),
            "film_quality_score_numeric": float(group["confidence"].mean()),
            "substrate_material": str(first.get("substrate_material") or ""),
            "substrate_orientation": str(first.get("substrate_orientation") or ""),
            "doped_elements": str(first.get("doping_elements") or "[]"),
            "alloy_elements": str(first.get("alloy_elements") or "[]"),
            "precursor": "",
            "film_material": str(first.get("entity") or ""),
            "film_quality_label": _quality_label(str(first.get("snippet") or "")),
            "method_family": _method_family(str(first.get("snippet") or "")),
            "citation_url": str(first.get("citation_url") or ""),
            "doi": str(first.get("doi") or ""),
            "locator": str(first.get("locator") or ""),
            "snippet": str(first.get("snippet") or ""),
            "mp_status": str(first.get("mp_status") or ""),
            "mp_query_key": str(first.get("mp_query_key") or ""),
            "mp_material_ids": str(first.get("mp_material_ids") or "[]"),
            "mp_best_material_id": str(first.get("mp_best_material_id") or ""),
            "mp_formula_match_score": float(first.get("mp_formula_match_score") or 0.0),
            "mp_phase_match_score": float(first.get("mp_phase_match_score") or 0.0),
            "mp_stability_score": float(first.get("mp_stability_score") or 0.0),
            "mp_interpretation_score": float(first.get("mp_interpretation_score") or 0.0),
            "mp_interpretation_label": str(first.get("mp_interpretation_label") or ""),
            "mp_energy_above_hull_min": first.get("mp_energy_above_hull_min"),
            "mp_spacegroup_symbol": str(first.get("mp_spacegroup_symbol") or ""),
            "mp_spacegroup_number": str(first.get("mp_spacegroup_number") or ""),
            "mp_conflict_reason": str(first.get("mp_conflict_reason") or ""),
            "created_at": now_iso(),
        }
        row["event_id"] = hashlib.sha1(
            f"{row['point_id']}|{row['phase_label']}|{row['thickness_nm']}|{row['anneal_temperature_c']}".encode("utf-8")
        ).hexdigest()[:20]
        rows.append(row)
    return rows


def _value_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).strip()


def _build_condition_graph(phase_events: pd.DataFrame) -> pd.DataFrame:
    if phase_events.empty:
        return pd.DataFrame(
            columns=[
                "source_key",
                "target_key",
                "cooccurrence_count",
                "support_event_count",
                "created_at",
            ]
        )

    condition_cols = [
        "phase_label",
        "substrate_material",
        "substrate_orientation",
        "method_family",
        "thickness_nm",
        "anneal_temperature_c",
        "anneal_time_s",
        "pressure_pa",
        "strain_pct",
        "mp_interpretation_label",
    ]
    edge_counts: dict[tuple[str, str], int] = {}

    for row in phase_events.to_dict(orient="records"):
        nodes: list[str] = []
        for col in condition_cols:
            value = _value_token(row.get(col))
            if not value:
                continue
            nodes.append(f"{col}:{value}")
        nodes = sorted(set(nodes))
        for a, b in combinations(nodes, 2):
            key = (a, b)
            edge_counts[key] = edge_counts.get(key, 0) + 1

    rows = [
        {
            "source_key": source,
            "target_key": target,
            "cooccurrence_count": count,
            "support_event_count": count,
            "created_at": now_iso(),
        }
        for (source, target), count in sorted(edge_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return pd.DataFrame(rows)


def _build_quality_outcomes(phase_events: pd.DataFrame) -> pd.DataFrame:
    if phase_events.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "phase_label",
                "film_quality_score_numeric",
                "film_quality_label",
                "method_family",
                "citation_url",
                "doi",
                "created_at",
            ]
        )
    cols = [
        "event_id",
        "phase_label",
        "film_quality_score_numeric",
        "film_quality_label",
        "method_family",
        "mp_status",
        "mp_best_material_id",
        "mp_interpretation_score",
        "mp_interpretation_label",
        "mp_energy_above_hull_min",
        "citation_url",
        "doi",
    ]
    out = phase_events[cols].copy()
    out["created_at"] = now_iso()
    return out


def build_knowledge(run_dir: str | Path) -> KnowledgeBundle:
    layout = ensure_run_layout(run_dir)
    artifacts = layout["artifacts"]
    state = load_run_state(run_dir)
    spec_path = str(state.get("spec_path") or "")
    if not spec_path:
        raise ValueError("run state missing spec_path")
    spec = load_topic_spec(spec_path)

    validated_path = artifacts / "validated_points.parquet"
    if not validated_path.exists():
        raise FileNotFoundError(f"missing validated points: {validated_path}")
    validated = pd.read_parquet(validated_path)

    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y
    phase_rows = _point_pairs(validated, x_name=x_name, y_name=y_name)
    phase_events = pd.DataFrame(phase_rows)
    if phase_events.empty:
        phase_events = pd.DataFrame(
            columns=[
                "event_id",
                "point_id",
                "article_key",
                "provider",
                "phase_label",
                "thickness_nm",
                "anneal_temperature_c",
                "anneal_time_s",
                "pressure_pa",
                "strain_pct",
                "vacancy_concentration_cm3",
                "composition_value",
                "film_quality_score_numeric",
                "substrate_material",
                "substrate_orientation",
                "doped_elements",
                "alloy_elements",
                "precursor",
                "film_material",
                "film_quality_label",
                "method_family",
                "mp_status",
                "mp_query_key",
                "mp_material_ids",
                "mp_best_material_id",
                "mp_formula_match_score",
                "mp_phase_match_score",
                "mp_stability_score",
                "mp_interpretation_score",
                "mp_interpretation_label",
                "mp_energy_above_hull_min",
                "mp_spacegroup_symbol",
                "mp_spacegroup_number",
                "mp_conflict_reason",
                "citation_url",
                "doi",
                "locator",
                "snippet",
                "created_at",
            ]
        )

    condition_graph = _build_condition_graph(phase_events)
    quality_outcomes = _build_quality_outcomes(phase_events)

    phase_events_path = artifacts / "phase_events.parquet"
    condition_graph_path = artifacts / "condition_graph.parquet"
    quality_outcomes_path = artifacts / "quality_outcomes.parquet"

    phase_events.to_parquet(phase_events_path, index=False)
    condition_graph.to_parquet(condition_graph_path, index=False)
    quality_outcomes.to_parquet(quality_outcomes_path, index=False)

    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    run_id = str(state.get("run_id") or "")
    upsert_artifact(db_path=db_path, run_id=run_id, name="phase_events", path=phase_events_path)
    upsert_artifact(db_path=db_path, run_id=run_id, name="condition_graph", path=condition_graph_path)
    upsert_artifact(db_path=db_path, run_id=run_id, name="quality_outcomes", path=quality_outcomes_path)
    append_lineage(
        db_path=db_path,
        run_id=run_id,
        stage="knowledge",
        source_name="validated_points.parquet",
        target_name="phase_events.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=run_id,
        stage="knowledge",
        source_name="phase_events.parquet",
        target_name="condition_graph.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=run_id,
        stage="knowledge",
        source_name="phase_events.parquet",
        target_name="quality_outcomes.parquet",
    )

    return KnowledgeBundle(
        phase_events_path=phase_events_path,
        condition_graph_path=condition_graph_path,
        quality_outcomes_path=quality_outcomes_path,
        phase_event_count=int(len(phase_events)),
        condition_edge_count=int(len(condition_graph)),
        quality_outcome_count=int(len(quality_outcomes)),
    )
