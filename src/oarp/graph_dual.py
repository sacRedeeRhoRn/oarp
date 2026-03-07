from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oarp.models import (
    ArticleGraphBuildResult,
    BridgeBuildResult,
    ConceptGraphBuildResult,
    GraphAuditResult,
    RunConfig,
)
from oarp.runtime import now_iso, write_json

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
_SPACEGROUP_CLEAN_RE = re.compile(r"[^A-Za-z0-9]")
_KEY_CLEAN_RE = re.compile(r"[^a-z0-9]+")

_DEFAULT_CONCEPT_GATES = {"film", "substrate", "dopant", "precursor", "method", "thermo", "electronic"}
_DETERMINISTIC_COMPONENT_WEIGHTS = {
    "extraction_conf": 0.35,
    "provenance_quality": 0.20,
    "context_proximity": 0.15,
    "cross_source_agreement": 0.15,
    "mp_alignment": 0.15,
}
_PERIODIC_TABLE_SYMBOLS = [
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
]
_GLOBAL_METHOD_FAMILIES = [
    "PVD",
    "CVD",
    "ALD",
    "MBE",
    "PLD",
    "SPUTTER",
    "EVAPORATION",
    "ANNEAL",
    "RTA",
    "FURNACE_ANNEAL",
    "PECVD",
    "LPCVD",
    "ELECTRODEPOSITION",
    "SOL_GEL",
    "ATOMIC_DIFFUSION",
    "unknown_method",
]
_GLOBAL_ORIENTATION_FAMILIES = [
    "(001)",
    "(100)",
    "(110)",
    "(111)",
    "(112)",
    "(113)",
    "<100>",
    "<110>",
    "<111>",
    "[001]",
    "[110]",
    "[111]",
    "c-plane",
    "a-plane",
    "m-plane",
    "r-plane",
    "unknown_orientation",
]
_GLOBAL_AMBIENT_GASES = [
    "N2",
    "Ar",
    "H2",
    "O2",
    "NH3",
    "forming_gas",
    "vacuum",
    "air",
    "inert",
    "unknown_ambient",
]
_GLOBAL_CRYSTAL_SYSTEMS = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "trigonal",
    "hexagonal",
    "cubic",
    "unknown_crystal_system",
]
_GLOBAL_TEMPERATURE_REGIMES = [
    "unknown_temperature_regime",
    "temp_regime_low",
    "temp_regime_mid",
    "temp_regime_high",
    "temp_regime_extreme",
]
_GLOBAL_TIME_REGIMES = [
    "unknown_time_regime",
    "time_regime_short",
    "time_regime_medium",
    "time_regime_long",
    "time_regime_very_long",
]
_GLOBAL_PRESSURE_REGIMES = [
    "unknown_pressure_regime",
    "pressure_regime_high_vacuum",
    "pressure_regime_low_pressure",
    "pressure_regime_mid_pressure",
    "pressure_regime_atm_like",
]
_GLOBAL_STABILITY_BANDS = [
    "stability_unknown",
    "stability_stable",
    "stability_metastable",
    "stability_unstable",
]
_GLOBAL_ENERGY_HULL_BUCKETS = [
    "energy_hull_unknown",
    "energy_hull_0_0p05",
    "energy_hull_0p05_0p10",
    "energy_hull_0p10_0p30",
    "energy_hull_gt_0p30",
]
_GLOBAL_BANDGAP_BUCKETS = [
    "bandgap_unknown",
    "bandgap_metallic",
    "bandgap_narrow",
    "bandgap_mid",
    "bandgap_wide",
]
_GLOBAL_METALLICITY_CLASSES = ["metallicity_unknown", "metallic", "non_metallic"]
_GLOBAL_MAGNETIC_CLASSES = ["magnetic_unknown", "ferromagnetic", "antiferromagnetic", "paramagnetic", "nonmagnetic"]


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if np.isnan(out) or np.isinf(out):
        return float(default)
    return float(out)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _as_text(value)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    return []


def _normalize_formula(raw: Any) -> str:
    text = _as_text(raw).replace(" ", "").replace("-", "")
    if not text:
        return ""
    tokens = _FORMULA_TOKEN_RE.findall(text)
    if not tokens:
        return ""
    rebuilt = "".join(f"{el}{count}" for el, count in tokens)
    compact = re.sub(r"[^A-Za-z0-9.]", "", text)
    if rebuilt.lower() != compact.lower():
        return ""
    return rebuilt


def _formula_components(formula: str) -> tuple[list[str], list[dict[str, Any]]]:
    elements: list[str] = []
    stoich: list[dict[str, Any]] = []
    for elem, count_text in _FORMULA_TOKEN_RE.findall(_as_text(formula)):
        if elem not in elements:
            elements.append(elem)
        ratio = _safe_float(count_text or 1.0, 1.0)
        if ratio <= 0.0:
            ratio = 1.0
        stoich.append({"element": elem, "ratio": float(ratio)})
    return elements, stoich


def _normalize_spacegroup_symbol(value: Any) -> str:
    text = _as_text(value)
    if not text:
        return ""
    return _SPACEGROUP_CLEAN_RE.sub("", text)


def _normalize_spacegroup_number(value: Any) -> str:
    text = _as_text(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits


def _normalize_key(value: Any) -> str:
    text = _as_text(value).lower()
    text = _KEY_CLEAN_RE.sub("_", text).strip("_")
    return text or "unknown"


def _stable_id(*parts: Any, length: int = 24) -> str:
    digest = hashlib.sha1("|".join(str(item) for item in parts).encode("utf-8", errors="replace")).hexdigest()
    return digest[: int(length)]


def _bucket_temperature(value: float) -> str:
    if value <= 0:
        return "unknown_temperature_regime"
    if value < 200:
        return "temp_regime_low"
    if value < 450:
        return "temp_regime_mid"
    if value < 750:
        return "temp_regime_high"
    return "temp_regime_extreme"


def _bucket_time(value: float) -> str:
    if value <= 0:
        return "unknown_time_regime"
    if value < 30:
        return "time_regime_short"
    if value < 600:
        return "time_regime_medium"
    if value < 3600:
        return "time_regime_long"
    return "time_regime_very_long"


def _bucket_pressure(value: float) -> str:
    if value <= 0:
        return "unknown_pressure_regime"
    if value < 100:
        return "pressure_regime_high_vacuum"
    if value < 1e4:
        return "pressure_regime_low_pressure"
    if value < 5e4:
        return "pressure_regime_mid_pressure"
    return "pressure_regime_atm_like"


def _bucket_stability(energy_above_hull: float) -> str:
    if energy_above_hull <= 0.0:
        return "stability_unknown"
    if energy_above_hull <= 0.05:
        return "stability_stable"
    if energy_above_hull <= 0.10:
        return "stability_metastable"
    return "stability_unstable"


def _bucket_energy_hull(energy_above_hull: float) -> str:
    if energy_above_hull <= 0.0:
        return "energy_hull_unknown"
    if energy_above_hull <= 0.05:
        return "energy_hull_0_0p05"
    if energy_above_hull <= 0.10:
        return "energy_hull_0p05_0p10"
    if energy_above_hull <= 0.30:
        return "energy_hull_0p10_0p30"
    return "energy_hull_gt_0p30"


def _bucket_bandgap(value: float) -> str:
    if value < 0:
        return "bandgap_unknown"
    if value == 0:
        return "bandgap_metallic"
    if value < 1.0:
        return "bandgap_narrow"
    if value < 3.0:
        return "bandgap_mid"
    return "bandgap_wide"


def _spacegroup_to_crystal_system(spacegroup_number: int) -> str:
    if spacegroup_number <= 2:
        return "triclinic"
    if spacegroup_number <= 15:
        return "monoclinic"
    if spacegroup_number <= 74:
        return "orthorhombic"
    if spacegroup_number <= 142:
        return "tetragonal"
    if spacegroup_number <= 167:
        return "trigonal"
    if spacegroup_number <= 194:
        return "hexagonal"
    return "cubic"


def _concept_gate_set(cfg: RunConfig) -> set[str]:
    gates = {str(item).strip().lower() for item in list(cfg.concept_gates or []) if str(item).strip()}
    if not gates:
        return set(_DEFAULT_CONCEPT_GATES)
    normalized = {item.replace("_gate", "") for item in gates}
    return normalized.intersection(_DEFAULT_CONCEPT_GATES) or set(_DEFAULT_CONCEPT_GATES)


def score_bridge_weight(edge_features: dict[str, Any], cfg: RunConfig) -> float:
    _ = cfg
    comp = {
        "extraction_conf": max(0.0, min(1.0, _safe_float(edge_features.get("extraction_conf"), 0.0))),
        "provenance_quality": max(0.0, min(1.0, _safe_float(edge_features.get("provenance_quality"), 0.0))),
        "context_proximity": max(0.0, min(1.0, _safe_float(edge_features.get("context_proximity"), 0.0))),
        "cross_source_agreement": max(0.0, min(1.0, _safe_float(edge_features.get("cross_source_agreement"), 0.0))),
        "mp_alignment": max(0.0, min(1.0, _safe_float(edge_features.get("mp_alignment"), 0.0))),
    }
    score = float(
        sum(_DETERMINISTIC_COMPONENT_WEIGHTS[key] * comp[key] for key in _DETERMINISTIC_COMPONENT_WEIGHTS.keys())
    )
    return max(0.0, min(1.0, score))


def _provenance_quality(row: dict[str, Any]) -> float:
    required = ["citation_url", "snippet", "locator"]
    filled = sum(1 for col in required if _as_text(row.get(col)))
    return float(filled / len(required))


def _mp_alignment_score(row: dict[str, Any]) -> float:
    label = _as_text(row.get("mp_interpretation_label")).lower()
    if label == "supports":
        return 1.0
    if label == "neutral":
        return 0.6
    if label == "conflicts":
        return 0.2
    if label == "insufficient":
        return 0.4
    return 0.5


def _context_proximity_score(gate_type: str) -> float:
    gate = gate_type.replace("_gate", "")
    if gate == "film":
        return 0.95
    if gate in {"substrate", "dopant", "method"}:
        return 0.85
    if gate in {"precursor", "thermo", "electronic"}:
        return 0.75
    return 0.70


def _cross_source_map(points: pd.DataFrame) -> dict[str, float]:
    if points.empty:
        return {}
    concept_support: dict[str, set[str]] = {}
    for row in points.to_dict(orient="records"):
        article_key = _as_text(row.get("article_key")) or "unknown_article"
        formula = _normalize_formula(row.get("film_material") or row.get("phase_label") or row.get("entity"))
        if formula:
            concept_support.setdefault(f"formula:{_normalize_key(formula)}", set()).add(article_key)
        elements, _stoich = _formula_components(formula)
        for elem in elements:
            concept_support.setdefault(f"element:{_normalize_key(elem)}", set()).add(article_key)
        substrate = _as_text(row.get("substrate_material"))
        if substrate:
            concept_support.setdefault(f"substrate_material:{_normalize_key(substrate)}", set()).add(article_key)
    out: dict[str, float] = {}
    for key, support in concept_support.items():
        out[key] = max(0.0, min(1.0, len(support) / 3.0))
    return out


def build_article_process_graph(points_df: pd.DataFrame, cfg: RunConfig) -> ArticleGraphBuildResult:
    artifacts = cfg.as_path() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    nodes_path = artifacts / "article_process_nodes.parquet"
    edges_path = artifacts / "article_process_edges.parquet"

    points = points_df.copy()
    if points.empty:
        pd.DataFrame(
            columns=[
                "node_id",
                "node_type",
                "article_key",
                "stage",
                "state_id",
                "label",
                "attributes_json",
                "created_at",
            ]
        ).to_parquet(nodes_path, index=False)
        pd.DataFrame(
            columns=[
                "edge_id",
                "article_key",
                "source_id",
                "target_id",
                "edge_type",
                "evidence_weight",
                "provenance_ref",
                "source_point_id",
                "created_at",
            ]
        ).to_parquet(edges_path, index=False)
        return ArticleGraphBuildResult(nodes_path=nodes_path, edges_path=edges_path, node_count=0, edge_count=0)

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(node_id: str, node_type: str, article_key: str, stage: str, state_id: str, label: str, attrs: dict[str, Any]) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        node_rows.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "article_key": article_key,
                "stage": stage,
                "state_id": state_id,
                "label": label,
                "attributes_json": _json_dump(attrs),
                "created_at": now_iso(),
            }
        )

    def add_edge(
        article_key: str,
        source_id: str,
        target_id: str,
        edge_type: str,
        source_point_id: str,
        provenance_ref: str,
        evidence_weight: float,
    ) -> None:
        edge_id = f"edge_{_stable_id(article_key, source_id, target_id, edge_type, source_point_id, length=20)}"
        if edge_id in seen_edges:
            return
        seen_edges.add(edge_id)
        edge_rows.append(
            {
                "edge_id": edge_id,
                "article_key": article_key,
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "evidence_weight": max(0.0, min(1.0, _safe_float(evidence_weight, 1.0))),
                "provenance_ref": provenance_ref,
                "source_point_id": source_point_id,
                "created_at": now_iso(),
            }
        )

    points["article_key"] = points.get("article_key", pd.Series(["unknown_article"] * len(points))).astype(str)
    points["point_id"] = points.get("point_id", pd.Series([""] * len(points))).astype(str)
    points["phase_label"] = points.get("phase_label", points.get("entity", "")).astype(str)
    points["film_material"] = points.get("film_material", points["phase_label"]).astype(str)

    for row in points.to_dict(orient="records"):
        article_key = _as_text(row.get("article_key")) or "unknown_article"
        point_id = _as_text(row.get("point_id")) or f"{article_key}:{_stable_id(now_iso(), article_key, length=8)}"
        phase_label = _as_text(row.get("phase_label")) or _as_text(row.get("entity")) or "unknown_phase"
        film_material = _as_text(row.get("film_material")) or phase_label
        substrate = _as_text(row.get("substrate_material")) or "unknown_substrate"
        orientation = _as_text(row.get("substrate_orientation")) or "unknown_orientation"
        method_family = _as_text(row.get("method_family")) or "unknown_method"
        precursor = _as_text(row.get("precursor")) or "unknown_precursor"
        dopant_sig = "+".join(sorted(_json_list(row.get("doping_elements")))) or _as_text(row.get("doping_state")) or "undoped"
        alloy_sig = "+".join(sorted(_json_list(row.get("alloy_elements")))) or _as_text(row.get("alloy_state")) or "unalloyed"
        confidence = _safe_float(row.get("confidence"), 0.9)
        provenance_ref = f"{_as_text(row.get('citation_url'))}|{_as_text(row.get('locator'))}"

        article_id = f"article:{article_key}"
        deposition_id = f"deposition_state:{article_key}:{point_id}"
        anneal_id = f"anneal_step:{article_key}:{point_id}"
        phase_dep_id = f"film_phase_instance_as_deposited:{article_key}:{point_id}"
        phase_ann_id = f"film_phase_instance_post_anneal:{article_key}:{point_id}"
        quality_id = f"quality_outcome:{article_key}:{point_id}"
        substrate_id = f"substrate:{_normalize_key(substrate)}"
        orientation_id = f"substrate_orientation:{_normalize_key(orientation)}"
        dopant_id = f"dopant_signature:{_normalize_key(dopant_sig)}"
        alloy_id = f"alloy_signature:{_normalize_key(alloy_sig)}"
        precursor_id = f"precursor:{_normalize_key(precursor)}"
        method_dep_id = f"method_step_deposition:{_normalize_key(method_family)}"
        method_ann_id = f"method_step_anneal:{_normalize_key(method_family)}"

        dep_condition = {
            "deposition_thickness_nm": _safe_float(row.get("thickness_nm"), 0.0),
            "deposition_pressure_pa": _safe_float(row.get("pressure_pa"), 0.0),
            "deposition_composition_value": _safe_float(row.get("composition_value"), 0.0),
            "deposition_vacancy_concentration_cm3": _safe_float(row.get("vacancy_concentration_cm3"), 0.0),
            "deposition_temperature_c": _safe_float(row.get("deposition_temperature_c"), 0.0),
            "deposition_rate_nm_s": _safe_float(row.get("deposition_rate_nm_s"), 0.0),
        }
        ann_condition = {
            "anneal_temperature_c": _safe_float(row.get("anneal_temperature_c"), _safe_float(row.get("temperature_c"), 0.0)),
            "anneal_time_s": _safe_float(row.get("anneal_time_s"), 0.0),
            "anneal_rate_c_s": _safe_float(row.get("anneal_rate_c_s"), 0.0),
            "anneal_pressure_pa": _safe_float(row.get("pressure_pa"), 0.0),
            "anneal_ambient": _as_text(row.get("ambient")) or "unknown_ambient",
        }
        dep_condition_id = f"condition_vector_deposition:{_stable_id(article_key, point_id, _json_dump(dep_condition), length=16)}"
        ann_condition_id = f"condition_vector_anneal:{_stable_id(article_key, point_id, _json_dump(ann_condition), length=16)}"

        add_node(article_id, "article", article_key, "deposition", article_id, article_key, {"article_key": article_key})
        add_node(
            deposition_id,
            "deposition_state",
            article_key,
            "deposition",
            deposition_id,
            f"deposition:{point_id}",
            {"point_id": point_id, "method_family": method_family, "film_material": film_material},
        )
        add_node(
            anneal_id,
            "anneal_step",
            article_key,
            "annealing",
            anneal_id,
            f"anneal:{point_id}",
            {"point_id": point_id, "method_family": method_family, "film_material": film_material},
        )
        add_node(
            phase_dep_id,
            "film_phase_instance_as_deposited",
            article_key,
            "deposition",
            phase_dep_id,
            film_material,
            {"phase_label": phase_label, "film_material": film_material, "point_id": point_id},
        )
        add_node(
            phase_ann_id,
            "film_phase_instance_post_anneal",
            article_key,
            "annealing",
            phase_ann_id,
            phase_label,
            {"phase_label": phase_label, "film_material": film_material, "point_id": point_id},
        )
        add_node(substrate_id, "substrate", article_key, "deposition", substrate_id, substrate, {"substrate_material": substrate})
        add_node(
            orientation_id,
            "substrate_orientation",
            article_key,
            "deposition",
            orientation_id,
            orientation,
            {"substrate_orientation": orientation},
        )
        add_node(dopant_id, "dopant_signature", article_key, "deposition", dopant_id, dopant_sig, {"doping_signature": dopant_sig})
        add_node(alloy_id, "alloy_signature", article_key, "deposition", alloy_id, alloy_sig, {"alloy_signature": alloy_sig})
        add_node(precursor_id, "precursor", article_key, "deposition", precursor_id, precursor, {"precursor": precursor})
        add_node(
            method_dep_id,
            "method_step_deposition",
            article_key,
            "deposition",
            method_dep_id,
            method_family,
            {"method_family": method_family},
        )
        add_node(
            method_ann_id,
            "method_step_anneal",
            article_key,
            "annealing",
            method_ann_id,
            method_family,
            {"method_family": method_family},
        )
        add_node(
            dep_condition_id,
            "condition_vector_deposition",
            article_key,
            "deposition",
            dep_condition_id,
            dep_condition_id,
            dep_condition,
        )
        add_node(
            ann_condition_id,
            "condition_vector_anneal",
            article_key,
            "annealing",
            ann_condition_id,
            ann_condition_id,
            ann_condition,
        )
        add_node(
            quality_id,
            "quality_outcome",
            article_key,
            "annealing",
            quality_id,
            "quality_outcome",
            {
                "film_quality_score_numeric": _safe_float(row.get("film_quality_score_numeric"), 0.0),
                "film_quality_label": _as_text(row.get("film_quality_label")),
                "interface_quality_label": _as_text(row.get("interface_quality_label")),
            },
        )

        add_edge(article_key, article_id, deposition_id, "article_has_deposition_state", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, anneal_id, "deposition_state_precedes_anneal_step", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, phase_dep_id, "deposition_state_yields_phase_instance", point_id, provenance_ref, confidence)
        add_edge(article_key, anneal_id, phase_ann_id, "anneal_step_yields_phase_instance", point_id, provenance_ref, confidence)
        add_edge(
            article_key,
            phase_dep_id,
            phase_ann_id,
            "phase_instance_transitions_to_phase_instance",
            point_id,
            provenance_ref,
            confidence,
        )
        add_edge(article_key, phase_dep_id, dep_condition_id, "phase_instance_observed_under_condition", point_id, provenance_ref, confidence)
        add_edge(article_key, phase_ann_id, ann_condition_id, "phase_instance_observed_under_condition", point_id, provenance_ref, confidence)
        add_edge(article_key, phase_ann_id, quality_id, "phase_instance_has_quality_outcome", point_id, provenance_ref, confidence)

        add_edge(article_key, deposition_id, substrate_id, "deposition_state_on_substrate", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, orientation_id, "deposition_state_has_orientation", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, dopant_id, "deposition_state_has_dopant_signature", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, alloy_id, "deposition_state_has_alloy_signature", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, precursor_id, "deposition_state_uses_precursor", point_id, provenance_ref, confidence)
        add_edge(article_key, deposition_id, method_dep_id, "deposition_state_uses_method_step", point_id, provenance_ref, confidence)
        add_edge(article_key, anneal_id, method_ann_id, "anneal_step_uses_method_step", point_id, provenance_ref, confidence)

    nodes = pd.DataFrame(node_rows)
    edges = pd.DataFrame(edge_rows)
    if nodes.empty:
        nodes = pd.DataFrame(
            columns=["node_id", "node_type", "article_key", "stage", "state_id", "label", "attributes_json", "created_at"]
        )
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "edge_id",
                "article_key",
                "source_id",
                "target_id",
                "edge_type",
                "evidence_weight",
                "provenance_ref",
                "source_point_id",
                "created_at",
            ]
        )
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    return ArticleGraphBuildResult(
        nodes_path=nodes_path,
        edges_path=edges_path,
        node_count=int(len(nodes)),
        edge_count=int(len(edges)),
    )


def build_global_concept_graph(points_df: pd.DataFrame, mp_df: pd.DataFrame | None, cfg: RunConfig) -> ConceptGraphBuildResult:
    artifacts = cfg.as_path() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    nodes_path = artifacts / "concept_nodes.parquet"
    edges_path = artifacts / "concept_edges.parquet"

    points = points_df.copy()
    if points.empty:
        pd.DataFrame(
            columns=["concept_id", "concept_type", "canonical_label", "aliases_json", "attributes_json", "created_at"]
        ).to_parquet(nodes_path, index=False)
        pd.DataFrame(
            columns=[
                "edge_id",
                "source_id",
                "target_id",
                "edge_type",
                "evidence_weight",
                "attributes_json",
                "created_at",
            ]
        ).to_parquet(edges_path, index=False)
        return ConceptGraphBuildResult(nodes_path=nodes_path, edges_path=edges_path, node_count=0, edge_count=0)

    if mp_df is not None and not mp_df.empty and "point_id" in points.columns and "point_id" in mp_df.columns:
        keep = [
            col
            for col in (
                "point_id",
                "mp_energy_above_hull_min",
                "band_gap",
                "is_metal",
                "magnetic_ordering",
                "mp_spacegroup_symbol",
                "mp_spacegroup_number",
            )
            if col in mp_df.columns
        ]
        if len(keep) > 1:
            points = points.drop(columns=[col for col in keep if col != "point_id" and col in points.columns], errors="ignore")
            points = points.merge(mp_df[keep], on="point_id", how="left")

    concept_rows: dict[str, dict[str, Any]] = {}
    edge_rows: list[dict[str, Any]] = []
    seen_edges: set[str] = set()

    def add_concept(
        concept_type: str,
        canonical_label: str,
        *,
        aliases: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        label = _as_text(canonical_label) or f"unknown_{concept_type}"
        key = f"{concept_type}:{_normalize_key(label)}"
        concept_id = f"concept:{key}"
        if concept_id not in concept_rows:
            concept_rows[concept_id] = {
                "concept_id": concept_id,
                "concept_type": concept_type,
                "canonical_label": label,
                "aliases_json": _json_dump(sorted(set(aliases or []))),
                "attributes_json": _json_dump(attrs or {}),
                "created_at": now_iso(),
            }
        return concept_id

    def add_edge(source_id: str, target_id: str, edge_type: str, evidence_weight: float, attrs: dict[str, Any] | None = None) -> None:
        edge_id = f"edge_{_stable_id(source_id, target_id, edge_type, length=20)}"
        if edge_id in seen_edges:
            return
        seen_edges.add(edge_id)
        edge_rows.append(
            {
                "edge_id": edge_id,
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
                "evidence_weight": max(0.0, min(1.0, _safe_float(evidence_weight, 0.7))),
                "attributes_json": _json_dump(attrs or {}),
                "created_at": now_iso(),
            }
        )

    def seed_global_ontology() -> None:
        if str(cfg.concept_ontology_profile or "").strip().lower() != "maximal":
            return
        for element in _PERIODIC_TABLE_SYMBOLS:
            add_concept("element", element, attrs={"element": element, "seeded": True})
            add_concept("dopant_element", element, attrs={"element": element, "seeded": True})
            add_concept("alloy_element", element, attrs={"element": element, "seeded": True})
        for system in _GLOBAL_CRYSTAL_SYSTEMS:
            add_concept("crystal_system", system, attrs={"seeded": True})
        for number in range(1, 231):
            sg_label = f"SG{number}"
            sg_id = add_concept(
                "space_group",
                sg_label,
                attrs={"spacegroup_symbol": "", "spacegroup_number": str(number), "seeded": True},
            )
            crystal_id = add_concept("crystal_system", _spacegroup_to_crystal_system(number), attrs={"seeded": True})
            add_edge(
                sg_id,
                crystal_id,
                "space_group_in_crystal_system",
                0.6,
                attrs={"seeded": True},
            )
        for orientation in _GLOBAL_ORIENTATION_FAMILIES:
            add_concept("orientation_family", orientation, attrs={"seeded": True})
        for method in _GLOBAL_METHOD_FAMILIES:
            add_concept("method_family", method, attrs={"seeded": True})
        for ambient in _GLOBAL_AMBIENT_GASES:
            add_concept("ambient_gas", ambient, attrs={"seeded": True})
        for label in _GLOBAL_TEMPERATURE_REGIMES:
            add_concept("temperature_regime", label, attrs={"seeded": True})
        for label in _GLOBAL_TIME_REGIMES:
            add_concept("time_regime", label, attrs={"seeded": True})
        for label in _GLOBAL_PRESSURE_REGIMES:
            add_concept("pressure_regime", label, attrs={"seeded": True})
        for label in _GLOBAL_STABILITY_BANDS:
            add_concept("stability_band", label, attrs={"seeded": True})
        for label in _GLOBAL_ENERGY_HULL_BUCKETS:
            add_concept("energy_hull_bucket", label, attrs={"seeded": True})
        for label in _GLOBAL_BANDGAP_BUCKETS:
            add_concept("bandgap_bucket", label, attrs={"seeded": True})
        for label in _GLOBAL_METALLICITY_CLASSES:
            add_concept("metallicity_class", label, attrs={"seeded": True})
        for label in _GLOBAL_MAGNETIC_CLASSES:
            add_concept("magnetic_class", label, attrs={"seeded": True})
        add_concept("substrate_material", "unknown_substrate", attrs={"seeded": True})
        add_concept("precursor_chemistry", "unknown_precursor", attrs={"seeded": True})
        add_concept("prototype_family", "prototype_unknown", attrs={"seeded": True})
        add_concept("quality_metric_type", "film_quality_score_numeric", attrs={"seeded": True})
        add_concept("defect_metric_type", "vacancy_concentration_cm3", attrs={"seeded": True})
        add_concept("defect_metric_type", "defect_unknown", attrs={"seeded": True})
        add_concept("interface_state_type", "interface_unknown", attrs={"seeded": True})

    seed_global_ontology()

    for row in points.to_dict(orient="records"):
        formula = _normalize_formula(row.get("film_material") or row.get("phase_label") or row.get("entity"))
        elements, stoich = _formula_components(formula)
        formula_id = add_concept("formula", formula or "unknown_formula", attrs={"reduced_formula": formula})
        for elem in elements:
            elem_id = add_concept("element", elem, attrs={"element": elem})
            add_edge(formula_id, elem_id, "formula_has_element", _safe_float(row.get("confidence"), 0.8))
        for comp in stoich:
            token = f"{_as_text(comp.get('element'))}:{_safe_float(comp.get('ratio'), 1.0):.4f}"
            comp_id = add_concept("stoich_component", token, attrs={"element": _as_text(comp.get("element")), "ratio": _safe_float(comp.get("ratio"), 1.0)})
            add_edge(formula_id, comp_id, "formula_has_stoich_component", _safe_float(row.get("confidence"), 0.8))

        spacegroup_symbol = _normalize_spacegroup_symbol(row.get("spacegroup_symbol") or row.get("mp_spacegroup_symbol"))
        spacegroup_number = _normalize_spacegroup_number(row.get("spacegroup_number") or row.get("mp_spacegroup_number"))
        if spacegroup_symbol or spacegroup_number:
            sg_label = spacegroup_symbol or f"sg{spacegroup_number}"
            sg_id = add_concept(
                "space_group",
                sg_label,
                attrs={"spacegroup_symbol": spacegroup_symbol, "spacegroup_number": spacegroup_number},
            )
            add_edge(formula_id, sg_id, "formula_has_space_group", _safe_float(row.get("confidence"), 0.8))
            crystal_system = "unknown_crystal_system"
            if spacegroup_number.isdigit():
                crystal_system = _spacegroup_to_crystal_system(int(spacegroup_number))
            crystal_id = add_concept("crystal_system", crystal_system)
            add_edge(sg_id, crystal_id, "space_group_in_crystal_system", 0.8)

        substrate = _as_text(row.get("substrate_material")) or "unknown_substrate"
        add_concept("substrate_material", substrate)
        orientation = _as_text(row.get("substrate_orientation")) or "unknown_orientation"
        orientation_id = add_concept("orientation_family", orientation)
        add_edge(
            add_concept("substrate_material", substrate),
            orientation_id,
            "substrate_has_orientation_family",
            _safe_float(row.get("confidence"), 0.8),
        )

        for element in _json_list(row.get("doping_elements")):
            add_concept("dopant_element", element, attrs={"element": element})
        for element in _json_list(row.get("alloy_elements")):
            add_concept("alloy_element", element, attrs={"element": element})
        precursor = _as_text(row.get("precursor")) or "unknown_precursor"
        add_concept("precursor_chemistry", precursor)

        method = _as_text(row.get("method_family")) or "unknown_method"
        add_concept("method_family", method)
        add_concept("ambient_gas", _as_text(row.get("ambient")) or "unknown_ambient")
        add_concept("temperature_regime", _bucket_temperature(_safe_float(row.get("anneal_temperature_c"), 0.0)))
        add_concept("time_regime", _bucket_time(_safe_float(row.get("anneal_time_s"), 0.0)))
        add_concept("pressure_regime", _bucket_pressure(_safe_float(row.get("pressure_pa"), 0.0)))

        e_hull = _safe_float(row.get("mp_energy_above_hull_min"), -1.0)
        add_concept("stability_band", _bucket_stability(e_hull), attrs={"energy_above_hull": e_hull})
        add_concept("energy_hull_bucket", _bucket_energy_hull(e_hull), attrs={"energy_above_hull": e_hull})
        add_concept("bandgap_bucket", _bucket_bandgap(_safe_float(row.get("band_gap"), -1.0)))
        is_metal = _as_text(row.get("is_metal")).lower()
        metallicity = "metallicity_unknown"
        if is_metal in {"true", "1", "yes", "metal"}:
            metallicity = "metallic"
        elif is_metal in {"false", "0", "no", "insulator"}:
            metallicity = "non_metallic"
        add_concept("metallicity_class", metallicity)
        add_concept("magnetic_class", _as_text(row.get("magnetic_ordering")) or "magnetic_unknown")

        add_concept("quality_metric_type", "film_quality_score_numeric")
        defect_metric = "vacancy_concentration_cm3" if _safe_float(row.get("vacancy_concentration_cm3"), 0.0) > 0 else "defect_unknown"
        add_concept("defect_metric_type", defect_metric)
        add_concept("interface_state_type", _as_text(row.get("interface_quality_label")) or "interface_unknown")
        add_concept("prototype_family", _as_text(row.get("prototype_family")) or "prototype_unknown")

    nodes = pd.DataFrame(list(concept_rows.values()))
    edges = pd.DataFrame(edge_rows)
    if nodes.empty:
        nodes = pd.DataFrame(
            columns=["concept_id", "concept_type", "canonical_label", "aliases_json", "attributes_json", "created_at"]
        )
    if edges.empty:
        edges = pd.DataFrame(
            columns=["edge_id", "source_id", "target_id", "edge_type", "evidence_weight", "attributes_json", "created_at"]
        )
    nodes.to_parquet(nodes_path, index=False)
    edges.to_parquet(edges_path, index=False)
    return ConceptGraphBuildResult(
        nodes_path=nodes_path,
        edges_path=edges_path,
        node_count=int(len(nodes)),
        edge_count=int(len(edges)),
    )


def build_bridge_edges(
    points_df: pd.DataFrame,
    article_graph: ArticleGraphBuildResult,
    concept_graph: ConceptGraphBuildResult,
    cfg: RunConfig,
) -> BridgeBuildResult:
    artifacts = cfg.as_path() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    bridge_path = artifacts / "bridge_edges.parquet"
    audit_path = artifacts / "bridge_weight_audit.parquet"

    points = points_df.copy()
    if points.empty:
        pd.DataFrame(
            columns=[
                "bridge_id",
                "article_key",
                "source_id",
                "target_id",
                "gate_type",
                "weight",
                "weight_components_json",
                "provenance_ref",
                "source_point_id",
                "created_at",
            ]
        ).to_parquet(bridge_path, index=False)
        pd.DataFrame(
            columns=[
                "article_key",
                "source_id",
                "target_id",
                "gate_type",
                "keep",
                "weight",
                "weight_components_json",
                "source_point_id",
                "created_at",
            ]
        ).to_parquet(audit_path, index=False)
        return BridgeBuildResult(edges_path=bridge_path, audit_path=audit_path, edge_count=0, candidate_count=0)

    concept_nodes = pd.read_parquet(concept_graph.nodes_path)
    concept_lookup: dict[tuple[str, str], str] = {}
    for row in concept_nodes.to_dict(orient="records"):
        ctype = _as_text(row.get("concept_type"))
        label = _normalize_key(row.get("canonical_label"))
        if ctype and label:
            concept_lookup[(ctype, label)] = _as_text(row.get("concept_id"))

    allowed_gates = _concept_gate_set(cfg)
    support_map = _cross_source_map(points)

    bridge_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen_bridge: set[str] = set()

    def add_candidate(
        *,
        article_key: str,
        source_id: str,
        target_id: str,
        gate_type: str,
        source_point_id: str,
        provenance_ref: str,
        components: dict[str, float],
    ) -> None:
        gate_clean = gate_type.replace("_gate", "")
        keep_gate = gate_clean in allowed_gates
        weight = score_bridge_weight(components, cfg)
        keep = bool(keep_gate and weight >= float(cfg.bridge_weight_threshold))
        row = {
            "article_key": article_key,
            "source_id": source_id,
            "target_id": target_id,
            "gate_type": gate_type,
            "keep": keep,
            "weight": weight,
            "weight_components_json": _json_dump(components),
            "source_point_id": source_point_id,
            "created_at": now_iso(),
        }
        audit_rows.append(row)
        if not keep:
            return
        bridge_id = f"bridge_{_stable_id(article_key, source_id, target_id, gate_type, source_point_id, length=20)}"
        if bridge_id in seen_bridge:
            return
        seen_bridge.add(bridge_id)
        bridge_rows.append(
            {
                "bridge_id": bridge_id,
                "article_key": article_key,
                "source_id": source_id,
                "target_id": target_id,
                "gate_type": gate_type,
                "weight": weight,
                "weight_components_json": _json_dump(components),
                "provenance_ref": provenance_ref,
                "source_point_id": source_point_id,
                "created_at": now_iso(),
            }
        )

    for row in points.to_dict(orient="records"):
        article_key = _as_text(row.get("article_key")) or "unknown_article"
        point_id = _as_text(row.get("point_id")) or f"{article_key}:unknown_point"
        phase_source = f"film_phase_instance_post_anneal:{article_key}:{point_id}"
        deposition_source = f"deposition_state:{article_key}:{point_id}"
        anneal_source = f"anneal_step:{article_key}:{point_id}"
        provenance_ref = f"{_as_text(row.get('citation_url'))}|{_as_text(row.get('locator'))}"
        extraction_conf = max(0.0, min(1.0, _safe_float(row.get("confidence"), 0.8)))
        provenance_quality = _provenance_quality(row)
        mp_alignment = _mp_alignment_score(row)

        formula = _normalize_formula(row.get("film_material") or row.get("phase_label") or row.get("entity"))
        formula_key = ("formula", _normalize_key(formula or "unknown_formula"))
        target_formula = concept_lookup.get(formula_key)
        if target_formula:
            concept_support = support_map.get(f"formula:{formula_key[1]}", 0.3)
            add_candidate(
                article_key=article_key,
                source_id=phase_source,
                target_id=target_formula,
                gate_type="film_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("film_gate"),
                    "cross_source_agreement": concept_support,
                    "mp_alignment": mp_alignment,
                },
            )

        elements, _stoich = _formula_components(formula)
        for element in elements:
            key = ("element", _normalize_key(element))
            target = concept_lookup.get(key)
            if not target:
                continue
            concept_support = support_map.get(f"element:{_normalize_key(element)}", 0.3)
            add_candidate(
                article_key=article_key,
                source_id=phase_source,
                target_id=target,
                gate_type="film_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("film_gate"),
                    "cross_source_agreement": concept_support,
                    "mp_alignment": mp_alignment,
                },
            )

        substrate = _as_text(row.get("substrate_material")) or "unknown_substrate"
        substrate_target = concept_lookup.get(("substrate_material", _normalize_key(substrate)))
        if substrate_target:
            add_candidate(
                article_key=article_key,
                source_id=deposition_source,
                target_id=substrate_target,
                gate_type="substrate_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("substrate_gate"),
                    "cross_source_agreement": support_map.get(f"substrate_material:{_normalize_key(substrate)}", 0.3),
                    "mp_alignment": mp_alignment,
                },
            )

        orientation = _as_text(row.get("substrate_orientation")) or "unknown_orientation"
        orient_target = concept_lookup.get(("orientation_family", _normalize_key(orientation)))
        if orient_target:
            add_candidate(
                article_key=article_key,
                source_id=deposition_source,
                target_id=orient_target,
                gate_type="substrate_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("substrate_gate"),
                    "cross_source_agreement": 0.4,
                    "mp_alignment": mp_alignment,
                },
            )

        precursor = _as_text(row.get("precursor")) or "unknown_precursor"
        precursor_target = concept_lookup.get(("precursor_chemistry", _normalize_key(precursor)))
        if precursor_target:
            add_candidate(
                article_key=article_key,
                source_id=deposition_source,
                target_id=precursor_target,
                gate_type="precursor_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("precursor_gate"),
                    "cross_source_agreement": 0.35,
                    "mp_alignment": mp_alignment,
                },
            )

        method = _as_text(row.get("method_family")) or "unknown_method"
        method_target = concept_lookup.get(("method_family", _normalize_key(method)))
        if method_target:
            add_candidate(
                article_key=article_key,
                source_id=deposition_source,
                target_id=method_target,
                gate_type="method_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("method_gate"),
                    "cross_source_agreement": 0.4,
                    "mp_alignment": mp_alignment,
                },
            )

        temp_target = concept_lookup.get(
            ("temperature_regime", _normalize_key(_bucket_temperature(_safe_float(row.get("anneal_temperature_c"), 0.0))))
        )
        if temp_target:
            add_candidate(
                article_key=article_key,
                source_id=anneal_source,
                target_id=temp_target,
                gate_type="method_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("method_gate"),
                    "cross_source_agreement": 0.35,
                    "mp_alignment": mp_alignment,
                },
            )

        e_hull = _safe_float(row.get("mp_energy_above_hull_min"), -1.0)
        stability_target = concept_lookup.get(("stability_band", _normalize_key(_bucket_stability(e_hull))))
        if stability_target:
            add_candidate(
                article_key=article_key,
                source_id=phase_source,
                target_id=stability_target,
                gate_type="thermo_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("thermo_gate"),
                    "cross_source_agreement": 0.30,
                    "mp_alignment": mp_alignment,
                },
            )

        metallicity_target = concept_lookup.get(
            ("metallicity_class", _normalize_key("metallic" if _as_text(row.get("is_metal")).lower() in {"true", "1", "yes", "metal"} else "metallicity_unknown"))
        )
        if metallicity_target:
            add_candidate(
                article_key=article_key,
                source_id=phase_source,
                target_id=metallicity_target,
                gate_type="electronic_gate",
                source_point_id=point_id,
                provenance_ref=provenance_ref,
                components={
                    "extraction_conf": extraction_conf,
                    "provenance_quality": provenance_quality,
                    "context_proximity": _context_proximity_score("electronic_gate"),
                    "cross_source_agreement": 0.30,
                    "mp_alignment": mp_alignment,
                },
            )

    edges = pd.DataFrame(bridge_rows)
    audit = pd.DataFrame(audit_rows)
    if edges.empty:
        edges = pd.DataFrame(
            columns=[
                "bridge_id",
                "article_key",
                "source_id",
                "target_id",
                "gate_type",
                "weight",
                "weight_components_json",
                "provenance_ref",
                "source_point_id",
                "created_at",
            ]
        )
    if audit.empty:
        audit = pd.DataFrame(
            columns=[
                "article_key",
                "source_id",
                "target_id",
                "gate_type",
                "keep",
                "weight",
                "weight_components_json",
                "source_point_id",
                "created_at",
            ]
        )
    edges.to_parquet(bridge_path, index=False)
    audit.to_parquet(audit_path, index=False)
    return BridgeBuildResult(
        edges_path=bridge_path,
        audit_path=audit_path,
        edge_count=int(len(edges)),
        candidate_count=int(len(audit)),
    )


def audit_dual_graph(run_dir: str | Path, cfg: RunConfig) -> GraphAuditResult:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = run_path / "artifacts"
    json_path = artifacts / "graph_audit.json"
    report_path = artifacts / "graph_audit.md"

    required = {
        "article_process_nodes": artifacts / "article_process_nodes.parquet",
        "article_process_edges": artifacts / "article_process_edges.parquet",
        "concept_nodes": artifacts / "concept_nodes.parquet",
        "concept_edges": artifacts / "concept_edges.parquet",
        "bridge_edges": artifacts / "bridge_edges.parquet",
        "bridge_weight_audit": artifacts / "bridge_weight_audit.parquet",
    }
    issues: list[str] = []
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        issues.append(f"missing artifacts: {', '.join(missing)}")

    summary: dict[str, Any] = {
        "created_at": now_iso(),
        "run_dir": str(run_path),
        "graph_schema_version": "v2_dual_concept",
        "bridge_weight_threshold": float(cfg.bridge_weight_threshold),
        "artifact_paths": {name: str(path) for name, path in required.items()},
    }
    if missing:
        payload = {"ok": False, "issues": issues, "summary": summary}
        write_json(json_path, payload)
        report_path.write_text(
            "# Graph Audit\n\n- status: failed\n- issues:\n" + "\n".join(f"  - {item}" for item in issues) + "\n",
            encoding="utf-8",
        )
        return GraphAuditResult(ok=False, summary=summary, issues=issues, json_path=json_path, report_path=report_path)

    ap_nodes = pd.read_parquet(required["article_process_nodes"])
    ap_edges = pd.read_parquet(required["article_process_edges"])
    concept_nodes = pd.read_parquet(required["concept_nodes"])
    concept_edges = pd.read_parquet(required["concept_edges"])
    bridge_edges = pd.read_parquet(required["bridge_edges"])
    bridge_audit = pd.read_parquet(required["bridge_weight_audit"])

    ap_node_ids = set(ap_nodes.get("node_id", pd.Series([], dtype=str)).astype(str).tolist())
    concept_node_ids = set(concept_nodes.get("concept_id", pd.Series([], dtype=str)).astype(str).tolist())

    if not ap_edges.empty:
        src_ok = ap_edges["source_id"].astype(str).isin(ap_node_ids)
        dst_ok = ap_edges["target_id"].astype(str).isin(ap_node_ids)
        if not bool((src_ok & dst_ok).all()):
            issues.append("article_process_edges contain orphan references")

    if not concept_edges.empty:
        src_ok = concept_edges["source_id"].astype(str).isin(concept_node_ids)
        dst_ok = concept_edges["target_id"].astype(str).isin(concept_node_ids)
        if not bool((src_ok & dst_ok).all()):
            issues.append("concept_edges contain orphan references")

    if not bridge_edges.empty:
        src_ok = bridge_edges["source_id"].astype(str).isin(ap_node_ids)
        dst_ok = bridge_edges["target_id"].astype(str).isin(concept_node_ids)
        if not bool((src_ok & dst_ok).all()):
            issues.append("bridge_edges contain orphan references")
        weights = bridge_edges["weight"].map(lambda v: _safe_float(v, -1.0))
        if not bool(((weights >= 0.0) & (weights <= 1.0)).all()):
            issues.append("bridge_edges contain weight outside [0,1]")
        if not bool((weights >= float(cfg.bridge_weight_threshold)).all()):
            issues.append("bridge_edges include edge below bridge_weight_threshold")

    if not bridge_audit.empty and "weight" in bridge_audit.columns:
        weights = bridge_audit["weight"].map(lambda v: _safe_float(v, -1.0))
        if not bool(((weights >= 0.0) & (weights <= 1.0)).all()):
            issues.append("bridge_weight_audit contains weight outside [0,1]")

    dep_to_ann_ok = True
    if not ap_edges.empty:
        dep_ann = ap_edges[ap_edges["edge_type"].astype(str) == "deposition_state_precedes_anneal_step"]
        if not dep_ann.empty:
            node_ids = ap_nodes["node_id"].astype(str).tolist()
            node_types = ap_nodes["node_type"].astype(str).tolist()
            bound = min(len(node_ids), len(node_types))
            type_lookup = {node_ids[idx]: node_types[idx] for idx in range(bound)}
            for row in dep_ann.to_dict(orient="records"):
                if type_lookup.get(_as_text(row.get("source_id"))) != "deposition_state":
                    dep_to_ann_ok = False
                    break
                if type_lookup.get(_as_text(row.get("target_id"))) != "anneal_step":
                    dep_to_ann_ok = False
                    break
    if not dep_to_ann_ok:
        issues.append("deposition_state_precedes_anneal_step edge has invalid source/target types")

    summary.update(
        {
            "article_process_nodes": int(len(ap_nodes)),
            "article_process_edges": int(len(ap_edges)),
            "concept_nodes": int(len(concept_nodes)),
            "concept_edges": int(len(concept_edges)),
            "bridge_edges": int(len(bridge_edges)),
            "bridge_candidates": int(len(bridge_audit)),
            "dep_to_ann_edge_type_valid": bool(dep_to_ann_ok),
            "no_orphans_article_process": "article_process_edges contain orphan references" not in issues,
            "no_orphans_concept": "concept_edges contain orphan references" not in issues,
            "no_orphans_bridge": "bridge_edges contain orphan references" not in issues,
        }
    )
    ok = len(issues) == 0
    payload = {"ok": ok, "issues": issues, "summary": summary}
    write_json(json_path, payload)

    md_lines = [
        "# Graph Audit",
        "",
        f"- status: {'ok' if ok else 'failed'}",
        f"- graph_schema_version: {summary['graph_schema_version']}",
        f"- article_process_nodes: {summary['article_process_nodes']}",
        f"- article_process_edges: {summary['article_process_edges']}",
        f"- concept_nodes: {summary['concept_nodes']}",
        f"- concept_edges: {summary['concept_edges']}",
        f"- bridge_edges: {summary['bridge_edges']}",
        f"- bridge_candidates: {summary['bridge_candidates']}",
    ]
    if issues:
        md_lines.append("- issues:")
        md_lines.extend([f"  - {item}" for item in issues])
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return GraphAuditResult(ok=ok, summary=summary, issues=issues, json_path=json_path, report_path=report_path)
