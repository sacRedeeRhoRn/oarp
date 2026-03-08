from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import oarp.graph_dual as graph_dual
from oarp.cache import prepare_feature_cache
from oarp.models import (
    FineTuneSliceSpec,
    GraphBuildResult,
    ProcessorDataset,
    ProcessorModelBundle,
    ProcessorModelRef,
    RunConfig,
)
from oarp.runtime import now_iso

_NUMERIC_FEATURES = [
    "thickness_nm",
    "anneal_temperature_c",
    "anneal_time_s",
    "pressure_pa",
    "strain_pct",
    "composition_value",
    "vacancy_concentration_cm3",
    "film_quality_score_numeric",
    "mp_interpretation_score",
    "mp_formula_match_score",
    "mp_phase_match_score",
    "mp_stability_score",
    "mp_energy_above_hull_min",
    "concept_bridge_count",
    "concept_bridge_mean_weight",
    "concept_bridge_max_weight",
    "concept_gate_coverage",
]

_CATEGORICAL_FEATURES = [
    "phase_label",
    "film_material",
    "phase_signature",
    "substrate_material",
    "substrate_orientation",
    "method_family",
    "doping_state",
    "alloy_state",
    "mp_interpretation_label",
    "mp_spacegroup_symbol",
    "spacegroup_symbol",
]

_NODE_TYPES = [
    "article",
    "material",
    "phase",
    "substrate",
    "orientation",
    "dopant",
    "method_step",
    "element",
    "stoich_component",
    "space_group",
    "condition_vector",
]
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")
_SPACEGROUP_CLEAN_RE = re.compile(r"[^A-Za-z0-9]")
_FEATURE_NAME_CLEAN_RE = re.compile(r"[^0-9A-Za-z_]+")


@dataclass
class _Split:
    train: pd.DataFrame
    test: pd.DataFrame


class _EncodedLabelClassifier:
    """Adapter for estimators trained on numeric labels but evaluated on string labels."""

    def __init__(self, base: Any, labels: list[str]) -> None:
        self.base = base
        self.classes_ = np.asarray(labels, dtype=object)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        idx = np.asarray(self.base.predict(x), dtype=int)
        idx = np.clip(idx, 0, max(0, len(self.classes_) - 1))
        return np.asarray([self.classes_[item] for item in idx], dtype=object)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.base.predict_proba(x), dtype=float)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if np.isnan(out) or np.isinf(out):
        return float(default)
    return float(out)


def _as_text(value: Any) -> str:
    return str(value or "").strip()


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


def _load_points_frame(points: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(points, pd.DataFrame):
        return points.copy()
    path = Path(points).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"missing points dataframe: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"unsupported points format: {path}")


def _point_level_from_variable_rows(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame()
    if "variable_name" not in points.columns:
        out = points.copy()
        if "article_key" not in out.columns:
            out["article_key"] = [f"article_{idx}" for idx in range(len(out))]
        if "point_id" not in out.columns:
            out["point_id"] = [f"point_{idx}" for idx in range(len(out))]
        if "phase_label" not in out.columns:
            out["phase_label"] = out.get("entity", "").astype(str)
        return out
    if "point_id" not in points.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for point_id, group in points.groupby("point_id"):
        row: dict[str, Any] = {"point_id": str(point_id)}
        for _, item in group.iterrows():
            var_name = _as_text(item.get("variable_name"))
            if not var_name:
                continue
            row[var_name] = _safe_float(item.get("normalized_value"), 0.0)
        first = group.iloc[0]
        passthrough_cols = [
            "article_key",
            "provider",
            "entity",
            "citation_url",
            "doi",
            "locator",
            "snippet",
            "substrate_material",
            "substrate_orientation",
            "doping_state",
            "alloy_state",
            "film_material",
            "phase_label",
            "method_family",
            "film_quality_score_numeric",
            "mp_interpretation_score",
            "mp_formula_match_score",
            "mp_phase_match_score",
            "mp_stability_score",
            "mp_energy_above_hull_min",
            "mp_interpretation_label",
            "mp_spacegroup_symbol",
            "mp_spacegroup_number",
            "doping_elements",
            "doping_composition",
            "alloy_elements",
            "alloy_composition",
        ]
        for col in passthrough_cols:
            if col in group.columns:
                row[col] = first.get(col)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if "article_key" not in frame.columns:
        frame["article_key"] = [f"article_{idx}" for idx in range(len(frame))]
    frame["article_key"] = frame["article_key"].astype(str).replace("", np.nan).fillna("unknown_article")
    if "phase_label" not in frame.columns:
        frame["phase_label"] = frame.get("entity", "").astype(str)
    if "film_material" not in frame.columns:
        frame["film_material"] = frame.get("entity", "").astype(str)
    return frame


def _fill_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in _NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].map(lambda v: _safe_float(v, 0.0))
    for col in _CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str).fillna("").str.strip()
    if "phase_label" not in out.columns:
        out["phase_label"] = out.get("entity", "").astype(str)
    out["phase_label"] = out["phase_label"].astype(str).str.strip().replace("", "unknown")
    if "film_material" not in out.columns:
        out["film_material"] = out.get("phase_label", "").astype(str)
    if "method_family" not in out.columns:
        out["method_family"] = ""
    out["created_at"] = now_iso()
    return out


def _attach_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    work = frame.copy()
    for col in ("phase_label", "substrate_material", "method_family"):
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].astype(str)

    phase_counts = work["phase_label"].value_counts()
    substrate_counts = work["substrate_material"].value_counts()
    method_counts = work["method_family"].value_counts()

    work["graph_degree_phase"] = work["phase_label"].map(phase_counts).fillna(0).astype(float)
    work["graph_degree_substrate"] = work["substrate_material"].map(substrate_counts).fillna(0).astype(float)
    work["graph_degree_method"] = work["method_family"].map(method_counts).fillna(0).astype(float)
    work["graph_support"] = (
        0.50 * work["graph_degree_phase"]
        + 0.30 * work["graph_degree_substrate"]
        + 0.20 * work["graph_degree_method"]
    )
    return work


def _stable_id(*parts: Any, length: int = 24) -> str:
    digest = hashlib.sha1("|".join(str(item) for item in parts).encode("utf-8", errors="replace")).hexdigest()
    return digest[: int(length)]


def _normalize_formula(raw: Any) -> str:
    text = _as_text(raw).replace(" ", "").replace("-", "")
    text = re.sub(r"[^A-Za-z0-9.]", "", text)
    if not text:
        return ""
    tokens = _FORMULA_TOKEN_RE.findall(text)
    if not tokens:
        return ""
    rebuilt = "".join(f"{el}{cnt}" for el, cnt in tokens)
    if rebuilt.lower() != text.lower():
        # Keep robust for OCR spacing but avoid malformed non-formula payloads.
        compact = re.sub(r"[^A-Za-z0-9]", "", text)
        if rebuilt.lower() != compact.lower():
            return ""
    return rebuilt


def _formula_components(formula: str) -> tuple[list[str], list[dict[str, Any]]]:
    elements: list[str] = []
    stoich: list[dict[str, Any]] = []
    for element, count_text in _FORMULA_TOKEN_RE.findall(formula):
        if element not in elements:
            elements.append(element)
        ratio = _safe_float(count_text or 1.0, 1.0)
        if ratio <= 0:
            ratio = 1.0
        stoich.append({"element": element, "ratio": float(ratio)})
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


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _frame_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "empty"
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "no_columns"
    work = frame[cols].copy().fillna("").astype(str)
    work = work.sort_values(cols).reset_index(drop=True)
    payload = "\n".join("|".join(row) for row in work.itertuples(index=False, name=None))
    return hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()


def _phase_signature(
    *,
    reduced_formula: str,
    stoich: list[dict[str, Any]],
    spacegroup_symbol: str,
    spacegroup_number: str,
) -> str:
    payload = {
        "reduced_formula": reduced_formula,
        "stoich": sorted(stoich, key=lambda item: (str(item.get("element") or ""), float(item.get("ratio") or 0.0))),
        "spacegroup_symbol": spacegroup_symbol,
        "spacegroup_number": spacegroup_number,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8", errors="replace")).hexdigest()


def build_phase_spec(points_df: pd.DataFrame | str | Path, cfg: RunConfig) -> pd.DataFrame:
    frame = _point_level_from_variable_rows(_load_points_frame(points_df))
    if frame.empty:
        out = pd.DataFrame(
            columns=[
                "phase_id",
                "reduced_formula",
                "elements_json",
                "stoichiometry_json",
                "spacegroup_symbol",
                "spacegroup_number",
                "phase_signature",
                "created_at",
            ]
        )
        out_path = cfg.as_path() / "artifacts" / "phase_specs.parquet"
        out.to_parquet(out_path, index=False)
        return out

    cache = prepare_feature_cache(cfg)
    key = cache.make_key(
        "phase_spec",
        str(cfg.phase_schema_version),
        _frame_fingerprint(frame, ["phase_label", "film_material", "mp_spacegroup_symbol", "mp_spacegroup_number"]),
    )
    cached = cache.get_json("phase_spec", key, ttl_hours=int(cfg.cache_ttl_hours))
    if isinstance(cached, dict):
        rows = cached.get("rows")
        if isinstance(rows, list):
            try:
                cached_df = pd.DataFrame(rows)
                required = {"phase_id", "reduced_formula", "elements_json", "stoichiometry_json", "phase_signature"}
                if required.issubset(set(cached_df.columns)):
                    out_path = cfg.as_path() / "artifacts" / "phase_specs.parquet"
                    cached_df.to_parquet(out_path, index=False)
                    cache.write_audit(cfg.as_path() / "artifacts" / "cache_audit.parquet")
                    return cached_df
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        formula = _normalize_formula(row.get("film_material") or row.get("phase_label") or row.get("entity"))
        elements, stoich = _formula_components(formula)
        sg_symbol = _normalize_spacegroup_symbol(row.get("mp_spacegroup_symbol"))
        sg_number = _normalize_spacegroup_number(row.get("mp_spacegroup_number"))

        phase_signature = _phase_signature(
            reduced_formula=formula,
            stoich=stoich,
            spacegroup_symbol=sg_symbol,
            spacegroup_number=sg_number,
        )
        rows.append(
            {
                "phase_id": f"phase_{_stable_id(phase_signature, length=16)}",
                "reduced_formula": formula,
                "elements_json": _json_dump(elements),
                "stoichiometry_json": _json_dump(stoich),
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": sg_number,
                "phase_signature": phase_signature,
                "created_at": now_iso(),
            }
        )
    out = pd.DataFrame(rows).drop_duplicates(subset=["phase_signature"], keep="first").reset_index(drop=True)
    out_path = cfg.as_path() / "artifacts" / "phase_specs.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    cache.put_json(
        "phase_spec",
        key,
        {"rows": out.to_dict(orient="records")},
        key_text=f"phase_schema:{cfg.phase_schema_version}",
        ttl_hours=int(cfg.cache_ttl_hours),
    )
    cache.write_audit(cfg.as_path() / "artifacts" / "cache_audit.parquet")
    return out


def build_article_phase_graph(
    points_df: pd.DataFrame | str | Path,
    phase_spec_df: pd.DataFrame,
    cfg: RunConfig,
) -> GraphBuildResult:
    frame = _point_level_from_variable_rows(_load_points_frame(points_df))
    artifacts = cfg.as_path() / "artifacts"
    nodes_path = artifacts / "graph_nodes.parquet"
    edges_path = artifacts / "graph_edges.parquet"
    tensor_index_path = artifacts / "graph_tensor_index.parquet"
    artifacts.mkdir(parents=True, exist_ok=True)

    if frame.empty:
        pd.DataFrame(
            columns=["node_id", "node_type", "article_key", "label", "attributes_json", "created_at"]
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
        pd.DataFrame(
            columns=["graph_sample_id", "article_key", "node_count", "edge_count", "tensor_path", "created_at"]
        ).to_parquet(tensor_index_path, index=False)
        return GraphBuildResult(
            nodes_path=nodes_path,
            edges_path=edges_path,
            tensor_index_path=tensor_index_path,
            graph_sample_count=0,
        )

    cache = prepare_feature_cache(cfg)
    key = cache.make_key(
        "graph_build",
        str(cfg.graph_core_mode),
        str(cfg.phase_schema_version),
        _frame_fingerprint(frame, ["article_key", "point_id", "phase_label", "film_material", "thickness_nm", "anneal_temperature_c"]),
        _frame_fingerprint(phase_spec_df, ["phase_signature", "phase_id"]),
    )
    cached = cache.get_json("graph_build_index", key, ttl_hours=int(cfg.cache_ttl_hours))
    if isinstance(cached, dict):
        cached_nodes = Path(str(cached.get("nodes_path") or ""))
        cached_edges = Path(str(cached.get("edges_path") or ""))
        cached_tensor = Path(str(cached.get("tensor_index_path") or ""))
        if cached_nodes.exists() and cached_edges.exists() and cached_tensor.exists():
            cache.write_audit(cfg.as_path() / "artifacts" / "cache_audit.parquet")
            tensor_rows = pd.read_parquet(cached_tensor) if cached_tensor.exists() else pd.DataFrame()
            return GraphBuildResult(
                nodes_path=cached_nodes,
                edges_path=cached_edges,
                tensor_index_path=cached_tensor,
                graph_sample_count=int(len(tensor_rows)),
            )

    phase_lookup: dict[str, dict[str, Any]] = {}
    if not phase_spec_df.empty and "phase_signature" in phase_spec_df.columns:
        for row in phase_spec_df.to_dict(orient="records"):
            phase_lookup[str(row.get("phase_signature") or "")] = row

    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(node_id: str, node_type: str, article_key: str, label: str, attrs: dict[str, Any]) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        node_rows.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "article_key": article_key,
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
        evidence_weight: float,
        provenance_ref: str,
        source_point_id: str,
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
                "evidence_weight": float(max(0.0, min(1.0, evidence_weight))),
                "provenance_ref": provenance_ref,
                "source_point_id": source_point_id,
                "created_at": now_iso(),
            }
        )

    work = frame.copy()
    work["article_key"] = work.get("article_key", pd.Series(["unknown_article"] * len(work))).astype(str)
    work["point_id"] = work.get("point_id", pd.Series([""] * len(work))).astype(str)
    work["phase_label"] = work.get("phase_label", pd.Series(["unknown"] * len(work))).astype(str)
    work["film_material"] = work.get("film_material", work["phase_label"]).astype(str)

    for article_key, group in work.groupby("article_key"):
        article_id = f"article:{article_key}"
        add_node(article_id, "article", article_key, article_key, {"article_key": article_key})
        ordered = group.sort_values(["anneal_temperature_c", "anneal_time_s", "thickness_nm"], na_position="last")

        prev_phase_instance = ""
        for idx, row in enumerate(ordered.to_dict(orient="records"), start=1):
            point_id = _as_text(row.get("point_id")) or f"{article_key}:p{idx}"
            phase_instance_id = f"phase_instance:{point_id}"
            formula = _normalize_formula(row.get("film_material") or row.get("phase_label"))
            elements, stoich = _formula_components(formula)
            sg_symbol = _normalize_spacegroup_symbol(row.get("mp_spacegroup_symbol"))
            sg_number = _normalize_spacegroup_number(row.get("mp_spacegroup_number"))
            signature = _phase_signature(
                reduced_formula=formula,
                stoich=stoich,
                spacegroup_symbol=sg_symbol,
                spacegroup_number=sg_number,
            )
            phase_meta = phase_lookup.get(signature, {})
            phase_id = str(phase_meta.get("phase_id") or f"phase_{_stable_id(signature, length=16)}")
            formula_node_id = f"phase_formula:{phase_id}"
            space_group_label = sg_symbol or sg_number or "unknown_spacegroup"
            space_group_id = f"space_group:{space_group_label or 'unknown'}"
            substrate_label = _as_text(row.get("substrate_material")) or "unknown_substrate"
            orientation_label = _as_text(row.get("substrate_orientation")) or "unknown_orientation"
            method_label = _as_text(row.get("method_family")) or "unknown_method"
            condition_payload = {
                "thickness_nm": _safe_float(row.get("thickness_nm"), 0.0),
                "anneal_temperature_c": _safe_float(row.get("anneal_temperature_c"), 0.0),
                "anneal_time_s": _safe_float(row.get("anneal_time_s"), 0.0),
                "pressure_pa": _safe_float(row.get("pressure_pa"), 0.0),
                "strain_pct": _safe_float(row.get("strain_pct"), 0.0),
                "vacancy_concentration_cm3": _safe_float(row.get("vacancy_concentration_cm3"), 0.0),
                "composition_value": _safe_float(row.get("composition_value"), 0.0),
            }
            condition_token = _stable_id(article_key, point_id, _json_dump(condition_payload), length=16)
            condition_id = f"condition_vector:{condition_token}"

            add_node(
                phase_instance_id,
                "phase_instance",
                article_key,
                _as_text(row.get("phase_label")) or formula or "unknown_phase",
                {
                    "point_id": point_id,
                    "phase_label": _as_text(row.get("phase_label")),
                    "citation_url": _as_text(row.get("citation_url")),
                    "locator": _as_text(row.get("locator")),
                },
            )
            add_node(
                formula_node_id,
                "phase_formula",
                article_key,
                formula or _as_text(row.get("phase_label")) or "unknown_formula",
                {
                    "phase_id": phase_id,
                    "phase_signature": signature,
                    "reduced_formula": formula,
                },
            )
            add_node(
                space_group_id,
                "space_group",
                article_key,
                space_group_label,
                {
                    "spacegroup_symbol": sg_symbol,
                    "spacegroup_number": sg_number,
                },
            )
            add_node(
                f"substrate:{substrate_label}",
                "substrate",
                article_key,
                substrate_label,
                {"substrate_material": substrate_label},
            )
            add_node(
                f"orientation:{orientation_label}",
                "orientation",
                article_key,
                orientation_label,
                {"substrate_orientation": orientation_label},
            )
            add_node(
                f"method_step:{method_label}",
                "method_step",
                article_key,
                method_label,
                {"method_family": method_label},
            )
            add_node(
                condition_id,
                "condition_vector",
                article_key,
                condition_token,
                condition_payload,
            )

            for element in elements:
                add_node(
                    f"element:{element}",
                    "element",
                    article_key,
                    element,
                    {"element": element},
                )
            for comp in stoich:
                elem = _as_text(comp.get("element"))
                ratio = _safe_float(comp.get("ratio"), 0.0)
                sto_id = f"stoich_component:{phase_id}:{elem}:{ratio:.4f}"
                add_node(
                    sto_id,
                    "stoich_component",
                    article_key,
                    f"{elem}:{ratio:.4f}",
                    {"element": elem, "ratio": ratio},
                )

            provenance_ref = f"{_as_text(row.get('citation_url'))}|{_as_text(row.get('locator'))}"
            point_conf = _safe_float(row.get("confidence"), 0.8)
            add_edge(article_key, article_id, phase_instance_id, "article_has_phase_instance", point_conf, provenance_ref, point_id)
            add_edge(article_key, phase_instance_id, formula_node_id, "phase_instance_has_formula", point_conf, provenance_ref, point_id)
            add_edge(article_key, phase_instance_id, space_group_id, "phase_instance_has_space_group", point_conf, provenance_ref, point_id)
            add_edge(article_key, phase_instance_id, f"substrate:{substrate_label}", "phase_instance_on_substrate", point_conf, provenance_ref, point_id)
            add_edge(article_key, phase_instance_id, f"orientation:{orientation_label}", "phase_instance_has_orientation", point_conf, provenance_ref, point_id)
            add_edge(article_key, phase_instance_id, f"method_step:{method_label}", "phase_instance_uses_method_step", point_conf, provenance_ref, point_id)
            add_edge(article_key, phase_instance_id, condition_id, "phase_instance_observed_under_condition", point_conf, provenance_ref, point_id)
            for element in elements:
                add_edge(article_key, formula_node_id, f"element:{element}", "phase_formula_has_element", point_conf, provenance_ref, point_id)
            for comp in stoich:
                elem = _as_text(comp.get("element"))
                ratio = _safe_float(comp.get("ratio"), 0.0)
                sto_id = f"stoich_component:{phase_id}:{elem}:{ratio:.4f}"
                add_edge(article_key, formula_node_id, sto_id, "phase_formula_has_stoich_component", point_conf, provenance_ref, point_id)

            if prev_phase_instance and prev_phase_instance != phase_instance_id:
                add_edge(
                    article_key,
                    prev_phase_instance,
                    phase_instance_id,
                    "phase_instance_transitions_to_phase_instance",
                    point_conf,
                    provenance_ref,
                    point_id,
                )
            prev_phase_instance = phase_instance_id

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)
    if nodes_df.empty:
        nodes_df = pd.DataFrame(columns=["node_id", "node_type", "article_key", "label", "attributes_json", "created_at"])
    if edges_df.empty:
        edges_df = pd.DataFrame(
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
    nodes_df.to_parquet(nodes_path, index=False)
    edges_df.to_parquet(edges_path, index=False)

    tensor_root = artifacts / "graph_tensors_v1"
    tensor_root.mkdir(parents=True, exist_ok=True)
    tensor_rows: list[dict[str, Any]] = []
    article_keys = sorted(set(nodes_df.get("article_key", pd.Series([], dtype=str)).astype(str).tolist()))
    for article_key in article_keys:
        ndf = nodes_df[nodes_df["article_key"].astype(str) == article_key].copy()
        edf = edges_df[edges_df["article_key"].astype(str) == article_key].copy()
        graph_sample_id = f"graph_{_stable_id(article_key, len(ndf), len(edf), length=16)}"
        tensor_path = tensor_root / f"{graph_sample_id}.npz"
        np.savez(
            tensor_path,
            node_ids=np.array(ndf["node_id"].astype(str).tolist(), dtype=object),
            node_types=np.array(ndf["node_type"].astype(str).tolist(), dtype=object),
            edge_source=np.array(edf["source_id"].astype(str).tolist(), dtype=object),
            edge_target=np.array(edf["target_id"].astype(str).tolist(), dtype=object),
            edge_type=np.array(edf["edge_type"].astype(str).tolist(), dtype=object),
        )
        tensor_rows.append(
            {
                "graph_sample_id": graph_sample_id,
                "article_key": article_key,
                "node_count": int(len(ndf)),
                "edge_count": int(len(edf)),
                "tensor_path": str(tensor_path),
                "created_at": now_iso(),
            }
        )
    tensor_df = pd.DataFrame(tensor_rows)
    if tensor_df.empty:
        tensor_df = pd.DataFrame(
            columns=["graph_sample_id", "article_key", "node_count", "edge_count", "tensor_path", "created_at"]
        )
    tensor_df.to_parquet(tensor_index_path, index=False)

    cache.put_json(
        "graph_build_index",
        key,
        {
            "nodes_path": str(nodes_path),
            "edges_path": str(edges_path),
            "tensor_index_path": str(tensor_index_path),
            "graph_sample_count": int(len(tensor_df)),
        },
        key_text=f"graph_core:{cfg.graph_core_mode}",
        ttl_hours=int(cfg.cache_ttl_hours),
    )
    cache.write_audit(cfg.as_path() / "artifacts" / "cache_audit.parquet")

    return GraphBuildResult(
        nodes_path=nodes_path,
        edges_path=edges_path,
        tensor_index_path=tensor_index_path,
        graph_sample_count=int(len(tensor_df)),
    )


def build_article_process_graph(
    points_df: pd.DataFrame | str | Path,
    cfg: RunConfig,
) -> Any:
    frame = _point_level_from_variable_rows(_load_points_frame(points_df))
    return graph_dual.build_article_process_graph(frame, cfg)


def build_global_concept_graph(
    points_df: pd.DataFrame | str | Path,
    mp_df: pd.DataFrame | None,
    cfg: RunConfig,
) -> Any:
    frame = _point_level_from_variable_rows(_load_points_frame(points_df))
    return graph_dual.build_global_concept_graph(frame, mp_df, cfg)


def build_bridge_edges(
    article_graph: Any,
    concept_graph: Any,
    cfg: RunConfig,
) -> Any:
    artifacts = cfg.as_path() / "artifacts"
    source_candidates = [
        artifacts / "processor_training_rows.parquet",
        artifacts / "phase_events.parquet",
        artifacts / "validated_points.parquet",
        artifacts / "evidence_points.parquet",
    ]
    source = next((path for path in source_candidates if path.exists()), None)
    if source is None:
        raise FileNotFoundError(
            f"missing points source for bridge build in {artifacts}; expected one of: "
            f"{', '.join(str(p.name) for p in source_candidates)}"
        )
    frame = _point_level_from_variable_rows(_load_points_frame(source))
    return graph_dual.build_bridge_edges(frame, article_graph, concept_graph, cfg)


def score_bridge_weight(edge_features: dict[str, Any], cfg: RunConfig) -> float:
    return graph_dual.score_bridge_weight(edge_features, cfg)


def audit_dual_graph(run_dir: str | Path, cfg: RunConfig) -> Any:
    return graph_dual.audit_dual_graph(run_dir, cfg)


def _hash_embedding(token: str, dim: int, seed: int) -> np.ndarray:
    digest = hashlib.sha1(f"{seed}|{token}".encode("utf-8", errors="replace")).hexdigest()
    local_seed = int(digest[:8], 16)
    rng = np.random.default_rng(local_seed)
    return rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)


def _graph_rows(frame: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "graph_sample_id",
                "article_key",
                "point_count",
                "node_counts_json",
                "edge_counts_json",
                "tensor_path",
                "created_at",
            ]
        )

    tensor_root = cfg.as_path() / "artifacts" / "graph_tensors"
    tensor_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for article_key, group in frame.groupby("article_key"):
        article_key_s = _as_text(article_key) or "unknown_article"
        materials = sorted({item for item in group.get("film_material", pd.Series([], dtype=str)).astype(str) if item})
        phases = sorted({item for item in group.get("phase_label", pd.Series([], dtype=str)).astype(str) if item})
        substrates = sorted({item for item in group.get("substrate_material", pd.Series([], dtype=str)).astype(str) if item})
        orientations = sorted({item for item in group.get("substrate_orientation", pd.Series([], dtype=str)).astype(str) if item})
        methods = sorted({item for item in group.get("method_family", pd.Series([], dtype=str)).astype(str) if item})
        dopants: set[str] = set()
        for item in group.to_dict(orient="records"):
            dopants.update(_json_list(item.get("doping_elements")))
        if not dopants:
            doping_state_values = {
                _as_text(item)
                for item in group.get("doping_state", pd.Series([], dtype=str)).astype(str).tolist()
                if _as_text(item)
            }
            if any(val and val not in {"undoped", "na_pure_ni"} for val in doping_state_values):
                dopants.update({val for val in doping_state_values if val})

        edge_counts = {
            "observed_under": int(len(group)),
            "deposited_on": int(len(group)),
            "doped_with": int(max(len(dopants), 1)),
            "annealed_to": int(max(len(methods), 1)),
            "transitions_to": int(max(0, len(phases) - 1)),
        }
        node_counts = {
            "article": 1,
            "material": max(len(materials), 1),
            "phase": max(len(phases), 1),
            "substrate": max(len(substrates), 1),
            "orientation": max(len(orientations), 1),
            "dopant": max(len(dopants), 1),
            "method_step": max(len(methods), 1),
        }

        graph_id = hashlib.sha1(f"{article_key_s}|{len(group)}".encode("utf-8", errors="replace")).hexdigest()[:20]
        tensor_path = tensor_root / f"{graph_id}.npz"
        payload = {
            "article_numeric": group[_NUMERIC_FEATURES].mean().astype(np.float32).to_numpy(),
            "materials": np.array(materials or ["unknown"], dtype=object),
            "phases": np.array(phases or ["unknown"], dtype=object),
            "substrates": np.array(substrates or ["unknown"], dtype=object),
            "orientations": np.array(orientations or ["unknown"], dtype=object),
            "dopants": np.array(sorted(dopants) or ["undoped"], dtype=object),
            "methods": np.array(methods or ["unknown"], dtype=object),
        }
        np.savez(tensor_path, **payload)

        rows.append(
            {
                "graph_sample_id": graph_id,
                "article_key": article_key_s,
                "point_count": int(len(group)),
                "node_counts_json": json.dumps(node_counts, sort_keys=True),
                "edge_counts_json": json.dumps(edge_counts, sort_keys=True),
                "tensor_path": str(tensor_path),
                "created_at": now_iso(),
            }
        )
    return pd.DataFrame(rows)


def build_processing_dataset(
    points: pd.DataFrame | str | Path,
    mp_data: pd.DataFrame | None,
    aux_data: pd.DataFrame | None,
    cfg: RunConfig,
) -> ProcessorDataset:
    points_frame = _load_points_frame(points)
    base = _point_level_from_variable_rows(points_frame)

    if mp_data is not None and not mp_data.empty:
        join_cols = [
            col
            for col in (
                "point_id",
                "mp_interpretation_score",
                "mp_formula_match_score",
                "mp_phase_match_score",
                "mp_stability_score",
                "mp_energy_above_hull_min",
                "mp_interpretation_label",
                "mp_spacegroup_symbol",
            )
            if col in mp_data.columns
        ]
        if join_cols and "point_id" in join_cols and "point_id" in base.columns:
            drop_cols = [col for col in join_cols if col != "point_id" and col in base.columns]
            base = base.drop(columns=drop_cols, errors="ignore")
            base = base.merge(mp_data[join_cols], on="point_id", how="left")

    if aux_data is not None and not aux_data.empty:
        aux = aux_data.copy()
        if "point_id" in aux.columns and "point_id" in base.columns:
            aux_cols = [col for col in aux.columns if col == "point_id" or col.startswith("aux_")]
            base = base.merge(aux[aux_cols], on="point_id", how="left")

    base = _fill_feature_columns(base)
    base = _attach_graph_features(base)
    phase_specs = build_phase_spec(base, cfg)
    phase_lookup: dict[str, dict[str, Any]] = {}
    if not phase_specs.empty:
        for row in phase_specs.to_dict(orient="records"):
            phase_lookup[str(row.get("phase_signature") or "")] = row

    phase_ids: list[str] = []
    phase_signatures: list[str] = []
    elements_json: list[str] = []
    stoich_json: list[str] = []
    sg_symbol_values: list[str] = []
    sg_number_values: list[str] = []
    for row in base.to_dict(orient="records"):
        formula = _normalize_formula(row.get("film_material") or row.get("phase_label"))
        elements, stoich = _formula_components(formula)
        sg_symbol = _normalize_spacegroup_symbol(row.get("mp_spacegroup_symbol"))
        sg_number = _normalize_spacegroup_number(row.get("mp_spacegroup_number"))
        signature = _phase_signature(
            reduced_formula=formula,
            stoich=stoich,
            spacegroup_symbol=sg_symbol,
            spacegroup_number=sg_number,
        )
        matched = phase_lookup.get(signature, {})
        phase_ids.append(str(matched.get("phase_id") or f"phase_{_stable_id(signature, length=16)}"))
        phase_signatures.append(signature)
        elements_json.append(_json_dump(elements))
        stoich_json.append(_json_dump(stoich))
        sg_symbol_values.append(sg_symbol)
        sg_number_values.append(sg_number)

    base["phase_id"] = phase_ids
    base["phase_signature"] = phase_signatures
    base["elements_json"] = elements_json
    base["stoichiometry_json"] = stoich_json
    base["spacegroup_symbol"] = sg_symbol_values
    base["spacegroup_number"] = sg_number_values
    base["graph_schema_version"] = "v1"

    dual_mode = str(cfg.graph_architecture or "").strip().lower() == "dual_concept"
    if dual_mode:
        article_graph = graph_dual.build_article_process_graph(base, cfg)
        concept_graph = graph_dual.build_global_concept_graph(base, mp_data, cfg)
        bridge_graph = graph_dual.build_bridge_edges(base, article_graph, concept_graph, cfg)
        _ = graph_dual.audit_dual_graph(cfg.as_path(), cfg)

        base["concept_bridge_count"] = 0.0
        base["concept_bridge_mean_weight"] = 0.0
        base["concept_bridge_max_weight"] = 0.0
        base["concept_gate_coverage"] = 0.0
        if bridge_graph.edges_path.exists():
            bridge_df = pd.read_parquet(bridge_graph.edges_path)
            if not bridge_df.empty and "source_point_id" in bridge_df.columns:
                grouped = bridge_df.groupby(bridge_df["source_point_id"].astype(str))
                stat_rows: list[dict[str, Any]] = []
                for point_id, group in grouped:
                    unique_gate_count = group.get("gate_type", pd.Series([], dtype=str)).astype(str).nunique()
                    stat_rows.append(
                        {
                            "point_id": str(point_id),
                            "concept_bridge_count": float(len(group)),
                            "concept_bridge_mean_weight": float(group["weight"].mean()),
                            "concept_bridge_max_weight": float(group["weight"].max()),
                            "concept_gate_coverage": float(unique_gate_count / 7.0),
                        }
                    )
                if stat_rows:
                    stats_df = pd.DataFrame(stat_rows)
                    base = base.merge(stats_df, on="point_id", how="left", suffixes=("", "_dual"))
                    for col in (
                        "concept_bridge_count",
                        "concept_bridge_mean_weight",
                        "concept_bridge_max_weight",
                        "concept_gate_coverage",
                    ):
                        if f"{col}_dual" in base.columns:
                            base[col] = (
                                base[f"{col}_dual"]
                                .map(lambda v: _safe_float(v, 0.0))
                                .where(base[f"{col}_dual"].notna(), base[col].map(lambda v: _safe_float(v, 0.0)))
                            )
                            base = base.drop(columns=[f"{col}_dual"], errors="ignore")
        base["graph_schema_version"] = "v2_dual_concept"

    artifacts = cfg.as_path() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    out_path = artifacts / "processor_training_rows.parquet"
    base.to_parquet(out_path, index=False)

    graph_result = build_article_phase_graph(base, phase_specs, cfg)
    graph_index = pd.read_parquet(graph_result.tensor_index_path)
    graph_samples_path = artifacts / "graph_samples.parquet"
    nodes_df = pd.read_parquet(graph_result.nodes_path) if graph_result.nodes_path.exists() else pd.DataFrame()
    edges_df = pd.read_parquet(graph_result.edges_path) if graph_result.edges_path.exists() else pd.DataFrame()
    sample_rows: list[dict[str, Any]] = []
    for row in graph_index.to_dict(orient="records"):
        article_key = str(row.get("article_key") or "")
        node_counts = {}
        edge_counts = {}
        point_count = int(len(base[base["article_key"].astype(str) == article_key])) if "article_key" in base.columns else 0
        if not nodes_df.empty and "article_key" in nodes_df.columns and "node_type" in nodes_df.columns:
            subset = nodes_df[nodes_df["article_key"].astype(str) == article_key]
            node_counts = subset["node_type"].astype(str).value_counts().to_dict()
        if not edges_df.empty and "article_key" in edges_df.columns and "edge_type" in edges_df.columns:
            subset = edges_df[edges_df["article_key"].astype(str) == article_key]
            edge_counts = subset["edge_type"].astype(str).value_counts().to_dict()
        sample_rows.append(
            {
                "graph_sample_id": str(row.get("graph_sample_id") or ""),
                "article_key": article_key,
                "point_count": point_count,
                "node_counts_json": _json_dump(node_counts),
                "edge_counts_json": _json_dump(edge_counts),
                "tensor_path": str(row.get("tensor_path") or ""),
                "created_at": str(row.get("created_at") or now_iso()),
            }
        )
    graph_samples = pd.DataFrame(sample_rows)
    if graph_samples.empty:
        graph_samples = pd.DataFrame(
            columns=[
                "graph_sample_id",
                "article_key",
                "point_count",
                "node_counts_json",
                "edge_counts_json",
                "tensor_path",
                "created_at",
            ]
        )
    graph_samples.to_parquet(graph_samples_path, index=False)

    return ProcessorDataset(frame=base, path=out_path)


def _split_by_article(frame: pd.DataFrame, seed: int) -> _Split:
    if frame.empty:
        return _Split(train=frame.copy(), test=frame.copy())
    if "article_key" not in frame.columns:
        shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split = int(max(1, round(len(shuffled) * 0.8)))
        train = shuffled.iloc[:split].copy()
        test = shuffled.iloc[split:].copy()
        if test.empty:
            test = train.copy()
        return _Split(train=train, test=test)

    keys = frame["article_key"].astype(str).drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    split = int(max(1, round(len(keys) * 0.8)))
    train_keys = set(keys[:split])
    train = frame[frame["article_key"].astype(str).isin(train_keys)].copy()
    test = frame[~frame["article_key"].astype(str).isin(train_keys)].copy()
    if test.empty:
        test = train.copy()
    if train.empty:
        train = test.copy()
    return _Split(train=train, test=test)


def _ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    if len(scores) == 0 or len(relevance) == 0:
        return 0.0
    order = np.argsort(-scores)
    rel_sorted = relevance[order][:k]
    gains = (2 ** rel_sorted - 1).astype(float)
    discounts = np.log2(np.arange(2, 2 + len(rel_sorted))).astype(float)
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(relevance)[::-1][:k]
    ideal_gains = (2 ** ideal - 1).astype(float)
    idcg = float(np.sum(ideal_gains / discounts[: len(ideal_gains)]))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _canonical_feature_name(raw_name: Any) -> str:
    raw = str(raw_name or "")
    base = _FEATURE_NAME_CLEAN_RE.sub("_", raw).strip("_").lower()
    if not base:
        base = "feature"
    if base[0].isdigit():
        base = f"f_{base}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{base}__{digest}"


def _canonicalize_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out.columns = [_canonical_feature_name(col) for col in out.columns]
    return out


def _prepare_matrix(frame: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = _NUMERIC_FEATURES + [
        "graph_degree_phase",
        "graph_degree_substrate",
        "graph_degree_method",
        "graph_support",
    ]
    out = frame.copy()
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = 0.0
    for col in _CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = ""

    numeric = out[numeric_cols].copy()
    cat = pd.get_dummies(out[_CATEGORICAL_FEATURES].astype(str), prefix=_CATEGORICAL_FEATURES, dtype=float)
    x = pd.concat([numeric, cat], axis=1).fillna(0.0)
    x = _canonicalize_feature_columns(x)
    if feature_columns is None:
        return x, list(x.columns)
    return x.reindex(columns=feature_columns, fill_value=0.0), list(feature_columns)


def _legacy_train_universal(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelRef:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import f1_score, mean_squared_error
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    split = _split_by_article(dataset.frame.copy(), seed=cfg.random_seed)
    x_train, feature_columns = _prepare_matrix(split.train)
    x_test, _ = _prepare_matrix(split.test, feature_columns=feature_columns)

    y_phase_train = split.train.get("phase_label", pd.Series(["unknown"] * len(split.train))).astype(str)
    y_phase_test = split.test.get("phase_label", pd.Series(["unknown"] * len(split.test))).astype(str)

    y_quality_train = split.train.get("film_quality_score_numeric", pd.Series([0.0] * len(split.train))).map(
        lambda v: _safe_float(v, 0.0)
    )
    y_quality_test = split.test.get("film_quality_score_numeric", pd.Series([0.0] * len(split.test))).map(
        lambda v: _safe_float(v, 0.0)
    )

    if len(set(y_phase_train.tolist())) <= 1:
        phase_model = DummyClassifier(strategy="most_frequent")
    else:
        phase_model = MLPClassifier(hidden_layer_sizes=(96, 48), random_state=cfg.random_seed, max_iter=300)
    quality_model = MLPRegressor(hidden_layer_sizes=(96, 48), random_state=cfg.random_seed, max_iter=300)
    if len(split.train) < 8:
        quality_model = DummyRegressor(strategy="mean")

    phase_model.fit(x_train, y_phase_train)
    quality_model.fit(x_train, y_quality_train)

    y_pred_phase = pd.Series(phase_model.predict(x_test), index=split.test.index).astype(str)
    y_pred_quality = pd.Series(quality_model.predict(x_test), index=split.test.index).map(lambda v: _safe_float(v, 0.0))
    phase_macro_f1 = float(f1_score(y_phase_test, y_pred_phase, average="macro", zero_division=0))
    quality_rmse = float(np.sqrt(mean_squared_error(y_quality_test, y_pred_quality)))

    model_dir = (
        Path(cfg.processor_model_dir).expanduser().resolve()
        if cfg.processor_model_dir
        else cfg.as_path() / "artifacts" / "models" / "universal"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "processor_model.pkl"
    metadata_path = model_dir / "processor_model_meta.json"
    metrics_path = cfg.as_path() / "artifacts" / "processor_eval_metrics.json"

    payload = {
        "phase_model": phase_model,
        "quality_model": quality_model,
        "feature_columns": feature_columns,
        "classes": sorted(set(y_phase_train.tolist() + y_phase_test.tolist())),
        "model_backend": "legacy_mlp_fallback",
    }
    with model_path.open("wb") as f:
        pickle.dump(payload, f)

    metrics = {
        "created_at": now_iso(),
        "model_type": "legacy_mlp_fallback",
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "phase_macro_f1": phase_macro_f1,
        "quality_rmse": quality_rmse,
        "quality_ndcg_at_10": 0.0,
        "dataset_path": str(dataset.path),
    }
    metadata_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ProcessorModelRef(
        model_dir=model_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        model_type="legacy_mlp_fallback",
    )


def _require_pyg_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as f
        from torch import nn
        from torch_geometric.data import HeteroData
        from torch_geometric.nn import HeteroConv, SAGEConv

        return torch, nn, f, HeteroData, HeteroConv, SAGEConv
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyG processor requires torch + torch_geometric. Install compatible versions to use processor-train-gnn."
        ) from exc


def _build_hetero_data(frame: pd.DataFrame, cfg: RunConfig) -> tuple[Any, dict[str, dict[str, int]], list[str], np.ndarray]:
    torch, _nn, _f, HeteroData, _HeteroConv, _SAGEConv = _require_pyg_dependencies()

    work = frame.copy()
    work["article_key"] = work.get("article_key", pd.Series(["unknown_article"] * len(work))).astype(str)
    work["phase_label"] = work.get("phase_label", pd.Series(["unknown"] * len(work))).astype(str)
    work["film_material"] = work.get("film_material", work["phase_label"]).astype(str)
    work["substrate_material"] = work.get("substrate_material", pd.Series(["unknown"] * len(work))).astype(str)
    work["substrate_orientation"] = work.get("substrate_orientation", pd.Series(["unknown"] * len(work))).astype(str)
    work["method_family"] = work.get("method_family", pd.Series(["unknown"] * len(work))).astype(str)
    work["spacegroup_symbol"] = work.get("spacegroup_symbol", work.get("mp_spacegroup_symbol", "")).astype(str)
    work["spacegroup_number"] = work.get("spacegroup_number", work.get("mp_spacegroup_number", "")).astype(str)

    dopant_tokens: list[str] = []
    space_group_tokens: list[str] = []
    condition_tokens: list[str] = []
    row_elements: list[list[str]] = []
    row_stoich: list[list[str]] = []
    for row in work.to_dict(orient="records"):
        elems = _json_list(row.get("doping_elements"))
        if elems:
            dopant_tokens.append("+".join(sorted(elems)))
        else:
            state = _as_text(row.get("doping_state"))
            dopant_tokens.append(state or "undoped")
        space_group = _normalize_spacegroup_symbol(row.get("spacegroup_symbol")) or _normalize_spacegroup_number(
            row.get("spacegroup_number")
        )
        space_group_tokens.append(space_group or "unknown_spacegroup")
        cond_token = (
            f"{_safe_float(row.get('thickness_nm'), 0.0):.2f}|"
            f"{_safe_float(row.get('anneal_temperature_c'), 0.0):.1f}|"
            f"{_safe_float(row.get('anneal_time_s'), 0.0):.1f}|"
            f"{_safe_float(row.get('pressure_pa'), 0.0):.1f}|"
            f"{_safe_float(row.get('strain_pct'), 0.0):.2f}"
        )
        condition_tokens.append(cond_token)

        phase_elements = _json_list(row.get("elements_json"))
        if not phase_elements:
            formula = _normalize_formula(row.get("film_material") or row.get("phase_label"))
            phase_elements, _phase_stoich = _formula_components(formula)
        phase_elements = [item for item in phase_elements if item]
        row_elements.append(phase_elements)

        stoich_items = row.get("stoichiometry_json")
        parsed_sto: list[str] = []
        if str(stoich_items or "").strip():
            try:
                payload = json.loads(str(stoich_items))
            except Exception:
                payload = []
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        elem = _as_text(item.get("element"))
                        ratio = _safe_float(item.get("ratio"), 0.0)
                        if elem:
                            parsed_sto.append(f"{elem}:{ratio:.4f}")
        if not parsed_sto:
            formula = _normalize_formula(row.get("film_material") or row.get("phase_label"))
            _, phase_sto = _formula_components(formula)
            parsed_sto = [f"{_as_text(item.get('element'))}:{_safe_float(item.get('ratio'), 0.0):.4f}" for item in phase_sto if _as_text(item.get("element"))]
        row_stoich.append(parsed_sto)
    work["_dopant"] = dopant_tokens
    work["_space_group"] = space_group_tokens
    work["_condition_vector"] = condition_tokens

    maps: dict[str, dict[str, int]] = {node: {} for node in _NODE_TYPES}
    token_columns = {
        "article": "article_key",
        "material": "film_material",
        "phase": "phase_label",
        "substrate": "substrate_material",
        "orientation": "substrate_orientation",
        "dopant": "_dopant",
        "method_step": "method_family",
        "space_group": "_space_group",
        "condition_vector": "_condition_vector",
    }
    element_tokens = sorted({item for items in row_elements for item in items if item}) or ["unknown_element"]
    stoich_tokens = sorted({item for items in row_stoich for item in items if item}) or ["unknown_stoich"]
    maps["element"] = {token: idx for idx, token in enumerate(element_tokens)}
    maps["stoich_component"] = {token: idx for idx, token in enumerate(stoich_tokens)}
    for node, col in token_columns.items():
        tokens = sorted({str(item).strip() or f"unknown_{node}" for item in work[col].astype(str).tolist()})
        maps[node] = {token: idx for idx, token in enumerate(tokens)}

    article_numeric = work.groupby("article_key")[_NUMERIC_FEATURES].mean()

    data = HeteroData()
    for node_type in _NODE_TYPES:
        dim = int(cfg.gnn_hidden_dim)
        x = np.zeros((max(1, len(maps[node_type])), dim), dtype=np.float32)
        for token, idx in maps[node_type].items():
            emb = _hash_embedding(token, dim, cfg.random_seed)
            x[idx, :] = emb
        if node_type == "article":
            for token, idx in maps[node_type].items():
                if token in article_numeric.index:
                    vals = article_numeric.loc[token].to_numpy(dtype=np.float32)
                    take = min(len(vals), dim)
                    x[idx, :take] = vals[:take]
        data[node_type].x = torch.tensor(x, dtype=torch.float32)

    def _edges(src_type: str, rel: str, dst_type: str, src_col: str, dst_col: str) -> None:
        pairs: list[tuple[int, int]] = []
        for row in work.to_dict(orient="records"):
            src_token = _as_text(row.get(src_col)) or f"unknown_{src_type}"
            dst_token = _as_text(row.get(dst_col)) or f"unknown_{dst_type}"
            src_idx = maps[src_type].get(src_token)
            dst_idx = maps[dst_type].get(dst_token)
            if src_idx is None or dst_idx is None:
                continue
            pairs.append((src_idx, dst_idx))
        if not pairs:
            return
        uniq = sorted(set(pairs))
        edge_index = torch.tensor(np.array(uniq, dtype=np.int64).T, dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_index = edge_index
        rev_edge = torch.tensor(np.array([(b, a) for a, b in uniq], dtype=np.int64).T, dtype=torch.long)
        data[(dst_type, f"rev_{rel}", src_type)].edge_index = rev_edge

    _edges("article", "observed_under", "phase", "article_key", "phase_label")
    _edges("article", "deposited_on", "substrate", "article_key", "substrate_material")
    _edges("article", "observed_under", "orientation", "article_key", "substrate_orientation")
    _edges("article", "doped_with", "dopant", "article_key", "_dopant")
    _edges("article", "annealed_to", "method_step", "article_key", "method_family")
    _edges("material", "transitions_to", "phase", "film_material", "phase_label")
    _edges("phase", "has_spacegroup", "space_group", "phase_label", "_space_group")
    _edges("phase", "observed_under_condition", "condition_vector", "phase_label", "_condition_vector")

    def _multi_edges(src_type: str, rel: str, dst_type: str, src_tokens: list[str], dst_tokens_list: list[list[str]]) -> None:
        pairs: list[tuple[int, int]] = []
        pair_count = min(len(src_tokens), len(dst_tokens_list))
        for idx in range(pair_count):
            src_token = src_tokens[idx]
            dst_tokens = dst_tokens_list[idx]
            src_idx = maps[src_type].get(_as_text(src_token))
            if src_idx is None:
                continue
            for dst_token in dst_tokens:
                dst_idx = maps[dst_type].get(_as_text(dst_token))
                if dst_idx is None:
                    continue
                pairs.append((src_idx, dst_idx))
        if not pairs:
            return
        uniq = sorted(set(pairs))
        edge_index = torch.tensor(np.array(uniq, dtype=np.int64).T, dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_index = edge_index
        rev_edge = torch.tensor(np.array([(b, a) for a, b in uniq], dtype=np.int64).T, dtype=torch.long)
        data[(dst_type, f"rev_{rel}", src_type)].edge_index = rev_edge

    phase_tokens = work["phase_label"].astype(str).tolist()
    _multi_edges("phase", "has_element", "element", phase_tokens, row_elements)
    _multi_edges("phase", "has_stoich", "stoich_component", phase_tokens, row_stoich)

    phase_pairs: list[tuple[int, int]] = []
    for _article_key, group in work.sort_values("anneal_temperature_c").groupby("article_key"):
        ordered = [maps["phase"][p] for p in group["phase_label"].astype(str).tolist() if p in maps["phase"]]
        for idx in range(max(0, len(ordered) - 1)):
            left = ordered[idx]
            right = ordered[idx + 1]
            if left != right:
                phase_pairs.append((left, right))
    if phase_pairs:
        uniq = sorted(set(phase_pairs))
        edge_index = torch.tensor(np.array(uniq, dtype=np.int64).T, dtype=torch.long)
        data[("phase", "transitions_to", "phase")].edge_index = edge_index

    classes = sorted({str(item).strip() or "unknown" for item in work["phase_label"].astype(str).tolist()})
    class_index = {label: idx for idx, label in enumerate(classes)}
    article_labels = work.groupby("article_key")["phase_label"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown")
    y = np.zeros((len(maps["article"]),), dtype=np.int64)
    for token, idx in maps["article"].items():
        label = str(article_labels.get(token, "unknown"))
        y[idx] = class_index.get(label, 0)
    data["article"].y = torch.tensor(y, dtype=torch.long)

    article_keys = list(maps["article"].keys())
    rng = np.random.default_rng(cfg.random_seed)
    perm = np.arange(len(article_keys))
    rng.shuffle(perm)
    split_idx = int(max(1, round(len(article_keys) * 0.8)))
    train_idx = set(int(item) for item in perm[:split_idx])
    test_idx = set(int(item) for item in perm[split_idx:])
    if not test_idx:
        test_idx = train_idx

    train_mask = np.array([idx in train_idx for idx in range(len(article_keys))], dtype=bool)
    test_mask = np.array([idx in test_idx for idx in range(len(article_keys))], dtype=bool)
    data["article"].train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data["article"].test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return data, maps, classes, train_mask


class _HeteroArticleClassifier:
    def __init__(
        self,
        *,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_dim: int,
        out_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        torch, nn, f, _HeteroData, HeteroConv, SAGEConv = _require_pyg_dependencies()
        self._torch = torch
        self._f = f
        self._dropout = nn.Dropout(float(dropout))
        self._convs = nn.ModuleList()
        self._linears = nn.ModuleList()
        edge_types = metadata[1]
        for _ in range(max(1, int(layers))):
            conv = HeteroConv({etype: SAGEConv((-1, -1), int(hidden_dim)) for etype in edge_types}, aggr="sum")
            self._convs.append(conv)
        self._classifier = nn.Linear(int(hidden_dim), int(out_dim))

    def to(self, device: Any) -> "_HeteroArticleClassifier":
        for conv in self._convs:
            conv.to(device)
        self._classifier.to(device)
        self._dropout.to(device)
        return self

    def parameters(self):
        params: list[Any] = []
        for conv in self._convs:
            params.extend(list(conv.parameters()))
        params.extend(list(self._classifier.parameters()))
        params.extend(list(self._dropout.parameters()))
        return params

    def train(self) -> None:
        for conv in self._convs:
            conv.train()
        self._classifier.train()
        self._dropout.train()

    def eval(self) -> None:
        for conv in self._convs:
            conv.eval()
        self._classifier.eval()
        self._dropout.eval()

    def __call__(self, data: Any) -> tuple[Any, dict[str, Any]]:
        x_dict = data.x_dict
        for conv in self._convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {key: self._f.relu(value) for key, value in x_dict.items()}
            x_dict = {key: self._dropout(value) for key, value in x_dict.items()}
        logits = self._classifier(x_dict["article"])
        return logits, x_dict

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"classifier": self._classifier.state_dict()}
        for idx, conv in enumerate(self._convs):
            payload[f"conv_{idx}"] = conv.state_dict()
        payload["dropout"] = self._dropout.state_dict()
        return payload


def train_gnn_base(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelRef:
    from sklearn.metrics import f1_score

    torch, _nn, _f, _HeteroData, _HeteroConv, _SAGEConv = _require_pyg_dependencies()

    frame = dataset.frame.copy()
    if frame.empty:
        raise ValueError("processor dataset is empty")

    data, maps, classes, _train_mask = _build_hetero_data(frame, cfg)
    metadata = data.metadata()

    model = _HeteroArticleClassifier(
        metadata=metadata,
        hidden_dim=int(cfg.gnn_hidden_dim),
        out_dim=max(1, len(classes)),
        layers=max(1, int(cfg.gnn_layers)),
        dropout=float(cfg.gnn_dropout),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.gnn_lr))
    criterion = torch.nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    epochs = max(1, int(cfg.gnn_epochs))

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits, _emb = model(data)
        mask = data["article"].train_mask
        loss = criterion(logits[mask], data["article"].y[mask])
        if torch.isnan(loss):
            raise RuntimeError("gnn training collapsed with NaN loss")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = model(data)
            pred = torch.argmax(eval_logits, dim=1)
            test_mask = data["article"].test_mask
            true_test = data["article"].y[test_mask].cpu().numpy()
            pred_test = pred[test_mask].cpu().numpy()
            if len(true_test) == 0:
                phase_macro_f1 = 0.0
            else:
                phase_macro_f1 = float(f1_score(true_test, pred_test, average="macro", zero_division=0))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(loss.detach().cpu().item()),
                "test_phase_macro_f1": phase_macro_f1,
            }
        )

    model.eval()
    with torch.no_grad():
        logits, emb = model(data)
        pred = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        test_mask = data["article"].test_mask
        true_test = data["article"].y[test_mask].cpu().numpy()
        pred_test = pred[test_mask].cpu().numpy()
        phase_macro_f1 = float(f1_score(true_test, pred_test, average="macro", zero_division=0)) if len(true_test) else 0.0

    article_tokens = list(maps["article"].keys())
    article_embeddings = emb["article"].cpu().numpy()
    emb_cols = [f"gnn_emb_{idx}" for idx in range(article_embeddings.shape[1])]
    emb_frame = pd.DataFrame(article_embeddings, columns=emb_cols)
    emb_frame.insert(0, "article_key", article_tokens)
    emb_frame["pred_class_idx"] = pred.cpu().numpy()
    emb_frame["pred_confidence"] = np.max(probs.cpu().numpy(), axis=1)

    model_dir = cfg.as_path() / "artifacts" / "models" / "gnn_base"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "gnn_model.pt"
    metadata_path = model_dir / "gnn_model_meta.json"
    metrics_path = cfg.as_path() / "artifacts" / "gnn_train_metrics.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "node_types": metadata[0],
                "edge_types": [list(item) for item in metadata[1]],
                "classes": classes,
                "gnn_hidden_dim": int(cfg.gnn_hidden_dim),
                "gnn_layers": int(cfg.gnn_layers),
                "gnn_dropout": float(cfg.gnn_dropout),
            },
        },
        model_path,
    )

    embedding_path = model_dir / "article_embeddings.parquet"
    emb_frame.to_parquet(embedding_path, index=False)

    metrics_payload = {
        "created_at": now_iso(),
        "model_type": "pyg_hetero_gnn",
        "backend": "torch_geometric",
        "train_rows": int(len(frame)),
        "article_node_count": int(len(article_tokens)),
        "phase_macro_f1": phase_macro_f1,
        "epochs": int(epochs),
        "history": history,
        "classes": classes,
        "embedding_path": str(embedding_path),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ProcessorModelRef(
        model_dir=model_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        model_type="pyg_hetero_gnn",
    )


def _select_tabular_backend(cfg: RunConfig) -> str:
    preferred = _as_text(cfg.tabular_model).lower() or "xgboost"
    if preferred == "xgboost":
        try:
            import xgboost  # noqa: F401

            return "xgboost"
        except Exception:
            pass
    if preferred in {"xgboost", "lightgbm"}:
        try:
            import lightgbm  # noqa: F401

            return "lightgbm"
        except Exception:
            pass
    return "sklearn"


def _train_tabular_models(
    *,
    train_x: pd.DataFrame,
    train_phase: pd.Series,
    train_quality: pd.Series,
    cfg: RunConfig,
) -> tuple[Any, Any, str]:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier

    backend = _select_tabular_backend(cfg)
    unique_classes = sorted(set(train_phase.astype(str).tolist()))

    if len(unique_classes) <= 1:
        classifier = DummyClassifier(strategy="most_frequent")
    elif backend == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        class_order = list(unique_classes)
        class_to_idx = {label: idx for idx, label in enumerate(class_order)}
        y_phase_idx = train_phase.astype(str).map(lambda value: class_to_idx.get(value, 0)).astype(int)
        classifier = XGBClassifier(
            n_estimators=240,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=cfg.random_seed,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=max(2, len(class_order)),
        )
        regressor = XGBRegressor(
            n_estimators=260,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=cfg.random_seed,
            objective="reg:squarederror",
        )
        classifier.fit(train_x, y_phase_idx)
        regressor.fit(train_x, train_quality)
        wrapped = _EncodedLabelClassifier(classifier, class_order)
        return wrapped, regressor, backend
    elif backend == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        classifier = LGBMClassifier(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=-1,
            random_state=cfg.random_seed,
        )
        regressor = LGBMRegressor(
            n_estimators=260,
            learning_rate=0.05,
            max_depth=-1,
            random_state=cfg.random_seed,
        )
        classifier.fit(train_x, train_phase.astype(str))
        regressor.fit(train_x, train_quality)
        return classifier, regressor, backend
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=cfg.random_seed,
            n_jobs=1,
        )

    if len(train_x) < 8:
        regressor: Any = DummyRegressor(strategy="mean")
    else:
        regressor = GradientBoostingRegressor(random_state=cfg.random_seed)

    classifier.fit(train_x, train_phase.astype(str))
    regressor.fit(train_x, train_quality)
    return classifier, regressor, backend


def _phase_probability(classifier: Any, x: pd.DataFrame, target_phase: str | None = None) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        proba = np.asarray(classifier.predict_proba(x), dtype=float)
        classes = [str(item) for item in getattr(classifier, "classes_", [])]
        if target_phase and target_phase in classes:
            idx = classes.index(target_phase)
            return np.asarray(proba[:, idx], dtype=float)
        return np.max(proba, axis=1)
    if hasattr(classifier, "predict"):
        pred = pd.Series(classifier.predict(x)).astype(str)
        if target_phase:
            return (pred == str(target_phase)).astype(float).to_numpy()
        return np.ones(len(x), dtype=float)
    return np.zeros(len(x), dtype=float)


def _inject_gnn_embeddings(frame: pd.DataFrame, gnn_ref: ProcessorModelRef) -> tuple[pd.DataFrame, list[str]]:
    embedding_path = gnn_ref.model_dir / "article_embeddings.parquet"
    if not embedding_path.exists():
        raise FileNotFoundError(f"missing GNN embedding artifact: {embedding_path}")
    emb = pd.read_parquet(embedding_path)
    emb_cols = [col for col in emb.columns if col.startswith("gnn_emb_")]
    merged = frame.merge(emb[["article_key", *emb_cols]], on="article_key", how="left")
    for col in emb_cols:
        merged[col] = merged[col].map(lambda v: _safe_float(v, 0.0))
    return merged, emb_cols


def _tabular_eval(
    *,
    frame: pd.DataFrame,
    classifier: Any,
    regressor: Any,
    feature_columns: list[str],
    target_phase: str | None = None,
) -> dict[str, float]:
    from sklearn.metrics import f1_score, mean_squared_error

    if frame.empty:
        return {
            "phase_macro_f1": 0.0,
            "quality_rmse": 0.0,
            "ndcg_at_10": 0.0,
            "row_count": 0.0,
        }
    x, _ = _prepare_matrix(frame.copy(), feature_columns=feature_columns)
    y_true_phase = frame.get("phase_label", pd.Series(["unknown"] * len(frame))).astype(str)
    y_true_quality = frame.get("film_quality_score_numeric", pd.Series([0.0] * len(frame))).map(lambda v: _safe_float(v, 0.0))

    y_pred_phase = pd.Series(classifier.predict(x), index=frame.index).astype(str)
    y_pred_quality = np.asarray(regressor.predict(x), dtype=float) if hasattr(regressor, "predict") else np.zeros(len(x))

    phase_macro_f1 = float(f1_score(y_true_phase, y_pred_phase, average="macro", zero_division=0))
    quality_rmse = float(np.sqrt(mean_squared_error(y_true_quality, y_pred_quality)))

    phase_prob = _phase_probability(classifier, x, target_phase=target_phase)
    relevance = (y_pred_phase == y_true_phase).astype(float).to_numpy()
    ndcg_at_10 = _ndcg_at_k(phase_prob, relevance, k=10)

    return {
        "phase_macro_f1": phase_macro_f1,
        "quality_rmse": quality_rmse,
        "ndcg_at_10": ndcg_at_10,
        "row_count": float(len(frame)),
    }


def train_tabular_head(dataset: ProcessorDataset, gnn_ref: ProcessorModelRef, cfg: RunConfig) -> ProcessorModelRef:
    frame = dataset.frame.copy()
    if frame.empty:
        raise ValueError("processor dataset is empty")

    merged, emb_cols = _inject_gnn_embeddings(frame, gnn_ref)
    split = _split_by_article(merged, seed=cfg.random_seed)

    x_train_base, base_feature_cols = _prepare_matrix(split.train)
    for col in emb_cols:
        if col not in x_train_base.columns:
            x_train_base[col] = split.train[col].map(lambda v: _safe_float(v, 0.0))
    feature_cols = list(x_train_base.columns)

    x_train = x_train_base.reindex(columns=feature_cols, fill_value=0.0)
    y_phase_train = split.train.get("phase_label", pd.Series(["unknown"] * len(split.train))).astype(str)
    y_quality_train = split.train.get("film_quality_score_numeric", pd.Series([0.0] * len(split.train))).map(
        lambda v: _safe_float(v, 0.0)
    )

    classifier, regressor, backend = _train_tabular_models(
        train_x=x_train,
        train_phase=y_phase_train,
        train_quality=y_quality_train,
        cfg=cfg,
    )

    test_eval = _tabular_eval(
        frame=split.test,
        classifier=classifier,
        regressor=regressor,
        feature_columns=feature_cols,
        target_phase=str(cfg.finetune_target_phase or "NiSi"),
    )

    model_dir = (
        Path(cfg.processor_model_dir).expanduser().resolve()
        if str(cfg.processor_model_dir or "").strip()
        else cfg.as_path() / "artifacts" / "models" / "tabular_head"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "processor_model.pkl"
    metadata_path = model_dir / "processor_model_meta.json"
    metrics_path = cfg.as_path() / "artifacts" / "tabular_train_metrics.json"

    payload = {
        "classifier": classifier,
        "regressor": regressor,
        "feature_columns": feature_cols,
        "embedding_columns": emb_cols,
        "model_backend": backend,
        "base_gnn_model_dir": str(gnn_ref.model_dir),
    }
    with model_path.open("wb") as f:
        pickle.dump(payload, f)

    meta_payload = {
        "created_at": now_iso(),
        "model_type": "graph_tabular_fusion",
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "dataset_path": str(dataset.path),
        "model_backend": backend,
        "base_gnn_model_dir": str(gnn_ref.model_dir),
        "feature_columns": feature_cols,
        **test_eval,
    }
    metadata_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ProcessorModelRef(
        model_dir=model_dir,
        model_path=model_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        model_type="graph_tabular_fusion",
    )


def _slice_finetune_rows(frame: pd.DataFrame, spec: FineTuneSliceSpec) -> pd.DataFrame:
    out = frame.copy()
    out["phase_label"] = out.get("phase_label", pd.Series(["unknown"] * len(out))).astype(str)
    out["thickness_nm"] = out.get("thickness_nm", pd.Series([0.0] * len(out))).map(lambda v: _safe_float(v, 0.0))

    mask = out["phase_label"].str.lower().eq(str(spec.target_phase).strip().lower()) & (
        out["thickness_nm"] <= float(spec.max_thickness_nm)
    )
    if spec.substrate_filter:
        allowed = {item.lower() for item in spec.substrate_filter}
        mask = mask & out.get("substrate_material", pd.Series([""] * len(out))).astype(str).str.lower().isin(allowed)
    if spec.doping_filter:
        doped = out.get("doping_elements", pd.Series(["[]"] * len(out))).astype(str)
        mask = mask & doped.str.lower().str.contains("|".join(item.lower() for item in spec.doping_filter), regex=True)
    return out[mask].copy()


def _load_system_finetune_frame(path_like: str | Path, *, template_frame: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    raw = Path(path_like).expanduser().resolve()
    target = raw
    if raw.is_dir():
        for candidate in (
            raw / "artifacts" / "processor_training_rows.parquet",
            raw / "processor_training_rows.parquet",
            raw / "artifacts" / "validated_points.parquet",
            raw / "validated_points.parquet",
            raw / "artifacts" / "evidence_points.parquet",
            raw / "evidence_points.parquet",
        ):
            if candidate.exists():
                target = candidate
                break
    if not target.exists():
        raise FileNotFoundError(f"system fine-tune dataset not found: {raw}")

    suffix = target.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(target)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(target, sep="," if suffix == ".csv" else "\t")
    else:
        raise ValueError(f"unsupported system fine-tune dataset format: {target}")

    if "variable_name" in frame.columns:
        frame = _point_level_from_variable_rows(frame)
    out = frame.copy()
    if "phase_label" not in out.columns:
        out["phase_label"] = out.get("entity", pd.Series(["unknown"] * len(out))).astype(str)
    if "thickness_nm" not in out.columns:
        out["thickness_nm"] = 0.0
    if "anneal_temperature_c" not in out.columns and "temperature_c" in out.columns:
        out["anneal_temperature_c"] = out["temperature_c"].map(lambda v: _safe_float(v, 0.0))

    for col in template_frame.columns:
        if col in out.columns:
            continue
        if col in _NUMERIC_FEATURES or str(template_frame.get(col).dtype).startswith(("float", "int")):
            out[col] = 0.0
        elif col == "article_key":
            out[col] = [f"system_article_{idx}" for idx in range(len(out))]
        else:
            out[col] = ""

    if "article_key" not in out.columns:
        out["article_key"] = [f"system_article_{idx}" for idx in range(len(out))]
    if "point_id" not in out.columns:
        out["point_id"] = [f"system_point_{idx}" for idx in range(len(out))]

    return out.reset_index(drop=True), target


def _split_system_slice(frame: pd.DataFrame, *, holdout_ratio: float, seed: int) -> _Split:
    if frame.empty:
        return _Split(train=frame.copy(), test=frame.copy())
    ratio = max(0.0, min(0.95, float(holdout_ratio)))
    if ratio <= 0.0:
        return _Split(train=frame.copy(), test=frame.copy())

    if "article_key" not in frame.columns:
        shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_eval = int(max(1, round(len(shuffled) * ratio)))
        n_eval = min(max(1, len(shuffled) - 1), n_eval) if len(shuffled) > 1 else 1
        test = shuffled.iloc[:n_eval].copy()
        train = shuffled.iloc[n_eval:].copy()
        if train.empty:
            train = test.copy()
        return _Split(train=train, test=test)

    keys = frame["article_key"].astype(str).drop_duplicates().tolist()
    if len(keys) <= 1:
        return _Split(train=frame.copy(), test=frame.copy())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_eval = int(max(1, round(len(keys) * ratio)))
    n_eval = min(max(1, len(keys) - 1), n_eval)
    test_keys = set(keys[:n_eval])
    test = frame[frame["article_key"].astype(str).isin(test_keys)].copy()
    train = frame[~frame["article_key"].astype(str).isin(test_keys)].copy()
    if train.empty:
        train = test.copy()
    if test.empty:
        test = train.copy()
    return _Split(train=train, test=test)


def finetune_nisi_sub200(
    model_ref: ProcessorModelRef,
    dataset: ProcessorDataset,
    cfg: RunConfig,
) -> ProcessorModelRef:
    slice_spec = FineTuneSliceSpec(
        target_phase=str(cfg.finetune_target_phase or "NiSi"),
        max_thickness_nm=float(cfg.finetune_max_thickness_nm or 200.0),
    )

    frame = dataset.frame.copy()
    finetune_source = "run_universal_dataset"
    finetune_source_path = ""
    if str(cfg.system_finetune_dataset or "").strip():
        system_frame, source_path = _load_system_finetune_frame(
            cfg.system_finetune_dataset,
            template_frame=frame,
        )
        finetune_source = "external_system_dataset"
        finetune_source_path = str(source_path)
    else:
        system_frame = frame.copy()
    slice_df = _slice_finetune_rows(system_frame, slice_spec)

    artifacts = cfg.as_path() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    finetune_slice_path = artifacts / "finetune_slice.parquet"
    slice_df.to_parquet(finetune_slice_path, index=False)
    if slice_df.empty:
        raise ValueError("finetune slice is empty for target constraints")

    split = _split_system_slice(
        slice_df,
        holdout_ratio=float(cfg.system_eval_holdout_ratio or 0.0),
        seed=int(cfg.random_seed),
    )
    train_slice = split.train if not split.train.empty else slice_df.copy()
    eval_slice = split.test if not split.test.empty else slice_df.copy()
    finetune_train_path = artifacts / "finetune_system_train.parquet"
    finetune_eval_path = artifacts / "finetune_system_eval.parquet"
    train_slice.to_parquet(finetune_train_path, index=False)
    eval_slice.to_parquet(finetune_eval_path, index=False)

    base_bundle = load_processor_model(model_ref.model_dir)
    gnn_dir_raw = _as_text(base_bundle.get("base_gnn_model_dir"))
    if not gnn_dir_raw:
        raise ValueError(
            "base model does not contain `base_gnn_model_dir`; use a tabular model created by processor-train-tabular."
        )
    gnn_dir = Path(gnn_dir_raw).expanduser().resolve()
    if not gnn_dir.exists():
        raise FileNotFoundError("base tabular model metadata missing base_gnn_model_dir")
    gnn_ref = ProcessorModelRef(
        model_dir=gnn_dir,
        model_path=gnn_dir / "gnn_model.pt",
        metadata_path=gnn_dir / "gnn_model_meta.json",
        metrics_path=cfg.as_path() / "artifacts" / "gnn_train_metrics.json",
        model_type="pyg_hetero_gnn",
    )

    boosted = pd.concat([frame, train_slice, train_slice], ignore_index=True)
    boosted_dataset = ProcessorDataset(frame=boosted, path=dataset.path)

    tuned_cfg = RunConfig(**{**cfg.__dict__})
    finetune_dir = cfg.as_path() / "artifacts" / "models" / "finetune_nisi_sub200"
    tuned_cfg.processor_model_dir = str(finetune_dir)

    tuned_ref = train_tabular_head(boosted_dataset, gnn_ref, tuned_cfg)
    target_metrics = cfg.as_path() / "artifacts" / "processor_eval_finetune.json"

    meta_payload = json.loads(Path(tuned_ref.metadata_path).read_text(encoding="utf-8"))
    meta_payload["finetune_target_phase"] = slice_spec.target_phase
    meta_payload["finetune_max_thickness_nm"] = slice_spec.max_thickness_nm
    meta_payload["finetune_slice_rows"] = int(len(slice_df))
    meta_payload["finetune_slice_path"] = str(finetune_slice_path)
    meta_payload["finetune_source"] = finetune_source
    meta_payload["finetune_source_path"] = finetune_source_path
    meta_payload["finetune_train_rows"] = int(len(train_slice))
    meta_payload["finetune_eval_rows"] = int(len(eval_slice))
    meta_payload["finetune_train_path"] = str(finetune_train_path)
    meta_payload["finetune_eval_path"] = str(finetune_eval_path)
    meta_payload["system_eval_holdout_ratio"] = float(cfg.system_eval_holdout_ratio or 0.0)
    Path(tuned_ref.metadata_path).write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target_metrics.write_text(json.dumps(meta_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return ProcessorModelRef(
        model_dir=finetune_dir,
        model_path=Path(tuned_ref.model_path),
        metadata_path=Path(tuned_ref.metadata_path),
        metrics_path=target_metrics,
        model_type="graph_tabular_fusion_finetune",
    )


def load_processor_model(model_dir_or_file: str | Path) -> dict[str, Any]:
    target = Path(model_dir_or_file).expanduser().resolve()
    if target.is_dir():
        target = target / "processor_model.pkl"
    if not target.exists():
        raise FileNotFoundError(f"processor model not found: {target}")
    with target.open("rb") as f:
        return pickle.load(f)


def predict_with_processor(
    candidates: pd.DataFrame,
    model_dir_or_file: str | Path,
    target_phase: str,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    bundle = load_processor_model(model_dir_or_file)
    classifier = bundle.get("classifier") or bundle.get("phase_model")
    regressor = bundle.get("regressor") or bundle.get("quality_model")
    feature_columns = list(bundle.get("feature_columns") or [])

    x, _ = _prepare_matrix(candidates.copy(), feature_columns=feature_columns)
    phase_prob = _phase_probability(classifier, x, target_phase=str(target_phase or ""))
    quality_pred = (
        np.asarray(regressor.predict(x), dtype=float)
        if regressor is not None and hasattr(regressor, "predict")
        else np.zeros(len(x), dtype=float)
    )
    quality_pred = np.clip(quality_pred, 0.0, 1.0)

    out = candidates.copy()
    out["processor_phase_prob"] = phase_prob
    out["processor_quality_score"] = quality_pred
    out["processor_uncertainty"] = np.clip(1.0 - (0.65 * phase_prob + 0.35 * quality_pred), 0.0, 1.0)
    out["phase_prob"] = out["processor_phase_prob"]
    out["film_quality_score_numeric"] = out["processor_quality_score"]
    out["uncertainty"] = out["processor_uncertainty"]
    return out


def _evaluate_model_on_frame(
    *,
    frame: pd.DataFrame,
    model_path: Path,
    target_phase: str,
) -> dict[str, float]:
    bundle = load_processor_model(model_path)
    classifier = bundle.get("classifier") or bundle.get("phase_model")
    regressor = bundle.get("regressor") or bundle.get("quality_model")
    feature_columns = list(bundle.get("feature_columns") or [])
    return _tabular_eval(
        frame=frame,
        classifier=classifier,
        regressor=regressor,
        feature_columns=feature_columns,
        target_phase=target_phase,
    )


def _policy_gate_payload(
    *,
    cfg: RunConfig,
    base_eval: dict[str, float],
    base_slice: dict[str, float],
    finetune_eval: dict[str, float],
    slice_rows: int,
    support_articles: int,
) -> tuple[dict[str, Any], dict[str, float], str]:
    f1_uplift = float(finetune_eval.get("phase_macro_f1", 0.0) - base_slice.get("phase_macro_f1", 0.0))
    ndcg_uplift = float(finetune_eval.get("ndcg_at_10", 0.0) - base_slice.get("ndcg_at_10", 0.0))
    policy = str(cfg.finetune_gate_policy or "absolute_uplift").strip().lower()
    if policy not in {"absolute_uplift", "ceiling_aware", "non_degradation"}:
        policy = "absolute_uplift"

    base_f1 = float(base_slice.get("phase_macro_f1", 0.0))
    base_ndcg = float(base_slice.get("ndcg_at_10", 0.0))
    tuned_f1 = float(finetune_eval.get("phase_macro_f1", 0.0))
    tuned_ndcg = float(finetune_eval.get("ndcg_at_10", 0.0))
    threshold = float(cfg.finetune_ceiling_threshold or 0.95)
    min_rows = max(1, int(cfg.finetune_min_slice_rows or 1))
    min_articles = max(1, int(cfg.finetune_min_support_articles or 1))

    if policy == "absolute_uplift":
        branch = "uplift_required"
        f1_gate = bool(f1_uplift >= 0.05)
        ndcg_gate = bool(ndcg_uplift >= 0.05)
    elif policy == "non_degradation":
        branch = "non_degradation_required"
        f1_gate = bool(tuned_f1 >= (base_f1 - 1e-9))
        ndcg_gate = bool(tuned_ndcg >= (base_ndcg - 1e-9))
    else:
        f1_at_ceiling = bool(base_f1 >= threshold)
        ndcg_at_ceiling = bool(base_ndcg >= threshold)
        if f1_at_ceiling and ndcg_at_ceiling:
            branch = "ceiling_non_degradation"
        elif f1_at_ceiling or ndcg_at_ceiling:
            branch = "ceiling_mixed"
        else:
            branch = "ceiling_uplift_required"
        f1_gate = bool(tuned_f1 >= (base_f1 - 1e-9)) if f1_at_ceiling else bool(f1_uplift >= 0.05)
        ndcg_gate = bool(tuned_ndcg >= (base_ndcg - 1e-9)) if ndcg_at_ceiling else bool(ndcg_uplift >= 0.05)

    gates = {
        "base_phase_macro_f1_ge_0_80": bool(base_eval.get("phase_macro_f1", 0.0) >= 0.80),
        "base_ndcg_at_10_ge_0_75": bool(base_eval.get("ndcg_at_10", 0.0) >= 0.75),
        "finetune_support_rows_gate": bool(slice_rows >= min_rows),
        "finetune_support_articles_gate": bool(support_articles >= min_articles),
        "finetune_f1_policy_gate": bool(f1_gate),
        "finetune_ndcg_policy_gate": bool(ndcg_gate),
    }
    uplift = {"phase_macro_f1": f1_uplift, "ndcg_at_10": ndcg_uplift}
    return gates, uplift, branch


def evaluate_processor_with_policy(run_dir: str | Path, cfg: RunConfig) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = run_path / "artifacts"
    dataset_path = artifacts / "processor_training_rows.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing processor training rows: {dataset_path}")
    frame = pd.read_parquet(dataset_path)

    base_model_path = artifacts / "models" / "tabular_head" / "processor_model.pkl"
    finetune_model_path = artifacts / "models" / "finetune_nisi_sub200" / "processor_model.pkl"
    if not base_model_path.exists():
        raise FileNotFoundError(f"missing base tabular model: {base_model_path}")
    if not finetune_model_path.exists():
        raise FileNotFoundError(f"missing finetune model: {finetune_model_path}")

    target_phase = str(cfg.finetune_target_phase or "NiSi")
    base_eval = _evaluate_model_on_frame(frame=frame, model_path=base_model_path, target_phase=target_phase)
    slice_spec = FineTuneSliceSpec(
        target_phase=target_phase,
        max_thickness_nm=float(cfg.finetune_max_thickness_nm or 200.0),
    )
    eval_source = "run_slice_fallback"
    eval_source_path = str(dataset_path)
    eval_frame = pd.DataFrame()

    stored_eval_path = artifacts / "finetune_system_eval.parquet"
    if stored_eval_path.exists():
        try:
            candidate = pd.read_parquet(stored_eval_path)
            if not candidate.empty:
                eval_frame = candidate
                eval_source = "finetune_system_eval_artifact"
                eval_source_path = str(stored_eval_path)
        except Exception:
            eval_frame = pd.DataFrame()

    if eval_frame.empty and str(cfg.system_finetune_dataset or "").strip():
        try:
            system_frame, source_path = _load_system_finetune_frame(
                cfg.system_finetune_dataset,
                template_frame=frame,
            )
            system_slice = _slice_finetune_rows(system_frame, slice_spec)
            split = _split_system_slice(
                system_slice,
                holdout_ratio=float(cfg.system_eval_holdout_ratio or 0.0),
                seed=int(cfg.random_seed),
            )
            candidate = split.test if not split.test.empty else system_slice
            if not candidate.empty:
                eval_frame = candidate
                eval_source = "external_system_holdout"
                eval_source_path = str(source_path)
        except Exception:
            eval_frame = pd.DataFrame()

    if eval_frame.empty:
        eval_frame = _slice_finetune_rows(frame, slice_spec)

    base_slice = _evaluate_model_on_frame(frame=eval_frame, model_path=base_model_path, target_phase=target_phase)
    finetune_eval = _evaluate_model_on_frame(frame=eval_frame, model_path=finetune_model_path, target_phase=target_phase)
    support_articles = int(eval_frame.get("article_key", pd.Series([], dtype=str)).astype(str).nunique()) if not eval_frame.empty else 0
    gates, uplift, policy_branch = _policy_gate_payload(
        cfg=cfg,
        base_eval=base_eval,
        base_slice=base_slice,
        finetune_eval=finetune_eval,
        slice_rows=int(len(eval_frame)),
        support_articles=support_articles,
    )

    payload = {
        "created_at": now_iso(),
        "target_phase": target_phase,
        "slice_rows": int(len(eval_frame)),
        "support_articles": support_articles,
        "eval_source": eval_source,
        "eval_source_path": eval_source_path,
        "gate_policy": str(cfg.finetune_gate_policy or "absolute_uplift"),
        "gate_policy_branch": policy_branch,
        "gate_policy_thresholds": {
            "ceiling_threshold": float(cfg.finetune_ceiling_threshold or 0.95),
            "min_slice_rows": int(cfg.finetune_min_slice_rows or 0),
            "min_support_articles": int(cfg.finetune_min_support_articles or 0),
        },
        "base": base_eval,
        "base_slice": base_slice,
        "finetune_slice": finetune_eval,
        "uplift": uplift,
        "gates": gates,
    }
    payload["all_gates_pass"] = bool(all(payload["gates"].values()))

    base_out = artifacts / "processor_eval_base.json"
    finetune_out = artifacts / "processor_eval_finetune.json"
    combined_out = artifacts / "processor_eval_metrics.json"
    explain_out = artifacts / "processor_gate_explain_v2.md"

    base_out.write_text(json.dumps(payload["base"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finetune_out.write_text(
        json.dumps({"base_slice": payload["base_slice"], "finetune_slice": payload["finetune_slice"], "uplift": payload["uplift"]}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    combined_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    explain_out.write_text(
        "\n".join(
            [
                "# Processor Gate Explain (v2)",
                "",
                f"- policy: `{payload['gate_policy']}`",
                f"- branch: `{payload['gate_policy_branch']}`",
                f"- slice_rows: `{payload['slice_rows']}`",
                f"- support_articles: `{payload['support_articles']}`",
                f"- all_gates_pass: `{payload['all_gates_pass']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def evaluate_processor(run_dir: str | Path, cfg: RunConfig) -> dict[str, Any]:
    return evaluate_processor_with_policy(run_dir=run_dir, cfg=cfg)


def train_universal_processor(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelRef:
    try:
        gnn_ref = train_gnn_base(dataset, cfg)
        return train_tabular_head(dataset, gnn_ref, cfg)
    except RuntimeError as exc:
        # Compatibility wrapper keeps legacy pipeline usable when PyG is unavailable.
        if "torch + torch_geometric" not in str(exc):
            raise
        return _legacy_train_universal(dataset, cfg)


def train_processor_v1(dataset: ProcessorDataset, cfg: RunConfig) -> ProcessorModelBundle:
    gnn_ref = train_gnn_base(dataset, cfg)
    tabular_ref = train_tabular_head(dataset, gnn_ref, cfg)
    finetune_ref = finetune_nisi_sub200(tabular_ref, dataset, cfg)
    eval_metrics = evaluate_processor(run_dir=cfg.as_path(), cfg=cfg)
    return ProcessorModelBundle(
        gnn_ref=gnn_ref,
        tabular_ref=tabular_ref,
        finetune_ref=finetune_ref,
        eval_metrics=eval_metrics,
    )


def finetune_processor_for_system(
    model_ref: ProcessorModelRef,
    system_dataset: ProcessorDataset,
    cfg: RunConfig,
    system_id: str,
) -> ProcessorModelRef:
    token = _as_text(system_id)
    if token:
        cfg.finetune_target_phase = token
    try:
        return finetune_nisi_sub200(model_ref, system_dataset, cfg)
    except Exception:
        # Compatibility fallback for old system fine-tuning flows.
        return _legacy_train_universal(system_dataset, cfg)
