from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from oarp.models import ConsensusSet, RunConfig
from oarp.plugins.base import ConsensusModel
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    upsert_artifact,
)
from oarp.topic_spec import TopicSpec


def _shannon_entropy(probabilities: list[float]) -> float:
    total = 0.0
    for p in probabilities:
        if p <= 0:
            continue
        total -= p * math.log(p, 2)
    return total


def _point_level_frame(validated: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:
    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y

    points: list[dict[str, Any]] = []
    for point_id, group in validated.groupby("point_id"):
        x_rows = group[group["variable_name"] == x_name]
        y_rows = group[group["variable_name"] == y_name]
        if x_rows.empty or y_rows.empty:
            continue
        x_val = float(x_rows.iloc[0]["normalized_value"])
        y_val = float(y_rows.iloc[0]["normalized_value"])
        entity = ""
        for item in group["entity"].tolist():
            text = str(item).strip()
            if text:
                entity = text
                break
        entity = entity or "unknown"
        confidence = float(np.mean(group["confidence"].astype(float).values))
        citation_url = str(group.iloc[0].get("citation_url") or "")
        doi = str(group.iloc[0].get("doi") or "")
        locator = str(group.iloc[0].get("locator") or "")
        snippet = str(group.iloc[0].get("snippet") or "")
        substrate_material = str(group.iloc[0].get("substrate_material") or "")
        substrate_orientation = str(group.iloc[0].get("substrate_orientation") or "")
        orientation_family = str(group.iloc[0].get("orientation_family") or "")
        doping_state = str(group.iloc[0].get("doping_state") or "")
        doping_elements = str(group.iloc[0].get("doping_elements") or "[]")
        doping_composition = str(group.iloc[0].get("doping_composition") or "[]")
        alloy_state = str(group.iloc[0].get("alloy_state") or "")
        alloy_elements = str(group.iloc[0].get("alloy_elements") or "[]")
        alloy_composition = str(group.iloc[0].get("alloy_composition") or "[]")
        context_confidence = float(group.iloc[0].get("context_confidence") or 0.0)
        pure_ni_evidence = bool(group.iloc[0].get("pure_ni_evidence") or False)
        mp_status = str(group.iloc[0].get("mp_status") or "")
        mp_query_key = str(group.iloc[0].get("mp_query_key") or "")
        mp_material_ids = str(group.iloc[0].get("mp_material_ids") or "[]")
        mp_best_material_id = str(group.iloc[0].get("mp_best_material_id") or "")
        mp_formula_match_score = float(group.iloc[0].get("mp_formula_match_score") or 0.0)
        mp_phase_match_score = float(group.iloc[0].get("mp_phase_match_score") or 0.0)
        mp_stability_score = float(group.iloc[0].get("mp_stability_score") or 0.0)
        mp_interpretation_score = float(group.iloc[0].get("mp_interpretation_score") or 0.0)
        mp_interpretation_label = str(group.iloc[0].get("mp_interpretation_label") or "")
        mp_energy_above_hull_min = float(group.iloc[0].get("mp_energy_above_hull_min") or 0.0)
        mp_spacegroup_symbol = str(group.iloc[0].get("mp_spacegroup_symbol") or "")
        mp_spacegroup_number = str(group.iloc[0].get("mp_spacegroup_number") or "")
        mp_conflict_reason = str(group.iloc[0].get("mp_conflict_reason") or "")
        points.append(
            {
                "point_id": point_id,
                "x_name": x_name,
                "x_value": x_val,
                "y_name": y_name,
                "y_value": y_val,
                "entity": entity,
                "confidence": confidence,
                "citation_url": citation_url,
                "doi": doi,
                "locator": locator,
                "snippet": snippet,
                "substrate_material": substrate_material,
                "substrate_orientation": substrate_orientation,
                "orientation_family": orientation_family,
                "doping_state": doping_state,
                "doping_elements": doping_elements,
                "doping_composition": doping_composition,
                "alloy_state": alloy_state,
                "alloy_elements": alloy_elements,
                "alloy_composition": alloy_composition,
                "context_confidence": context_confidence,
                "pure_ni_evidence": pure_ni_evidence,
                "mp_status": mp_status,
                "mp_query_key": mp_query_key,
                "mp_material_ids": mp_material_ids,
                "mp_best_material_id": mp_best_material_id,
                "mp_formula_match_score": mp_formula_match_score,
                "mp_phase_match_score": mp_phase_match_score,
                "mp_stability_score": mp_stability_score,
                "mp_interpretation_score": mp_interpretation_score,
                "mp_interpretation_label": mp_interpretation_label,
                "mp_energy_above_hull_min": mp_energy_above_hull_min,
                "mp_spacegroup_symbol": mp_spacegroup_symbol,
                "mp_spacegroup_number": mp_spacegroup_number,
                "mp_conflict_reason": mp_conflict_reason,
            }
        )
    return pd.DataFrame(points)


def _assign_outliers(frame: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    out = frame.copy()
    out["is_outlier"] = False
    out["outlier_score"] = 0.0

    if len(out) < 5:
        return out

    try:
        from sklearn.ensemble import IsolationForest  # type: ignore

        clf = IsolationForest(contamination=0.15, random_state=random_seed)
        feat = out[["x_value", "y_value"]].values
        pred = clf.fit_predict(feat)
        score = clf.decision_function(feat)
        out["is_outlier"] = pred == -1
        out["outlier_score"] = -score
        return out
    except Exception:
        median = float(out["y_value"].median())
        mad = float(np.median(np.abs(out["y_value"] - median))) or 1e-6
        robust_z = 0.6745 * (out["y_value"] - median) / mad
        out["is_outlier"] = np.abs(robust_z) > 3.5
        out["outlier_score"] = np.abs(robust_z)
        return out


class ConfidenceConsensusModel:
    model_name = "confidence-consensus"

    def score(self, points: list[dict], spec: TopicSpec, cfg: RunConfig) -> list[dict]:
        frame = pd.DataFrame(points)
        if frame.empty:
            return []

        frame["x_bin"] = frame["x_value"].round(3)
        frame["context_signature"] = frame.apply(
            lambda row: "|".join(
                [
                    str(row.get("substrate_material") or ""),
                    str(row.get("substrate_orientation") or ""),
                    str(row.get("doping_state") or ""),
                    str(row.get("alloy_state") or ""),
                    str(row.get("doping_elements") or ""),
                    str(row.get("alloy_elements") or ""),
                ]
            ),
            axis=1,
        )
        frame = _assign_outliers(frame, cfg.random_seed)

        consensus_rows: list[dict[str, Any]] = []
        by_bin = frame.groupby(["x_bin", "context_signature"])
        for (_x_bin, context_signature), group in by_bin:
            support_count = int(len(group))
            if support_count <= 0:
                continue

            weights = group.groupby("entity")["confidence"].sum()
            total = float(weights.sum()) or 1.0
            probs = [float(item) / total for item in weights.tolist()]
            entropy = _shannon_entropy(probs)
            consensus_entity = str(weights.idxmax())

            consensus_group = group[group["entity"] == consensus_entity]
            if consensus_group.empty:
                continue
            y = consensus_group["y_value"].astype(float)
            w = consensus_group["confidence"].astype(float)
            weighted_mean = float(np.average(y, weights=w))
            y_std = float(np.std(y)) if len(y) > 1 else 0.0

            low_support = support_count < max(1, int(cfg.min_support_per_bin))
            support_penalty = 1.0 - float(cfg.low_n_confidence_penalty) if low_support else 1.0

            for _, row in group.iterrows():
                is_consensus = str(row["entity"]) == consensus_entity
                is_outlier = bool(row["is_outlier"])
                if is_outlier:
                    alpha = 0.15
                elif is_consensus:
                    alpha = 0.9
                else:
                    alpha = 0.35

                base_conf = float(row["confidence"])
                base_model_confidence = base_conf * (1.0 / (1.0 + entropy)) * support_penalty
                mp_score = float(row.get("mp_interpretation_score") or 0.0)
                model_confidence = base_model_confidence * (0.90 + 0.20 * mp_score)
                model_confidence = max(0.0, min(1.0, model_confidence))
                mp_label = str(row.get("mp_interpretation_label") or "").strip().lower()
                if mp_label == "conflicts" and not is_outlier:
                    alpha = min(alpha, 0.20)

                consensus_rows.append(
                    {
                        "point_id": str(row["point_id"]),
                        "x_name": str(row["x_name"]),
                        "x_value": float(row["x_value"]),
                        "y_name": str(row["y_name"]),
                        "y_value": float(row["y_value"]),
                        "entity": str(row["entity"]),
                        "consensus_entity": consensus_entity,
                        "consensus_y": weighted_mean,
                        "consensus_y_std": y_std,
                        "entropy": entropy,
                        "support_count": support_count,
                        "low_support": low_support,
                        "support_penalty": support_penalty,
                        "base_confidence": base_conf,
                        "base_model_confidence": base_model_confidence,
                        "model_confidence": model_confidence,
                        "is_outlier": is_outlier,
                        "outlier_score": float(row["outlier_score"]),
                        "display_alpha": alpha,
                        "citation_url": str(row["citation_url"]),
                        "doi": str(row["doi"]),
                        "locator": str(row["locator"]),
                        "snippet": str(row["snippet"]),
                        "context_signature": context_signature,
                        "substrate_material": str(row.get("substrate_material") or ""),
                        "substrate_orientation": str(row.get("substrate_orientation") or ""),
                        "orientation_family": str(row.get("orientation_family") or ""),
                        "doping_state": str(row.get("doping_state") or ""),
                        "doping_elements": str(row.get("doping_elements") or "[]"),
                        "doping_composition": str(row.get("doping_composition") or "[]"),
                        "alloy_state": str(row.get("alloy_state") or ""),
                        "alloy_elements": str(row.get("alloy_elements") or "[]"),
                        "alloy_composition": str(row.get("alloy_composition") or "[]"),
                        "context_confidence": float(row.get("context_confidence") or 0.0),
                        "pure_ni_evidence": bool(row.get("pure_ni_evidence") or False),
                        "mp_status": str(row.get("mp_status") or ""),
                        "mp_query_key": str(row.get("mp_query_key") or ""),
                        "mp_material_ids": str(row.get("mp_material_ids") or "[]"),
                        "mp_best_material_id": str(row.get("mp_best_material_id") or ""),
                        "mp_formula_match_score": float(row.get("mp_formula_match_score") or 0.0),
                        "mp_phase_match_score": float(row.get("mp_phase_match_score") or 0.0),
                        "mp_stability_score": float(row.get("mp_stability_score") or 0.0),
                        "mp_interpretation_score": float(row.get("mp_interpretation_score") or 0.0),
                        "mp_interpretation_label": str(row.get("mp_interpretation_label") or ""),
                        "mp_energy_above_hull_min": float(row.get("mp_energy_above_hull_min") or 0.0),
                        "mp_spacegroup_symbol": str(row.get("mp_spacegroup_symbol") or ""),
                        "mp_spacegroup_number": str(row.get("mp_spacegroup_number") or ""),
                        "mp_conflict_reason": str(row.get("mp_conflict_reason") or ""),
                    }
                )

        return consensus_rows


def build(*, spec: TopicSpec, cfg: RunConfig, model: ConsensusModel | None = None) -> ConsensusSet:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]

    validated_path = artifacts / "validated_points.parquet"
    if not validated_path.exists():
        raise FileNotFoundError(f"missing validated points: {validated_path}")

    validated = pd.read_parquet(validated_path)
    point_frame = _point_level_frame(validated, spec)

    selected_model = model or ConfidenceConsensusModel()
    rows = selected_model.score(point_frame.to_dict(orient="records"), spec, cfg)
    consensus = pd.DataFrame(rows)
    if consensus.empty:
        consensus = pd.DataFrame(
            columns=[
                "point_id",
                "x_name",
                "x_value",
                "y_name",
                "y_value",
                "entity",
                "consensus_entity",
                "consensus_y",
                "consensus_y_std",
                "entropy",
                "support_count",
                "low_support",
                "support_penalty",
                "base_confidence",
                "base_model_confidence",
                "model_confidence",
                "is_outlier",
                "outlier_score",
                "display_alpha",
                "citation_url",
                "doi",
                "locator",
                "snippet",
                "context_signature",
                "substrate_material",
                "substrate_orientation",
                "orientation_family",
                "doping_state",
                "doping_elements",
                "doping_composition",
                "alloy_state",
                "alloy_elements",
                "alloy_composition",
                "context_confidence",
                "pure_ni_evidence",
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
            ]
        )

    path = artifacts / "consensus_points.parquet"
    consensus.to_parquet(path, index=False)

    summary = {
        "point_count": int(len(consensus)),
        "consensus_entity_count": int(consensus["consensus_entity"].nunique()) if not consensus.empty else 0,
        "outlier_count": int(consensus["is_outlier"].sum()) if not consensus.empty else 0,
        "mean_entropy": float(consensus["entropy"].mean()) if not consensus.empty else 0.0,
        "low_support_bin_count": int(consensus["low_support"].sum()) if not consensus.empty else 0,
        "mean_model_confidence": float(consensus["model_confidence"].mean()) if not consensus.empty else 0.0,
        "mp_support_rate": float(
            (
                consensus["mp_interpretation_label"].astype(str).str.strip().str.lower() == "supports"
            ).mean()
        )
        if not consensus.empty and "mp_interpretation_label" in consensus.columns
        else 0.0,
        "mp_conflict_rate": float(
            (
                consensus["mp_interpretation_label"].astype(str).str.strip().str.lower() == "conflicts"
            ).mean()
        )
        if not consensus.empty and "mp_interpretation_label" in consensus.columns
        else 0.0,
        "mp_coverage": float(
            consensus["mp_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"success", "no_match", "fetch_error", "parse_error", "no_api_key", "disabled"})
            .mean()
        )
        if not consensus.empty and "mp_status" in consensus.columns
        else 0.0,
    }

    state = load_run_state(cfg.as_path())
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="consensus_points", path=path)
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="consensus",
        source_name="validated_points.parquet",
        target_name="consensus_points.parquet",
    )

    return ConsensusSet(points=consensus, summary=summary, parquet_path=path)
