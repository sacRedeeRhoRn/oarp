from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oarp.context import ContextValidator
from oarp.models import ExtractionCalibration, RunConfig, ValidatedEvidence
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
)
from oarp.topic_spec import TopicSpec, VariableSpec


def _point_has_required_variables(group: pd.DataFrame, x_name: str, y_name: str) -> bool:
    names = {str(item).strip() for item in group["variable_name"].tolist()}
    return x_name in names and y_name in names


def _variable_map(spec: TopicSpec) -> dict[str, VariableSpec]:
    return {var.name: var for var in spec.variables}


def _value_in_range(variable: VariableSpec, value: float) -> bool:
    if variable.min_value is not None and value < float(variable.min_value):
        return False
    if variable.max_value is not None and value > float(variable.max_value):
        return False
    name = variable.name.lower()
    if "thickness" in name and value < 0:
        return False
    if "temperature" in name and value < -273.2:
        return False
    return True


def _build_reason_table(rejected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if rejected.empty:
        return pd.DataFrame(columns=["reason", "count", "extraction_type", "created_at"])

    for _, row in rejected.iterrows():
        reason_text = str(row.get("reject_reason") or "").strip()
        extraction_type = str(row.get("extraction_type") or "")
        reasons = [item.strip() for item in reason_text.split(";") if item.strip()]
        for reason in reasons:
            rows.append(
                {
                    "reason": reason,
                    "extraction_type": extraction_type,
                    "created_at": now_iso(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["reason", "count", "extraction_type", "created_at"])

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["reason", "extraction_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["count", "reason"], ascending=[False, True])
    )
    grouped["created_at"] = now_iso()
    return grouped


def _write_validation_metrics(path: Path, accepted: pd.DataFrame, rejected: pd.DataFrame, reasons: pd.DataFrame) -> None:
    payload = {
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "acceptance_ratio": float(len(accepted) / max(len(accepted) + len(rejected), 1)),
        "top_reasons": reasons.head(15).to_dict(orient="records"),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extractor_gate_payload(
    *,
    accepted: pd.DataFrame,
    extraction_calibration: pd.DataFrame,
) -> dict[str, object]:
    if accepted.empty:
        provenance_complete_ratio = 0.0
        context_precision_proxy = 0.0
    else:
        required = ["citation_url", "snippet", "locator"]
        required_present = all(col in accepted.columns for col in required)
        if required_present:
            provenance_mask = accepted[required].fillna("").astype(str).apply(
                lambda col: col.str.strip().ne(""),
                axis=0,
            ).all(axis=1)
            provenance_complete_ratio = float(provenance_mask.mean())
        else:
            provenance_complete_ratio = 0.0

        context_cols = ["substrate_material", "substrate_orientation", "doping_state", "alloy_state"]
        if all(col in accepted.columns for col in context_cols):
            context_mask = accepted[context_cols].fillna("").astype(str).apply(
                lambda col: col.str.strip().ne(""),
                axis=0,
            ).all(axis=1)
            context_precision_proxy = float(context_mask.mean())
        else:
            context_precision_proxy = 0.0

    ece = 0.0
    if not extraction_calibration.empty and "ece_component" in extraction_calibration.columns:
        ece = float(extraction_calibration["ece_component"].sum())

    gates = {
        "provenance_complete_ratio_eq_1": bool(abs(provenance_complete_ratio - 1.0) < 1e-12),
        "context_precision_ge_0_80": bool(context_precision_proxy >= 0.80),
        "ece_le_0_08": bool(ece <= 0.08),
        "accepted_points_gt_0": bool(len(accepted) > 0),
    }
    payload = {
        "accepted_points": int(len(accepted)),
        "provenance_complete_ratio": float(provenance_complete_ratio),
        "context_precision_proxy": float(context_precision_proxy),
        "ece": float(ece),
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }
    return payload


def validate_extraction_precision(*, votes: pd.DataFrame, accepted: pd.DataFrame, cfg: RunConfig) -> ExtractionCalibration:
    if votes.empty:
        frame = pd.DataFrame(
            columns=[
                "bin_idx",
                "bin_left",
                "bin_right",
                "point_count",
                "mean_predicted",
                "empirical_precision",
                "ece_component",
            ]
        )
        summary = {"ece": 0.0, "point_count": 0}
        path = cfg.as_path() / "artifacts" / "extraction_calibration.parquet"
        frame.to_parquet(path, index=False)
        return ExtractionCalibration(frame=frame, summary=summary, path=path)

    accepted_ids = set(accepted["point_id"].astype(str).tolist()) if not accepted.empty else set()
    grouped = (
        votes.groupby("point_id", as_index=False)
        .agg(
            predicted_confidence=("aggregated_support", "max")
            if "aggregated_support" in votes.columns
            else ("confidence", "mean")
        )
        .copy()
    )
    grouped["is_correct"] = grouped["point_id"].astype(str).isin(accepted_ids)

    bins = max(2, int(cfg.calibration_bins or 10))
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    total = max(1, len(grouped))
    for idx in range(bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == bins - 1:
            mask = (grouped["predicted_confidence"] >= left) & (grouped["predicted_confidence"] <= right)
        else:
            mask = (grouped["predicted_confidence"] >= left) & (grouped["predicted_confidence"] < right)
        bucket = grouped[mask]
        count = int(len(bucket))
        if count == 0:
            rows.append(
                {
                    "bin_idx": idx,
                    "bin_left": left,
                    "bin_right": right,
                    "point_count": 0,
                    "mean_predicted": 0.0,
                    "empirical_precision": 0.0,
                    "ece_component": 0.0,
                }
            )
            continue
        mean_pred = float(bucket["predicted_confidence"].mean())
        empirical = float(bucket["is_correct"].mean())
        comp = abs(empirical - mean_pred) * (count / total)
        ece += comp
        rows.append(
            {
                "bin_idx": idx,
                "bin_left": left,
                "bin_right": right,
                "point_count": count,
                "mean_predicted": mean_pred,
                "empirical_precision": empirical,
                "ece_component": comp,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {"ece": float(ece), "point_count": int(len(grouped))}
    path = cfg.as_path() / "artifacts" / "extraction_calibration.parquet"
    frame.to_parquet(path, index=False)
    return ExtractionCalibration(frame=frame, summary=summary, path=path)


def validate(*, spec: TopicSpec, cfg: RunConfig) -> ValidatedEvidence:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]

    evidence_path = artifacts / "evidence_points.parquet"
    mp_enriched_path = artifacts / "materials_project_enriched_points.parquet"
    if mp_enriched_path.exists():
        evidence_path = mp_enriched_path
    provenance_path = artifacts / "provenance.parquet"
    if not evidence_path.exists() or not provenance_path.exists():
        raise FileNotFoundError("missing evidence/provenance parquet artifacts")

    points = pd.read_parquet(evidence_path)
    provenance = pd.read_parquet(provenance_path)
    variable_map = _variable_map(spec)
    context_validator = ContextValidator()

    if points.empty:
        accepted = points.copy()
        rejected = points.copy()
        rejected["reject_reason"] = "no_points"
        warning_frame = pd.DataFrame(
            columns=[
                "point_id",
                "reason",
                "mp_status",
                "mp_interpretation_label",
                "mp_interpretation_score",
                "mp_conflict_reason",
                "created_at",
            ]
        )
    else:
        merged = points.merge(
            provenance[["point_id", "citation_url", "snippet", "locator", "provenance_mode"]],
            on="point_id",
            how="left",
            suffixes=("", "_prov"),
        )
        has_phase_schema_cols = any(
            col in merged.columns
            for col in ("elements_json", "stoichiometry_json", "spacegroup_symbol", "spacegroup_number")
        )

        merged["reject_reason"] = ""
        warnings_rows: list[dict[str, object]] = []

        for idx, row in merged.iterrows():
            reasons: list[str] = []
            variable_name = str(row.get("variable_name") or "")
            variable = variable_map.get(variable_name)
            if variable is None:
                reasons.append("unknown_variable")
            try:
                value = float(row.get("normalized_value"))
                if not np.isfinite(value):
                    reasons.append("non_finite_value")
                elif variable is not None and not _value_in_range(variable, value):
                    reasons.append("out_of_range")
            except Exception:
                reasons.append("non_numeric_value")

            confidence = float(row.get("confidence") or 0.0)
            etype = str(row.get("extraction_type") or "").strip().lower()
            extraction_min = float(spec.extraction_rules.confidence_thresholds.get(etype, 0.0))
            threshold = max(extraction_min, spec.validation.min_confidence)
            if confidence < threshold:
                reasons.append(f"low_confidence<{threshold:.2f}")
            if etype == "assembled" and confidence < float(cfg.assembled_confidence_floor):
                reasons.append(f"assembled_confidence<{cfg.assembled_confidence_floor:.2f}")

            for field in spec.validation.required_provenance_fields:
                value = str(row.get(field) or row.get(f"{field}_prov") or "").strip()
                if not value:
                    reasons.append(f"missing_{field}")

            entity = str(row.get("entity") or "").strip()
            if not entity:
                reasons.append("missing_entity")
            elif entity.lower() == "unknown":
                reasons.append("unknown_entity")

            if has_phase_schema_cols:
                elements_json = str(row.get("elements_json") or "").strip()
                stoich_json = str(row.get("stoichiometry_json") or "").strip()
                sg_symbol = str(row.get("spacegroup_symbol") or row.get("mp_spacegroup_symbol") or "").strip()
                sg_number = str(row.get("spacegroup_number") or row.get("mp_spacegroup_number") or "").strip()
                if bool(cfg.phase_require_elements) and (not elements_json or elements_json == "[]"):
                    reasons.append("missing_phase_elements")
                if bool(cfg.phase_require_stoich) and (not stoich_json or stoich_json == "[]"):
                    reasons.append("missing_phase_stoichiometry")
                if bool(cfg.phase_require_spacegroup) and (not sg_symbol and not sg_number):
                    reasons.append("missing_phase_spacegroup")

            if cfg.require_context_fields:
                reasons.extend(context_validator.validate_context(row.to_dict(), spec, cfg))

            mp_status = str(row.get("mp_status") or "").strip()
            mp_label = str(row.get("mp_interpretation_label") or "").strip().lower()
            mp_score = float(row.get("mp_interpretation_score") or 0.0)
            mp_conflict_reason = str(row.get("mp_conflict_reason") or "").strip()
            if mp_label == "conflicts":
                warnings_rows.append(
                    {
                        "point_id": str(row.get("point_id") or ""),
                        "reason": "mp_conflict",
                        "mp_status": mp_status,
                        "mp_interpretation_label": mp_label,
                        "mp_interpretation_score": mp_score,
                        "mp_conflict_reason": mp_conflict_reason,
                        "created_at": now_iso(),
                    }
                )
                mode = str(cfg.mp_mode or "interpreter").strip().lower()
                if mode == "hard":
                    reasons.append("mp_conflict_hard")
                elif mode == "hybrid" and confidence >= 0.90 and mp_score <= 0.20:
                    reasons.append("mp_conflict_hybrid")

            merged.at[idx, "reject_reason"] = ";".join(sorted(set(reasons)))

        x_name = spec.plot.primary.x
        y_name = spec.plot.primary.y
        for point_id, group in merged.groupby("point_id"):
            if _point_has_required_variables(group, x_name, y_name):
                continue
            mask = merged["point_id"] == point_id
            for idx in merged.index[mask]:
                reason = str(merged.at[idx, "reject_reason"])
                extra = "missing_primary_variables"
                merged.at[idx, "reject_reason"] = f"{reason};{extra}" if reason else extra

        accepted = merged[merged["reject_reason"] == ""].copy()
        rejected = merged[merged["reject_reason"] != ""].copy()
        warning_frame = pd.DataFrame(warnings_rows)
        if warning_frame.empty:
            warning_frame = pd.DataFrame(
                columns=[
                    "point_id",
                    "reason",
                    "mp_status",
                    "mp_interpretation_label",
                    "mp_interpretation_score",
                    "mp_conflict_reason",
                    "created_at",
                ]
            )

    accepted_path = artifacts / "validated_points.parquet"
    rejected_path = artifacts / "rejected_points.parquet"
    reasons_path = artifacts / "validation_reasons.parquet"
    context_validation_path = artifacts / "context_validation.parquet"
    warnings_path = artifacts / "validation_warnings.parquet"
    extraction_calibration_path = artifacts / "extraction_calibration.parquet"

    accepted.to_parquet(accepted_path, index=False)
    rejected.to_parquet(rejected_path, index=False)
    reason_table = _build_reason_table(rejected)
    reason_table.to_parquet(reasons_path, index=False)
    context_reason_table = reason_table[
        reason_table["reason"].astype(str).str.contains(
            "substrate|orientation|doping|alloy|composition|pure_ni",
            case=False,
            regex=True,
        )
    ].copy()
    if context_reason_table.empty:
        context_reason_table = pd.DataFrame(
            columns=["reason", "count", "extraction_type", "created_at"]
        )
    context_reason_table.to_parquet(context_validation_path, index=False)
    warning_frame.to_parquet(warnings_path, index=False)

    if cfg.emit_extraction_calibration:
        votes_path = artifacts / "extraction_votes.parquet"
        votes = pd.read_parquet(votes_path) if votes_path.exists() else pd.DataFrame()
        calibration = validate_extraction_precision(votes=votes, accepted=accepted, cfg=cfg)
        extraction_calibration_path = calibration.path
    else:
        pd.DataFrame(
            columns=[
                "bin_idx",
                "bin_left",
                "bin_right",
                "point_count",
                "mean_predicted",
                "empirical_precision",
                "ece_component",
            ]
        ).to_parquet(extraction_calibration_path, index=False)

    calibration_frame = pd.read_parquet(extraction_calibration_path) if extraction_calibration_path.exists() else pd.DataFrame()
    gate_payload = _extractor_gate_payload(accepted=accepted, extraction_calibration=calibration_frame)
    gate_path = artifacts / "slm_eval_metrics.json"
    gate_path.write_text(json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if str(cfg.extractor_gate_profile or "").strip().lower() == "strict_v1" and not gate_payload["all_gates_pass"]:
        failed = [
            key
            for key, passed in dict(gate_payload["gates"]).items()
            if not bool(passed)
        ]
        raise RuntimeError(f"extractor strict_v1 gate failed: {', '.join(failed)}")

    if cfg.emit_validation_metrics:
        metrics_path = artifacts / "validation_metrics.json"
        _write_validation_metrics(metrics_path, accepted, rejected, reason_table)

    state = load_run_state(cfg.as_path())
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="validated_points", path=accepted_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="rejected_points", path=rejected_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="validation_reasons", path=reasons_path)
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="context_validation",
        path=context_validation_path,
    )
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="validation_warnings",
        path=warnings_path,
    )
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="extraction_calibration",
        path=extraction_calibration_path,
    )
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="slm_eval_metrics",
        path=gate_path,
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="validate",
        source_name=evidence_path.name,
        target_name="validated_points.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="validate",
        source_name="extraction_votes.parquet",
        target_name="extraction_calibration.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="validate",
        source_name="validated_points.parquet",
        target_name="slm_eval_metrics.json",
    )

    return ValidatedEvidence(
        accepted=accepted,
        rejected=rejected,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        validation_reasons_path=reasons_path,
        context_validation_path=context_validation_path,
        warnings_path=warnings_path,
        extraction_calibration_path=extraction_calibration_path,
    )
