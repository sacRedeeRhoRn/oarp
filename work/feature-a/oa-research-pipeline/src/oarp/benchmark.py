from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oarp.topic_spec import TopicSpec


@dataclass
class BenchmarkResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    threshold_met: bool
    report_path: Path
    json_path: Path
    shadow_metrics_path: Path | None = None
    mp_json_path: Path | None = None
    mp_report_path: Path | None = None
    context_report_path: Path | None = None


def _update_run_metrics(
    *,
    run_path: Path,
    precision: float,
    recall: float,
    f1: float,
    threshold_met: bool,
    precision_gate: float,
    strict_gold: bool,
    context_payload: dict[str, Any],
    context_completeness_gate: float,
    context_precision_gate: float,
    json_path: Path,
    report_path: Path,
    shadow_metrics_path: Path | None,
    context_report_path: Path | None,
    mp_json_path: Path | None,
    mp_report_path: Path | None,
) -> None:
    metrics_path = run_path / "artifacts" / "run_metrics.json"
    if not metrics_path.exists():
        return

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(metrics, dict):
        return

    day3 = metrics.get("day3_summary")
    if not isinstance(day3, dict):
        day3 = {}

    condition_completeness = float(context_payload.get("condition_completeness") or 0.0)
    condition_precision = float(context_payload.get("condition_precision") or 0.0)
    condition_recall = float(context_payload.get("condition_recall") or 0.0)
    condition_f1 = float(context_payload.get("condition_f1") or 0.0)
    context_gate_pass = bool(context_payload.get("context_threshold_met"))

    day3["benchmark"] = {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "precision_gate": float(precision_gate),
        "threshold_met": bool(threshold_met),
        "strict_gold": bool(strict_gold),
        "benchmark_json": str(json_path),
        "benchmark_report": str(report_path),
        "shadow_metrics_path": str(shadow_metrics_path) if shadow_metrics_path else "",
        "context_report_path": str(context_report_path) if context_report_path else "",
        "mp_json_path": str(mp_json_path) if mp_json_path else "",
        "mp_report_path": str(mp_report_path) if mp_report_path else "",
        "context_precision_gate": float(context_precision_gate),
    }
    day3["context_quality"] = {
        "validated_context_completeness": round(condition_completeness, 6),
        "condition_precision": round(condition_precision, 6),
        "condition_recall": round(condition_recall, 6),
        "condition_f1": round(condition_f1, 6),
        "context_completeness_gate": float(context_completeness_gate),
        "context_precision_gate": float(context_precision_gate),
        "completeness_gate_pass": condition_completeness >= float(context_completeness_gate),
        "precision_gate_pass": condition_precision >= float(context_precision_gate),
        "context_gate_pass": context_gate_pass,
    }

    metrics["day3_summary"] = day3
    metrics["benchmarks"] = {
        "benchmark_json": str(json_path),
        "benchmark_report": str(report_path),
        "shadow_metrics_path": str(shadow_metrics_path) if shadow_metrics_path else "",
        "context_report_path": str(context_report_path) if context_report_path else "",
        "benchmark_mp_json": str(mp_json_path) if mp_json_path else "",
        "benchmark_mp_report": str(mp_report_path) if mp_report_path else "",
    }

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_gold(gold_dir: Path) -> pd.DataFrame:
    parquet_path = gold_dir / "gold_points.parquet"
    csv_path = gold_dir / "gold_points.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"gold dataset missing. expected {parquet_path} or {csv_path}")


def _load_predicted(run_dir: Path) -> pd.DataFrame:
    artifacts = run_dir / "artifacts"
    consensus_path = artifacts / "consensus_points.parquet"
    if consensus_path.exists():
        return pd.read_parquet(consensus_path)
    validated_path = artifacts / "validated_points.parquet"
    if not validated_path.exists():
        raise FileNotFoundError(f"predicted points missing: {validated_path}")
    validated = pd.read_parquet(validated_path)

    if validated.empty:
        return pd.DataFrame(columns=["point_id", "x_value", "y_value", "entity"])

    points = []
    for point_id, group in validated.groupby("point_id"):
        x_rows = group[group["variable_name"] == "thickness_nm"]
        y_rows = group[group["variable_name"] == "temperature_c"]
        if x_rows.empty or y_rows.empty:
            continue
        row = {
            "point_id": point_id,
            "x_value": float(x_rows.iloc[0]["normalized_value"]),
            "y_value": float(y_rows.iloc[0]["normalized_value"]),
            "entity": str(group.iloc[0].get("entity") or "unknown"),
        }
        for col in (
            "substrate_material",
            "substrate_orientation",
            "doping_state",
            "alloy_state",
        ):
            row[col] = str(group.iloc[0].get(col) or "")
        points.append(row)
    return pd.DataFrame(points)


def _normalize_gold_columns(df: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:
    out = df.copy()
    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y

    rename_map = {}
    for col in out.columns:
        low = col.strip().lower()
        if low in {"x", x_name.lower(), "x_value"}:
            rename_map[col] = "x_value"
        elif low in {"y", y_name.lower(), "y_value"}:
            rename_map[col] = "y_value"
        elif low in {"phase", "entity", spec.plot.primary.color_by.lower()}:
            rename_map[col] = "entity"
    out = out.rename(columns=rename_map)
    for col in ("x_value", "y_value", "entity"):
        if col not in out.columns:
            raise ValueError(f"gold dataset missing required column: {col}")
    return out[["x_value", "y_value", "entity"]]


def _normalize_context_gold_columns(df: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:
    out = _normalize_gold_columns(df, spec)
    src = df.copy()
    rename_map = {}
    for col in src.columns:
        low = str(col).strip().lower()
        if low in {"substrate", "substrate_material"}:
            rename_map[col] = "substrate_material"
        elif low in {"orientation", "substrate_orientation", "crystallographic_orientation"}:
            rename_map[col] = "substrate_orientation"
        elif low in {"doping", "doping_state"}:
            rename_map[col] = "doping_state"
        elif low in {"alloy", "alloy_state"}:
            rename_map[col] = "alloy_state"
    src = src.rename(columns=rename_map)
    for col in ("substrate_material", "substrate_orientation", "doping_state", "alloy_state"):
        if col not in src.columns:
            src[col] = ""
        out[col] = src[col].astype(str)
    return out


def _nearest_gold_reason(pred_row: pd.Series, gold: pd.DataFrame, tol_x: float, tol_y: float) -> str:
    if gold.empty:
        return "no_gold_points"

    px = float(pred_row["x_value"])
    py = float(pred_row["y_value"])
    pe = str(pred_row.get("entity") or "").strip().lower()

    distances = []
    for _, grow in gold.iterrows():
        gx = float(grow["x_value"])
        gy = float(grow["y_value"])
        dx = abs(px - gx)
        dy = abs(py - gy)
        distances.append((dx + dy, dx, dy, str(grow.get("entity") or "").strip().lower()))
    distances.sort(key=lambda item: item[0])
    _, dx, dy, ge = distances[0]

    if pe != ge:
        return "entity_mismatch"
    if dx > tol_x and dy > tol_y:
        return "x_y_outside_tolerance"
    if dx > tol_x:
        return "x_outside_tolerance"
    if dy > tol_y:
        return "y_outside_tolerance"
    return "duplicate_or_collision"


def _evaluate(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    tol_x: float,
    tol_y: float,
) -> tuple[int, int, int, dict[str, float], dict[str, int], list[dict[str, Any]]]:
    if pred.empty and gold.empty:
        return 0, 0, 0, {}, {}, []
    if pred.empty:
        return 0, 0, len(gold), {}, {}, []
    if gold.empty:
        return 0, len(pred), 0, {}, {"no_gold_points": int(len(pred))}, []

    matched_gold = set()
    tp = 0
    entity_tp: dict[str, int] = {}
    entity_fp: dict[str, int] = {}
    fp_reasons: dict[str, int] = {}
    fp_rows: list[dict[str, Any]] = []

    for _, prow in pred.iterrows():
        px = float(prow["x_value"])
        py = float(prow["y_value"])
        pe = str(prow.get("entity") or "").strip().lower()
        found_index = None
        for gidx, grow in gold.iterrows():
            if gidx in matched_gold:
                continue
            gx = float(grow["x_value"])
            gy = float(grow["y_value"])
            ge = str(grow.get("entity") or "").strip().lower()
            if abs(px - gx) <= tol_x and abs(py - gy) <= tol_y and pe == ge:
                found_index = gidx
                break
        if found_index is not None:
            matched_gold.add(found_index)
            tp += 1
            entity_tp[pe] = entity_tp.get(pe, 0) + 1
            continue

        reason = _nearest_gold_reason(prow, gold, tol_x=tol_x, tol_y=tol_y)
        fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
        entity_fp[pe] = entity_fp.get(pe, 0) + 1
        fp_rows.append(
            {
                "point_id": str(prow.get("point_id") or ""),
                "entity": pe,
                "x_value": px,
                "y_value": py,
                "reason": reason,
            }
        )

    fp = len(pred) - tp
    fn = len(gold) - tp

    entity_precision: dict[str, float] = {}
    for entity in sorted(set(entity_tp.keys()) | set(entity_fp.keys())):
        etp = entity_tp.get(entity, 0)
        efp = entity_fp.get(entity, 0)
        entity_precision[entity] = etp / (etp + efp) if (etp + efp) else 0.0

    return tp, fp, fn, entity_precision, fp_reasons, fp_rows


def _context_completeness(pred: pd.DataFrame) -> float:
    required = ["substrate_material", "substrate_orientation", "doping_state", "alloy_state"]
    if pred.empty or not all(col in pred.columns for col in required):
        return 0.0
    mask = pred[required].apply(lambda col: col.astype(str).str.strip().ne(""), axis=0).all(axis=1)
    return float(mask.mean())


def _context_match(pred_row: pd.Series, gold_row: pd.Series) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for field in ("substrate_material", "substrate_orientation", "doping_state", "alloy_state"):
        pv = str(pred_row.get(field) or "").strip().lower()
        gv = str(gold_row.get(field) or "").strip().lower()
        if pv != gv:
            mismatches.append(field)
    return len(mismatches) == 0, mismatches


def _evaluate_context(
    pred: pd.DataFrame,
    gold: pd.DataFrame,
    tol_x: float,
    tol_y: float,
) -> tuple[float, float, float, dict[str, int], list[dict[str, Any]]]:
    if pred.empty or gold.empty:
        return 0.0, 0.0, 0.0, {}, []

    matched_gold = set()
    tp = 0
    fp = 0
    error_breakdown: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for _, prow in pred.iterrows():
        px = float(prow.get("x_value") or 0.0)
        py = float(prow.get("y_value") or 0.0)
        pe = str(prow.get("entity") or "").strip().lower()

        matched_idx = None
        matched_row = None
        for gidx, grow in gold.iterrows():
            if gidx in matched_gold:
                continue
            gx = float(grow.get("x_value") or 0.0)
            gy = float(grow.get("y_value") or 0.0)
            ge = str(grow.get("entity") or "").strip().lower()
            if abs(px - gx) <= tol_x and abs(py - gy) <= tol_y and pe == ge:
                matched_idx = gidx
                matched_row = grow
                break

        if matched_idx is None or matched_row is None:
            fp += 1
            error_breakdown["no_xy_entity_match"] = error_breakdown.get("no_xy_entity_match", 0) + 1
            continue

        matched_gold.add(matched_idx)
        ok, mismatches = _context_match(prow, matched_row)
        if ok:
            tp += 1
        else:
            fp += 1
            for field in mismatches:
                error_breakdown[field] = error_breakdown.get(field, 0) + 1
            examples.append(
                {
                    "point_id": str(prow.get("point_id") or ""),
                    "entity": pe,
                    "x_value": px,
                    "y_value": py,
                    "mismatches": mismatches,
                }
            )

    fn = len(gold) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, error_breakdown, examples


def _evaluate_mp(pred: pd.DataFrame, gold_context: pd.DataFrame | None = None) -> dict[str, Any]:
    if pred.empty:
        return {
            "mp_coverage": 0.0,
            "mp_support_precision": 0.0,
            "mp_conflict_rate": 0.0,
            "mp_alignment_by_entity": {},
            "labeled_rows": 0,
        }

    work = pred.copy()
    for col in ("mp_status", "mp_interpretation_label", "entity"):
        if col not in work.columns:
            work[col] = ""

    coverage_mask = work["mp_status"].astype(str).str.strip().str.lower().isin(
        {"success", "no_match", "fetch_error", "parse_error", "no_api_key", "disabled"}
    )
    mp_coverage = float(coverage_mask.mean()) if len(work) else 0.0
    support_mask = work["mp_interpretation_label"].astype(str).str.strip().str.lower() == "supports"
    conflict_mask = work["mp_interpretation_label"].astype(str).str.strip().str.lower() == "conflicts"
    mp_conflict_rate = float(conflict_mask.mean()) if len(work) else 0.0

    by_entity: dict[str, float] = {}
    for entity, group in work.groupby(work["entity"].astype(str).str.strip().str.lower()):
        if not str(entity).strip():
            continue
        by_entity[str(entity)] = float(
            (
                group["mp_interpretation_label"].astype(str).str.strip().str.lower() == "supports"
            ).mean()
        )

    mp_support_precision = float(support_mask.mean()) if len(work) else 0.0
    labeled_rows = 0
    if gold_context is not None and not gold_context.empty and "mp_expected_label" in gold_context.columns:
        labeled_rows = 0
        tp = 0
        fp = 0
        for _, prow in work.iterrows():
            px = float(prow.get("x_value") or 0.0)
            py = float(prow.get("y_value") or 0.0)
            pe = str(prow.get("entity") or "").strip().lower()
            match = gold_context[
                (gold_context["x_value"].astype(float).sub(px).abs() <= 2.0)
                & (gold_context["y_value"].astype(float).sub(py).abs() <= 8.0)
                & (gold_context["entity"].astype(str).str.strip().str.lower() == pe)
            ]
            if match.empty:
                continue
            expected = str(match.iloc[0].get("mp_expected_label") or "").strip().lower()
            if not expected:
                continue
            labeled_rows += 1
            pred_label = str(prow.get("mp_interpretation_label") or "").strip().lower()
            if pred_label == "supports":
                if expected == "supports":
                    tp += 1
                else:
                    fp += 1
        mp_support_precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "mp_coverage": mp_coverage,
        "mp_support_precision": mp_support_precision,
        "mp_conflict_rate": mp_conflict_rate,
        "mp_alignment_by_entity": by_entity,
        "labeled_rows": labeled_rows,
    }


def run_benchmark(
    *,
    spec: TopicSpec,
    gold_dir: str | Path,
    out_dir: str | Path,
    run_dir: str | Path,
    precision_gate: float = 0.85,
    tol_x: float = 2.0,
    tol_y: float = 8.0,
    shadow_gold_dir: str | Path | None = None,
    strict_gold: bool = True,
    gold_context_dir: str | Path | None = None,
    context_completeness_gate: float = 0.70,
    context_precision_gate: float = 0.80,
) -> BenchmarkResult:
    gold_path = Path(gold_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    run_path = Path(run_dir).expanduser().resolve()

    gold_raw = _load_gold(gold_path)
    gold = _normalize_gold_columns(gold_raw, spec)
    pred = _load_predicted(run_path)

    tp, fp, fn, entity_precision, fp_reasons, fp_rows = _evaluate(pred, gold, tol_x=tol_x, tol_y=tol_y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    threshold_met = precision >= precision_gate

    payload: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_gate": precision_gate,
        "threshold_met": threshold_met,
        "tol_x": tol_x,
        "tol_y": tol_y,
        "precision_by_entity": entity_precision,
        "false_positive_reasons": fp_reasons,
        "false_positive_examples": fp_rows[:30],
        "strict_gold": bool(strict_gold),
    }

    json_path = out_path / "benchmark.json"

    report_lines = [
        "# Benchmark Result",
        "",
        f"- precision: `{precision:.4f}`",
        f"- recall: `{recall:.4f}`",
        f"- f1: `{f1:.4f}`",
        f"- tp/fp/fn: `{tp}/{fp}/{fn}`",
        f"- precision gate ({precision_gate:.2f}): `{threshold_met}`",
        "",
        "## Precision by Entity",
    ]
    if entity_precision:
        for entity, val in sorted(entity_precision.items()):
            report_lines.append(f"- {entity}: `{val:.4f}`")
    else:
        report_lines.append("- no entity precision rows available")

    report_lines.extend(["", "## False Positive Reasons"])
    if fp_reasons:
        for reason, count in sorted(fp_reasons.items(), key=lambda item: item[1], reverse=True):
            report_lines.append(f"- {reason}: `{count}`")
    else:
        report_lines.append("- none")

    report_path = out_path / "benchmark.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    shadow_metrics_path: Path | None = None
    if shadow_gold_dir:
        shadow_path = Path(shadow_gold_dir).expanduser().resolve()
        if shadow_path.exists():
            shadow_gold_raw = _load_gold(shadow_path)
            shadow_gold = _normalize_gold_columns(shadow_gold_raw, spec)
            stp, sfp, sfn, sentity_precision, sfp_reasons, sfp_rows = _evaluate(
                pred,
                shadow_gold,
                tol_x=tol_x,
                tol_y=tol_y,
            )
            sprecision = stp / (stp + sfp) if (stp + sfp) else 0.0
            srecall = stp / (stp + sfn) if (stp + sfn) else 0.0
            sf1 = (2 * sprecision * srecall / (sprecision + srecall)) if (sprecision + srecall) else 0.0
            if not strict_gold:
                threshold_met = max(precision, sprecision) >= precision_gate
            shadow_payload: dict[str, Any] = {
                "precision": sprecision,
                "recall": srecall,
                "f1": sf1,
                "tp": stp,
                "fp": sfp,
                "fn": sfn,
                "precision_gate": precision_gate,
                "threshold_met": sprecision >= precision_gate,
                "tol_x": tol_x,
                "tol_y": tol_y,
                "precision_by_entity": sentity_precision,
                "false_positive_reasons": sfp_reasons,
                "false_positive_examples": sfp_rows[:30],
                "shadow_gold_dir": str(shadow_path),
            }
            shadow_metrics_path = out_path / "benchmark_shadow.json"
            shadow_metrics_path.write_text(
                json.dumps(shadow_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shadow_report_path = out_path / "benchmark_shadow.md"
            shadow_lines = [
                "# Benchmark Shadow Result",
                "",
                f"- precision: `{sprecision:.4f}`",
                f"- recall: `{srecall:.4f}`",
                f"- f1: `{sf1:.4f}`",
                f"- tp/fp/fn: `{stp}/{sfp}/{sfn}`",
                f"- precision gate ({precision_gate:.2f}): `{sprecision >= precision_gate}`",
            ]
            shadow_report_path.write_text("\n".join(shadow_lines) + "\n", encoding="utf-8")

    context_report_path: Path | None = None
    context_gold_norm: pd.DataFrame | None = None
    context_payload: dict[str, Any] = {
        "condition_completeness": _context_completeness(pred),
        "condition_precision": 0.0,
        "condition_recall": 0.0,
        "condition_f1": 0.0,
        "condition_error_breakdown": {},
        "context_completeness_gate": float(context_completeness_gate),
        "context_precision_gate": float(context_precision_gate),
        "context_threshold_met": False,
    }

    if gold_context_dir:
        ctx_gold_path = Path(gold_context_dir).expanduser().resolve()
        if ctx_gold_path.exists():
            ctx_gold_raw = _load_gold(ctx_gold_path)
            ctx_gold = _normalize_context_gold_columns(ctx_gold_raw, spec)
            for col in ctx_gold_raw.columns:
                low = str(col).strip().lower()
                if low in {"mp_expected_label", "mp_label", "expected_mp_label"}:
                    ctx_gold["mp_expected_label"] = ctx_gold_raw[col].astype(str)
                    break
            context_gold_norm = ctx_gold
            ctx_precision, ctx_recall, ctx_f1, ctx_errors, ctx_examples = _evaluate_context(
                pred,
                ctx_gold,
                tol_x=tol_x,
                tol_y=tol_y,
            )
            context_payload.update(
                {
                    "condition_precision": ctx_precision,
                    "condition_recall": ctx_recall,
                    "condition_f1": ctx_f1,
                    "condition_error_breakdown": ctx_errors,
                    "context_examples": ctx_examples[:30],
                    "context_threshold_met": bool(
                        context_payload["condition_completeness"] >= float(context_completeness_gate)
                        and ctx_precision >= float(context_precision_gate)
                    ),
                }
            )
            context_json_path = out_path / "benchmark_context.json"
            context_json_path.write_text(
                json.dumps(context_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            context_report_path = out_path / "benchmark_context.md"
            context_report_path.write_text(
                "\n".join(
                    [
                        "# Benchmark Context Result",
                        "",
                        f"- condition completeness: `{context_payload['condition_completeness']:.4f}`",
                        f"- condition precision: `{ctx_precision:.4f}`",
                        f"- condition recall: `{ctx_recall:.4f}`",
                        f"- condition f1: `{ctx_f1:.4f}`",
                        f"- context gate ({context_completeness_gate:.2f} completeness, {context_precision_gate:.2f} precision): `{context_payload['context_threshold_met']}`",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

    mp_payload = _evaluate_mp(pred, context_gold_norm)
    mp_json_path = out_path / "benchmark_mp.json"
    mp_json_path.write_text(json.dumps(mp_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mp_report_path = out_path / "benchmark_mp.md"
    mp_report_lines = [
        "# Benchmark MP Result",
        "",
        f"- mp coverage: `{float(mp_payload.get('mp_coverage') or 0.0):.4f}`",
        f"- mp support precision: `{float(mp_payload.get('mp_support_precision') or 0.0):.4f}`",
        f"- mp conflict rate: `{float(mp_payload.get('mp_conflict_rate') or 0.0):.4f}`",
        f"- labeled rows: `{int(mp_payload.get('labeled_rows') or 0)}`",
        "",
        "## MP Alignment By Entity",
    ]
    alignment = dict(mp_payload.get("mp_alignment_by_entity") or {})
    if alignment:
        for entity, score in sorted(alignment.items()):
            mp_report_lines.append(f"- {entity}: `{float(score):.4f}`")
    else:
        mp_report_lines.append("- no entity-level MP alignment rows")
    mp_report_path.write_text("\n".join(mp_report_lines) + "\n", encoding="utf-8")

    payload["threshold_met"] = threshold_met
    payload["shadow_metrics_path"] = str(shadow_metrics_path) if shadow_metrics_path else ""
    payload["condition_completeness"] = context_payload["condition_completeness"]
    payload["condition_precision"] = context_payload["condition_precision"]
    payload["condition_recall"] = context_payload["condition_recall"]
    payload["condition_error_breakdown"] = context_payload["condition_error_breakdown"]
    payload["context_report_path"] = str(context_report_path) if context_report_path else ""
    payload["benchmark_mp_json"] = str(mp_json_path)
    payload["benchmark_mp_report"] = str(mp_report_path)
    payload["mp_coverage"] = float(mp_payload.get("mp_coverage") or 0.0)
    payload["mp_support_precision"] = float(mp_payload.get("mp_support_precision") or 0.0)
    payload["mp_conflict_rate"] = float(mp_payload.get("mp_conflict_rate") or 0.0)
    payload["mp_alignment_by_entity"] = dict(mp_payload.get("mp_alignment_by_entity") or {})
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _update_run_metrics(
        run_path=run_path,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold_met=threshold_met,
        precision_gate=precision_gate,
        strict_gold=strict_gold,
        context_payload=context_payload,
        context_completeness_gate=context_completeness_gate,
        context_precision_gate=context_precision_gate,
        json_path=json_path,
        report_path=report_path,
        shadow_metrics_path=shadow_metrics_path,
        context_report_path=context_report_path,
        mp_json_path=mp_json_path,
        mp_report_path=mp_report_path,
    )

    return BenchmarkResult(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        threshold_met=threshold_met,
        report_path=report_path,
        json_path=json_path,
        shadow_metrics_path=shadow_metrics_path,
        mp_json_path=mp_json_path,
        context_report_path=context_report_path,
        mp_report_path=mp_report_path,
    )


def run_extraction_benchmark(
    *,
    suite: str,
    spec: TopicSpec,
    run_dir: str | Path,
    out_dir: str | Path,
    gold_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    suite_key = str(suite or "").strip().lower()

    if suite_key == "thinfilm_gold":
        if gold_dir is None:
            raise ValueError("gold_dir is required for thinfilm_gold suite")
        result = run_benchmark(
            spec=spec,
            gold_dir=gold_dir,
            out_dir=out_path,
            run_dir=run_path,
            strict_gold=True,
        )
        payload = {
            "suite": suite_key,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "threshold_met": result.threshold_met,
            "benchmark_json": str(result.json_path),
            "benchmark_report": str(result.report_path),
        }
    else:
        votes_path = run_path / "artifacts" / "extraction_votes.parquet"
        votes = pd.read_parquet(votes_path) if votes_path.exists() else pd.DataFrame()
        accepted_ratio = (
            float(votes["accepted"].mean()) if (not votes.empty and "accepted" in votes.columns) else 0.0
        )
        mean_support = (
            float(votes["aggregated_support"].mean())
            if (not votes.empty and "aggregated_support" in votes.columns)
            else 0.0
        )
        payload = {
            "suite": suite_key,
            "status": "informational_only",
            "detail": "external suite assets are not bundled in this repository",
            "vote_rows": int(len(votes)),
            "accepted_ratio": accepted_ratio,
            "mean_support": mean_support,
        }

    json_path = out_path / f"benchmark_extraction_{suite_key}.json"
    md_path = out_path / f"benchmark_extraction_{suite_key}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# Extraction Benchmark: {suite_key}",
        "",
    ]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(md_path)
    return payload


def run_processing_benchmark(
    *,
    suite: str,
    run_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    suite_key = str(suite or "").strip().lower()
    artifacts = run_path / "artifacts"

    payload: dict[str, Any] = {"suite": suite_key}
    metrics_path = artifacts / "processor_eval_metrics.json"
    if metrics_path.exists():
        try:
            payload["processor_eval"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            payload["processor_eval"] = {}
    else:
        payload["processor_eval"] = {}

    if suite_key == "recipe_rank":
        ranked_path = artifacts / "recipe_ranked.parquet"
        ranked = pd.read_parquet(ranked_path) if ranked_path.exists() else pd.DataFrame()
        if ranked.empty:
            payload["ndcg_at_10"] = 0.0
            payload["ranked_rows"] = 0
        else:
            ranked = ranked.sort_values("rank") if "rank" in ranked.columns else ranked.copy()
            rel = ranked.get("weighted_score", pd.Series([0.0] * len(ranked))).astype(float).tolist()[:10]
            if not rel:
                ndcg = 0.0
            else:
                dcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(rel))
                ideal = sorted(rel, reverse=True)
                idcg = sum((2**v - 1) / np.log2(i + 2) for i, v in enumerate(ideal))
                ndcg = float(dcg / idcg) if idcg > 0 else 0.0
            payload["ndcg_at_10"] = ndcg
            payload["ranked_rows"] = int(len(ranked))
    elif suite_key == "context_quality":
        validated_path = artifacts / "validated_points.parquet"
        validated = pd.read_parquet(validated_path) if validated_path.exists() else pd.DataFrame()
        payload["context_completeness"] = _context_completeness(_load_predicted(run_path))
        payload["validated_rows"] = int(len(validated))
    elif suite_key == "phase_transition":
        phase_path = artifacts / "phase_events.parquet"
        phase = pd.read_parquet(phase_path) if phase_path.exists() else pd.DataFrame()
        payload["phase_event_count"] = int(len(phase))
        payload["phase_diversity"] = int(phase["phase_label"].nunique()) if "phase_label" in phase.columns else 0
    else:
        payload["status"] = "unknown_suite"

    json_path = out_path / f"benchmark_processing_{suite_key}.json"
    md_path = out_path / f"benchmark_processing_{suite_key}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# Processing Benchmark: {suite_key}",
        "",
    ]
    for key, value in payload.items():
        lines.append(f"- {key}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["json_path"] = str(json_path)
    payload["report_path"] = str(md_path)
    return payload
