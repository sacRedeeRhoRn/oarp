from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from oarp.models import ConsensusSet, OutputBundle, RunConfig
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    upsert_artifact,
)
from oarp.topic_spec import TopicSpec


def _entity_color_map(values: list[str]) -> dict[str, str]:
    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#666666",
    ]
    out: dict[str, str] = {}
    for idx, entity in enumerate(sorted(set(values))):
        out[entity] = palette[idx % len(palette)]
    return out


def _write_citation_table(frame: pd.DataFrame, path: Path) -> Path:
    cols = [
        "point_id",
        "entity",
        "x_value",
        "y_value",
        "substrate_material",
        "substrate_orientation",
        "doping_state",
        "alloy_state",
        "model_confidence",
        "entropy",
        "display_alpha",
        "mp_status",
        "mp_best_material_id",
        "mp_interpretation_score",
        "mp_interpretation_label",
        "mp_conflict_reason",
        "doi",
        "citation_url",
        "locator",
    ]
    if frame.empty:
        citation = pd.DataFrame(columns=cols)
    else:
        work = frame.copy()
        for col in cols:
            if col not in work.columns:
                work[col] = ""
        citation = work[cols].copy()
    citation.to_csv(path, index=False)
    return path


def _render_scatter(frame: pd.DataFrame, spec: TopicSpec, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y

    if frame.empty:
        ax.set_title(f"No validated points for {spec.topic_id}")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return path

    colors = _entity_color_map(frame["entity"].tolist())
    for entity, group in frame.groupby("entity"):
        ax.scatter(
            group["x_value"],
            group["y_value"],
            c=colors.get(entity, "#333333"),
            alpha=group["display_alpha"].astype(float).clip(0.05, 1.0).tolist(),
            edgecolors="none",
            s=45,
            label=f"{entity} (n={len(group)})",
        )

    # Draw consensus curve for dominant entity per x point.
    consensus_rows = (
        frame.sort_values("model_confidence", ascending=False)
        .drop_duplicates(subset=["x_value"], keep="first")
        .sort_values("x_value")
    )
    if not consensus_rows.empty:
        ax.plot(
            consensus_rows["x_value"],
            consensus_rows["consensus_y"],
            color="#1f2937",
            linewidth=1.3,
            linestyle="--",
            label="consensus trend",
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{spec.topic_id}: citation-backed scatter")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _build_report(
    *,
    spec: TopicSpec,
    consensus: pd.DataFrame,
    summary: dict[str, Any],
    plot_rel_path: str,
    citation_rel_path: str,
    validation_reason_rows: list[dict[str, Any]],
) -> str:
    context_coverage = 0.0
    mp_coverage = 0.0
    mp_conflict_rate = 0.0
    mp_support_rate = 0.0
    if not consensus.empty:
        required = {"substrate_material", "substrate_orientation", "doping_state", "alloy_state"}
        if required.issubset(set(consensus.columns)):
            valid = consensus[
                consensus["substrate_material"].astype(str).str.strip().ne("")
                & consensus["substrate_orientation"].astype(str).str.strip().ne("")
                & consensus["doping_state"].astype(str).str.strip().ne("")
                & consensus["alloy_state"].astype(str).str.strip().ne("")
            ]
            context_coverage = float(len(valid) / max(len(consensus), 1))
        if "mp_status" in consensus.columns:
            mp_coverage = float(
                consensus["mp_status"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"success", "no_match", "fetch_error", "parse_error", "no_api_key", "disabled"})
                .mean()
            )
        if "mp_interpretation_label" in consensus.columns:
            labels = consensus["mp_interpretation_label"].astype(str).str.strip().str.lower()
            mp_conflict_rate = float((labels == "conflicts").mean())
            mp_support_rate = float((labels == "supports").mean())

    lines = [
        f"# Research Report: {spec.topic_id}",
        "",
        "## Summary",
        f"- points plotted: `{summary.get('point_count', 0)}`",
        f"- outliers flagged: `{summary.get('outlier_count', 0)}`",
        f"- mean entropy: `{summary.get('mean_entropy', 0.0):.4f}`",
        f"- context completeness: `{context_coverage:.4f}`",
        f"- MP coverage: `{mp_coverage:.4f}`",
        f"- MP support rate: `{mp_support_rate:.4f}`",
        f"- MP conflict rate: `{mp_conflict_rate:.4f}`",
        "",
        "## Plot",
        f"![primary scatter]({plot_rel_path})",
        "",
        "## Citation Table",
        f"- CSV: `{citation_rel_path}`",
        "",
        "## Conflict Notes",
    ]

    if consensus.empty:
        lines.append("- no validated points available")
    else:
        high_entropy = consensus[consensus["entropy"] >= spec.consensus.entropy_threshold]
        if high_entropy.empty:
            lines.append("- no high-entropy conflict regions detected")
        else:
            grouped = high_entropy.groupby("x_value").head(1)
            for _, row in grouped.head(10).iterrows():
                lines.append(
                    "- x={0:.3f}: entropy={1:.3f}, consensus={2}, source={3}".format(
                        float(row["x_value"]),
                        float(row["entropy"]),
                        str(row["consensus_entity"]),
                        str(row["citation_url"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Validation Rejection Summary",
        ]
    )

    if validation_reason_rows:
        for row in validation_reason_rows[:12]:
            lines.append(
                "- {0}: `{1}` (extraction={2})".format(
                    str(row.get("reason") or ""),
                    int(row.get("count") or 0),
                    str(row.get("extraction_type") or ""),
                )
            )
    else:
        lines.append("- no rejection reasons recorded")

    lines.extend(
        [
            "",
            "## Materials Project",
        ]
    )
    if consensus.empty or "mp_interpretation_label" not in consensus.columns:
        lines.append("- MP interpretation not available")
    else:
        conflicts = consensus[
            consensus["mp_interpretation_label"].astype(str).str.strip().str.lower() == "conflicts"
        ]
        if conflicts.empty:
            lines.append("- no MP conflicts detected")
        else:
            by_ctx = (
                conflicts.groupby(["entity", "context_signature"], as_index=False)
                .size()
                .sort_values("size", ascending=False)
                .head(10)
            )
            for _, row in by_ctx.iterrows():
                lines.append(
                    "- entity={0}, context=`{1}` conflicts=`{2}`".format(
                        str(row.get("entity") or ""),
                        str(row.get("context_signature") or ""),
                        int(row.get("size") or 0),
                    )
                )

    lines.extend(
        [
            "",
            "## Context Slices",
        ]
    )
    if consensus.empty or "context_signature" not in consensus.columns:
        lines.append("- no context slices available")
    else:
        top_ctx = (
            consensus.groupby("context_signature", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(8)
        )
        for _, row in top_ctx.iterrows():
            sig = str(row.get("context_signature") or "")
            lines.append(f"- `{sig or 'unknown'}`: `{int(row.get('size') or 0)}`")

    lines.extend(
        [
            "",
            "## Methods",
            "- OA-only discovery + acquisition",
            "- Layered extraction (text/table/figure cues)",
            "- Strict confidence/provenance validation",
            "- Entropy-aware consensus with transparent outlier rendering",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render(*, spec: TopicSpec, consensus_set: ConsensusSet, cfg: RunConfig) -> OutputBundle:
    layout = ensure_run_layout(cfg.as_path())
    outputs = layout["outputs"]
    plots = layout["plots"]
    artifacts = layout["artifacts"]

    consensus = consensus_set.points
    citation_path = outputs / "citation_points.csv"
    plot_path = plots / "primary_scatter.png"
    validation_path = artifacts / "validation_reasons.parquet"
    validation_rows: list[dict[str, Any]] = []
    if validation_path.exists():
        validation_rows = pd.read_parquet(validation_path).to_dict(orient="records")

    _write_citation_table(consensus, citation_path)
    _render_scatter(consensus, spec, plot_path)

    report_path = outputs / "report.md"
    report_body = _build_report(
        spec=spec,
        consensus=consensus,
        summary=consensus_set.summary,
        plot_rel_path=str(plot_path.relative_to(outputs)),
        citation_rel_path=str(citation_path.relative_to(outputs)),
        validation_reason_rows=validation_rows,
    )
    report_path.write_text(report_body, encoding="utf-8")

    state = load_run_state(cfg.as_path())
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="report", path=report_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="plot_primary", path=plot_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="citation_table", path=citation_path)
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="render",
        source_name="consensus_points.parquet",
        target_name="outputs/report.md",
    )

    return OutputBundle(
        report_path=report_path,
        plot_paths=[plot_path],
        citation_table_path=citation_path,
    )
