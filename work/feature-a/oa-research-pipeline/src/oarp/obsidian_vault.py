from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from oarp.models import RunConfig, VaultBenchmarkResult, VaultExportResult, VaultImportResult
from oarp.runtime import ensure_run_layout, load_run_state, now_iso, write_json

_WIKILINK_RE = re.compile(r"!?(\[\[([^\]]+)\]\])")
_HEADING_RE = re.compile(r"^\s{0,3}#{2,6}\s+(.+?)\s*$")


def _slug(value: Any, default: str = "item") -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return token or default


def _hash_id(*parts: Any, length: int = 16) -> str:
    digest = hashlib.sha1("|".join(str(item) for item in parts).encode("utf-8", errors="replace")).hexdigest()
    return digest[: max(8, int(length))]


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value or "")
    if not text:
        return '""'
    if re.search(r"[:#\[\]\n\r\t]", text) or text.strip() != text:
        return json.dumps(text, ensure_ascii=True)
    return text


def _frontmatter(payload: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {_yaml_scalar(item)}")
            continue
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    return "\n".join(lines)


def _read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _note_path(vault_root: Path, rel: str) -> Path:
    target = vault_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _load_run_id(run_dir: Path) -> str:
    try:
        state = load_run_state(run_dir)
        run_id = str(state.get("run_id") or "").strip()
        if run_id:
            return run_id
    except Exception:
        pass
    return f"run-{_hash_id(run_dir.name, now_iso())}"


def _article_key_from_row(row: dict[str, Any]) -> str:
    raw = str(row.get("article_key") or "").strip()
    if raw:
        return raw
    token = str(row.get("doi") or row.get("source_id") or row.get("title") or "").strip()
    if token:
        return f"article_{_hash_id(token, length=14)}"
    return f"article_{_hash_id(json.dumps(row, sort_keys=True), length=14)}"


def _link_target_name(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    base = raw.split("|", 1)[0].strip()
    block = ""
    if "#" in base:
        base, block = base.split("#", 1)
        base = base.strip()
        block = block.strip()
    return base, block


def _edge_type_from_heading(heading: str) -> str:
    token = str(heading or "").strip().lower()
    if "uses method" in token:
        return "uses_method"
    if "observed under" in token:
        return "observed_under"
    if "transitions to" in token:
        return "transitions_to"
    if "mentions" in token:
        return "mentions"
    if "supporting evidence" in token:
        return "supports"
    return "link"


def export_obsidian_vault(run_dir: str | Path, out_dir: str | Path, cfg: RunConfig) -> VaultExportResult:
    run_path = Path(run_dir).expanduser().resolve()
    layout = ensure_run_layout(run_path)
    artifacts = layout["artifacts"]
    vault_root = Path(out_dir).expanduser().resolve()
    vault_root.mkdir(parents=True, exist_ok=True)

    run_id = _load_run_id(run_path)
    articles = _read_parquet_if_exists(artifacts / "articles.parquet")
    validated = _read_parquet_if_exists(artifacts / "validated_points.parquet")
    phase_events = _read_parquet_if_exists(artifacts / "phase_events.parquet")
    processor_eval_path = artifacts / "processor_eval_metrics.json"
    recipe_cards_path = artifacts / "recipe_cards.json"

    note_counts: dict[str, int] = {
        "runs": 0,
        "articles": 0,
        "concepts": 0,
        "events": 0,
        "recipes": 0,
        "models": 0,
    }
    warnings: list[str] = []
    links: list[dict[str, Any]] = []

    # Run note and index note.
    run_note_rel = f"Runs/{_slug(run_id, 'run')}.md"
    run_note = _note_path(vault_root, run_note_rel)
    run_note.write_text(
        "\n".join(
            [
                _frontmatter(
                    {
                        "type": "run",
                        "run_id": run_id,
                        "id": run_id,
                        "tags": ["oarp/run"],
                    }
                ),
                "",
                f"# Run {run_id}",
                "",
                "## Artifacts",
                f"- [run_metrics.json]({(artifacts / 'run_metrics.json').as_posix()})",
                f"- [full_workflow_metrics.json]({(artifacts / 'full_workflow_metrics.json').as_posix()})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    note_counts["runs"] += 1

    # Article notes with evidence block refs.
    article_key_to_rel: dict[str, str] = {}
    if articles.empty and not validated.empty:
        articles = pd.DataFrame([{"article_key": key} for key in sorted(validated.get("article_key", pd.Series([], dtype=str)).astype(str).unique())])
    for row in articles.to_dict(orient="records"):
        article_key = _article_key_from_row(row)
        article_rel = f"Articles/{_slug(article_key, 'article')}.md"
        article_key_to_rel[article_key] = article_rel
        article_note = _note_path(vault_root, article_rel)
        title = str(row.get("title") or article_key).strip() or article_key
        abstract = str(row.get("abstract") or "").strip()
        provider = str(row.get("provider") or "").strip()
        doi = str(row.get("doi") or "").strip()
        source_url = str(row.get("source_url") or row.get("oa_url") or "").strip()
        subset = validated[validated.get("article_key", pd.Series([], dtype=str)).astype(str) == article_key] if not validated.empty else pd.DataFrame()

        mention_lines: list[str] = []
        evidence_lines: list[str] = []
        for _, pt in subset.iterrows():
            entity = str(pt.get("entity") or pt.get("phase_label") or "").strip()
            if entity:
                concept_rel = f"Concepts/film/{_slug(entity)}.md"
                mention_lines.append(f"- [[{concept_rel[:-3]}]]")
                links.append(
                    {
                        "source_note": article_rel,
                        "target_note": concept_rel,
                        "edge_type": "mentions",
                        "source_point_id": str(pt.get("point_id") or ""),
                    }
                )
            point_id = str(pt.get("point_id") or _hash_id(article_key, entity, pt.get("locator")))
            snippet = str(pt.get("snippet") or "").strip()
            locator = str(pt.get("locator") or "").strip()
            citation = str(pt.get("citation_url") or "").strip()
            confidence = float(pt.get("confidence") or 0.0)
            evidence_lines.extend(
                [
                    f"- {snippet} ^point_{point_id}",
                    f"  - locator: {locator}",
                    f"  - citation: {citation}",
                    f"  - confidence: {confidence:.3f}",
                ]
            )

        body = [
            _frontmatter(
                {
                    "type": "article",
                    "run_id": run_id,
                    "id": article_key,
                    "doi": doi,
                    "provider": provider,
                    "source_url": source_url,
                    "tags": ["oarp/article", f"oarp/run/{run_id}"],
                }
            ),
            "",
            f"# {title}",
            "",
        ]
        if abstract:
            body.extend(["## Abstract", abstract, ""])
        if mention_lines:
            body.extend(["## Mentions", *sorted(set(mention_lines)), ""])
        if evidence_lines:
            body.extend(["## Validated evidence", *evidence_lines, ""])
        article_note.write_text("\n".join(body).strip() + "\n", encoding="utf-8")
        note_counts["articles"] += 1

    # Concept notes.
    concept_support: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if not validated.empty:
        for _, pt in validated.iterrows():
            point_id = str(pt.get("point_id") or "").strip()
            article_key = str(pt.get("article_key") or "").strip()
            article_rel = article_key_to_rel.get(article_key, "")

            entity = str(pt.get("entity") or pt.get("phase_label") or "").strip()
            if entity:
                concept_support.setdefault(("film", entity), []).append((article_rel, point_id))
            substrate = str(pt.get("substrate_material") or "").strip()
            if substrate:
                concept_support.setdefault(("substrate", substrate), []).append((article_rel, point_id))
            method = str(pt.get("method_family") or "").strip()
            if method:
                concept_support.setdefault(("method", method), []).append((article_rel, point_id))

    for (concept_type, label), refs in sorted(concept_support.items(), key=lambda item: (item[0][0], item[0][1].lower())):
        concept_rel = f"Concepts/{_slug(concept_type)}/{_slug(label)}.md"
        concept_note = _note_path(vault_root, concept_rel)
        support_lines: list[str] = []
        for article_rel, point_id in refs[:100]:
            if not article_rel or not point_id:
                continue
            support_lines.append(f"- ![[{article_rel[:-3]}#^point_{point_id}]]")
            links.append(
                {
                    "source_note": concept_rel,
                    "target_note": article_rel,
                    "edge_type": "supports",
                    "source_point_id": point_id,
                }
            )

        concept_note.write_text(
            "\n".join(
                [
                    _frontmatter(
                        {
                            "type": "concept",
                            "run_id": run_id,
                            "id": f"{concept_type}:{label}",
                            "concept_type": concept_type,
                            "aliases": [label],
                            "tags": ["oarp/concept", f"oarp/concept/{concept_type}"],
                        }
                    ),
                    "",
                    f"# {label}",
                    "",
                    "## Supporting evidence",
                    *support_lines,
                ]
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        note_counts["concepts"] += 1

    # Event notes.
    if not phase_events.empty:
        for idx, row in phase_events.head(5000).iterrows():
            event_id = str(row.get("event_id") or _hash_id("event", row.get("article_key"), idx))
            rel = f"Events/{_slug(event_id)}.md"
            note = _note_path(vault_root, rel)
            phase = str(row.get("phase_label") or row.get("entity") or "").strip()
            article_key = str(row.get("article_key") or "").strip()
            article_rel = article_key_to_rel.get(article_key, "")
            t = row.get("anneal_temperature_c")
            x = row.get("thickness_nm")
            lines = [
                _frontmatter(
                    {
                        "type": "event",
                        "run_id": run_id,
                        "id": event_id,
                        "tags": ["oarp/event"],
                    }
                ),
                "",
                f"# Event {event_id}",
                "",
                f"- phase: {phase}",
                f"- thickness_nm: {t if x is None else x}",
                f"- anneal_temperature_c: {t}",
            ]
            if article_rel:
                lines.extend(["", "## Evidence", f"- [[{article_rel[:-3]}]]"])
                links.append(
                    {
                        "source_note": rel,
                        "target_note": article_rel,
                        "edge_type": "observed_under",
                        "source_point_id": str(row.get("point_id") or ""),
                    }
                )
            note.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
            note_counts["events"] += 1

    # Recipe notes.
    if recipe_cards_path.exists():
        try:
            cards = json.loads(recipe_cards_path.read_text(encoding="utf-8"))
            if isinstance(cards, list):
                for item in cards[:200]:
                    if not isinstance(item, dict):
                        continue
                    recipe_id = str(item.get("recipe_id") or _hash_id("recipe", json.dumps(item, sort_keys=True)))
                    rel = f"Recipes/{_slug(recipe_id)}.md"
                    note = _note_path(vault_root, rel)
                    note.write_text(
                        "\n".join(
                            [
                                _frontmatter(
                                    {
                                        "type": "recipe",
                                        "run_id": run_id,
                                        "id": recipe_id,
                                        "tags": ["oarp/recipe"],
                                    }
                                ),
                                "",
                                f"# Recipe {recipe_id}",
                                "",
                                "```json",
                                json.dumps(item, indent=2, sort_keys=True),
                                "```",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    note_counts["recipes"] += 1
        except Exception:
            warnings.append("failed_to_parse_recipe_cards")

    # Model note.
    if processor_eval_path.exists():
        try:
            payload = json.loads(processor_eval_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        rel = "Models/processor_eval.md"
        note = _note_path(vault_root, rel)
        note.write_text(
            "\n".join(
                [
                    _frontmatter(
                        {
                            "type": "model_card",
                            "run_id": run_id,
                            "id": "processor_eval",
                            "tags": ["oarp/model"],
                        }
                    ),
                    "",
                    "# Processor Eval",
                    "",
                    "```json",
                    json.dumps(payload, indent=2, sort_keys=True),
                    "```",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        note_counts["models"] += 1

    # Vault index.
    index_path = _note_path(vault_root, "00_Index.md")
    index_path.write_text(
        "\n".join(
            [
                _frontmatter(
                    {
                        "type": "index",
                        "run_id": run_id,
                        "id": "vault_index",
                        "tags": ["oarp/index"],
                    }
                ),
                "",
                "# OARP Vault Index",
                "",
                f"- [[{run_note_rel[:-3]}]]",
                f"- Articles: {note_counts['articles']}",
                f"- Concepts: {note_counts['concepts']}",
                f"- Events: {note_counts['events']}",
                f"- Recipes: {note_counts['recipes']}",
                f"- Models: {note_counts['models']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    link_df = pd.DataFrame(links)
    if link_df.empty:
        link_df = pd.DataFrame(columns=["source_note", "target_note", "edge_type", "source_point_id"])
    link_index_path = artifacts / "vault_link_index.parquet"
    link_df.to_parquet(link_index_path, index=False)
    write_json(
        artifacts / "vault_export_manifest.json",
        {
            "created_at": now_iso(),
            "vault_path": str(vault_root),
            "run_id": run_id,
            "note_counts_by_type": note_counts,
            "link_count": int(len(link_df)),
            "warnings": warnings,
        },
    )
    return VaultExportResult(
        vault_path=vault_root,
        note_counts_by_type=note_counts,
        link_count=int(len(link_df)),
        warnings=warnings,
        index_path=index_path,
    )


def import_obsidian_vault(vault_dir: str | Path, run_dir: str | Path, cfg: RunConfig) -> VaultImportResult:
    _ = cfg
    vault_root = Path(vault_dir).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = ensure_run_layout(run_path)["artifacts"]
    baseline_path = artifacts / "vault_link_index.parquet"
    baseline = _read_parquet_if_exists(baseline_path)

    parsed_rows: list[dict[str, Any]] = []
    for note in sorted(vault_root.rglob("*.md")):
        rel_source = note.relative_to(vault_root).as_posix()
        heading = ""
        try:
            lines = note.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(lines, start=1):
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                heading = str(heading_match.group(1) or "").strip()
            for match in _WIKILINK_RE.finditer(line):
                token = str(match.group(2) or "")
                target, block = _link_target_name(token)
                if not target:
                    continue
                target_note = target if target.endswith(".md") else f"{target}.md"
                parsed_rows.append(
                    {
                        "source_note": rel_source,
                        "target_note": target_note,
                        "target_block": block,
                        "edge_type": _edge_type_from_heading(heading),
                        "line_no": line_no,
                    }
                )

    parsed = pd.DataFrame(parsed_rows)
    if parsed.empty:
        parsed = pd.DataFrame(columns=["source_note", "target_note", "target_block", "edge_type", "line_no"])

    parsed["edge_key"] = (
        parsed["source_note"].astype(str)
        + "->"
        + parsed["target_note"].astype(str)
        + "|"
        + parsed["edge_type"].astype(str)
    )
    baseline_keys: set[str] = set()
    if not baseline.empty and {"source_note", "target_note", "edge_type"}.issubset(set(baseline.columns)):
        baseline_keys = set(
            (
                baseline["source_note"].astype(str)
                + "->"
                + baseline["target_note"].astype(str)
                + "|"
                + baseline["edge_type"].astype(str)
            ).tolist()
        )

    parsed_keys = set(parsed["edge_key"].astype(str).tolist()) if not parsed.empty else set()
    add_keys = sorted(parsed_keys - baseline_keys)
    remove_keys = sorted(baseline_keys - parsed_keys)

    delta_rows: list[dict[str, Any]] = []
    parsed_by_key = {str(row["edge_key"]): row for row in parsed.to_dict(orient="records")}
    for key in add_keys:
        row = parsed_by_key.get(key, {})
        delta_rows.append(
            {
                "edge_key": key,
                "delta_type": "add",
                "source_note": str(row.get("source_note") or ""),
                "target_note": str(row.get("target_note") or ""),
                "edge_type": str(row.get("edge_type") or "link"),
                "confidence_hint": 0.65,
            }
        )
    for key in remove_keys:
        source_target, edge_type = key.rsplit("|", 1)
        source_note, target_note = source_target.split("->", 1)
        delta_rows.append(
            {
                "edge_key": key,
                "delta_type": "remove",
                "source_note": source_note,
                "target_note": target_note,
                "edge_type": edge_type,
                "confidence_hint": 0.35,
            }
        )
    deltas = pd.DataFrame(delta_rows)
    if deltas.empty:
        deltas = pd.DataFrame(columns=["edge_key", "delta_type", "source_note", "target_note", "edge_type", "confidence_hint"])

    parsed_path = artifacts / "vault_links.parquet"
    delta_path = artifacts / "vault_import_audit.parquet"
    soft_path = artifacts / "vault_soft_constraints.parquet"
    conflict_path = artifacts / "vault_conflicts.parquet"
    parsed.to_parquet(parsed_path, index=False)
    deltas.to_parquet(delta_path, index=False)

    soft = deltas.copy()
    if not soft.empty:
        soft["weight"] = soft["confidence_hint"].astype(float)
        soft["mode"] = "soft_supervision"
        soft["created_at"] = now_iso()
    soft.to_parquet(soft_path, index=False)

    conflicts = parsed[parsed["source_note"].astype(str) == parsed["target_note"].astype(str)].copy()
    conflicts.to_parquet(conflict_path, index=False)
    return VaultImportResult(
        parsed_links=parsed,
        link_deltas=deltas,
        soft_constraints_path=soft_path,
        conflicts=[f"self_links:{len(conflicts)}"] if len(conflicts) else [],
        parsed_links_path=parsed_path,
        delta_path=delta_path,
        audit_path=conflict_path,
    )


def apply_vault_soft_supervision(run_dir: str | Path, vault_links: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    _ = cfg
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = ensure_run_layout(run_path)["artifacts"]
    if vault_links.empty:
        out = pd.DataFrame(columns=["point_id", "vault_soft_supervision_score", "support_count", "penalty_count"])
        out.to_parquet(artifacts / "vault_point_supervision.parquet", index=False)
        return out

    work = vault_links.copy()
    if "delta_type" not in work.columns and "mode" not in work.columns:
        if "confidence_hint" not in work.columns:
            work["confidence_hint"] = 0.5
        work["delta_type"] = "add"

    def _point_from_target(value: Any) -> str:
        token = str(value or "").strip()
        if token.startswith("point_"):
            return token[6:]
        if token.startswith("^point_"):
            return token[7:]
        return ""

    if "target_block" not in work.columns:
        work["target_block"] = ""
    work["point_id"] = work["target_block"].map(_point_from_target)
    scoped = work[work["point_id"].astype(str).str.strip().ne("")].copy()
    if scoped.empty:
        out = pd.DataFrame(columns=["point_id", "vault_soft_supervision_score", "support_count", "penalty_count"])
        out.to_parquet(artifacts / "vault_point_supervision.parquet", index=False)
        return out

    scoped["sign"] = scoped["delta_type"].astype(str).map(lambda token: 1.0 if token == "add" else -1.0)
    scoped["weight"] = scoped.get("confidence_hint", 0.5).astype(float) * scoped["sign"]
    out = (
        scoped.groupby("point_id", as_index=False)
        .agg(
            vault_soft_supervision_score=("weight", "sum"),
            support_count=("sign", lambda s: int((s > 0).sum())),
            penalty_count=("sign", lambda s: int((s < 0).sum())),
        )
    )
    out.to_parquet(artifacts / "vault_point_supervision.parquet", index=False)

    proc_path = artifacts / "processor_training_rows.parquet"
    if proc_path.exists():
        try:
            proc = pd.read_parquet(proc_path)
            if "point_id" in proc.columns:
                merged = proc.merge(out, on="point_id", how="left")
                for col in ("vault_soft_supervision_score", "support_count", "penalty_count"):
                    if col in merged.columns:
                        merged[col] = merged[col].fillna(0.0)
                merged.to_parquet(proc_path, index=False)
        except Exception:
            pass
    return out


def benchmark_vault_alignment(run_dir: str | Path, vault_dir: str | Path, cfg: RunConfig) -> VaultBenchmarkResult:
    run_path = Path(run_dir).expanduser().resolve()
    artifacts = ensure_run_layout(run_path)["artifacts"]
    import_result = import_obsidian_vault(vault_dir=vault_dir, run_dir=run_path, cfg=cfg)
    parsed = import_result.parsed_links
    baseline = _read_parquet_if_exists(artifacts / "vault_link_index.parquet")
    notes = list(Path(vault_dir).expanduser().resolve().rglob("*.md"))

    parsed_set = set(
        (
            parsed.get("source_note", pd.Series([], dtype=str)).astype(str)
            + "->"
            + parsed.get("target_note", pd.Series([], dtype=str)).astype(str)
            + "|"
            + parsed.get("edge_type", pd.Series([], dtype=str)).astype(str)
        ).tolist()
    ) if not parsed.empty else set()
    base_set = set(
        (
            baseline.get("source_note", pd.Series([], dtype=str)).astype(str)
            + "->"
            + baseline.get("target_note", pd.Series([], dtype=str)).astype(str)
            + "|"
            + baseline.get("edge_type", pd.Series([], dtype=str)).astype(str)
        ).tolist()
    ) if not baseline.empty else set()

    tp = len(parsed_set & base_set)
    fp = len(parsed_set - base_set)
    fn = len(base_set - parsed_set)
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = 0.0 if precision + recall <= 0 else float((2.0 * precision * recall) / (precision + recall))
    vault_coverage = float(len({str(x) for x in parsed.get("source_note", pd.Series([], dtype=str)).astype(str).tolist()}) / max(1, len(notes)))
    vault_link_density = float(len(parsed_set) / max(1, len(notes)))
    vault_import_delta_rate = float((fp + fn) / max(1, len(base_set)))

    json_path = artifacts / "benchmark_vault.json"
    report_path = artifacts / "benchmark_vault.md"
    payload = {
        "created_at": now_iso(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "vault_coverage": vault_coverage,
        "vault_link_density": vault_link_density,
        "vault_import_delta_rate": vault_import_delta_rate,
        "parsed_links_path": str(import_result.parsed_links_path),
        "delta_path": str(import_result.delta_path),
    }
    write_json(json_path, payload)
    report_path.write_text(
        "\n".join(
            [
                "# Vault Benchmark",
                "",
                f"- precision: `{precision:.4f}`",
                f"- recall: `{recall:.4f}`",
                f"- f1: `{f1:.4f}`",
                f"- vault_coverage: `{vault_coverage:.4f}`",
                f"- vault_link_density: `{vault_link_density:.4f}`",
                f"- vault_import_delta_rate: `{vault_import_delta_rate:.4f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return VaultBenchmarkResult(
        precision=precision,
        recall=recall,
        f1=f1,
        vault_coverage=vault_coverage,
        vault_link_density=vault_link_density,
        vault_import_delta_rate=vault_import_delta_rate,
        report_path=report_path,
        json_path=json_path,
    )
