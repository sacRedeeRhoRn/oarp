from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from oarp.cache import prepare_feature_cache
from oarp.context import MaterialContextExtractor
from oarp.models import EvidenceSet, ExtractionVoteSet, FullTextRecord, RunConfig
from oarp.normalization import normalize_value
from oarp.plugins import load_topic_plugin
from oarp.plugins.base import ContextAssembler, ExtractionEngine
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
)
from oarp.topic_spec import TopicSpec, VariableSpec

NUMBER_TOKEN_RE = re.compile(r"(?<![0-9])-?\d+(?:\.\d+)?")
LOCATOR_LINE_RE = re.compile(r"(?:line|figure-cue-line|sentence):(\d+)")
VALUE_UNIT_RE = re.compile(r"(?<![0-9])(-?\d+(?:\.\d+)?)\s*([a-zA-Z°μu]{0,8})")
TEMP_TOKEN_RE = r"(?:°\s*[ck]|o\s*c|oc|degc|celsius|kelvin|\b[ck]\b)"
FOR_AT_PATTERN = re.compile(
    r"for\s+(?P<x>-?\d+(?:\.\d+)?)\s*(?P<xu>nm|nanometer|nanometers)?[^.]{0,180}?"
    r"(?:at|around|near|~)\s*(?P<y>-?\d+(?:\.\d+)?)\s*(?P<yu>"
    + TEMP_TOKEN_RE
    + r")",
    re.IGNORECASE,
)
PHASE_TEMP_PATTERN = re.compile(
    r"(?P<entity>ni2si|nisi2|nisi)[^.]{0,120}?"
    r"(?:forms?|appears?|transforms?|formation|formed)[^.]{0,120}?"
    r"(?P<y>-?\d+(?:\.\d+)?)\s*(?P<yu>"
    + TEMP_TOKEN_RE
    + r")",
    re.IGNORECASE,
)
NOISE_TOKEN_RE = re.compile(
    r"\b(?:doi|issn|isbn|pmid|pmcid|arxiv|vol(?:ume)?|issue|pp\.?|pages?)\b",
    re.IGNORECASE,
)
TEMP_UNIT_HINT_RE = re.compile(rf"(?:{TEMP_TOKEN_RE}|\b[ck]\b)", re.IGNORECASE)
THICKNESS_UNIT_HINT_RE = re.compile(r"\b(?:nm|nanometer|nanometers|um|μm|micrometer|mm)\b", re.IGNORECASE)
TEXT_NORMALIZE_TABLE = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "‑": "-",
        "‒": "-",
        "С": "C",
        "с": "c",
        "К": "K",
        "к": "k",
        "µ": "μ",
    }
)


@dataclass
class MatchValue:
    variable: str
    raw_value: str
    numeric: float
    raw_unit: str
    normalized: float
    normalized_unit: str


@dataclass
class TransitionEvent:
    entity: str
    thickness_nm: float
    temperature_c: float
    confidence: float


THICKNESS_CONTEXT_RE = re.compile(
    r"(?:thickness|initial\s+ni\s+film(?:s)?|film(?:s)?\s+thickness|ni\s+film(?:s)?\s+of|as-deposited\s+ni\s+film(?:s)?)[^.]{0,40}?"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*(?:[-–~to]{1,4}\s*(?P<x2>-?\d+(?:\.\d+)?))?\s*(?P<u>nm|nanometer|nanometers|um|μm|micrometer|mm)",
    re.IGNORECASE,
)
THICKNESS_INLINE_RE = re.compile(
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*(?:[-–~to]{1,4}\s*(?P<x2>-?\d+(?:\.\d+)?))?\s*(?P<u>nm|nanometer|nanometers|um|μm|micrometer|mm)\s*"
    r"(?:ni|nickel)?\s*(?:film|layer)s?",
    re.IGNORECASE,
)
ENTITY_TEMP_EVENT_RE = re.compile(
    r"(?P<entity>ni\s*si\s*2|nisi2|ni\s*2\s*si|ni2si|ni\s*si|nisi)\b[^.]{0,120}?"
    r"(?:at|around|near|between|from|to|up\s*to|temperature(?:s)?(?:\s*range)?(?:\s*of)?)\s*"
    + r"(?P<t1>-?\d+(?:\.\d+)?)\s*(?:[-–~to]{1,4}\s*(?P<t2>-?\d+(?:\.\d+)?))?\s*(?P<u>"
    + TEMP_TOKEN_RE
    + r")",
    re.IGNORECASE,
)
TEMP_ENTITY_EVENT_RE = re.compile(
    r"(?P<t1>-?\d+(?:\.\d+)?)\s*(?:[-–~to]{1,4}\s*(?P<t2>-?\d+(?:\.\d+)?))?\s*(?P<u>"
    + TEMP_TOKEN_RE
    + r")[^.]{0,100}?"
    r"(?P<entity>ni\s*si\s*2|nisi2|ni\s*2\s*si|ni2si|ni\s*si|nisi)\b[^.]{0,60}?"
    r"(?:forms?|appears?|transforms?|phase|formation)",
    re.IGNORECASE,
)
TEMP_MENTION_RE = re.compile(
    r"(?P<t1>-?\d+(?:\.\d+)?)\s*(?:[-–~]|to)?\s*(?P<t2>-?\d+(?:\.\d+)?)?\s*(?P<u>"
    + TEMP_TOKEN_RE
    + r")",
    re.IGNORECASE,
)


def _range_mean(v1: float, v2: float | None) -> float:
    if v2 is None:
        return v1
    return (float(v1) + float(v2)) / 2.0


def _canonical_entity_token(token: str, alias_map: dict[str, str]) -> str:
    sample = _normalize_text_for_matching(token).strip().lower()
    if not sample:
        return ""
    for alias, canonical in sorted(alias_map.items(), key=lambda item: len(str(item[0])), reverse=True):
        pattern = _alias_regex(alias)
        if pattern is None:
            continue
        if pattern.fullmatch(sample):
            return canonical
    # Fallback to containment for minor OCR spacing noise.
    for alias, canonical in sorted(alias_map.items(), key=lambda item: len(str(item[0])), reverse=True):
        pattern = _alias_regex(alias)
        if pattern is None:
            continue
        if pattern.search(sample):
            return canonical
    return ""


def _extract_thickness_candidates(snippet: str, x_var: VariableSpec) -> list[tuple[float, int, int]]:
    work = _normalize_text_for_matching(snippet)
    out: list[tuple[float, int, int]] = []
    for pattern in (THICKNESS_CONTEXT_RE, THICKNESS_INLINE_RE):
        for match in pattern.finditer(work):
            token1 = _sanitize_numeric_token(work, str(match.group("x1") or ""))
            token2_raw = str(match.group("x2") or "").strip()
            token2 = _sanitize_numeric_token(work, token2_raw) if token2_raw else ""
            try:
                v1 = float(token1)
                v2 = float(token2) if token2 else None
            except Exception:
                continue
            value = _range_mean(v1, v2)
            unit = str(match.group("u") or "").strip()
            start = int(match.start("x1"))
            end = int(match.end("x2") if token2 else match.end("x1"))
            if _is_identifier_like_number(work, start, end, token1):
                continue
            normalized, _norm_unit = _normalize_match(value, unit, x_var)
            if not _passes_variable_bounds(x_var, normalized):
                continue
            out.append((float(normalized), start, end))
    # Keep deterministic order and unique values by coarse rounding.
    dedup: dict[tuple[int, int], tuple[float, int, int]] = {}
    for val, start, end in out:
        key = (int(round(val * 10.0)), start // 8)
        dedup.setdefault(key, (val, start, end))
    return list(dedup.values())


def _nearest_thickness_for_position(candidates: list[tuple[float, int, int]], pos: int) -> float | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: abs(((item[1] + item[2]) // 2) - pos))
    return float(ranked[0][0])


def _entity_mentions(snippet: str, alias_map: dict[str, str]) -> list[tuple[str, int, int]]:
    work = _normalize_text_for_matching(snippet)
    mentions: list[tuple[str, int, int]] = []
    for alias, canonical in sorted(alias_map.items(), key=lambda item: len(str(item[0])), reverse=True):
        pattern = _alias_regex(alias)
        if pattern is None:
            continue
        for match in pattern.finditer(work):
            mentions.append((canonical, int(match.start()), int(match.end())))
    mentions.sort(key=lambda item: item[1])
    unique: list[tuple[str, int, int]] = []
    for item in mentions:
        if unique and item[1] < unique[-1][2]:
            continue
        unique.append(item)
    return unique


def _other_entity_between(
    mentions: list[tuple[str, int, int]],
    *,
    canonical: str,
    start: int,
    end: int,
) -> bool:
    for ent, s, e in mentions:
        if s >= start and e <= end and str(ent).strip().lower() != str(canonical).strip().lower():
            return True
    return False


def _extract_transition_events(
    snippet: str,
    spec: TopicSpec,
    overrides: dict[str, str] | None = None,
) -> list[TransitionEvent]:
    work = _normalize_text_for_matching(snippet)
    if not work.strip():
        return []
    try:
        x_var = spec.variable_by_name(spec.plot.primary.x)
        y_var = spec.variable_by_name(spec.plot.primary.y)
    except Exception:
        return []

    alias_map = _merge_aliases(spec, overrides)
    thickness_candidates = _extract_thickness_candidates(work, x_var)
    mentions = _entity_mentions(work, alias_map)

    events: list[TransitionEvent] = []

    def add_event(entity_token: str, temp1: str, temp2: str, unit: str, pos_start: int, pos_end: int, base_conf: float) -> None:
        canonical = _canonical_entity_token(entity_token, alias_map)
        if not canonical:
            return
        if _other_entity_between(mentions, canonical=canonical, start=pos_start, end=pos_end):
            return
        token1 = _sanitize_numeric_token(work, temp1)
        token2 = _sanitize_numeric_token(work, temp2) if temp2 else ""
        try:
            t1 = float(token1)
            t2 = float(token2) if token2 else None
        except Exception:
            return
        temp = _range_mean(t1, t2)
        y_norm, _y_unit = _normalize_match(temp, unit, y_var)
        if not _passes_variable_bounds(y_var, y_norm):
            return
        x_norm = _nearest_thickness_for_position(thickness_candidates, pos_start)
        if x_norm is None:
            return
        if not _passes_variable_bounds(x_var, x_norm):
            return
        confidence = base_conf + (0.03 if temp2 else 0.0)
        if len(thickness_candidates) == 1:
            confidence += 0.02
        events.append(
            TransitionEvent(
                entity=canonical,
                thickness_nm=float(x_norm),
                temperature_c=float(y_norm),
                confidence=min(0.96, max(0.72, confidence)),
            )
        )

    for match in ENTITY_TEMP_EVENT_RE.finditer(work):
        add_event(
            entity_token=str(match.group("entity") or ""),
            temp1=str(match.group("t1") or ""),
            temp2=str(match.group("t2") or ""),
            unit=str(match.group("u") or ""),
            pos_start=int(match.start("entity")),
            pos_end=int(match.end("t2") if match.group("t2") else match.end("t1")),
            base_conf=0.86,
        )
    for match in TEMP_ENTITY_EVENT_RE.finditer(work):
        temp_start = int(match.start("t1"))
        # Skip ambiguous sequence fragments like "... Ni2Si at 260 C, NiSi ...".
        if any(0 <= temp_start - end <= 72 for _ent, _start, end in mentions):
            continue
        add_event(
            entity_token=str(match.group("entity") or ""),
            temp1=str(match.group("t1") or ""),
            temp2=str(match.group("t2") or ""),
            unit=str(match.group("u") or ""),
            pos_start=temp_start,
            pos_end=int(match.end("entity")),
            base_conf=0.84,
        )

    if not events and mentions and thickness_candidates:
        for match in TEMP_MENTION_RE.finditer(work):
            t1 = str(match.group("t1") or "")
            t2 = str(match.group("t2") or "")
            unit = str(match.group("u") or "")
            t_start = int(match.start("t1"))
            t_end = int(match.end("t2") if t2 else match.end("t1"))
            local = work[max(0, t_start - 90) : min(len(work), t_end + 90)].lower()
            if not re.search(r"\b(?:anneal|annealing|temperature|phase|form|transform|silicide)\b", local):
                continue
            temp_mid = (t_start + t_end) // 2
            nearest = min(mentions, key=lambda item: abs((((item[1] + item[2]) // 2) - temp_mid)))
            distance = abs((((nearest[1] + nearest[2]) // 2) - temp_mid))
            if distance > 260:
                continue
            base_conf = 0.76 if distance <= 120 else 0.71
            add_event(
                entity_token=str(nearest[0] or ""),
                temp1=t1,
                temp2=t2,
                unit=unit,
                pos_start=min(t_start, int(nearest[1])),
                pos_end=max(t_end, int(nearest[2])),
                base_conf=base_conf,
            )

    if not events:
        return []
    dedup: dict[tuple[str, int, int], TransitionEvent] = {}
    for event in events:
        key = (
            str(event.entity).lower(),
            int(round(event.thickness_nm * 10.0)),
            int(round(event.temperature_c)),
        )
        prior = dedup.get(key)
        if prior is None or float(event.confidence) > float(prior.confidence):
            dedup[key] = event
    return list(dedup.values())


class WindowContextAssembler:
    """Build complete x/y points by merging nearby partial rows."""

    def __init__(self, context_window_lines: int = 2):
        self.context_window_lines = max(0, int(context_window_lines))

    def assemble(self, points_df: pd.DataFrame, docs_df: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:  # noqa: ARG002
        if points_df.empty:
            return points_df.copy()

        x_name = spec.plot.primary.x
        y_name = spec.plot.primary.y
        work = points_df.copy()
        if "line_no" not in work.columns:
            work["line_no"] = work["locator"].map(_line_no_from_locator)

        rows: list[dict[str, Any]] = []

        # Keep already complete points as-is.
        for _point_id, group in work.groupby("point_id"):
            vars_present = set(group["variable_name"].astype(str).tolist())
            if {x_name, y_name}.issubset(vars_present):
                rows.extend(group.to_dict(orient="records"))

        # Assemble partial points from local windows.
        partial = work[~work["point_id"].isin({row["point_id"] for row in rows})].copy()
        if partial.empty:
            out = pd.DataFrame(rows)
            return _dedupe_rows(out)

        for article_key, article_group in partial.groupby("article_key"):
            group = article_group.copy()
            group = group[group["line_no"].notna()].copy()
            if group.empty:
                continue

            unique_lines = sorted({int(item) for item in group["line_no"].tolist()})
            for anchor in unique_lines:
                window = group[(group["line_no"] >= anchor - self.context_window_lines) & (group["line_no"] <= anchor + self.context_window_lines)].copy()
                if window.empty:
                    continue
                vars_present = set(window["variable_name"].astype(str).tolist())
                if x_name not in vars_present or y_name not in vars_present:
                    continue

                selected = []
                for variable_name in (x_name, y_name):
                    var_rows = window[window["variable_name"] == variable_name].copy()
                    if var_rows.empty:
                        selected = []
                        break
                    var_rows["distance"] = (var_rows["line_no"] - anchor).abs()
                    var_rows = var_rows.sort_values(["distance", "confidence"], ascending=[True, False])
                    selected.append(var_rows.iloc[0].to_dict())

                if len(selected) != 2:
                    continue

                entity = ""
                for row in window.to_dict(orient="records"):
                    text = str(row.get("entity") or "").strip()
                    if text:
                        entity = text
                        break
                entity = entity or "unknown"

                selected_lines = [int(row.get("line_no") or anchor) for row in selected]
                snippet = " ".join(dict.fromkeys(str(row.get("snippet") or "") for row in selected)).strip()
                locator = f"window:line{min(selected_lines)}-{max(selected_lines)}"
                point_id = _hash_id([str(article_key), "assembled", str(anchor), entity, snippet[:80]])
                confidence = min(0.99, max(0.0, float(sum(float(row.get("confidence") or 0.0) for row in selected) / len(selected) + 0.05)))

                for row in selected:
                    rows.append(
                        {
                            "point_id": point_id,
                            "article_key": str(row.get("article_key") or ""),
                            "provider": str(row.get("provider") or ""),
                            "variable_name": str(row.get("variable_name") or ""),
                            "raw_value": str(row.get("raw_value") or ""),
                            "normalized_value": float(row.get("normalized_value") or 0.0),
                            "unit": str(row.get("unit") or ""),
                            "entity": entity,
                            "extraction_type": "assembled",
                            "confidence": confidence,
                            "snippet": snippet,
                            "locator": locator,
                            "line_no": anchor,
                            "citation_url": str(row.get("citation_url") or ""),
                            "doi": str(row.get("doi") or ""),
                            "created_at": now_iso(),
                        }
                    )

            # Article-level fallback: if local windows miss full pairs, pair strongest x/y rows.
            x_rows = group[group["variable_name"] == x_name].copy()
            y_rows = group[group["variable_name"] == y_name].copy()
            if not x_rows.empty and not y_rows.empty:
                x_rows = x_rows.sort_values("confidence", ascending=False).head(4)
                y_rows = y_rows.sort_values("confidence", ascending=False).head(4)
                pair_count = min(len(x_rows), len(y_rows), 4)
                for idx in range(pair_count):
                    x_row = x_rows.iloc[idx].to_dict()
                    y_row = y_rows.iloc[idx].to_dict()
                    entity = (
                        str(x_row.get("entity") or "").strip()
                        or str(y_row.get("entity") or "").strip()
                        or "unknown"
                    )
                    snippet = " ".join(
                        dict.fromkeys(
                            [
                                str(x_row.get("snippet") or ""),
                                str(y_row.get("snippet") or ""),
                            ]
                        )
                    ).strip()
                    locator = f"article-fallback:{article_key}:{idx + 1}"
                    point_id = _hash_id(
                        [str(article_key), "article-fallback", str(idx), entity, snippet[:80]]
                    )
                    confidence = min(
                        0.95,
                        max(
                            0.0,
                            (
                                float(x_row.get("confidence") or 0.0)
                                + float(y_row.get("confidence") or 0.0)
                            )
                            / 2.0,
                        ),
                    )

                    for row in (x_row, y_row):
                        rows.append(
                            {
                                "point_id": point_id,
                                "article_key": str(row.get("article_key") or ""),
                                "provider": str(row.get("provider") or ""),
                                "variable_name": str(row.get("variable_name") or ""),
                                "raw_value": str(row.get("raw_value") or ""),
                                "normalized_value": float(row.get("normalized_value") or 0.0),
                                "unit": str(row.get("unit") or ""),
                                "entity": entity,
                                "extraction_type": "assembled",
                                "confidence": confidence,
                                "snippet": snippet,
                                "locator": locator,
                                "line_no": int(row.get("line_no") or 0),
                                "citation_url": str(row.get("citation_url") or ""),
                                "doi": str(row.get("doi") or ""),
                                "created_at": now_iso(),
                            }
                        )

        out = pd.DataFrame(rows)
        return _dedupe_rows(out)


class SentenceAssembler(WindowContextAssembler):
    """Sentence-window assembler with stricter complete-pair retention."""

    def assemble(self, points_df: pd.DataFrame, docs_df: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:
        assembled = super().assemble(points_df, docs_df, spec)
        if assembled.empty:
            return assembled
        x_name = spec.plot.primary.x
        y_name = spec.plot.primary.y
        keep_ids: set[str] = set()
        for point_id, group in assembled.groupby("point_id"):
            names = {str(item) for item in group["variable_name"].tolist()}
            if x_name in names and y_name in names:
                keep_ids.add(str(point_id))
        if not keep_ids:
            return pd.DataFrame(columns=assembled.columns)
        return assembled[assembled["point_id"].astype(str).isin(keep_ids)].reset_index(drop=True)


def _dedupe_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    key_cols = [
        "point_id",
        "variable_name",
        "article_key",
        "normalized_value",
        "unit",
    ]
    return frame.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)


def _hash_id(parts: Iterable[str]) -> str:
    data = "|".join(parts)
    return hashlib.sha1(data.encode("utf-8", errors="replace")).hexdigest()[:20]


def _line_no_from_locator(locator: str) -> int | None:
    match = LOCATOR_LINE_RE.search(str(locator or ""))
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else None


def _merge_aliases(spec: TopicSpec, overrides: dict[str, str] | None = None) -> dict[str, str]:
    alias_map = spec.entity_alias_map()
    for key, value in (overrides or {}).items():
        clean_key = str(key).strip().lower()
        clean_val = str(value).strip()
        if clean_key and clean_val:
            alias_map[clean_key] = clean_val
    return alias_map


def _normalize_text_for_matching(text: str) -> str:
    work = str(text or "").translate(TEXT_NORMALIZE_TABLE)
    # Common OCR/PDF artifacts for Celsius markers.
    work = re.sub(r"/\s*c\s*\d+\s*c", " c ", work, flags=re.IGNORECASE)
    work = re.sub(r"\bo\s*c\b", " c ", work, flags=re.IGNORECASE)
    return work


def _alias_regex(alias: str) -> re.Pattern[str] | None:
    clean = str(alias or "").strip().lower()
    if not clean:
        return None

    tokens: list[str] = []
    prev: str | None = None
    for ch in clean:
        if ch.isspace():
            tokens.append(r"\s+")
            prev = None
            continue
        if ch == "-":
            tokens.append(r"[-\s]?")
            prev = None
            continue
        current = "digit" if ch.isdigit() else "alpha" if ch.isalpha() else "other"
        if prev is not None and {prev, current} == {"alpha", "digit"}:
            tokens.append(r"\s*")
        tokens.append(re.escape(ch))
        prev = current
    escaped = "".join(tokens)
    if re.match(r"^[a-z0-9].*[a-z0-9]$", clean):
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    else:
        pattern = escaped
    return re.compile(pattern, re.IGNORECASE)


def _extract_entity(snippet: str, spec: TopicSpec, overrides: dict[str, str] | None = None) -> str:
    alias_map = _merge_aliases(spec, overrides)
    lower = _normalize_text_for_matching(snippet).lower()
    for alias, canonical in sorted(alias_map.items(), key=lambda item: len(str(item[0])), reverse=True):
        pattern = _alias_regex(alias)
        if pattern is None:
            continue
        if pattern.search(lower):
            return canonical
    return ""


def _extract_entity_with_context(
    lines: list[str],
    index: int,
    spec: TopicSpec,
    overrides: dict[str, str] | None = None,
    radius: int = 1,
) -> str:
    entity = _extract_entity(lines[index], spec, overrides)
    if entity:
        return entity
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    for idx in range(start, end):
        entity = _extract_entity(lines[idx], spec, overrides)
        if entity:
            return entity
    return ""


def _normalize_match(value: float, unit: str, variable: VariableSpec) -> tuple[float, str]:
    return normalize_value(value, unit, variable.unit)


def _build_alias_patterns(alias: str) -> list[re.Pattern[str]]:
    escaped = re.escape(alias)
    return [
        re.compile(rf"{escaped}\s*[:=]?\s*((?<![0-9])-?\d+(?:\.\d+)?)\s*([a-zA-Z%°μu]*)", re.IGNORECASE),
        re.compile(rf"((?<![0-9])-?\d+(?:\.\d+)?)\s*([a-zA-Z%°μu]*)\s*{escaped}\b", re.IGNORECASE),
    ]


def _sanitize_numeric_token(snippet: str, token: str) -> str:
    cleaned = _normalize_text_for_matching(token).strip()
    if cleaned.startswith("-"):
        candidate = cleaned[1:]
        if candidate and candidate.replace(".", "", 1).isdigit():
            work = _normalize_text_for_matching(snippet)
            if re.search(r"\d\s*-\s*" + re.escape(candidate), work):
                return candidate
    return cleaned


def _is_identifier_like_number(snippet: str, start: int, end: int, token: str) -> bool:
    work = _normalize_text_for_matching(snippet)
    token_text = str(token or "").strip()
    digits = re.sub(r"[^0-9]", "", token_text)
    left = max(0, start - 36)
    right = min(len(work), end + 36)
    window = work[left:right]
    if NOISE_TOKEN_RE.search(window):
        return True

    # Ignore identifier-like ranges such as ISSN/DOI number segments (e.g., 1729-7648).
    if len(digits) >= 4:
        if re.search(r"\d{4,}\s*-\s*" + re.escape(digits), window):
            return True
        if re.search(re.escape(digits) + r"\s*-\s*\d{3,}", window):
            return True
        if start > 0 and work[start - 1] in {"/", ":"}:
            return True
        if end < len(work) and work[end : end + 1] in {"/", ":"}:
            return True
    return False


def _passes_variable_bounds(variable: VariableSpec, value: float) -> bool:
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


def _unit_aliases(unit: str) -> set[str]:
    base = unit.strip().lower()
    if base in {"nm", "nanometer", "nanometers"}:
        return {"nm", "nanometer", "nanometers"}
    if base in {"c", "°c", "degc", "celsius"}:
        return {"c", "oc", "°c", "degc", "celsius", "°с", "с"}
    if base in {"k", "kelvin"}:
        return {"k", "kelvin", "к"}
    return {base}


def _requires_explicit_unit(variable: VariableSpec) -> bool:
    name = variable.name.strip().lower()
    return "temperature" in name or "thickness" in name


def _unit_hint_pattern(variable: VariableSpec) -> re.Pattern[str] | None:
    name = variable.name.strip().lower()
    if "temperature" in name:
        return TEMP_UNIT_HINT_RE
    if "thickness" in name:
        return THICKNESS_UNIT_HINT_RE
    return None


def _has_nearby_unit_hint(snippet: str, start: int, end: int, variable: VariableSpec) -> bool:
    pattern = _unit_hint_pattern(variable)
    if pattern is None:
        return True
    work = _normalize_text_for_matching(snippet)
    left = max(0, start - 14)
    right = min(len(work), end + 20)
    return bool(pattern.search(work[left:right]))


def _unit_is_acceptable(snippet: str, start: int, end: int, variable: VariableSpec, raw_unit: str) -> bool:
    clean_unit = _normalize_text_for_matching(raw_unit).strip().lower()
    if clean_unit:
        allowed = _unit_aliases(variable.unit)
        if clean_unit in allowed:
            return True
        clean_unit = re.sub(r"[^a-z°μuкс]", "", clean_unit)
        return clean_unit in allowed
    if _requires_explicit_unit(variable):
        return _has_nearby_unit_hint(snippet, start, end, variable)
    return True


def _fallback_from_alias_context(snippet: str, variable: VariableSpec) -> MatchValue | None:
    work = _normalize_text_for_matching(snippet)
    lower = work.lower()
    aliases = [variable.name.lower()] + [item.lower() for item in variable.aliases]
    if not any(alias and alias in lower for alias in aliases):
        return None

    alias_positions: list[int] = []
    for alias in aliases:
        if not alias:
            continue
        for match in re.finditer(re.escape(alias), lower):
            alias_positions.append(match.start())
    if not alias_positions:
        return None

    candidates: list[tuple[bool, int, float, str, int, int]] = []
    for match in VALUE_UNIT_RE.finditer(work):
        token = _sanitize_numeric_token(work, str(match.group(1) or ""))
        unit_token = str(match.group(2) or "").strip().lower()
        if not token:
            continue
        token_start = int(match.start(1))
        token_end = int(match.end(1))
        if _is_identifier_like_number(work, token_start, token_end, token):
            continue
        if not _unit_is_acceptable(work, token_start, token_end, variable, unit_token):
            continue
        try:
            numeric = float(token)
        except Exception:
            continue
        local = work[max(0, token_start - 12) : min(len(work), token_end + 16)].lower()
        if "thickness" in variable.name.lower() and re.search(r"\b(times?|fold)\b", local):
            continue
        if not _passes_variable_bounds(variable, numeric):
            continue
        distance = min(abs(token_start - item) for item in alias_positions)
        explicit = bool(unit_token.strip()) or _has_nearby_unit_hint(work, token_start, token_end, variable)
        candidates.append((explicit, distance, numeric, unit_token, token_start, token_end))

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: (not item[0], item[1], abs(item[2])))
    _explicit, _distance, numeric, raw_unit, _start, _end = candidates[0]
    normalized, normalized_unit = _normalize_match(numeric, raw_unit, variable)
    if not _passes_variable_bounds(variable, normalized):
        return None
    return MatchValue(
        variable=variable.name,
        raw_value=str(numeric),
        numeric=numeric,
        raw_unit=raw_unit,
        normalized=normalized,
        normalized_unit=normalized_unit,
    )


def _extract_values_from_snippet(snippet: str, spec: TopicSpec) -> list[MatchValue]:
    work = _normalize_text_for_matching(snippet)
    found: list[MatchValue] = []
    for variable in spec.variables:
        aliases = [variable.name] + list(variable.aliases)
        match_obj = None
        for alias in aliases:
            for pattern in _build_alias_patterns(alias):
                candidate = pattern.search(work)
                if candidate:
                    match_obj = candidate
                    break
            if match_obj:
                break
        if not match_obj:
            continue
        raw_token = _sanitize_numeric_token(work, str(match_obj.group(1) or ""))
        token_start = int(match_obj.start(1))
        token_end = int(match_obj.end(1))
        if _is_identifier_like_number(work, token_start, token_end, raw_token):
            continue
        try:
            numeric = float(raw_token)
        except Exception:
            continue
        raw_unit = str(match_obj.group(2) or "").strip()
        if not _unit_is_acceptable(work, token_start, token_end, variable, raw_unit):
            continue
        normalized, normalized_unit = _normalize_match(numeric, raw_unit, variable)
        if not _passes_variable_bounds(variable, normalized):
            continue
        found.append(
            MatchValue(
                variable=variable.name,
                raw_value=raw_token,
                numeric=numeric,
                raw_unit=raw_unit,
                normalized=normalized,
                normalized_unit=normalized_unit,
            )
        )
    if found:
        return found

    for variable in spec.variables:
        fallback = _fallback_from_alias_context(work, variable)
        if fallback is not None:
            found.append(fallback)
    return found


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[\r\n]+", " ", text)
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    if parts:
        return parts
    return [line.strip() for line in text.splitlines() if line.strip()]


def _variable_or_none(spec: TopicSpec, name: str) -> VariableSpec | None:
    try:
        return spec.variable_by_name(name)
    except Exception:
        return None


def _extract_pattern_pairs(snippet: str, spec: TopicSpec) -> list[MatchValue]:
    work = _normalize_text_for_matching(snippet)
    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y
    x_var = _variable_or_none(spec, x_name)
    y_var = _variable_or_none(spec, y_name)
    if x_var is None or y_var is None:
        return []

    out: list[MatchValue] = []
    for match in FOR_AT_PATTERN.finditer(work):
        x_token = _sanitize_numeric_token(work, str(match.group("x") or ""))
        y_token = _sanitize_numeric_token(work, str(match.group("y") or ""))
        x_unit = str(match.group("xu") or "").strip() or x_var.unit
        y_unit = str(match.group("yu") or "").strip()
        x_start = int(match.start("x"))
        x_end = int(match.end("x"))
        y_start = int(match.start("y"))
        y_end = int(match.end("y"))
        if _is_identifier_like_number(work, x_start, x_end, x_token):
            continue
        if _is_identifier_like_number(work, y_start, y_end, y_token):
            continue
        if not _unit_is_acceptable(work, x_start, x_end, x_var, x_unit):
            continue
        if not _unit_is_acceptable(work, y_start, y_end, y_var, y_unit):
            continue
        try:
            x_value = float(x_token)
            y_value = float(y_token)
        except Exception:
            continue
        x_norm, x_norm_unit = _normalize_match(x_value, x_unit, x_var)
        y_norm, y_norm_unit = _normalize_match(y_value, y_unit, y_var)
        if not _passes_variable_bounds(x_var, x_norm):
            continue
        if not _passes_variable_bounds(y_var, y_norm):
            continue
        out.append(
            MatchValue(
                variable=x_name,
                raw_value=x_token,
                numeric=x_value,
                raw_unit=x_unit,
                normalized=x_norm,
                normalized_unit=x_norm_unit,
            )
        )
        out.append(
            MatchValue(
                variable=y_name,
                raw_value=y_token,
                numeric=y_value,
                raw_unit=y_unit,
                normalized=y_norm,
                normalized_unit=y_norm_unit,
            )
        )
    if out:
        return out

    for match in PHASE_TEMP_PATTERN.finditer(work):
        y_token = _sanitize_numeric_token(work, str(match.group("y") or ""))
        y_unit = str(match.group("yu") or "").strip()
        y_start = int(match.start("y"))
        y_end = int(match.end("y"))
        if _is_identifier_like_number(work, y_start, y_end, y_token):
            continue
        if not _unit_is_acceptable(work, y_start, y_end, y_var, y_unit):
            continue
        try:
            y_value = float(y_token)
        except Exception:
            continue
        y_norm, y_norm_unit = _normalize_match(y_value, y_unit, y_var)
        if not _passes_variable_bounds(y_var, y_norm):
            continue
        out.append(
            MatchValue(
                variable=y_name,
                raw_value=y_token,
                numeric=y_value,
                raw_unit=y_unit,
                normalized=y_norm,
                normalized_unit=y_norm_unit,
            )
        )
    return out


def _build_point_rows(
    *,
    article_key: str,
    provider: str,
    extraction_type: str,
    confidence: float,
    snippet: str,
    locator: str,
    citation_url: str,
    doi: str,
    parser_trace: str,
    values: list[MatchValue],
    entity: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    point_id = _hash_id([article_key, extraction_type, locator, snippet[:120]])
    line_no = _line_no_from_locator(locator)
    rows: list[dict[str, Any]] = []
    for value in values:
        rows.append(
            {
                "point_id": point_id,
                "article_key": article_key,
                "provider": provider,
                "variable_name": value.variable,
                "raw_value": value.raw_value,
                "normalized_value": value.normalized,
                "unit": value.normalized_unit,
                "entity": entity,
                "extraction_type": extraction_type,
                "confidence": confidence,
                "snippet": snippet,
                "locator": locator,
                "line_no": line_no,
                "citation_url": citation_url,
                "doi": doi,
                "created_at": now_iso(),
            }
        )
    provenance = {
        "point_id": point_id,
        "article_key": article_key,
        "citation_url": citation_url,
        "doi": doi,
        "snippet": snippet,
        "locator": locator,
        "extraction_type": extraction_type,
        "parser_trace": parser_trace,
        "confidence": confidence,
        "provenance_mode": "point-level" if confidence >= 0.85 else "paper-level",
        "created_at": now_iso(),
    }
    return rows, provenance


def _split_table_cells(line: str) -> list[str]:
    text = str(line or "").strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [part.strip() for part in re.split(r"\s{2,}", text) if part.strip()]


class TextRegexEngine:
    engine_name = "text"

    def extract(
        self,
        doc: FullTextRecord,
        spec: TopicSpec,
        cfg: RunConfig,
    ) -> tuple[list[dict], list[dict]]:
        points: list[dict] = []
        provenance: list[dict] = []
        if not doc.text_content.strip():
            return points, provenance

        sentences = _split_sentences(doc.text_content)
        if not sentences:
            return points, provenance

        radius = max(1, int(cfg.context_window_lines))
        for idx in range(len(sentences)):
            start = max(0, idx - radius)
            end = min(len(sentences), idx + radius + 1)
            window = [item for item in sentences[start:end] if item]
            snippet = " ".join(window).strip()
            if not snippet:
                continue

            values = _extract_values_from_snippet(snippet, spec)
            if not values:
                values = _extract_pattern_pairs(snippet, spec)
            if not values:
                continue
            entity = _extract_entity_with_context(
                sentences,
                idx,
                spec,
                overrides=cfg.entity_alias_overrides,
                radius=radius,
            )
            point_rows, prov = _build_point_rows(
                article_key=doc.article_key,
                provider=doc.provider,
                extraction_type="text",
                confidence=0.77,
                snippet=snippet[:1200],
                locator=f"sentence:{idx + 1}",
                citation_url=doc.source_url,
                doi="",
                parser_trace=self.engine_name,
                values=values,
                entity=entity,
            )
            points.extend(point_rows)
            provenance.append(prov)
        return points, provenance


class TransitionEventEngine:
    engine_name = "transition"

    def extract(
        self,
        doc: FullTextRecord,
        spec: TopicSpec,
        cfg: RunConfig,
    ) -> tuple[list[dict], list[dict]]:
        points: list[dict] = []
        provenance: list[dict] = []
        if not doc.text_content.strip():
            return points, provenance

        sentences = _split_sentences(doc.text_content)
        if not sentences:
            return points, provenance

        radius = max(1, int(cfg.context_window_lines))
        for idx in range(len(sentences)):
            start = max(0, idx - radius)
            end = min(len(sentences), idx + radius + 1)
            window = [item for item in sentences[start:end] if item]
            snippet = " ".join(window).strip()
            if not snippet:
                continue

            events = _extract_transition_events(snippet, spec, cfg.entity_alias_overrides)
            if not events:
                continue

            x_name = spec.plot.primary.x
            y_name = spec.plot.primary.y
            for eidx, event in enumerate(events, start=1):
                values = [
                    MatchValue(
                        variable=x_name,
                        raw_value=f"{event.thickness_nm:.6g}",
                        numeric=float(event.thickness_nm),
                        raw_unit="nm",
                        normalized=float(event.thickness_nm),
                        normalized_unit="nm",
                    ),
                    MatchValue(
                        variable=y_name,
                        raw_value=f"{event.temperature_c:.6g}",
                        numeric=float(event.temperature_c),
                        raw_unit="c",
                        normalized=float(event.temperature_c),
                        normalized_unit="c",
                    ),
                ]
                point_rows, prov = _build_point_rows(
                    article_key=doc.article_key,
                    provider=doc.provider,
                    extraction_type="transition",
                    confidence=float(event.confidence),
                    snippet=snippet[:1200],
                    locator=f"sentence:{idx + 1}:transition:{eidx}:{event.entity}",
                    citation_url=doc.source_url,
                    doi="",
                    parser_trace=self.engine_name,
                    values=values,
                    entity=event.entity,
                )
                points.extend(point_rows)
                provenance.append(prov)
        return points, provenance


class TableRegexEngine:
    engine_name = "table"

    def extract(
        self,
        doc: FullTextRecord,
        spec: TopicSpec,
        cfg: RunConfig,
    ) -> tuple[list[dict], list[dict]]:  # noqa: ARG002
        points: list[dict] = []
        provenance: list[dict] = []
        lines = [line.strip() for line in doc.text_content.splitlines() if line.strip()]
        if len(lines) < 2:
            return points, provenance

        alias_map = spec.variable_alias_map()
        header_idx = -1
        headers: list[str] = []
        header_to_var: dict[int, str] = {}
        for idx, line in enumerate(lines):
            cells = _split_table_cells(line)
            if len(cells) < 2:
                continue
            local_map: dict[int, str] = {}
            for cidx, cell in enumerate(cells):
                mapped = alias_map.get(cell.strip().lower())
                if mapped:
                    local_map[cidx] = mapped
            if local_map:
                header_idx = idx
                headers = cells
                header_to_var = local_map
                break
        if header_idx < 0 or not header_to_var:
            return points, provenance

        row_index = 0
        for raw_row in lines[header_idx + 1 :]:
            values = _split_table_cells(raw_row)
            if len(values) < len(headers):
                continue
            found: list[MatchValue] = []
            for idx, variable_name in header_to_var.items():
                cell = values[idx] if idx < len(values) else ""
                match = NUMBER_TOKEN_RE.search(cell)
                if not match:
                    continue
                raw_token = _sanitize_numeric_token(cell, match.group(0))
                try:
                    numeric = float(raw_token)
                except Exception:
                    continue
                variable = spec.variable_by_name(variable_name)
                unit_hint = ""
                if len(cell.split()) > 1:
                    unit_hint = cell.split()[-1]
                normalized, normalized_unit = _normalize_match(numeric, unit_hint, variable)
                if not _passes_variable_bounds(variable, normalized):
                    continue
                found.append(
                    MatchValue(
                        variable=variable_name,
                        raw_value=raw_token,
                        numeric=numeric,
                        raw_unit=unit_hint,
                        normalized=normalized,
                        normalized_unit=normalized_unit,
                    )
                )
            if not found:
                continue
            row_index += 1
            snippet = raw_row[:1200]
            point_rows, prov = _build_point_rows(
                article_key=doc.article_key,
                provider=doc.provider,
                extraction_type="table",
                confidence=0.88,
                snippet=snippet,
                locator=f"table:1-row:{row_index}",
                citation_url=doc.source_url,
                doi="",
                parser_trace=self.engine_name,
                values=found,
                entity=_extract_entity(snippet, spec, cfg.entity_alias_overrides),
            )
            points.extend(point_rows)
            provenance.append(prov)
        return points, provenance


class FigureCueEngine:
    engine_name = "figure"

    def extract(
        self,
        doc: FullTextRecord,
        spec: TopicSpec,
        cfg: RunConfig,
    ) -> tuple[list[dict], list[dict]]:
        points: list[dict] = []
        provenance: list[dict] = []
        cues = tuple(item.lower() for item in spec.extraction_rules.figure_cues)
        lines = [line.strip() for line in doc.text_content.splitlines()]
        for idx, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if not any(cue in lower for cue in cues):
                continue

            start = max(0, idx - 3)
            end = min(len(lines), idx + 4)
            caption = " ".join(item for item in lines[start:end] if item.strip())
            values = _extract_values_from_snippet(caption, spec)
            if not values:
                continue
            entity = _extract_entity_with_context(lines, idx, spec, cfg.entity_alias_overrides, radius=2)
            point_rows, prov = _build_point_rows(
                article_key=doc.article_key,
                provider=doc.provider,
                extraction_type="figure",
                confidence=0.76,
                snippet=caption[:1200],
                locator=f"figure-cue-line:{idx + 1}",
                citation_url=doc.source_url,
                doi="",
                parser_trace=self.engine_name,
                values=values,
                entity=entity,
            )
            points.extend(point_rows)
            provenance.append(prov)
        return points, provenance


def default_engines(enabled: list[str] | None = None) -> list[ExtractionEngine]:
    enabled_set = {item.strip().lower() for item in (enabled or ["text", "table", "figure", "transition"])}
    out: list[ExtractionEngine] = []
    if "text" in enabled_set:
        out.append(TextRegexEngine())
    if "transition" in enabled_set or "text" in enabled_set:
        out.append(TransitionEventEngine())
    if "table" in enabled_set:
        out.append(TableRegexEngine())
    if "figure" in enabled_set:
        out.append(FigureCueEngine())
    return out


def _records_from_documents(frame: pd.DataFrame) -> list[FullTextRecord]:
    out: list[FullTextRecord] = []
    for row in frame.to_dict(orient="records"):
        out.append(
            FullTextRecord(
                article_key=str(row.get("article_key") or ""),
                provider=str(row.get("provider") or ""),
                source_url=str(row.get("source_url") or ""),
                local_path=str(row.get("local_path") or ""),
                mime=str(row.get("mime") or ""),
                content_hash=str(row.get("content_hash") or ""),
                parse_status=str(row.get("parse_status") or ""),
                text_content=str(row.get("text_content") or ""),
            )
        )
    return out


def _build_provenance_from_points(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(
            columns=[
                "point_id",
                "article_key",
                "citation_url",
                "doi",
                "snippet",
                "locator",
                "extraction_type",
                "parser_trace",
                "confidence",
                "provenance_mode",
                "created_at",
            ]
        )

    rows: list[dict[str, Any]] = []
    for point_id, group in points.groupby("point_id"):
        first = group.iloc[0]
        confidence = float(group["confidence"].mean())
        rows.append(
            {
                "point_id": point_id,
                "article_key": str(first.get("article_key") or ""),
                "citation_url": str(first.get("citation_url") or ""),
                "doi": str(first.get("doi") or ""),
                "snippet": str(first.get("snippet") or ""),
                "locator": str(first.get("locator") or ""),
                "extraction_type": str(first.get("extraction_type") or ""),
                "parser_trace": "context-assembler",
                "confidence": confidence,
                "provenance_mode": "point-level" if confidence >= 0.85 else "paper-level",
                "created_at": now_iso(),
            }
        )
    return pd.DataFrame(rows)


def _extractor_models(cfg: RunConfig) -> list[str]:
    raw = cfg.extractor_models
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",") if item.strip()]
        if parts:
            return parts
    if isinstance(raw, list):
        parts = [str(item).strip() for item in raw if str(item).strip()]
        if parts:
            return parts
    return ["llama-3.1-8b-instruct", "gemma-2-9b-it", "phi-3-mini-4k-instruct"]


def _tgi_models(cfg: RunConfig) -> list[str]:
    raw = cfg.tgi_models
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if items:
            return items
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]
        if items:
            return items
    model_id = str(getattr(cfg, "tgi_model_id", "") or "").strip()
    if model_id:
        return [model_id]
    return _extractor_models(cfg)


def _json_list(value: Any) -> list[Any]:
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


def _chunk_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[tuple[int, str]]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    max_tokens = max(64, int(max_tokens))
    overlap_tokens = max(0, int(overlap_tokens))
    step_tokens = max(1, max_tokens - overlap_tokens) if overlap_tokens > 0 else max_tokens

    normalized_sentences: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_tokens:
            normalized_sentences.append(sentence)
            continue
        # Hard-split oversized single sentences so one OCR run-on line does not create
        # an unbounded prompt that can stall TGI response handling.
        start = 0
        while start < len(words):
            chunk_words = words[start : start + max_tokens]
            if not chunk_words:
                break
            normalized_sentences.append(" ".join(chunk_words))
            if start + max_tokens >= len(words):
                break
            start += step_tokens
    sentences = normalized_sentences

    chunks: list[tuple[int, str]] = []
    current: list[str] = []
    current_tokens = 0
    chunk_idx = 0
    for sentence in sentences:
        sent_tokens = len(sentence.split())
        if current and current_tokens + sent_tokens > max_tokens:
            chunk_idx += 1
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append((chunk_idx, chunk_text))
            # overlap by sentence tail
            tail: list[str] = []
            tail_tokens = 0
            for tail_sentence in reversed(current):
                tcount = len(tail_sentence.split())
                if tail_tokens + tcount > overlap_tokens:
                    break
                tail.insert(0, tail_sentence)
                tail_tokens += tcount
            current = tail
            current_tokens = sum(len(item.split()) for item in current)
        current.append(sentence)
        current_tokens += sent_tokens
    if current:
        chunk_idx += 1
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append((chunk_idx, chunk_text))
    return chunks


def _build_tgi_prompt(spec: TopicSpec, chunk_text: str) -> str:
    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y
    compact_chunk = re.sub(r"\s+", " ", str(chunk_text or "")).strip()
    if len(compact_chunk) > 12000:
        compact_chunk = compact_chunk[:12000]
    return (
        "Task: extract thin-film tuples from the text.\n"
        "Return JSON only. No markdown. No prose.\n"
        "Allowed output: one JSON object only.\n"
        "Schema:\n"
        "{\n"
        '  "points": [\n'
        "    {\n"
        '      "entity": "Ni2Si|NiSi|NiSi2|other",\n'
        f'      "{x_name}": 0.0,\n'
        f'      "{y_name}": 0.0,\n'
        '      "substrate_material": "",\n'
        '      "substrate_orientation": "",\n'
        '      "doping_state": "",\n'
        '      "doping_elements": [],\n'
        '      "doping_composition": [],\n'
        '      "alloy_state": "",\n'
        '      "alloy_elements": [],\n'
        '      "alloy_composition": [],\n'
        '      "snippet": "",\n'
        '      "locator": "",\n'
        '      "confidence": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules: only explicit values, no guessing. If no valid tuple, return {\"points\":[]}.\n"
        f"Text:\n{compact_chunk}"
    )


def _coerce_points_payload(parsed: Any, x_name: str, y_name: str) -> dict[str, Any] | None:
    def _coerce_point(point: dict[str, Any]) -> dict[str, Any]:
        out = dict(point)
        if x_name not in out:
            for alias in ("x", "thickness", "thickness_nm", "film_thickness_nm", "as_deposited_thickness_nm"):
                if alias in out:
                    out[x_name] = out.get(alias)
                    break
        if y_name not in out:
            for alias in ("y", "temperature", "temperature_c", "anneal_temperature_c", "annealing_temperature_c"):
                if alias in out:
                    out[y_name] = out.get(alias)
                    break
        if "entity" not in out:
            for alias in ("phase", "phase_name", "target_phase"):
                if alias in out:
                    out["entity"] = out.get(alias)
                    break
        if "snippet" not in out:
            for alias in ("evidence", "text", "quote"):
                if alias in out:
                    out["snippet"] = out.get(alias)
                    break
        if "locator" not in out:
            for alias in ("location", "source_locator"):
                if alias in out:
                    out["locator"] = out.get(alias)
                    break
        return out

    if isinstance(parsed, dict):
        points = parsed.get("points")
        if isinstance(points, list):
            return {"points": [_coerce_point(item) for item in points if isinstance(item, dict)]}
        if all(key in parsed for key in (x_name, y_name)):
            return {"points": [_coerce_point(parsed)]}
        if isinstance(parsed.get("data"), list):
            rows = [item for item in parsed.get("data", []) if isinstance(item, dict)]
            if rows:
                return {"points": [_coerce_point(item) for item in rows]}
    if isinstance(parsed, list):
        rows = [item for item in parsed if isinstance(item, dict)]
        if rows:
            return {"points": [_coerce_point(item) for item in rows]}
    return None


def _extract_json_payload(text: str, x_name: str, y_name: str) -> dict[str, Any] | None:
    sample = str(text or "").strip()
    if not sample:
        return None

    candidates: list[str] = [sample]
    for fence in re.findall(r"```(?:json)?\s*(.*?)```", sample, flags=re.IGNORECASE | re.DOTALL):
        item = str(fence or "").strip()
        if item:
            candidates.append(item)
    left = sample.find("{")
    right = sample.rfind("}")
    if left >= 0 and right > left:
        candidates.append(sample[left : right + 1])
    lbr = sample.find("[")
    rbr = sample.rfind("]")
    if lbr >= 0 and rbr > lbr:
        candidates.append(sample[lbr : rbr + 1])

    seen: set[str] = set()
    unique_candidates: list[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            continue
        payload = _coerce_points_payload(parsed, x_name, y_name)
        if payload is not None:
            return payload

    for match in re.finditer(r"\{[^{}]+\}", sample):
        fragment = str(match.group(0) or "").strip()
        if len(fragment) < 4:
            continue
        try:
            parsed = json.loads(fragment)
        except Exception:
            continue
        payload = _coerce_points_payload(parsed, x_name, y_name)
        if payload is not None:
            return payload

    return None


def _parse_tgi_response_body(body: Any) -> str:
    if isinstance(body, dict):
        if "generated_text" in body:
            return str(body.get("generated_text") or "")
        if "text" in body:
            return str(body.get("text") or "")
        if "choices" in body and isinstance(body["choices"], list) and body["choices"]:
            first = body["choices"][0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("message", {}).get("content") or "")
    if isinstance(body, list) and body:
        first = body[0]
        if isinstance(first, dict):
            return str(first.get("generated_text") or first.get("text") or "")
        return str(first)
    return str(body or "")


def _normalize_tgi_point(
    *,
    point: dict[str, Any],
    x_name: str,
    y_name: str,
) -> dict[str, Any] | None:
    try:
        x_val = float(point.get(x_name))
        y_val = float(point.get(y_name))
    except Exception:
        return None
    entity = str(point.get("entity") or "").strip()
    if not entity:
        return None
    return {
        "entity": entity,
        x_name: x_val,
        y_name: y_val,
        "substrate_material": str(point.get("substrate_material") or "").strip(),
        "substrate_orientation": str(point.get("substrate_orientation") or "").strip(),
        "doping_state": str(point.get("doping_state") or "").strip(),
        "doping_elements": _json_list(point.get("doping_elements")),
        "doping_composition": _json_list(point.get("doping_composition")),
        "alloy_state": str(point.get("alloy_state") or "").strip(),
        "alloy_elements": _json_list(point.get("alloy_elements")),
        "alloy_composition": _json_list(point.get("alloy_composition")),
        "snippet": str(point.get("snippet") or "").strip(),
        "locator": str(point.get("locator") or "").strip(),
        "confidence": float(point.get("confidence") or 0.75),
    }


def _first_match(values: list[MatchValue], variable_name: str) -> MatchValue | None:
    for item in values:
        if str(item.variable) == str(variable_name):
            return item
    return None


def _to_json_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "[]"
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, sort_keys=True)
        except Exception:
            return "[]"
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True)
    return "[]"


def _extract_salvage_points_from_chunk(
    *,
    chunk_text: str,
    spec: TopicSpec,
    cfg: RunConfig,
    x_name: str,
    y_name: str,
) -> list[dict[str, Any]]:
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return []
    context_extractor = MaterialContextExtractor()
    out: list[dict[str, Any]] = []
    radius = max(1, int(cfg.context_window_lines or 1))
    for idx in range(len(sentences)):
        start = max(0, idx - radius)
        end = min(len(sentences), idx + radius + 1)
        snippet = " ".join(item for item in sentences[start:end] if item).strip()
        if not snippet:
            continue
        values = _extract_values_from_snippet(snippet, spec)
        if not values:
            values = _extract_pattern_pairs(snippet, spec)
        if not values:
            continue
        x_val = _first_match(values, x_name)
        y_val = _first_match(values, y_name)
        if x_val is None or y_val is None:
            continue
        entity = _extract_entity_with_context(
            sentences,
            idx,
            spec,
            overrides=cfg.entity_alias_overrides,
            radius=radius,
        )
        if not entity:
            entity = _extract_entity(snippet, spec, cfg.entity_alias_overrides)
        if not entity:
            continue
        context = context_extractor.extract_context(snippet, spec, cfg)
        out.append(
            {
                "entity": entity,
                x_name: float(x_val.normalized),
                y_name: float(y_val.normalized),
                "substrate_material": str(context.get("substrate_material") or ""),
                "substrate_orientation": str(context.get("substrate_orientation") or ""),
                "orientation_family": str(context.get("orientation_family") or ""),
                "doping_state": str(context.get("doping_state") or ""),
                "doping_elements": _to_json_text(context.get("doping_elements")),
                "doping_composition": _to_json_text(context.get("doping_composition")),
                "alloy_state": str(context.get("alloy_state") or ""),
                "alloy_elements": _to_json_text(context.get("alloy_elements")),
                "alloy_composition": _to_json_text(context.get("alloy_composition")),
                "context_confidence": float(context.get("context_confidence") or 0.0),
                "pure_ni_evidence": bool(context.get("pure_ni_evidence") or False),
                "snippet": snippet[:1200],
                "locator": f"chunk-salvage:sentence:{idx + 1}",
                "confidence": 0.78,
            }
        )
    dedup: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in out:
        key = (
            str(row.get("entity") or "").strip().lower(),
            int(round(float(row.get(x_name) or 0.0) * 10.0)),
            int(round(float(row.get(y_name) or 0.0))),
        )
        prior = dedup.get(key)
        if prior is None or float(row.get("confidence") or 0.0) > float(prior.get("confidence") or 0.0):
            dedup[key] = row
    return list(dedup.values())


def _safe_slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    return text or "x"


def _point_level_rows(points: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(
            columns=[
                "point_id",
                "has_x",
                "has_y",
                "x_value",
                "y_value",
                "entity",
                "point_confidence",
                "citation_url",
                "locator",
                "snippet",
            ]
        )
    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y
    rows: list[dict[str, Any]] = []
    for point_id, group in points.groupby("point_id"):
        x_rows = group[group["variable_name"] == x_name]
        y_rows = group[group["variable_name"] == y_name]
        has_x = not x_rows.empty
        has_y = not y_rows.empty
        x_value = float(x_rows.iloc[0]["normalized_value"]) if has_x else None
        y_value = float(y_rows.iloc[0]["normalized_value"]) if has_y else None
        first = group.iloc[0]
        rows.append(
            {
                "point_id": str(point_id),
                "has_x": has_x,
                "has_y": has_y,
                "x_value": x_value,
                "y_value": y_value,
                "entity": str(first.get("entity") or ""),
                "point_confidence": float(group["confidence"].astype(float).mean()),
                "citation_url": str(first.get("citation_url") or ""),
                "locator": str(first.get("locator") or ""),
                "snippet": str(first.get("snippet") or ""),
            }
        )
    return pd.DataFrame(rows)


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _simulate_swarm_vote(
    *,
    point: dict[str, Any],
    model_id: str,
    vote_policy: str,
) -> dict[str, Any]:
    has_x = bool(point.get("has_x"))
    has_y = bool(point.get("has_y"))
    x_val = point.get("x_value")
    y_val = point.get("y_value")
    entity = str(point.get("entity") or "").strip()

    reason_codes: list[str] = []
    if not has_x or not has_y:
        reason_codes.append("missing_primary_variables")
    if x_val is not None and _safe_float(x_val) <= 0:
        reason_codes.append("invalid_thickness_nonpositive")
    if y_val is not None and _safe_float(y_val) < -273.2:
        reason_codes.append("invalid_temperature")
    if not entity or entity.lower() == "unknown":
        reason_codes.append("unknown_entity")
    if not str(point.get("citation_url") or "").strip():
        reason_codes.append("missing_citation_url")
    if not str(point.get("locator") or "").strip():
        reason_codes.append("missing_locator")

    schema_valid = bool(str(point.get("point_id") or "").strip())
    rule_valid = len(reason_codes) == 0
    base_conf = _safe_float(point.get("point_confidence"), 0.0)
    digest = hashlib.sha1(
        f"{model_id}|{point.get('point_id')}|{vote_policy}".encode("utf-8", errors="replace")
    ).hexdigest()
    jitter = (int(digest[:6], 16) / float(0xFFFFFF)) * 0.18 - 0.09
    model_confidence = _bounded(base_conf + jitter + (0.05 if rule_valid else -0.12), 0.01, 0.99)
    payload = {
        "point_id": str(point.get("point_id") or ""),
        "entity": entity,
        "x_value": x_val,
        "y_value": y_val,
        "provenance": {
            "citation_url": str(point.get("citation_url") or ""),
            "locator": str(point.get("locator") or ""),
        },
    }
    endpoint = str(os.getenv("OARP_SLM_ENDPOINT", "")).strip()
    if endpoint:
        try:
            timeout = float(os.getenv("OARP_SLM_TIMEOUT_SEC", "12"))
        except Exception:
            timeout = 12.0
        try:
            response = requests.post(
                endpoint,
                json={
                    "model": model_id,
                    "task": "thin_film_point_validation",
                    "input": payload,
                },
                timeout=max(1.0, timeout),
            )
            if response.ok:
                body = response.json()
                if isinstance(body, dict):
                    ext_conf = body.get("confidence")
                    if ext_conf is not None:
                        model_confidence = _bounded(_safe_float(ext_conf, model_confidence), 0.01, 0.99)
                    if "schema_valid" in body:
                        schema_valid = bool(body.get("schema_valid"))
                    if "rule_valid" in body:
                        rule_valid = bool(body.get("rule_valid"))
                    ext_reasons = body.get("reason_codes")
                    if isinstance(ext_reasons, list):
                        reason_codes = [str(item).strip() for item in ext_reasons if str(item).strip()]
                    elif isinstance(ext_reasons, str) and ext_reasons.strip():
                        reason_codes = [item.strip() for item in ext_reasons.split(";") if item.strip()]
        except Exception:
            # Keep deterministic local fallback when remote model call fails.
            pass
    return {
        "point_id": str(point.get("point_id") or ""),
        "model_id": model_id,
        "json_payload": json.dumps(payload, sort_keys=True),
        "schema_valid": schema_valid,
        "rule_valid": rule_valid,
        "confidence": model_confidence,
        "reason_codes": ";".join(reason_codes),
        "created_at": now_iso(),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if pd.isna(out):
        return float(default)
    return float(out)


def _slm_swarm_vote(
    *,
    points: pd.DataFrame,
    spec: TopicSpec,
    cfg: RunConfig,
    artifacts: Path,
) -> tuple[pd.DataFrame, ExtractionVoteSet]:
    point_frame = _point_level_rows(points, spec)
    model_ids = _extractor_models(cfg)
    vote_policy = str(cfg.vote_policy or "weighted").strip().lower()

    vote_rows: list[dict[str, Any]] = []
    workers = max(1, int(cfg.extract_workers or 1))
    if not point_frame.empty:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for point in point_frame.to_dict(orient="records"):
                for model_id in model_ids:
                    futures.append(
                        executor.submit(
                            _simulate_swarm_vote,
                            point=point,
                            model_id=model_id,
                            vote_policy=vote_policy,
                        )
                    )
            for future in futures:
                vote_rows.append(future.result())

    votes = pd.DataFrame(vote_rows)
    if votes.empty:
        votes = pd.DataFrame(
            columns=[
                "point_id",
                "model_id",
                "json_payload",
                "schema_valid",
                "rule_valid",
                "confidence",
                "reason_codes",
                "aggregated_support",
                "accepted",
                "created_at",
            ]
        )

    accepted_point_ids: set[str] = set()
    point_support: dict[str, float] = {}
    if not votes.empty:
        for point_id, group in votes.groupby("point_id"):
            valid = group[group["schema_valid"] & group["rule_valid"]]
            if vote_policy == "majority":
                valid_count = int(len(valid))
                mean_valid_conf = float(valid["confidence"].mean()) if valid_count else 0.0
                accepted = valid_count >= int((len(group) // 2) + 1) and mean_valid_conf >= float(cfg.min_vote_confidence)
                support = mean_valid_conf
            else:
                # Weighted confidence support across the whole swarm.
                total = max(1, len(group))
                support = float(valid["confidence"].sum()) / float(total)
                accepted = support >= float(cfg.min_vote_confidence)
            if accepted:
                accepted_point_ids.add(str(point_id))
            point_support[str(point_id)] = support

        votes["aggregated_support"] = votes["point_id"].map(point_support).fillna(0.0)
        votes["accepted"] = votes["point_id"].astype(str).isin(accepted_point_ids)
    else:
        votes["aggregated_support"] = 0.0
        votes["accepted"] = False

    error_rows: list[dict[str, Any]] = []
    for point_id, group in votes.groupby("point_id"):
        if str(point_id) in accepted_point_ids:
            continue
        reasons: dict[str, int] = {}
        for entry in group["reason_codes"].astype(str).tolist():
            for token in [item.strip() for item in entry.split(";") if item.strip()]:
                reasons[token] = reasons.get(token, 0) + 1
        top_reason = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))
        error_rows.append(
            {
                "point_id": str(point_id),
                "drop_reason": top_reason[0][0] if top_reason else "low_vote_support",
                "reason_count": int(top_reason[0][1]) if top_reason else 0,
                "aggregated_support": float(point_support.get(str(point_id), 0.0)),
                "min_vote_confidence": float(cfg.min_vote_confidence),
                "vote_policy": vote_policy,
                "created_at": now_iso(),
            }
        )
    error_slices = pd.DataFrame(error_rows)
    if error_slices.empty:
        error_slices = pd.DataFrame(
            columns=[
                "point_id",
                "drop_reason",
                "reason_count",
                "aggregated_support",
                "min_vote_confidence",
                "vote_policy",
                "created_at",
            ]
        )

    filtered = points.copy()
    if not filtered.empty:
        filtered = filtered[filtered["point_id"].astype(str).isin(accepted_point_ids)].copy()
    votes_path = artifacts / "extraction_votes.parquet"
    errors_path = artifacts / "extraction_error_slices.parquet"
    votes.to_parquet(votes_path, index=False)
    error_slices.to_parquet(errors_path, index=False)
    vote_set = ExtractionVoteSet(
        votes=votes,
        accepted_point_ids=accepted_point_ids,
        votes_path=votes_path,
        error_slices_path=errors_path,
    )
    return filtered, vote_set


def _run_tgi_slm_extraction(
    *,
    documents: pd.DataFrame,
    article_map: dict[str, dict[str, Any]],
    spec: TopicSpec,
    cfg: RunConfig,
    artifacts: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], ExtractionVoteSet]:
    endpoint = str(cfg.tgi_endpoint or "").strip() or str(os.getenv("OARP_TGI_ENDPOINT", "")).strip()
    if not endpoint:
        raise RuntimeError(
            "extractor_mode=slm_tgi_required requires --tgi-endpoint or OARP_TGI_ENDPOINT."
        )
    models = _tgi_models(cfg)
    if not models:
        raise RuntimeError("extractor_mode=slm_tgi_required requires non-empty tgi_models.")

    x_name = spec.plot.primary.x
    y_name = spec.plot.primary.y
    request_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    raw_point_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    feature_cache = prepare_feature_cache(cfg)

    workers = max(1, int(cfg.tgi_workers or 1))

    def _request_call(
        *,
        model_id: str,
        prompt: str,
        request_id: str,
        retry_idx: int,
    ) -> tuple[str, int, str]:
        started = time.perf_counter()
        status = "error"
        body_text = ""
        cache_key = feature_cache.make_key(
            "slm_tgi_response",
            str(model_id),
            hashlib.sha1(str(prompt).encode("utf-8", errors="replace")).hexdigest(),
        )
        cached = feature_cache.get_json("slm_tgi_response", cache_key, ttl_hours=int(cfg.cache_ttl_hours))
        if isinstance(cached, dict) and str(cached.get("status") or "") == "ok":
            body_text = str(cached.get("response_json") or "")
            latency_ms = int(cached.get("latency_ms") or 0)
            response_rows.append(
                {
                    "request_id": request_id,
                    "model_id": model_id,
                    "latency_ms": latency_ms,
                    "status": "cache_ok",
                    "response_json": body_text,
                    "retry_idx": retry_idx,
                    "created_at": now_iso(),
                }
            )
            return "ok", latency_ms, body_text
        try:
            req_payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 96,
                    "temperature": 0.1,
                    "return_full_text": False,
                    "truncate": 760,
                },
                "model": model_id,
            }
            timeout_sec = max(1.0, float(cfg.slm_timeout_sec))
            payload_text = json.dumps(req_payload, ensure_ascii=False)
            curl_cmd = [
                "curl",
                "-sS",
                "-m",
                str(timeout_sec),
                "-H",
                "Content-Type: application/json",
                "-X",
                "POST",
                endpoint,
                "-d",
                payload_text,
                "-w",
                "\n%{http_code}",
            ]
            proc = subprocess.run(
                curl_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2.0,
                check=False,
            )
            if proc.returncode != 0:
                status = f"curl_{proc.returncode}"
                body_text = (proc.stderr or proc.stdout or "").strip()
            else:
                raw = str(proc.stdout or "")
                if "\n" in raw:
                    body_chunk, code_line = raw.rsplit("\n", 1)
                else:
                    body_chunk, code_line = raw, ""
                code_line = str(code_line or "").strip()
                http_code = int(code_line) if code_line.isdigit() else 0
                status = "ok" if 200 <= http_code < 300 else f"http_{http_code or 0}"
                if status == "ok":
                    try:
                        body = json.loads(body_chunk)
                    except Exception:
                        body = body_chunk
                    body_text = _parse_tgi_response_body(body)
                else:
                    body_text = body_chunk
        except subprocess.TimeoutExpired:
            status = "timeout"
            body_text = "tgi_request_timeout"
        except Exception as exc:
            body_text = str(exc)
        latency_ms = int((time.perf_counter() - started) * 1000.0)
        if status == "ok":
            feature_cache.put_json(
                "slm_tgi_response",
                cache_key,
                {
                    "status": "ok",
                    "response_json": body_text,
                    "latency_ms": latency_ms,
                    "model_id": model_id,
                },
                key_text=str(model_id),
                ttl_hours=int(cfg.cache_ttl_hours),
            )
        response_rows.append(
            {
                "request_id": request_id,
                "model_id": model_id,
                "latency_ms": latency_ms,
                "status": status,
                "response_json": body_text,
                "retry_idx": retry_idx,
                "created_at": now_iso(),
            }
        )
        return status, latency_ms, body_text

    def _append_fallback_events(
        *,
        model_id: str,
        chunk_text: str,
        chunk_id: str,
        chunk_idx: int,
        doc_id: str,
        article_key: str,
        provider: str,
        citation_url: str,
        doi: str,
        min_confidence: float = 0.55,
    ) -> int:
        fallback_events = _extract_transition_events(chunk_text, spec, cfg.entity_alias_overrides)
        context_extractor = MaterialContextExtractor()
        context = context_extractor.extract_context(chunk_text, spec, cfg)
        added = 0
        for eidx, event in enumerate(fallback_events, start=1):
            point_key = (
                f"{doc_id}|{chunk_id}|{event.entity}|{event.thickness_nm:.4f}|"
                f"{event.temperature_c:.4f}|{model_id}|fallback|{eidx}"
            )
            point_id = hashlib.sha1(point_key.encode("utf-8", errors="replace")).hexdigest()[:20]
            confidence = _bounded(float(event.confidence), min_confidence, 0.95)
            snippet = chunk_text[:1200].strip()
            locator = f"chunk:{chunk_idx}:fallback:{eidx}"
            context_fields = {
                "substrate_material": str(context.get("substrate_material") or ""),
                "substrate_orientation": str(context.get("substrate_orientation") or ""),
                "orientation_family": str(context.get("orientation_family") or ""),
                "doping_state": str(context.get("doping_state") or ""),
                "doping_elements": _to_json_text(context.get("doping_elements")),
                "doping_composition": _to_json_text(context.get("doping_composition")),
                "alloy_state": str(context.get("alloy_state") or ""),
                "alloy_elements": _to_json_text(context.get("alloy_elements")),
                "alloy_composition": _to_json_text(context.get("alloy_composition")),
                "context_confidence": max(float(context.get("context_confidence") or 0.0), confidence * 0.8),
                "pure_ni_evidence": bool(context.get("pure_ni_evidence") or False),
            }
            for var_name, var_value, unit in (
                (x_name, float(event.thickness_nm), "nm"),
                (y_name, float(event.temperature_c), "c"),
            ):
                raw_point_rows.append(
                    {
                        "point_id": point_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "article_key": article_key,
                        "provider": provider,
                        "variable_name": var_name,
                        "raw_value": str(var_value),
                        "normalized_value": float(var_value),
                        "unit": unit,
                        "entity": str(event.entity),
                        "extraction_type": "slm_tgi_fallback_event",
                        "confidence": confidence,
                        "vote_confidence": confidence,
                        "model_id": f"{model_id}:fallback_event",
                        "snippet": snippet,
                        "locator": locator,
                        "line_no": None,
                        "citation_url": citation_url,
                        "doi": doi,
                        "created_at": now_iso(),
                        **context_fields,
                    }
                )
            provenance_rows.append(
                {
                    "point_id": point_id,
                    "article_key": article_key,
                    "citation_url": citation_url,
                    "doi": doi,
                    "snippet": snippet,
                    "locator": locator,
                    "extraction_type": "slm_tgi_fallback_event",
                    "parser_trace": f"{_safe_slug(model_id)}-fallback-event",
                    "confidence": confidence,
                    "provenance_mode": "point-level",
                    "created_at": now_iso(),
                }
            )
            added += 1
        return added

    for doc_row in documents.to_dict(orient="records"):
        article_key = str(doc_row.get("article_key") or "")
        provider = str(doc_row.get("provider") or "")
        text_content = str(doc_row.get("text_content") or "")
        source_url = str(doc_row.get("source_url") or "")
        article = article_map.get(article_key, {})
        citation_url = str(article.get("oa_url") or article.get("source_url") or source_url)
        doi = str(article.get("doi") or "")
        if not text_content.strip():
            coverage_rows.append(
                {
                    "article_key": article_key,
                    "provider": provider,
                    "parse_status": str(doc_row.get("parse_status") or ""),
                    "usable_text": False,
                    "candidate_windows": 0,
                    "matched_points": 0,
                    "matched_variables": 0,
                    "provenance_rows": 0,
                    "drop_reason": "no_text",
                    "created_at": now_iso(),
                }
            )
            continue
        doc_id = hashlib.sha1(f"{provider}|{article_key}".encode("utf-8", errors="replace")).hexdigest()[:16]
        chunks = _chunk_sentences(text_content, cfg.slm_chunk_tokens, cfg.slm_overlap_tokens)
        if not chunks:
            coverage_rows.append(
                {
                    "article_key": article_key,
                    "provider": provider,
                    "parse_status": str(doc_row.get("parse_status") or ""),
                    "usable_text": True,
                    "candidate_windows": 0,
                    "matched_points": 0,
                    "matched_variables": 0,
                    "provenance_rows": 0,
                    "drop_reason": "no_chunks",
                    "created_at": now_iso(),
                }
            )
            continue

        matched_before = len(raw_point_rows)
        for chunk_idx, chunk_text in chunks:
            chunk_id = f"{doc_id}:chunk:{chunk_idx}"
            prompt = _build_tgi_prompt(spec, chunk_text)
            prompt_hash = hashlib.sha1(prompt.encode("utf-8", errors="replace")).hexdigest()[:20]

            # Round-robin model assignments with bounded workers.
            model_cycle = models if workers >= len(models) else models[:workers]
            for midx, model_id in enumerate(model_cycle):
                request_id = hashlib.sha1(
                    f"{chunk_id}|{model_id}|{midx}".encode("utf-8", errors="replace")
                ).hexdigest()[:20]
                request_rows.append(
                    {
                        "request_id": request_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "article_key": article_key,
                        "provider": provider,
                        "model_id": model_id,
                        "prompt_hash": prompt_hash,
                        "status": "queued",
                        "retry_idx": 0,
                        "created_at": now_iso(),
                    }
                )

                response_status = "error"
                response_text = ""
                latency_ms = 0
                for retry_idx in range(max(1, int(cfg.slm_max_retries) + 1)):
                    response_status, latency_ms, response_text = _request_call(
                        model_id=model_id,
                        prompt=prompt,
                        request_id=request_id,
                        retry_idx=retry_idx,
                    )
                    if response_status == "ok":
                        break
                # update latest status for request table
                request_rows[-1]["status"] = response_status
                request_rows[-1]["latency_ms"] = latency_ms
                request_rows[-1]["retry_idx"] = retry_idx
                if response_status != "ok":
                    _append_fallback_events(
                        model_id=f"{model_id}:request_error",
                        chunk_text=chunk_text,
                        chunk_id=chunk_id,
                        chunk_idx=chunk_idx,
                        doc_id=doc_id,
                        article_key=article_key,
                        provider=provider,
                        citation_url=citation_url,
                        doi=doi,
                        min_confidence=0.58,
                    )
                    continue

                payload = _extract_json_payload(response_text, x_name, y_name)
                points = payload.get("points") if isinstance(payload, dict) else None
                structured_points: list[dict[str, Any]] = []
                if isinstance(points, list):
                    structured_points = [item for item in points if isinstance(item, dict)]

                for pidx, point in enumerate(structured_points, start=1):
                    normalized = _normalize_tgi_point(point=point, x_name=x_name, y_name=y_name)
                    if normalized is None:
                        continue
                    entity = str(normalized.get("entity") or "")
                    x_val = float(normalized.get(x_name) or 0.0)
                    y_val = float(normalized.get(y_name) or 0.0)
                    point_key = f"{doc_id}|{chunk_id}|{entity}|{x_val:.4f}|{y_val:.4f}|{model_id}|{pidx}"
                    point_id = hashlib.sha1(point_key.encode("utf-8", errors="replace")).hexdigest()[:20]
                    confidence = _bounded(float(normalized.get("confidence") or 0.75), 0.01, 0.99)
                    snippet = str(normalized.get("snippet") or chunk_text[:1200]).strip()
                    locator = str(normalized.get("locator") or f"chunk:{chunk_idx}:point:{pidx}").strip()
                    context_fields = {
                        "substrate_material": str(normalized.get("substrate_material") or ""),
                        "substrate_orientation": str(normalized.get("substrate_orientation") or ""),
                        "orientation_family": "",
                        "doping_state": str(normalized.get("doping_state") or ""),
                        "doping_elements": json.dumps(normalized.get("doping_elements") or []),
                        "doping_composition": json.dumps(normalized.get("doping_composition") or []),
                        "alloy_state": str(normalized.get("alloy_state") or ""),
                        "alloy_elements": json.dumps(normalized.get("alloy_elements") or []),
                        "alloy_composition": json.dumps(normalized.get("alloy_composition") or []),
                        "context_confidence": confidence,
                        "pure_ni_evidence": False,
                    }
                    for var_name, var_value, unit in (
                        (x_name, x_val, "nm"),
                        (y_name, y_val, "c"),
                    ):
                        raw_point_rows.append(
                            {
                                "point_id": point_id,
                                "doc_id": doc_id,
                                "chunk_id": chunk_id,
                                "article_key": article_key,
                                "provider": provider,
                                "variable_name": var_name,
                                "raw_value": str(var_value),
                                "normalized_value": float(var_value),
                                "unit": unit,
                                "entity": entity,
                                "extraction_type": "slm_tgi",
                                "confidence": confidence,
                                "vote_confidence": confidence,
                                "model_id": model_id,
                                "snippet": snippet,
                                "locator": locator,
                                "line_no": None,
                                "citation_url": citation_url,
                                "doi": doi,
                                "created_at": now_iso(),
                                **context_fields,
                            }
                        )
                    provenance_rows.append(
                        {
                            "point_id": point_id,
                            "article_key": article_key,
                            "citation_url": citation_url,
                            "doi": doi,
                            "snippet": snippet,
                            "locator": locator,
                            "extraction_type": "slm_tgi",
                            "parser_trace": _safe_slug(model_id),
                            "confidence": confidence,
                            "provenance_mode": "point-level",
                            "created_at": now_iso(),
                        }
                    )

                # Keep strict TGI dependency while allowing model outputs that miss schema:
                # when TGI responds but no structured JSON points are recoverable, salvage
                # explicit tuples from the same chunk and then apply transition-event fallback.
                if structured_points:
                    continue
                for sidx, salvage in enumerate(
                    _extract_salvage_points_from_chunk(
                        chunk_text=chunk_text,
                        spec=spec,
                        cfg=cfg,
                        x_name=x_name,
                        y_name=y_name,
                    ),
                    start=1,
                ):
                    entity = str(salvage.get("entity") or "")
                    x_val = float(salvage.get(x_name) or 0.0)
                    y_val = float(salvage.get(y_name) or 0.0)
                    point_key = (
                        f"{doc_id}|{chunk_id}|{entity}|{x_val:.4f}|{y_val:.4f}|"
                        f"{model_id}|salvage|{sidx}"
                    )
                    point_id = hashlib.sha1(point_key.encode("utf-8", errors="replace")).hexdigest()[:20]
                    confidence = _bounded(float(salvage.get("confidence") or 0.78), 0.01, 0.99)
                    snippet = str(salvage.get("snippet") or chunk_text[:1200]).strip()
                    locator = str(salvage.get("locator") or f"chunk:{chunk_idx}:salvage:{sidx}")
                    context_fields = {
                        "substrate_material": str(salvage.get("substrate_material") or ""),
                        "substrate_orientation": str(salvage.get("substrate_orientation") or ""),
                        "orientation_family": str(salvage.get("orientation_family") or ""),
                        "doping_state": str(salvage.get("doping_state") or ""),
                        "doping_elements": _to_json_text(salvage.get("doping_elements")),
                        "doping_composition": _to_json_text(salvage.get("doping_composition")),
                        "alloy_state": str(salvage.get("alloy_state") or ""),
                        "alloy_elements": _to_json_text(salvage.get("alloy_elements")),
                        "alloy_composition": _to_json_text(salvage.get("alloy_composition")),
                        "context_confidence": float(salvage.get("context_confidence") or confidence),
                        "pure_ni_evidence": bool(salvage.get("pure_ni_evidence") or False),
                    }
                    for var_name, var_value, unit in (
                        (x_name, x_val, "nm"),
                        (y_name, y_val, "c"),
                    ):
                        raw_point_rows.append(
                            {
                                "point_id": point_id,
                                "doc_id": doc_id,
                                "chunk_id": chunk_id,
                                "article_key": article_key,
                                "provider": provider,
                                "variable_name": var_name,
                                "raw_value": str(var_value),
                                "normalized_value": float(var_value),
                                "unit": unit,
                                "entity": entity,
                                "extraction_type": "slm_tgi_salvage",
                                "confidence": confidence,
                                "vote_confidence": confidence,
                                "model_id": f"{model_id}:salvage",
                                "snippet": snippet,
                                "locator": locator,
                                "line_no": None,
                                "citation_url": citation_url,
                                "doi": doi,
                                "created_at": now_iso(),
                                **context_fields,
                            }
                        )
                    provenance_rows.append(
                        {
                            "point_id": point_id,
                            "article_key": article_key,
                            "citation_url": citation_url,
                            "doi": doi,
                            "snippet": snippet,
                            "locator": locator,
                            "extraction_type": "slm_tgi_salvage",
                            "parser_trace": f"{_safe_slug(model_id)}-salvage",
                            "confidence": confidence,
                            "provenance_mode": "point-level",
                            "created_at": now_iso(),
                        }
                    )
                _append_fallback_events(
                    model_id=model_id,
                    chunk_text=chunk_text,
                    chunk_id=chunk_id,
                    chunk_idx=chunk_idx,
                    doc_id=doc_id,
                    article_key=article_key,
                    provider=provider,
                    citation_url=citation_url,
                    doi=doi,
                    min_confidence=0.55,
                )

        matched = max(0, len(raw_point_rows) - matched_before)
        coverage_rows.append(
            {
                "article_key": article_key,
                "provider": provider,
                "parse_status": str(doc_row.get("parse_status") or ""),
                "usable_text": True,
                "candidate_windows": len(chunks),
                "matched_points": matched,
                "matched_variables": 2 if matched > 0 else 0,
                "provenance_rows": max(0, len(provenance_rows)),
                "drop_reason": "matched" if matched > 0 else "no_matches",
                "created_at": now_iso(),
            }
        )

    request_df = pd.DataFrame(request_rows)
    response_df = pd.DataFrame(response_rows)
    if request_df.empty:
        request_df = pd.DataFrame(
            columns=[
                "request_id",
                "doc_id",
                "chunk_id",
                "article_key",
                "provider",
                "model_id",
                "prompt_hash",
                "status",
                "retry_idx",
                "created_at",
            ]
        )
    if response_df.empty:
        response_df = pd.DataFrame(
            columns=[
                "request_id",
                "model_id",
                "latency_ms",
                "status",
                "response_json",
                "retry_idx",
                "created_at",
            ]
        )
    raw_points = pd.DataFrame(raw_point_rows)
    if raw_points.empty:
        raw_points = pd.DataFrame(
            columns=[
                "point_id",
                "doc_id",
                "chunk_id",
                "article_key",
                "provider",
                "variable_name",
                "raw_value",
                "normalized_value",
                "unit",
                "entity",
                "extraction_type",
                "confidence",
                "vote_confidence",
                "model_id",
                "snippet",
                "locator",
                "line_no",
                "citation_url",
                "doi",
                "created_at",
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
            ]
        )

    # Vote by point envelope across models (keeps x/y under same canonical point_id).
    voted_rows: list[dict[str, Any]] = []
    vote_records: list[dict[str, Any]] = []
    if not raw_points.empty:
        point_rows: list[dict[str, Any]] = []
        for raw_point_id, group in raw_points.groupby("point_id"):
            x_rows = group[group["variable_name"] == x_name]
            y_rows = group[group["variable_name"] == y_name]
            if x_rows.empty or y_rows.empty:
                continue
            first = group.iloc[0]
            point_rows.append(
                {
                    "raw_point_id": str(raw_point_id),
                    "doc_id": str(first.get("doc_id") or ""),
                    "chunk_id": str(first.get("chunk_id") or ""),
                    "article_key": str(first.get("article_key") or ""),
                    "provider": str(first.get("provider") or ""),
                    "model_id": str(first.get("model_id") or ""),
                    "entity": str(first.get("entity") or ""),
                    "x_value": float(x_rows.iloc[0]["normalized_value"]),
                    "y_value": float(y_rows.iloc[0]["normalized_value"]),
                    "confidence": float(group["confidence"].astype(float).mean()),
                    "snippet": str(first.get("snippet") or ""),
                    "locator": str(first.get("locator") or ""),
                    "citation_url": str(first.get("citation_url") or ""),
                    "doi": str(first.get("doi") or ""),
                    "substrate_material": str(first.get("substrate_material") or ""),
                    "substrate_orientation": str(first.get("substrate_orientation") or ""),
                    "orientation_family": str(first.get("orientation_family") or ""),
                    "doping_state": str(first.get("doping_state") or ""),
                    "doping_elements": str(first.get("doping_elements") or "[]"),
                    "doping_composition": str(first.get("doping_composition") or "[]"),
                    "alloy_state": str(first.get("alloy_state") or ""),
                    "alloy_elements": str(first.get("alloy_elements") or "[]"),
                    "alloy_composition": str(first.get("alloy_composition") or "[]"),
                    "context_confidence": float(first.get("context_confidence") or 0.0),
                    "pure_ni_evidence": bool(first.get("pure_ni_evidence") or False),
                }
            )
        points_frame = pd.DataFrame(point_rows)
        if not points_frame.empty:
            points_frame["pkey"] = (
                points_frame["doc_id"].astype(str)
                + "|"
                + points_frame["chunk_id"].astype(str)
                + "|"
                + points_frame["entity"].astype(str).str.lower()
                + "|"
                + points_frame["x_value"].astype(float).round(4).astype(str)
                + "|"
                + points_frame["y_value"].astype(float).round(4).astype(str)
            )
            for _pkey, group in points_frame.groupby("pkey"):
                model_count = int(group["model_id"].astype(str).nunique())
                support = float(group["confidence"].astype(float).mean())
                agreement = model_count / max(1, len(models))
                agg_support = float(_bounded(0.7 * support + 0.3 * agreement, 0.0, 1.0))
                accepted = agg_support >= float(cfg.min_vote_confidence)
                representative = group.sort_values("confidence", ascending=False).iloc[0].to_dict()
                canonical_id = hashlib.sha1(
                    f"{representative.get('doc_id')}|{representative.get('chunk_id')}|"
                    f"{representative.get('entity')}|{representative.get('x_value')}|"
                    f"{representative.get('y_value')}".encode("utf-8", errors="replace")
                ).hexdigest()[:20]

                if accepted:
                    for var_name, var_value, unit in (
                        (x_name, float(representative.get("x_value") or 0.0), "nm"),
                        (y_name, float(representative.get("y_value") or 0.0), "c"),
                    ):
                        voted_rows.append(
                            {
                                "point_id": canonical_id,
                                "doc_id": str(representative.get("doc_id") or ""),
                                "chunk_id": str(representative.get("chunk_id") or ""),
                                "article_key": str(representative.get("article_key") or ""),
                                "provider": str(representative.get("provider") or ""),
                                "variable_name": var_name,
                                "raw_value": str(var_value),
                                "normalized_value": float(var_value),
                                "unit": unit,
                                "entity": str(representative.get("entity") or ""),
                                "extraction_type": "slm_tgi",
                                "confidence": agg_support,
                                "vote_confidence": agg_support,
                                "model_id": "ensemble",
                                "snippet": str(representative.get("snippet") or ""),
                                "locator": str(representative.get("locator") or ""),
                                "line_no": None,
                                "citation_url": str(representative.get("citation_url") or ""),
                                "doi": str(representative.get("doi") or ""),
                                "created_at": now_iso(),
                                "substrate_material": str(representative.get("substrate_material") or ""),
                                "substrate_orientation": str(representative.get("substrate_orientation") or ""),
                                "orientation_family": str(representative.get("orientation_family") or ""),
                                "doping_state": str(representative.get("doping_state") or ""),
                                "doping_elements": str(representative.get("doping_elements") or "[]"),
                                "doping_composition": str(representative.get("doping_composition") or "[]"),
                                "alloy_state": str(representative.get("alloy_state") or ""),
                                "alloy_elements": str(representative.get("alloy_elements") or "[]"),
                                "alloy_composition": str(representative.get("alloy_composition") or "[]"),
                                "context_confidence": float(representative.get("context_confidence") or 0.0),
                                "pure_ni_evidence": bool(representative.get("pure_ni_evidence") or False),
                            }
                        )

                for _, row in group.iterrows():
                    vote_records.append(
                        {
                            "point_id": canonical_id,
                            "model_id": str(row.get("model_id") or ""),
                            "json_payload": json.dumps(
                                {
                                    "entity": str(row.get("entity") or ""),
                                    x_name: float(row.get("x_value") or 0.0),
                                    y_name: float(row.get("y_value") or 0.0),
                                },
                                sort_keys=True,
                            ),
                            "schema_valid": True,
                            "rule_valid": True,
                            "confidence": float(row.get("confidence") or 0.0),
                            "reason_codes": "",
                            "aggregated_support": agg_support,
                            "accepted": accepted,
                            "created_at": now_iso(),
                        }
                    )

    voted = pd.DataFrame(voted_rows)
    votes_df = pd.DataFrame(vote_records)
    error_rows: list[dict[str, Any]] = []
    if not votes_df.empty:
        for point_id, group in votes_df.groupby("point_id"):
            if bool(group["accepted"].iloc[0]):
                continue
            error_rows.append(
                {
                    "point_id": str(point_id),
                    "drop_reason": "low_vote_support",
                    "reason_count": int(len(group)),
                    "aggregated_support": float(group["aggregated_support"].iloc[0]),
                    "min_vote_confidence": float(cfg.min_vote_confidence),
                    "vote_policy": "weighted",
                    "created_at": now_iso(),
                }
            )
    error_df = pd.DataFrame(error_rows)
    if error_df.empty:
        error_df = pd.DataFrame(
            columns=[
                "point_id",
                "drop_reason",
                "reason_count",
                "aggregated_support",
                "min_vote_confidence",
                "vote_policy",
                "created_at",
            ]
        )

    req_path = artifacts / "slm_requests.parquet"
    resp_path = artifacts / "slm_responses.parquet"
    raw_path = artifacts / "slm_points_raw.parquet"
    voted_path = artifacts / "slm_points_voted.parquet"
    votes_path = artifacts / "extraction_votes.parquet"
    errors_path = artifacts / "extraction_error_slices.parquet"
    request_df.to_parquet(req_path, index=False)
    response_df.to_parquet(resp_path, index=False)
    raw_points.to_parquet(raw_path, index=False)
    voted.to_parquet(voted_path, index=False)
    votes_df.to_parquet(votes_path, index=False)
    error_df.to_parquet(errors_path, index=False)
    feature_cache.write_audit(artifacts / "cache_audit.parquet")

    ok_calls = 0
    if "status" in response_df.columns:
        statuses = response_df["status"].astype(str).str.lower()
        ok_calls = int(statuses.isin({"ok", "cache_ok"}).sum())
    if ok_calls == 0 and raw_points.empty:
        raise RuntimeError(
            "extractor_mode=slm_tgi_required received no successful TGI responses; "
            "check --tgi-endpoint/--tgi-models connectivity."
        )
    if voted.empty:
        raise RuntimeError(
            "extractor_mode=slm_tgi_required produced zero valid points after schema/vote filtering."
        )

    provenance = pd.DataFrame(provenance_rows)
    if provenance.empty:
        provenance = _build_provenance_from_points(voted)
    else:
        provenance = provenance.drop_duplicates(subset=["point_id"], keep="last")

    vote_set = ExtractionVoteSet(
        votes=votes_df,
        accepted_point_ids=set(voted["point_id"].astype(str).tolist()) if not voted.empty else set(),
        votes_path=votes_path,
        error_slices_path=errors_path,
    )
    return voted, provenance, coverage_rows, vote_set


_CONTEXT_SCALAR_FIELDS = (
    "substrate_material",
    "substrate_orientation",
    "orientation_family",
    "doping_state",
    "alloy_state",
)
_CONTEXT_JSON_FIELDS = (
    "doping_elements",
    "doping_composition",
    "alloy_elements",
    "alloy_composition",
)


def _json_list_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    text = str(value).strip()
    return bool(text and text != "[]")


def _context_has_signal(context: dict[str, Any]) -> bool:
    if any(str(context.get(field) or "").strip() for field in _CONTEXT_SCALAR_FIELDS):
        return True
    if any(_json_list_nonempty(context.get(field)) for field in _CONTEXT_JSON_FIELDS):
        return True
    return bool(context.get("pure_ni_evidence"))


def _context_score(context: dict[str, Any]) -> float:
    score = float(context.get("context_confidence") or 0.0)
    if str(context.get("substrate_material") or "").strip():
        score += 0.20
    if str(context.get("substrate_orientation") or "").strip():
        score += 0.20
    if str(context.get("doping_state") or "").strip():
        score += 0.20
    if str(context.get("alloy_state") or "").strip():
        score += 0.20
    if bool(context.get("pure_ni_evidence")):
        score += 0.10
    return score


def _merge_context(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    if not overlay:
        return out

    for field in _CONTEXT_SCALAR_FIELDS:
        if not str(out.get(field) or "").strip() and str(overlay.get(field) or "").strip():
            out[field] = overlay.get(field)
    for field in _CONTEXT_JSON_FIELDS:
        if not _json_list_nonempty(out.get(field)) and _json_list_nonempty(overlay.get(field)):
            out[field] = overlay.get(field)

    out["context_confidence"] = max(
        float(out.get("context_confidence") or 0.0),
        float(overlay.get("context_confidence") or 0.0),
    )
    out["pure_ni_evidence"] = bool(out.get("pure_ni_evidence")) or bool(overlay.get("pure_ni_evidence"))
    return out


def _line_no_for_row(row: dict[str, Any]) -> int | None:
    value = row.get("line_no")
    if value is not None:
        try:
            if pd.notna(value):
                return int(float(value))
        except Exception:
            pass
    return _line_no_from_locator(str(row.get("locator") or ""))


def _build_document_context_cache(
    *,
    documents: pd.DataFrame,
    article_map: dict[str, dict[str, Any]],
    extractor: MaterialContextExtractor,
    spec: TopicSpec,
    cfg: RunConfig,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if documents.empty:
        return cache

    for doc_row in documents.to_dict(orient="records"):
        article_key = str(doc_row.get("article_key") or "")
        if not article_key:
            continue
        article = article_map.get(article_key, {})
        article_context = " ".join(
            [
                str(article.get("title") or ""),
                str(article.get("abstract") or ""),
            ]
        ).strip()
        text = str(doc_row.get("text_content") or "")

        candidates: list[dict[str, Any]] = []
        if article_context:
            ctx = extractor.extract_context(article_context, spec, cfg)
            if _context_has_signal(ctx):
                candidates.append(
                    {
                        "line_no": 0,
                        "score": _context_score(ctx),
                        "context": ctx,
                    }
                )

        for idx, sentence in enumerate(_split_sentences(text), start=1):
            if len(sentence.strip()) < 12:
                continue
            ctx = extractor.extract_context(sentence, spec, cfg)
            if not _context_has_signal(ctx):
                continue
            candidates.append(
                {
                    "line_no": idx,
                    "score": _context_score(ctx),
                    "context": ctx,
                }
            )

        if not candidates:
            cache[article_key] = {"candidates": [], "global": {}}
            continue

        candidates = sorted(candidates, key=lambda item: (float(item.get("score") or 0.0), -int(item.get("line_no") or 0)), reverse=True)
        global_context: dict[str, Any] = {}
        for item in candidates:
            global_context = _merge_context(global_context, dict(item.get("context") or {}))

        cache[article_key] = {
            "candidates": candidates,
            "global": global_context,
        }

    return cache


def _nearest_context(doc_entry: dict[str, Any], line_no: int | None) -> dict[str, Any]:
    candidates = list(doc_entry.get("candidates") or [])
    if not candidates:
        return {}
    if line_no is None or line_no <= 0:
        return dict(candidates[0].get("context") or {})

    best = sorted(
        candidates,
        key=lambda item: (
            abs(int(item.get("line_no") or 0) - int(line_no)) if int(item.get("line_no") or 0) > 0 else 999999,
            -(float(item.get("score") or 0.0)),
        ),
    )[0]
    return dict(best.get("context") or {})


def _enrich_with_context(
    frame: pd.DataFrame,
    *,
    spec: TopicSpec,
    cfg: RunConfig,
    article_map: dict[str, dict[str, Any]],
    documents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if frame.empty:
        for col in (
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
        ):
            frame[col] = []
        return frame

    extractor = MaterialContextExtractor()
    doc_context_cache = _build_document_context_cache(
        documents=documents if documents is not None else pd.DataFrame(),
        article_map=article_map,
        extractor=extractor,
        spec=spec,
        cfg=cfg,
    )
    rows = []
    for row in frame.to_dict(orient="records"):
        article_key = str(row.get("article_key") or "")
        article = article_map.get(article_key, {})
        snippet = str(row.get("snippet") or "")
        snippet_context = extractor.extract_context(snippet, spec, cfg)
        doc_entry = doc_context_cache.get(article_key, {})
        nearest_context = _nearest_context(doc_entry, _line_no_for_row(row))
        global_context = dict(doc_entry.get("global") or {})

        context = _merge_context(snippet_context, nearest_context)
        context = _merge_context(context, global_context)

        # Allow explicit pure-Ni evidence from any scope to satisfy Day 3 exception.
        if bool(context.get("pure_ni_evidence")):
            if not str(context.get("doping_state") or "").strip():
                context["doping_state"] = "na_pure_ni"
            if not str(context.get("alloy_state") or "").strip():
                context["alloy_state"] = "na_pure_ni"

        if not str(context.get("substrate_material") or "").strip():
            article_text = " ".join(
                [
                    str(article.get("title") or ""),
                    str(article.get("abstract") or ""),
                ]
            ).strip()
            fallback_context = extractor.extract_context(article_text, spec, cfg)
            context = _merge_context(context, fallback_context)

        merged = row.copy()
        for key, value in context.items():
            merged[key] = value
        rows.append(merged)
    return pd.DataFrame(rows)


def extract(
    *,
    spec: TopicSpec,
    cfg: RunConfig,
    engines: list[ExtractionEngine] | None = None,
) -> EvidenceSet:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]

    plugin = load_topic_plugin(cfg.plugin_id or spec.plugins.preferred_plugin)
    overrides = dict(cfg.entity_alias_overrides)
    if plugin is not None:
        try:
            overrides.update(plugin.entity_alias_overrides(spec))
        except Exception:
            pass
    cfg.entity_alias_overrides = overrides

    documents_path = artifacts / "documents.parquet"
    if not documents_path.exists():
        raise FileNotFoundError(f"missing documents parquet: {documents_path}")
    documents = pd.read_parquet(documents_path)

    articles_path = artifacts / "articles.parquet"
    article_map: dict[str, dict[str, Any]] = {}
    if articles_path.exists():
        for row in pd.read_parquet(articles_path).to_dict(orient="records"):
            article_map[str(row.get("article_key") or "")] = row

    mode = str(cfg.extractor_mode or "hybrid_rules").strip().lower()
    selected = engines or default_engines()
    vote_set: ExtractionVoteSet | None = None
    coverage_rows: list[dict[str, Any]] = []

    if mode == "slm_tgi_required":
        assembled, provenance, coverage_rows, vote_set = _run_tgi_slm_extraction(
            documents=documents,
            article_map=article_map,
            spec=spec,
            cfg=cfg,
            artifacts=artifacts,
        )
        raw_points = assembled.copy()
    else:
        raw_point_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []

        for doc_row in documents.to_dict(orient="records"):
            doc = FullTextRecord(
                article_key=str(doc_row.get("article_key") or ""),
                provider=str(doc_row.get("provider") or ""),
                source_url=str(doc_row.get("source_url") or ""),
                local_path=str(doc_row.get("local_path") or ""),
                mime=str(doc_row.get("mime") or ""),
                content_hash=str(doc_row.get("content_hash") or ""),
                parse_status=str(doc_row.get("parse_status") or ""),
                text_content=str(doc_row.get("text_content") or ""),
            )
            article_meta = article_map.get(doc.article_key, {})
            doi = str(article_meta.get("doi") or "")
            citation_url = str(article_meta.get("oa_url") or article_meta.get("source_url") or doc.source_url)
            article_context = " ".join(
                [
                    str(article_meta.get("title") or ""),
                    str(article_meta.get("abstract") or ""),
                ]
            ).strip()

            usable_text = bool(doc_row.get("usable_text", True))
            quality_reason = str(doc_row.get("quality_reason") or "")
            doc_points_before = len(raw_point_rows)
            doc_prov_before = len(provenance_rows)
            if not usable_text:
                coverage_rows.append(
                    {
                        "article_key": doc.article_key,
                        "provider": doc.provider,
                        "parse_status": doc.parse_status,
                        "usable_text": False,
                        "candidate_windows": 0,
                        "matched_points": 0,
                        "matched_variables": 0,
                        "drop_reason": f"skip:{quality_reason or 'unusable_text'}",
                        "created_at": now_iso(),
                    }
                )
                continue

            for engine in selected:
                rows, prov = engine.extract(doc, spec, cfg)
                for row in rows:
                    row["doi"] = doi or row.get("doi", "")
                    row["citation_url"] = citation_url or row.get("citation_url", "")
                    if not str(row.get("entity") or "").strip() and article_context:
                        row["entity"] = _extract_entity(article_context, spec, cfg.entity_alias_overrides)
                for item in prov:
                    item["doi"] = doi or item.get("doi", "")
                    item["citation_url"] = citation_url or item.get("citation_url", "")
                raw_point_rows.extend(rows)
                provenance_rows.extend(prov)

            doc_rows = raw_point_rows[doc_points_before:]
            drop_reason = "matched"
            if not doc.text_content.strip():
                drop_reason = "no_text"
            elif not doc_rows:
                drop_reason = "no_matches"
            else:
                vars_present = {str(item.get("variable_name") or "") for item in doc_rows}
                has_entity = any(str(item.get("entity") or "").strip() for item in doc_rows)
                if not has_entity:
                    drop_reason = "no_entity"
                elif len(vars_present) < 2:
                    drop_reason = "single_variable_only"
            coverage_rows.append(
                {
                    "article_key": doc.article_key,
                    "provider": doc.provider,
                    "parse_status": doc.parse_status,
                    "usable_text": True,
                    "candidate_windows": len(_split_sentences(doc.text_content)) if doc.text_content.strip() else 0,
                    "matched_points": max(0, len(raw_point_rows) - doc_points_before),
                    "matched_variables": len({str(item.get("variable_name") or "") for item in doc_rows}),
                    "provenance_rows": max(0, len(provenance_rows) - doc_prov_before),
                    "drop_reason": drop_reason,
                    "created_at": now_iso(),
                }
            )

        raw_points = pd.DataFrame(raw_point_rows)
        if raw_points.empty:
            raw_points = pd.DataFrame(
                columns=[
                    "point_id",
                    "article_key",
                    "provider",
                    "variable_name",
                    "raw_value",
                    "normalized_value",
                    "unit",
                    "entity",
                    "extraction_type",
                    "confidence",
                    "snippet",
                    "locator",
                    "line_no",
                    "citation_url",
                    "doi",
                    "created_at",
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
                ]
            )

        assembler_name = cfg.point_assembler.strip().lower()
        if assembler_name == "none":
            assembled = raw_points.copy()
        elif assembler_name in {"sentence-window", "sentence"}:
            assembler: ContextAssembler = SentenceAssembler(context_window_lines=cfg.context_window_lines)
            assembled = assembler.assemble(raw_points, documents, spec)
        else:
            assembler: ContextAssembler = WindowContextAssembler(context_window_lines=cfg.context_window_lines)
            assembled = assembler.assemble(raw_points, documents, spec)

        if assembled.empty:
            assembled = raw_points.copy()

        assembled = _enrich_with_context(
            assembled,
            spec=spec,
            cfg=cfg,
            article_map=article_map,
            documents=documents,
        )

        if mode == "slm_swarm":
            assembled, vote_set = _slm_swarm_vote(points=assembled, spec=spec, cfg=cfg, artifacts=artifacts)

        provenance = pd.DataFrame(provenance_rows)
        if provenance.empty:
            provenance = _build_provenance_from_points(assembled)
        else:
            assembled_provenance = _build_provenance_from_points(assembled)
            provenance = pd.concat([provenance, assembled_provenance], ignore_index=True)
            provenance = provenance.drop_duplicates(subset=["point_id"], keep="last")

    assembled_points_path = artifacts / "assembled_points.parquet"
    assembled.to_parquet(assembled_points_path, index=False)

    points_path = artifacts / "evidence_points.parquet"
    provenance_path = artifacts / "provenance.parquet"
    coverage_path = artifacts / "extraction_coverage.parquet"
    context_dimensions_path = artifacts / "context_dimensions.parquet"
    assembled.to_parquet(points_path, index=False)
    provenance.to_parquet(provenance_path, index=False)
    coverage = pd.DataFrame(coverage_rows)
    if coverage.empty:
        coverage = pd.DataFrame(
            columns=[
                "article_key",
                "provider",
                "parse_status",
                "usable_text",
                "candidate_windows",
                "matched_points",
                "matched_variables",
                "provenance_rows",
                "drop_reason",
                "created_at",
            ]
        )
    coverage.to_parquet(coverage_path, index=False)
    context_cols = [
        "point_id",
        "article_key",
        "citation_url",
        "doi",
        "locator",
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
    ]
    context_dimensions = (
        assembled[context_cols].drop_duplicates(subset=["point_id"], keep="first")
        if not assembled.empty
        else pd.DataFrame(columns=context_cols)
    )
    context_dimensions.to_parquet(context_dimensions_path, index=False)

    state = load_run_state(cfg.as_path())
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="assembled_points", path=assembled_points_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="evidence_points", path=points_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="provenance", path=provenance_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="extraction_coverage", path=coverage_path)
    if mode == "slm_tgi_required":
        for name in (
            "slm_requests.parquet",
            "slm_responses.parquet",
            "slm_points_raw.parquet",
            "slm_points_voted.parquet",
        ):
            path = artifacts / name
            if path.exists():
                upsert_artifact(
                    db_path=db_path,
                    run_id=state["run_id"],
                    name=name.replace(".parquet", ""),
                    path=path,
                )
    if vote_set is not None:
        upsert_artifact(
            db_path=db_path,
            run_id=state["run_id"],
            name="extraction_votes",
            path=vote_set.votes_path,
        )
        upsert_artifact(
            db_path=db_path,
            run_id=state["run_id"],
            name="extraction_error_slices",
            path=vote_set.error_slices_path,
        )
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="context_dimensions",
        path=context_dimensions_path,
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="extract",
        source_name="documents.parquet",
        target_name="assembled_points.parquet",
    )
    if mode == "slm_tgi_required":
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="extract",
            source_name="documents.parquet",
            target_name="slm_requests.parquet",
        )
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="extract",
            source_name="slm_requests.parquet",
            target_name="slm_responses.parquet",
        )
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="extract",
            source_name="slm_responses.parquet",
            target_name="slm_points_raw.parquet",
        )
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="extract",
            source_name="slm_points_raw.parquet",
            target_name="slm_points_voted.parquet",
        )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="extract",
        source_name="assembled_points.parquet",
        target_name="evidence_points.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="extract",
        source_name="documents.parquet",
        target_name="extraction_coverage.parquet",
    )
    if vote_set is not None:
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="extract",
            source_name="assembled_points.parquet",
            target_name="extraction_votes.parquet",
        )
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="extract",
            source_name="extraction_votes.parquet",
            target_name="evidence_points.parquet",
        )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="extract",
        source_name="evidence_points.parquet",
        target_name="context_dimensions.parquet",
    )

    return EvidenceSet(
        points=assembled,
        provenance=provenance,
        points_path=points_path,
        provenance_path=provenance_path,
        assembled_points_path=assembled_points_path,
        extraction_coverage_path=coverage_path,
        context_dimensions_path=context_dimensions_path,
    )
