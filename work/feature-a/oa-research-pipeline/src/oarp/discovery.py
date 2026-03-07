from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict
from hashlib import sha1
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from oarp.cache import CacheManager, prepare_feature_cache
from oarp.models import ArticleCandidate, ArticleIndex, RunConfig
from oarp.plugins import load_topic_plugin
from oarp.plugins.base import CandidateScorer, DiscoveryProvider, TopicPlugin
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
)
from oarp.topic_spec import TopicSpec

_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}

_PROVIDER_WEIGHT = {
    "openalex": 0.9,
    "europepmc": 0.85,
    "crossref": 0.7,
    "arxiv": 0.65,
    "localfs": 0.72,
}

_NI_MATERIAL_TOKENS = {"nickel", "ni"}
_NI_PROCESS_TOKENS = {"anneal", "annealing", "thermal", "temperature"}
_NI_SILICIDE_TOKENS = {"silicide", "ni2si", "nisi", "nisi2"}
_NI_NEGATIVE_TOKENS = {
    "diamond",
    "curie",
    "ferromagnetic",
    "ferromagnetism",
    "hfo2",
    "hafnia",
}
_DISCOVERY_CACHE: dict[str, CacheManager] = {}


class _CachedResponse:
    def __init__(self, *, status_code: int, text: str, url: str, headers: dict[str, Any] | None = None) -> None:
        self.status_code = int(status_code)
        self.text = str(text or "")
        self.url = str(url or "")
        self.headers = dict(headers or {})

    @property
    def ok(self) -> bool:
        return 200 <= int(self.status_code) < 300

    def json(self) -> Any:
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if self.ok:
            return
        raise requests.HTTPError(f"http_{self.status_code} for {self.url}")


class WeightedCandidateScorer:
    """Rank candidates using lexical overlap + source quality + optional plugin signal."""

    def __init__(self, plugin: TopicPlugin | None = None):
        self._plugin = plugin

    def score_candidates(self, frame: pd.DataFrame, spec: TopicSpec, query: str) -> pd.DataFrame:
        if frame.empty:
            out = frame.copy()
            out["discovery_score"] = []
            return out

        query_tokens = set(_tokenize(query))
        keyword_tokens = set(_tokenize(" ".join(spec.keywords)))
        entity_tokens = set(_tokenize(" ".join([row.name for row in spec.entities])))
        ni_gate_enabled = _should_enable_ni_gate(query=query, spec=spec)
        for row in spec.entities:
            entity_tokens.update(_tokenize(" ".join(row.aliases)))

        rows = []
        for _, row in frame.iterrows():
            title = str(row.get("title") or "")
            abstract = str(row.get("abstract") or "")
            provider = str(row.get("provider") or "").strip().lower()
            oa_url = str(row.get("oa_url") or "")

            title_tokens = set(_tokenize(title))
            abstract_tokens = set(_tokenize(abstract))
            merged_tokens = title_tokens | abstract_tokens

            title_overlap = _overlap_ratio(query_tokens, title_tokens)
            abstract_overlap = _overlap_ratio(query_tokens, abstract_tokens)
            keyword_overlap = _overlap_ratio(keyword_tokens, merged_tokens)
            entity_overlap = _overlap_ratio(entity_tokens, merged_tokens)
            provider_weight = float(_PROVIDER_WEIGHT.get(provider, 0.55))
            oa_quality = _oa_quality(oa_url)
            plugin_boost = 0.0
            if self._plugin is not None:
                try:
                    plugin_boost = float(self._plugin.candidate_boost(row.to_dict(), spec))
                except Exception:
                    plugin_boost = 0.0

            phrase_overlap = _phrase_overlap(query, f"{title} {abstract}")
            relevance_ok, gate_reasons = _passes_relevance_gate(
                title,
                abstract,
                enabled=ni_gate_enabled,
            )
            negative_penalty = _negative_context_penalty(
                title,
                abstract,
                enabled=ni_gate_enabled,
            )

            score = (
                0.34 * title_overlap
                + 0.22 * abstract_overlap
                + 0.16 * keyword_overlap
                + 0.14 * entity_overlap
                + 0.08 * phrase_overlap
                + 0.08 * provider_weight
                + 0.06 * oa_quality
                + plugin_boost
                - negative_penalty
            )
            if not relevance_ok:
                score -= 0.35

            scored = row.to_dict()
            scored.update(
                {
                    "title_overlap": round(title_overlap, 6),
                    "abstract_overlap": round(abstract_overlap, 6),
                    "keyword_overlap": round(keyword_overlap, 6),
                    "entity_overlap": round(entity_overlap, 6),
                    "phrase_overlap": round(phrase_overlap, 6),
                    "provider_weight": round(provider_weight, 6),
                    "oa_quality": round(oa_quality, 6),
                    "plugin_boost": round(plugin_boost, 6),
                    "negative_penalty": round(negative_penalty, 6),
                    "relevance_gate_ok": relevance_ok,
                    "gate_reasons": ",".join(gate_reasons),
                    "discovery_score": round(score, 6),
                }
            )
            rows.append(scored)

        out = pd.DataFrame(rows)
        out = out.sort_values(["discovery_score", "year"], ascending=[False, False], na_position="last")
        out = out.reset_index(drop=True)
        return out


class RollingSaturationController:
    def __init__(
        self,
        *,
        window_pages: int,
        min_yield: float,
        min_pages_before: int,
    ) -> None:
        self.window_pages = max(1, int(window_pages))
        self.min_yield = float(min_yield)
        self.min_pages_before = max(1, int(min_pages_before))

    def should_stop(self, key: str, page_metrics: list[dict[str, Any]]) -> bool:  # noqa: ARG002
        if len(page_metrics) < self.min_pages_before:
            return False
        window = page_metrics[-self.window_pages :]
        if not window:
            return False
        avg_yield = float(sum(float(item.get("page_yield") or 0.0) for item in window) / len(window))
        return avg_yield < self.min_yield


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [tok for tok in tokens if tok and tok not in _STOPWORDS and len(tok) > 1]


def _overlap_ratio(required: set[str], observed: set[str]) -> float:
    if not required or not observed:
        return 0.0
    hit = len(required & observed)
    return hit / max(len(required), 1)


def _phrase_overlap(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    if len(q_tokens) < 2:
        return 0.0
    phrases = {" ".join(q_tokens[idx : idx + 2]) for idx in range(len(q_tokens) - 1)}
    if not phrases:
        return 0.0
    normalized = re.sub(r"\s+", " ", text.lower())
    hits = sum(1 for phrase in phrases if phrase and phrase in normalized)
    return hits / len(phrases)


def _should_enable_ni_gate(*, query: str, spec: TopicSpec) -> bool:
    text = " ".join(
        [
            query,
            " ".join(spec.keywords),
            " ".join(entity.name for entity in spec.entities),
            " ".join(" ".join(entity.aliases) for entity in spec.entities),
        ]
    ).lower()
    return "nickel" in text or "ni2si" in text or "nisi2" in text or "silicide" in text


def _passes_relevance_gate(title: str, abstract: str, *, enabled: bool) -> tuple[bool, list[str]]:
    if not enabled:
        return True, []
    text = f"{title} {abstract}".lower()
    reasons: list[str] = []
    has_material = any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in _NI_MATERIAL_TOKENS)
    has_process = any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in _NI_PROCESS_TOKENS)
    has_silicide = any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in _NI_SILICIDE_TOKENS)

    # Day 3 broad-first policy: keep silicide-anchored records even when
    # process details are sparse, but reject records missing both material
    # and process cues to avoid unrelated chemistry/noise.
    if not has_silicide:
        reasons.append("missing_silicide_term")
    if not (has_material or has_process):
        reasons.append("missing_material_or_process_term")
    return (not reasons), reasons


def _negative_context_penalty(title: str, abstract: str, *, enabled: bool) -> float:
    if not enabled:
        return 0.0
    text = f"{title} {abstract}".lower()
    if any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in _NI_NEGATIVE_TOKENS):
        if not (
            any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in _NI_SILICIDE_TOKENS)
            and any(re.search(rf"\b{re.escape(tok)}\b", text) for tok in _NI_PROCESS_TOKENS)
        ):
            return 0.18
    return 0.0


def _oa_quality(url: str) -> float:
    clean = str(url or "").strip().lower()
    if not clean:
        return 0.0
    if clean.endswith(".pdf"):
        return 1.0
    if clean.endswith(".xml") or "format=xml" in clean:
        return 0.9
    if clean.endswith(".txt") or clean.endswith(".html"):
        return 0.8
    if "doi.org" in clean:
        return 0.55
    return 0.65


def _req(url: str, *, params: dict[str, Any], cfg: RunConfig) -> requests.Response:
    cache = _DISCOVERY_CACHE.get(str(cfg.as_path()))
    if cache is None:
        cache = prepare_feature_cache(cfg)
        _DISCOVERY_CACHE[str(cfg.as_path())] = cache
    cache_key = cache.make_key(
        "discovery_http",
        url,
        json.dumps(params, sort_keys=True),
        str(cfg.user_agent or ""),
    )
    cached = cache.get_json("discovery_http", cache_key, ttl_hours=int(cfg.cache_ttl_hours))
    if isinstance(cached, dict):
        status_code = int(cached.get("status_code") or 0)
        text = str(cached.get("text") or "")
        if status_code > 0 and text:
            resp = _CachedResponse(
                status_code=status_code,
                text=text,
                url=str(cached.get("url") or url),
                headers=cached.get("headers") if isinstance(cached.get("headers"), dict) else {},
            )
            resp.raise_for_status()
            return resp  # type: ignore[return-value]

    response = requests.get(
        url,
        params=params,
        timeout=cfg.timeout_sec,
        headers={"User-Agent": cfg.user_agent},
    )
    response.raise_for_status()
    cache.put_json(
        "discovery_http",
        cache_key,
        {
            "status_code": int(response.status_code),
            "text": str(response.text or ""),
            "url": str(response.url or url),
            "headers": dict(response.headers or {}),
        },
        key_text=f"{url}?{json.dumps(params, sort_keys=True)}",
        ttl_hours=int(cfg.cache_ttl_hours),
    )
    return response


def _normalize_language(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _clean_html(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    rows: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                rows.append((pos, word))
    rows.sort(key=lambda item: item[0])
    return " ".join(word for _, word in rows)


def _select_oa_url_from_openalex(item: dict[str, Any]) -> str:
    best = item.get("best_oa_location")
    if isinstance(best, dict):
        for key in ("pdf_url", "landing_page_url"):
            val = str(best.get(key) or "").strip()
            if val:
                return val
    open_access = item.get("open_access")
    if isinstance(open_access, dict):
        val = str(open_access.get("oa_url") or "").strip()
        if val:
            return val
    return ""


def _safe_year(item: dict[str, Any]) -> int | None:
    year = item.get("publication_year")
    if isinstance(year, int):
        return year
    if isinstance(year, str) and year.isdigit():
        return int(year)
    return None


def _normalize_article_type(value: str) -> str:
    raw = value.strip().lower().replace(" ", "-").replace("_", "-")
    if raw in {"article", "journal-article", "research-article"}:
        return "journal-article"
    if raw in {"posted-content", "preprint"}:
        return "preprint"
    return raw or "unknown"


class OpenAlexProvider:
    provider_name = "openalex"

    def search(self, query: str, spec: TopicSpec, cfg: RunConfig) -> list[ArticleCandidate]:
        rows, _next_token, _meta = self.search_page(query, spec, cfg, page_token=None)
        return rows

    def search_page(
        self,
        query: str,
        spec: TopicSpec,
        cfg: RunConfig,
        page_token: str | None,
    ) -> tuple[list[ArticleCandidate], str | None, dict[str, Any]]:
        cursor = page_token or "*"
        params = {
            "search": query,
            "cursor": cursor,
            "per-page": cfg.max_per_provider,
            "filter": "open_access.is_oa:true,has_fulltext:true",
            "select": (
                "id,doi,display_name,publication_year,type,language,"
                "best_oa_location,open_access,abstract_inverted_index,locations"
            ),
        }
        resp = _req("https://api.openalex.org/works", params=params, cfg=cfg)
        payload = resp.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        next_cursor = payload.get("meta", {}).get("next_cursor") if isinstance(payload, dict) else None

        out: list[ArticleCandidate] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            oa_url = _select_oa_url_from_openalex(item)
            doi = str(item.get("doi") or "").replace("https://doi.org/", "").strip().lower()
            source_id = str(item.get("id") or "").strip()
            article_type = _normalize_article_type(str(item.get("type") or ""))
            lang = _normalize_language(item.get("language"))
            best = item.get("best_oa_location") if isinstance(item.get("best_oa_location"), dict) else {}
            license_val = str(best.get("license") or "") if isinstance(best, dict) else ""
            abstract = _openalex_abstract(item.get("abstract_inverted_index"))
            if spec.eligibility.oa_required and not oa_url:
                continue
            out.append(
                ArticleCandidate(
                    provider=self.provider_name,
                    source_id=source_id,
                    doi=doi,
                    title=str(item.get("display_name") or "").strip(),
                    abstract=abstract,
                    year=_safe_year(item),
                    venue="",
                    article_type=article_type,
                    is_oa=True,
                    oa_url=oa_url,
                    source_url=f"https://api.openalex.org/works/{source_id}" if source_id else "",
                    license=license_val,
                    language=lang,
                    raw=item,
                )
            )
        return out, (str(next_cursor) if next_cursor else None), {"cursor": cursor}


class CrossrefProvider:
    provider_name = "crossref"

    def search(self, query: str, spec: TopicSpec, cfg: RunConfig) -> list[ArticleCandidate]:
        rows, _next_token, _meta = self.search_page(query, spec, cfg, page_token=None)
        return rows

    def search_page(
        self,
        query: str,
        spec: TopicSpec,
        cfg: RunConfig,
        page_token: str | None,
    ) -> tuple[list[ArticleCandidate], str | None, dict[str, Any]]:
        cursor = page_token or "*"
        params = {
            "query": query,
            "rows": cfg.max_per_provider,
            "cursor": cursor,
            "select": "DOI,title,issued,container-title,type,language,link,URL,license,abstract",
        }
        resp = _req("https://api.crossref.org/works", params=params, cfg=cfg)
        payload = resp.json()
        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        items = message.get("items", []) if isinstance(message, dict) else []
        next_cursor = message.get("next-cursor") if isinstance(message, dict) else None

        out: list[ArticleCandidate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            doi = str(item.get("DOI") or "").strip().lower()
            title = ""
            titles = item.get("title")
            if isinstance(titles, list) and titles:
                title = str(titles[0] or "").strip()
            year = None
            issued = item.get("issued") if isinstance(item.get("issued"), dict) else {}
            parts = issued.get("date-parts") if isinstance(issued, dict) else None
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                if isinstance(parts[0][0], int):
                    year = parts[0][0]
            links = item.get("link") if isinstance(item.get("link"), list) else []
            oa_url = ""
            for link in links:
                if not isinstance(link, dict):
                    continue
                candidate = str(link.get("URL") or "").strip()
                content_type = str(link.get("content-type") or "").lower()
                if candidate and ("pdf" in content_type or "xml" in content_type or "text" in content_type):
                    oa_url = candidate
                    break
            if not oa_url:
                oa_url = str(item.get("URL") or "").strip()
            license_items = item.get("license") if isinstance(item.get("license"), list) else []
            license_url = ""
            if license_items and isinstance(license_items[0], dict):
                license_url = str(license_items[0].get("URL") or "").strip()
            is_oa = bool(oa_url)
            if spec.eligibility.oa_required and not is_oa:
                continue
            out.append(
                ArticleCandidate(
                    provider=self.provider_name,
                    source_id=doi,
                    doi=doi,
                    title=title,
                    abstract=_clean_html(str(item.get("abstract") or "")),
                    year=year,
                    venue=(item.get("container-title") or [""])[0] if item.get("container-title") else "",
                    article_type=_normalize_article_type(str(item.get("type") or "")),
                    is_oa=is_oa,
                    oa_url=oa_url,
                    source_url=str(item.get("URL") or "").strip(),
                    license=license_url,
                    language=_normalize_language(item.get("language")),
                    raw=item,
                )
            )

        next_token = str(next_cursor) if next_cursor and next_cursor != cursor else None
        return out, next_token, {"cursor": cursor}


class ArxivProvider:
    provider_name = "arxiv"

    def search(self, query: str, spec: TopicSpec, cfg: RunConfig) -> list[ArticleCandidate]:
        rows, _next_token, _meta = self.search_page(query, spec, cfg, page_token=None)
        return rows

    def search_page(
        self,
        query: str,
        spec: TopicSpec,
        cfg: RunConfig,
        page_token: str | None,
    ) -> tuple[list[ArticleCandidate], str | None, dict[str, Any]]:
        start = int(page_token or 0)
        encoded = quote(query)
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{encoded}&start={start}&max_results={cfg.max_per_provider}&sortBy=relevance"
        )
        response = requests.get(url, timeout=cfg.timeout_sec, headers={"User-Agent": cfg.user_agent})
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        out: list[ArticleCandidate] = []
        for entry in root.findall("atom:entry", ns):
            source_id = str(entry.findtext("atom:id", default="", namespaces=ns)).strip()
            title = " ".join(str(entry.findtext("atom:title", default="", namespaces=ns)).split())
            abstract = " ".join(str(entry.findtext("atom:summary", default="", namespaces=ns)).split())
            published = str(entry.findtext("atom:published", default="", namespaces=ns)).strip()
            year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None
            doi = str(entry.findtext("arxiv:doi", default="", namespaces=ns)).strip().lower()
            link = ""
            for node in entry.findall("atom:link", ns):
                href = str(node.attrib.get("href") or "").strip()
                rel = str(node.attrib.get("rel") or "").strip()
                if href and (not rel or rel == "alternate"):
                    link = href
                    break
            out.append(
                ArticleCandidate(
                    provider=self.provider_name,
                    source_id=source_id,
                    doi=doi,
                    title=title,
                    abstract=abstract,
                    year=year,
                    venue="arXiv",
                    article_type="preprint",
                    is_oa=True,
                    oa_url=link,
                    source_url=source_id,
                    license="",
                    language="en",
                    raw={"entry_id": source_id},
                )
            )

        next_start = start + len(out)
        next_token = str(next_start) if len(out) >= cfg.max_per_provider else None
        return out, next_token, {"start": start}


class EuropePMCProvider:
    provider_name = "europepmc"

    def search(self, query: str, spec: TopicSpec, cfg: RunConfig) -> list[ArticleCandidate]:
        rows, _next_token, _meta = self.search_page(query, spec, cfg, page_token=None)
        return rows

    def search_page(
        self,
        query: str,
        spec: TopicSpec,
        cfg: RunConfig,
        page_token: str | None,
    ) -> tuple[list[ArticleCandidate], str | None, dict[str, Any]]:
        page = int(page_token or 1)
        full_query = f"({query}) AND OPEN_ACCESS:y"
        params = {
            "query": full_query,
            "format": "json",
            "page": page,
            "pageSize": cfg.max_per_provider,
            "resultType": "core",
        }
        resp = _req("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, cfg=cfg)
        payload = resp.json()
        result_list = payload.get("resultList", {}).get("result", []) if isinstance(payload, dict) else []
        out: list[ArticleCandidate] = []
        for item in result_list:
            if not isinstance(item, dict):
                continue
            doi = str(item.get("doi") or "").strip().lower()
            source_id = str(item.get("id") or "").strip()
            title = str(item.get("title") or "").strip()
            year = None
            year_raw = item.get("pubYear")
            if isinstance(year_raw, str) and year_raw.isdigit():
                year = int(year_raw)
            fulltext_list = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
            oa_url = ""
            if isinstance(fulltext_list, list):
                for node in fulltext_list:
                    if not isinstance(node, dict):
                        continue
                    url = str(node.get("url") or "").strip()
                    if url:
                        oa_url = url
                        break
            if spec.eligibility.oa_required and not oa_url:
                continue
            out.append(
                ArticleCandidate(
                    provider=self.provider_name,
                    source_id=source_id,
                    doi=doi,
                    title=title,
                    abstract=_clean_html(str(item.get("abstractText") or "")),
                    year=year,
                    venue=str(item.get("journalTitle") or "").strip(),
                    article_type=_normalize_article_type(str(item.get("pubType") or "journal-article")),
                    is_oa=bool(oa_url),
                    oa_url=oa_url,
                    source_url=str(item.get("source") or "").strip(),
                    license="",
                    language=_normalize_language(str(item.get("language") or "")),
                    raw=item,
                )
            )

        next_page = page + 1 if len(result_list) >= cfg.max_per_provider else None
        next_token = str(next_page) if next_page else None
        return out, next_token, {"page": page}


def _title_year_from_local_filename(path: Path) -> tuple[str, int | None]:
    stem = str(path.stem or "").strip()
    if not stem:
        return "", None
    m = re.match(r"^\((\d{4})\)\s*(.+)$", stem)
    if m:
        year_text = str(m.group(1) or "")
        title = str(m.group(2) or "").strip()
        year = int(year_text) if year_text.isdigit() else None
        return title, year
    return stem, None


class LocalFilesystemProvider:
    provider_name = "localfs"

    def __init__(self) -> None:
        self.last_index = pd.DataFrame()
        self.last_manifest: dict[str, Any] = {}

    def search(self, query: str, spec: TopicSpec, cfg: RunConfig) -> list[ArticleCandidate]:  # noqa: ARG002
        rows, _next_token, _meta = self.search_page(query, spec, cfg, page_token=None)
        return rows

    def search_page(
        self,
        query: str,  # noqa: ARG002
        spec: TopicSpec,  # noqa: ARG002
        cfg: RunConfig,
        page_token: str | None,
    ) -> tuple[list[ArticleCandidate], str | None, dict[str, Any]]:
        if page_token:
            return [], None, {"localfs": "done"}

        rows: list[dict[str, Any]] = []
        candidates: list[ArticleCandidate] = []
        errors: list[str] = []
        max_files = max(0, int(cfg.local_max_files or 0))
        file_glob = str(cfg.local_file_glob or "*.pdf").strip() or "*.pdf"

        for root in cfg.local_repo_paths:
            base = Path(root).expanduser().resolve()
            if not base.exists():
                errors.append(f"missing:{base}")
                continue
            if not base.is_dir():
                errors.append(f"not_directory:{base}")
                continue

            iterator = base.rglob(file_glob) if bool(cfg.local_repo_recursive) else base.glob(file_glob)
            for path in iterator:
                if not path.is_file():
                    continue
                readable = os.access(path, os.R_OK)
                if bool(cfg.local_require_readable) and not readable:
                    rows.append(
                        {
                            "repo_root": str(base),
                            "file_path": str(path),
                            "readable": False,
                            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                            "mtime_epoch": float(path.stat().st_mtime) if path.exists() else 0.0,
                            "title": "",
                            "year": None,
                            "source_id": "",
                            "selection_reason": "drop:unreadable",
                        }
                    )
                    continue

                title, year = _title_year_from_local_filename(path)
                source_id = sha1(str(path).encode("utf-8", errors="replace")).hexdigest()[:20]
                file_uri = path.as_uri()
                candidate = ArticleCandidate(
                    provider=self.provider_name,
                    source_id=source_id,
                    doi="",
                    title=title,
                    abstract="",
                    year=year,
                    venue="local_repository",
                    article_type="journal-article",
                    is_oa=True,
                    oa_url=file_uri,
                    source_url=file_uri,
                    license="localfs",
                    language="en",
                    raw={
                        "local_path": str(path),
                        "repo_root": str(base),
                        "size_bytes": int(path.stat().st_size),
                        "mtime_epoch": float(path.stat().st_mtime),
                    },
                )
                candidates.append(candidate)
                rows.append(
                    {
                        "repo_root": str(base),
                        "file_path": str(path),
                        "readable": True,
                        "size_bytes": int(path.stat().st_size),
                        "mtime_epoch": float(path.stat().st_mtime),
                        "title": title,
                        "year": year,
                        "source_id": source_id,
                        "selection_reason": "keep",
                    }
                )
                if max_files > 0 and len(candidates) >= max_files:
                    break
            if max_files > 0 and len(candidates) >= max_files:
                break

        self.last_index = pd.DataFrame(rows)
        if self.last_index.empty:
            self.last_index = pd.DataFrame(
                columns=[
                    "repo_root",
                    "file_path",
                    "readable",
                    "size_bytes",
                    "mtime_epoch",
                    "title",
                    "year",
                    "source_id",
                    "selection_reason",
                ]
            )
        self.last_manifest = {
            "created_at": now_iso(),
            "local_repo_paths": list(cfg.local_repo_paths),
            "recursive": bool(cfg.local_repo_recursive),
            "glob": file_glob,
            "scanned_rows": int(len(self.last_index)),
            "candidate_count": int(len(candidates)),
            "error_count": int(len(errors)),
            "errors": errors,
        }
        return candidates, None, dict(self.last_manifest)


def _normalize_title(title: str) -> str:
    return re.sub(r"\W+", "", title.lower())


def _candidate_quality(candidate: ArticleCandidate) -> tuple[int, int, int, int, int]:
    raw = candidate.raw if isinstance(candidate.raw, dict) else {}
    local_path = Path(str(raw.get("local_path") or "")).expanduser() if raw else Path("")
    local_readable = 1 if (candidate.provider == "localfs" and local_path.exists() and os.access(local_path, os.R_OK)) else 0
    metadata_count = sum(
        int(bool(item))
        for item in (
            candidate.title,
            candidate.abstract,
            candidate.venue,
            candidate.doi,
            candidate.oa_url,
            candidate.language,
            candidate.license,
        )
    )
    oa_score = 1 if bool(candidate.oa_url) else 0
    provider_pref = 1 if candidate.provider == "openalex" else 0
    year_score = int(candidate.year or 0)
    return (local_readable, metadata_count, oa_score, provider_pref, year_score)


def _prefer_candidate(current: ArticleCandidate, incoming: ArticleCandidate) -> ArticleCandidate:
    if _candidate_quality(incoming) > _candidate_quality(current):
        return incoming
    return current


def deduplicate_articles(rows: list[ArticleCandidate]) -> list[ArticleCandidate]:
    by_key: dict[str, ArticleCandidate] = {}
    no_doi_title_map: dict[str, list[ArticleCandidate]] = {}
    for row in rows:
        doi = row.doi.strip().lower()
        title_key = _normalize_title(row.title)
        if doi:
            key = f"doi:{doi}"
            current = by_key.get(key)
            by_key[key] = row if current is None else _prefer_candidate(current, row)
            continue
        key = f"title:{title_key}"
        no_doi_title_map.setdefault(key, []).append(row)

    out = list(by_key.values())
    for title_key, items in no_doi_title_map.items():
        clusters: list[list[ArticleCandidate]] = []
        for item in sorted(items, key=lambda x: int(x.year or 0)):
            placed = False
            for cluster in clusters:
                anchor = cluster[0]
                if anchor.year is None or item.year is None or abs(int(anchor.year) - int(item.year)) <= 1:
                    cluster.append(item)
                    placed = True
                    break
            if not placed:
                clusters.append([item])
        for idx, cluster in enumerate(clusters):
            best = cluster[0]
            for candidate in cluster[1:]:
                best = _prefer_candidate(best, candidate)
            out.append(best)
            _ = f"{title_key}:{idx}"  # deterministic cluster key reserved for future debugging
    return out


def _article_key(item: ArticleCandidate) -> str:
    raw = item.doi or item.source_id or item.title
    return sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def _article_to_dict(item: ArticleCandidate) -> dict[str, Any]:
    data = asdict(item)
    data["article_key"] = _article_key(item)
    data["discovered_at"] = now_iso()
    data["raw_json"] = json.dumps(data.pop("raw", {}), sort_keys=True)
    return data


def _expand_queries(spec: TopicSpec, query: str, plugin: TopicPlugin | None) -> list[str]:
    out: list[str] = []
    if plugin is not None:
        for item in plugin.expand_query(spec, query):
            clean = " ".join(str(item).split())
            if clean and clean not in out:
                out.append(clean)
    for item in [query] + list(spec.plugins.extra_queries):
        clean = " ".join(str(item).split())
        if clean and clean not in out:
            out.append(clean)

    # Keyword subset expansion for broader crawl coverage.
    keyword_blob = " ".join(spec.keywords).strip()
    if keyword_blob and keyword_blob not in out:
        out.append(keyword_blob)

    tokens = [tok for tok in _tokenize(keyword_blob) if tok]
    if len(tokens) >= 3:
        subset = " ".join(tokens[: min(6, len(tokens))])
        if subset and subset not in out:
            out.append(subset)

    return out[:20]


def _page_frame(rows: list[ArticleCandidate]) -> pd.DataFrame:
    frame = pd.DataFrame([_article_to_dict(item) for item in rows])
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "article_key",
                "provider",
                "source_id",
                "doi",
                "title",
                "abstract",
                "year",
                "venue",
                "article_type",
                "is_oa",
                "oa_url",
                "source_url",
                "license",
                "language",
                "discovered_at",
                "raw_json",
            ]
        )
    return frame


def _label_selection(scored: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    if scored.empty:
        out = scored.copy()
        out["selection_reason"] = []
        return out
    out = scored.copy()
    out["selection_reason"] = "keep"

    below_score_mask = out["discovery_score"] < float(cfg.min_discovery_score)
    out.loc[below_score_mask, "selection_reason"] = "drop:min_discovery_score"

    if "relevance_gate_ok" in out.columns:
        gate_mask = out["relevance_gate_ok"] != True  # noqa: E712
        out.loc[gate_mask, "selection_reason"] = "drop:relevance_gate"
    return out


def _load_resume_cursors(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def default_providers() -> list[DiscoveryProvider]:
    return [OpenAlexProvider(), CrossrefProvider(), ArxivProvider(), EuropePMCProvider()]


def index_local_repository(*, cfg: RunConfig) -> ArticleIndex:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]
    provider = LocalFilesystemProvider()
    rows, _next_token, meta = provider.search_page("", None, cfg, None)  # type: ignore[arg-type]
    frame = _page_frame(rows)
    index_path = artifacts / "local_repository_index.parquet"
    manifest_path = artifacts / "local_repository_manifest.json"
    debug_path = artifacts / "local_discovery_debug.parquet"

    provider.last_index.to_parquet(index_path, index=False)
    manifest_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame([_article_to_dict(item) for item in rows]).to_parquet(debug_path, index=False)
    return ArticleIndex(frame=frame, parquet_path=index_path)


def discover(
    *,
    spec: TopicSpec,
    query: str,
    cfg: RunConfig,
    providers: list[DiscoveryProvider] | None = None,
) -> ArticleIndex:
    online_provider_set = providers or default_providers()
    merge_mode = str(cfg.local_merge_mode or "union").strip().lower()
    local_provider: LocalFilesystemProvider | None = None
    if merge_mode in {"union", "local_only"} and cfg.local_repo_paths:
        local_provider = LocalFilesystemProvider()

    if merge_mode == "local_only":
        provider_set: list[DiscoveryProvider] = [local_provider] if local_provider is not None else []
    elif merge_mode == "online_only":
        provider_set = list(online_provider_set)
    else:
        provider_set = list(online_provider_set)
        if local_provider is not None:
            provider_set.append(local_provider)

    plugin = load_topic_plugin(cfg.plugin_id or spec.plugins.preferred_plugin)
    expanded_queries = _expand_queries(spec, query, plugin)

    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]

    ranked_path = artifacts / "articles_ranked.parquet"
    cursors_path = artifacts / "discovery_cursors.json"

    all_rows: list[ArticleCandidate] = []
    if cfg.resume_discovery and ranked_path.exists():
        existing = pd.read_parquet(ranked_path)
        for row in existing.to_dict(orient="records"):
            all_rows.append(
                ArticleCandidate(
                    provider=str(row.get("provider") or ""),
                    source_id=str(row.get("source_id") or ""),
                    doi=str(row.get("doi") or ""),
                    title=str(row.get("title") or ""),
                    abstract=str(row.get("abstract") or ""),
                    year=int(row.get("year")) if str(row.get("year") or "").isdigit() else None,
                    venue=str(row.get("venue") or ""),
                    article_type=str(row.get("article_type") or ""),
                    is_oa=bool(row.get("is_oa")),
                    oa_url=str(row.get("oa_url") or ""),
                    source_url=str(row.get("source_url") or ""),
                    license=str(row.get("license") or ""),
                    language=str(row.get("language") or ""),
                    raw={},
                )
            )

    resume_cursors = _load_resume_cursors(cursors_path) if cfg.resume_discovery else {}

    scorer: CandidateScorer = WeightedCandidateScorer(plugin=plugin)
    saturation = RollingSaturationController(
        window_pages=cfg.saturation_window_pages,
        min_yield=cfg.saturation_min_yield,
        min_pages_before=cfg.min_pages_before_saturation,
    )

    page_rows: list[dict[str, Any]] = []
    page_metrics_by_key: dict[str, list[dict[str, Any]]] = {}
    cursor_state: dict[str, dict[str, Any]] = {}
    emergency_stop = False

    for search_query in expanded_queries:
        if emergency_stop:
            break
        for provider in provider_set:
            if emergency_stop:
                break
            provider_name = str(getattr(provider, "provider_name", provider.__class__.__name__)).strip().lower()
            if provider_name == "localfs" and search_query != expanded_queries[0]:
                continue
            key = f"{provider_name}::{search_query}"
            token = str(resume_cursors.get(key, {}).get("next_token") or "").strip() or None

            page_metrics_by_key.setdefault(key, [])
            stop_reason = "provider_exhausted"
            last_next_token = token

            for page_idx in range(1, max(1, int(cfg.max_pages_per_provider)) + 1):
                try:
                    page_candidates, next_token, page_meta = provider.search_page(search_query, spec, cfg, token)  # type: ignore[attr-defined]
                except Exception as exc:
                    page_rows.append(
                        {
                            "provider": provider_name,
                            "query": search_query,
                            "page_index": page_idx,
                            "cursor_in": str(token or ""),
                            "cursor_out": "",
                            "candidate_count": 0,
                            "keep_count": 0,
                            "page_yield": 0.0,
                            "stop_reason": f"provider_error:{type(exc).__name__}",
                            "created_at": now_iso(),
                        }
                    )
                    stop_reason = "provider_error"
                    break

                page_count = len(page_candidates)
                keep_count = 0
                page_yield = 0.0
                if page_count > 0:
                    page_frame = _page_frame(page_candidates)
                    page_scored = scorer.score_candidates(page_frame, spec, query)
                    page_labeled = _label_selection(page_scored, cfg)
                    keep_count = int((page_labeled["selection_reason"] == "keep").sum())
                    page_yield = float(keep_count / max(page_count, 1))
                    all_rows.extend(page_candidates)

                page_metrics_by_key[key].append(
                    {
                        "provider": provider_name,
                        "query": search_query,
                        "page_index": page_idx,
                        "page_yield": page_yield,
                    }
                )

                page_rows.append(
                    {
                        "provider": provider_name,
                        "query": search_query,
                        "page_index": page_idx,
                        "cursor_in": str(token or ""),
                        "cursor_out": str(next_token or ""),
                        "candidate_count": page_count,
                        "keep_count": keep_count,
                        "page_yield": page_yield,
                        "stop_reason": "",
                        "created_at": now_iso(),
                        "meta": json.dumps(page_meta or {}, sort_keys=True),
                    }
                )

                cursor_state[key] = {
                    "provider": provider_name,
                    "query": search_query,
                    "next_token": str(next_token or ""),
                    "last_page": page_idx,
                    "updated_at": now_iso(),
                }

                if len(all_rows) >= int(cfg.max_discovered_records):
                    stop_reason = "emergency_cap"
                    emergency_stop = True
                    break

                if page_count == 0:
                    stop_reason = "provider_exhausted"
                    break

                if saturation.should_stop(key, page_metrics_by_key[key]):
                    stop_reason = "saturation_stop"
                    break

                if not next_token:
                    stop_reason = "provider_exhausted"
                    break

                token = next_token
                last_next_token = next_token

            # Stamp stop reason on latest row for this provider/query pass.
            if page_rows:
                for idx in range(len(page_rows) - 1, -1, -1):
                    row = page_rows[idx]
                    if row.get("provider") == provider_name and row.get("query") == search_query:
                        row["stop_reason"] = stop_reason
                        break

            if key in cursor_state:
                cursor_state[key]["next_token"] = str(last_next_token or "")

    filtered: list[ArticleCandidate] = []
    allowed_types = {item.lower() for item in spec.eligibility.article_types}
    allowed_languages = {item.lower() for item in spec.eligibility.languages}
    for row in deduplicate_articles(all_rows):
        if spec.eligibility.oa_required and not row.is_oa:
            continue
        if allowed_types and row.article_type and row.article_type.lower() not in allowed_types:
            continue
        # English-first mode keeps non-English records in discovery; acquisition can prioritize/skip later.
        if not cfg.english_first and allowed_languages and row.language and row.language.lower() not in allowed_languages:
            continue
        filtered.append(row)

    base_frame = _page_frame(filtered)
    scored = scorer.score_candidates(base_frame, spec, query)
    debug = _label_selection(scored, cfg)

    selected = debug[debug["selection_reason"] == "keep"].copy()

    if cfg.per_provider_cap:
        selected = selected.reset_index(drop=True)
        keep_rows: list[dict[str, Any]] = []
        provider_counts: dict[str, int] = {}
        dropped_keys: set[str] = set()
        for _, row in selected.iterrows():
            provider = str(row.get("provider") or "").strip().lower()
            cap = int(cfg.per_provider_cap.get(provider, 0))
            if cap > 0 and provider_counts.get(provider, 0) >= cap:
                dropped_keys.add(str(row.get("article_key") or ""))
                continue
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            keep_rows.append(row.to_dict())
        selected = pd.DataFrame(keep_rows) if keep_rows else pd.DataFrame(columns=debug.columns)
        if dropped_keys:
            debug.loc[debug["article_key"].isin(dropped_keys), "selection_reason"] = "drop:provider_cap"

    if not selected.empty and cfg.max_discovered_records > 0:
        selected = selected.head(int(cfg.max_discovered_records)).reset_index(drop=True)
        selected_keys = set(selected["article_key"].astype(str).tolist())
        over_limit_mask = (debug["selection_reason"] == "keep") & (~debug["article_key"].isin(selected_keys))
        debug.loc[over_limit_mask, "selection_reason"] = "drop:max_discovered_records"
    else:
        selected = selected.reset_index(drop=True)

    articles_path = artifacts / "articles.parquet"
    debug_path = artifacts / "discovery_debug.parquet"
    pages_path = artifacts / "discovery_pages.parquet"
    local_index_path = artifacts / "local_repository_index.parquet"
    local_manifest_path = artifacts / "local_repository_manifest.json"
    local_debug_path = artifacts / "local_discovery_debug.parquet"

    selected.to_parquet(articles_path, index=False)
    scored.to_parquet(ranked_path, index=False)
    debug.to_parquet(debug_path, index=False)

    pages_df = pd.DataFrame(page_rows)
    if pages_df.empty:
        pages_df = pd.DataFrame(
            columns=[
                "provider",
                "query",
                "page_index",
                "cursor_in",
                "cursor_out",
                "candidate_count",
                "keep_count",
                "page_yield",
                "stop_reason",
                "meta",
                "created_at",
            ]
        )
    pages_df.to_parquet(pages_path, index=False)
    if local_provider is not None:
        local_index = local_provider.last_index.copy()
        if local_index.empty:
            local_index = pd.DataFrame(
                columns=[
                    "repo_root",
                    "file_path",
                    "readable",
                    "size_bytes",
                    "mtime_epoch",
                    "title",
                    "year",
                    "source_id",
                    "selection_reason",
                ]
            )
        local_index.to_parquet(local_index_path, index=False)
        local_manifest_payload = dict(local_provider.last_manifest or {})
        local_manifest_payload.setdefault("created_at", now_iso())
        local_manifest_path.write_text(
            json.dumps(local_manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        local_debug = debug[debug["provider"].astype(str).str.lower() == "localfs"].copy()
        if local_debug.empty:
            local_debug = pd.DataFrame(columns=debug.columns)
        local_debug.to_parquet(local_debug_path, index=False)

    cursors_path.write_text(json.dumps(cursor_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cache_audit_path = artifacts / "cache_audit.parquet"
    cache = _DISCOVERY_CACHE.get(str(cfg.as_path()))
    if cache is not None:
        cache.write_audit(cache_audit_path)

    state = load_run_state(cfg.as_path())
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="articles", path=articles_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="articles_ranked", path=ranked_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="discovery_debug", path=debug_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="discovery_pages", path=pages_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="discovery_cursors", path=cursors_path)
    if cache is not None and cache_audit_path.exists():
        upsert_artifact(db_path=db_path, run_id=state["run_id"], name="cache_audit", path=cache_audit_path)
    if local_provider is not None:
        upsert_artifact(db_path=db_path, run_id=state["run_id"], name="local_repository_index", path=local_index_path)
        upsert_artifact(
            db_path=db_path,
            run_id=state["run_id"],
            name="local_repository_manifest",
            path=local_manifest_path,
        )
        upsert_artifact(db_path=db_path, run_id=state["run_id"], name="local_discovery_debug", path=local_debug_path)
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="discover",
        source_name="providers",
        target_name="articles_ranked.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="discover",
        source_name="articles_ranked.parquet",
        target_name="articles.parquet",
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="discover",
        source_name="providers",
        target_name="discovery_pages.parquet",
    )
    if local_provider is not None:
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="discover",
            source_name="local_repository",
            target_name="local_repository_index.parquet",
        )
        append_lineage(
            db_path=db_path,
            run_id=state["run_id"],
            stage="discover",
            source_name="local_repository_index.parquet",
            target_name="local_discovery_debug.parquet",
        )

    return ArticleIndex(frame=selected, parquet_path=articles_path)
