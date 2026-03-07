from __future__ import annotations

import hashlib
import json
import mimetypes
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from oarp.models import DocumentIndex, RunConfig
from oarp.runtime import (
    append_lineage,
    ensure_run_layout,
    init_index_db,
    load_run_state,
    now_iso,
    upsert_artifact,
)


class DomainThrottle:
    def __init__(self, requests_per_second: float) -> None:
        self.requests_per_second = max(requests_per_second, 0.1)
        self._next_allowed: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, domain: str) -> None:
        with self._lock:
            now = time.monotonic()
            target = self._next_allowed.get(domain, now)
            sleep_sec = max(0.0, target - now)
            self._next_allowed[domain] = max(target, now) + (1.0 / self.requests_per_second)
        if sleep_sec > 0:
            time.sleep(sleep_sec)


class RobotsCache:
    def __init__(self, user_agent: str, timeout_sec: int = 8) -> None:
        self.user_agent = user_agent
        self.timeout_sec = max(2, int(timeout_sec))
        self._cache: dict[str, RobotFileParser] = {}
        self._lock = threading.Lock()

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return False
        domain = parsed.netloc

        with self._lock:
            parser = self._cache.get(domain)
        if parser is None:
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            try:
                response = requests.get(
                    robots_url,
                    timeout=self.timeout_sec,
                    headers={"User-Agent": self.user_agent},
                )
                response.raise_for_status()
                parser.parse(response.text.splitlines())
            except Exception:
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Allow: /"])
            with self._lock:
                self._cache[domain] = parser
        return parser.can_fetch(self.user_agent, url)


class AcquisitionPlanner:
    """Plan candidate fulltext URLs from article metadata with OA-aware prioritization."""

    def plan(self, article_row: dict[str, Any], cfg: RunConfig) -> list[str]:
        provider = str(article_row.get("provider") or "").strip().lower()
        urls: list[str] = []

        def push(url: str) -> None:
            clean = str(url or "").strip()
            if not clean:
                return
            if not (clean.startswith("http") or clean.startswith("file://") or Path(clean).expanduser().is_absolute()):
                return
            if clean.startswith("http") and _is_metadata_api_url(clean):
                return
            if clean not in urls:
                urls.append(clean)

        oa_url = str(article_row.get("oa_url") or "").strip()
        source_url = str(article_row.get("source_url") or "").strip()
        push(oa_url)

        raw_json = str(article_row.get("raw_json") or "").strip()
        if raw_json:
            try:
                raw = json.loads(raw_json)
            except Exception:
                raw = {}
            for candidate in _provider_urls_from_raw(provider, raw):
                push(candidate)

        if cfg.allow_direct_crawl and source_url and source_url != oa_url:
            push(source_url)

        return urls


class DocumentQualityGate:
    """Classify parse outputs for extraction readiness."""

    def __init__(self, cfg: RunConfig) -> None:
        self._cfg = cfg

    def assess(self, doc_row: dict[str, Any]) -> tuple[bool, str]:
        parse_status = str(doc_row.get("parse_status") or "").strip().lower()
        mime = str(doc_row.get("mime") or "").strip().lower()
        text = str(doc_row.get("text_content") or "")
        text_len = len(text.strip())

        if parse_status in {"fetch_failed", "skipped_non_english"}:
            return False, parse_status
        if "json" in mime:
            return False, "json_not_fulltext"
        if parse_status not in {"parsed_html", "parsed_xml", "parsed_text", "parsed_pdf"}:
            return False, "unsupported_parse_status"
        if text_len < max(1, int(self._cfg.min_text_length)):
            return False, f"short_text<{self._cfg.min_text_length}"
        return True, "usable"


def _is_metadata_api_url(url: str) -> bool:
    lower = url.lower()
    if "api.openalex.org/works" in lower:
        return True
    return False


def _provider_urls_from_raw(provider: str, raw: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    if provider == "openalex":
        best = raw.get("best_oa_location") if isinstance(raw.get("best_oa_location"), dict) else {}
        if isinstance(best, dict):
            for key in ("pdf_url", "landing_page_url"):
                val = str(best.get(key) or "").strip()
                if val:
                    urls.append(val)
        open_access = raw.get("open_access") if isinstance(raw.get("open_access"), dict) else {}
        if isinstance(open_access, dict):
            val = str(open_access.get("oa_url") or "").strip()
            if val:
                urls.append(val)
        locations = raw.get("locations") if isinstance(raw.get("locations"), list) else []
        for node in locations:
            if not isinstance(node, dict):
                continue
            for key in ("pdf_url", "landing_page_url"):
                val = str(node.get(key) or "").strip()
                if val:
                    urls.append(val)
    elif provider == "crossref":
        links = raw.get("link") if isinstance(raw.get("link"), list) else []
        for link in links:
            if not isinstance(link, dict):
                continue
            val = str(link.get("URL") or "").strip()
            if val:
                urls.append(val)
    elif provider == "europepmc":
        items = raw.get("fullTextUrlList", {}).get("fullTextUrl", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                val = str(item.get("url") or "").strip()
                if val:
                    urls.append(val)
    return urls


def _guess_extension(content_type: str, url: str) -> str:
    ctype = (content_type or "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return ".pdf"
    if "xml" in ctype:
        return ".xml"
    if "html" in ctype:
        return ".html"
    if "json" in ctype:
        return ".json"
    if "plain" in ctype or ctype.startswith("text/"):
        return ".txt"
    return ".bin"


def _resolve_local_source(url: str) -> Path | None:
    raw = str(url or "").strip()
    if not raw:
        return None
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        if parsed.scheme != "file":
            return None
        path = Path(unquote(parsed.path)).expanduser()
        return path.resolve()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return None


def _read_local_source(url: str) -> tuple[bytes, str, Path] | None:
    path = _resolve_local_source(url)
    if path is None or not path.exists() or not path.is_file():
        return None
    data = path.read_bytes()
    guessed, _enc = mimetypes.guess_type(str(path))
    content_type = str(guessed or "")
    if not content_type:
        if path.suffix.lower() == ".pdf":
            content_type = "application/pdf"
        elif path.suffix.lower() in {".txt", ".text"}:
            content_type = "text/plain"
        elif path.suffix.lower() in {".xml"}:
            content_type = "application/xml"
        elif path.suffix.lower() in {".html", ".htm"}:
            content_type = "text/html"
    return data, content_type, path


def _extract_html(content: bytes) -> str:
    decoded = content.decode("utf-8", errors="replace")
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(decoded, include_tables=True, include_comments=False)
        if extracted:
            return extracted.strip()
    except Exception:
        pass

    soup = BeautifulSoup(decoded, "lxml")
    return soup.get_text("\n", strip=True)


def _extract_pdf(local_path: Path, min_text_length: int) -> str:
    text = ""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(local_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
    except Exception:
        text = ""

    if len(text) >= min_text_length:
        return text

    try:
        import pdfplumber  # type: ignore

        rows: list[str] = []
        with pdfplumber.open(str(local_path)) as pdf:
            for page in pdf.pages:
                rows.append(page.extract_text() or "")
        fallback = "\n".join(rows).strip()
        if len(fallback) > len(text):
            text = fallback
    except Exception:
        pass

    return text


def _extract_text(
    content: bytes,
    content_type: str,
    local_path: Path,
    min_text_length: int,
) -> tuple[str, str]:
    ctype = (content_type or "").lower()
    if "html" in ctype:
        text = _extract_html(content)
        return text, "parsed_html"
    if "xml" in ctype or local_path.suffix.lower() == ".xml":
        soup = BeautifulSoup(content.decode("utf-8", errors="replace"), "xml")
        return soup.get_text("\n", strip=True), "parsed_xml"
    if "text" in ctype:
        return content.decode("utf-8", errors="replace"), "parsed_text"
    if "pdf" in ctype or local_path.suffix.lower() == ".pdf":
        text = _extract_pdf(local_path, min_text_length=min_text_length)
        if text:
            return text, "parsed_pdf"
        return "", "stored_binary"
    return "", "stored_binary"


def _is_fulltext_mime(content_type: str, url: str) -> bool:
    ctype = str(content_type or "").lower()
    lower_url = str(url or "").lower()
    if "json" in ctype:
        return False
    if "pdf" in ctype or "xml" in ctype or "html" in ctype or ctype.startswith("text/"):
        return True
    if lower_url.endswith((".pdf", ".xml", ".html", ".txt")):
        return True
    return False


def _download(
    *,
    url: str,
    cfg: RunConfig,
    robots: RobotsCache,
    throttle: DomainThrottle,
) -> requests.Response | None:
    parsed = urlparse(url)
    domain = parsed.netloc
    if not domain:
        return None
    if not robots.can_fetch(url):
        return None
    throttle.wait(domain)

    for attempt in range(cfg.retries + 1):
        try:
            response = requests.get(
                url,
                timeout=cfg.timeout_sec,
                headers={"User-Agent": cfg.user_agent},
                allow_redirects=True,
            )
            if response.status_code >= 400:
                continue
            return response
        except Exception:
            if attempt >= cfg.retries:
                return None
            time.sleep(cfg.backoff_sec * (attempt + 1))
    return None


def _load_existing_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = str(row.get("article_key") or "")
        if key:
            out[key] = row
    return out


def _acquire_one(
    *,
    article_row: dict[str, Any],
    cfg: RunConfig,
    raw_root: Path,
    planner: AcquisitionPlanner,
    robots: RobotsCache,
    throttle: DomainThrottle,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    article_key = str(article_row.get("article_key") or "")
    provider = str(article_row.get("provider") or "")
    language = str(article_row.get("language") or "")

    debug_rows: list[dict[str, Any]] = []

    if cfg.english_first and language and language.lower() not in {"en", "eng"}:
        row = {
            "article_key": article_key,
            "provider": provider,
            "source_url": "",
            "local_path": "",
            "mime": "",
            "content_hash": "",
            "parse_status": "skipped_non_english",
            "text_content": "",
            "language": language,
        }
        progress = {
            "article_key": article_key,
            "provider": provider,
            "language": language,
            "status": "skipped",
            "reason": "non_english_backlog",
            "source_url": "",
            "local_path": "",
            "updated_at": now_iso(),
        }
        return row, debug_rows, progress

    candidates = planner.plan(article_row, cfg)
    selected_url = ""
    selected_rank = 0
    selected_content: bytes | None = None
    selected_content_type = ""
    selected_is_local = False
    selected_local_source = ""

    for rank, candidate in enumerate(candidates, start=1):
        local_payload = _read_local_source(candidate)
        if local_payload is not None:
            local_data, local_content_type, local_path = local_payload
            if cfg.require_fulltext_mime and not _is_fulltext_mime(local_content_type, candidate):
                debug_rows.append(
                    {
                        "article_key": article_key,
                        "provider": provider,
                        "candidate_url": candidate,
                        "candidate_rank": rank,
                        "status": "drop",
                        "reason": "non_fulltext_mime",
                        "status_code": 0,
                        "content_type": local_content_type,
                    }
                )
                continue
            selected_url = candidate
            selected_rank = rank
            selected_content = local_data
            selected_content_type = local_content_type
            selected_is_local = True
            selected_local_source = str(local_path)
            debug_rows.append(
                {
                    "article_key": article_key,
                    "provider": provider,
                    "candidate_url": candidate,
                    "candidate_rank": rank,
                    "status": "selected",
                    "reason": "local_selected",
                    "status_code": 200,
                    "content_type": local_content_type,
                }
            )
            break

        attempt = _download(url=candidate, cfg=cfg, robots=robots, throttle=throttle)
        if attempt is None:
            debug_rows.append(
                {
                    "article_key": article_key,
                    "provider": provider,
                    "candidate_url": candidate,
                    "candidate_rank": rank,
                    "status": "drop",
                    "reason": "fetch_failed",
                    "status_code": 0,
                    "content_type": "",
                }
            )
            continue

        content_type = str(attempt.headers.get("Content-Type") or "")
        if cfg.require_fulltext_mime and not _is_fulltext_mime(content_type, candidate):
            debug_rows.append(
                {
                    "article_key": article_key,
                    "provider": provider,
                    "candidate_url": candidate,
                    "candidate_rank": rank,
                    "status": "drop",
                    "reason": "non_fulltext_mime",
                    "status_code": int(attempt.status_code),
                    "content_type": content_type,
                }
            )
            continue

        selected_url = candidate
        selected_rank = rank
        selected_content = attempt.content
        selected_content_type = content_type
        selected_is_local = False
        selected_local_source = ""
        debug_rows.append(
            {
                "article_key": article_key,
                "provider": provider,
                "candidate_url": candidate,
                "candidate_rank": rank,
                "status": "selected",
                "reason": "ok",
                "status_code": int(attempt.status_code),
                "content_type": content_type,
            }
        )
        break

    if selected_content is None:
        row = {
            "article_key": article_key,
            "provider": provider,
            "source_url": "",
            "local_path": "",
            "mime": "",
            "content_hash": "",
            "parse_status": "fetch_failed",
            "text_content": "",
            "language": language,
        }
        progress = {
            "article_key": article_key,
            "provider": provider,
            "language": language,
            "status": "failed",
            "reason": "fetch_failed",
            "source_url": "",
            "local_path": "",
            "updated_at": now_iso(),
        }
        return row, debug_rows, progress

    ext = _guess_extension(selected_content_type, selected_url)
    if ext == ".bin" and selected_is_local:
        suffix = Path(selected_local_source).suffix
        if suffix:
            ext = suffix
    target = raw_root / provider / f"{article_key}{ext}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(selected_content)
    digest = hashlib.sha256(selected_content).hexdigest()

    text_content, parse_status = _extract_text(
        selected_content,
        selected_content_type,
        target,
        min_text_length=max(1, int(cfg.min_text_length)),
    )

    debug_rows.append(
        {
            "article_key": article_key,
            "provider": provider,
            "candidate_url": selected_url,
            "candidate_rank": selected_rank,
            "status": "parsed",
            "reason": (
                "local_copy_ok"
                if selected_is_local and parse_status in {"parsed_html", "parsed_xml", "parsed_text", "parsed_pdf"}
                else "local_parse_failed"
                if selected_is_local
                else "ok"
            ),
            "status_code": 200,
            "content_type": selected_content_type,
        }
    )

    row = {
        "article_key": article_key,
        "provider": provider,
        "source_url": selected_url,
        "local_path": str(target),
        "mime": selected_content_type,
        "content_hash": digest,
        "parse_status": parse_status,
        "text_content": text_content,
        "language": language,
    }
    progress = {
        "article_key": article_key,
        "provider": provider,
        "language": language,
        "status": "success",
        "reason": "ok",
        "source_url": selected_url,
        "local_path": str(target),
        "updated_at": now_iso(),
    }
    return row, debug_rows, progress


def acquire(
    *,
    cfg: RunConfig,
) -> DocumentIndex:
    layout = ensure_run_layout(cfg.as_path())
    artifacts = layout["artifacts"]
    raw_root = layout["raw"]

    articles_path = artifacts / "articles.parquet"
    if not articles_path.exists():
        raise FileNotFoundError(f"missing articles parquet: {articles_path}")
    articles = pd.read_parquet(articles_path)

    planner = AcquisitionPlanner()
    quality_gate = DocumentQualityGate(cfg)
    robots = RobotsCache(cfg.user_agent, timeout_sec=cfg.timeout_sec)
    throttle = DomainThrottle(cfg.requests_per_second)

    progress_path = artifacts / "crawl_progress.parquet"
    existing_progress = _load_existing_progress(progress_path) if cfg.resume_discovery else {}

    existing_docs_by_key: dict[str, dict[str, Any]] = {}
    docs_path_existing = artifacts / "documents.parquet"
    if cfg.resume_discovery and docs_path_existing.exists():
        try:
            existing_docs = pd.read_parquet(docs_path_existing)
            for row in existing_docs.to_dict(orient="records"):
                key = str(row.get("article_key") or "")
                if key:
                    existing_docs_by_key[key] = row
        except Exception:
            existing_docs_by_key = {}

    pending_rows: list[tuple[int, dict[str, Any]]] = []
    documents_rows: list[dict[str, Any]] = []
    progress_rows: dict[str, dict[str, Any]] = dict(existing_progress)

    for idx, article in articles.iterrows():
        article_row = article.to_dict()
        key = str(article_row.get("article_key") or "")
        prior = existing_progress.get(key, {})
        status = str(prior.get("status") or "").strip().lower()
        if cfg.resume_discovery and status in {"success", "skipped"}:
            if key in existing_docs_by_key:
                documents_rows.append(existing_docs_by_key[key])
            continue
        pending_rows.append((idx, article_row))

    if cfg.english_first:
        pending_rows = sorted(
            pending_rows,
            key=lambda item: 0 if str(item[1].get("language") or "").lower() in {"en", "eng", ""} else 1,
        )

    pending_rows = pending_rows[: max(1, int(cfg.max_downloads))]

    debug_rows: list[dict[str, Any]] = []
    if pending_rows:
        with ThreadPoolExecutor(max_workers=max(1, int(cfg.acquire_workers))) as executor:
            futures = {
                executor.submit(
                    _acquire_one,
                    article_row=row,
                    cfg=cfg,
                    raw_root=raw_root,
                    planner=planner,
                    robots=robots,
                    throttle=throttle,
                ): idx
                for idx, row in pending_rows
            }
            for future in as_completed(futures):
                _idx = futures[future]
                try:
                    doc_row, dbg, progress = future.result()
                except Exception as exc:
                    doc_row = {
                        "article_key": "",
                        "provider": "",
                        "source_url": "",
                        "local_path": "",
                        "mime": "",
                        "content_hash": "",
                        "parse_status": "fetch_failed",
                        "text_content": "",
                        "language": "",
                    }
                    dbg = []
                    progress = {
                        "article_key": "",
                        "provider": "",
                        "language": "",
                        "status": "failed",
                        "reason": f"worker_error:{type(exc).__name__}",
                        "source_url": "",
                        "local_path": "",
                        "updated_at": now_iso(),
                    }
                if str(doc_row.get("article_key") or ""):
                    documents_rows.append(doc_row)
                debug_rows.extend(dbg)
                key = str(progress.get("article_key") or "")
                if key:
                    progress_rows[key] = progress

    documents = pd.DataFrame(documents_rows)
    if documents.empty:
        documents = pd.DataFrame(
            columns=[
                "article_key",
                "provider",
                "source_url",
                "local_path",
                "mime",
                "content_hash",
                "parse_status",
                "text_content",
                "language",
            ]
        )

    quality_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    for row in documents.to_dict(orient="records"):
        usable, reason = quality_gate.assess(row)
        text_len = len(str(row.get("text_content") or "").strip())
        row["usable_text"] = usable
        row["quality_reason"] = reason
        row["text_length"] = text_len
        enriched_rows.append(row)
        quality_rows.append(
            {
                "article_key": str(row.get("article_key") or ""),
                "provider": str(row.get("provider") or ""),
                "language": str(row.get("language") or ""),
                "parse_status": str(row.get("parse_status") or ""),
                "mime": str(row.get("mime") or ""),
                "text_length": text_len,
                "usable_text": usable,
                "quality_reason": reason,
            }
        )

    documents = pd.DataFrame(enriched_rows)

    path = artifacts / "documents.parquet"
    documents.to_parquet(path, index=False)

    acquisition_debug = pd.DataFrame(debug_rows)
    if acquisition_debug.empty:
        acquisition_debug = pd.DataFrame(
            columns=[
                "article_key",
                "provider",
                "candidate_url",
                "candidate_rank",
                "status",
                "reason",
                "status_code",
                "content_type",
            ]
        )
    acquisition_debug_path = artifacts / "acquisition_debug.parquet"
    acquisition_debug.to_parquet(acquisition_debug_path, index=False)

    document_quality = pd.DataFrame(quality_rows)
    if document_quality.empty:
        document_quality = pd.DataFrame(
            columns=[
                "article_key",
                "provider",
                "language",
                "parse_status",
                "mime",
                "text_length",
                "usable_text",
                "quality_reason",
            ]
        )
    document_quality_path = artifacts / "document_quality.parquet"
    document_quality.to_parquet(document_quality_path, index=False)

    crawl_progress = pd.DataFrame(list(progress_rows.values()))
    if crawl_progress.empty:
        crawl_progress = pd.DataFrame(
            columns=[
                "article_key",
                "provider",
                "language",
                "status",
                "reason",
                "source_url",
                "local_path",
                "updated_at",
            ]
        )
    crawl_progress.to_parquet(progress_path, index=False)

    state = load_run_state(cfg.as_path())
    db_path = artifacts / "index.sqlite"
    init_index_db(db_path)
    upsert_artifact(db_path=db_path, run_id=state["run_id"], name="documents", path=path)
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="acquisition_debug",
        path=acquisition_debug_path,
    )
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="document_quality",
        path=document_quality_path,
    )
    upsert_artifact(
        db_path=db_path,
        run_id=state["run_id"],
        name="crawl_progress",
        path=progress_path,
    )
    append_lineage(
        db_path=db_path,
        run_id=state["run_id"],
        stage="acquire",
        source_name="articles.parquet",
        target_name="documents.parquet",
    )

    return DocumentIndex(frame=documents, parquet_path=path)
