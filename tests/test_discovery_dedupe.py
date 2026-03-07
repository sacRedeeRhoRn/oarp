from __future__ import annotations

from oarp.discovery import deduplicate_articles
from oarp.models import ArticleCandidate


def _article(*, provider: str, doi: str, title: str, oa_url: str) -> ArticleCandidate:
    return ArticleCandidate(
        provider=provider,
        source_id=doi or title,
        doi=doi,
        title=title,
        abstract="",
        year=2020,
        venue="",
        article_type="journal-article",
        is_oa=bool(oa_url),
        oa_url=oa_url,
        source_url="",
        license="",
        language="en",
        raw={},
    )


def test_deduplicate_prefers_rows_with_oa_url() -> None:
    rows = [
        _article(provider="crossref", doi="10.1/demo", title="T", oa_url=""),
        _article(provider="openalex", doi="10.1/demo", title="T", oa_url="https://oa.example/a"),
    ]
    deduped = deduplicate_articles(rows)
    assert len(deduped) == 1
    assert deduped[0].provider == "openalex"


def test_deduplicate_by_title_when_no_doi() -> None:
    rows = [
        _article(provider="crossref", doi="", title="Same Title!", oa_url="https://a"),
        _article(provider="europepmc", doi="", title="same title", oa_url="https://b"),
    ]
    deduped = deduplicate_articles(rows)
    assert len(deduped) == 1
