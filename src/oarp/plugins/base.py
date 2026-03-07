from __future__ import annotations

from typing import Protocol

import pandas as pd

from oarp.models import ArticleCandidate, FullTextRecord, RunConfig
from oarp.topic_spec import TopicSpec


class DiscoveryProvider(Protocol):
    provider_name: str

    def search(self, query: str, spec: TopicSpec, cfg: RunConfig) -> list[ArticleCandidate]:
        ...


class FullTextFetcher(Protocol):
    fetcher_name: str

    def fetch(self, article: ArticleCandidate, cfg: RunConfig) -> FullTextRecord:
        ...


class ExtractionEngine(Protocol):
    engine_name: str

    def extract(
        self,
        doc: FullTextRecord,
        spec: TopicSpec,
        cfg: RunConfig,
    ) -> tuple[list[dict], list[dict]]:
        """Return evidence points and provenance rows."""


class ConsensusModel(Protocol):
    model_name: str

    def score(self, points: list[dict], spec: TopicSpec, cfg: RunConfig) -> list[dict]:
        ...


class TopicPlugin(Protocol):
    plugin_id: str

    def expand_query(self, spec: TopicSpec, query: str) -> list[str]:
        ...

    def candidate_boost(self, article_row: dict, spec: TopicSpec) -> float:
        ...

    def entity_alias_overrides(self, spec: TopicSpec) -> dict[str, str]:
        ...


class CandidateScorer(Protocol):
    def score_candidates(self, frame: pd.DataFrame, spec: TopicSpec, query: str) -> pd.DataFrame:
        ...


class ContextAssembler(Protocol):
    def assemble(self, points_df: pd.DataFrame, docs_df: pd.DataFrame, spec: TopicSpec) -> pd.DataFrame:
        ...


class DiscoveryPaginator(Protocol):
    def iter_pages(self, provider: DiscoveryProvider, query: str, spec: TopicSpec, cfg: RunConfig):
        ...


class SaturationController(Protocol):
    def should_stop(self, key: str, page_metrics: list[dict]) -> bool:
        ...


class MaterialContextExtractor(Protocol):
    def extract_context(self, snippet: str, spec: TopicSpec, cfg: RunConfig) -> dict:
        ...


class ContextValidator(Protocol):
    def validate_context(self, point_row: dict, spec: TopicSpec, cfg: RunConfig) -> list[str]:
        ...
