from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class EligibilitySpec(BaseModel):
    article_types: list[str] = Field(default_factory=lambda: ["journal-article", "preprint"])
    oa_required: bool = True
    languages: list[str] = Field(default_factory=lambda: ["en"])


class VariableSpec(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    unit: str
    datatype: str = "float"
    normalization: str = "identity"
    min_value: float | None = None
    max_value: float | None = None

    @field_validator("name", "unit", "datatype", "normalization")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def _range_valid(self) -> "VariableSpec":
        if self.min_value is not None and self.max_value is not None and self.max_value < self.min_value:
            raise ValueError("max_value must be >= min_value")
        return self


class EntitySpec(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("entity name must be non-empty")
        return value


class ExtractionRulesSpec(BaseModel):
    confidence_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"text": 0.6, "table": 0.75, "figure": 0.7}
    )
    variable_patterns: dict[str, str] = Field(default_factory=dict)
    figure_cues: list[str] = Field(default_factory=lambda: ["figure", "fig.", "fig "])


class PlotPrimarySpec(BaseModel):
    x: str
    y: str
    color_by: str = "entity"
    transparency_policy: str = "outlier-transparent"


class PlotSpec(BaseModel):
    primary: PlotPrimarySpec


class ConsensusSpec(BaseModel):
    model: str = "confidence-consensus"
    entropy_threshold: float = 0.7
    outlier_policy: str = "show-transparent"


class ValidationSpec(BaseModel):
    min_confidence: float = 0.65
    required_provenance_fields: list[str] = Field(
        default_factory=lambda: ["citation_url", "snippet", "locator"]
    )


class PluginsSpec(BaseModel):
    preferred_plugin: str = ""
    extra_queries: list[str] = Field(default_factory=list)


class TopicSpec(BaseModel):
    topic_id: str
    question_template: str | None = None
    keywords: list[str] = Field(default_factory=list)
    eligibility: EligibilitySpec = Field(default_factory=EligibilitySpec)
    variables: list[VariableSpec]
    entities: list[EntitySpec] = Field(default_factory=list)
    extraction_rules: ExtractionRulesSpec = Field(default_factory=ExtractionRulesSpec)
    plot: PlotSpec
    consensus: ConsensusSpec = Field(default_factory=ConsensusSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    plugins: PluginsSpec = Field(default_factory=PluginsSpec)

    @field_validator("topic_id")
    @classmethod
    def _topic_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("topic_id must be non-empty")
        return value

    @field_validator("variables")
    @classmethod
    def _has_variables(cls, value: list[VariableSpec]) -> list[VariableSpec]:
        if not value:
            raise ValueError("variables must include at least one variable")
        return value

    def variable_alias_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.variables:
            out[item.name.lower()] = item.name
            for alias in item.aliases:
                alias_key = alias.strip().lower()
                if alias_key:
                    out[alias_key] = item.name
        return out

    def entity_alias_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.entities:
            out[item.name.lower()] = item.name
            for alias in item.aliases:
                alias_key = alias.strip().lower()
                if alias_key:
                    out[alias_key] = item.name
        return out

    def variable_by_name(self, name: str) -> VariableSpec:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(f"variable not found: {name}")


def load_topic_spec(path: str | Path) -> TopicSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"topic spec must be a YAML mapping: {spec_path}")
    try:
        return TopicSpec.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"topic spec validation failed: {exc}") from exc


def dump_topic_spec(spec: TopicSpec, path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.write_text(yaml.safe_dump(spec.model_dump(mode="python"), sort_keys=False), encoding="utf-8")


def ensure_query(spec: TopicSpec, query: str | None) -> str:
    if query and query.strip():
        return query.strip()
    if spec.question_template and spec.question_template.strip():
        return spec.question_template.strip()
    if spec.keywords:
        return " ".join(spec.keywords)
    raise ValueError("query not provided and topic spec has no question_template/keywords")


def spec_from_dict(payload: dict[str, Any]) -> TopicSpec:
    return TopicSpec.model_validate(payload)
