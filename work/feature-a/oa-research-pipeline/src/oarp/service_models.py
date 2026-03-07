from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class JobType(str, Enum):
    RESEARCH_INDEX = "research-index"
    RECIPE_GENERATE = "recipe-generate"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class NumericRange(BaseModel):
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> "NumericRange":
        if self.min is not None and self.max is not None and self.max < self.min:
            raise ValueError("max must be >= min")
        return self


class PhaseTarget(BaseModel):
    type: Literal["space_group", "amorphous", "phase_label"] = "phase_label"
    value: str

    @field_validator("value")
    @classmethod
    def _value_non_empty(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("value must be non-empty")
        return clean


class RecipeObjective(BaseModel):
    target_quality_numeric: NumericRange | None = None
    priority: str = "target-satisfaction"

    @field_validator("priority")
    @classmethod
    def _priority_non_empty(cls, value: str) -> str:
        clean = value.strip().lower()
        if not clean:
            raise ValueError("priority must be non-empty")
        return clean


class SafetyPolicy(BaseModel):
    forbidden_precursors: list[str] = Field(default_factory=list)
    forbidden_methods: list[str] = Field(default_factory=list)
    max_pressure_pa: float | None = None
    max_temperature_c: float | None = None
    notes: str = ""


class RecipeGenerateRequest(BaseModel):
    target_film_material: str
    target_phase_target: PhaseTarget
    system_family_hint: str = ""
    knowledge_run_dir: str = ""
    numeric_constraints: dict[str, NumericRange] = Field(default_factory=dict)
    text_constraints: dict[str, list[str]] = Field(default_factory=dict)
    objective: RecipeObjective = Field(default_factory=RecipeObjective)
    gate_profile: str = "progressive_default"
    max_candidates: int = 200
    top_k: int = 10
    max_retry_loops: int = 2
    use_materials_project: bool = True
    safety_policy: SafetyPolicy = Field(default_factory=SafetyPolicy)

    @field_validator("target_film_material")
    @classmethod
    def _material_non_empty(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("target_film_material must be non-empty")
        return clean

    @field_validator("max_candidates")
    @classmethod
    def _max_candidates_valid(cls, value: int) -> int:
        return max(1, min(int(value), 5000))

    @field_validator("top_k")
    @classmethod
    def _top_k_valid(cls, value: int) -> int:
        return max(1, min(int(value), 500))

    @field_validator("max_retry_loops")
    @classmethod
    def _retry_valid(cls, value: int) -> int:
        return max(0, min(int(value), 5))


class ResearchIndexRequest(BaseModel):
    spec_path: str
    query: str
    run_dir: str
    cfg_overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("spec_path", "query", "run_dir")
    @classmethod
    def _required_text(cls, value: str) -> str:
        clean = value.strip()
        if not clean:
            raise ValueError("value must be non-empty")
        return clean


class RecipeCardStep(BaseModel):
    step: int
    action: str
    method_family: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)


class RecipeCard(BaseModel):
    recipe_id: str
    target: dict[str, Any]
    method_flow: list[RecipeCardStep]
    condition_vector: dict[str, Any]
    predicted_outcomes: dict[str, Any]
    uncertainty: dict[str, Any]
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    gate_stage_passed: str


class RecipeGenerateResult(BaseModel):
    job_id: str
    status: str
    phase: str
    recipes: list[RecipeCard]
    gate_report: dict[str, Any]
    evidence_report: dict[str, Any]


class RecipeResult(RecipeGenerateResult):
    pass


class JobRef(BaseModel):
    job_id: str
    status: JobStatus


class JobStatusPayload(BaseModel):
    job_id: str
    job_type: JobType
    status: JobStatus
    run_dir: str
    retries: int = 0
    max_retries: int = 0
    created_at: str
    started_at: str = ""
    finished_at: str = ""
    error: str = ""


class ServiceConfig(BaseModel):
    data_dir: str = "~/.oarp_service"
    jobs_db_path: str = ""
    host: str = "127.0.0.1"
    port: int = 8787
    worker_poll_sec: float = 0.5
    user_agent_suffix: str = "service"
    materials_project_enabled: bool = True
    materials_project_api_key: str = Field(default_factory=lambda: os.getenv("MP_API_KEY", ""))
    materials_project_endpoint: str = "https://api.materialsproject.org"
    materials_project_scope: str = "summary_thermo"

    @field_validator("port")
    @classmethod
    def _port_valid(cls, value: int) -> int:
        value = int(value)
        if value <= 0 or value > 65535:
            raise ValueError("port must be in [1, 65535]")
        return value

    @field_validator("worker_poll_sec")
    @classmethod
    def _poll_valid(cls, value: float) -> float:
        return max(0.1, min(float(value), 10.0))

    def data_root(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    def jobs_db(self) -> Path:
        if self.jobs_db_path.strip():
            return Path(self.jobs_db_path).expanduser().resolve()
        return self.data_root() / "jobs.sqlite"

    def jobs_runs_root(self) -> Path:
        return self.data_root() / "jobs"
