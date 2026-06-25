from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field


CleaningStatus = Literal["passed", "review", "excluded", "unscored"]
DecisionSource = Literal["auto", "manual"]
FindingSeverity = Literal["info", "warn", "error"]
VlmProvider = Literal["OpenAI", "Gemini", "Local"]


DEFAULT_VLM_PROMPT = (
    "You are an automated robot episode evaluator. Return only JSON with success, "
    "score, and reason. Judge whether the task was successfully completed from the "
    "visual evidence."
)


class VlmSettings(BaseModel):
    enabled: bool = False
    provider: VlmProvider = "OpenAI"
    model: str = "gpt-4o-mini"
    api_base_url: str | None = None
    api_key: str | None = Field(default=None, exclude=True)
    prompt: str = DEFAULT_VLM_PROMPT
    sample_frames: int = Field(default=4, ge=1, le=12)

    @computed_field
    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key)


class VlmEvaluation(BaseModel):
    success: bool
    score: float = Field(ge=0, le=1)
    reason: str = ""
    raw_response: dict | None = None


class CleaningConfig(BaseModel):
    pass_threshold: float = Field(default=0.8, ge=0, le=1)
    review_threshold: float = Field(default=0.6, ge=0, le=1)
    overwrite_manual: bool = False
    vlm: VlmSettings = Field(default_factory=VlmSettings)


class QualityFinding(BaseModel):
    code: str
    severity: FindingSeverity
    message: str


class EpisodeQualityResult(BaseModel):
    episode_index: int
    score: float | None
    status: CleaningStatus
    source: DecisionSource
    per_attribute_scores: dict[str, float]
    findings: list[QualityFinding]
    review_note: str | None = None
    updated_at: datetime


class CleaningSummary(BaseModel):
    total: int
    passed_count: int
    review_count: int
    excluded_count: int
    unscored_count: int
    results: list[EpisodeQualityResult]
    config: CleaningConfig
    scorer_version: str


class CleaningRun(BaseModel):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    summary: CleaningSummary


class EpisodeDecisionRequest(BaseModel):
    status: Literal["passed", "review", "excluded"]
    note: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
