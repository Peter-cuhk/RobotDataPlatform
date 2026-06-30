from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator


CleaningStatus = Literal["passed", "review", "excluded", "unscored"]
DecisionSource = Literal["auto", "manual"]
FindingSeverity = Literal["info", "warn", "error", "critical"]
VlmProvider = Literal["OpenAI", "Gemini", "Local"]
FilterStageId = Literal[
    "visual_quality",
    "sudden_change",
    "state_action_alignment",
    "extreme_value",
    "kinematic_consistency",
    "orientation_alignment",
    "metadata_completeness",
]


DEFAULT_ENABLED_FILTER_STAGES: list[FilterStageId] = [
    "visual_quality",
    "sudden_change",
    "state_action_alignment",
    "extreme_value",
    "kinematic_consistency",
    "orientation_alignment",
    "metadata_completeness",
]

DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "visual_quality": 1.5,
    "sudden_change": 1.5,
    "state_action_alignment": 1.5,
    "extreme_value": 2.0,
    "kinematic_consistency": 2.0,
    "orientation_alignment": 1.0,
    "task_success": 2.0,
}


DEFAULT_VLM_PROMPT = (
    "You are an automated robot episode evaluator. The images are sampled from the "
    "main camera at important trajectory moments: the start, gripper open/close "
    "events, and the final state. Judge whether the task was successfully completed "
    "from the sequence and final object state. Transparent objects, grayscale video, "
    "and small buttons can be subtle; inspect the full sequence carefully instead of "
    "assuming absence from one low-contrast frame. Return only JSON with success, "
    "score, and reason."
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
    enabled_filter_stages: list[FilterStageId] = Field(
        default_factory=lambda: DEFAULT_ENABLED_FILTER_STAGES.copy()
    )
    quality_weights: dict[str, float] = Field(default_factory=lambda: DEFAULT_QUALITY_WEIGHTS.copy())
    vlm: VlmSettings = Field(default_factory=VlmSettings)

    @field_validator("quality_weights")
    @classmethod
    def validate_quality_weights(cls, weights: dict[str, float]) -> dict[str, float]:
        invalid = {
            name: value
            for name, value in weights.items()
            if not 0.25 <= value <= 3.0
        }
        if invalid:
            raise ValueError("quality weights must be between 0.25 and 3.0")
        return weights


class QualityFinding(BaseModel):
    code: str
    severity: FindingSeverity
    message: str


class EpisodeQualityResult(BaseModel):
    episode_index: int
    score: float | None
    data_quality_score: float | None = None
    task_success_score: float | None = None
    status: CleaningStatus
    source: DecisionSource
    per_attribute_scores: dict[str, float]
    findings: list[QualityFinding]
    review_note: str | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def populate_compatible_scores(self) -> EpisodeQualityResult:
        if self.data_quality_score is None and self.score is not None:
            self.data_quality_score = self.score
        elif self.score is None and self.data_quality_score is not None:
            self.score = self.data_quality_score
        return self


class CleaningSummary(BaseModel):
    total: int
    passed_count: int
    review_count: int
    excluded_count: int
    unscored_count: int
    results: list[EpisodeQualityResult]
    config: CleaningConfig
    scorer_version: str
    requires_rerun: bool = False
    previous_scorer_version: str | None = None


class CleaningRun(BaseModel):
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    summary: CleaningSummary


class EpisodeDecisionRequest(BaseModel):
    status: Literal["passed", "review", "excluded"]
    note: str | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
