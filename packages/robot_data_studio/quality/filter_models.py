from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


FilterStageId = Literal[
    "sudden_change",
    "state_action_alignment",
    "extreme_value",
    "kinematic_consistency",
    "orientation_alignment",
]
FilterStatus = Literal["passed", "review", "skipped"]


class FilterFinding(BaseModel):
    code: str
    severity: Literal["info", "warn", "error"]
    message: str


class FilterStageSummary(BaseModel):
    id: FilterStageId
    label: str
    count: int
    status: FilterStatus
    skipped_reason: str | None = None


class EpisodeFilterStageStatus(BaseModel):
    count: int
    status: FilterStatus
    skipped_reason: str | None = None


class EpisodeFilterSummary(BaseModel):
    episode_index: int
    stage_status: dict[FilterStageId, EpisodeFilterStageStatus]


class FilterSummary(BaseModel):
    dataset_path: str
    total_episodes: int
    total_frames: int
    stages: list[FilterStageSummary]
    episodes: list[EpisodeFilterSummary]


class FilterRun(BaseModel):
    run_id: str
    status: Literal["succeeded", "failed"]
    summary: FilterSummary


class FilterDetail(BaseModel):
    stage_id: FilterStageId
    episode_index: int
    title: str
    status: FilterStatus
    series: dict[str, list[float]] = Field(default_factory=dict)
    thresholds: dict[str, dict[str, float]] = Field(default_factory=dict)
    table_rows: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    findings: list[FilterFinding] = Field(default_factory=list)
    skipped_reason: str | None = None


class FilterKinematicsConfig(BaseModel):
    urdf_path: str | None = None
    end_effector_link: str | None = None
    joint_names: list[str] = Field(default_factory=list)
    joint_state_indices: list[int] = Field(default_factory=list)
    eef_position_indices: list[int] = Field(default_factory=list)
    position_tolerance: float = 0.05
    resolve_tcp_offset: bool = True


class FilterConfig(BaseModel):
    gripper_indices: list[int] = Field(default_factory=list)
    kinematics: FilterKinematicsConfig = Field(default_factory=FilterKinematicsConfig)


class FilterConfigPatch(BaseModel):
    gripper_indices: list[int] | None = None
    kinematics: FilterKinematicsConfig | None = None
