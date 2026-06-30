from __future__ import annotations

import math

from pydantic import BaseModel, Field

from robot_data_studio.formats.models import DatasetAdapter


class SignalPoint(BaseModel):
    timestamp: float
    value: float


class GripperSeries(BaseModel):
    label: str
    dimension_index: int
    points: list[SignalPoint] = Field(default_factory=list)


class EpisodeDurationRow(BaseModel):
    episode_index: int
    duration_seconds: float


class ReportSignals(BaseModel):
    episode_index: int
    gripper_series: list[GripperSeries] = Field(default_factory=list)
    episode_durations: list[EpisodeDurationRow] = Field(default_factory=list)
    mean_episode_duration_seconds: float
    gripper_unavailable_reason: str | None = None


def build_report_signals(
    adapter: DatasetAdapter,
    episode_index: int,
) -> ReportSignals:
    adapter.episode(episode_index)
    episodes = sorted(adapter.list_episodes(), key=lambda episode: episode.episode_index)
    duration_rows = [
        EpisodeDurationRow(
            episode_index=episode.episode_index,
            duration_seconds=episode.duration_seconds,
        )
        for episode in episodes
    ]
    mean_duration = (
        sum(row.duration_seconds for row in duration_rows) / len(duration_rows)
        if duration_rows
        else 0.0
    )
    dimensions = _find_named_dimensions(
        adapter.metadata().features,
        feature_key="action",
        token="gripper",
    )
    if not dimensions:
        return ReportSignals(
            episode_index=episode_index,
            episode_durations=duration_rows,
            mean_episode_duration_seconds=mean_duration,
            gripper_unavailable_reason="no_named_gripper_dimensions",
        )

    frames = adapter.read_episode_frames(episode_index)
    series = [
        GripperSeries(
            label=label,
            dimension_index=dimension_index,
            points=[
                SignalPoint(
                    timestamp=float(frame.timestamp),
                    value=float(frame.action[dimension_index]),
                )
                for frame in frames
                if len(frame.action) > dimension_index
                and math.isfinite(float(frame.timestamp))
                and math.isfinite(float(frame.action[dimension_index]))
            ],
        )
        for dimension_index, label in dimensions
    ]
    if not any(item.points for item in series):
        return ReportSignals(
            episode_index=episode_index,
            episode_durations=duration_rows,
            mean_episode_duration_seconds=mean_duration,
            gripper_unavailable_reason="no_gripper_samples",
        )
    return ReportSignals(
        episode_index=episode_index,
        gripper_series=series,
        episode_durations=duration_rows,
        mean_episode_duration_seconds=mean_duration,
    )


def _find_named_dimensions(
    features: dict[str, dict],
    feature_key: str,
    token: str,
) -> list[tuple[int, str]]:
    feature = features.get(feature_key)
    if not isinstance(feature, dict):
        return []
    names = feature.get("names")
    if not isinstance(names, dict):
        return []
    normalized_token = token.lower()
    dimensions: list[tuple[int, str]] = []
    for values in names.values():
        if not isinstance(values, list):
            continue
        for index, name in enumerate(values):
            if isinstance(name, str) and normalized_token in name.lower():
                dimensions.append((index, name))
    return dimensions
