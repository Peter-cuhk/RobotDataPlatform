from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robot_data_studio.lerobot.models import EpisodeFrame, EpisodeSummary
from robot_data_studio.quality.filter_models import FilterTimeSyncConfig


@dataclass(frozen=True)
class TimeSyncRow:
    issue: str
    value: float
    expected: float
    delta: float
    threshold: float
    frame: int | None = None
    camera: str | None = None


@dataclass
class TimeSyncResult:
    frame_count: int
    issue_count: int
    score: float
    rows: list[TimeSyncRow] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)


def analyze_time_sync(
    episode: EpisodeSummary,
    frames: list[EpisodeFrame],
    fps: float,
    config: FilterTimeSyncConfig | None = None,
) -> TimeSyncResult:
    config = config or FilterTimeSyncConfig()
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    expected_delta = 1.0 / fps if fps > 0 else 0.0
    jitter_threshold = max(
        config.timestamp_jitter_seconds,
        expected_delta * config.timestamp_jitter_ratio,
    )
    rows: list[TimeSyncRow] = []
    deltas = np.diff(timestamps) if len(timestamps) >= 2 else np.asarray([], dtype=np.float64)

    for frame, delta in enumerate(deltas, start=1):
        offset = abs(float(delta) - expected_delta)
        if offset > jitter_threshold:
            rows.append(
                TimeSyncRow(
                    issue="timestamp_gap",
                    frame=frame,
                    value=float(delta),
                    expected=expected_delta,
                    delta=offset,
                    threshold=jitter_threshold,
                )
            )

    if len(timestamps) >= 2:
        actual_span = float(timestamps[-1] - timestamps[0])
        expected_span = (len(timestamps) - 1) * expected_delta
        span_delta = abs(actual_span - expected_span)
        if span_delta > config.duration_tolerance_seconds:
            rows.append(
                TimeSyncRow(
                    issue="timestamp_span_mismatch",
                    value=actual_span,
                    expected=expected_span,
                    delta=span_delta,
                    threshold=config.duration_tolerance_seconds,
                )
            )

    first_timestamp = float(timestamps[0]) if len(timestamps) else 0.0
    expected_video_end = first_timestamp + episode.length * expected_delta
    for camera in episode.video_files:
        if camera in episode.video_start_seconds:
            start = float(episode.video_start_seconds[camera])
            start_delta = abs(start - first_timestamp)
            if start_delta > config.video_boundary_tolerance_seconds:
                rows.append(
                    TimeSyncRow(
                        issue="video_start_offset",
                        camera=camera,
                        value=start,
                        expected=first_timestamp,
                        delta=start_delta,
                        threshold=config.video_boundary_tolerance_seconds,
                    )
                )
        if camera in episode.video_end_seconds:
            end = float(episode.video_end_seconds[camera])
            end_delta = abs(end - expected_video_end)
            if end_delta > config.video_boundary_tolerance_seconds:
                rows.append(
                    TimeSyncRow(
                        issue="video_end_offset",
                        camera=camera,
                        value=end,
                        expected=expected_video_end,
                        delta=end_delta,
                        threshold=config.video_boundary_tolerance_seconds,
                    )
                )

    denominator = max(len(deltas) + max(len(episode.video_files) * 2, 1), 1)
    issue_count = len(rows)
    score = max(0.0, 1.0 - issue_count / denominator)
    return TimeSyncResult(
        frame_count=len(frames),
        issue_count=issue_count,
        score=score,
        rows=rows,
        series={"timestamp_delta": _sample_series(deltas)},
    )


def _sample_series(values: np.ndarray, limit: int = 240) -> list[float]:
    if len(values) <= limit:
        return [round(float(value), 6) for value in values]
    indexes = np.linspace(0, len(values) - 1, limit).astype(int)
    return [round(float(values[index]), 6) for index in indexes]
