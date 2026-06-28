from __future__ import annotations

import numpy as np
import pytest

from robot_data_studio.quality.filter_models import FilterVisualQualityConfig
from robot_data_studio.quality.visual_quality import (
    VisualQualityRow,
    aggregate_visual_quality_incidents,
    analyze_sampled_frames,
    sample_video_frames,
)


def _flat(value: int) -> np.ndarray:
    return np.full((24, 32, 3), value, dtype=np.uint8)


def _checkerboard() -> np.ndarray:
    pattern = (np.indices((24, 32)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def test_clear_high_contrast_frames_pass_visual_quality() -> None:
    result = analyze_sampled_frames(
        [_checkerboard(), np.roll(_checkerboard(), 1, axis=1)],
        FilterVisualQualityConfig(),
        has_motion=True,
    )

    assert result.issue_count == 0
    assert result.score == 1
    assert result.rows == []


def test_blur_dark_bright_and_low_contrast_frames_are_flagged() -> None:
    result = analyze_sampled_frames(
        [_flat(128), _flat(3), _flat(252), _flat(40)],
        FilterVisualQualityConfig(),
        has_motion=False,
    )

    issues = [row.issue for row in result.rows]

    assert "blur" in issues
    assert "dark" in issues
    assert "bright" in issues
    assert "low_contrast" in issues
    assert result.issue_count >= 4
    assert result.score < 1


def test_global_bright_washout_is_flagged_before_white_screen() -> None:
    frame = np.full((24, 32, 3), 165, dtype=np.uint8)
    frame[:12] = 210

    result = analyze_sampled_frames(
        [frame],
        FilterVisualQualityConfig(),
        has_motion=False,
    )

    bright_rows = [row for row in result.rows if row.issue == "bright"]
    assert bright_rows
    assert "p75" in str(bright_rows[0].value)


def test_global_dark_exposure_is_flagged_before_black_screen() -> None:
    frame = np.full((24, 32, 3), 70, dtype=np.uint8)

    result = analyze_sampled_frames(
        [frame],
        FilterVisualQualityConfig(),
        has_motion=False,
    )

    dark_rows = [row for row in result.rows if row.issue == "dark"]
    assert dark_rows
    assert "p75" in str(dark_rows[0].value)


def test_repeated_frames_with_motion_are_flagged_as_freeze() -> None:
    frame = _checkerboard()

    result = analyze_sampled_frames(
        [frame.copy() for _ in range(5)],
        FilterVisualQualityConfig(freeze_min_run=3),
        has_motion=True,
        frame_indexes=[0, 25, 50, 75, 100],
        timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
    )

    freeze = next(row for row in result.rows if row.issue == "freeze")
    assert freeze.frame == 0
    assert freeze.end_frame == 100
    assert freeze.timestamp == 0.0
    assert freeze.end_timestamp == 2.0
    assert result.critical_issue


def test_repeated_frames_without_motion_do_not_trigger_freeze() -> None:
    frame = _checkerboard()

    result = analyze_sampled_frames(
        [frame.copy() for _ in range(5)],
        FilterVisualQualityConfig(freeze_min_run=3),
        has_motion=False,
    )

    assert not any(row.issue == "freeze" for row in result.rows)


def test_sampled_rows_and_metrics_use_episode_frame_indexes() -> None:
    result = analyze_sampled_frames(
        [_flat(128), _flat(128)],
        FilterVisualQualityConfig(
            dark_mean_threshold=0,
            dark_global_mean_threshold=0,
            dark_global_p75_threshold=0,
            bright_mean_threshold=255,
            bright_global_mean_threshold=255,
            bright_global_p75_threshold=255,
            low_contrast_std_threshold=0,
        ),
        has_motion=False,
        camera="cam_high",
        frame_indexes=[25, 50],
        timestamps=[0.5, 1.0],
    )

    assert [row.frame for row in result.rows] == [25, 50]
    assert [metric.frame for metric in result.metrics["cam_high"]] == [25, 50]
    assert [metric.timestamp for metric in result.metrics["cam_high"]] == [0.5, 1.0]


def test_contiguous_rows_are_grouped_with_start_worst_and_end_evidence() -> None:
    rows = [
        VisualQualityRow(0, 0.0, "cam_high", "blur", 12.0, 18.0, metric_value=12.0),
        VisualQualityRow(25, 0.5, "cam_high", "blur", 4.0, 18.0, metric_value=4.0),
        VisualQualityRow(50, 1.0, "cam_high", "blur", 10.0, 18.0, metric_value=10.0),
        VisualQualityRow(100, 2.0, "cam_high", "blur", 8.0, 18.0, metric_value=8.0),
        VisualQualityRow(25, 0.5, "cam_wrist", "blur", 5.0, 18.0, metric_value=5.0),
    ]

    incidents = aggregate_visual_quality_incidents(rows, sample_interval_seconds=0.5)

    assert len(incidents) == 3
    first = incidents[0]
    assert first.camera == "cam_high"
    assert first.issue == "blur"
    assert (first.start_frame, first.end_frame) == (0, 50)
    assert first.sample_count == 3
    assert first.worst_value == 4.0
    assert [frame.frame for frame in first.representative_frames] == [0, 25, 50]


def test_video_sampling_is_limited_to_episode_time_bounds(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "shared.mp4"
    video.write_bytes(b"video")
    captured: list[list[str]] = []

    class Completed:
        stdout = bytes(160 * 120 * 3)

    def fake_run(command, **_kwargs):
        captured.append(command)
        return Completed()

    monkeypatch.setattr("robot_data_studio.quality.visual_quality._ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr("robot_data_studio.quality.visual_quality.subprocess.run", fake_run)

    sample_video_frames(
        video,
        FilterVisualQualityConfig(max_frames_per_video=1),
        start_seconds=12.5,
        duration_seconds=2.25,
    )

    command = captured[0]
    assert command[command.index("-ss") + 1] == "12.5"
    assert command[command.index("-t") + 1] == "2.25"
    assert command.index("-ss") < command.index("-i")
