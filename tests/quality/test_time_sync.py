from __future__ import annotations

from pathlib import Path

import pytest

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary
from robot_data_studio.quality.filter_models import FilterConfig
from robot_data_studio.quality.filter_service import DatasetFilterService
from robot_data_studio.quality.time_sync import analyze_time_sync


class TimeSyncReader:
    root = Path("/tmp/time-sync-dataset")

    def __init__(
        self,
        *,
        timestamps: list[float] | None = None,
        fps: float = 10.0,
        video_start_seconds: dict[str, float] | None = None,
        video_end_seconds: dict[str, float] | None = None,
        video_files: dict[str, str] | None = None,
    ) -> None:
        self.timestamps = timestamps or [index / fps for index in range(5)]
        self.fps = fps
        self.video_files = video_files if video_files is not None else {"observation.image": "video.mp4"}
        self.video_start_seconds = (
            video_start_seconds
            if video_start_seconds is not None
            else {key: self.timestamps[0] for key in self.video_files}
        )
        self.video_end_seconds = (
            video_end_seconds
            if video_end_seconds is not None
            else {key: self.timestamps[0] + len(self.timestamps) / fps for key in self.video_files}
        )

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            path=str(self.root),
            format="lerobot",
            version="v3.0",
            robot_type="test",
            total_episodes=1,
            total_frames=len(self.timestamps),
            fps=self.fps,
            video_keys=list(self.video_files),
            scalar_keys=["observation.state", "action"],
            features={},
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes = [self.episode(0)]
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        return EpisodeSummary(
            episode_index=episode_index,
            length=len(self.timestamps),
            duration_seconds=len(self.timestamps) / self.fps,
            tasks=["time sync"],
            data_file="data.parquet",
            video_files=self.video_files,
            video_start_seconds=self.video_start_seconds,
            video_end_seconds=self.video_end_seconds,
        )

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        return [
            EpisodeFrame(
                frame_index=index,
                timestamp=timestamp,
                observation_state=[float(index)],
                action=[float(index)],
            )
            for index, timestamp in enumerate(self.timestamps)
        ]


def test_time_sync_accepts_regular_scalar_timestamps() -> None:
    reader = TimeSyncReader(timestamps=[0.0, 0.1, 0.2, 0.3, 0.4])

    result = analyze_time_sync(reader.episode(0), reader.read_episode_frames(0), reader.metadata().fps)

    assert result.issue_count == 0
    assert result.score == 1.0
    assert result.rows == []


def test_time_sync_flags_timestamp_gap() -> None:
    reader = TimeSyncReader(timestamps=[0.0, 0.1, 0.35, 0.45])
    service = DatasetFilterService(reader, FilterConfig())

    summary = service.summary()
    detail = service.detail("state_action_alignment", 0)

    stage = summary.episodes[0].stage_status["state_action_alignment"]
    assert stage.count > 0
    assert stage.status == "review"
    assert detail.stage_id == "state_action_alignment"
    assert detail.title == "Time sync"
    assert "timestamp_delta" in detail.series
    assert any(row["issue"] == "timestamp_gap" for row in detail.table_rows)


def test_time_sync_flags_timestamp_span_mismatch() -> None:
    reader = TimeSyncReader(timestamps=[0.0, 0.1, 0.2, 0.65])
    detail = DatasetFilterService(reader, FilterConfig()).detail("state_action_alignment", 0)

    assert any(row["issue"] == "timestamp_span_mismatch" for row in detail.table_rows)


def test_time_sync_flags_video_boundary_offsets() -> None:
    reader = TimeSyncReader(
        timestamps=[0.0, 0.1, 0.2, 0.3],
        video_start_seconds={"observation.image": 0.25},
        video_end_seconds={"observation.image": 0.7},
    )

    detail = DatasetFilterService(reader, FilterConfig()).detail("state_action_alignment", 0)
    issues = {row["issue"] for row in detail.table_rows}

    assert {"video_start_offset", "video_end_offset"} <= issues


def test_time_sync_passes_without_video_streams() -> None:
    reader = TimeSyncReader(
        timestamps=[0.0, 0.1, 0.2, 0.3],
        video_files={},
        video_start_seconds={},
        video_end_seconds={},
    )

    summary = DatasetFilterService(reader, FilterConfig()).summary()

    stage = summary.episodes[0].stage_status["state_action_alignment"]
    assert stage.status == "passed"
    assert stage.count == 0
    assert stage.skipped_reason is None
    assert stage.score == pytest.approx(1.0)
