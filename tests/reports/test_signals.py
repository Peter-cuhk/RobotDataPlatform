from __future__ import annotations

from pathlib import Path

import pytest

from robot_data_studio.lerobot.models import (
    DatasetMetadata,
    EpisodeFrame,
    EpisodeSummary,
)
from robot_data_studio.reports import build_report_signals


class MemoryAdapter:
    root = Path(".")

    def __init__(
        self,
        *,
        features: dict[str, dict] | None = None,
        frames: list[EpisodeFrame] | None = None,
    ) -> None:
        self._features = features or {}
        self._frames = frames or []
        self._episodes = [
            _episode(1, 2.0),
            _episode(0, 1.0),
        ]

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            path="/tmp/dataset",
            format="lerobot",
            version="v2.1",
            robot_type="aloha",
            total_episodes=2,
            total_frames=4,
            fps=2,
            video_keys=[],
            scalar_keys=["action"],
            features=self._features,
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        return self._episodes[:limit] if limit is not None else self._episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        for episode in self._episodes:
            if episode.episode_index == episode_index:
                return episode
        raise KeyError(f"Episode {episode_index} not found")

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        self.episode(episode_index)
        return self._frames


def test_builds_ordered_duration_and_named_gripper_series() -> None:
    adapter = MemoryAdapter(
        features={
            "action": {
                "dtype": "float32",
                "shape": [4],
                "names": {
                    "motors": [
                        "left_waist",
                        "left_gripper",
                        "right_waist",
                        "right_gripper",
                    ]
                },
            }
        },
        frames=[
            _frame(0, 0.0, [0.0, 0.1, 0.0, 0.2]),
            _frame(1, 0.5, [0.0, 0.8, 0.0, 0.7]),
        ],
    )

    signals = build_report_signals(adapter, episode_index=0)

    assert [row.episode_index for row in signals.episode_durations] == [0, 1]
    assert signals.mean_episode_duration_seconds == pytest.approx(1.5)
    assert [series.label for series in signals.gripper_series] == [
        "left_gripper",
        "right_gripper",
    ]
    assert signals.gripper_series[0].points[0].model_dump() == {
        "timestamp": 0.0,
        "value": 0.1,
    }
    assert signals.gripper_unavailable_reason is None


def test_reports_missing_named_gripper_dimensions_without_losing_durations() -> None:
    signals = build_report_signals(MemoryAdapter(), episode_index=0)

    assert signals.gripper_series == []
    assert signals.gripper_unavailable_reason == "no_named_gripper_dimensions"
    assert len(signals.episode_durations) == 2


def test_reports_empty_gripper_samples_and_omits_non_finite_points() -> None:
    features = {
        "action": {
            "dtype": "float32",
            "shape": [1],
            "names": {"motors": ["main_gripper"]},
        }
    }
    empty = build_report_signals(
        MemoryAdapter(features=features),
        episode_index=0,
    )
    filtered = build_report_signals(
        MemoryAdapter(
            features=features,
            frames=[
                _frame(0, float("nan"), [0.1]),
                _frame(1, 0.5, [float("inf")]),
                _frame(2, 1.0, [0.9]),
            ],
        ),
        episode_index=0,
    )

    assert empty.gripper_unavailable_reason == "no_gripper_samples"
    assert filtered.gripper_unavailable_reason is None
    assert [point.model_dump() for point in filtered.gripper_series[0].points] == [
        {"timestamp": 1.0, "value": 0.9}
    ]


def _episode(episode_index: int, duration_seconds: float) -> EpisodeSummary:
    return EpisodeSummary(
        episode_index=episode_index,
        length=2,
        duration_seconds=duration_seconds,
        tasks=[],
        data_file=f"episode-{episode_index}.parquet",
        video_files={},
        video_start_seconds={},
        video_end_seconds={},
    )


def _frame(
    frame_index: int,
    timestamp: float,
    action: list[float],
) -> EpisodeFrame:
    return EpisodeFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        observation_state=[],
        action=action,
    )
