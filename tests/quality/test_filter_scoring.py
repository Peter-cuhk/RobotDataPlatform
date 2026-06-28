from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary
from robot_data_studio.quality.filter_models import FilterConfig
from robot_data_studio.quality.filter_service import DatasetFilterService, infer_gripper_indices
from robot_data_studio.quality.models import CleaningConfig
from robot_data_studio.quality.scorer import EpisodeQualityScorer


class SpikeReader:
    root = Path("/tmp/spike-dataset")

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            path=str(self.root),
            format="lerobot",
            version="v3.0",
            robot_type="test",
            total_episodes=1,
            total_frames=101,
            fps=50,
            video_keys=[],
            scalar_keys=["observation.state", "action"],
            features={},
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes = [self.episode(0)]
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        return EpisodeSummary(
            episode_index=episode_index,
            length=101,
            duration_seconds=2.02,
            tasks=["test spike"],
            data_file="data.parquet",
            video_files={},
            video_start_seconds={},
            video_end_seconds={},
        )

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = []
        for index in range(101):
            value = 10.0 * self.scale if index == 50 else 0.0
            frames.append(
                EpisodeFrame(
                    frame_index=index,
                    timestamp=index / 50,
                    observation_state=[0.0],
                    action=[value],
                )
            )
        return frames


class BurstReader(SpikeReader):
    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = []
        for index in range(101):
            value = 10.0 if 45 <= index <= 55 and index % 2 else 0.0
            frames.append(
                EpisodeFrame(
                    frame_index=index,
                    timestamp=index / 50,
                    observation_state=[0.0],
                    action=[value],
                )
            )
        return frames


class MultiDimSpikeReader(SpikeReader):
    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = []
        for index in range(101):
            action = [0.0, 8.0 if index == 50 else 0.0]
            frames.append(
                EpisodeFrame(
                    frame_index=index,
                    timestamp=index / 50,
                    observation_state=[0.0, 0.0],
                    action=action,
                )
            )
        return frames


class NonFiniteReader(SpikeReader):
    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = super().read_episode_frames(episode_index)
        frames[50] = frames[50].model_copy(
            update={"observation_state": [float("nan")]}
        )
        return frames


class TimestampRegressionReader(SpikeReader):
    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = super().read_episode_frames(episode_index)
        frames[50] = frames[50].model_copy(update={"timestamp": 0.5})
        return frames


class CrossEpisodeDimensionReader(SpikeReader):
    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes = [self.episode(0), self.episode(1)]
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        episode = super().episode(episode_index)
        return episode.model_copy(update={"episode_index": episode_index})

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = super().read_episode_frames(episode_index)
        if episode_index == 1:
            return [
                frame.model_copy(
                    update={"observation_state": [*frame.observation_state, 0.0]}
                )
                for frame in frames
            ]
        return frames


class ThreeEpisodeReader(SpikeReader):
    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes = [self.episode(0), self.episode(1), self.episode(2)]
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        episode = super().episode(episode_index)
        return episode.model_copy(update={"episode_index": episode_index})


class CountingFrameReader(ThreeEpisodeReader):
    def __init__(self) -> None:
        super().__init__()
        self.read_counts: dict[int, int] = {}

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        self.read_counts[episode_index] = self.read_counts.get(episode_index, 0) + 1
        return super().read_episode_frames(episode_index)


class NamedGripperReader(SpikeReader):
    def metadata(self) -> DatasetMetadata:
        metadata = super().metadata()
        return metadata.model_copy(
            update={
                "features": {
                    "observation.state": {
                        "names": ["joint", "main_gripper"],
                    }
                }
            }
        )


class SharedVideoReader(SpikeReader):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root

    def metadata(self) -> DatasetMetadata:
        metadata = super().metadata()
        return metadata.model_copy(update={"video_keys": ["observation.images.cam_high"]})

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes = [self.episode(0), self.episode(1)]
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        episode = super().episode(episode_index)
        start = episode_index * episode.duration_seconds
        return episode.model_copy(
            update={
                "episode_index": episode_index,
                "video_files": {"observation.images.cam_high": "videos/shared.mp4"},
                "video_start_seconds": {"observation.images.cam_high": start},
                "video_end_seconds": {
                    "observation.images.cam_high": start + episode.duration_seconds
                },
            }
        )


def test_visual_quality_samples_each_episode_segment_from_shared_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "videos" / "shared.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake video")
    calls = []

    def sample_once(path: Path, _config, *, start_seconds: float, duration_seconds: float):
        calls.append((path, start_seconds, duration_seconds))
        return [np.zeros((120, 160, 3), dtype=np.uint8)]

    monkeypatch.setattr(
        "robot_data_studio.quality.filter_service.sample_video_frames",
        sample_once,
    )

    DatasetFilterService(SharedVideoReader(tmp_path), FilterConfig()).summary()

    assert calls == [
        (video_path, 0.0, 2.02),
        (video_path, 2.02, 2.02),
    ]


def test_visual_quality_detail_exposes_real_episode_frames_and_incidents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "videos" / "shared.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake video")

    monkeypatch.setattr(
        "robot_data_studio.quality.filter_service.sample_video_frames",
        lambda *_args, **_kwargs: [
            np.full((120, 160, 3), 128, dtype=np.uint8),
            np.full((120, 160, 3), 128, dtype=np.uint8),
        ],
    )

    detail = DatasetFilterService(SharedVideoReader(tmp_path), FilterConfig()).detail(
        "visual_quality", 1
    )

    assert detail.visual_quality is not None
    assert detail.visual_quality.sampled_frame_count == 2
    assert detail.visual_quality.camera_count == 1
    incident = detail.visual_quality.incidents[0]
    assert (incident.start_frame, incident.end_frame) == (0, 25)
    assert [frame.frame for frame in incident.representative_frames] == [0, 25]


def test_filter_summary_reuses_reference_normalization_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original_quantile = np.quantile

    def counted_quantile(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_quantile(*args, **kwargs)

    monkeypatch.setattr(np, "quantile", counted_quantile)

    DatasetFilterService(ThreeEpisodeReader(), FilterConfig()).summary()

    assert calls <= 8


def test_filter_summary_reads_each_episode_frames_once() -> None:
    reader = CountingFrameReader()

    DatasetFilterService(reader, FilterConfig()).summary()

    assert reader.read_counts == {0: 1, 1: 1, 2: 1}


def test_sudden_change_stage_exposes_continuous_health_score() -> None:
    summary = DatasetFilterService(SpikeReader(), FilterConfig()).summary()

    stage = summary.episodes[0].stage_status["sudden_change"]

    assert stage.count > 0
    assert stage.score is not None
    assert 0 <= stage.score < 1
    assert stage.severity in {"warning", "critical"}


def test_sudden_change_score_is_invariant_to_signal_units() -> None:
    radians = DatasetFilterService(SpikeReader(scale=0.001), FilterConfig()).summary()
    servo_units = DatasetFilterService(SpikeReader(scale=1000.0), FilterConfig()).summary()

    radians_stage = radians.episodes[0].stage_status["sudden_change"]
    servo_stage = servo_units.episodes[0].stage_status["sudden_change"]

    assert radians_stage.count == servo_stage.count
    assert radians_stage.score == servo_stage.score


def test_sudden_change_detail_uses_the_same_normalized_analysis() -> None:
    service = DatasetFilterService(SpikeReader(scale=0.001), FilterConfig())

    summary_stage = service.summary().episodes[0].stage_status["sudden_change"]
    detail = service.detail("sudden_change", 0)

    assert summary_stage.status == "review"
    assert detail.status == summary_stage.status
    assert detail.table_rows


def test_sudden_change_detail_reports_dimension_metric_and_ratio() -> None:
    detail = DatasetFilterService(MultiDimSpikeReader(), FilterConfig()).detail("sudden_change", 0)

    first_row = detail.table_rows[0]

    assert first_row["source"] == "action"
    assert first_row["dimension"] == 1
    assert first_row["metric"] in {"delta", "acceleration", "jerk"}
    assert first_row["ratio"] > 1


def test_sudden_change_detail_keeps_original_dimension_after_gripper_exemption() -> None:
    detail = DatasetFilterService(
        MultiDimSpikeReader(),
        FilterConfig(gripper_indices=[0]),
    ).detail("sudden_change", 0)

    assert detail.table_rows[0]["dimension"] == 1


def test_balanced_default_quality_weights_prioritize_hard_failures() -> None:
    assert CleaningConfig().quality_weights == {
        "visual_quality": 1.5,
        "sudden_change": 1.5,
        "state_action_alignment": 1.5,
        "extreme_value": 2.0,
        "kinematic_consistency": 2.0,
        "orientation_alignment": 1.0,
        "task_success": 2.0,
    }


def test_quality_weights_are_limited_to_the_slider_range() -> None:
    with pytest.raises(ValueError, match="between 0.25 and 3.0"):
        CleaningConfig(quality_weights={"sudden_change": 3.25})


def test_gripper_indices_are_inferred_from_list_style_motor_names() -> None:
    assert infer_gripper_indices(NamedGripperReader()) == [1]


def test_sudden_change_ignores_configured_gripper_dimensions() -> None:
    summary = DatasetFilterService(
        SpikeReader(),
        FilterConfig(gripper_indices=[0]),
    ).summary()

    assert summary.episodes[0].stage_status["sudden_change"].count == 0


def test_sudden_change_detail_handles_all_dimensions_exempted_as_gripper() -> None:
    detail = DatasetFilterService(
        SpikeReader(),
        FilterConfig(gripper_indices=[0]),
    ).detail("sudden_change", 0)

    assert detail.status == "passed"
    assert detail.table_rows == []


def test_cleaning_score_uses_the_exact_filter_stage_scores() -> None:
    reader = SpikeReader()
    filter_summary = DatasetFilterService(reader, FilterConfig()).summary()

    result = EpisodeQualityScorer().score_dataset(
        reader,
        CleaningConfig(),
        filter_summary=filter_summary,
    )[0]
    stage_status = filter_summary.episodes[0].stage_status
    expected_scores = {
        stage_id: stage.score
        for stage_id, stage in stage_status.items()
        if stage.score is not None
    }
    weighted_total = sum(
        expected_scores[stage_id] * CleaningConfig().quality_weights[stage_id]
        for stage_id in expected_scores
    )
    weight_total = sum(
        CleaningConfig().quality_weights[stage_id] for stage_id in expected_scores
    )

    assert result.per_attribute_scores == expected_scores
    assert result.data_quality_score == weighted_total / weight_total
    assert result.score == result.data_quality_score
    assert result.task_success_score is None


def test_visual_quality_stage_is_skipped_when_video_is_missing() -> None:
    summary = DatasetFilterService(SpikeReader(), FilterConfig()).summary()

    stage = summary.episodes[0].stage_status["visual_quality"]

    assert summary.stages[0].id == "visual_quality"
    assert stage.status == "skipped"
    assert stage.score is None
    assert stage.skipped_reason == "video_missing"


def test_visual_quality_detail_reports_missing_video_without_blocking_other_filters() -> None:
    detail = DatasetFilterService(SpikeReader(), FilterConfig()).detail("visual_quality", 0)

    assert detail.stage_id == "visual_quality"
    assert detail.status == "skipped"
    assert detail.skipped_reason == "video_missing"
    assert detail.findings[0].code == "video_missing"
    assert detail.visual_quality is not None
    assert detail.visual_quality.sampled_frame_count == 0
    assert detail.table_rows[0]["issue"] == "video_missing"


def test_sudden_change_is_weighted_but_not_a_critical_integrity_gate() -> None:
    reader = BurstReader()
    filter_summary = DatasetFilterService(reader, FilterConfig()).summary()
    config = CleaningConfig(
        enabled_filter_stages=["sudden_change", "extreme_value"],
        quality_weights={
            "sudden_change": 0.25,
            "extreme_value": 3.0,
        }
    )

    result = EpisodeQualityScorer().score_dataset(
        reader,
        config,
        filter_summary=filter_summary,
    )[0]

    assert filter_summary.episodes[0].stage_status["sudden_change"].severity == "warning"
    assert result.status != "excluded"


def test_non_finite_signal_is_a_critical_integrity_gate() -> None:
    reader = NonFiniteReader()
    filter_summary = DatasetFilterService(reader, FilterConfig()).summary()

    result = EpisodeQualityScorer().score_dataset(
        reader,
        CleaningConfig(),
        filter_summary=filter_summary,
    )[0]

    assert [finding.code for finding in filter_summary.episodes[0].critical_findings] == [
        "non_finite_signal"
    ]
    assert result.data_quality_score == 0
    assert result.status == "excluded"


def test_timestamp_regression_is_a_critical_integrity_gate() -> None:
    reader = TimestampRegressionReader()
    filter_summary = DatasetFilterService(reader, FilterConfig()).summary()

    episode = filter_summary.episodes[0]

    assert [finding.code for finding in episode.critical_findings] == [
        "non_monotonic_timestamp"
    ]


def test_cross_episode_dimension_mismatch_is_a_critical_integrity_gate() -> None:
    reader = CrossEpisodeDimensionReader()
    filter_summary = DatasetFilterService(reader, FilterConfig()).summary()

    assert filter_summary.episodes[0].critical_findings == []
    assert [finding.code for finding in filter_summary.episodes[1].critical_findings] == [
        "dimension_mismatch"
    ]
