from __future__ import annotations

from pathlib import Path

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary
from robot_data_studio.quality.models import CleaningConfig, VlmEvaluation, VlmSettings
from robot_data_studio.quality.scorer import EpisodeQualityScorer


class FakeReader:
    root = Path("/tmp/fake-dataset")

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            path=str(self.root),
            format="lerobot",
            version="v3.0",
            robot_type="test",
            total_episodes=1,
            total_frames=3,
            fps=10,
            video_keys=["observation.image"],
            scalar_keys=["observation.state", "action"],
            features={},
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        return [self.episode(0)]

    def episode(self, episode_index: int) -> EpisodeSummary:
        return EpisodeSummary(
            episode_index=episode_index,
            length=3,
            duration_seconds=0.3,
            tasks=["push the block to the target"],
            data_file="data.parquet",
            video_files={"observation.image": "videos/episode.mp4"},
            video_start_seconds={"observation.image": 0.0},
            video_end_seconds={"observation.image": 0.3},
        )

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        return [
            EpisodeFrame(frame_index=0, timestamp=0.0, observation_state=[0.0, 0.0], action=[0.0, 0.0]),
            EpisodeFrame(frame_index=1, timestamp=0.1, observation_state=[1.0, 0.0], action=[1.0, 0.0]),
            EpisodeFrame(frame_index=2, timestamp=0.2, observation_state=[2.0, 0.0], action=[2.0, 0.0]),
        ]


class FakeVlmEvaluator:
    def evaluate(self, reader, episode_index: int, settings: VlmSettings) -> VlmEvaluation:
        assert settings.prompt == "Judge task: {task}"
        assert episode_index == 0
        return VlmEvaluation(success=False, score=0.2, reason="The block never reached the target.")


def test_vlm_task_success_is_included_as_a_quality_dimension() -> None:
    scorer = EpisodeQualityScorer(vlm_evaluator=FakeVlmEvaluator())
    config = CleaningConfig(vlm=VlmSettings(enabled=True, prompt="Judge task: {task}"))

    result = scorer.score_episode(FakeReader(), 0, 0.3, 0.3, config)

    assert result.per_attribute_scores["task_success"] == 0.2
    assert any(finding.code == "vlm_failed" for finding in result.findings)
    assert any("never reached" in finding.message for finding in result.findings)
