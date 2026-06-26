from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary
from robot_data_studio.quality.models import CleaningConfig, VlmEvaluation, VlmSettings
from robot_data_studio.quality.scorer import EpisodeQualityScorer
from robot_data_studio.quality.vlm import VlmTaskSuccessEvaluator


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


def test_vlm_task_failure_is_weighted_and_prevents_auto_pass() -> None:
    scorer = EpisodeQualityScorer(vlm_evaluator=FakeVlmEvaluator())
    config = CleaningConfig(
        pass_threshold=0.8,
        review_threshold=0.6,
        vlm=VlmSettings(enabled=True, prompt="Judge task: {task}"),
        quality_weights={"task_success": 3.0},
    )

    result = scorer.score_episode(FakeReader(), 0, 0.3, 0.3, config)

    assert result.score == pytest.approx(0.7)
    assert result.status == "review"
    assert any(finding.code == "vlm_failed" for finding in result.findings)


def test_openai_compatible_requests_include_mimo_api_key_header(monkeypatch) -> None:
    captured_headers = {}

    def fake_sample_frames(self, video_path, duration_seconds, sample_frames, sample_times=None):
        return [b"jpeg bytes"]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":'
                b'"{\\"success\\":true,\\"score\\":1,\\"reason\\":\\"ok\\"}"}}]}'
            )

    def fake_urlopen(request, timeout):
        captured_headers.update(dict(request.header_items()))
        return FakeResponse()

    monkeypatch.setattr(VlmTaskSuccessEvaluator, "_sample_video_frames", fake_sample_frames)
    monkeypatch.setattr("robot_data_studio.quality.vlm.urlrequest.urlopen", fake_urlopen)

    result = VlmTaskSuccessEvaluator()._evaluate_openai_compatible(
        Path("episode.mp4"),
        "make coffee",
        1.0,
        VlmSettings(api_key="tp-secret", model="mimo-v2.5"),
    )

    assert result.success is True
    assert captured_headers["Authorization"] == "Bearer tp-secret"
    assert captured_headers["Api-key"] == "tp-secret"


def test_ffmpeg_path_falls_back_to_imageio_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    class FakeImageioFfmpeg:
        @staticmethod
        def get_ffmpeg_exe():
            return "/tmp/imageio-ffmpeg"

    monkeypatch.setattr(
        "robot_data_studio.quality.vlm._import_imageio_ffmpeg",
        lambda: FakeImageioFfmpeg,
    )

    assert VlmTaskSuccessEvaluator()._ffmpeg_path() == "/tmp/imageio-ffmpeg"


def test_evaluator_prefers_high_camera_and_passes_trajectory_frames(tmp_path, monkeypatch) -> None:
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "left.mp4").write_bytes(b"left")
    (tmp_path / "videos" / "high.mp4").write_bytes(b"high")
    (tmp_path / "videos" / "low.mp4").write_bytes(b"low")
    captured = {}

    class MultiCameraReader(FakeReader):
        root = tmp_path

        def episode(self, episode_index: int) -> EpisodeSummary:
            episode = super().episode(episode_index)
            return episode.model_copy(
                update={
                    "video_files": {
                        "observation.images.cam_left_wrist": "videos/left.mp4",
                        "observation.images.cam_high": "videos/high.mp4",
                        "observation.images.cam_low": "videos/low.mp4",
                    }
                }
            )

    def fake_evaluate_openai(
        self,
        video_path,
        task,
        duration_seconds,
        settings,
        trajectory_frames=None,
        supplemental_video_paths=None,
    ):
        captured["video_path"] = video_path
        captured["trajectory_frames"] = trajectory_frames
        captured["supplemental_video_paths"] = supplemental_video_paths
        return VlmEvaluation(success=True, score=1.0, reason="ok")

    monkeypatch.setattr(
        VlmTaskSuccessEvaluator,
        "_evaluate_openai_compatible",
        fake_evaluate_openai,
    )

    VlmTaskSuccessEvaluator().evaluate(MultiCameraReader(), 0, VlmSettings(enabled=True))

    assert captured["video_path"] == tmp_path / "videos" / "high.mp4"
    assert captured["supplemental_video_paths"] == [
        ("observation.images.cam_low", tmp_path / "videos" / "low.mp4")
    ]
    assert len(captured["trajectory_frames"]) == 3


def test_keyframe_times_include_gripper_events_and_final_frame() -> None:
    frames = []
    for index in range(1100):
        state = [0.0] * 14
        state[6] = 0.1
        state[13] = 0.1
        if index >= 606:
            state[6] = 0.7
        if index >= 1035:
            state[13] = 0.6
        frames.append(
            EpisodeFrame(
                frame_index=index,
                timestamp=index / 50,
                observation_state=state,
                action=state,
            )
        )

    times = VlmTaskSuccessEvaluator()._keyframe_times(frames, sample_frames=4)

    assert times == pytest.approx([0.0, 12.12, 20.7, 21.98], abs=0.02)
