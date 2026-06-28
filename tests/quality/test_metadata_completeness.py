from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_data_studio.formats.registry import FormatRegistry
from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary
from robot_data_studio.quality.filter_models import FilterConfig, MetadataCompletenessConfig
from robot_data_studio.quality.filter_service import DatasetFilterService
from robot_data_studio.quality.metadata_completeness import (
    analyze_metadata_completeness,
    build_metadata_inspection,
)
from robot_data_studio.quality.models import CleaningConfig
from robot_data_studio.quality.scorer import EpisodeQualityScorer
from tests.formats.test_hdf5_and_zarr_formats import write_act_hdf5


class MetadataReader:
    def __init__(
        self,
        *,
        root: Path | None = None,
        video_keys: list[str] | None = None,
        features: dict | None = None,
        episodes: list[EpisodeSummary] | None = None,
    ) -> None:
        self.root = root or Path("/tmp/metadata-dataset")
        self._video_keys = video_keys or []
        self._features = features or {
            "observation.state": {"shape": [7], "names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]},
            "action": {"shape": [7], "names": ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]},
            "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
        }
        self._episodes = episodes or [
            EpisodeSummary(
                episode_index=0,
                length=10,
                duration_seconds=0.2,
                tasks=["pick cube"],
                data_file="data.parquet",
                video_files={
                    "observation.images.top": "videos/top.mp4",
                    "observation.images.wrist": "videos/wrist.mp4",
                },
                video_start_seconds={
                    "observation.images.top": 0.0,
                    "observation.images.wrist": 0.0,
                },
                video_end_seconds={
                    "observation.images.top": 0.2,
                    "observation.images.wrist": 0.2,
                },
            )
        ]

    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            path=str(self.root),
            format="lerobot",
            version="v3.0",
            robot_type="test",
            total_episodes=len(self._episodes),
            total_frames=sum(episode.length for episode in self._episodes),
            fps=50,
            video_keys=self._video_keys or list(self._episodes[0].video_files),
            scalar_keys=["observation.state", "action"],
            features=self._features,
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes = self._episodes
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        return next(item for item in self._episodes if item.episode_index == episode_index)

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        episode = self.episode(episode_index)
        return [
            EpisodeFrame(
                frame_index=index,
                timestamp=index / 50,
                observation_state=[0.0] * 7,
                action=[0.0] * 7,
            )
            for index in range(episode.length)
        ]


def test_analyze_metadata_completeness_flags_task_description_missing() -> None:
    reader = MetadataReader(
        episodes=[
            MetadataReader().episode(0).model_copy(update={"tasks": [], "episode_index": 0})
        ]
    )

    result = analyze_metadata_completeness(reader)

    codes = [finding.code for finding in result.episode_findings[0]]
    assert "task_description_missing" in codes


def test_analyze_metadata_completeness_flags_camera_naming() -> None:
    reader = MetadataReader(
        video_keys=["top", "wrist"],
        episodes=[
            MetadataReader().episode(0).model_copy(
                update={
                    "video_files": {"top": "videos/top.mp4", "wrist": "videos/wrist.mp4"},
                }
            )
        ],
    )

    result = analyze_metadata_completeness(reader)

    assert any(finding.code == "camera_naming" for finding in result.dataset_findings)


def test_analyze_metadata_completeness_flags_resolution_issues() -> None:
    reader = MetadataReader(
        features={
            "observation.state": {"shape": [7]},
            "action": {"shape": [7]},
            "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [240, 320, 3]},
        }
    )

    result = analyze_metadata_completeness(reader)

    codes = [finding.code for finding in result.dataset_findings]
    assert "resolution_inconsistent" in codes


def test_analyze_metadata_completeness_flags_single_view() -> None:
    reader = MetadataReader(
        video_keys=["observation.images.top"],
        episodes=[
            MetadataReader().episode(0).model_copy(
                update={
                    "video_files": {"observation.images.top": "videos/top.mp4"},
                    "video_start_seconds": {"observation.images.top": 0.0},
                    "video_end_seconds": {"observation.images.top": 0.2},
                }
            )
        ],
    )

    result = analyze_metadata_completeness(reader)

    assert any(finding.code == "single_view" for finding in result.dataset_findings)


def test_analyze_metadata_completeness_flags_action_metadata_issues() -> None:
    reader = MetadataReader(
        features={
            "observation.state": {"shape": [7]},
            "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "video", "shape": [480, 640, 3]},
        }
    )

    result = analyze_metadata_completeness(reader)

    codes = [finding.code for finding in result.dataset_findings]
    assert "action_missing" in codes


def test_analyze_metadata_completeness_flags_missing_video_file(tmp_path: Path) -> None:
    reader = MetadataReader(root=tmp_path)

    result = analyze_metadata_completeness(reader)

    assert any(finding.code == "video_file_missing" for finding in result.episode_findings[0])


def test_metadata_completeness_stage_has_no_score_and_does_not_affect_cleaning_score() -> None:
    reader = MetadataReader(
        video_keys=["top", "wrist"],
        episodes=[
            MetadataReader().episode(0).model_copy(
                update={
                    "tasks": [],
                    "video_files": {"top": "videos/top.mp4", "wrist": "videos/wrist.mp4"},
                    "video_start_seconds": {"top": 0.0, "wrist": 0.0},
                    "video_end_seconds": {"top": 0.2, "wrist": 0.2},
                }
            )
        ],
    )
    filter_summary = DatasetFilterService(reader, FilterConfig()).summary()
    stage = filter_summary.episodes[0].stage_status["metadata_completeness"]

    assert stage.score is None
    assert stage.count > 0
    assert stage.status == "review"
    assert stage.severity == "warning"

    before = EpisodeQualityScorer().score_dataset(
        reader,
        CleaningConfig(enabled_filter_stages=["sudden_change"]),
        filter_summary=filter_summary,
    )[0]
    after = EpisodeQualityScorer().score_dataset(
        reader,
        CleaningConfig(
            enabled_filter_stages=["sudden_change", "metadata_completeness"],
        ),
        filter_summary=filter_summary,
    )[0]

    assert "metadata_completeness" not in after.per_attribute_scores
    assert after.data_quality_score == before.data_quality_score
    assert any(finding.code == "metadata_completeness" for finding in after.findings)


def test_metadata_inspection_rows_always_present_on_pass(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "top.mp4").write_bytes(b"")
    (videos / "wrist.mp4").write_bytes(b"")
    reader = MetadataReader(root=tmp_path)
    rows = build_metadata_inspection(reader, 0)

    assert len(rows) == 6
    assert all(row["kind"] == "inspection" for row in rows)
    assert all(row.get("value") for row in rows)
    assert all(row["status"] == "passed" for row in rows)


def test_metadata_completeness_is_enabled_by_default() -> None:
    assert "metadata_completeness" in CleaningConfig().enabled_filter_stages


def test_metadata_completeness_detail_lists_findings() -> None:
    reader = MetadataReader(
        video_keys=["top"],
        episodes=[
            MetadataReader().episode(0).model_copy(
                update={
                    "tasks": [],
                    "video_files": {"top": "videos/top.mp4"},
                    "video_start_seconds": {"top": 0.0},
                    "video_end_seconds": {"top": 0.2},
                }
            )
        ],
    )
    detail = DatasetFilterService(reader, FilterConfig()).detail("metadata_completeness", 0)

    assert detail.stage_id == "metadata_completeness"
    assert detail.status == "review"
    assert detail.findings
    assert len(detail.table_rows) == 6
    assert all(row["kind"] == "inspection" for row in detail.table_rows)
    assert detail.parameters["summary"]["warnings"] >= 1


def test_metadata_completeness_detail_includes_summary_on_pass(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "top.mp4").write_bytes(b"")
    (videos / "wrist.mp4").write_bytes(b"")
    reader = MetadataReader(root=tmp_path)
    detail = DatasetFilterService(reader, FilterConfig()).detail("metadata_completeness", 0)

    assert detail.status == "passed"
    assert detail.findings == []
    assert len(detail.table_rows) == 6
    assert detail.parameters["summary"] == {
        "total_checks": 6,
        "passed": 6,
        "warnings": 0,
        "infos": 0,
    }


def test_export_conversion_report_includes_metadata_completeness(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    registry = FormatRegistry.default()
    adapter = registry.open_dataset(source)

    export = registry.export_dataset(
        adapter=adapter,
        target_format="act_hdf5",
        episode_indexes=[0],
        output_root=tmp_path / "exports",
    )

    report = json.loads(export.report_path.read_text(encoding="utf-8"))
    assert "metadata_completeness" in report
    assert "dataset_findings" in report["metadata_completeness"]
    assert "episodes" in report["metadata_completeness"]
    assert "inspection" in report["metadata_completeness"]
    assert report["metadata_completeness"]["inspection"]["0"]
