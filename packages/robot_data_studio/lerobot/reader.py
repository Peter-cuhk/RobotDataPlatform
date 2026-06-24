from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .models import DatasetMetadata, DatasetProbe, EpisodeFrame, EpisodeSummary


class NotLeRobotDataset(ValueError):
    pass


class LeRobotDatasetReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.probe(self.root)
        self._info = json.loads((self.root / "meta" / "info.json").read_text())

    @classmethod
    def probe(cls, root: str | Path) -> DatasetProbe:
        root_path = Path(root).expanduser().resolve()
        info_path = root_path / "meta" / "info.json"
        if not info_path.is_file():
            raise NotLeRobotDataset(f"Missing LeRobot metadata: {info_path}")
        try:
            info = json.loads(info_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise NotLeRobotDataset(f"Invalid LeRobot metadata: {info_path}") from error
        version = str(info.get("codebase_version", ""))
        if not version.startswith("v3"):
            raise NotLeRobotDataset(f"Only LeRobot v3 is supported, got {version!r}")
        return DatasetProbe(format="lerobot", version=version, confidence=1.0)

    def metadata(self) -> DatasetMetadata:
        features = self._info.get("features", {})
        video_keys = [key for key, value in features.items() if value.get("dtype") == "video"]
        preferred_scalars = ["observation.state", "action"]
        scalar_keys = [key for key in preferred_scalars if key in features]
        return DatasetMetadata(
            path=str(self.root),
            format="lerobot",
            version=self._info["codebase_version"],
            robot_type=self._info.get("robot_type", "unknown"),
            total_episodes=self._info["total_episodes"],
            total_frames=self._info["total_frames"],
            fps=float(self._info["fps"]),
            video_keys=video_keys,
            scalar_keys=scalar_keys,
            features=features,
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        files = sorted((self.root / "meta" / "episodes").glob("**/*.parquet"))
        table = ds.dataset(files, format="parquet").to_table()
        if limit is not None:
            table = table.slice(0, limit)
        summaries = []
        for row in table.to_pylist():
            video_files: dict[str, str] = {}
            video_starts: dict[str, float] = {}
            video_ends: dict[str, float] = {}
            for key in self.metadata().video_keys:
                chunk = row[f"videos/{key}/chunk_index"]
                file_index = row[f"videos/{key}/file_index"]
                video_files[key] = self._info["video_path"].format(
                    video_key=key,
                    chunk_index=chunk,
                    file_index=file_index,
                )
                video_starts[key] = row[f"videos/{key}/from_timestamp"]
                video_ends[key] = row[f"videos/{key}/to_timestamp"]
            data_file = self._info["data_path"].format(
                chunk_index=row["data/chunk_index"],
                file_index=row["data/file_index"],
            )
            summaries.append(
                EpisodeSummary(
                    episode_index=row["episode_index"],
                    length=row["length"],
                    duration_seconds=row["length"] / float(self._info["fps"]),
                    tasks=row.get("tasks") or [],
                    data_file=data_file,
                    video_files=video_files,
                    video_start_seconds=video_starts,
                    video_end_seconds=video_ends,
                )
            )
        return summaries

    def episode(self, episode_index: int) -> EpisodeSummary:
        episodes = self.list_episodes()
        try:
            return next(item for item in episodes if item.episode_index == episode_index)
        except StopIteration as error:
            raise KeyError(f"Episode {episode_index} not found") from error

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        episode = self.episode(episode_index)
        parquet_path = self.root / episode.data_file
        table = pq.read_table(parquet_path)
        table = table.filter(pc.equal(table["episode_index"], episode_index))
        return [
            EpisodeFrame(
                frame_index=row["frame_index"],
                timestamp=row["timestamp"],
                observation_state=row.get("observation.state") or [],
                action=row.get("action") or [],
            )
            for row in table.to_pylist()
        ]

    def video_path(self, episode_index: int, video_key: str) -> Path:
        episode = self.episode(episode_index)
        if video_key not in episode.video_files:
            raise KeyError(f"Video key {video_key!r} not found")
        return self.root / episode.video_files[video_key]

