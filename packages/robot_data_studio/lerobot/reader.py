from __future__ import annotations

import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .models import DatasetMetadata, DatasetProbe, EpisodeFrame, EpisodeSubtask, EpisodeSummary


class NotLeRobotDataset(ValueError):
    pass


class LeRobotDatasetReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.probe(self.root)
        self._meta_root = self._metadata_root(self.root)
        self._info = json.loads((self._meta_root / "info.json").read_text())
        self._version = str(self._info.get("codebase_version", ""))

    @staticmethod
    def _metadata_root(root: Path) -> Path:
        nested = root / "meta"
        if (nested / "info.json").is_file():
            return nested
        if (root / "info.json").is_file():
            return root
        return nested

    @classmethod
    def probe(cls, root: str | Path) -> DatasetProbe:
        root_path = Path(root).expanduser().resolve()
        info_path = cls._metadata_root(root_path) / "info.json"
        if not info_path.is_file():
            raise NotLeRobotDataset(f"Missing LeRobot metadata: {info_path}")
        try:
            info = json.loads(info_path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise NotLeRobotDataset(f"Invalid LeRobot metadata: {info_path}") from error
        version = str(info.get("codebase_version", ""))
        if not version.startswith(("v2", "v3")):
            raise NotLeRobotDataset(f"Only LeRobot v2/v3 is supported, got {version!r}")
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
        if self._version.startswith("v2"):
            return self._list_v2_episodes(limit)
        return self._list_v3_episodes(limit)

    def _list_v3_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        files = sorted((self._meta_root / "episodes").glob("**/*.parquet"))
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
                    subtasks=[],
                    data_file=data_file,
                    video_files=video_files,
                    video_start_seconds=video_starts,
                    video_end_seconds=video_ends,
                )
            )
        return summaries

    def _list_v2_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        episodes_path = self._meta_root / "episodes.jsonl"
        if not episodes_path.is_file():
            raise NotLeRobotDataset(f"Missing LeRobot v2 episodes metadata: {episodes_path}")
        rows = _read_jsonl(episodes_path)
        if limit is not None:
            rows = rows[:limit]
        fps = float(self._info["fps"])
        annotations = self._v2_subtasks_by_episode()
        video_keys = self.metadata().video_keys
        summaries = []
        for row in rows:
            episode_index = int(row["episode_index"])
            length = int(row["length"])
            episode_chunk = int(row.get("episode_chunk", episode_index // int(self._info.get("chunks_size", 1000))))
            video_files = {
                key: self._format_template(
                    self._info["video_path"],
                    episode_index=episode_index,
                    episode_chunk=episode_chunk,
                    video_key=key,
                )
                for key in video_keys
            }
            summaries.append(
                EpisodeSummary(
                    episode_index=episode_index,
                    length=length,
                    duration_seconds=length / fps,
                    tasks=row.get("tasks") or [],
                    subtasks=annotations.get(episode_index, []),
                    data_file=self._format_template(
                        self._info["data_path"],
                        episode_index=episode_index,
                        episode_chunk=episode_chunk,
                    ),
                    video_files=video_files,
                    video_start_seconds={key: 0.0 for key in video_keys},
                    video_end_seconds={key: length / fps for key in video_keys},
                )
            )
        return summaries

    def _v2_subtasks_by_episode(self) -> dict[int, list[EpisodeSubtask]]:
        annotations_path = self._meta_root / "annotations.json"
        if not annotations_path.is_file():
            return {}
        try:
            payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        fps = float(self._info["fps"])
        by_episode: dict[int, list[EpisodeSubtask]] = {}
        for annotation in payload.values():
            if not isinstance(annotation, dict) or "episode_index" not in annotation:
                continue
            episode_index = int(annotation["episode_index"])
            subtasks = []
            for step in annotation.get("action_steps") or []:
                if not isinstance(step, dict):
                    continue
                start_frame = int(step.get("start_frame", 0))
                end_frame = int(step.get("end_frame", start_frame))
                prompt = str(step.get("action_text") or "").strip()
                if not prompt:
                    continue
                subtasks.append(
                    EpisodeSubtask(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_seconds=start_frame / fps,
                        end_seconds=end_frame / fps,
                        prompt=prompt,
                        skill=step.get("skill"),
                        track=step.get("track"),
                        is_mistake=bool(step.get("is_mistake", False)),
                    )
                )
            if subtasks:
                by_episode[episode_index] = subtasks
        return by_episode

    @staticmethod
    def _format_template(template: str, **values: object) -> str:
        return template.format(
            chunk_index=values.get("episode_chunk"),
            file_index=values.get("episode_index"),
            **values,
        )

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


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
