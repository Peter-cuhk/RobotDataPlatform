from __future__ import annotations

from pathlib import Path

import numpy as np

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary


class NotUMIZarrDataset(ValueError):
    pass


def _zarr_module():
    try:
        import zarr
    except ImportError as error:
        raise NotUMIZarrDataset("zarr is required for umi_zarr datasets") from error
    return zarr


def _as_float_list(value: object) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), 6) for item in array]


class UMIZarrDatasetAdapter:
    format_id = "umi_zarr"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.probe(self.root):
            raise NotUMIZarrDataset(f"Not a UMI Zarr dataset: {self.root}")

    @classmethod
    def probe(cls, root: str | Path) -> bool:
        path = Path(root).expanduser().resolve()
        if not (path.exists() and (path.suffix == ".zarr" or path.name.endswith(".zarr.zip"))):
            return False
        try:
            zarr = _zarr_module()
            group = zarr.open_group(path, mode="r")
            return "data" in group and "meta" in group and "episode_ends" in group["meta"]
        except Exception:
            return False

    def _root_group(self):
        return _zarr_module().open_group(self.root, mode="r")

    def _episode_bounds(self) -> list[tuple[int, int]]:
        ends = [int(value) for value in np.asarray(self._root_group()["meta"]["episode_ends"])]
        starts = [0, *ends[:-1]]
        return list(zip(starts, ends, strict=True))

    def metadata(self) -> DatasetMetadata:
        bounds = self._episode_bounds()
        total_frames = bounds[-1][1] if bounds else 0
        return DatasetMetadata(
            path=str(self.root),
            format=self.format_id,
            version="umi_zarr",
            robot_type="unknown",
            total_episodes=len(bounds),
            total_frames=total_frames,
            fps=60.0,
            video_keys=[],
            scalar_keys=["observation.state", "action"],
            features={
                "robot0_eef_pos": {"dtype": "float32"},
                "robot0_eef_rot_axis_angle": {"dtype": "float32"},
                "robot0_gripper_width": {"dtype": "float32"},
            },
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        summaries = [
            EpisodeSummary(
                episode_index=index,
                length=end - start,
                duration_seconds=(end - start) / 60.0,
                tasks=[],
                data_file=str(self.root),
                video_files={},
                video_start_seconds={},
                video_end_seconds={},
            )
            for index, (start, end) in enumerate(self._episode_bounds())
        ]
        return summaries[:limit] if limit is not None else summaries

    def episode(self, episode_index: int) -> EpisodeSummary:
        for episode in self.list_episodes():
            if episode.episode_index == episode_index:
                return episode
        raise KeyError(f"Episode {episode_index} not found")

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        self.episode(episode_index)
        start, end = self._episode_bounds()[episode_index]
        data = self._root_group()["data"]
        states = self._state_array(data)[start:end]
        actions = self._action_array(data, states, start, end)
        return [
            EpisodeFrame(
                frame_index=index,
                timestamp=round(index / 60.0, 6),
                observation_state=_as_float_list(states[index]),
                action=_as_float_list(actions[index]),
            )
            for index in range(end - start)
        ]

    def _state_array(self, data) -> np.ndarray:
        parts = []
        for key in ["robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width"]:
            if key in data:
                parts.append(np.asarray(data[key]))
        if not parts:
            raise NotUMIZarrDataset("UMI Zarr profile requires at least one robot0_* state array")
        return np.concatenate([part.reshape((part.shape[0], -1)) for part in parts], axis=1)

    def _action_array(self, data, states: np.ndarray, start: int, end: int) -> np.ndarray:
        if "action" in data:
            return np.asarray(data["action"])[start:end]
        if len(states) == 1:
            return states.copy()
        shifted = np.vstack([states[1:], states[-1:]])
        return shifted


def write_umi_zarr(adapter: object, episode_indexes: list[int], output: Path) -> Path:
    zarr = _zarr_module()
    output = output.with_suffix(".zarr")
    root = zarr.open_group(output, mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    states = []
    actions = []
    episode_ends = []
    total = 0
    for episode_index in episode_indexes:
        frames = adapter.read_episode_frames(episode_index)
        states.extend(frame.observation_state for frame in frames)
        actions.extend(frame.action for frame in frames)
        total += len(frames)
        episode_ends.append(total)
    data.create_dataset("robot0_eef_pos", data=np.asarray(states, dtype=np.float32))
    data.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
    meta.create_dataset("episode_ends", data=np.asarray(episode_ends, dtype=np.int64))
    return output
