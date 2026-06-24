from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary


class NotHDF5Profile(ValueError):
    pass


def _as_float_list(value: object) -> list[float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return [round(float(item), 6) for item in array]


def _episode_file_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob("**/*.hdf5")) + sorted(root.glob("**/*.h5"))


class ActHDF5DatasetAdapter:
    format_id = "act_hdf5"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._files = _episode_file_paths(self.root)
        if not self._files:
            raise NotHDF5Profile(f"No HDF5 files found under {self.root}")
        if not self.probe(self.root):
            raise NotHDF5Profile(f"Not an ACT HDF5 dataset: {self.root}")

    @classmethod
    def probe(cls, root: str | Path) -> bool:
        try:
            files = _episode_file_paths(Path(root).expanduser().resolve())
            if not files:
                return False
            with h5py.File(files[0], "r") as file:
                return "action" in file and "observations/qpos" in file
        except OSError:
            return False

    def metadata(self) -> DatasetMetadata:
        episodes = self.list_episodes()
        total_frames = sum(episode.length for episode in episodes)
        return DatasetMetadata(
            path=str(self.root),
            format=self.format_id,
            version="act_hdf5",
            robot_type="unknown",
            total_episodes=len(episodes),
            total_frames=total_frames,
            fps=30.0,
            video_keys=[],
            scalar_keys=["observation.state", "action"],
            features={
                "observation.state": {"dtype": "float32"},
                "action": {"dtype": "float32"},
            },
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        summaries = []
        for index, path in enumerate(self._files):
            with h5py.File(path, "r") as file:
                length = int(file["action"].shape[0])
            summaries.append(
                EpisodeSummary(
                    episode_index=index,
                    length=length,
                    duration_seconds=length / 30.0,
                    tasks=[],
                    data_file=str(path),
                    video_files={},
                    video_start_seconds={},
                    video_end_seconds={},
                )
            )
        return summaries[:limit] if limit is not None else summaries

    def episode(self, episode_index: int) -> EpisodeSummary:
        for episode in self.list_episodes():
            if episode.episode_index == episode_index:
                return episode
        raise KeyError(f"Episode {episode_index} not found")

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        episode = self.episode(episode_index)
        with h5py.File(episode.data_file, "r") as file:
            actions = np.asarray(file["action"])
            states = np.asarray(file["observations/qpos"])
            if "observations/timestamp" in file:
                timestamps = np.asarray(file["observations/timestamp"], dtype=np.float64)
            else:
                timestamps = np.arange(actions.shape[0], dtype=np.float64) / 30.0
        return [
            EpisodeFrame(
                frame_index=index,
                timestamp=round(float(timestamps[index]), 6),
                observation_state=_as_float_list(states[index]),
                action=_as_float_list(actions[index]),
            )
            for index in range(actions.shape[0])
        ]


class RobomimicHDF5DatasetAdapter:
    format_id = "robomimic_hdf5"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.probe(self.root):
            raise NotHDF5Profile(f"Not a robomimic HDF5 dataset: {self.root}")

    @classmethod
    def probe(cls, root: str | Path) -> bool:
        path = Path(root).expanduser().resolve()
        if not path.is_file():
            return False
        try:
            with h5py.File(path, "r") as file:
                return "data" in file and any(key.startswith("demo_") for key in file["data"].keys())
        except OSError:
            return False

    def _demo_keys(self) -> list[str]:
        with h5py.File(self.root, "r") as file:
            return sorted(file["data"].keys(), key=lambda key: int(key.split("_")[-1]))

    def metadata(self) -> DatasetMetadata:
        episodes = self.list_episodes()
        return DatasetMetadata(
            path=str(self.root),
            format=self.format_id,
            version="robomimic_hdf5",
            robot_type="unknown",
            total_episodes=len(episodes),
            total_frames=sum(episode.length for episode in episodes),
            fps=30.0,
            video_keys=[],
            scalar_keys=["observation.state", "action"],
            features={"observation.state": {"dtype": "float32"}, "action": {"dtype": "float32"}},
        )

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        summaries = []
        with h5py.File(self.root, "r") as file:
            for index, key in enumerate(self._demo_keys()):
                demo = file[f"data/{key}"]
                length = int(demo.attrs.get("num_samples", demo["actions"].shape[0]))
                summaries.append(
                    EpisodeSummary(
                        episode_index=index,
                        length=length,
                        duration_seconds=length / 30.0,
                        tasks=[],
                        data_file=f"data/{key}",
                        video_files={},
                        video_start_seconds={},
                        video_end_seconds={},
                    )
                )
        return summaries[:limit] if limit is not None else summaries

    def episode(self, episode_index: int) -> EpisodeSummary:
        for episode in self.list_episodes():
            if episode.episode_index == episode_index:
                return episode
        raise KeyError(f"Episode {episode_index} not found")

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        episode = self.episode(episode_index)
        with h5py.File(self.root, "r") as file:
            demo = file[episode.data_file]
            actions = np.asarray(demo["actions"])
            if "obs/qpos" in demo:
                states = np.asarray(demo["obs/qpos"])
            elif "states" in demo:
                states = np.asarray(demo["states"])
            else:
                states = actions
        return [
            EpisodeFrame(
                frame_index=index,
                timestamp=round(index / 30.0, 6),
                observation_state=_as_float_list(states[index]),
                action=_as_float_list(actions[index]),
            )
            for index in range(actions.shape[0])
        ]


def write_act_hdf5(adapter: object, episode_indexes: list[int], output_root: Path) -> Path:
    if len(episode_indexes) == 1:
        output = output_root.with_suffix(".hdf5")
        _write_act_file(adapter, episode_indexes[0], output)
        return output
    output_root.mkdir(parents=True, exist_ok=True)
    for episode_index in episode_indexes:
        _write_act_file(adapter, episode_index, output_root / f"episode_{episode_index:06d}.hdf5")
    return output_root


def _write_act_file(adapter: object, episode_index: int, output: Path) -> None:
    frames = adapter.read_episode_frames(episode_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as file:
        file.attrs["sim"] = True
        file.attrs["source_format"] = adapter.metadata().format
        file.attrs["episode_index"] = episode_index
        file.create_dataset("action", data=np.asarray([frame.action for frame in frames], dtype=np.float32))
        observations = file.create_group("observations")
        observations.create_dataset(
            "qpos",
            data=np.asarray([frame.observation_state for frame in frames], dtype=np.float32),
        )
        observations.create_dataset(
            "timestamp",
            data=np.asarray([frame.timestamp for frame in frames], dtype=np.float32),
        )


def write_robomimic_hdf5(adapter: object, episode_indexes: list[int], output: Path) -> Path:
    output = output.with_suffix(".hdf5")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as file:
        data = file.create_group("data")
        data.attrs["total"] = 0
        for demo_index, episode_index in enumerate(episode_indexes):
            frames = adapter.read_episode_frames(episode_index)
            demo = data.create_group(f"demo_{demo_index}")
            actions = np.asarray([frame.action for frame in frames], dtype=np.float32)
            states = np.asarray([frame.observation_state for frame in frames], dtype=np.float32)
            demo.create_dataset("actions", data=actions)
            demo.create_dataset("states", data=states)
            obs = demo.create_group("obs")
            obs.create_dataset("qpos", data=states)
            demo.attrs["num_samples"] = len(frames)
            demo.attrs["source_episode_index"] = episode_index
            data.attrs["total"] = int(data.attrs["total"]) + len(frames)
    return output
