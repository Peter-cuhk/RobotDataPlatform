from pathlib import Path

import h5py
import numpy as np

from robot_data_studio.lerobot.reader import LeRobotDatasetReader


def export_act_hdf5(
    reader: LeRobotDatasetReader,
    episode_index: int,
    output_path: Path,
) -> Path:
    frames = reader.read_episode_frames(episode_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    actions = np.asarray([frame.action for frame in frames], dtype=np.float32)
    states = np.asarray([frame.observation_state for frame in frames], dtype=np.float32)
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float32)
    with h5py.File(output_path, "w") as file:
        file.attrs["sim"] = True
        file.attrs["source_format"] = "lerobot_v3"
        file.attrs["episode_index"] = episode_index
        file.create_dataset("action", data=actions)
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=states)
        observations.create_dataset("timestamp", data=timestamps)
    return output_path

