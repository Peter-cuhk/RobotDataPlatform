from __future__ import annotations

from pathlib import Path

import numpy as np


def write_lerobot_dataset(adapter: object, episode_indexes: list[int], output_root: Path, target_format: str) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first_frames = adapter.read_episode_frames(episode_indexes[0])
    state_dim = len(first_frames[0].observation_state) if first_frames else 0
    action_dim = len(first_frames[0].action) if first_frames else 0
    dataset = LeRobotDataset.create(
        repo_id=f"robot-data-studio/{output_root.name}",
        root=output_root,
        fps=int(adapter.metadata().fps or 30),
        robot_type=adapter.metadata().robot_type,
        features={
            "observation.state": {"dtype": "float32", "shape": (state_dim,)},
            "action": {"dtype": "float32", "shape": (action_dim,)},
        },
        use_videos=False,
    )
    for episode_index in episode_indexes:
        for frame in adapter.read_episode_frames(episode_index):
            dataset.add_frame(
                {
                    "observation.state": np.asarray(frame.observation_state, dtype=np.float32),
                    "action": np.asarray(frame.action, dtype=np.float32),
                    "timestamp": frame.timestamp,
                },
                task=adapter.episode(episode_index).tasks[0] if adapter.episode(episode_index).tasks else "default",
            )
        dataset.save_episode()
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    return output_root
