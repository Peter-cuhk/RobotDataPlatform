from __future__ import annotations

import json
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from robot_data_studio.formats.registry import FormatRegistry


DEFAULT_FORGE_ROOT = Path(__file__).resolve().parents[2] / "forge"
DEFAULT_FORGE_PYTHON = DEFAULT_FORGE_ROOT / ".venv/bin/python"


@dataclass(frozen=True)
class EpisodeSnapshot:
    episode_index: int
    length: int
    timestamps: list[float]
    states: list[list[float]]
    actions: list[list[float]]


@dataclass(frozen=True)
class DatasetSnapshot:
    format: str
    episode_count: int
    episodes: list[EpisodeSnapshot]


def rds_snapshot(path: Path, format_hint: str) -> DatasetSnapshot:
    adapter = FormatRegistry.default().open_dataset(path, format_hint=format_hint)
    episodes = []
    for episode in adapter.list_episodes():
        frames = adapter.read_episode_frames(episode.episode_index)
        episodes.append(
            EpisodeSnapshot(
                episode_index=episode.episode_index,
                length=len(frames),
                timestamps=[frame.timestamp for frame in frames],
                states=[frame.observation_state for frame in frames],
                actions=[frame.action for frame in frames],
            )
        )
    return DatasetSnapshot(
        format=format_hint,
        episode_count=len(episodes),
        episodes=episodes,
    )


def forge_snapshot(path: Path, rds_format: str) -> DatasetSnapshot:
    forge_python = Path(os.environ.get("FORGE_PYTHON", DEFAULT_FORGE_PYTHON))
    if not forge_python.is_file():
        pytest.skip(f"Forge Python not found: {forge_python}")

    script = _forge_snapshot_script()
    try:
        completed = subprocess.run(
            [
                str(forge_python),
                "-c",
                script,
                str(Path(path).resolve()),
                _forge_format_for(rds_format),
                rds_format,
            ],
            cwd=DEFAULT_FORGE_ROOT if DEFAULT_FORGE_ROOT.is_dir() else None,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        pytest.skip(f"Forge Python could not be executed: {error}")

    if completed.returncode != 0:
        pytest.fail(
            "Forge snapshot failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return _snapshot_from_json(json.loads(completed.stdout))


def assert_snapshots_match(
    expected: DatasetSnapshot,
    actual: DatasetSnapshot,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    assert actual.episode_count == expected.episode_count
    assert len(actual.episodes) == len(expected.episodes)
    for expected_episode, actual_episode in zip(
        expected.episodes,
        actual.episodes,
        strict=True,
    ):
        assert actual_episode.episode_index == expected_episode.episode_index
        assert actual_episode.length == expected_episode.length
        np.testing.assert_allclose(
            actual_episode.timestamps,
            expected_episode.timestamps,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            actual_episode.states,
            expected_episode.states,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            actual_episode.actions,
            expected_episode.actions,
            rtol=rtol,
            atol=atol,
        )


def _forge_format_for(rds_format: str) -> str:
    mapping = {
        "act_hdf5": "hdf5",
        "robomimic_hdf5": "hdf5",
        "umi_zarr": "zarr",
        "lerobot_v3": "lerobot-v3",
    }
    return mapping[rds_format]


def _snapshot_from_json(data: dict) -> DatasetSnapshot:
    return DatasetSnapshot(
        format=data["format"],
        episode_count=data["episode_count"],
        episodes=[
            EpisodeSnapshot(
                episode_index=episode["episode_index"],
                length=episode["length"],
                timestamps=episode["timestamps"],
                states=episode["states"],
                actions=episode["actions"],
            )
            for episode in data["episodes"]
        ],
    )


def _forge_snapshot_script() -> str:
    return textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path

        import numpy as np

        import forge.formats  # noqa: F401 - registers readers
        from forge.formats.registry import FormatRegistry

        def _as_float_list(value):
            if value is None:
                return []
            return [round(float(item), 6) for item in np.asarray(value, dtype=np.float64).reshape(-1)]


        def _fallback_timestamp(rds_format, frame_index):
            if rds_format == "umi_zarr":
                return frame_index / 60.0
            return frame_index / 30.0


        def _source_timestamp(path, rds_format, episode_index, frame_index):
            if rds_format != "act_hdf5":
                return None
            try:
                import h5py

                files = [path] if path.is_file() else sorted(path.glob("**/*.hdf5")) + sorted(path.glob("**/*.h5"))
                file_path = files[episode_index]
                with h5py.File(file_path, "r") as file:
                    if "observations/timestamp" in file:
                        return float(file["observations/timestamp"][frame_index])
            except Exception:
                return None
            return None


        path = Path(sys.argv[1])
        forge_format = sys.argv[2]
        rds_format = sys.argv[3]

        reader = FormatRegistry.get_reader(forge_format)
        episodes = []
        for episode_index, episode in enumerate(reader.read_episodes(path)):
            frames = episode.load_frames()
            timestamps = []
            states = []
            actions = []
            for frame_index, frame in enumerate(frames):
                timestamp = frame.timestamp
                if timestamp is None:
                    timestamp = _source_timestamp(path, rds_format, episode_index, frame_index)
                if timestamp is None:
                    timestamp = _fallback_timestamp(rds_format, frame_index)
                timestamps.append(round(float(timestamp), 6))
                states.append(_as_float_list(frame.state))
                actions.append(_as_float_list(frame.action))
            episodes.append(
                {
                    "episode_index": episode_index,
                    "length": len(frames),
                    "timestamps": timestamps,
                    "states": states,
                    "actions": actions,
                }
            )

        print(
            json.dumps(
                {
                    "format": forge_format,
                    "episode_count": len(episodes),
                    "episodes": episodes,
                }
            )
        )
        """
    )
