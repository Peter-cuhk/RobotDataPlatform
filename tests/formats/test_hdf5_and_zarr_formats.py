from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from robot_data_studio.formats.registry import FormatRegistry


def test_act_hdf5_import_and_export_round_trip(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    registry = FormatRegistry.default()
    adapter = registry.open_dataset(source)

    assert adapter.metadata().format == "act_hdf5"
    assert adapter.list_episodes()[0].length == 3
    assert adapter.read_episode_frames(0)[1].action == [1.0, 1.5]

    export = registry.export_dataset(
        adapter=adapter,
        target_format="act_hdf5",
        episode_indexes=[0],
        output_root=tmp_path / "exports",
    )

    assert export.format == "act_hdf5"
    assert export.episode_count == 1
    assert export.output_path.exists()
    assert export.report_path.exists()
    round_trip = registry.open_dataset(export.output_path, format_hint="act_hdf5")
    assert round_trip.read_episode_frames(0)[2].observation_state == [2.0, 2.5]


def test_robomimic_hdf5_import_and_export_round_trip(tmp_path: Path) -> None:
    source = write_robomimic_hdf5(tmp_path / "robomimic.hdf5")
    registry = FormatRegistry.default()
    adapter = registry.open_dataset(source)

    assert adapter.metadata().format == "robomimic_hdf5"
    assert [episode.episode_index for episode in adapter.list_episodes()] == [0, 1]
    assert adapter.read_episode_frames(1)[0].action == [10.0, 10.5]

    export = registry.export_dataset(
        adapter=adapter,
        target_format="robomimic_hdf5",
        episode_indexes=[0, 1],
        output_root=tmp_path / "exports",
    )

    round_trip = registry.open_dataset(export.output_path, format_hint="robomimic_hdf5")
    assert round_trip.list_episodes()[1].length == 2
    assert round_trip.read_episode_frames(0)[1].observation_state == [1.0, 1.5]


def test_umi_zarr_import_and_export_round_trip(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    source = tmp_path / "umi_dataset.zarr"
    root = zarr.open_group(source, mode="w")
    data = root.create_group("data")
    data.create_dataset("robot0_eef_pos", data=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="f4"))
    data.create_dataset("robot0_eef_rot_axis_angle", data=np.asarray([[0, 0, 0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype="f4"))
    data.create_dataset("robot0_gripper_width", data=np.asarray([[0.01], [0.02], [0.03]], dtype="f4"))
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.asarray([2, 3], dtype="i8"))

    registry = FormatRegistry.default()
    adapter = registry.open_dataset(source)

    assert adapter.metadata().format == "umi_zarr"
    assert [episode.length for episode in adapter.list_episodes()] == [2, 1]
    assert adapter.read_episode_frames(0)[1].observation_state == [3.0, 4.0, 5.0, 0.1, 0.2, 0.3, 0.02]

    export = registry.export_dataset(
        adapter=adapter,
        target_format="umi_zarr",
        episode_indexes=[0, 1],
        output_root=tmp_path / "exports",
    )

    round_trip = registry.open_dataset(export.output_path, format_hint="umi_zarr")
    assert round_trip.list_episodes()[0].length == 2
    assert round_trip.read_episode_frames(1)[0].observation_state == [6.0, 7.0, 8.0, 0.4, 0.5, 0.6, 0.03]


def test_api_imports_with_hint_and_exports_report(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    client = TestClient(create_app(artifact_root=tmp_path / "artifacts"))

    project_response = client.post(
        "/api/projects",
        json={"path": str(source), "format_hint": "act_hdf5"},
    )
    assert project_response.status_code == 201
    project = project_response.json()
    assert project["dataset"]["format"] == "act_hdf5"

    export_response = client.post(
        f"/api/projects/{project['id']}/exports",
        json={"episode_indexes": [0], "format": "robomimic_hdf5", "options": {}},
    )

    assert export_response.status_code == 201
    body = export_response.json()
    assert body["format"] == "robomimic_hdf5"
    assert Path(body["output_path"]).exists()
    assert Path(body["report_path"]).exists()
    assert body["episode_count"] == 1


def write_act_hdf5(path: Path) -> Path:
    with h5py.File(path, "w") as file:
        file.create_dataset("action", data=np.asarray([[0, 0.5], [1, 1.5], [2, 2.5]], dtype="f4"))
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=np.asarray([[0, 0.5], [1, 1.5], [2, 2.5]], dtype="f4"))
        observations.create_dataset("timestamp", data=np.asarray([0, 0.1, 0.2], dtype="f4"))
    return path


def write_robomimic_hdf5(path: Path) -> Path:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for index, offset in enumerate([0, 10]):
            demo = data.create_group(f"demo_{index}")
            actions = np.asarray([[offset, offset + 0.5], [offset + 1, offset + 1.5]], dtype="f4")
            demo.create_dataset("actions", data=actions)
            demo.attrs["num_samples"] = len(actions)
            obs = demo.create_group("obs")
            obs.create_dataset("qpos", data=actions)
    return path
