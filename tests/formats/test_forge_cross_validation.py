from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from robot_data_studio.formats.registry import FormatRegistry
from tests.formats.forge_cross_validation import (
    assert_snapshots_match,
    forge_snapshot,
    rds_snapshot,
)


def test_act_hdf5_export_matches_forge_snapshot(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    export = export_with_rds(source, "act_hdf5", "act_hdf5", [0], tmp_path / "exports")

    rds = rds_snapshot(export.output_path, "act_hdf5")
    forge = forge_snapshot(export.output_path, "act_hdf5")

    assert_snapshots_match(rds, forge)
    assert_report(export.report_path, "act_hdf5", "act_hdf5", [0])


def test_act_hdf5_to_robomimic_export_matches_forge_snapshot(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    export = export_with_rds(source, "act_hdf5", "robomimic_hdf5", [0], tmp_path / "exports")

    rds = rds_snapshot(export.output_path, "robomimic_hdf5")
    forge = forge_snapshot(export.output_path, "robomimic_hdf5")

    assert_snapshots_match(rds, forge)
    assert_report(export.report_path, "act_hdf5", "robomimic_hdf5", [0])


def test_robomimic_export_preserves_multi_episode_order_for_forge(tmp_path: Path) -> None:
    source = write_robomimic_hdf5(tmp_path / "robomimic.hdf5")
    export = export_with_rds(
        source,
        "robomimic_hdf5",
        "robomimic_hdf5",
        [0, 1],
        tmp_path / "exports",
    )

    rds = rds_snapshot(export.output_path, "robomimic_hdf5")
    forge = forge_snapshot(export.output_path, "robomimic_hdf5")

    assert [episode.episode_index for episode in rds.episodes] == [0, 1]
    assert_snapshots_match(rds, forge)
    assert_report(export.report_path, "robomimic_hdf5", "robomimic_hdf5", [0, 1])


def test_umi_zarr_export_matches_forge_snapshot(tmp_path: Path) -> None:
    source = write_umi_zarr(tmp_path / "umi_dataset.zarr")
    export = export_with_rds(source, "umi_zarr", "umi_zarr", [0, 1], tmp_path / "exports")

    rds = rds_snapshot(export.output_path, "umi_zarr")
    forge = forge_snapshot(export.output_path, "umi_zarr")

    assert_snapshots_match(rds, forge)
    assert_report(export.report_path, "umi_zarr", "umi_zarr", [0, 1])


def test_api_export_result_is_forge_readable(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    client = TestClient(create_app(artifact_root=tmp_path / "artifacts"))
    project_response = client.post(
        "/api/projects",
        json={"path": str(source), "format_hint": "act_hdf5"},
    )
    project_response.raise_for_status()
    project = project_response.json()

    export_response = client.post(
        f"/api/projects/{project['id']}/exports",
        json={"episode_indexes": [0], "format": "robomimic_hdf5", "options": {}},
    )
    export_response.raise_for_status()
    body = export_response.json()

    rds = rds_snapshot(Path(body["output_path"]), "robomimic_hdf5")
    forge = forge_snapshot(Path(body["output_path"]), "robomimic_hdf5")

    assert_snapshots_match(rds, forge)
    assert_report(Path(body["report_path"]), "act_hdf5", "robomimic_hdf5", [0])


def test_lerobot_v3_export_is_forge_readable_or_explicit_fallback(tmp_path: Path) -> None:
    source = write_act_hdf5(tmp_path / "episode_000000.hdf5")
    export = export_with_rds(source, "act_hdf5", "lerobot_v3", [0], tmp_path / "lerobot")
    report = json.loads(export.report_path.read_text())

    if (export.output_path / "frames.jsonl").is_file():
        assert report["backend"]["lerobot_available"] is False
        assert report["backend"]["forge_available"] is True
        assert report["backend"]["writer_backend"] == "jsonl_fallback"
        return

    rds = rds_snapshot(export.output_path, "lerobot_v3")
    forge = forge_snapshot(export.output_path, "lerobot_v3")
    assert_snapshots_match(rds, forge)
    assert report["backend"]["writer_backend"] == "lerobot"


def export_with_rds(
    source: Path,
    source_format: str,
    target_format: str,
    episode_indexes: list[int],
    output_root: Path,
):
    registry = FormatRegistry.default()
    adapter = registry.open_dataset(source, format_hint=source_format)
    return registry.export_dataset(
        adapter=adapter,
        target_format=target_format,
        episode_indexes=episode_indexes,
        output_root=output_root,
    )


def assert_report(
    report_path: Path,
    source_format: str,
    target_format: str,
    episode_indexes: list[int],
) -> None:
    report = json.loads(report_path.read_text())
    assert report["source_format"] == source_format
    assert report["target_format"] == target_format
    assert report["episode_indexes"] == episode_indexes
    assert report["episode_count"] == len(episode_indexes)
    assert "observation.state" in report["field_mapping"]
    assert "action" in report["field_mapping"]
    assert report["backend"]["forge_available"] is True


def write_act_hdf5(path: Path) -> Path:
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "action",
            data=np.asarray([[0, 0.5], [1, 1.5], [2, 2.5]], dtype="f4"),
        )
        observations = file.create_group("observations")
        observations.create_dataset(
            "qpos",
            data=np.asarray([[0, 0.5], [1, 1.5], [2, 2.5]], dtype="f4"),
        )
        observations.create_dataset("timestamp", data=np.asarray([0, 0.1, 0.2], dtype="f4"))
    return path


def write_robomimic_hdf5(path: Path) -> Path:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for index, offset in enumerate([0, 10]):
            demo = data.create_group(f"demo_{index}")
            actions = np.asarray(
                [[offset, offset + 0.5], [offset + 1, offset + 1.5]],
                dtype="f4",
            )
            demo.create_dataset("actions", data=actions)
            demo.attrs["num_samples"] = len(actions)
            obs = demo.create_group("obs")
            obs.create_dataset("qpos", data=actions)
            obs.create_dataset("robot0_joint_pos", data=actions)
    return path


def write_umi_zarr(path: Path) -> Path:
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(path, mode="w")
    data = root.create_group("data")
    data.create_dataset(
        "robot0_eef_pos",
        data=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype="f4"),
    )
    data.create_dataset(
        "action",
        data=np.asarray([[0, 0.5], [1, 1.5], [2, 2.5]], dtype="f4"),
    )
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.asarray([2, 3], dtype="i8"))
    return path
