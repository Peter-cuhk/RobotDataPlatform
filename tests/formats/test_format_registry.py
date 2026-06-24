from pathlib import Path

import h5py
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from robot_data_studio.formats.registry import FormatRegistry, UnsupportedDatasetFormat


def test_registry_lists_mvp_formats() -> None:
    formats = FormatRegistry.default().list_formats()

    ids = {item.id for item in formats}
    assert {"lerobot_v2_1", "lerobot_v3", "act_hdf5", "robomimic_hdf5", "umi_zarr"} <= ids
    assert next(item for item in formats if item.id == "act_hdf5").can_import is True
    assert next(item for item in formats if item.id == "lerobot_v3").can_export is True


def test_registry_uses_format_hint(tmp_path: Path) -> None:
    dataset = tmp_path / "episode_000000.hdf5"
    with h5py.File(dataset, "w") as file:
        file.create_dataset("action", data=[[1.0, 2.0]])
        observations = file.create_group("observations")
        observations.create_dataset("qpos", data=[[0.5, 0.25]])

    adapter = FormatRegistry.default().open_dataset(dataset, format_hint="act_hdf5")

    assert adapter.metadata().format == "act_hdf5"
    assert adapter.list_episodes()[0].length == 1


def test_registry_rejects_unknown_format_hint(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedDatasetFormat, match="unknown_format"):
        FormatRegistry.default().open_dataset(tmp_path, format_hint="unknown_format")


def test_api_lists_formats(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))

    response = client.get("/api/formats")

    assert response.status_code == 200
    body = response.json()
    assert any(item["id"] == "lerobot_v3" for item in body)
    assert any(item["id"] == "umi_zarr" for item in body)
