from pathlib import Path

import h5py
from fastapi.testclient import TestClient

from apps.api.main import create_app


SAMPLE = Path("data/samples/lerobot-pusht").resolve()


def test_import_dataset_and_list_episodes(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))

    response = client.post("/api/projects", json={"path": str(SAMPLE)})

    assert response.status_code == 201
    project = response.json()
    assert project["dataset"]["total_episodes"] == 206
    episodes = client.get(f"/api/projects/{project['id']}/episodes?limit=2")
    assert episodes.status_code == 200
    assert [item["episode_index"] for item in episodes.json()] == [0, 1]


def test_get_episode_frames(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.get(f"/api/projects/{project['id']}/episodes/0/frames")

    assert response.status_code == 200
    assert len(response.json()) == 161
    assert response.json()[0]["action"] == [233.0, 71.0]


def test_generate_rerun_recording(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.post(f"/api/projects/{project['id']}/episodes/0/recording")

    assert response.status_code == 201
    body = response.json()
    assert body["recording_url"].endswith(".rrd")
    assert (tmp_path / body["recording_url"].split("/")[-1]).stat().st_size > 0


def test_export_episode_to_act_hdf5(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.post(
        f"/api/projects/{project['id']}/exports",
        json={"episode_indexes": [0], "format": "act_hdf5"},
    )

    assert response.status_code == 201
    output_path = Path(response.json()["output_path"])
    assert output_path.exists()
    with h5py.File(output_path, "r") as file:
        assert file["action"].shape == (161, 2)
        assert file["observations/qpos"].shape == (161, 2)
        assert file.attrs["sim"] is True or file.attrs["sim"] == 1
