from pathlib import Path

import h5py
import numpy as np
import pinocchio as pin
from fastapi.testclient import TestClient

from apps.api.main import create_app


SAMPLE = Path("data/samples/lerobot-pusht").resolve()


def test_run_filters_and_fetch_extreme_value_detail(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    run = client.post(f"/api/projects/{project['id']}/filters/runs")

    assert run.status_code == 201
    summary = run.json()["summary"]
    assert summary["total_episodes"] == 206
    assert {stage["id"] for stage in summary["stages"]} == {
        "visual_quality",
        "sudden_change",
        "state_action_alignment",
        "extreme_value",
        "kinematic_consistency",
        "orientation_alignment",
        "metadata_completeness",
    }
    assert summary["episodes"][0]["stage_status"]["extreme_value"]["count"] >= 0

    detail = client.get(f"/api/projects/{project['id']}/filters/extreme_value/episodes/0")

    assert detail.status_code == 200
    body = detail.json()
    assert body["stage_id"] == "extreme_value"
    assert body["episode_index"] == 0
    assert body["status"] in {"passed", "review", "skipped"}
    assert "state[0]" in body["series"]


def test_fetch_metadata_completeness_detail(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    run = client.post(f"/api/projects/{project['id']}/filters/runs")
    assert run.status_code == 201

    detail = client.get(
        f"/api/projects/{project['id']}/filters/metadata_completeness/episodes/0"
    )

    assert detail.status_code == 200
    body = detail.json()
    assert body["stage_id"] == "metadata_completeness"
    assert body["episode_index"] == 0
    assert body["status"] in {"passed", "review"}
    assert "findings" in body
    assert "table_rows" in body
    assert len(body["table_rows"]) == 6
    assert body["table_rows"][0]["kind"] == "inspection"
    assert "summary" in body["parameters"]


def test_fetch_time_sync_detail(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    run = client.post(f"/api/projects/{project['id']}/filters/runs")
    assert run.status_code == 201

    detail = client.get(
        f"/api/projects/{project['id']}/filters/state_action_alignment/episodes/0"
    )

    assert detail.status_code == 200
    body = detail.json()
    assert body["stage_id"] == "state_action_alignment"
    assert body["title"] == "Time sync"
    assert body["status"] in {"passed", "review"}
    assert "timestamp_delta" in body["series"]
    assert "time_sync" in body["thresholds"]
    assert "timestamp_jitter_seconds" in body["parameters"]


def test_filter_detail_rejects_unknown_stage(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.get(f"/api/projects/{project['id']}/filters/not_a_filter/episodes/0")

    assert response.status_code == 404


def test_kinematic_filter_accepts_urdf_upload_and_reports_missing_pinocchio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    upload = client.post(
        f"/api/projects/{project['id']}/filters/kinematics/urdf?filename=test_robot.urdf",
        content=b"<robot name='test_robot'/>",
        headers={"content-type": "application/octet-stream"},
    )

    assert upload.status_code == 201
    assert upload.json()["filename"] == "test_robot.urdf"
    assert Path(upload.json()["path"]).exists()

    config = client.patch(
        f"/api/projects/{project['id']}/filters/config",
        json={
            "kinematics": {
                "end_effector_link": "tool0",
                "joint_names": ["joint0", "joint1"],
                "joint_state_indices": [0, 1],
                "eef_position_indices": [0, 1, 2],
            }
        },
    )

    assert config.status_code == 200
    assert config.json()["kinematics"]["urdf_path"].endswith("test_robot.urdf")

    def fake_find_spec(name: str):
        if name == "pinocchio":
            return None
        return None

    monkeypatch.setattr(
        "packages.robot_data_studio.quality.filter_service.importlib.util.find_spec",
        fake_find_spec,
    )

    detail = client.get(f"/api/projects/{project['id']}/filters/kinematic_consistency/episodes/0")

    assert detail.status_code == 200
    assert detail.json()["status"] == "skipped"
    assert detail.json()["skipped_reason"] == "backend_missing"
    assert "Pinocchio" in detail.json()["findings"][0]["message"]


def test_project_response_includes_updated_filter_config(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    upload = client.post(
        f"/api/projects/{project['id']}/filters/kinematics/urdf?filename=test_robot.urdf",
        content=b"<robot name='test_robot'/>",
        headers={"content-type": "application/octet-stream"},
    )
    assert upload.status_code == 201

    config = client.patch(
        f"/api/projects/{project['id']}/filters/config",
        json={
            "kinematics": {
                "end_effector_link": "tool0",
                "joint_names": ["joint0", "joint1"],
                "joint_state_indices": [0, 1],
                "eef_position_indices": [0, 1, 2],
            }
        },
    )
    assert config.status_code == 200

    response = client.get(f"/api/projects/{project['id']}")

    assert response.status_code == 200
    kinematics = response.json()["filter_config"]["kinematics"]
    assert kinematics["urdf_path"].endswith("test_robot.urdf")
    assert kinematics["end_effector_link"] == "tool0"
    assert kinematics["joint_names"] == ["joint0", "joint1"]
    assert kinematics["eef_position_indices"] == [0, 1, 2]


def test_kinematic_filter_autoconfigures_robomimic_panda_dataset(tmp_path: Path) -> None:
    dataset = _write_robomimic_panda_fk_dataset(tmp_path / "robomimic_panda.hdf5")
    client = TestClient(create_app(artifact_root=tmp_path / "artifacts"))
    project = client.post("/api/projects", json={"path": str(dataset), "format_hint": "robomimic_hdf5"}).json()

    summary = client.post(f"/api/projects/{project['id']}/filters/runs").json()["summary"]
    detail = client.get(f"/api/projects/{project['id']}/filters/kinematic_consistency/episodes/0")

    stage = summary["episodes"][0]["stage_status"]["kinematic_consistency"]
    assert stage["score"] == 1.0
    assert stage["severity"] == "none"
    assert stage["skipped_reason"] is None
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] == "passed"
    assert body["skipped_reason"] is None
    assert max(body["series"]["position_error"]) < 0.001
    assert body["parameters"]["end_effector_link"] == "panda_grasptarget"
    assert body["parameters"]["joint_names"] == [f"panda_joint{index}" for index in range(1, 8)]
    assert body["parameters"]["joint_state_indices"] == list(range(7))
    assert body["parameters"]["eef_position_indices"] == [7, 8, 9]


def _write_robomimic_panda_fk_dataset(path: Path) -> Path:
    urdf_path = Path("data/urdf/panda_bullet.urdf").resolve()
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    frame_id = model.getFrameId("panda_grasptarget")
    offset = np.asarray([0.55, -0.1, 0.9], dtype=np.float64)
    joint_rows = []
    eef_rows = []
    for index in range(12):
        q = np.asarray(
            [
                0.2 * np.sin(index / 10),
                -0.4 + 0.05 * np.cos(index / 8),
                0.15 * np.sin(index / 7),
                -2.1 + 0.04 * np.cos(index / 6),
                0.05 * np.sin(index / 5),
                1.8 + 0.03 * np.cos(index / 4),
                0.7 + 0.02 * np.sin(index / 3),
            ],
            dtype=np.float64,
        )
        full_q = np.zeros(model.nq)
        full_q[:7] = q
        pin.forwardKinematics(model, data, full_q)
        pin.updateFramePlacements(model, data)
        joint_rows.append(q)
        eef_rows.append(np.asarray(data.oMf[frame_id].translation) + offset)
    joints = np.asarray(joint_rows, dtype=np.float64)
    eef = np.asarray(eef_rows, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        group = file.create_group("data")
        group.attrs["total"] = len(joints)
        group.attrs["env_args"] = '{"env_name":"Lift","env_kwargs":{"robots":["Panda"]}}'
        demo = group.create_group("demo_0")
        demo.attrs["num_samples"] = len(joints)
        demo.create_dataset("actions", data=np.zeros((len(joints), 7), dtype=np.float64))
        obs = demo.create_group("obs")
        obs.create_dataset("robot0_joint_pos", data=joints)
        obs.create_dataset("robot0_eef_pos", data=eef)
    return path
