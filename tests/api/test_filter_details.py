from pathlib import Path

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
        "sudden_change",
        "state_action_alignment",
        "extreme_value",
        "kinematic_consistency",
        "orientation_alignment",
    }
    assert summary["episodes"][0]["stage_status"]["extreme_value"]["count"] >= 0

    detail = client.get(f"/api/projects/{project['id']}/filters/extreme_value/episodes/0")

    assert detail.status_code == 200
    body = detail.json()
    assert body["stage_id"] == "extreme_value"
    assert body["episode_index"] == 0
    assert body["status"] in {"passed", "review", "skipped"}
    assert "state[0]" in body["series"]
    assert {"q01", "q99", "low", "high"} <= set(body["thresholds"]["state[0]"])
    assert body["parameters"]["alpha"] == 0.5


def test_filter_detail_rejects_unknown_stage(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.get(f"/api/projects/{project['id']}/filters/not_a_filter/episodes/0")

    assert response.status_code == 404


def test_kinematic_filter_accepts_urdf_upload_and_reports_missing_pinocchio(tmp_path: Path) -> None:
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

    detail = client.get(f"/api/projects/{project['id']}/filters/kinematic_consistency/episodes/0")

    assert detail.status_code == 200
    assert detail.json()["status"] == "skipped"
    assert detail.json()["skipped_reason"] == "backend_missing"
    assert "Pinocchio" in detail.json()["findings"][0]["message"]
