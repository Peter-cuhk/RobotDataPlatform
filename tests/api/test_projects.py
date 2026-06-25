from pathlib import Path

import h5py
from fastapi.testclient import TestClient

from apps.api.main import create_app
from tests.lerobot.test_reader import write_agibot_v2_episode_fixture


SAMPLE = Path("data/samples/lerobot-pusht").resolve()
ALOHA_SAMPLE = Path("data/samples/aloha_static_coffee").resolve()


def test_import_dataset_and_list_episodes(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))

    response = client.post("/api/projects", json={"path": str(SAMPLE)})

    assert response.status_code == 201
    project = response.json()
    assert project["dataset"]["total_episodes"] == 206
    episodes = client.get(f"/api/projects/{project['id']}/episodes?limit=2")
    assert episodes.status_code == 200
    assert [item["episode_index"] for item in episodes.json()] == [0, 1]


def test_import_agibot_v2_dataset_exposes_subtasks(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path / "artifacts"))
    sample = write_agibot_v2_episode_fixture(tmp_path / "agibot-one-episode")

    response = client.post("/api/projects", json={"path": str(sample), "format_hint": "lerobot_v2_1"})

    assert response.status_code == 201
    project = response.json()
    assert project["dataset"]["version"] == "v2.1"
    episodes = client.get(f"/api/projects/{project['id']}/episodes?limit=1")
    assert episodes.status_code == 200
    episode = episodes.json()[0]
    assert episode["tasks"] == ["Clear the desktop"]
    assert episode["subtasks"][0]["prompt"] == "Put the pen into the pen holder."
    assert episode["subtasks"][0]["start_seconds"] == 0.0
    assert episode["subtasks"][1]["skill"] == "Rotation"


def test_import_dataset_requires_non_empty_path(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))

    response = client.post("/api/projects", json={"path": ""})

    assert response.status_code == 400
    assert response.json()["detail"] == "Dataset path is required"


def test_import_dataset_accepts_shell_quoted_path(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))

    response = client.post("/api/projects", json={"path": f"'{SAMPLE}'"})

    assert response.status_code == 201
    assert response.json()["dataset"]["path"] == str(SAMPLE)


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


def test_aloha_static_coffee_exposes_all_video_views_and_generates_recording(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(ALOHA_SAMPLE)}).json()

    episodes = client.get(f"/api/projects/{project['id']}/episodes?limit=1")

    assert episodes.status_code == 200
    video_files = episodes.json()[0]["video_files"]
    assert set(video_files) == {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_low",
        "observation.images.cam_right_wrist",
    }

    response = client.post(f"/api/projects/{project['id']}/episodes/0/recording")

    assert response.status_code == 201
    artifact = tmp_path / response.json()["recording_url"].split("/")[-1]
    assert artifact.stat().st_size > 0


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


def test_export_episode_to_user_selected_output_directory(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path / "artifacts"))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()
    output_dir = tmp_path / "user-exports"

    response = client.post(
        f"/api/projects/{project['id']}/exports",
        json={
            "episode_indexes": [0],
            "format": "act_hdf5",
            "options": {"output_dir": str(output_dir)},
        },
    )

    assert response.status_code == 201
    output_path = Path(response.json()["output_path"])
    assert output_path.exists()
    assert output_path.parent == output_dir
    assert Path(response.json()["report_path"]).parent == output_dir


def test_run_cleaning_pipeline_and_fetch_summary(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.post(
        f"/api/projects/{project['id']}/cleaning/runs",
        json={"pass_threshold": 0.85, "review_threshold": 0.65},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["run_id"]
    summary = body["summary"]
    assert summary["total"] == 206
    assert summary["passed_count"] + summary["review_count"] + summary["excluded_count"] == 206
    assert summary["unscored_count"] == 0
    assert summary["config"] == {
        "pass_threshold": 0.85,
        "review_threshold": 0.65,
        "overwrite_manual": False,
        "vlm": {
            "enabled": False,
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "api_base_url": None,
            "api_key_configured": False,
            "prompt": (
                "You are an automated robot episode evaluator. Return only JSON with "
                "success, score, and reason. Judge whether the task was successfully "
                "completed from the visual evidence."
            ),
            "sample_frames": 4,
        },
    }
    first_result = summary["results"][0]
    assert first_result["episode_index"] == 0
    assert 0 <= first_result["score"] <= 1
    assert first_result["source"] == "auto"
    assert "smoothness" in first_result["per_attribute_scores"]

    stored = client.get(f"/api/projects/{project['id']}/cleaning")

    assert stored.status_code == 200
    assert stored.json()["total"] == 206


def test_manual_episode_decision_is_persisted_and_preserved_by_rerun(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()
    client.post(
        f"/api/projects/{project['id']}/cleaning/runs",
        json={"pass_threshold": 0.85, "review_threshold": 0.65},
    )

    decision = client.patch(
        f"/api/projects/{project['id']}/episodes/0/decision",
        json={"status": "excluded", "note": "bad wrist camera"},
    )

    assert decision.status_code == 200
    result = decision.json()
    assert result["episode_index"] == 0
    assert result["status"] == "excluded"
    assert result["source"] == "manual"
    assert result["review_note"] == "bad wrist camera"

    client.post(
        f"/api/projects/{project['id']}/cleaning/runs",
        json={"pass_threshold": 0.1, "review_threshold": 0.05},
    )
    preserved = client.get(f"/api/projects/{project['id']}/cleaning").json()["results"][0]

    assert preserved["status"] == "excluded"
    assert preserved["source"] == "manual"
    assert preserved["review_note"] == "bad wrist camera"


def test_cleaning_rerun_can_overwrite_manual_decisions(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()
    client.post(
        f"/api/projects/{project['id']}/cleaning/runs",
        json={"pass_threshold": 0.85, "review_threshold": 0.65},
    )
    client.patch(
        f"/api/projects/{project['id']}/episodes/0/decision",
        json={"status": "excluded", "note": "bad wrist camera"},
    )

    response = client.post(
        f"/api/projects/{project['id']}/cleaning/runs",
        json={"pass_threshold": 0.1, "review_threshold": 0.05, "overwrite_manual": True},
    )

    assert response.status_code == 201
    first_result = response.json()["summary"]["results"][0]
    assert first_result["source"] == "auto"
    assert first_result["status"] == "passed"
    assert first_result["review_note"] is None


def test_project_vlm_settings_are_persisted_without_returning_api_key(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    response = client.patch(
        f"/api/projects/{project['id']}/vlm-settings",
        json={
            "enabled": True,
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "api_base_url": "http://localhost:11434/v1",
            "api_key": "secret-key",
            "prompt": "Return JSON. Task: {task}",
            "sample_frames": 6,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "enabled": True,
        "provider": "OpenAI",
        "model": "gpt-4o-mini",
        "api_base_url": "http://localhost:11434/v1",
        "api_key_configured": True,
        "prompt": "Return JSON. Task: {task}",
        "sample_frames": 6,
    }
    assert "api_key" not in body

    stored = client.get(f"/api/projects/{project['id']}/vlm-settings")

    assert stored.status_code == 200
    assert stored.json() == body


def test_saved_vlm_api_key_is_used_when_cleaning_runs_without_key(tmp_path: Path) -> None:
    client = TestClient(create_app(artifact_root=tmp_path))
    project = client.post("/api/projects", json={"path": str(SAMPLE)}).json()

    client.patch(
        f"/api/projects/{project['id']}/vlm-settings",
        json={
            "enabled": True,
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "api_base_url": "http://localhost:11434/v1",
            "api_key": "secret-key",
            "prompt": "Return JSON. Task: {task}",
            "sample_frames": 6,
        },
    )

    response = client.post(
        f"/api/projects/{project['id']}/cleaning/runs",
        json={
            "pass_threshold": 0.8,
            "review_threshold": 0.6,
            "vlm": {
                "enabled": True,
                "provider": "OpenAI",
                "model": "gpt-4o-mini",
                "api_base_url": "http://localhost:11434/v1",
                "prompt": "Return JSON. Task: {task}",
                "sample_frames": 6,
            },
        },
    )

    assert response.status_code == 201
    first_result = response.json()["summary"]["results"][0]
    messages = [finding["message"] for finding in first_result["findings"]]
    assert "Set OPENAI_API_KEY or enter an API key to use OpenAI VLM scoring." not in messages
