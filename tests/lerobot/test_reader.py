from pathlib import Path
import json

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from robot_data_studio.lerobot.reader import LeRobotDatasetReader, NotLeRobotDataset


SAMPLE = Path("data/samples/lerobot-pusht")


def write_agibot_v2_episode_fixture(root: Path) -> Path:
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "g2a",
                "total_episodes": 1,
                "total_frames": 4,
                "total_tasks": 1,
                "fps": 30,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.top_head": {"dtype": "video"},
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "action": {"dtype": "float32", "shape": [2]},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "Clear the desktop"}) + "\n",
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["Clear the desktop"], "length": 4}) + "\n",
        encoding="utf-8",
    )
    (meta / "annotations.json").write_text(
        json.dumps(
            {
                "27": {
                    "episode_index": 0,
                    "action_steps": [
                        {
                            "track": "default",
                            "start_frame": 0,
                            "end_frame": 2,
                            "action_text": "Put the pen into the pen holder.",
                            "skill": "Insert",
                            "is_mistake": False,
                        },
                        {
                            "track": "default",
                            "start_frame": 2,
                            "end_frame": 4,
                            "action_text": "Close the laptop.",
                            "skill": "Rotation",
                            "is_mistake": False,
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "episode_index": pa.array([0, 0, 0, 0], type=pa.int64()),
            "frame_index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "timestamp": pa.array([0.0, 1 / 30, 2 / 30, 3 / 30], type=pa.float64()),
            "observation.state": pa.array([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]]),
            "action": pa.array([[3.0, 4.0], [3.1, 4.1], [3.2, 4.2], [3.3, 4.3]]),
        }
    )
    pq.write_table(table, data / "episode_000000.parquet")
    return root


def test_probe_recognizes_lerobot_v3_dataset() -> None:
    probe = LeRobotDatasetReader.probe(SAMPLE)

    assert probe.format == "lerobot"
    assert probe.version == "v3.0"
    assert probe.confidence == 1.0


def test_probe_recognizes_agibot_lerobot_v2_dataset(tmp_path: Path) -> None:
    fixture = write_agibot_v2_episode_fixture(tmp_path / "agibot-one-episode")

    probe = LeRobotDatasetReader.probe(fixture)

    assert probe.format == "lerobot"
    assert probe.version == "v2.1"
    assert probe.confidence == 1.0


def test_reader_accepts_legacy_v2_metadata_at_dataset_root(tmp_path: Path) -> None:
    fixture = write_agibot_v2_episode_fixture(tmp_path / "root-meta-v2")
    meta = fixture / "meta"
    for path in list(meta.iterdir()):
        path.rename(fixture / path.name)
    meta.rmdir()

    reader = LeRobotDatasetReader(fixture)

    assert reader.metadata().version == "v2.1"
    assert reader.list_episodes()[0].length == 4


def test_probe_rejects_directory_without_info_file(tmp_path: Path) -> None:
    with pytest.raises(NotLeRobotDataset):
        LeRobotDatasetReader.probe(tmp_path)


def test_reader_exposes_dataset_metadata() -> None:
    reader = LeRobotDatasetReader(SAMPLE)

    metadata = reader.metadata()

    assert metadata.total_episodes == 206
    assert metadata.total_frames == 25650
    assert metadata.fps == 10
    assert metadata.video_keys == ["observation.image"]
    assert metadata.scalar_keys == ["observation.state", "action"]


def test_reader_lists_episode_summaries() -> None:
    reader = LeRobotDatasetReader(SAMPLE)

    episodes = reader.list_episodes(limit=2)

    assert [episode.episode_index for episode in episodes] == [0, 1]
    assert episodes[0].length == 161
    assert episodes[0].duration_seconds == pytest.approx(16.1)
    assert episodes[0].tasks == ["Push the T-shaped block onto the T-shaped target."]
    assert episodes[0].subtasks == []


def test_reader_lists_agibot_v2_subtask_prompts(tmp_path: Path) -> None:
    fixture = write_agibot_v2_episode_fixture(tmp_path / "agibot-one-episode")
    reader = LeRobotDatasetReader(fixture)

    episodes = reader.list_episodes(limit=1)

    assert episodes[0].tasks == ["Clear the desktop"]
    assert len(episodes[0].subtasks) == 2
    assert episodes[0].subtasks[0].prompt == "Put the pen into the pen holder."
    assert episodes[0].subtasks[0].skill == "Insert"
    assert episodes[0].subtasks[0].start_seconds == pytest.approx(0.0)
    assert episodes[0].subtasks[0].end_seconds == pytest.approx(2 / 30)
    assert episodes[0].subtasks[1].prompt == "Close the laptop."


def test_reader_handles_agibot_v2_without_annotations(tmp_path: Path) -> None:
    fixture = write_agibot_v2_episode_fixture(tmp_path / "agibot-one-episode")
    (fixture / "meta" / "annotations.json").unlink()
    reader = LeRobotDatasetReader(fixture)

    episodes = reader.list_episodes(limit=1)

    assert episodes[0].subtasks == []


def test_reader_reads_agibot_v2_episode_frame_data(tmp_path: Path) -> None:
    fixture = write_agibot_v2_episode_fixture(tmp_path / "agibot-one-episode")
    reader = LeRobotDatasetReader(fixture)

    frames = reader.read_episode_frames(episode_index=0)

    assert len(frames) == 4
    assert frames[2].timestamp == pytest.approx(2 / 30)
    assert frames[2].observation_state == [1.2, 2.2]
    assert frames[2].action == [3.2, 4.2]


def test_reader_reads_episode_frame_data() -> None:
    reader = LeRobotDatasetReader(SAMPLE)

    frames = reader.read_episode_frames(episode_index=0)

    assert len(frames) == 161
    assert frames[0].timestamp == pytest.approx(0.0)
    assert frames[0].observation_state == [222.0, 97.0]
    assert frames[0].action == [233.0, 71.0]
