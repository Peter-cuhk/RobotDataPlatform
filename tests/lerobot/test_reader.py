from pathlib import Path

import pytest

from robot_data_studio.lerobot.reader import LeRobotDatasetReader, NotLeRobotDataset


SAMPLE = Path("data/samples/lerobot-pusht")


def test_probe_recognizes_lerobot_v3_dataset() -> None:
    probe = LeRobotDatasetReader.probe(SAMPLE)

    assert probe.format == "lerobot"
    assert probe.version == "v3.0"
    assert probe.confidence == 1.0


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


def test_reader_reads_episode_frame_data() -> None:
    reader = LeRobotDatasetReader(SAMPLE)

    frames = reader.read_episode_frames(episode_index=0)

    assert len(frames) == 161
    assert frames[0].timestamp == pytest.approx(0.0)
    assert frames[0].observation_state == [222.0, 97.0]
    assert frames[0].action == [233.0, 71.0]

