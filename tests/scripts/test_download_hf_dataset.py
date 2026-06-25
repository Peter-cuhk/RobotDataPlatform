from pathlib import Path
import json
import tarfile

from scripts.download_hf_dataset import (
    destination_for_repo,
    episode_member_filter,
    extract_episode_archive,
    files_from_tree,
    prune_episode_metadata,
    resolve_url,
)


def test_destination_for_repo_uses_dataset_name() -> None:
    assert destination_for_repo("lerobot/aloha_static_coffee") == Path("data/samples/aloha_static_coffee")


def test_files_from_tree_skips_directories_and_keeps_paths() -> None:
    tree = [
        {"type": "directory", "path": "meta"},
        {"type": "file", "path": "meta/info.json", "size": 123},
        {"type": "file", "path": "data/chunk-000/file-000.parquet", "size": 456},
    ]

    files = files_from_tree(tree)

    assert [item.path for item in files] == [
        "meta/info.json",
        "data/chunk-000/file-000.parquet",
    ]
    assert files[1].size == 456


def test_resolve_url_points_at_main_dataset_file() -> None:
    assert resolve_url("lerobot/aloha_static_coffee", "meta/info.json") == (
        "https://huggingface.co/datasets/lerobot/aloha_static_coffee/resolve/main/meta/info.json"
    )


def test_episode_member_filter_keeps_meta_and_target_episode_files() -> None:
    keep = episode_member_filter(episode_index=7)

    assert keep("meta/info.json")
    assert keep("meta/annotations.json")
    assert keep("data/chunk-000/episode_000007.parquet")
    assert keep("videos/chunk-000/top_head/episode_000007.mp4")
    assert not keep("data/chunk-000/episode_000008.parquet")
    assert not keep("videos/chunk-000/top_head/episode_000008.mp4")


def test_extract_episode_archive_only_writes_target_episode_members(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta").mkdir()
    (source / "data").mkdir()
    (source / "meta" / "info.json").write_text("{}", encoding="utf-8")
    (source / "data" / "episode_000000.parquet").write_text("target", encoding="utf-8")
    (source / "data" / "episode_000001.parquet").write_text("other", encoding="utf-8")
    archive = tmp_path / "data.tar.gz.000"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source / "meta" / "info.json", arcname="meta/info.json")
        tar.add(source / "data" / "episode_000000.parquet", arcname="data/episode_000000.parquet")
        tar.add(source / "data" / "episode_000001.parquet", arcname="data/episode_000001.parquet")

    extract_episode_archive(archive, tmp_path / "out", episode_index=0)

    assert (tmp_path / "out" / "meta" / "info.json").is_file()
    assert (tmp_path / "out" / "data" / "episode_000000.parquet").read_text() == "target"
    assert not (tmp_path / "out" / "data" / "episode_000001.parquet").exists()


def test_prune_episode_metadata_keeps_only_requested_episode(tmp_path: Path) -> None:
    meta = tmp_path / "dataset" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": 30}),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 10}) + "\n"
        + json.dumps({"episode_index": 1, "length": 20}) + "\n",
        encoding="utf-8",
    )
    (meta / "annotations.json").write_text(
        json.dumps(
            {
                "a": {"episode_index": 0, "action_steps": []},
                "b": {"episode_index": 1, "action_steps": []},
            }
        ),
        encoding="utf-8",
    )

    prune_episode_metadata(tmp_path / "dataset", episode_index=1)

    assert json.loads((meta / "info.json").read_text()) == {"total_episodes": 1, "total_frames": 20}
    assert (meta / "episodes.jsonl").read_text() == json.dumps({"episode_index": 1, "length": 20}) + "\n"
    assert list(json.loads((meta / "annotations.json").read_text())) == ["b"]
