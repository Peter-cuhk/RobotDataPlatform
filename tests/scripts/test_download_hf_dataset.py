from pathlib import Path

from scripts.download_hf_dataset import destination_for_repo, files_from_tree, resolve_url


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
