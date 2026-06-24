from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_REPO = "lerobot/aloha_static_coffee"
DEFAULT_REVISION = "main"


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int | None = None


def destination_for_repo(repo_id: str) -> Path:
    return Path("data") / "samples" / repo_id.split("/")[-1]


def files_from_tree(tree: list[dict]) -> list[RepoFile]:
    files = []
    for item in tree:
        if item.get("type") != "file":
            continue
        path = item.get("path")
        if not path:
            continue
        files.append(RepoFile(path=path, size=item.get("size")))
    return files


def resolve_url(repo_id: str, relative_path: str, revision: str = DEFAULT_REVISION) -> str:
    encoded_path = quote(relative_path, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded_path}"


def tree_url(repo_id: str, revision: str = DEFAULT_REVISION) -> str:
    return f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}?recursive=1"


def fetch_tree(repo_id: str, revision: str = DEFAULT_REVISION) -> list[RepoFile]:
    with urlopen(tree_url(repo_id, revision), timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return files_from_tree(payload)


def download_file(repo_id: str, repo_file: RepoFile, destination: Path, revision: str, force: bool) -> None:
    output = destination / repo_file.path
    if output.is_file() and not force:
        existing_size = output.stat().st_size
        if repo_file.size is None or existing_size == repo_file.size:
            print(f"exists   {output}")
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {repo_file.path}")
    with urlopen(resolve_url(repo_id, repo_file.path, revision), timeout=120) as response:
        output.write_bytes(response.read())


def download_dataset(repo_id: str, destination: Path, revision: str = DEFAULT_REVISION, force: bool = False) -> Path:
    files = fetch_tree(repo_id, revision)
    if not files:
        raise RuntimeError(f"No files found for dataset repo {repo_id!r}")
    destination.mkdir(parents=True, exist_ok=True)
    for repo_file in files:
        download_file(repo_id, repo_file, destination, revision, force)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset repo into data/samples.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Dataset repo id. Default: {DEFAULT_REPO}")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset revision. Default: main")
    parser.add_argument("--destination", type=Path, help="Output directory. Default: data/samples/<repo-name>")
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination or destination_for_repo(args.repo)
    output = download_dataset(args.repo, destination, args.revision, args.force)
    print(f"Dataset ready: {output.resolve()}")


if __name__ == "__main__":
    main()
