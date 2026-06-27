from __future__ import annotations

import argparse
import json
import re
import tarfile
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
    local_path: str | None = None


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
    return (
        f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}"
        "?recursive=1&expand=1&limit=100"
    )


def fetch_tree(repo_id: str, revision: str = DEFAULT_REVISION) -> list[RepoFile]:
    files: list[RepoFile] = []
    url: str | None = tree_url(repo_id, revision)
    while url:
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            link = response.headers.get("Link", "")
        files.extend(files_from_tree(payload))
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return files


def select_repo_files(
    files: list[RepoFile],
    source_prefix: str | None = None,
    data_only: bool = False,
) -> list[RepoFile]:
    prefix = (source_prefix or "").strip("/")
    selected: list[RepoFile] = []
    media_suffixes = {
        ".avi",
        ".gif",
        ".jpeg",
        ".jpg",
        ".mov",
        ".mp4",
        ".png",
        ".webm",
    }
    for repo_file in files:
        if prefix:
            marker = f"{prefix}/"
            if not repo_file.path.startswith(marker):
                continue
            local_path = repo_file.path[len(marker) :]
        else:
            local_path = repo_file.path
        path = Path(local_path)
        if data_only and (
            "videos" in path.parts or path.suffix.lower() in media_suffixes
        ):
            continue
        selected.append(
            RepoFile(
                path=repo_file.path,
                size=repo_file.size,
                local_path=local_path,
            )
        )
    return selected


def download_file(repo_id: str, repo_file: RepoFile, destination: Path, revision: str, force: bool) -> None:
    output = destination / (repo_file.local_path or repo_file.path)
    if output.is_file() and not force:
        existing_size = output.stat().st_size
        if repo_file.size is None or existing_size == repo_file.size:
            print(f"exists   {output}")
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {repo_file.path}")
    with urlopen(resolve_url(repo_id, repo_file.path, revision), timeout=120) as response:
        output.write_bytes(response.read())


def episode_member_filter(episode_index: int):
    marker = f"episode_{episode_index:06d}"

    def keep(member_name: str) -> bool:
        normalized = member_name.lstrip("./")
        if normalized.startswith("meta/"):
            return True
        if marker in Path(normalized).name:
            return True
        return False

    return keep


def extract_episode_archive(archive: Path, destination: Path, episode_index: int) -> None:
    keep = episode_member_filter(episode_index)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            if keep(member.name):
                if hasattr(tarfile, "data_filter"):
                    tar.extract(member, destination, filter="data")
                else:
                    tar.extract(member, destination)


def prune_episode_metadata(destination: Path, episode_index: int) -> None:
    meta = destination / "meta"
    if not meta.is_dir():
        return
    episode_length: int | None = None
    episodes_path = meta / "episodes.jsonl"
    if episodes_path.is_file():
        rows = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if int(row.get("episode_index", -1)) == episode_index]
        if rows:
            episode_length = int(rows[0].get("length", 0))
        episodes_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    stats_path = meta / "episodes_stats.jsonl"
    if stats_path.is_file():
        rows = [
            json.loads(line)
            for line in stats_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if int(row.get("episode_index", -1)) == episode_index]
        stats_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    annotations_path = meta / "annotations.json"
    if annotations_path.is_file():
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = {
                key: value
                for key, value in payload.items()
                if isinstance(value, dict) and int(value.get("episode_index", -1)) == episode_index
            }
            annotations_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    info_path = meta / "info.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["total_episodes"] = 1
        if episode_length is not None:
            info["total_frames"] = episode_length
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_archive(path: str) -> bool:
    return ".tar.gz" in path or path.endswith((".tgz", ".tar"))


def _file_matches_episode(repo_file: RepoFile, episode_index: int) -> bool:
    path = repo_file.path
    if _is_archive(path):
        return "/meta" in path or "/data" in path or Path(path).name.startswith(("meta", "data"))
    return episode_member_filter(episode_index)(path)


def download_file_or_episode_archive(
    repo_id: str,
    repo_file: RepoFile,
    destination: Path,
    revision: str,
    force: bool,
    episode_index: int | None,
) -> None:
    if episode_index is None or not _is_archive(repo_file.path):
        download_file(repo_id, repo_file, destination, revision, force)
        return
    temp = destination / ".archives" / Path(repo_file.path).name
    temp.parent.mkdir(parents=True, exist_ok=True)
    if not temp.is_file() or force:
        print(f"download {repo_file.path}")
        with urlopen(resolve_url(repo_id, repo_file.path, revision), timeout=120) as response:
            temp.write_bytes(response.read())
    print(f"extract  {repo_file.path} episode {episode_index}")
    extract_episode_archive(temp, destination, episode_index)


def download_dataset(
    repo_id: str,
    destination: Path,
    revision: str = DEFAULT_REVISION,
    force: bool = False,
    episode_index: int | None = None,
    source_prefix: str | None = None,
    data_only: bool = False,
) -> Path:
    files = fetch_tree(repo_id, revision)
    if not files:
        raise RuntimeError(f"No files found for dataset repo {repo_id!r}")
    files = select_repo_files(files, source_prefix=source_prefix, data_only=data_only)
    if episode_index is not None:
        files = [repo_file for repo_file in files if _file_matches_episode(repo_file, episode_index)]
        if not files:
            raise RuntimeError(f"No episode {episode_index} files found for dataset repo {repo_id!r}")
    destination.mkdir(parents=True, exist_ok=True)
    for repo_file in files:
        download_file_or_episode_archive(repo_id, repo_file, destination, revision, force, episode_index)
    if episode_index is not None:
        prune_episode_metadata(destination, episode_index)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset repo into data/samples.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Dataset repo id. Default: {DEFAULT_REPO}")
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Dataset revision. Default: main")
    parser.add_argument("--destination", type=Path, help="Output directory. Default: data/samples/<repo-name>")
    parser.add_argument("--episode-index", type=int, help="Keep only one episode when downloading archive-based datasets.")
    parser.add_argument(
        "--source-prefix",
        help="Only download this repository subdirectory and strip it from local paths.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Skip video and image assets while keeping metadata, labels, and numeric data.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files even if they already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination or destination_for_repo(args.repo)
    output = download_dataset(
        args.repo,
        destination,
        args.revision,
        args.force,
        args.episode_index,
        args.source_prefix,
        args.data_only,
    )
    print(f"Dataset ready: {output.resolve()}")


if __name__ == "__main__":
    main()
