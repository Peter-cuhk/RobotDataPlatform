from pathlib import Path
from urllib.request import urlopen


REPOSITORY = "https://huggingface.co/datasets/lerobot/pusht/resolve/main"
FILES = (
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
)


def main() -> None:
    destination = Path("data/samples/lerobot-pusht")
    for relative_path in FILES:
        output = destination / relative_path
        if output.is_file():
            print(f"exists   {output}")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {relative_path}")
        with urlopen(f"{REPOSITORY}/{relative_path}") as response:
            output.write_bytes(response.read())
    print(f"LeRobot sample ready: {destination.resolve()}")


if __name__ == "__main__":
    main()
