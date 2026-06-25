from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

from robot_data_studio.viewer.rerun_recording import create_episode_recording


class FakeFrame(SimpleNamespace):
    timestamp: float
    observation_state: list[float]
    action: list[float]


class FakeReader:
    def __init__(self) -> None:
        self.video_keys = [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_low",
            "observation.images.cam_right_wrist",
        ]
        self.video_path_calls: list[tuple[int, str]] = []

    def metadata(self) -> SimpleNamespace:
        return SimpleNamespace(video_keys=self.video_keys)

    def read_episode_frames(self, episode_index: int) -> list[FakeFrame]:
        assert episode_index == 7
        return [
            FakeFrame(timestamp=0.0, observation_state=[1.0, 2.0], action=[3.0, 4.0]),
            FakeFrame(timestamp=0.02, observation_state=[1.5, 2.5], action=[3.5, 4.5]),
        ]

    def episode(self, episode_index: int) -> SimpleNamespace:
        assert episode_index == 7
        return SimpleNamespace(tasks=["Clear the desktop"], subtasks=[])

    def video_path(self, episode_index: int, video_key: str) -> Path:
        self.video_path_calls.append((episode_index, video_key))
        return Path(f"/tmp/{video_key.replace('.', '_')}.mp4")


def install_fake_rerun(monkeypatch, fail_missing_video: bool = False):
    streams: list[FakeRecordingStream] = []

    class FakeRecordingStream:
        def __init__(self, application_id: str) -> None:
            self.application_id = application_id
            self.logged_paths: list[str] = []
            self.logged_values: list[object] = []
            self.column_paths: list[str] = []
            self.blueprints: list[object] = []
            streams.append(self)

        def set_time(self, *_args, **_kwargs) -> None:
            pass

        def log(self, path: str, _value: object, static: bool = False) -> None:
            self.logged_paths.append(path)
            self.logged_values.append(_value)

        def send_columns(self, path: str, **_kwargs) -> None:
            self.column_paths.append(path)

        def send_blueprint(self, blueprint: object) -> None:
            self.blueprints.append(blueprint)

        def save(self, output_path: Path) -> None:
            output_path.write_bytes(b"fake rrd")

    class FakeAssetVideo:
        def __init__(self, path: Path) -> None:
            if fail_missing_video and not path.is_file():
                raise FileNotFoundError(path)
            self.path = path

        def read_frame_timestamps_nanos(self) -> list[int]:
            return [0, 20_000_000]

    rr = ModuleType("rerun")
    rr.RecordingStream = FakeRecordingStream
    rr.AssetVideo = FakeAssetVideo
    rr.Scalars = lambda value: ("scalars", value)
    rr.TextDocument = lambda text, media_type=None: ("text-document", text, media_type)
    rr.TextLog = lambda text, level=None: ("text-log", text, level)
    rr.TextLogLevel = SimpleNamespace(INFO="INFO")
    rr.StateChange = lambda state: ("state-change", state)
    rr.MediaType = SimpleNamespace(MARKDOWN="text/markdown")
    rr.TimeColumn = lambda *_args, **_kwargs: ("time-column", _args, _kwargs)
    rr.VideoFrameReference = SimpleNamespace(columns_nanos=lambda timestamps: ("video-frames", timestamps))

    blueprint = ModuleType("rerun.blueprint")
    blueprint.Spatial2DView = lambda **kwargs: ("spatial2d", kwargs)
    blueprint.TimeSeriesView = lambda **kwargs: ("timeseries", kwargs)
    blueprint.TextDocumentView = lambda **kwargs: ("text-document-view", kwargs)
    blueprint.TextLogView = lambda **kwargs: ("text-log-view", kwargs)
    blueprint.StateTimelineView = lambda **kwargs: ("state-timeline-view", kwargs)
    blueprint.Vertical = lambda **kwargs: ("vertical", kwargs)
    blueprint.Horizontal = lambda **kwargs: ("horizontal", kwargs)
    blueprint.Grid = lambda **kwargs: ("grid", kwargs)
    blueprint.Blueprint = lambda *args, **kwargs: ("blueprint", args, kwargs)

    monkeypatch.setitem(__import__("sys").modules, "rerun", rr)
    monkeypatch.setitem(__import__("sys").modules, "rerun.blueprint", blueprint)
    return streams


def blueprint_node_names(value: object) -> list[str]:
    names = []
    if isinstance(value, tuple):
        if value and isinstance(value[0], str):
            names.append(value[0])
            children = value[1:]
        else:
            children = value
        for item in children:
            names.extend(blueprint_node_names(item))
    elif isinstance(value, dict):
        for item in value.values():
            names.extend(blueprint_node_names(item))
    elif isinstance(value, list):
        for item in value:
            names.extend(blueprint_node_names(item))
    return names


def test_create_episode_recording_logs_every_video_key(monkeypatch, tmp_path: Path) -> None:
    streams = install_fake_rerun(monkeypatch)
    reader = FakeReader()

    create_episode_recording(reader, episode_index=7, output_path=tmp_path / "episode.rrd")

    assert reader.video_path_calls == [(7, key) for key in reader.video_keys]
    recording = streams[0]
    assert recording.logged_paths == [
        "annotations/task",
        "observation/state",
        "action",
        "observation/state",
        "action",
        "observation/videos/cam_high",
        "observation/videos/cam_left_wrist",
        "observation/videos/cam_low",
        "observation/videos/cam_right_wrist",
    ]
    assert recording.column_paths == [
        "observation/videos/cam_high",
        "observation/videos/cam_left_wrist",
        "observation/videos/cam_low",
        "observation/videos/cam_right_wrist",
    ]
    assert recording.blueprints
    names = blueprint_node_names(recording.blueprints)
    assert "text-document-view" in names
    assert "state-timeline-view" not in names
    assert "text-log-view" not in names


def test_create_episode_recording_logs_task_and_subtask_annotations(monkeypatch, tmp_path: Path) -> None:
    streams = install_fake_rerun(monkeypatch)
    reader = FakeReader()
    reader.episode = lambda episode_index: SimpleNamespace(
        tasks=["Clear the desktop"],
        subtasks=[
            SimpleNamespace(
                start_frame=0,
                end_frame=2,
                start_seconds=0.0,
                end_seconds=0.02,
                prompt="Put the pen into the pen holder.",
                skill="Insert",
                track="default",
                is_mistake=False,
            ),
            SimpleNamespace(
                start_frame=2,
                end_frame=4,
                start_seconds=0.02,
                end_seconds=0.04,
                prompt="Close the laptop.",
                skill="Rotation",
                track="default",
                is_mistake=False,
            ),
        ],
    )

    create_episode_recording(reader, episode_index=7, output_path=tmp_path / "episode.rrd")

    recording = streams[0]
    assert "annotations/task" in recording.logged_paths
    assert recording.logged_paths.count("annotations/subtask_log") == 2
    assert recording.logged_paths.count("annotations/subtask_state") == 4
    markdown = next(value for value in recording.logged_values if value[0] == "text-document")[1]
    assert "Clear the desktop" in markdown
    assert "Put the pen into the pen holder." in markdown
    assert "Close the laptop." in markdown
    assert recording.blueprints
    names = blueprint_node_names(recording.blueprints)
    assert "text-document-view" in names
    assert "state-timeline-view" in names
    assert "text-log-view" in names


def test_create_episode_recording_skips_missing_video_assets(monkeypatch, tmp_path: Path) -> None:
    streams = install_fake_rerun(monkeypatch, fail_missing_video=True)
    reader = FakeReader()
    reader.video_keys = ["observation.images.top_head"]
    reader.video_path = lambda episode_index, video_key: tmp_path / "missing.mp4"

    create_episode_recording(reader, episode_index=7, output_path=tmp_path / "episode.rrd")

    recording = streams[0]
    assert "observation/videos/top_head" not in recording.logged_paths
    assert "annotations/task" in recording.logged_paths
