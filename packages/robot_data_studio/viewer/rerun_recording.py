from pathlib import Path
import re

from robot_data_studio.lerobot.reader import LeRobotDatasetReader


def _video_reference_timestamps_nanos(video_asset: object, fallback_timestamps: list[float]) -> list[int]:
    if hasattr(video_asset, "read_frame_timestamps_nanos"):
        timestamps = list(video_asset.read_frame_timestamps_nanos())
        if timestamps:
            return [int(timestamp) for timestamp in timestamps]
    return [int(timestamp * 1_000_000_000) for timestamp in fallback_timestamps]


def _camera_name(video_key: str) -> str:
    name = video_key.rsplit(".", 1)[-1] or video_key
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "video"


def _episode_annotation_markdown(episode: object) -> str:
    tasks = list(getattr(episode, "tasks", []) or [])
    subtasks = list(getattr(episode, "subtasks", []) or [])
    lines = ["# Episode task"]
    if tasks:
        lines.extend(f"- {task}" for task in tasks)
    else:
        lines.append("- No task description provided.")
    if subtasks:
        lines.extend(["", "## Subtasks", "| Time | Skill | Prompt |", "| --- | --- | --- |"])
        for subtask in subtasks:
            start = float(getattr(subtask, "start_seconds", 0.0))
            end = float(getattr(subtask, "end_seconds", start))
            skill = getattr(subtask, "skill", None) or ""
            prompt = str(getattr(subtask, "prompt", "")).replace("|", "\\|")
            lines.append(f"| {start:.2f}s - {end:.2f}s | {skill} | {prompt} |")
    return "\n".join(lines)


def _log_episode_annotations(recording: object, rr: object, episode: object) -> bool:
    tasks = list(getattr(episode, "tasks", []) or [])
    subtasks = list(getattr(episode, "subtasks", []) or [])
    if not tasks and not subtasks:
        return False
    recording.log(
        "annotations/task",
        rr.TextDocument(
            _episode_annotation_markdown(episode),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )
    level = getattr(getattr(rr, "TextLogLevel", object()), "INFO", None)
    for subtask in subtasks:
        start = float(getattr(subtask, "start_seconds", 0.0))
        end = float(getattr(subtask, "end_seconds", start))
        skill = getattr(subtask, "skill", None)
        prompt = str(getattr(subtask, "prompt", "")).strip()
        label = f"{skill}: {prompt}" if skill else prompt
        recording.set_time("episode_time", duration=start)
        recording.log("annotations/subtask_log", rr.TextLog(label, level=level))
        recording.log("annotations/subtask_state", rr.StateChange(state=label))
        recording.set_time("episode_time", duration=end)
        recording.log("annotations/subtask_state", rr.StateChange(state=None))
    return True


def create_episode_recording(
    reader: LeRobotDatasetReader,
    episode_index: int,
    output_path: Path,
) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb

    episode = reader.episode(episode_index)
    frames = reader.read_episode_frames(episode_index)
    recording = rr.RecordingStream("robot_data_studio")
    recording.set_time("episode_time", duration=0.0)
    has_task_document = _log_episode_annotations(recording, rr, episode)
    has_subtasks = bool(getattr(episode, "subtasks", []) or [])
    for frame in frames:
        recording.set_time("episode_time", duration=frame.timestamp)
        recording.log("observation/state", rr.Scalars(frame.observation_state))
        recording.log("action", rr.Scalars(frame.action))
    video_keys = reader.metadata().video_keys
    if video_keys:
        video_views = []
        fallback_timestamps = [frame.timestamp for frame in frames]
        for video_key in video_keys:
            camera_name = _camera_name(video_key)
            entity_path = f"observation/videos/{camera_name}"
            video_path = reader.video_path(episode_index, video_key)
            try:
                video_asset = rr.AssetVideo(path=video_path)
            except FileNotFoundError:
                continue
            recording.log(entity_path, video_asset, static=True)
            frame_timestamps_ns = _video_reference_timestamps_nanos(
                video_asset,
                fallback_timestamps=fallback_timestamps,
            )
            recording.send_columns(
                entity_path,
                indexes=[
                    rr.TimeColumn(
                        "episode_time",
                        duration=[timestamp / 1_000_000_000 for timestamp in frame_timestamps_ns],
                    )
                ],
                columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
            )
            video_views.append(
                rrb.Spatial2DView(origin=f"/{entity_path}", name=camera_name),
            )
        recording.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    contents=[
                        rrb.Grid(contents=video_views, grid_columns=2, name="Observation videos"),
                        rrb.Vertical(
                            contents=[
                                *(
                                    [
                                        rrb.TextDocumentView(origin="/annotations/task", name="Task"),
                                    ]
                                    if has_task_document
                                    else []
                                ),
                                *(
                                    [
                                        rrb.StateTimelineView(
                                            origin="/annotations/subtask_state",
                                            name="Subtasks",
                                        ),
                                        rrb.TextLogView(
                                            origin="/annotations/subtask_log",
                                            name="Subtask prompts",
                                        ),
                                    ]
                                    if has_subtasks
                                    else []
                                ),
                                rrb.TimeSeriesView(origin="/action", name="Action"),
                                rrb.TimeSeriesView(origin="/observation/state", name="Observation state"),
                            ]
                        ),
                    ]
                ),
                collapse_panels=False,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recording.save(output_path)
    return output_path
