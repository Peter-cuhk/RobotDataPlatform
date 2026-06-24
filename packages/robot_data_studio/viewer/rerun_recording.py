from pathlib import Path

from robot_data_studio.lerobot.reader import LeRobotDatasetReader


def _video_reference_timestamps_nanos(video_asset: object, fallback_timestamps: list[float]) -> list[int]:
    if hasattr(video_asset, "read_frame_timestamps_nanos"):
        timestamps = list(video_asset.read_frame_timestamps_nanos())
        if timestamps:
            return [int(timestamp) for timestamp in timestamps]
    return [int(timestamp * 1_000_000_000) for timestamp in fallback_timestamps]


def create_episode_recording(
    reader: LeRobotDatasetReader,
    episode_index: int,
    output_path: Path,
) -> Path:
    import rerun as rr
    import rerun.blueprint as rrb

    frames = reader.read_episode_frames(episode_index)
    recording = rr.RecordingStream("robot_data_studio")
    recording.set_time("episode_time", duration=0.0)
    for frame in frames:
        recording.set_time("episode_time", duration=frame.timestamp)
        recording.log("observation/state", rr.Scalars(frame.observation_state))
        recording.log("action", rr.Scalars(frame.action))
    video_keys = reader.metadata().video_keys
    if video_keys:
        video_path = reader.video_path(episode_index, video_keys[0])
        video_asset = rr.AssetVideo(path=video_path)
        recording.log("observation/video", video_asset, static=True)
        frame_timestamps_ns = _video_reference_timestamps_nanos(
            video_asset,
            fallback_timestamps=[frame.timestamp for frame in frames],
        )
        recording.send_columns(
            "observation/video",
            indexes=[rr.TimeColumn("episode_time", duration=[timestamp / 1_000_000_000 for timestamp in frame_timestamps_ns])],
            columns=rr.VideoFrameReference.columns_nanos(frame_timestamps_ns),
        )
        recording.send_blueprint(
            rrb.Blueprint(
                rrb.Horizontal(
                    contents=[
                        rrb.Spatial2DView(origin="/observation/video", name="Observation video"),
                        rrb.Vertical(
                            contents=[
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
