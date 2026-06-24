from pydantic import BaseModel


class DatasetProbe(BaseModel):
    format: str
    version: str
    confidence: float


class DatasetMetadata(BaseModel):
    path: str
    format: str
    version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    fps: float
    video_keys: list[str]
    scalar_keys: list[str]
    features: dict[str, dict]


class EpisodeSummary(BaseModel):
    episode_index: int
    length: int
    duration_seconds: float
    tasks: list[str]
    data_file: str
    video_files: dict[str, str]
    video_start_seconds: dict[str, float]
    video_end_seconds: dict[str, float]


class EpisodeFrame(BaseModel):
    frame_index: int
    timestamp: float
    observation_state: list[float]
    action: list[float]

