from pydantic import BaseModel, Field


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


class EpisodeSubtask(BaseModel):
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    prompt: str
    skill: str | None = None
    track: str | None = None
    is_mistake: bool = False


class EpisodeSummary(BaseModel):
    episode_index: int
    length: int
    duration_seconds: float
    tasks: list[str]
    subtasks: list[EpisodeSubtask] = Field(default_factory=list)
    data_file: str
    video_files: dict[str, str]
    video_start_seconds: dict[str, float]
    video_end_seconds: dict[str, float]


class EpisodeFrame(BaseModel):
    frame_index: int
    timestamp: float
    observation_state: list[float]
    action: list[float]
