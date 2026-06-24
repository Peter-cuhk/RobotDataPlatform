from pydantic import BaseModel

from robot_data_studio.lerobot.models import DatasetMetadata
from robot_data_studio.quality import CleaningConfig


class Project(BaseModel):
    id: str
    dataset: DatasetMetadata


class CreateProjectRequest(BaseModel):
    path: str
    format_hint: str | None = None


class ExportRequest(BaseModel):
    episode_indexes: list[int]
    format: str
    options: dict = {}


class CleaningRunRequest(CleaningConfig):
    pass
