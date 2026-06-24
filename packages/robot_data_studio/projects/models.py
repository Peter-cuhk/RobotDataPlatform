from pydantic import BaseModel

from robot_data_studio.lerobot.models import DatasetMetadata
from robot_data_studio.quality import CleaningConfig


class Project(BaseModel):
    id: str
    dataset: DatasetMetadata


class CreateProjectRequest(BaseModel):
    path: str


class ExportRequest(BaseModel):
    episode_indexes: list[int]
    format: str


class CleaningRunRequest(CleaningConfig):
    pass
