from pydantic import BaseModel, Field

from robot_data_studio.lerobot.models import DatasetMetadata
from robot_data_studio.quality import CleaningConfig


class Project(BaseModel):
    id: str
    dataset: DatasetMetadata


class CreateProjectRequest(BaseModel):
    path: str
    format_hint: str | None = None


class ExportOptions(BaseModel):
    output_dir: str | None = None


class ExportRequest(BaseModel):
    episode_indexes: list[int]
    format: str
    options: ExportOptions = Field(default_factory=ExportOptions)


class CleaningRunRequest(CleaningConfig):
    episode_indexes: list[int] | None = None
