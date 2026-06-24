from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from robot_data_studio.exporters import export_act_hdf5
from robot_data_studio.lerobot.reader import LeRobotDatasetReader

from .models import Project


class ProjectService:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._projects: dict[str, tuple[Project, LeRobotDatasetReader]] = {}

    def create(self, path: str) -> Project:
        reader = LeRobotDatasetReader(path)
        project = Project(id=uuid4().hex[:12], dataset=reader.metadata())
        self._projects[project.id] = (project, reader)
        return project

    def project(self, project_id: str) -> Project:
        return self._entry(project_id)[0]

    def reader(self, project_id: str) -> LeRobotDatasetReader:
        return self._entry(project_id)[1]

    def export_act(self, project_id: str, episode_index: int) -> Path:
        output = self.artifact_root / f"{project_id}-episode-{episode_index:06d}.hdf5"
        return export_act_hdf5(self.reader(project_id), episode_index, output)

    def _entry(self, project_id: str) -> tuple[Project, LeRobotDatasetReader]:
        try:
            return self._projects[project_id]
        except KeyError as error:
            raise KeyError(f"Project {project_id!r} not found") from error

