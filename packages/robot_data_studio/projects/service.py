from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from robot_data_studio.exporters import export_act_hdf5
from robot_data_studio.lerobot.reader import LeRobotDatasetReader
from robot_data_studio.quality import (
    CleaningConfig,
    CleaningRun,
    CleaningSummary,
    EpisodeDecisionRequest,
    EpisodeQualityScorer,
    EpisodeQualityResult,
)
from robot_data_studio.quality.models import utc_now
from robot_data_studio.quality.store import CleaningStateStore, build_summary

from .models import Project


class ProjectService:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._projects: dict[str, tuple[Project, LeRobotDatasetReader]] = {}
        self._scorer = EpisodeQualityScorer()

    def create(self, path: str) -> Project:
        reader = LeRobotDatasetReader(path)
        project = Project(id=uuid4().hex[:12], dataset=reader.metadata())
        self.project_artifact_dir(project.id).mkdir(parents=True, exist_ok=True)
        self._projects[project.id] = (project, reader)
        return project

    def project(self, project_id: str) -> Project:
        return self._entry(project_id)[0]

    def reader(self, project_id: str) -> LeRobotDatasetReader:
        return self._entry(project_id)[1]

    def export_act(self, project_id: str, episode_index: int) -> Path:
        output = self.artifact_root / f"{project_id}-episode-{episode_index:06d}.hdf5"
        return export_act_hdf5(self.reader(project_id), episode_index, output)

    def project_artifact_dir(self, project_id: str) -> Path:
        return self.artifact_root / "projects" / project_id

    def cleaning_summary(self, project_id: str) -> CleaningSummary:
        reader = self.reader(project_id)
        episodes = reader.list_episodes()
        existing = self._cleaning_store(project_id).load()
        if existing is not None:
            return existing
        return build_summary(episodes, [], CleaningConfig())

    def run_cleaning(self, project_id: str, config: CleaningConfig) -> CleaningRun:
        if config.review_threshold > config.pass_threshold:
            raise ValueError("review_threshold must be less than or equal to pass_threshold")
        reader = self.reader(project_id)
        episodes = reader.list_episodes()
        previous = self._cleaning_store(project_id).load()
        previous_manual = {
            result.episode_index: result
            for result in (previous.results if previous else [])
            if result.source == "manual"
        }
        results = self._scorer.score_dataset(reader, config)
        if not config.overwrite_manual:
            results = [
                previous_manual.get(result.episode_index, result)
                for result in results
            ]
        summary = build_summary(episodes, results, config, self._scorer.scorer_version)
        self._cleaning_store(project_id).save(summary)
        return CleaningRun(run_id=uuid4().hex[:12], status="succeeded", summary=summary)

    def update_episode_decision(
        self,
        project_id: str,
        episode_index: int,
        request: EpisodeDecisionRequest,
    ) -> EpisodeQualityResult:
        reader = self.reader(project_id)
        reader.episode(episode_index)
        summary = self.cleaning_summary(project_id)
        by_index = {result.episode_index: result for result in summary.results}
        existing = by_index[episode_index]
        updated = existing.model_copy(
            update={
                "status": request.status,
                "source": "manual",
                "review_note": request.note,
                "updated_at": utc_now(),
            }
        )
        by_index[episode_index] = updated
        new_summary = build_summary(
            reader.list_episodes(),
            list(by_index.values()),
            summary.config,
            summary.scorer_version,
        )
        self._cleaning_store(project_id).save(new_summary)
        return updated

    def _entry(self, project_id: str) -> tuple[Project, LeRobotDatasetReader]:
        try:
            return self._projects[project_id]
        except KeyError as error:
            raise KeyError(f"Project {project_id!r} not found") from error

    def _cleaning_store(self, project_id: str) -> CleaningStateStore:
        return CleaningStateStore(self.project_artifact_dir(project_id) / "cleaning_state.json")
