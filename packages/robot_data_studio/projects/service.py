from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from robot_data_studio.formats import ExportResult, FormatRegistry
from robot_data_studio.formats.models import DatasetAdapter
from robot_data_studio.quality import (
    CleaningConfig,
    CleaningRun,
    CleaningSummary,
    EpisodeDecisionRequest,
    EpisodeQualityScorer,
    EpisodeQualityResult,
    VlmSettings,
)
from robot_data_studio.quality.models import utc_now
from robot_data_studio.quality.store import CleaningStateStore, VlmSettingsStore, build_summary

from .models import Project


class ProjectService:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._projects: dict[str, tuple[Project, DatasetAdapter]] = {}
        self._scorer = EpisodeQualityScorer()
        self._formats = FormatRegistry.default()

    def formats(self):
        return self._formats.list_formats()

    def create(self, path: str, format_hint: str | None = None) -> Project:
        path = self._normalize_user_path(path)
        if not path:
            raise ValueError("Dataset path is required")
        reader = self._formats.open_dataset(path, format_hint=format_hint)
        project = Project(id=uuid4().hex[:12], dataset=reader.metadata())
        self.project_artifact_dir(project.id).mkdir(parents=True, exist_ok=True)
        self._projects[project.id] = (project, reader)
        return project

    def project(self, project_id: str) -> Project:
        return self._entry(project_id)[0]

    def reader(self, project_id: str) -> DatasetAdapter:
        return self._entry(project_id)[1]

    def export_dataset(
        self,
        project_id: str,
        format_id: str,
        episode_indexes: list[int],
    ) -> ExportResult:
        output = self.artifact_root / f"{project_id}-{format_id}"
        if len(episode_indexes) == 1:
            output = self.artifact_root / f"{project_id}-episode-{episode_indexes[0]:06d}-{format_id}"
        return self._formats.export_dataset(
            adapter=self.reader(project_id),
            target_format=format_id,
            episode_indexes=episode_indexes,
            output_root=output,
        )

    def project_artifact_dir(self, project_id: str) -> Path:
        return self.artifact_root / "projects" / project_id

    def cleaning_summary(self, project_id: str) -> CleaningSummary:
        reader = self.reader(project_id)
        episodes = reader.list_episodes()
        existing = self._cleaning_store(project_id).load()
        if existing is not None:
            return existing
        return build_summary(episodes, [], CleaningConfig())

    def vlm_settings(self, project_id: str) -> VlmSettings:
        self.project(project_id)
        return self._vlm_settings_store(project_id).load()

    def update_vlm_settings(self, project_id: str, settings: VlmSettings) -> VlmSettings:
        self.project(project_id)
        return self._vlm_settings_store(project_id).save(settings)

    def run_cleaning(self, project_id: str, config: CleaningConfig) -> CleaningRun:
        if config.review_threshold > config.pass_threshold:
            raise ValueError("review_threshold must be less than or equal to pass_threshold")
        if "vlm" in config.model_fields_set:
            self.update_vlm_settings(project_id, config.vlm)
        else:
            config = config.model_copy(update={"vlm": self.vlm_settings(project_id)})
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

    def _entry(self, project_id: str) -> tuple[Project, DatasetAdapter]:
        try:
            return self._projects[project_id]
        except KeyError as error:
            raise KeyError(f"Project {project_id!r} not found") from error

    @staticmethod
    def _normalize_user_path(path: str) -> str:
        normalized = path.strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
            normalized = normalized[1:-1].strip()
        return normalized

    def _cleaning_store(self, project_id: str) -> CleaningStateStore:
        return CleaningStateStore(self.project_artifact_dir(project_id) / "cleaning_state.json")

    def _vlm_settings_store(self, project_id: str) -> VlmSettingsStore:
        return VlmSettingsStore(self.project_artifact_dir(project_id) / "vlm_settings.json")
