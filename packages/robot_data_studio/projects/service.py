from __future__ import annotations

import json
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
    FilterConfig,
    FilterConfigPatch,
    FilterDetail,
    FilterKinematicsConfig,
    FilterRun,
    FilterSummary,
    VlmSettings,
)
from robot_data_studio.quality.filter_service import DatasetFilterService, infer_gripper_indices
from robot_data_studio.quality.models import utc_now
from robot_data_studio.quality.store import CleaningStateStore, VlmSettingsStore, build_summary

from .models import Project


def _normalize_filesystem_path(path: str) -> str:
    stripped = path.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1].strip()
    return stripped


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
        output_dir: str | None = None,
    ) -> ExportResult:
        output_base = self.artifact_root
        if output_dir and output_dir.strip():
            output_base = Path(_normalize_filesystem_path(output_dir)).expanduser().resolve()
            if output_base.exists() and not output_base.is_dir():
                raise ValueError("Export output directory must be a directory")

        output = output_base / f"{project_id}-{format_id}"
        if len(episode_indexes) == 1:
            output = output_base / f"{project_id}-episode-{episode_indexes[0]:06d}-{format_id}"
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
        if "api_key" not in settings.model_fields_set:
            existing = self._vlm_settings_store(project_id).load()
            settings = settings.model_copy(update={"api_key": existing.api_key})
        return self._vlm_settings_store(project_id).save(settings)

    def run_cleaning(self, project_id: str, config: CleaningConfig) -> CleaningRun:
        episode_indexes = getattr(config, "episode_indexes", None)
        scoring_config = CleaningConfig.model_validate(
            config.model_dump(exclude={"episode_indexes"})
        )
        if config.review_threshold > config.pass_threshold:
            raise ValueError("review_threshold must be less than or equal to pass_threshold")
        if "vlm" in config.model_fields_set:
            scoring_config = scoring_config.model_copy(
                update={"vlm": self.update_vlm_settings(project_id, config.vlm)}
            )
        else:
            scoring_config = scoring_config.model_copy(update={"vlm": self.vlm_settings(project_id)})
        reader = self.reader(project_id)
        episodes = reader.list_episodes()
        if episode_indexes is not None:
            valid_indexes = {episode.episode_index for episode in episodes}
            invalid_indexes = sorted(set(episode_indexes) - valid_indexes)
            if invalid_indexes:
                raise ValueError(f"Episode indexes not found: {invalid_indexes}")
        previous = self._cleaning_store(project_id).load()
        previous_manual = {
            result.episode_index: result
            for result in (previous.results if previous else [])
            if result.source == "manual"
        }
        results = self._scorer.score_dataset(reader, scoring_config, episode_indexes)
        if not config.overwrite_manual:
            results = [
                previous_manual.get(result.episode_index, result)
                for result in results
            ]
        summary = build_summary(episodes, results, scoring_config, self._scorer.scorer_version)
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

    def run_filters(self, project_id: str) -> FilterRun:
        self.project(project_id)
        result = self._filter_service(project_id).run()
        self._filter_summary_store(project_id).write_text(
            result.summary.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return result

    def filter_summary(self, project_id: str) -> FilterSummary:
        self.project(project_id)
        store = self._filter_summary_store(project_id)
        if store.is_file():
            return FilterSummary.model_validate_json(store.read_text())
        return self._filter_service(project_id).summary()

    def filter_detail(self, project_id: str, stage_id: str, episode_index: int) -> FilterDetail:
        self.reader(project_id).episode(episode_index)
        allowed = {
            "sudden_change",
            "state_action_alignment",
            "extreme_value",
            "kinematic_consistency",
            "orientation_alignment",
        }
        if stage_id not in allowed:
            raise KeyError(f"Filter stage {stage_id!r} not found")
        return self._filter_service(project_id).detail(stage_id, episode_index)  # type: ignore[arg-type]

    def filter_config(self, project_id: str) -> FilterConfig:
        self.project(project_id)
        store = self._filter_config_store(project_id)
        if store.is_file():
            return FilterConfig.model_validate_json(store.read_text())
        config = FilterConfig(
            gripper_indices=infer_gripper_indices(self.reader(project_id)),
            kinematics=self._infer_kinematics_config(project_id),
        )
        self._save_filter_config(project_id, config)
        return config

    def update_filter_config(self, project_id: str, patch: FilterConfigPatch) -> FilterConfig:
        config = self.filter_config(project_id)
        update = {}
        if patch.gripper_indices is not None:
            update["gripper_indices"] = patch.gripper_indices
        if patch.kinematics is not None:
            update["kinematics"] = config.kinematics.model_copy(
                update=patch.kinematics.model_dump(exclude_unset=True)
            )
        updated = config.model_copy(update=update)
        self._save_filter_config(project_id, updated)
        return updated

    def save_filter_urdf(self, project_id: str, filename: str, content: bytes) -> dict[str, str]:
        self.project(project_id)
        safe_name = Path(filename).name or "robot.urdf"
        if not safe_name.endswith(".urdf"):
            raise ValueError("URDF filename must end with .urdf")
        output = self.project_artifact_dir(project_id) / "filters" / "urdf" / safe_name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
        config = self.filter_config(project_id)
        updated_kinematics = config.kinematics.model_copy(update={"urdf_path": str(output)})
        self._save_filter_config(project_id, config.model_copy(update={"kinematics": updated_kinematics}))
        return {"filename": safe_name, "path": str(output)}

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

    def _filter_service(self, project_id: str) -> DatasetFilterService:
        return DatasetFilterService(self.reader(project_id), self.filter_config(project_id))

    def _filter_config_store(self, project_id: str) -> Path:
        return self.project_artifact_dir(project_id) / "filter_config.json"

    def _filter_summary_store(self, project_id: str) -> Path:
        return self.project_artifact_dir(project_id) / "filter_summary.json"

    def _infer_kinematics_config(self, project_id: str) -> FilterKinematicsConfig:
        metadata = self.project(project_id).dataset
        state_feature = metadata.features.get("observation.state", {})
        names = state_feature.get("names", {})
        motor_names = names.get("motors", []) if isinstance(names, dict) else []
        if (
            metadata.format == "robomimic_hdf5"
            and metadata.robot_type.lower() == "panda"
            and len(motor_names) >= 10
            and motor_names[:7] == [f"panda_joint{index}" for index in range(1, 8)]
        ):
            urdf_path = Path(__file__).resolve().parents[3] / "data" / "urdf" / "panda_bullet.urdf"
            if urdf_path.is_file():
                return FilterKinematicsConfig(
                    urdf_path=str(urdf_path),
                    end_effector_link="panda_grasptarget",
                    joint_names=[f"panda_joint{index}" for index in range(1, 8)],
                    joint_state_indices=list(range(7)),
                    eef_position_indices=[7, 8, 9],
                    position_tolerance=0.02,
                    resolve_tcp_offset=True,
                )
        return FilterKinematicsConfig()

    def _save_filter_config(self, project_id: str, config: FilterConfig) -> None:
        path = self._filter_config_store(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8")
