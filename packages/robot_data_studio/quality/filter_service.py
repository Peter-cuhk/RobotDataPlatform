from __future__ import annotations

import importlib.util
from uuid import uuid4

import numpy as np

from robot_data_studio.formats.models import DatasetAdapter
from robot_data_studio.quality.filter_models import (
    EpisodeFilterStageStatus,
    EpisodeFilterSummary,
    FilterConfig,
    FilterDetail,
    FilterFinding,
    FilterKinematicsConfig,
    FilterRun,
    FilterStageId,
    FilterStageSummary,
    FilterStatus,
    FilterSummary,
)
from robot_data_studio.quality.qwen_filters import (
    ExtremeValueConfig,
    StateActionAlignmentConfig,
    SuddenChangeConfig,
    detect_extreme_values,
    detect_state_action_trend_alignment,
    detect_sudden_changes,
)


STAGE_LABELS: dict[FilterStageId, str] = {
    "sudden_change": "突变检测",
    "state_action_alignment": "时间不同步",
    "extreme_value": "极值检测",
    "kinematic_consistency": "运动学一致性",
    "orientation_alignment": "姿态对齐",
}


def infer_gripper_indices(reader: DatasetAdapter) -> list[int]:
    features = reader.metadata().features
    names = features.get("observation.state", {}).get("names")
    motors = names.get("motors") if isinstance(names, dict) else None
    if not isinstance(motors, list):
        return []
    return [index for index, name in enumerate(motors) if "gripper" in str(name).lower()]


class DatasetFilterService:
    def __init__(self, reader: DatasetAdapter, config: FilterConfig) -> None:
        self.reader = reader
        self.config = config

    def run(self) -> FilterRun:
        summary = self.summary()
        return FilterRun(run_id=uuid4().hex[:12], status="succeeded", summary=summary)

    def summary(self) -> FilterSummary:
        episodes = self.reader.list_episodes()
        states_by_episode, actions_by_episode = self._episode_arrays()
        reference = self._reference_arrays(states_by_episode, actions_by_episode)
        episode_summaries = []
        totals = {stage_id: 0 for stage_id in STAGE_LABELS}
        for episode in episodes:
            states = states_by_episode[episode.episode_index]
            actions = actions_by_episode[episode.episode_index]
            status = self._stage_statuses(states, actions, reference)
            for stage_id, item in status.items():
                totals[stage_id] += item.count
            episode_summaries.append(
                EpisodeFilterSummary(episode_index=episode.episode_index, stage_status=status)
            )
        stages = [
            FilterStageSummary(
                id=stage_id,
                label=label,
                count=totals[stage_id],
                status="review" if totals[stage_id] else self._default_stage_status(stage_id),
                skipped_reason=self._default_skipped_reason(stage_id),
            )
            for stage_id, label in STAGE_LABELS.items()
        ]
        return FilterSummary(
            dataset_path=str(self.reader.root),
            total_episodes=len(episodes),
            total_frames=sum(episode.length for episode in episodes),
            stages=stages,
            episodes=episode_summaries,
        )

    def detail(self, stage_id: FilterStageId, episode_index: int) -> FilterDetail:
        frames = self.reader.read_episode_frames(episode_index)
        states = np.asarray([frame.observation_state for frame in frames], dtype=np.float64)
        actions = np.asarray([frame.action for frame in frames], dtype=np.float64)
        states_by_episode, actions_by_episode = self._episode_arrays()
        reference = self._reference_arrays(states_by_episode, actions_by_episode)
        if stage_id == "sudden_change":
            return self._sudden_change_detail(episode_index, states, actions)
        if stage_id == "state_action_alignment":
            return self._state_action_alignment_detail(episode_index, states, actions)
        if stage_id == "extreme_value":
            return self._extreme_value_detail(episode_index, states, actions, reference)
        if stage_id == "kinematic_consistency":
            return self._kinematic_detail(episode_index)
        if stage_id == "orientation_alignment":
            return self._orientation_detail(episode_index, states)
        raise ValueError(f"Unsupported filter stage: {stage_id}")

    def _episode_arrays(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        states_by_episode = {}
        actions_by_episode = {}
        for episode in self.reader.list_episodes():
            frames = self.reader.read_episode_frames(episode.episode_index)
            states_by_episode[episode.episode_index] = np.asarray(
                [frame.observation_state for frame in frames], dtype=np.float64
            )
            actions_by_episode[episode.episode_index] = np.asarray(
                [frame.action for frame in frames], dtype=np.float64
            )
        return states_by_episode, actions_by_episode

    def _reference_arrays(
        self,
        states_by_episode: dict[int, np.ndarray],
        actions_by_episode: dict[int, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            "state": np.vstack(list(states_by_episode.values())) if states_by_episode else np.empty((0, 0)),
            "action": np.vstack(list(actions_by_episode.values())) if actions_by_episode else np.empty((0, 0)),
        }

    def _stage_statuses(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        reference: dict[str, np.ndarray],
    ) -> dict[FilterStageId, EpisodeFilterStageStatus]:
        sudden = sorted(
            set(detect_sudden_changes(states, self._sudden_config()).flagged_frames)
            | set(detect_sudden_changes(actions, self._sudden_config()).flagged_frames)
        )
        alignment = detect_state_action_trend_alignment(states, actions, self._alignment_config())
        extreme = detect_extreme_values(
            {"state": states, "action": actions},
            self._extreme_config(),
            reference_values_by_key=reference,
        )
        return {
            "sudden_change": self._count_status(len(sudden)),
            "state_action_alignment": self._count_status(len(alignment.flagged_dimensions)),
            "extreme_value": self._count_status(len(extreme.flagged_frames)),
            "kinematic_consistency": EpisodeFilterStageStatus(
                count=0,
                status=self._default_stage_status("kinematic_consistency"),
                skipped_reason=self._default_skipped_reason("kinematic_consistency"),
            ),
            "orientation_alignment": EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                skipped_reason="not_configured",
            ),
        }

    def _sudden_change_detail(self, episode_index: int, states: np.ndarray, actions: np.ndarray) -> FilterDetail:
        state_result = detect_sudden_changes(states, self._sudden_config())
        action_result = detect_sudden_changes(actions, self._sudden_config())
        rows = []
        for source, result in [("state", state_result), ("action", action_result)]:
            for frame in result.flagged_frames[:200]:
                rows.append(
                    {
                        "frame": frame,
                        "source": source,
                        "dimension": 0,
                        "reason": "residual + acceleration/jerk",
                    }
                )
        return FilterDetail(
            stage_id="sudden_change",
            episode_index=episode_index,
            title=STAGE_LABELS["sudden_change"],
            status="review" if rows else "passed",
            series={"raw": self._sample(states[:, 0]), "action": self._sample(actions[:, 0])},
            thresholds={
                "state[0]": {
                    "residual": state_result.dimension_scores[0].residual_threshold,
                    "acceleration": state_result.dimension_scores[0].acceleration_threshold,
                    "jerk": state_result.dimension_scores[0].jerk_threshold,
                }
            },
            table_rows=rows,
            parameters={
                "median_windows": list(self._sudden_config().median_windows),
                "savgol_window": self._sudden_config().savgol_window,
                "residual_scale": self._sudden_config().residual_scale,
            },
            findings=self._finding_list(len(rows), "sudden_change", "检测到突变帧。"),
        )

    def _state_action_alignment_detail(self, episode_index: int, states: np.ndarray, actions: np.ndarray) -> FilterDetail:
        result = detect_state_action_trend_alignment(states, actions, self._alignment_config())
        rows = [
            {
                "dimension": item.dimension,
                "lag": item.lag,
                "correlation": round(item.correlation, 4),
                "directional_agreement": round(item.directional_agreement, 4),
                "status": "review" if item.flagged else "passed",
            }
            for item in result.dimension_results
        ]
        return FilterDetail(
            stage_id="state_action_alignment",
            episode_index=episode_index,
            title=STAGE_LABELS["state_action_alignment"],
            status="review" if result.flagged_dimensions else "passed",
            series={"state[0]": self._sample(states[:, 0]), "action[0]": self._sample(actions[:, 0])},
            table_rows=rows,
            parameters={
                "max_lag": self._alignment_config().max_lag,
                "directional_agreement_threshold": self._alignment_config().directional_agreement_threshold,
                "ignored_indices": self._alignment_config().ignored_indices,
            },
            findings=self._finding_list(
                len(result.flagged_dimensions),
                "state_action_alignment",
                "检测到 state/action 趋势不同步维度。",
            ),
        )

    def _extreme_value_detail(
        self,
        episode_index: int,
        states: np.ndarray,
        actions: np.ndarray,
        reference: dict[str, np.ndarray],
    ) -> FilterDetail:
        result = detect_extreme_values(
            {"state": states, "action": actions},
            self._extreme_config(),
            reference_values_by_key=reference,
        )
        thresholds = {}
        rows = []
        for key, dimensions in result.dimension_results.items():
            values = states if key == "state" else actions
            for dim in dimensions:
                label = f"{key}[{dim.dimension}]"
                ref = reference[key][:, dim.dimension]
                q01 = float(np.quantile(ref, self._extreme_config().lower_quantile))
                q99 = float(np.quantile(ref, self._extreme_config().upper_quantile))
                thresholds[label] = {"q01": q01, "q99": q99, "low": dim.low, "high": dim.high}
                if dim.exempt:
                    continue
                mask = (values[:, dim.dimension] < dim.low) | (values[:, dim.dimension] > dim.high)
                for frame in np.flatnonzero(mask)[:100]:
                    rows.append(
                        {
                            "frame": int(frame),
                            "dimension": label,
                            "value": round(float(values[frame, dim.dimension]), 6),
                            "low": round(dim.low, 6),
                            "high": round(dim.high, 6),
                            "gripper_exempt": dim.exempt,
                        }
                    )
        return FilterDetail(
            stage_id="extreme_value",
            episode_index=episode_index,
            title=STAGE_LABELS["extreme_value"],
            status="review" if result.flagged_frames else "passed",
            series={"state[0]": self._sample(states[:, 0]), "action[0]": self._sample(actions[:, 0])},
            thresholds=thresholds,
            table_rows=rows,
            parameters={
                "alpha": self._extreme_config().alpha,
                "q01": self._extreme_config().lower_quantile,
                "q99": self._extreme_config().upper_quantile,
                "gripper_exempt": self._extreme_config().gripper_indices,
            },
            findings=self._finding_list(len(result.flagged_frames), "extreme_value", "检测到越界帧。"),
        )

    def _kinematic_detail(self, episode_index: int) -> FilterDetail:
        if importlib.util.find_spec("pinocchio") is None:
            return self._skipped_detail(
                "kinematic_consistency",
                episode_index,
                "backend_missing",
                "Pinocchio 未安装，运动学一致性暂不可运行。",
            )
        if not self.config.kinematics.urdf_path:
            return self._skipped_detail(
                "kinematic_consistency",
                episode_index,
                "urdf_missing",
                "请先导入 URDF 文件。",
            )
        config = self.config.kinematics
        if not (
            config.end_effector_link
            and config.joint_names
            and config.eef_position_indices
        ):
            return self._skipped_detail(
                "kinematic_consistency",
                episode_index,
                "eef_pose_missing",
                "当前数据未配置 logged EEF pose 索引。",
            )
        frames = self.reader.read_episode_frames(episode_index)
        states = np.asarray([frame.observation_state for frame in frames], dtype=np.float64)
        try:
            result = _pinocchio_fk_detail(states, config)
        except Exception as error:
            return self._skipped_detail(
                "kinematic_consistency",
                episode_index,
                "fk_failed",
                f"Pinocchio FK 计算失败：{error}",
            )
        rows = [
            {
                "frame": index,
                "error": round(float(error), 6),
                "logged": [round(float(item), 6) for item in logged],
                "fk": [round(float(item), 6) for item in fk],
            }
            for index, (error, logged, fk) in enumerate(
                zip(result["errors"], result["logged"], result["predicted"], strict=True)
            )
            if error > config.position_tolerance
        ][:100]
        return FilterDetail(
            stage_id="kinematic_consistency",
            episode_index=episode_index,
            title=STAGE_LABELS["kinematic_consistency"],
            status="review" if rows else "passed",
            series={"position_error": self._sample(result["errors"])},
            table_rows=rows,
            parameters={
                **config.model_dump(),
                "max_error": round(float(np.max(result["errors"])), 6) if len(result["errors"]) else 0.0,
                "mean_error": round(float(np.mean(result["errors"])), 6) if len(result["errors"]) else 0.0,
            },
            findings=self._finding_list(len(rows), "kinematic_consistency", "检测到 FK 与 logged EEF 不一致帧。"),
        )

    def _orientation_detail(self, episode_index: int, states: np.ndarray) -> FilterDetail:
        return FilterDetail(
            stage_id="orientation_alignment",
            episode_index=episode_index,
            title=STAGE_LABELS["orientation_alignment"],
            status="skipped",
            skipped_reason="not_configured",
            series={"before_axis": self._sample(states[:, 0])},
            parameters={
                "representation": "rotation_6d or rotation_matrix",
                "positive_x": "forward",
                "correction": "not configured",
            },
            findings=[
                FilterFinding(
                    code="not_configured",
                    severity="info",
                    message="尚未配置姿态对齐修正矩阵。",
                )
            ],
        )

    def _skipped_detail(
        self,
        stage_id: FilterStageId,
        episode_index: int,
        reason: str,
        message: str,
    ) -> FilterDetail:
        return FilterDetail(
            stage_id=stage_id,
            episode_index=episode_index,
            title=STAGE_LABELS[stage_id],
            status="skipped",
            skipped_reason=reason,
            parameters=self.config.kinematics.model_dump() if stage_id == "kinematic_consistency" else {},
            findings=[FilterFinding(code=reason, severity="warn", message=message)],
        )

    def _default_stage_status(self, stage_id: FilterStageId) -> FilterStatus:
        if stage_id == "kinematic_consistency" and self._default_skipped_reason(stage_id):
            return "skipped"
        if stage_id == "orientation_alignment":
            return "skipped"
        return "passed"

    def _default_skipped_reason(self, stage_id: FilterStageId) -> str | None:
        if stage_id == "kinematic_consistency":
            if importlib.util.find_spec("pinocchio") is None:
                return "backend_missing"
            if not self.config.kinematics.urdf_path:
                return "urdf_missing"
        if stage_id == "orientation_alignment":
            return "not_configured"
        return None

    def _count_status(self, count: int) -> EpisodeFilterStageStatus:
        return EpisodeFilterStageStatus(count=count, status="review" if count else "passed")

    def _sudden_config(self) -> SuddenChangeConfig:
        return SuddenChangeConfig(
            residual_min_threshold=0.02,
            acceleration_min_threshold=0.02,
            jerk_min_threshold=0.02,
        )

    def _alignment_config(self) -> StateActionAlignmentConfig:
        return StateActionAlignmentConfig(max_lag=8, ignored_indices=self.config.gripper_indices)

    def _extreme_config(self) -> ExtremeValueConfig:
        return ExtremeValueConfig(alpha=0.5, gripper_indices=self.config.gripper_indices)

    def _finding_list(self, count: int, code: str, message: str) -> list[FilterFinding]:
        if not count:
            return []
        return [FilterFinding(code=code, severity="warn", message=f"{message} 数量：{count}")]

    def _sample(self, values: np.ndarray, max_points: int = 240) -> list[float]:
        if len(values) <= max_points:
            return [round(float(value), 6) for value in values]
        indexes = np.linspace(0, len(values) - 1, max_points).astype(int)
        return [round(float(values[index]), 6) for index in indexes]


def _pinocchio_fk_detail(states: np.ndarray, config: FilterKinematicsConfig) -> dict[str, np.ndarray]:
    import pinocchio as pin  # type: ignore[import-not-found]

    if not config.urdf_path:
        raise ValueError("URDF path is required")
    model = pin.buildModelFromUrdf(config.urdf_path)
    data = model.createData()
    frame_id = model.getFrameId(config.end_effector_link)
    if frame_id == len(model.frames):
        raise ValueError(f"End-effector link {config.end_effector_link!r} was not found")
    joint_indices = config.joint_state_indices or list(range(len(config.joint_names)))
    predicted = []
    for state in states:
        q = np.zeros(model.nq)
        for joint_name, state_index in zip(config.joint_names, joint_indices, strict=True):
            joint_id = model.getJointId(joint_name)
            if joint_id == 0:
                raise ValueError(f"Joint {joint_name!r} was not found")
            q_index = model.joints[joint_id].idx_q
            if q_index < model.nq:
                q[q_index] = state[state_index]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        predicted.append(np.array(data.oMf[frame_id].translation).reshape(3))
    predicted_array = np.asarray(predicted, dtype=np.float64)
    logged = states[:, config.eef_position_indices[:3]]
    if config.resolve_tcp_offset and len(logged):
        predicted_array = predicted_array + np.median(logged - predicted_array, axis=0)
    errors = np.linalg.norm(logged - predicted_array, axis=1)
    return {"predicted": predicted_array, "logged": logged, "errors": errors}
