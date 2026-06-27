from __future__ import annotations

import importlib.util
from collections import Counter
from collections.abc import Callable
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
    "sudden_change": "Sudden change",
    "state_action_alignment": "Time sync",
    "extreme_value": "Extreme value",
    "kinematic_consistency": "Kinematic consistency",
    "orientation_alignment": "Orientation alignment",
}

ProgressCallback = Callable[[int, int], None]


def infer_gripper_indices(reader: DatasetAdapter) -> list[int]:
    features = reader.metadata().features
    names = features.get("observation.state", {}).get("names")
    motors = names.get("motors") if isinstance(names, dict) else names
    if not isinstance(motors, list):
        return []
    return [index for index, name in enumerate(motors) if "gripper" in str(name).lower()]


class DatasetFilterService:
    def __init__(self, reader: DatasetAdapter, config: FilterConfig) -> None:
        self.reader = reader
        self.config = config

    def run(self, on_progress: ProgressCallback | None = None) -> FilterRun:
        summary = self.summary(on_progress=on_progress)
        return FilterRun(run_id=uuid4().hex[:12], status="succeeded", summary=summary)

    def summary(self, on_progress: ProgressCallback | None = None) -> FilterSummary:
        episodes = self.reader.list_episodes()
        total_episodes = len(episodes)
        total_units = total_episodes * 2
        load_progress: ProgressCallback | None = None
        if on_progress is not None:
            def load_progress(completed: int, _total: int) -> None:
                on_progress(completed, total_units)
        states_by_episode, actions_by_episode, integrity_by_episode = self._episode_arrays(
            on_progress=load_progress
        )
        reference = self._reference_arrays(states_by_episode, actions_by_episode)
        episode_summaries = []
        totals = {stage_id: 0 for stage_id in STAGE_LABELS}
        for index, episode in enumerate(episodes, start=1):
            states = states_by_episode[episode.episode_index]
            actions = actions_by_episode[episode.episode_index]
            integrity_findings = integrity_by_episode[episode.episode_index]
            status = (
                self._integrity_failed_statuses()
                if integrity_findings
                else self._stage_statuses(states, actions, reference)
            )
            for stage_id, item in status.items():
                totals[stage_id] += item.count
            episode_summaries.append(
                EpisodeFilterSummary(
                    episode_index=episode.episode_index,
                    stage_status=status,
                    critical_findings=integrity_findings,
                )
            )
            if on_progress is not None:
                on_progress(total_episodes + index, total_units)
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
        states_by_episode, actions_by_episode, _integrity_by_episode = self._episode_arrays()
        reference = self._reference_arrays(states_by_episode, actions_by_episode)
        normalized_states = self._normalize(states, reference["state"])
        normalized_actions = self._normalize(actions, reference["action"])
        if stage_id == "sudden_change":
            return self._sudden_change_detail(
                episode_index,
                states,
                actions,
                normalized_states,
                normalized_actions,
            )
        if stage_id == "state_action_alignment":
            return self._state_action_alignment_detail(
                episode_index,
                states,
                actions,
                normalized_states,
                normalized_actions,
            )
        if stage_id == "extreme_value":
            return self._extreme_value_detail(episode_index, states, actions, reference)
        if stage_id == "kinematic_consistency":
            return self._kinematic_detail(episode_index)
        if stage_id == "orientation_alignment":
            return self._orientation_detail(episode_index, states)
        raise ValueError(f"Unsupported filter stage: {stage_id}")

    def _episode_arrays(
        self, on_progress: ProgressCallback | None = None
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, np.ndarray],
        dict[int, list[FilterFinding]],
    ]:
        states_by_episode = {}
        actions_by_episode = {}
        integrity_by_episode = {}
        episodes = self.reader.list_episodes()
        total = len(episodes)
        for completed, episode in enumerate(episodes, start=1):
            frames = self.reader.read_episode_frames(episode.episode_index)
            integrity_by_episode[episode.episode_index] = self._integrity_findings(frames)
            state_rows = [frame.observation_state for frame in frames]
            action_rows = [frame.action for frame in frames]
            states_by_episode[episode.episode_index] = (
                np.asarray(state_rows, dtype=np.float64)
                if len({len(row) for row in state_rows}) <= 1
                else np.empty((0, 0), dtype=np.float64)
            )
            actions_by_episode[episode.episode_index] = (
                np.asarray(action_rows, dtype=np.float64)
                if len({len(row) for row in action_rows}) <= 1
                else np.empty((0, 0), dtype=np.float64)
            )
            if on_progress is not None:
                on_progress(completed, total)
        self._mark_cross_episode_dimension_mismatches(
            states_by_episode,
            actions_by_episode,
            integrity_by_episode,
        )
        return states_by_episode, actions_by_episode, integrity_by_episode

    @staticmethod
    def _mark_cross_episode_dimension_mismatches(
        states_by_episode: dict[int, np.ndarray],
        actions_by_episode: dict[int, np.ndarray],
        integrity_by_episode: dict[int, list[FilterFinding]],
    ) -> None:
        for stream_name, arrays in (
            ("state", states_by_episode),
            ("action", actions_by_episode),
        ):
            widths = {
                episode_index: array.shape[1]
                for episode_index, array in arrays.items()
                if array.ndim == 2 and array.shape[1] > 0
            }
            if not widths:
                continue
            canonical_width = Counter(widths.values()).most_common(1)[0][0]
            for episode_index, width in widths.items():
                if width == canonical_width or any(
                    finding.code == "dimension_mismatch"
                    for finding in integrity_by_episode[episode_index]
                ):
                    continue
                integrity_by_episode[episode_index].append(
                    FilterFinding(
                        code="dimension_mismatch",
                        severity="critical",
                        message=(
                            f"Episode {stream_name} width {width} differs from "
                            f"dataset width {canonical_width}."
                        ),
                    )
                )

    @staticmethod
    def _integrity_findings(frames: list) -> list[FilterFinding]:
        findings: list[FilterFinding] = []
        if not frames:
            return [
                FilterFinding(
                    code="empty_episode",
                    severity="critical",
                    message="Episode contains no frames.",
                )
            ]
        state_dimensions = {len(frame.observation_state) for frame in frames}
        action_dimensions = {len(frame.action) for frame in frames}
        if (
            len(state_dimensions) != 1
            or len(action_dimensions) != 1
            or next(iter(state_dimensions), 0) <= 0
            or next(iter(action_dimensions), 0) <= 0
        ):
            findings.append(
                FilterFinding(
                    code="dimension_mismatch",
                    severity="critical",
                    message="State/action dimensions are inconsistent.",
                )
            )
        numeric_values = [
            value
            for frame in frames
            for value in [*frame.observation_state, *frame.action]
        ]
        if numeric_values and not np.all(np.isfinite(numeric_values)):
            findings.append(
                FilterFinding(
                    code="non_finite_signal",
                    severity="critical",
                    message="State/action contains NaN or infinite values.",
                )
            )
        timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0):
            findings.append(
                FilterFinding(
                    code="non_monotonic_timestamp",
                    severity="critical",
                    message="Episode timestamps must be finite and must not move backward.",
                )
            )
        return findings

    @staticmethod
    def _integrity_failed_statuses() -> dict[FilterStageId, EpisodeFilterStageStatus]:
        return {
            stage_id: EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                score=None,
                severity="none",
                skipped_reason="integrity_failure",
            )
            for stage_id in STAGE_LABELS
        }

    def _reference_arrays(
        self,
        states_by_episode: dict[int, np.ndarray],
        actions_by_episode: dict[int, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            "state": self._stack_finite_arrays(states_by_episode.values()),
            "action": self._stack_finite_arrays(actions_by_episode.values()),
        }

    @staticmethod
    def _stack_finite_arrays(values) -> np.ndarray:
        arrays = [
            array
            for array in values
            if array.ndim == 2 and array.shape[1] > 0 and np.all(np.isfinite(array))
        ]
        if not arrays:
            return np.empty((0, 0), dtype=np.float64)
        width = arrays[0].shape[1]
        compatible = [array for array in arrays if array.shape[1] == width]
        return np.vstack(compatible) if compatible else np.empty((0, 0), dtype=np.float64)

    def _stage_statuses(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        reference: dict[str, np.ndarray],
    ) -> dict[FilterStageId, EpisodeFilterStageStatus]:
        normalized_states = self._normalize(states, reference["state"])
        normalized_actions = self._normalize(actions, reference["action"])
        sudden_states = self._without_indices(
            normalized_states,
            self.config.gripper_indices,
        )
        sudden_actions = self._without_indices(
            normalized_actions,
            self.config.gripper_indices,
        )
        sudden = sorted(
            set(
                detect_sudden_changes(
                    sudden_states, self._sudden_config()
                ).flagged_frames
            )
            | set(
                detect_sudden_changes(
                    sudden_actions, self._sudden_config()
                ).flagged_frames
            )
        )
        alignment = detect_state_action_trend_alignment(
            normalized_states,
            normalized_actions,
            self._alignment_config(),
        )
        extreme = detect_extreme_values(
            {"state": states, "action": actions},
            self._extreme_config(),
            reference_values_by_key=reference,
        )
        active_dimensions = max(
            0,
            min(states.shape[1], actions.shape[1]) - len(self.config.gripper_indices),
        )
        return {
            "sudden_change": self._count_status(
                len(sudden),
                len(states),
                critical_rate=0.02,
                critical=False,
            ),
            "state_action_alignment": self._count_status(
                len(alignment.flagged_dimensions),
                active_dimensions,
                critical_rate=0.5,
                critical=False,
            ),
            "extreme_value": self._count_status(
                len(extreme.flagged_frames),
                len(states),
                critical_rate=0.01,
                critical=True,
            ),
            "kinematic_consistency": self._kinematic_status(states),
            "orientation_alignment": EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                score=None,
                skipped_reason="not_configured",
            ),
        }

    def _kinematic_status(self, states: np.ndarray) -> EpisodeFilterStageStatus:
        skipped_reason = self._default_skipped_reason("kinematic_consistency")
        if skipped_reason:
            return EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                score=None,
                skipped_reason=skipped_reason,
            )
        config = self.config.kinematics
        if not (
            config.end_effector_link
            and config.joint_names
            and config.eef_position_indices
        ):
            return EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                score=None,
                skipped_reason="eef_pose_missing",
            )
        try:
            result = _pinocchio_fk_detail(states, config)
        except Exception:
            return EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                score=None,
                skipped_reason="fk_failed",
            )
        count = int(np.sum(result["errors"] > config.position_tolerance))
        return self._count_status(
            count,
            len(states),
            critical_rate=0.05,
            critical=True,
        )

    @staticmethod
    def _normalize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if values.size == 0 or reference.size == 0:
            return values.astype(np.float64, copy=True)
        center = np.median(reference, axis=0)
        low = np.quantile(reference, 0.01, axis=0)
        high = np.quantile(reference, 0.99, axis=0)
        robust_span = high - low
        full_span = np.max(reference, axis=0) - np.min(reference, axis=0)
        scale = np.where(robust_span > 1e-9, robust_span, full_span)
        scale = np.where(scale > 1e-9, scale, 1.0)
        return (values - center) / scale

    @staticmethod
    def _without_indices(values: np.ndarray, ignored_indices: list[int]) -> np.ndarray:
        ignored = set(ignored_indices)
        columns = [index for index in range(values.shape[1]) if index not in ignored]
        return values[:, columns]

    def _sudden_change_detail(
        self,
        episode_index: int,
        states: np.ndarray,
        actions: np.ndarray,
        normalized_states: np.ndarray,
        normalized_actions: np.ndarray,
    ) -> FilterDetail:
        state_result = detect_sudden_changes(
            self._without_indices(normalized_states, self.config.gripper_indices),
            self._sudden_config(),
        )
        action_result = detect_sudden_changes(
            self._without_indices(normalized_actions, self.config.gripper_indices),
            self._sudden_config(),
        )
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
                "normalized_by": "dataset_q01_q99",
            },
            findings=self._finding_list(len(rows), "sudden_change", "Detected sudden change frames."),
        )

    def _state_action_alignment_detail(
        self,
        episode_index: int,
        states: np.ndarray,
        actions: np.ndarray,
        normalized_states: np.ndarray,
        normalized_actions: np.ndarray,
    ) -> FilterDetail:
        result = detect_state_action_trend_alignment(
            normalized_states,
            normalized_actions,
            self._alignment_config(),
        )
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
                "normalized_by": "dataset_q01_q99",
            },
            findings=self._finding_list(
                len(result.flagged_dimensions),
                "state_action_alignment",
                "Detected state/action trend desync dimension(s).",
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
            findings=self._finding_list(len(result.flagged_frames), "extreme_value", "Detected out-of-bounds frames."),
        )

    def _kinematic_detail(self, episode_index: int) -> FilterDetail:
        if importlib.util.find_spec("pinocchio") is None:
            return self._skipped_detail(
                "kinematic_consistency",
                episode_index,
                "backend_missing",
                "Pinocchio not installed; kinematic consistency unavailable.",
            )
        if not self.config.kinematics.urdf_path:
            return self._skipped_detail(
                "kinematic_consistency",
                episode_index,
                "urdf_missing",
                "Please import a URDF file first.",
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
                "Logged EEF pose indices not configured.",
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
                f"Pinocchio FK failed: {error}",
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
            findings=self._finding_list(len(rows), "kinematic_consistency", "Detected FK vs logged EEF mismatch frames."),
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
                    message="Orientation alignment correction matrix not configured.",
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

    def _count_status(
        self,
        count: int,
        total: int,
        critical_rate: float,
        critical: bool,
    ) -> EpisodeFilterStageStatus:
        rate = count / total if total > 0 else 1.0
        score = max(0.0, min(1.0, 1.0 - rate / critical_rate))
        severity = (
            "critical"
            if critical and count and rate >= critical_rate
            else "warning"
            if count
            else "none"
        )
        return EpisodeFilterStageStatus(
            count=count,
            status="review" if count else "passed",
            score=score,
            severity=severity,
        )

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
        return [FilterFinding(code=code, severity="warn", message=f"{message} Count: {count}")]

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
