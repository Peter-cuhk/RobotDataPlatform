from __future__ import annotations

import importlib.util
from collections import Counter
from collections.abc import Callable
from pathlib import Path
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
    VisualQualityDetail,
    VisualQualityEvidenceFrame,
    VisualQualityIncident,
    VisualQualityMetricSample,
)
from robot_data_studio.quality.metadata_completeness import (
    analyze_metadata_completeness,
    build_metadata_inspection,
    inspection_summary,
)
from robot_data_studio.quality.qwen_filters import (
    ExtremeValueConfig,
    SuddenChangeConfig,
    detect_extreme_values,
    detect_sudden_changes,
)
from robot_data_studio.quality.time_sync import analyze_time_sync
from robot_data_studio.quality.visual_quality import (
    VisualQualityResult,
    VisualQualityRow,
    aggregate_visual_quality_incidents,
    analyze_sampled_frames,
    has_signal_motion,
    sample_video_frames,
)


STAGE_LABELS: dict[FilterStageId, str] = {
    "visual_quality": "Visual quality",
    "sudden_change": "Sudden change",
    "state_action_alignment": "Time sync",
    "extreme_value": "Extreme value",
    "kinematic_consistency": "Kinematic consistency",
    "orientation_alignment": "Orientation alignment",
    "metadata_completeness": "Metadata completeness",
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
        self._sampled_video_cache: dict[tuple[Path, float, float], list[np.ndarray]] = {}
        self._episode_frame_cache: dict[int, list] = {}

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
        reference_stats = {
            "state": self._normalization_stats(reference["state"]),
            "action": self._normalization_stats(reference["action"]),
        }
        extreme_bounds = {
            "state": self._extreme_bounds(reference["state"]),
            "action": self._extreme_bounds(reference["action"]),
        }
        metadata_result = analyze_metadata_completeness(
            self.reader,
            self.config.metadata_completeness,
        )
        episode_summaries = []
        totals = {stage_id: 0 for stage_id in STAGE_LABELS}
        for index, episode in enumerate(episodes, start=1):
            states = states_by_episode[episode.episode_index]
            actions = actions_by_episode[episode.episode_index]
            integrity_findings = integrity_by_episode[episode.episode_index]
            status = (
                self._integrity_failed_statuses()
                if integrity_findings
                else self._stage_statuses(
                    episode,
                    states,
                    actions,
                    reference,
                    reference_stats,
                    extreme_bounds,
                )
            )
            status["metadata_completeness"] = self._metadata_completeness_status(
                metadata_result,
                episode.episode_index,
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
        frames = self._read_episode_frames(episode_index)
        states = np.asarray([frame.observation_state for frame in frames], dtype=np.float64)
        actions = np.asarray([frame.action for frame in frames], dtype=np.float64)
        states_by_episode, actions_by_episode, _integrity_by_episode = self._episode_arrays()
        reference = self._reference_arrays(states_by_episode, actions_by_episode)
        normalized_states = self._normalize(states, reference["state"])
        normalized_actions = self._normalize(actions, reference["action"])
        if stage_id == "visual_quality":
            return self._visual_quality_detail(episode_index, states, actions)
        if stage_id == "sudden_change":
            return self._sudden_change_detail(
                episode_index,
                states,
                actions,
                normalized_states,
                normalized_actions,
            )
        if stage_id == "state_action_alignment":
            return self._time_sync_detail(
                episode_index,
                frames,
            )
        if stage_id == "extreme_value":
            return self._extreme_value_detail(episode_index, states, actions, reference)
        if stage_id == "kinematic_consistency":
            return self._kinematic_detail(episode_index)
        if stage_id == "orientation_alignment":
            return self._orientation_detail(episode_index, states)
        if stage_id == "metadata_completeness":
            return self._metadata_completeness_detail(episode_index)
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
            frames = self._read_episode_frames(episode.episode_index)
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
        episode,
        states: np.ndarray,
        actions: np.ndarray,
        reference: dict[str, np.ndarray],
        reference_stats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        extreme_bounds: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[FilterStageId, EpisodeFilterStageStatus]:
        if reference_stats is None:
            reference_stats = {
                "state": self._normalization_stats(reference["state"]),
                "action": self._normalization_stats(reference["action"]),
            }
        if extreme_bounds is None:
            extreme_bounds = {
                "state": self._extreme_bounds(reference["state"]),
                "action": self._extreme_bounds(reference["action"]),
            }
        normalized_states = self._normalize_with_stats(states, reference_stats["state"])
        normalized_actions = self._normalize_with_stats(actions, reference_stats["action"])
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
        time_sync = analyze_time_sync(
            episode,
            self._read_episode_frames(episode.episode_index),
            self.reader.metadata().fps,
            self.config.time_sync,
        )
        extreme = detect_extreme_values(
            {"state": states, "action": actions},
            self._extreme_config(),
            reference_values_by_key=reference,
            reference_bounds_by_key=extreme_bounds,
        )
        return {
            "visual_quality": self._visual_quality_status(episode, states, actions),
            "sudden_change": self._count_status(
                len(sudden),
                len(states),
                critical_rate=0.02,
                critical=False,
            ),
            "state_action_alignment": self._count_status(
                time_sync.issue_count,
                max(len(states), 1),
                critical_rate=0.05,
                critical=False,
                score=time_sync.score,
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

    def _visual_quality_status(
        self,
        episode,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> EpisodeFilterStageStatus:
        result, skipped_reason = self._visual_quality_result(episode, states, actions)
        if skipped_reason:
            return EpisodeFilterStageStatus(
                count=0,
                status="skipped",
                score=None,
                skipped_reason=skipped_reason,
            )
        issue_rate = result.issue_count / max(result.frame_count, 1)
        severity = (
            "critical"
            if result.issue_count and (issue_rate >= 0.10 or result.critical_issue)
            else "warning"
            if result.issue_count
            else "none"
        )
        return EpisodeFilterStageStatus(
            count=result.issue_count,
            status="review" if issue_rate >= 0.02 else "passed",
            score=result.score,
            severity=severity,
        )

    def _visual_quality_result(
        self,
        episode,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> tuple[VisualQualityResult, str | None]:
        video_files = getattr(episode, "video_files", {}) or {}
        if not video_files:
            return VisualQualityResult(frame_count=0, issue_count=0, score=0.0), "video_missing"
        config = self.config.visual_quality
        metadata = self.reader.metadata()
        has_motion = has_signal_motion(states, actions)
        combined = VisualQualityResult(frame_count=0, issue_count=0, score=1.0)
        existing_videos = 0
        for camera, relative_path in video_files.items():
            path = self.reader.root / relative_path
            if not path.is_file():
                combined.rows.append(
                    _visual_row(None, None, camera, "video_missing", str(relative_path), "exists")
                )
                combined.issue_count += 1
                combined.issue_counts["video_missing"] = combined.issue_counts.get("video_missing", 0) + 1
                continue
            existing_videos += 1
            start_seconds = float(episode.video_start_seconds.get(camera, 0.0))
            end_seconds = float(
                episode.video_end_seconds.get(
                    camera,
                    start_seconds + episode.duration_seconds,
                )
            )
            duration_seconds = max(0.0, end_seconds - start_seconds)
            cache_key = (path, start_seconds, duration_seconds)
            frames = self._sampled_video_cache.get(cache_key)
            if frames is None:
                frames = sample_video_frames(
                    path,
                    config,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                )
                self._sampled_video_cache[cache_key] = frames
            if not frames:
                combined.rows.append(
                    _visual_row(None, None, camera, "decode_failed", "unavailable", "decodable")
                )
                combined.issue_count += 1
                combined.issue_counts["decode_failed"] = (
                    combined.issue_counts.get("decode_failed", 0) + 1
                )
                continue
            timestamps = [index / config.sample_fps for index in range(len(frames))]
            frame_indexes = [
                min(
                    max(episode.length - 1, 0),
                    round(timestamp * metadata.fps),
                )
                for timestamp in timestamps
            ]
            result = analyze_sampled_frames(
                frames,
                config,
                has_motion=has_motion,
                camera=camera,
                timestamps=timestamps,
                frame_indexes=frame_indexes,
            )
            combined.frame_count += result.frame_count
            combined.issue_count += result.issue_count
            combined.rows.extend(result.rows)
            combined.critical_issue = combined.critical_issue or result.critical_issue
            combined.metrics.update(result.metrics)
            for name, values in result.series.items():
                key = f"{camera}:{name}"
                combined.series[key] = values
            for issue, count in result.issue_counts.items():
                combined.issue_counts[issue] = combined.issue_counts.get(issue, 0) + count
        if existing_videos == 0:
            return combined, "video_missing"
        combined.camera_count = existing_videos
        denominator = max(combined.frame_count, 1)
        combined.score = max(0.0, 1.0 - combined.issue_count / denominator)
        if combined.issue_counts.get("video_missing", 0):
            combined.critical_issue = True
        return combined, None

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
    def _normalization_stats(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if reference.size == 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        center = np.median(reference, axis=0)
        low = np.quantile(reference, 0.01, axis=0)
        high = np.quantile(reference, 0.99, axis=0)
        robust_span = high - low
        full_span = np.max(reference, axis=0) - np.min(reference, axis=0)
        scale = np.where(robust_span > 1e-9, robust_span, full_span)
        scale = np.where(scale > 1e-9, scale, 1.0)
        return center, scale

    @staticmethod
    def _normalize_with_stats(
        values: np.ndarray,
        stats: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        center, scale = stats
        if values.size == 0 or center.size == 0 or scale.size == 0:
            return values.astype(np.float64, copy=True)
        return (values - center) / scale

    @staticmethod
    def _normalize(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return DatasetFilterService._normalize_with_stats(
            values,
            DatasetFilterService._normalization_stats(reference),
        )

    def _read_episode_frames(self, episode_index: int) -> list:
        frames = self._episode_frame_cache.get(episode_index)
        if frames is None:
            frames = self.reader.read_episode_frames(episode_index)
            self._episode_frame_cache[episode_index] = frames
        return frames

    def _extreme_bounds(self, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if reference.size == 0:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        config = self._extreme_config()
        low = np.quantile(reference, config.lower_quantile, axis=0)
        high = np.quantile(reference, config.upper_quantile, axis=0)
        return low, high

    @staticmethod
    def _without_indices(values: np.ndarray, ignored_indices: list[int]) -> np.ndarray:
        return values[:, DatasetFilterService._retained_indices(values, ignored_indices)]

    @staticmethod
    def _retained_indices(values: np.ndarray, ignored_indices: list[int]) -> list[int]:
        ignored = set(ignored_indices)
        return [index for index in range(values.shape[1]) if index not in ignored]

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
        state_dimensions = self._retained_indices(normalized_states, self.config.gripper_indices)
        action_dimensions = self._retained_indices(normalized_actions, self.config.gripper_indices)
        rows = []
        for source, result, dimensions in [
            ("state", state_result, state_dimensions),
            ("action", action_result, action_dimensions),
        ]:
            for event in result.events[:200]:
                original_dimension = (
                    dimensions[event.dimension]
                    if event.dimension < len(dimensions)
                    else event.dimension
                )
                rows.append(
                    {
                        "frame": event.peak_frame,
                        "start_frame": event.start_frame,
                        "end_frame": event.end_frame,
                        "source": source,
                        "dimension": original_dimension,
                        "metric": event.source_metric,
                        "ratio": round(event.severity_ratio, 4),
                        "residual_ratio": round(event.residual_ratio, 4),
                        "metric_ratio": round(event.metric_ratio, 4),
                        "reason": f"residual + {event.source_metric}",
                    }
                )
        thresholds = {}
        if state_result.dimension_scores:
            thresholds["state[0]"] = {
                "residual": state_result.dimension_scores[0].residual_threshold,
                "delta": state_result.dimension_scores[0].delta_threshold,
                "acceleration": state_result.dimension_scores[0].acceleration_threshold,
                "jerk": state_result.dimension_scores[0].jerk_threshold,
            }
        return FilterDetail(
            stage_id="sudden_change",
            episode_index=episode_index,
            title=STAGE_LABELS["sudden_change"],
            status="review" if rows else "passed",
            series={"raw": self._sample(states[:, 0]), "action": self._sample(actions[:, 0])},
            thresholds=thresholds,
            table_rows=rows,
            parameters={
                "median_windows": list(self._sudden_config().median_windows),
                "savgol_window": self._sudden_config().savgol_window,
                "residual_scale": self._sudden_config().residual_scale,
                "delta_scale": self._sudden_config().delta_scale,
                "event_merge_gap": self._sudden_config().event_merge_gap,
                "normalized_by": "dataset_q01_q99",
            },
            findings=self._finding_list(len(rows), "sudden_change", "Detected sudden change frames."),
        )

    def _time_sync_detail(
        self,
        episode_index: int,
        frames: list,
    ) -> FilterDetail:
        episode = self.reader.episode(episode_index)
        result = analyze_time_sync(
            episode,
            frames,
            self.reader.metadata().fps,
            self.config.time_sync,
        )
        rows = [
            {
                "issue": item.issue,
                "frame": item.frame,
                "camera": item.camera,
                "value": round(item.value, 6),
                "expected": round(item.expected, 6),
                "delta": round(item.delta, 6),
                "threshold": round(item.threshold, 6),
            }
            for item in result.rows
        ]
        return FilterDetail(
            stage_id="state_action_alignment",
            episode_index=episode_index,
            title=STAGE_LABELS["state_action_alignment"],
            status="review" if result.issue_count else "passed",
            series=result.series,
            thresholds={
                "time_sync": {
                    "timestamp_jitter_seconds": self.config.time_sync.timestamp_jitter_seconds,
                    "timestamp_jitter_ratio": self.config.time_sync.timestamp_jitter_ratio,
                    "duration_tolerance_seconds": self.config.time_sync.duration_tolerance_seconds,
                    "video_boundary_tolerance_seconds": (
                        self.config.time_sync.video_boundary_tolerance_seconds
                    ),
                }
            },
            table_rows=rows,
            parameters=self.config.time_sync.model_dump(),
            findings=self._finding_list(
                result.issue_count,
                "state_action_alignment",
                "Detected timestamp synchronization issue(s).",
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

    def _metadata_completeness_status(
        self,
        metadata_result,
        episode_index: int,
    ) -> EpisodeFilterStageStatus:
        count = metadata_result.count_for_episode(episode_index)
        return EpisodeFilterStageStatus(
            count=count,
            status="review" if count else "passed",
            score=None,
            severity="warning" if count else "none",
        )

    def _metadata_completeness_detail(self, episode_index: int) -> FilterDetail:
        result = analyze_metadata_completeness(
            self.reader,
            self.config.metadata_completeness,
        )
        findings = result.findings_for_episode(episode_index)
        inspection_rows = build_metadata_inspection(
            self.reader,
            episode_index,
            self.config.metadata_completeness,
        )
        return FilterDetail(
            stage_id="metadata_completeness",
            episode_index=episode_index,
            title=STAGE_LABELS["metadata_completeness"],
            status="review" if findings else "passed",
            table_rows=inspection_rows,
            parameters={
                **self.config.metadata_completeness.model_dump(),
                "summary": inspection_summary(inspection_rows),
            },
            findings=findings,
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

    def _visual_quality_detail(
        self,
        episode_index: int,
        states: np.ndarray,
        actions: np.ndarray,
    ) -> FilterDetail:
        episode = self.reader.episode(episode_index)
        result, skipped_reason = self._visual_quality_result(episode, states, actions)
        if skipped_reason:
            skipped = self._skipped_detail(
                "visual_quality",
                episode_index,
                skipped_reason,
                "No local video stream is available for visual quality scoring.",
            )
            return skipped.model_copy(
                update={
                    "table_rows": [
                        {
                            "camera": None,
                            "frame": None,
                            "timestamp": None,
                            "issue": skipped_reason,
                            "value": "unavailable",
                            "threshold": "available local video",
                        }
                    ],
                    "visual_quality": VisualQualityDetail(
                        sampled_frame_count=0,
                        camera_count=0,
                        issue_sample_count=0,
                        affected_camera_count=0,
                        episode_frame_count=episode.length,
                        episode_duration_seconds=episode.duration_seconds,
                    ),
                }
            )
        rows = [
            {
                "camera": row.camera,
                "frame": row.frame,
                "timestamp": round(row.timestamp, 4) if row.timestamp is not None else None,
                "end_frame": row.end_frame,
                "end_timestamp": (
                    round(row.end_timestamp, 4) if row.end_timestamp is not None else None
                ),
                "issue": row.issue,
                "value": row.value,
                "threshold": row.threshold,
            }
            for row in result.rows[:200]
        ]
        issue_summary = ", ".join(
            f"{name}: {count}" for name, count in sorted(result.issue_counts.items())
        )
        findings = (
            [
                FilterFinding(
                    code="visual_quality",
                    severity="warn",
                    message=f"Detected visual quality issue(s). {issue_summary}",
                )
            ]
            if result.issue_count
            else []
        )
        incidents = aggregate_visual_quality_incidents(
            result.rows,
            sample_interval_seconds=1 / self.config.visual_quality.sample_fps,
        )
        issue_samples = {
            (row.camera, row.frame)
            for row in result.rows
            if row.frame is not None and row.issue not in {"video_missing", "decode_failed"}
        }
        visual_quality = VisualQualityDetail(
            sampled_frame_count=result.frame_count,
            camera_count=result.camera_count,
            issue_sample_count=len(issue_samples),
            affected_camera_count=len({incident.camera for incident in incidents}),
            episode_frame_count=episode.length,
            episode_duration_seconds=episode.duration_seconds,
            incidents=[
                VisualQualityIncident(
                    id=(
                        f"{incident.camera}:{incident.issue}:"
                        f"{incident.start_frame}:{incident.end_frame}"
                    ),
                    camera=incident.camera,
                    issue=incident.issue,
                    start_frame=incident.start_frame,
                    end_frame=incident.end_frame,
                    start_timestamp=round(incident.start_timestamp, 4),
                    end_timestamp=round(incident.end_timestamp, 4),
                    sample_count=incident.sample_count,
                    worst_value=incident.worst_value,
                    threshold=incident.threshold,
                    representative_frames=[
                        VisualQualityEvidenceFrame(
                            frame=frame.frame,
                            timestamp=round(frame.timestamp, 4),
                        )
                        for frame in incident.representative_frames
                    ],
                )
                for incident in incidents
            ],
            metrics={
                camera: [
                    VisualQualityMetricSample(
                        frame=metric.frame,
                        timestamp=round(metric.timestamp, 4),
                        sharpness=metric.sharpness,
                        brightness=metric.brightness,
                        contrast=metric.contrast,
                    )
                    for metric in samples
                ]
                for camera, samples in result.metrics.items()
            },
        )
        return FilterDetail(
            stage_id="visual_quality",
            episode_index=episode_index,
            title=STAGE_LABELS["visual_quality"],
            status="review" if result.issue_count / max(result.frame_count, 1) >= 0.02 else "passed",
            series=result.series,
            thresholds={
                "visual_quality": {
                    "blur_laplacian": self.config.visual_quality.blur_laplacian_threshold,
                    "dark_mean": self.config.visual_quality.dark_mean_threshold,
                    "bright_mean": self.config.visual_quality.bright_mean_threshold,
                    "dark_global_mean": self.config.visual_quality.dark_global_mean_threshold,
                    "dark_global_p75": self.config.visual_quality.dark_global_p75_threshold,
                    "bright_global_mean": self.config.visual_quality.bright_global_mean_threshold,
                    "bright_global_p75": self.config.visual_quality.bright_global_p75_threshold,
                    "low_contrast_std": self.config.visual_quality.low_contrast_std_threshold,
                    "freeze_mse": self.config.visual_quality.freeze_mse_threshold,
                }
            },
            table_rows=rows,
            parameters=self.config.visual_quality.model_dump(),
            findings=findings,
            visual_quality=visual_quality,
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
        score: float | None = None,
    ) -> EpisodeFilterStageStatus:
        rate = count / total if total > 0 else 1.0
        computed_score = max(0.0, min(1.0, 1.0 - rate / critical_rate))
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
            score=computed_score if score is None else score,
            severity=severity,
        )

    def _sudden_config(self) -> SuddenChangeConfig:
        return SuddenChangeConfig(
            residual_min_threshold=0.02,
            acceleration_min_threshold=0.02,
            jerk_min_threshold=0.02,
        )

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


def _visual_row(
    frame: int | None,
    timestamp: float | None,
    camera: str,
    issue: str,
    value: float | str,
    threshold: float | str,
) -> VisualQualityRow:
    return VisualQualityRow(
        frame=frame,
        timestamp=timestamp,
        camera=camera,
        issue=issue,
        value=value,
        threshold=threshold,
    )


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
