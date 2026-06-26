from __future__ import annotations

import math
from statistics import median

from robot_data_studio.lerobot.models import EpisodeFrame
from robot_data_studio.lerobot.reader import LeRobotDatasetReader

from .models import CleaningConfig, EpisodeQualityResult, QualityFinding, VlmEvaluation, utc_now
from .vlm import VlmEvaluationError, VlmTaskSuccessEvaluator


def _distance(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _path_efficiency(values: list[list[float]]) -> float:
    if len(values) < 2:
        return 0.0
    path_length = sum(_distance(values[index - 1], values[index]) for index in range(1, len(values)))
    if path_length <= 1e-9:
        return 0.0
    return _clamp_score(_distance(values[0], values[-1]) / path_length)


def _smoothness(values: list[list[float]]) -> float:
    if len(values) < 3:
        return 0.0
    second_derivatives = []
    for index in range(2, len(values)):
        previous = values[index - 2]
        current = values[index - 1]
        next_value = values[index]
        if len(previous) != len(current) or len(current) != len(next_value):
            continue
        second_derivatives.append(
            math.sqrt(
                sum(
                    (next_value[axis] - 2 * current[axis] + previous[axis]) ** 2
                    for axis in range(len(current))
                )
            )
        )
    if not second_derivatives:
        return 0.0
    mean_acceleration = sum(second_derivatives) / len(second_derivatives)
    return _clamp_score(1.0 / (1.0 + mean_acceleration / 20.0))


def _runtime_score(duration: float, nominal_duration: float) -> float:
    if nominal_duration <= 0:
        return 1.0
    relative_error = abs(duration - nominal_duration) / nominal_duration
    return _clamp_score(1.0 - relative_error)


def _tracking_score(frames: list[EpisodeFrame]) -> float:
    errors = [
        _distance(frame.action, frame.observation_state)
        for frame in frames
        if frame.action and frame.observation_state and len(frame.action) == len(frame.observation_state)
    ]
    if not errors:
        return 0.0
    mean_error = sum(errors) / len(errors)
    return _clamp_score(1.0 / (1.0 + mean_error / 100.0))


def _status_for_score(score: float, config: CleaningConfig) -> str:
    if score >= config.pass_threshold:
        return "passed"
    if score >= config.review_threshold:
        return "review"
    return "excluded"


def _status_for_episode(
    score: float,
    config: CleaningConfig,
    vlm_evaluation: VlmEvaluation | None,
) -> str:
    status = _status_for_score(score, config)
    if vlm_evaluation and not vlm_evaluation.success and status == "passed":
        return "review"
    return status


def _configured_weight(config: CleaningConfig, key: str) -> float:
    value = config.quality_weights.get(key, 1.0)
    if not math.isfinite(value):
        return 1.0
    return max(0.0, value)


def _weighted_score(scores: dict[str, float], config: CleaningConfig) -> float:
    weighted_total = 0.0
    weight_total = 0.0
    for stage_id in config.enabled_filter_stages:
        if stage_id not in scores:
            continue
        weight = _configured_weight(config, stage_id)
        if weight <= 0:
            continue
        weighted_total += scores[stage_id] * weight
        weight_total += weight
    if config.vlm.enabled and "task_success" in scores:
        weight = _configured_weight(config, "task_success")
        if weight > 0:
            weighted_total += scores["task_success"] * weight
            weight_total += weight
    if weight_total <= 0:
        fallback_scores = [
            score for name, score in scores.items() if name != "task_success" or config.vlm.enabled
        ]
        return sum(fallback_scores) / len(fallback_scores) if fallback_scores else 0.0
    return weighted_total / weight_total


def _findings(scores: dict[str, float], has_video: bool) -> list[QualityFinding]:
    findings = []
    if not has_video:
        findings.append(
            QualityFinding(
                code="video_missing",
                severity="warn",
                message="No video stream is available for visual scoring.",
            )
        )
    for name, score in scores.items():
        if score < 0.5:
            findings.append(
                QualityFinding(
                    code=f"low_{name}",
                    severity="warn",
                    message=f"{name.replace('_', ' ').title()} score is low ({score:.2f}).",
                )
            )
    return findings


def _deterministic_findings(scores: dict[str, float], config: CleaningConfig) -> list[QualityFinding]:
    messages = {
        "sudden_change": ("action_jump", "Sudden motion or action changes require review."),
        "state_action_alignment": ("time_sync", "State/action alignment score is low."),
        "extreme_value": ("extreme_value", "Extreme values are outside the expected range."),
        "kinematic_consistency": ("kinematic_consistency", "Kinematic consistency score is low."),
        "orientation_alignment": ("orientation_alignment", "Orientation alignment score is low."),
    }
    findings: list[QualityFinding] = []
    for stage_id in config.enabled_filter_stages:
        score = scores.get(stage_id)
        if score is None or score >= 0.5:
            continue
        code, message = messages[stage_id]
        findings.append(QualityFinding(code=code, severity="warn", message=message))
    return findings


def _vlm_findings(evaluation: VlmEvaluation | None, error: str | None) -> list[QualityFinding]:
    if error:
        return [
            QualityFinding(
                code="vlm_unavailable",
                severity="warn",
                message=error,
            )
        ]
    if evaluation and not evaluation.success:
        return [
            QualityFinding(
                code="vlm_failed",
                severity="warn",
                message=evaluation.reason or "VLM task success check failed.",
            )
        ]
    return []


class EpisodeQualityScorer:
    """A local scorer adapter shaped after RoboticsData/score_lerobot_episodes output."""

    scorer_version = "score_lerobot_episodes-compatible-v1"

    def __init__(self, vlm_evaluator: VlmTaskSuccessEvaluator | None = None) -> None:
        self._vlm_evaluator = vlm_evaluator or VlmTaskSuccessEvaluator()

    def score_dataset(
        self,
        reader: LeRobotDatasetReader,
        config: CleaningConfig,
        episode_indexes: list[int] | None = None,
    ) -> list[EpisodeQualityResult]:
        episodes = reader.list_episodes()
        if episode_indexes is not None:
            selected = set(episode_indexes)
            episodes = [episode for episode in episodes if episode.episode_index in selected]
        nominal_duration = median([episode.duration_seconds for episode in episodes]) if episodes else 0.0
        return [
            self.score_episode(reader, episode.episode_index, episode.duration_seconds, nominal_duration, config)
            for episode in episodes
        ]

    def score_episode(
        self,
        reader: LeRobotDatasetReader,
        episode_index: int,
        duration_seconds: float,
        nominal_duration: float,
        config: CleaningConfig,
    ) -> EpisodeQualityResult:
        frames = reader.read_episode_frames(episode_index)
        actions = [frame.action for frame in frames]
        states = [frame.observation_state for frame in frames]
        has_video = bool(reader.episode(episode_index).video_files)
        per_attribute_scores = {
            "visual_clarity": 1.0 if has_video else 0.0,
            "smoothness": _smoothness(actions),
            "path_efficiency": _path_efficiency(states),
            "runtime": _runtime_score(duration_seconds, nominal_duration),
            "actuator_saturation": _tracking_score(frames),
        }
        per_attribute_scores.update(
            {
                "sudden_change": per_attribute_scores["smoothness"],
                "state_action_alignment": per_attribute_scores["actuator_saturation"],
                "extreme_value": min(
                    per_attribute_scores["smoothness"],
                    per_attribute_scores["runtime"],
                ),
                "kinematic_consistency": per_attribute_scores["path_efficiency"],
                "orientation_alignment": per_attribute_scores["visual_clarity"],
            }
        )
        vlm_evaluation = None
        vlm_error = None
        if config.vlm.enabled:
            try:
                vlm_evaluation = self._vlm_evaluator.evaluate(reader, episode_index, config.vlm)
                per_attribute_scores["task_success"] = vlm_evaluation.score
            except VlmEvaluationError as error:
                vlm_error = str(error)
        score = _weighted_score(per_attribute_scores, config)
        return EpisodeQualityResult(
            episode_index=episode_index,
            score=score,
            status=_status_for_episode(score, config, vlm_evaluation),
            source="auto",
            per_attribute_scores=per_attribute_scores,
            findings=[
                *_findings(per_attribute_scores, has_video),
                *_deterministic_findings(per_attribute_scores, config),
                *_vlm_findings(vlm_evaluation, vlm_error),
            ],
            updated_at=utc_now(),
        )
