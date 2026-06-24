from __future__ import annotations

import math
from statistics import median

from robot_data_studio.lerobot.models import EpisodeFrame
from robot_data_studio.lerobot.reader import LeRobotDatasetReader

from .models import CleaningConfig, EpisodeQualityResult, QualityFinding, utc_now


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


class EpisodeQualityScorer:
    """A local scorer adapter shaped after RoboticsData/score_lerobot_episodes output."""

    scorer_version = "score_lerobot_episodes-compatible-v1"

    def score_dataset(
        self,
        reader: LeRobotDatasetReader,
        config: CleaningConfig,
    ) -> list[EpisodeQualityResult]:
        episodes = reader.list_episodes()
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
        score = sum(per_attribute_scores.values()) / len(per_attribute_scores)
        return EpisodeQualityResult(
            episode_index=episode_index,
            score=score,
            status=_status_for_score(score, config),
            source="auto",
            per_attribute_scores=per_attribute_scores,
            findings=_findings(per_attribute_scores, has_video),
            updated_at=utc_now(),
        )
