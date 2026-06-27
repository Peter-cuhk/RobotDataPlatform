from __future__ import annotations

import json
from pathlib import Path

from robot_data_studio.lerobot.models import EpisodeSummary

from .models import CleaningConfig, CleaningSummary, EpisodeQualityResult, VlmSettings, utc_now

SCORER_VERSION = "quality-rules-v2"


class CleaningStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> CleaningSummary | None:
        if not self.path.is_file():
            return None
        return CleaningSummary.model_validate_json(self.path.read_text())

    def save(self, summary: CleaningSummary) -> CleaningSummary:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2))
        return summary


class VlmSettingsStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> VlmSettings:
        if not self.path.is_file():
            return VlmSettings()
        return VlmSettings.model_validate_json(self.path.read_text())

    def save(self, settings: VlmSettings) -> VlmSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(settings.__dict__, indent=2))
        return settings


def build_summary(
    episodes: list[EpisodeSummary],
    results: list[EpisodeQualityResult],
    config: CleaningConfig,
    scorer_version: str = SCORER_VERSION,
) -> CleaningSummary:
    by_index = {result.episode_index: result for result in results}
    complete_results = []
    for episode in episodes:
        result = by_index.get(episode.episode_index)
        if result is None:
            result = EpisodeQualityResult(
                episode_index=episode.episode_index,
                score=None,
                status="unscored",
                source="auto",
                per_attribute_scores={},
                findings=[],
                updated_at=utc_now(),
            )
        complete_results.append(result)
    complete_results.sort(key=lambda item: item.episode_index)
    return CleaningSummary(
        total=len(complete_results),
        passed_count=sum(1 for result in complete_results if result.status == "passed"),
        review_count=sum(1 for result in complete_results if result.status == "review"),
        excluded_count=sum(1 for result in complete_results if result.status == "excluded"),
        unscored_count=sum(1 for result in complete_results if result.status == "unscored"),
        results=complete_results,
        config=config,
        scorer_version=scorer_version,
    )
