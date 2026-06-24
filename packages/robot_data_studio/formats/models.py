from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from robot_data_studio.lerobot.models import DatasetMetadata, EpisodeFrame, EpisodeSummary


class FormatInfo(BaseModel):
    id: str
    label: str
    profile: str
    can_import: bool
    can_export: bool
    requires_extra: bool = False


class ExportResult(BaseModel):
    output_path: Path
    report_path: Path
    format: str
    episode_count: int


class DatasetAdapter(Protocol):
    root: Path

    @classmethod
    def probe(cls, root: str | Path) -> bool: ...

    def metadata(self) -> DatasetMetadata: ...

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]: ...

    def episode(self, episode_index: int) -> EpisodeSummary: ...

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]: ...


def conversion_report(
    *,
    source: DatasetMetadata,
    target_format: str,
    episode_indexes: list[int],
    output_path: Path,
    field_mapping: dict[str, str],
    warnings: list[str] | None = None,
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_format": source.format,
        "source_version": source.version,
        "target_format": target_format,
        "episode_indexes": episode_indexes,
        "episode_count": len(episode_indexes),
        "output_path": str(output_path),
        "field_mapping": field_mapping,
        "warnings": warnings or [],
        "backend": backend or {},
    }
