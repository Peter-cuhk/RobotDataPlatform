from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from robot_data_studio.formats.models import DatasetAdapter
from robot_data_studio.lerobot.models import EpisodeSummary

from .filter_models import FilterFinding, MetadataCompletenessConfig

CHECK_LABELS: dict[str, str] = {
    "task_description": "Task description",
    "camera_naming": "Camera naming",
    "resolution": "Resolution",
    "multi_view": "Multi-view",
    "action_field": "Action field",
    "episode_integrity": "Episode integrity",
}


@dataclass
class MetadataCompletenessResult:
    dataset_findings: list[FilterFinding] = field(default_factory=list)
    episode_findings: dict[int, list[FilterFinding]] = field(default_factory=dict)
    table_rows: dict[int, list[dict[str, object]]] = field(default_factory=dict)

    def findings_for_episode(self, episode_index: int) -> list[FilterFinding]:
        return [*self.dataset_findings, *self.episode_findings.get(episode_index, [])]

    def count_for_episode(self, episode_index: int) -> int:
        return len(self.findings_for_episode(episode_index))

    def to_report_payload(self) -> dict[str, object]:
        return {
            "dataset_findings": [finding.model_dump() for finding in self.dataset_findings],
            "episodes": {
                str(episode_index): [finding.model_dump() for finding in findings]
                for episode_index, findings in sorted(self.episode_findings.items())
            },
        }


def _feature_shape(features: dict, key: str) -> tuple[int, ...] | None:
    feature = features.get(key, {})
    shape = feature.get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    return tuple(int(item) for item in shape)


def _feature_names(features: dict, key: str) -> list[str] | None:
    feature = features.get(key, {})
    names = feature.get("names")
    if isinstance(names, dict):
        motors = names.get("motors")
        if isinstance(motors, list):
            return [str(item) for item in motors]
        return [str(item) for item in names.values()]
    if isinstance(names, list):
        return [str(item) for item in names]
    return None


def _state_action_dims(features: dict) -> tuple[int | None, int | None]:
    state_shape = _feature_shape(features, "observation.state")
    action_shape = _feature_shape(features, "action")
    state_dim = state_shape[-1] if state_shape else None
    action_dim = action_shape[-1] if action_shape else None
    return state_dim, action_dim


def _video_resolution(features: dict, video_key: str) -> tuple[int, int] | None:
    shape = _feature_shape(features, video_key)
    if shape is None or len(shape) < 2:
        return None
    height = int(shape[-2])
    width = int(shape[-1])
    return height, width


def _inspection_row(
    *,
    check: str,
    scope: str,
    status: str,
    expected: str,
    value: str,
    detail: str | None = None,
) -> dict[str, object]:
    return {
        "kind": "inspection",
        "check": check,
        "scope": scope,
        "status": status,
        "label": CHECK_LABELS[check],
        "expected": expected,
        "value": value,
        "detail": detail,
    }


def _inspect_task_description(
    episode: EpisodeSummary,
    config: MetadataCompletenessConfig,
) -> tuple[dict[str, object], FilterFinding | None]:
    tasks = [task.strip() for task in episode.tasks if str(task).strip()]
    value = "; ".join(tasks) if tasks else "(empty)"
    expected = "Non-empty task description"
    if not config.require_task_description:
        return (
            _inspection_row(
                check="task_description",
                scope="episode",
                status="passed",
                expected="Task description optional",
                value=value,
            ),
            None,
        )
    if tasks:
        return (
            _inspection_row(
                check="task_description",
                scope="episode",
                status="passed",
                expected=expected,
                value=value,
            ),
            None,
        )
    return (
        _inspection_row(
            check="task_description",
            scope="episode",
            status="warning",
            expected=expected,
            value=value,
        ),
        FilterFinding(
            code="task_description_missing",
            severity="warn",
            message=f"Episode {episode.episode_index} has no task description.",
        ),
    )


def _inspect_camera_naming(
    video_keys: list[str],
    config: MetadataCompletenessConfig,
) -> tuple[dict[str, object], FilterFinding | None]:
    prefix = config.expected_camera_prefix
    expected = f"Keys use prefix {prefix!r}"
    if not video_keys:
        return (
            _inspection_row(
                check="camera_naming",
                scope="dataset",
                status="warning",
                expected=expected,
                value="(no camera keys)",
            ),
            FilterFinding(
                code="camera_naming",
                severity="warn",
                message="Dataset defines no video camera keys.",
            ),
        )
    short_names = [key.removeprefix(prefix) if key.startswith(prefix) else key for key in video_keys]
    value = ", ".join(short_names)
    invalid = [key for key in video_keys if not key.startswith(prefix)]
    if invalid:
        return (
            _inspection_row(
                check="camera_naming",
                scope="dataset",
                status="warning",
                expected=expected,
                value=value,
                detail=f"Non-conforming: {', '.join(invalid)}",
            ),
            FilterFinding(
                code="camera_naming",
                severity="warn",
                message=(
                    f"Camera keys should use prefix {prefix!r}; "
                    f"found non-conforming keys: {', '.join(invalid)}."
                ),
            ),
        )
    return (
        _inspection_row(
            check="camera_naming",
            scope="dataset",
            status="passed",
            expected=expected,
            value=value,
        ),
        None,
    )


def _inspect_resolution(
    features: dict,
    video_keys: list[str],
) -> tuple[dict[str, object], list[FilterFinding]]:
    expected = "Resolvable and consistent camera resolutions"
    findings: list[FilterFinding] = []
    if not video_keys:
        return (
            _inspection_row(
                check="resolution",
                scope="dataset",
                status="passed",
                expected=expected,
                value="(no cameras)",
            ),
            findings,
        )
    resolutions: dict[str, tuple[int, int]] = {}
    unknown: list[str] = []
    parts: list[str] = []
    for key in video_keys:
        resolution = _video_resolution(features, key)
        short = key.split(".")[-1]
        if resolution is None:
            unknown.append(key)
            parts.append(f"{short}=(unknown)")
            continue
        resolutions[key] = resolution
        height, width = resolution
        parts.append(f"{short}={height}x{width}")
    value = ", ".join(parts)
    status = "passed"
    detail = None
    if unknown:
        status = "warning"
        detail = f"Unknown: {', '.join(unknown)}"
        findings.append(
            FilterFinding(
                code="resolution_unknown",
                severity="warn",
                message=f"Missing video resolution metadata for: {', '.join(unknown)}.",
            )
        )
    unique = set(resolutions.values())
    if len(resolutions) > 1 and len(unique) > 1:
        status = "warning"
        mismatch = ", ".join(
            f"{key.split('.')[-1]}={height}x{width}"
            for key, (height, width) in resolutions.items()
        )
        detail = f"Inconsistent: {mismatch}"
        findings.append(
            FilterFinding(
                code="resolution_inconsistent",
                severity="warn",
                message="Camera resolutions are inconsistent across views.",
            )
        )
    return (
        _inspection_row(
            check="resolution",
            scope="dataset",
            status=status,
            expected=expected,
            value=value,
            detail=detail,
        ),
        findings,
    )


def _inspect_multi_view(
    video_keys: list[str],
    config: MetadataCompletenessConfig,
) -> tuple[dict[str, object], FilterFinding | None]:
    count = len(video_keys)
    expected = f"At least {config.min_camera_count} camera view(s)"
    value = f"{count} camera(s)"
    if count < config.min_camera_count:
        return (
            _inspection_row(
                check="multi_view",
                scope="dataset",
                status="info",
                expected=expected,
                value=value,
            ),
            FilterFinding(
                code="single_view",
                severity="info",
                message=(
                    f"Dataset has {count} camera view(s); "
                    f"expected at least {config.min_camera_count}."
                ),
            ),
        )
    return (
        _inspection_row(
            check="multi_view",
            scope="dataset",
            status="passed",
            expected=expected,
            value=value,
        ),
        None,
    )


def _inspect_action_field(features: dict) -> tuple[dict[str, object], list[FilterFinding]]:
    expected = "action feature present with names and matching state/action dims"
    findings: list[FilterFinding] = []
    if "action" not in features:
        return (
            _inspection_row(
                check="action_field",
                scope="dataset",
                status="warning",
                expected=expected,
                value="action feature missing",
            ),
            [
                FilterFinding(
                    code="action_missing",
                    severity="warn",
                    message="Dataset metadata is missing the action feature.",
                )
            ],
        )
    action_names = _feature_names(features, "action")
    state_dim, action_dim = _state_action_dims(features)
    name_count = len(action_names) if action_names else 0
    value_parts = [
        f"action dim={action_dim if action_dim is not None else '?'}",
        f"names={name_count if action_names is not None else 'missing'}",
        f"state dim={state_dim if state_dim is not None else '?'}",
    ]
    value = ", ".join(value_parts)
    status = "passed"
    detail = None
    if action_names is None:
        status = "warning"
        detail = "Action names missing"
        findings.append(
            FilterFinding(
                code="action_names_missing",
                severity="warn",
                message="Action feature is missing motor/dimension names.",
            )
        )
    if state_dim is not None and action_dim is not None and state_dim != action_dim:
        status = "warning"
        detail = f"state={state_dim}, action={action_dim}"
        findings.append(
            FilterFinding(
                code="dimension_metadata_mismatch",
                severity="warn",
                message=(
                    f"observation.state dimension ({state_dim}) does not match "
                    f"action dimension ({action_dim}) in metadata."
                ),
            )
        )
    return (
        _inspection_row(
            check="action_field",
            scope="dataset",
            status=status,
            expected=expected,
            value=value,
            detail=detail,
        ),
        findings,
    )


def _inspect_episode_integrity(
    root: Path,
    episode: EpisodeSummary,
) -> tuple[dict[str, object], list[FilterFinding]]:
    expected = "length > 0 and all declared video files exist"
    findings: list[FilterFinding] = []
    missing: list[str] = []
    present = 0
    total = len(episode.video_files)
    for camera, relative_path in episode.video_files.items():
        path = root / relative_path
        if path.is_file():
            present += 1
            continue
        missing.append(f"{camera}: {relative_path}")
    value = f"{episode.length} frames, {present}/{total} videos present"
    status = "passed"
    detail = None
    if episode.length <= 0:
        status = "warning"
        detail = f"length={episode.length}"
        findings.append(
            FilterFinding(
                code="episode_incomplete",
                severity="warn",
                message=f"Episode {episode.episode_index} has zero or negative length.",
            )
        )
    if missing:
        status = "warning"
        detail = "; ".join(missing)
        for camera, relative_path in episode.video_files.items():
            path = root / relative_path
            if path.is_file():
                continue
            findings.append(
                FilterFinding(
                    code="video_file_missing",
                    severity="warn",
                    message=(
                        f"Episode {episode.episode_index} is missing video file for camera "
                        f"{camera!r}."
                    ),
                )
            )
    return (
        _inspection_row(
            check="episode_integrity",
            scope="episode",
            status=status,
            expected=expected,
            value=value,
            detail=detail,
        ),
        findings,
    )


def build_metadata_inspection(
    reader: DatasetAdapter,
    episode_index: int,
    config: MetadataCompletenessConfig | None = None,
) -> list[dict[str, object]]:
    config = config or MetadataCompletenessConfig()
    metadata = reader.metadata()
    episode = reader.episode(episode_index)

    task_row, _task_finding = _inspect_task_description(episode, config)
    camera_row, _camera_finding = _inspect_camera_naming(metadata.video_keys, config)
    resolution_row, _resolution_findings = _inspect_resolution(metadata.features, metadata.video_keys)
    multi_view_row, _multi_view_finding = _inspect_multi_view(metadata.video_keys, config)
    action_row, _action_findings = _inspect_action_field(metadata.features)
    integrity_row, _integrity_findings = _inspect_episode_integrity(reader.root, episode)

    return [
        task_row,
        camera_row,
        resolution_row,
        multi_view_row,
        action_row,
        integrity_row,
    ]


def inspection_summary(rows: list[dict[str, object]]) -> dict[str, int]:
    passed = sum(1 for row in rows if row.get("status") == "passed")
    warnings = sum(1 for row in rows if row.get("status") == "warning")
    infos = sum(1 for row in rows if row.get("status") == "info")
    return {
        "total_checks": len(rows),
        "passed": passed,
        "warnings": warnings,
        "infos": infos,
    }


def analyze_metadata_completeness(
    reader: DatasetAdapter,
    config: MetadataCompletenessConfig | None = None,
) -> MetadataCompletenessResult:
    config = config or MetadataCompletenessConfig()
    metadata = reader.metadata()
    episodes = reader.list_episodes()
    result = MetadataCompletenessResult(
        episode_findings={episode.episode_index: [] for episode in episodes},
        table_rows={episode.episode_index: [] for episode in episodes},
    )

    camera_row, camera_finding = _inspect_camera_naming(metadata.video_keys, config)
    if camera_finding:
        result.dataset_findings.append(camera_finding)

    _, resolution_findings = _inspect_resolution(metadata.features, metadata.video_keys)
    result.dataset_findings.extend(resolution_findings)

    _, multi_view_finding = _inspect_multi_view(metadata.video_keys, config)
    if multi_view_finding:
        result.dataset_findings.append(multi_view_finding)

    _, action_findings = _inspect_action_field(metadata.features)
    result.dataset_findings.extend(action_findings)

    for episode in episodes:
        inspection_rows = build_metadata_inspection(reader, episode.episode_index, config)
        result.table_rows[episode.episode_index] = inspection_rows

        task_row, task_finding = _inspect_task_description(episode, config)
        if task_finding:
            result.episode_findings[episode.episode_index].append(task_finding)

        _, integrity_findings = _inspect_episode_integrity(reader.root, episode)
        result.episode_findings[episode.episode_index].extend(integrity_findings)

    return result


def build_metadata_inspection_report(
    reader: DatasetAdapter,
    config: MetadataCompletenessConfig | None = None,
) -> dict[str, object]:
    config = config or MetadataCompletenessConfig()
    result = analyze_metadata_completeness(reader, config)
    return {
        **result.to_report_payload(),
        "inspection": {
            str(episode_index): rows
            for episode_index, rows in sorted(result.table_rows.items())
        },
    }
