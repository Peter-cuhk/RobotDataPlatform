from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median

import numpy as np

from robot_data_studio.formats.models import DatasetAdapter
from robot_data_studio.lerobot.models import EpisodeFrame, EpisodeSummary
from robot_data_studio.lerobot.reader import LeRobotDatasetReader
from robot_data_studio.quality.filter_models import FilterConfig, FilterSummary
from robot_data_studio.quality.filter_service import DatasetFilterService, infer_gripper_indices
from robot_data_studio.quality.models import CleaningConfig, EpisodeQualityResult
from robot_data_studio.quality.scorer import (
    EpisodeQualityScorer,
    _smoothness,
    _tracking_score,
)
try:
    from scripts.download_hf_dataset import download_dataset
except ModuleNotFoundError:
    from download_hf_dataset import download_dataset


FrameTransform = Callable[[list[EpisodeFrame]], list[EpisodeFrame]]


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = np.asarray(_average_ranks(left), dtype=np.float64)
    right_ranks = np.asarray(_average_ranks(right), dtype=np.float64)
    left_ranks -= np.mean(left_ranks)
    right_ranks -= np.mean(right_ranks)
    denominator = float(np.linalg.norm(left_ranks) * np.linalg.norm(right_ranks))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_ranks, right_ranks) / denominator)


class DatasetView:
    def __init__(
        self,
        reader: DatasetAdapter,
        episode_indexes: list[int],
        transform: FrameTransform | None = None,
    ) -> None:
        self.reader = reader
        self.root = reader.root
        self.episode_indexes = episode_indexes
        self.transform = transform

    def metadata(self):
        return self.reader.metadata()

    def list_episodes(self, limit: int | None = None) -> list[EpisodeSummary]:
        selected = set(self.episode_indexes)
        episodes = [
            episode
            for episode in self.reader.list_episodes()
            if episode.episode_index in selected
        ]
        return episodes[:limit] if limit is not None else episodes

    def episode(self, episode_index: int) -> EpisodeSummary:
        return self.reader.episode(episode_index)

    def read_episode_frames(self, episode_index: int) -> list[EpisodeFrame]:
        frames = self.reader.read_episode_frames(episode_index)
        return self.transform(frames) if self.transform else frames


def _replace_action(frame: EpisodeFrame, action: list[float]) -> EpisodeFrame:
    return frame.model_copy(update={"action": action})


def inject_sudden_change(frames: list[EpisodeFrame]) -> list[EpisodeFrame]:
    output = list(frames)
    if not output or not output[0].action:
        return output
    injected_count = max(4, int(np.ceil(len(output) * 0.03)))
    indexes = np.linspace(
        max(1, len(output) // 5),
        min(len(output) - 2, len(output) * 4 // 5),
        injected_count,
        dtype=int,
    )
    for offset, index in enumerate(np.unique(indexes)):
        action = list(output[index].action)
        action[0] += 20.0 if offset % 2 == 0 else -20.0
        output[index] = _replace_action(output[index], action)
    return output


def inject_alignment_shift(frames: list[EpisodeFrame]) -> list[EpisodeFrame]:
    if len(frames) < 16:
        return list(frames)
    shift = min(12, max(2, len(frames) // 10))
    original = [list(frame.action) for frame in frames]
    return [
        _replace_action(frame, original[max(0, index - shift)])
        for index, frame in enumerate(frames)
    ]


def inject_extreme_values(frames: list[EpisodeFrame]) -> list[EpisodeFrame]:
    output = list(frames)
    if not output:
        return output
    center = len(output) // 2
    for index in range(max(0, center - 2), min(len(output), center + 3)):
        state = list(output[index].observation_state)
        action = list(output[index].action)
        if state:
            state[0] += 100.0
        if action:
            action[0] += 100.0
        output[index] = output[index].model_copy(
            update={"observation_state": state, "action": action}
        )
    return output


def inject_structural_failure(frames: list[EpisodeFrame]) -> list[EpisodeFrame]:
    output = list(frames)
    if not output or not output[0].observation_state:
        return output
    state = list(output[0].observation_state)
    state[0] = float("nan")
    output[0] = output[0].model_copy(update={"observation_state": state})
    return output


def evaluate_reader(
    reader: DatasetAdapter,
    cleaning_config: CleaningConfig | None = None,
    filter_config: FilterConfig | None = None,
) -> tuple[FilterSummary, list[EpisodeQualityResult]]:
    filter_config = filter_config or FilterConfig(
        gripper_indices=infer_gripper_indices(reader)
    )
    filters = DatasetFilterService(reader, filter_config).summary()
    results = EpisodeQualityScorer().score_dataset(
        reader,
        cleaning_config or CleaningConfig(),
        filter_summary=filters,
    )
    return filters, results


def _score_stats(results: list[EpisodeQualityResult]) -> dict:
    scores = sorted(
        result.data_quality_score
        for result in results
        if result.data_quality_score is not None
    )
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "iqr": None,
            "statuses": {},
        }
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": mean(scores),
        "median": median(scores),
        "iqr": float(np.quantile(scores, 0.75) - np.quantile(scores, 0.25)),
        "statuses": {
            status: sum(result.status == status for result in results)
            for status in ("passed", "review", "excluded", "unscored")
        },
    }


def legacy_proxy_scores(reader: DatasetAdapter) -> list[float]:
    scores = []
    for episode in reader.list_episodes():
        frames = reader.read_episode_frames(episode.episode_index)
        smoothness = _smoothness([frame.action for frame in frames])
        tracking = _tracking_score(frames)
        scores.append((smoothness * 2 + tracking) / 3)
    return scores


def benchmark_coffee(path: Path) -> dict:
    reader = LeRobotDatasetReader(path)
    _filters, results = evaluate_reader(reader)
    legacy = legacy_proxy_scores(reader)
    validation_indexes = [
        episode.episode_index
        for episode in reader.list_episodes()
        if episode.episode_index % 2 == 1
    ]
    baseline_view = DatasetView(reader, validation_indexes)
    _baseline_filters, baseline_results = evaluate_reader(baseline_view)
    baseline_by_episode = {
        result.episode_index: result for result in baseline_results
    }
    faults = {
        "sudden_change": inject_sudden_change,
        "state_action_alignment": inject_alignment_shift,
        "extreme_value": inject_extreme_values,
        "structural_integrity": inject_structural_failure,
    }
    fault_reports = {}
    combined_labels: list[int] = []
    combined_anomaly_scores: list[float] = []
    for name, transform in faults.items():
        corrupted_view = DatasetView(reader, validation_indexes, transform)
        _fault_filters, fault_results = evaluate_reader(corrupted_view)
        overall_drops = []
        stage_drops = []
        recalls = []
        labels = []
        anomaly_scores = []
        for result in fault_results:
            baseline = baseline_by_episode[result.episode_index]
            baseline_score = baseline.data_quality_score or 0.0
            fault_score = result.data_quality_score or 0.0
            overall_drops.append(baseline_score - fault_score)
            if name in result.per_attribute_scores:
                stage_drops.append(
                    baseline.per_attribute_scores.get(name, 1.0)
                    - result.per_attribute_scores[name]
                )
            if name == "structural_integrity":
                recalls.append(result.status == "excluded")
            else:
                material_drop = bool(stage_drops and stage_drops[-1] >= 0.3)
                recalls.append(
                    result.per_attribute_scores.get(name, 1.0) <= 0.4
                    or material_drop
                )
            labels.extend([0, 1])
            anomaly_scores.extend([1 - baseline_score, 1 - fault_score])
            combined_labels.extend([0, 1])
            combined_anomaly_scores.extend([1 - baseline_score, 1 - fault_score])
        fault_reports[name] = {
            "auc": roc_auc(labels, anomaly_scores),
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "median_stage_drop": median(stage_drops) if stage_drops else None,
            "median_overall_drop": median(overall_drops) if overall_drops else 0.0,
        }
    legacy_sorted = sorted(legacy)
    combined_auc = roc_auc(
        combined_labels,
        combined_anomaly_scores,
    )
    initial_thresholds_passed = (
        combined_auc >= 0.9
        and all(item["recall"] >= 0.95 for item in fault_reports.values())
        and all(
            (item["median_stage_drop"] or 0.0) >= 0.3
            for name, item in fault_reports.items()
            if name != "structural_integrity"
        )
        and all(
            item["median_overall_drop"] >= 0.1
            for item in fault_reports.values()
        )
    )
    return {
        "before": {
            "count": len(legacy),
            "min": min(legacy_sorted),
            "max": max(legacy_sorted),
            "mean": mean(legacy_sorted),
            "median": median(legacy_sorted),
            "iqr": float(
                np.quantile(legacy_sorted, 0.75)
                - np.quantile(legacy_sorted, 0.25)
            ),
            "passed_at_0_8": sum(score >= 0.8 for score in legacy),
        },
        "after": _score_stats(results),
        "injected_faults": fault_reports,
        "combined_injection_auc": combined_auc,
        "threshold_calibration": {
            "performed": False,
            "reason": (
                "Initial thresholds passed the controlled-injection gate; "
                "the reserved even-episode tuning split was not used."
                if initial_thresholds_passed
                else "Initial thresholds failed; calibration is required."
            ),
            "training_split": "coffee even episode indexes (reserved)",
            "validation_split": "coffee odd episode indexes",
        },
    }


def benchmark_exylos(path: Path) -> dict:
    reader = LeRobotDatasetReader(path)
    filters, results = evaluate_reader(reader)
    annotations_path = path / "annotations.json"
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    annotations = {
        int(item["episode_id"].split("_")[-1]): item
        for item in payload.get("episodes", [])
    }
    result_by_episode = {result.episode_index: result for result in results}
    sudden_scores = []
    smoothness_labels = []
    discontinuities = []
    frame_counts = []
    for episode in filters.episodes:
        annotation = annotations.get(episode.episode_index)
        stage = episode.stage_status["sudden_change"]
        if annotation is None or stage.score is None:
            continue
        sudden_scores.append(stage.score)
        smoothness_labels.append(float(annotation["scores"]["D5_motion_smoothness"]))
        discontinuities.append(
            float(annotation["raw_measurements"]["discontinuity_count"])
        )
        frame_counts.append(float(reader.episode(episode.episode_index).length))
    return {
        "scores": _score_stats(list(result_by_episode.values())),
        "sudden_vs_motion_smoothness_spearman": spearman(
            sudden_scores,
            smoothness_labels,
        ),
        "sudden_vs_discontinuity_spearman": spearman(
            sudden_scores,
            discontinuities,
        ),
        "annotation_motion_vs_discontinuity_spearman": spearman(
            smoothness_labels,
            discontinuities,
        ),
        "frame_count_vs_motion_smoothness_spearman": spearman(
            frame_counts,
            smoothness_labels,
        ),
        "frame_count_vs_discontinuity_spearman": spearman(
            frame_counts,
            discontinuities,
        ),
        "sudden_flagged_episode_count": sum(score < 1 for score in sudden_scores),
    }


def _label_frames(path: Path) -> set[int]:
    return {
        index
        for index, value in enumerate(path.read_text(encoding="utf-8").splitlines())
        if value.strip() and int(float(value.strip())) != 0
    }


def benchmark_botfails(normal_path: Path, test_path: Path, labels_path: Path) -> dict:
    normal_reader = LeRobotDatasetReader(normal_path)
    _normal_filters, normal_results = evaluate_reader(normal_reader)
    test_reader = LeRobotDatasetReader(test_path)
    test_service = DatasetFilterService(
        test_reader,
        FilterConfig(gripper_indices=infer_gripper_indices(test_reader)),
    )
    test_filters = test_service.summary()
    test_results = EpisodeQualityScorer().score_dataset(
        test_reader,
        CleaningConfig(),
        filter_summary=test_filters,
    )
    anomaly_episodes = 0
    numeric_coverage = 0
    flagged_frames: set[tuple[int, int]] = set()
    labeled_frames: set[tuple[int, int]] = set()
    for episode in test_filters.episodes:
        label_file = labels_path / f"episode_{episode.episode_index:06d}_labels.csv"
        labels = _label_frames(label_file) if label_file.is_file() else set()
        labeled_frames.update((episode.episode_index, frame) for frame in labels)
        if not labels:
            continue
        anomaly_episodes += 1
        if any(stage.count > 0 for stage in episode.stage_status.values()):
            numeric_coverage += 1
        for stage_id in ("sudden_change", "extreme_value"):
            if episode.stage_status[stage_id].count <= 0:
                continue
            detail = test_service.detail(stage_id, episode.episode_index)
            for row in detail.table_rows:
                if "frame" in row:
                    flagged_frames.add((episode.episode_index, int(row["frame"])))
    overlap = flagged_frames & labeled_frames
    normal_review = sum(result.status == "review" for result in normal_results)
    normal_excluded = sum(result.status == "excluded" for result in normal_results)
    return {
        "expert_scores": _score_stats(normal_results),
        "test_scores": _score_stats(test_results),
        "expert_false_review_rate": normal_review / len(normal_results),
        "expert_false_excluded_rate": normal_excluded / len(normal_results),
        "anomaly_episode_count": anomaly_episodes,
        "numeric_episode_coverage": (
            numeric_coverage / anomaly_episodes if anomaly_episodes else 0.0
        ),
        "temporal_overlap_precision": (
            len(overlap) / len(flagged_frames) if flagged_frames else 0.0
        ),
        "temporal_overlap_recall": (
            len(overlap) / len(labeled_frames) if labeled_frames else 0.0
        ),
    }


def acceptance(report: dict) -> dict[str, bool]:
    injection_reports = report["coffee"]["injected_faults"]
    stage_reports = [
        item
        for name, item in injection_reports.items()
        if name != "structural_integrity"
    ]
    return {
        "combined_injection_auc_gte_0_90": (
            report["coffee"]["combined_injection_auc"] >= 0.9
        ),
        "each_fault_recall_gte_0_95": all(
            item["recall"] >= 0.95 for item in injection_reports.values()
        ),
        "stage_median_drop_gte_0_30": all(
            (item["median_stage_drop"] or 0.0) >= 0.3 for item in stage_reports
        ),
        "overall_median_drop_gte_0_10": all(
            item["median_overall_drop"] >= 0.1
            for item in injection_reports.values()
        ),
        "exylos_smoothness_rho_gte_0_40": (
            report["exylos"]["sudden_vs_motion_smoothness_spearman"] >= 0.4
        ),
        "exylos_discontinuity_rho_lte_minus_0_40": (
            report["exylos"]["sudden_vs_discontinuity_spearman"] <= -0.4
        ),
        "botfails_expert_false_excluded_zero": (
            report["botfails"]["expert_false_excluded_rate"] == 0
        ),
        "botfails_expert_false_review_lte_0_10": (
            report["botfails"]["expert_false_review_rate"] <= 0.1
        ),
    }


def markdown_report(report: dict) -> str:
    checks = report["acceptance"]
    coffee_after = report["coffee"]["after"]
    fault_lines = [
        (
            f"- {name}: AUC {metrics['auc']:.4f}, recall "
            f"{metrics['recall']:.4f}, rule-score median drop "
            f"{metrics['median_stage_drop'] if metrics['median_stage_drop'] is not None else 0:.4f}, "
            f"data-score median drop {metrics['median_overall_drop']:.4f}"
        )
        for name, metrics in report["coffee"]["injected_faults"].items()
    ]
    lines = [
        "# Quality scoring benchmark",
        "",
        "## Coffee before / after",
        "",
        f"- Before range: {report['coffee']['before']['min']:.4f}–{report['coffee']['before']['max']:.4f}",
        f"- Before IQR: {report['coffee']['before']['iqr']:.4f}",
        f"- After range: {report['coffee']['after']['min']:.4f}–{report['coffee']['after']['max']:.4f}",
        f"- After IQR: {coffee_after['iqr']:.4f}",
        f"- After status distribution: {coffee_after['statuses']}",
        f"- Combined injected-fault AUC: {report['coffee']['combined_injection_auc']:.4f}",
        f"- Threshold calibration: {report['coffee']['threshold_calibration']['reason']}",
        "",
        "### Injected-fault validation",
        "",
        *fault_lines,
        "",
        "## External datasets",
        "",
        f"- Exylos smoothness ρ: {report['exylos']['sudden_vs_motion_smoothness_spearman']:.4f}",
        f"- Exylos discontinuity ρ: {report['exylos']['sudden_vs_discontinuity_spearman']:.4f}",
        f"- Exylos annotation D5 vs discontinuity ρ: {report['exylos']['annotation_motion_vs_discontinuity_spearman']:.4f}",
        f"- Exylos frame count vs D5 ρ: {report['exylos']['frame_count_vs_motion_smoothness_spearman']:.4f}",
        f"- Exylos frame count vs discontinuity ρ: {report['exylos']['frame_count_vs_discontinuity_spearman']:.4f}",
        f"- Exylos episodes with numerical sudden-change findings: {report['exylos']['sudden_flagged_episode_count']}/50",
        f"- Exylos score range: {report['exylos']['scores']['min']:.4f}–{report['exylos']['scores']['max']:.4f}",
        f"- Exylos score IQR: {report['exylos']['scores']['iqr']:.4f}",
        f"- Exylos status distribution: {report['exylos']['scores']['statuses']}",
        f"- BotFails expert review rate: {report['botfails']['expert_false_review_rate']:.4f}",
        f"- BotFails expert excluded rate: {report['botfails']['expert_false_excluded_rate']:.4f}",
        f"- BotFails anomaly numeric coverage: {report['botfails']['numeric_episode_coverage']:.4f}",
        f"- BotFails temporal overlap precision: {report['botfails']['temporal_overlap_precision']:.4f}",
        f"- BotFails temporal overlap recall: {report['botfails']['temporal_overlap_recall']:.4f}",
        "",
        "## Acceptance",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name}"
        for name, passed in checks.items()
    )
    return "\n".join(lines) + "\n"


def download_benchmark_data(root: Path) -> dict[str, Path]:
    paths = {
        "exylos": root / "exylos-pick-and-place",
        "botfails_normal": root / "botfails-making-coffee-expert",
        "botfails_test": root / "botfails-making-coffee-test",
        "botfails_labels": root / "botfails-making-coffee-labels",
    }
    download_dataset(
        "ExylosAi/pick_and_place_sample",
        paths["exylos"],
        data_only=True,
    )
    download_dataset(
        "kantine/BotFails",
        paths["botfails_normal"],
        source_prefix="BotFails/normal_train/domotic_makingCoffee_expert",
        data_only=True,
    )
    download_dataset(
        "kantine/BotFails",
        paths["botfails_test"],
        source_prefix="BotFails/test/domotic_makingCoffee_anomaly",
        data_only=True,
    )
    download_dataset(
        "kantine/BotFails",
        paths["botfails_labels"],
        source_prefix="BotFails/labels/domotic_makingCoffee_anomaly",
        data_only=True,
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark quality scoring on controlled and external data."
    )
    parser.add_argument(
        "--coffee",
        type=Path,
        default=Path("data/samples/aloha_static_coffee"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/samples/quality-benchmark"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/quality-benchmark"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = (
        download_benchmark_data(args.data_root)
        if args.download
        else {
            "exylos": args.data_root / "exylos-pick-and-place",
            "botfails_normal": args.data_root / "botfails-making-coffee-expert",
            "botfails_test": args.data_root / "botfails-making-coffee-test",
            "botfails_labels": args.data_root / "botfails-making-coffee-labels",
        }
    )
    report = {
        "coffee": benchmark_coffee(args.coffee),
        "exylos": benchmark_exylos(paths["exylos"]),
        "botfails": benchmark_botfails(
            paths["botfails_normal"],
            paths["botfails_test"],
            paths["botfails_labels"],
        ),
    }
    report["acceptance"] = acceptance(report)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(json.dumps(report["acceptance"], indent=2))
    print(f"Report: {(args.output / 'report.md').resolve()}")
    if args.strict and not all(report["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
