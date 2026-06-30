from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from robot_data_studio.lerobot.reader import LeRobotDatasetReader


ChartSeries = tuple[str, list[tuple[float, float]]]


def find_named_dimensions(features: dict[str, object], feature_key: str, token: str) -> list[tuple[int, str]]:
    feature = features.get(feature_key)
    if not isinstance(feature, dict):
        return []
    names = feature.get("names")
    if not isinstance(names, dict):
        return []

    token = token.lower()
    dimensions: list[tuple[int, str]] = []
    for values in names.values():
        if not isinstance(values, list):
            continue
        for index, name in enumerate(values):
            if isinstance(name, str) and token in name.lower():
                dimensions.append((index, name))
    return dimensions


def episode_duration_rows(episodes: Iterable[object]) -> list[tuple[int, float]]:
    rows = [
        (int(getattr(episode, "episode_index")), float(getattr(episode, "duration_seconds")))
        for episode in episodes
    ]
    return sorted(rows, key=lambda row: row[0])


def collect_gripper_series(reader: LeRobotDatasetReader, episode_index: int) -> list[ChartSeries]:
    dimensions = find_named_dimensions(reader.metadata().features, "action", "gripper")
    if not dimensions:
        raise ValueError("No gripper dimensions found in action feature metadata.")

    frames = reader.read_episode_frames(episode_index)
    series: list[ChartSeries] = []
    for dimension_index, label in dimensions:
        points = [
            (float(frame.timestamp), float(frame.action[dimension_index]))
            for frame in frames
            if len(frame.action) > dimension_index
        ]
        series.append((label, points))
    return series


def draw_line_chart(
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[ChartSeries],
    *,
    width: int = 1200,
    height: int = 720,
) -> None:
    image, draw, font, title_font = _new_chart(width, height)
    plot = _plot_area(width, height)
    colors = ["#2f80ed", "#eb5757", "#27ae60", "#f2994a"]

    all_points = [point for _, points in series for point in points]
    if not all_points:
        raise ValueError("Cannot draw a line chart without points.")

    min_x, max_x = _range([point[0] for point in all_points], padding_ratio=0.0)
    min_y, max_y = _range([point[1] for point in all_points])
    _draw_axes(draw, plot, title, x_label, y_label, min_x, max_x, min_y, max_y, font, title_font)

    for series_index, (label, points) in enumerate(series):
        color = colors[series_index % len(colors)]
        scaled = [_scale_point(point, plot, min_x, max_x, min_y, max_y) for point in points]
        if len(scaled) > 1:
            draw.line(scaled, fill=color, width=3, joint="curve")
        for x, y in scaled[:: max(1, len(scaled) // 60)]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

        legend_x = plot[0] + 260 + series_index * 210
        legend_y = plot[1] - 34
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 14, legend_y + 14), radius=3, fill=color)
        draw.text((legend_x + 21, legend_y - 1), label, fill="#334155", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_episode_duration_chart(
    output_path: Path,
    rows: Sequence[tuple[int, float]],
    *,
    width: int = 1200,
    height: int = 720,
) -> None:
    image, draw, font, title_font = _new_chart(width, height)
    plot = _plot_area(width, height)
    if not rows:
        raise ValueError("Cannot draw an episode duration chart without rows.")

    min_x, max_x = _range([float(index) for index, _ in rows], padding_ratio=0.0)
    min_y = 0.0
    _, max_y = _range([duration for _, duration in rows])
    _draw_axes(
        draw,
        plot,
        "Episode Duration Distribution",
        "Episode index",
        "Duration (seconds)",
        min_x,
        max_x,
        min_y,
        max_y,
        font,
        title_font,
    )

    bar_slot = (plot[2] - plot[0]) / max(1, len(rows))
    bar_width = max(3, min(18, int(bar_slot * 0.72)))
    for offset, (episode_index, duration) in enumerate(rows):
        x_center = plot[0] + bar_slot * offset + bar_slot / 2
        _, y = _scale_point((float(episode_index), duration), plot, min_x, max_x, min_y, max_y)
        draw.rounded_rectangle(
            (x_center - bar_width / 2, y, x_center + bar_width / 2, plot[3]),
            radius=3,
            fill="#2f80ed",
        )

    durations = [duration for _, duration in rows]
    mean_duration = sum(durations) / len(durations)
    _, mean_y = _scale_point((min_x, mean_duration), plot, min_x, max_x, min_y, max_y)
    draw.line((plot[0], mean_y, plot[2], mean_y), fill="#eb5757", width=2)
    draw.text((plot[2] - 145, mean_y - 24), f"mean {mean_duration:.2f}s", fill="#b91c1c", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def generate_report_charts(dataset_path: Path, output_dir: Path, episode_index: int) -> tuple[Path, Path]:
    reader = LeRobotDatasetReader(dataset_path)
    gripper_path = output_dir / "gripper_opening_curve.png"
    duration_path = output_dir / "episode_duration_distribution.png"

    draw_line_chart(
        gripper_path,
        f"Gripper Opening Curve - Episode {episode_index:06d}",
        "Time (seconds)",
        "Action value",
        collect_gripper_series(reader, episode_index),
    )
    draw_episode_duration_chart(duration_path, episode_duration_rows(reader.list_episodes()))
    return gripper_path, duration_path


def _new_chart(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, ImageFont.ImageFont, ImageFont.ImageFont]:
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=30)
    return image, draw, font, title_font


def _plot_area(width: int, height: int) -> tuple[int, int, int, int]:
    return (96, 112, width - 54, height - 92)


def _range(values: Sequence[float], *, padding_ratio: float = 0.08) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.1)
        return lower - padding, upper + padding
    padding = (upper - lower) * padding_ratio
    return lower - padding, upper + padding


def _scale_point(
    point: tuple[float, float],
    plot: tuple[int, int, int, int],
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[float, float]:
    x, y = point
    left, top, right, bottom = plot
    scaled_x = left + (x - min_x) / (max_x - min_x) * (right - left)
    scaled_y = bottom - (y - min_y) / (max_y - min_y) * (bottom - top)
    return scaled_x, scaled_y


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    title: str,
    x_label: str,
    y_label: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = plot
    draw.text((left, 38), title, fill="#0f172a", font=title_font)
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    draw.line((left, top, left, bottom), fill="#334155", width=2)

    for tick in range(6):
        ratio = tick / 5
        x = left + ratio * (right - left)
        y = bottom - ratio * (bottom - top)
        x_value = min_x + ratio * (max_x - min_x)
        y_value = min_y + ratio * (max_y - min_y)
        draw.line((x, bottom, x, bottom + 7), fill="#64748b", width=1)
        draw.line((left - 7, y, left, y), fill="#64748b", width=1)
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        draw.text((x - 22, bottom + 14), f"{x_value:.1f}", fill="#475569", font=font)
        draw.text((left - 82, y - 9), f"{y_value:.2f}", fill="#475569", font=font)

    draw.text(((left + right) / 2 - 65, bottom + 48), x_label, fill="#334155", font=font)
    draw.text((left, top - 28), y_label, fill="#334155", font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone dataset report charts.")
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/report-charts"))
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    gripper_path, duration_path = generate_report_charts(
        args.dataset_path,
        args.output_dir,
        args.episode_index,
    )
    print(gripper_path)
    print(duration_path)


if __name__ == "__main__":
    main()
