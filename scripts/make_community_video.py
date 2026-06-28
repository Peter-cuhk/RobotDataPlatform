from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from imageio_ffmpeg import get_ffmpeg_exe


ROOT = Path.cwd()
OUT = ROOT / "output" / "community-video"
FINAL_DIR = OUT / "final"
TEMP_DIR = OUT / "segments"
QA_DIR = OUT / "qa"
FINAL = FINAL_DIR / "robot-data-studio-community-demo.mp4"
THUMBNAIL = FINAL_DIR / "thumbnail.png"

W, H = 1920, 1080
FPS = 30
BG = "#070b10"
PANEL = "#0d151f"
FG = "#edf5ff"
MUTED = "#a9bacb"
ACCENT = "#7ed6a8"
BLUE = "#4da3ff"

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def ffmpeg() -> str:
    return get_ffmpeg_exe()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size=size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=face)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def make_card(path: Path, kind: str) -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    for y in range(H):
        r = 7 + int(6 * y / H)
        g = 11 + int(12 * y / H)
        b = 16 + int(20 * y / H)
        draw.line((0, y, W, y), fill=(r, g, b))

    for i in range(22):
        x = 105 + i * 92
        draw.line((x, 92, x - 330, 1020), fill="#142331", width=1)
    for x in range(120, W, 160):
        draw.line((x, 0, x, H), fill="#0c1824", width=1)

    if kind == "title":
        draw.text((124, 282), "Robot Data Studio", font=font(86, True), fill=FG)
        draw.text((128, 400), "From raw robot episodes to trusted training data.", font=font(42, True), fill=ACCENT)
        body = "A local-first open-source workspace for robot dataset quality."
        draw.text((132, 480), body, font=font(30), fill=MUTED)
        draw.rounded_rectangle((128, 610, 642, 686), radius=20, fill="#101c28", outline="#29435e", width=2)
        draw.text((162, 632), "Inspect · Clean · Replay · Export", font=font(26, True), fill=FG)
    else:
        draw.text((124, 310), "Open source. Local-first.", font=font(76, True), fill=FG)
        draw.text((128, 420), "Built for robot learning datasets before training.", font=font(38, True), fill=ACCENT)
        cards = [
            ("Local data", "Source files stay on your machine."),
            ("Explainable QA", "Scores, findings, thresholds, and review."),
            ("Training export", "ACT HDF5, robomimic, UMI Zarr, LeRobot."),
        ]
        for i, (head, body) in enumerate(cards):
            x = 128 + i * 548
            draw.rounded_rectangle((x, 572, x + 488, 744), radius=18, fill=PANEL, outline="#29435e", width=2)
            draw.text((x + 32, 612), head, font=font(31, True), fill=ACCENT)
            lines = wrap_text(draw, body, font(23), 410)
            for j, line in enumerate(lines):
                draw.text((x + 32, 666 + j * 32), line, font=font(23), fill=FG)
        draw.text((132, 850), "From raw robot episodes to trusted training data.", font=font(36, True), fill=FG)

    image.save(path)


def render_card(image_path: Path, out_path: Path, duration: float) -> None:
    cmd = [
        ffmpeg(),
        "-y",
        "-loop",
        "1",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(image_path),
        "-an",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def write_textfile(text: str, temp_root: Path, name: str) -> Path:
    path = temp_root / f"{name}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def render_video_segment(
    source: Path,
    out_path: Path,
    start: float,
    duration: float,
    subtitle: str,
    eyebrow: str,
    temp_root: Path,
) -> None:
    subtitle_file = write_textfile(subtitle, temp_root, out_path.stem)
    eyebrow_file = write_textfile(eyebrow.upper(), temp_root, f"{out_path.stem}-eyebrow")
    filter_graph = (
        f"[0:v]fps={FPS},"
        "scale=1584:990:force_original_aspect_ratio=decrease,"
        "pad=1760:990:(ow-iw)/2:(oh-ih)/2:color=0x071017,"
        "setsar=1[screen];"
        f"color=c=0x070b10:s={W}x{H}:d={duration:.3f}:r={FPS}[bg];"
        "[bg][screen]overlay=80:45:shortest=1[v0];"
        "[v0]drawbox=x=80:y=45:w=1760:h=990:color=0x2b3b4d:t=2,"
        "drawbox=x=0:y=900:w=1920:h=180:color=0x05090D@0.93:t=fill,"
        f"drawtext=fontfile='{FONT_BOLD}':textfile='{eyebrow_file}':"
        "x=100:y=924:fontsize=22:fontcolor=0x7ED6A8,"
        f"drawtext=fontfile='{FONT_BOLD}':textfile='{subtitle_file}':"
        "x=100:y=962:fontsize=38:fontcolor=0xEDF5FF:line_spacing=10"
    )
    cmd = [
        ffmpeg(),
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v0]",
        "-an",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(out_path),
    ]
    # The final label is consumed by the filter chain itself, so map the implicit
    # final output when the filter graph has no explicit output label.
    cmd[cmd.index("[v0]")] = "0:v"
    cmd = [
        ffmpeg(),
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-an",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def render_still_segment(
    source: Path,
    out_path: Path,
    duration: float,
    subtitle: str,
    eyebrow: str,
    temp_root: Path,
) -> None:
    subtitle_file = write_textfile(subtitle, temp_root, out_path.stem)
    eyebrow_file = write_textfile(eyebrow.upper(), temp_root, f"{out_path.stem}-eyebrow")
    filter_graph = (
        "scale=1584:990:force_original_aspect_ratio=decrease,"
        "pad=1760:990:(ow-iw)/2:(oh-ih)/2:color=0x071017,"
        "setsar=1[screen];"
        f"color=c=0x070b10:s={W}x{H}:d={duration:.3f}:r={FPS}[bg];"
        "[bg][screen]overlay=80:45:shortest=1[v0];"
        "[v0]drawbox=x=80:y=45:w=1760:h=990:color=0x2b3b4d:t=2,"
        "drawbox=x=0:y=900:w=1920:h=180:color=0x05090D@0.93:t=fill,"
        f"drawtext=fontfile='{FONT_BOLD}':textfile='{eyebrow_file}':"
        "x=100:y=924:fontsize=22:fontcolor=0x7ED6A8,"
        f"drawtext=fontfile='{FONT_BOLD}':textfile='{subtitle_file}':"
        "x=100:y=962:fontsize=38:fontcolor=0xEDF5FF:line_spacing=10"
    )
    cmd = [
        ffmpeg(),
        "-y",
        "-loop",
        "1",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
        "-filter_complex",
        filter_graph,
        "-an",
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def concat_segments(segments: list[Path], output: Path) -> None:
    list_file = TEMP_DIR / "concat.txt"
    list_file.write_text("".join(f"file '{segment.resolve()}'\n" for segment in segments), encoding="utf-8")
    cmd = [
        ffmpeg(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def extract_frame(video: Path, timestamp: float, output: Path) -> None:
    cmd = [
        ffmpeg(),
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def render() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    new_capture = OUT / "raw" / "page@2a5dac28e6abcb4c66d0bea05a5ff664.webm"
    promo_a = ROOT / "output" / "promo-video" / "browser-video" / "page@0bf6aa97cae11957ac0ee5e6a9207b36.webm"
    tail = ROOT / "output" / "promo-video" / "tail-video" / "page@2d7467e3c532b2ab45c4d0640c44b0f3.webm"
    quality_screen = ROOT / "output" / "promo-video" / "screens" / "04-cleaned.png"
    review_screen = ROOT / "output" / "promo-video" / "screens" / "05-filter-detail.png"
    export_screen = ROOT / "output" / "promo-video" / "screens" / "08-exported.png"
    for source in [new_capture, promo_a, tail, quality_screen, review_screen, export_screen]:
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(source)

    title_png = TEMP_DIR / "title.png"
    end_png = TEMP_DIR / "end.png"
    make_card(title_png, "title")
    make_card(end_png, "end")
    shutil.copyfile(title_png, THUMBNAIL)

    segments: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="rds-video-text-") as tmp:
        temp_root = Path(tmp)
        segment_specs = [
            ("00-title.mp4", "card", title_png, 0, 6.0, "", ""),
            (
                "01-local-import.mp4",
                "video",
                new_capture,
                0.0,
                14.0,
                "Robot datasets are hard to trust before training. Start by importing local data.",
                "Local-first import",
            ),
            (
                "02-dataset-overview.mp4",
                "video",
                promo_a,
                4.0,
                12.0,
                "Robot Data Studio indexes episodes, frames, rates, and task metadata in one workspace.",
                "Dataset overview",
            ),
            (
                "03-quality-pipeline.mp4",
                "still",
                quality_screen,
                0.0,
                17.0,
                "Run quality checks and turn raw robot episodes into reviewable evidence.",
                "Quality pipeline",
            ),
            (
                "04-explainable-review.mp4",
                "still",
                review_screen,
                0.0,
                15.0,
                "Scores, findings, thresholds, and manual decisions stay explainable.",
                "Explainable review",
            ),
            (
                "05-rerun-replay.mp4",
                "video",
                tail,
                0.0,
                8.0,
                "Replay video, state, action, and timelines together with Rerun.",
                "Synchronized replay",
            ),
            (
                "06-training-export.mp4",
                "still",
                export_screen,
                0.0,
                6.4,
                "Export clean episodes to training formats like ACT HDF5, robomimic, UMI Zarr, and LeRobot.",
                "Training export",
            ),
            ("07-end.mp4", "card", end_png, 0, 8.0, "", ""),
        ]

        for spec in segment_specs:
            name, kind, source, start, duration, subtitle, eyebrow = spec
            out_path = TEMP_DIR / name
            if kind == "card":
                render_card(Path(source), out_path, float(duration))
            elif kind == "still":
                render_still_segment(
                    Path(source),
                    out_path,
                    float(duration),
                    str(subtitle),
                    str(eyebrow),
                    temp_root,
                )
            else:
                render_video_segment(
                    Path(source),
                    out_path,
                    float(start),
                    float(duration),
                    str(subtitle),
                    str(eyebrow),
                    temp_root,
                )
            segments.append(out_path)

    concat_segments(segments, FINAL)
    make_qa()


def make_qa() -> None:
    if not FINAL.exists():
        raise FileNotFoundError(FINAL)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    for timestamp in [2, 12, 28, 48, 63, 72, 82]:
        extract_frame(FINAL, timestamp, QA_DIR / f"qa-{timestamp:02d}s.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()
    if args.qa_only:
        make_qa()
    else:
        render()
    print(FINAL)
    print(THUMBNAIL)


if __name__ == "__main__":
    main()
