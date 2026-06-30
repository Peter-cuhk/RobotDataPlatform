from __future__ import annotations

import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from imageio_ffmpeg import get_ffmpeg_exe


ROOT = Path.cwd()
OUT = ROOT / "output" / "promo-video"
SCREENS = OUT / "screens"
FINAL = OUT / "robot-data-studio-promo.mp4"
TEMP_VIDEO = OUT / "robot-data-studio-promo-silent.mp4"
TEMP_AUDIO = OUT / "robot-data-studio-promo-bed.wav"

W, H = 1920, 1080
FPS = 30
DURATION = 60


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_HERO = font(86, True)
FONT_HERO_SMALL = font(36, True)
FONT_TITLE = font(52, True)
FONT_SUBTITLE = font(32)
FONT_CAPTION = font(34, True)
FONT_SMALL = font(24)
FONT_TINY = font(19)


def ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_image(img: Image.Image, box_w: int, box_h: int, zoom: float = 1.0) -> Image.Image:
    scale = min(box_w / img.width, box_h / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (box_w, box_h), "#071017")
    left = (box_w - resized.width) // 2
    top = (box_h - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def screen_frame(img: Image.Image, t: float, title: str, caption: str, metric: str, progress: float) -> Image.Image:
    base = Image.new("RGB", (W, H), "#070b10")
    draw = ImageDraw.Draw(base)

    for y in range(H):
        r = 7 + int(10 * y / H)
        g = 11 + int(15 * y / H)
        b = 16 + int(18 * y / H)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    zoom = 1.015 + 0.035 * ease(t)
    shot = fit_image(img, 1760, 990, zoom)
    shot = shot.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=3))
    x, y = 80, 58
    shadow = Image.new("RGBA", (1788, 1018), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    rounded_rect(sdraw, (14, 14, 1774, 1004), 24, (0, 0, 0, 155))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    base.paste(shadow, (x - 14, y - 14), shadow)

    mask = Image.new("L", (1760, 990), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 1760, 990), radius=22, fill=255)
    base.paste(shot, (x, y), mask)
    draw = ImageDraw.Draw(base)
    rounded_rect(draw, (x, y, x + 1760, y + 990), 22, None, "#2b3b4d", 2)

    overlay_h = 156
    overlay = Image.new("RGBA", (W, overlay_h), (5, 9, 13, 218))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle((0, 0, W, 1), fill=(80, 120, 160, 150))
    odraw.text((100, 28), title, font=FONT_TITLE, fill=(242, 248, 255, 255))
    odraw.text((100, 94), caption, font=FONT_SMALL, fill=(169, 188, 208, 255))
    rounded_rect(odraw, (1385, 40, 1818, 110), 22, (14, 29, 43, 245), "#35506b", 2)
    odraw.text((1414, 59), metric, font=FONT_CAPTION, fill=(132, 214, 169, 255))
    base.paste(overlay, (0, H - overlay_h), overlay)

    draw.rounded_rectangle((100, H - 34, 1820, H - 20), radius=7, fill="#132031")
    draw.rounded_rectangle((100, H - 34, 100 + int(1720 * progress), H - 20), radius=7, fill="#2f89e7")
    return base


def title_frame(t: float) -> Image.Image:
    base = Image.new("RGB", (W, H), "#070b10")
    draw = ImageDraw.Draw(base)
    for y in range(H):
        draw.line([(0, y), (W, y)], fill=(7, 12 + y // 80, 18 + y // 55))
    for i in range(22):
        x = 120 + i * 82
        color = (28, 54, 78) if i % 2 else (35, 76, 94)
        draw.line((x, 120, x - 360, 1020), fill=color, width=1)
    alpha = int(255 * ease(min(1.0, t * 3.0)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ldraw = ImageDraw.Draw(layer)
    ldraw.text((120, 310), "Robot Data Studio", font=FONT_HERO, fill=(245, 250, 255, alpha))
    ldraw.text((126, 422), "机器人数据集检查、清洗与格式转换工作台", font=FONT_HERO_SMALL, fill=(138, 202, 255, alpha))
    ldraw.text((128, 500), "从本地数据到可训练资产，一分钟看清全流程", font=FONT_SUBTITLE, fill=(178, 195, 212, alpha))
    rounded_rect(ldraw, (126, 605, 595, 682), 24, (24, 53, 77, alpha), "#426d93", 2)
    ldraw.text((158, 626), "Local-first · Quality-first", font=FONT_SMALL, fill=(238, 247, 255, alpha))
    base.paste(layer, (0, 0), layer)
    return base


def end_frame(t: float) -> Image.Image:
    base = Image.new("RGB", (W, H), "#070b10")
    draw = ImageDraw.Draw(base)
    for y in range(H):
        draw.line([(0, y), (W, y)], fill=(7, 11 + y // 100, 16 + y // 80))
    title = "Robot Data Studio"
    subtitle = "把机器人数据清洗成可追溯、可复核、可导出的训练资产"
    draw.text((126, 265), title, font=FONT_HERO, fill="#f4f9ff")
    draw.text((132, 382), subtitle, font=FONT_SUBTITLE, fill="#b9c8d8")
    cards = [
        ("本地优先", "源数据留在本机"),
        ("自动质检", "分数、证据、问题同屏"),
        ("多格式导出", "ACT / robomimic / UMI / LeRobot"),
    ]
    for i, (head, body) in enumerate(cards):
        x = 126 + i * 548
        rounded_rect(draw, (x, 540, x + 486, 724), 22, "#101923", "#2b435a", 2)
        draw.text((x + 34, 585), head, font=FONT_CAPTION, fill="#84d6a9")
        draw.text((x + 34, 648), body, font=FONT_SMALL, fill="#d7e3ee")
    draw.text((132, 840), "Inspect · Clean · Replay · Export", font=FONT_TITLE, fill="#eef6ff")
    return base


def write_video():
    scenes = [
        ("title", None, 0, 5, "", "", ""),
        ("screen", "02-path.png", 5, 12, "导入本地机器人数据集", "指定 ALOHA coffee 数据路径，源数据不复制、不改写。", "Local-first"),
        ("screen", "03-imported.png", 12, 19, "索引数据规模与 episode", "50 episodes、55,000 frames、50 Hz，一眼看清数据资产。", "50 episodes"),
        ("screen", "04-cleaned.png", 19, 31, "自动清洗 Pipeline", "为每个 episode 生成质量分数，并保留可复核的问题证据。", "Pass 50"),
        ("screen", "05-filter-detail.png", 31, 41, "筛选详情可追溯", "信号曲线、阈值、问题列表同屏，人工审核更快更稳。", "0 issues"),
        ("screen", "07-rerun.png", 41, 52, "Rerun 多模态回放", "Action、state、任务文本与多相机视频同步检查。", "Replay ready"),
        ("screen", "08-exported.png", 52, 56, "导出训练格式", "把选中的 episode 转成 ACT HDF5 等下游训练格式。", "Exported"),
        ("end", None, 56, 60, "", "", ""),
    ]
    images = {p.name: Image.open(p).convert("RGB") for p in SCREENS.glob("*.png")}
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        str(TEMP_VIDEO),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame_idx in range(DURATION * FPS):
        now = frame_idx / FPS
        progress = now / DURATION
        active = scenes[-1]
        for scene in scenes:
            if scene[2] <= now < scene[3]:
                active = scene
                break
        kind, image_name, start, end, title, caption, metric = active
        local_t = (now - start) / max(0.001, end - start)
        if kind == "title":
            frame = title_frame(local_t)
        elif kind == "end":
            frame = end_frame(local_t)
        else:
            frame = screen_frame(images[image_name], local_t, title, caption, metric, progress)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg video encode failed")


def write_audio():
    rate = 48000
    samples = np.arange(DURATION * rate) / rate
    beat = ((samples * 2) % 1 < 0.055).astype(float)
    kick = np.sin(2 * math.pi * 72 * samples) * np.exp(-((samples * 2) % 1) * 18) * beat
    pad = 0.14 * np.sin(2 * math.pi * 146.83 * samples) + 0.10 * np.sin(2 * math.pi * 220 * samples)
    shimmer = 0.05 * np.sin(2 * math.pi * 440 * samples + 0.8 * np.sin(2 * math.pi * 0.2 * samples))
    audio = pad + shimmer + 0.33 * kick
    envelope = np.minimum(1, samples / 2) * np.minimum(1, (DURATION - samples) / 2)
    audio = audio * envelope * 0.35
    stereo = np.stack([audio, audio * 0.92], axis=1)
    pcm = np.clip(stereo * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(TEMP_AUDIO), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def mux():
    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(TEMP_VIDEO),
        "-i",
        str(TEMP_AUDIO),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-shortest",
        str(FINAL),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write_video()
    write_audio()
    mux()
    print(FINAL)
