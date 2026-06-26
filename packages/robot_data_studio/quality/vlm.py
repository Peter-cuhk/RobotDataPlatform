from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from robot_data_studio.formats.models import DatasetAdapter

from .models import VlmEvaluation, VlmSettings


class VlmEvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SampledVideoFrame:
    image: bytes
    timestamp: float | None = None
    label: str = ""


def _import_imageio_ffmpeg():
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except ImportError:
        return None
    return imageio_ffmpeg


class VlmTaskSuccessEvaluator:
    def evaluate(
        self,
        reader: DatasetAdapter,
        episode_index: int,
        settings: VlmSettings,
    ) -> VlmEvaluation:
        episode = reader.episode(episode_index)
        task = episode.tasks[0] if episode.tasks else "No task description provided."
        video_path = self._select_video_path(reader.root, episode.video_files)
        if video_path is None:
            raise VlmEvaluationError("VLM task success requires at least one local video file.")
        supplemental_video_paths = self._supplemental_video_paths(reader.root, episode.video_files, video_path)
        if settings.provider == "Gemini":
            return self._evaluate_gemini(video_path, task, settings)
        if settings.provider == "Local":
            return self._evaluate_local(video_path, task)
        return self._evaluate_openai_compatible(
            video_path,
            task,
            episode.duration_seconds,
            settings,
            trajectory_frames=reader.read_episode_frames(episode_index),
            supplemental_video_paths=supplemental_video_paths,
        )

    def _evaluate_local(self, video_path: Path, task: str) -> VlmEvaluation:
        return VlmEvaluation(
            success=True,
            score=0.5,
            reason=(
                "Local VLM provider is selected but no local vision model adapter is configured. "
                f"Video exists at {video_path.name}; task was not semantically judged: {task}"
            ),
        )

    def _evaluate_gemini(self, video_path: Path, task: str, settings: VlmSettings) -> VlmEvaluation:
        api_key = settings.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise VlmEvaluationError("Set GOOGLE_API_KEY or enter an API key to use Gemini VLM scoring.")
        try:
            import google.generativeai as genai  # type: ignore[import-not-found]
        except ImportError as error:
            raise VlmEvaluationError("Install google-generativeai to use Gemini VLM scoring.") from error
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(settings.model)
        prompt = self._prompt(settings, task)
        response = model.generate_content(
            [
                {"mime_type": "video/mp4", "data": video_path.read_bytes()},
                prompt,
            ],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        )
        return self._parse_evaluation(response.text)

    def _evaluate_openai_compatible(
        self,
        video_path: Path,
        task: str,
        duration_seconds: float,
        settings: VlmSettings,
        trajectory_frames: list[Any] | None = None,
        supplemental_video_paths: list[tuple[str, Path]] | None = None,
    ) -> VlmEvaluation:
        api_key = settings.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise VlmEvaluationError("Set OPENAI_API_KEY or enter an API key to use OpenAI VLM scoring.")
        sample_times = self._keyframe_times(trajectory_frames or [], settings.sample_frames)
        sampled_frames = self._sample_video_frames(
            video_path,
            duration_seconds,
            settings.sample_frames,
            sample_times=sample_times,
        )
        if not sampled_frames:
            raise VlmEvaluationError("Could not sample frames for OpenAI-compatible VLM scoring.")
        if sample_times and supplemental_video_paths:
            for label, path in supplemental_video_paths:
                supplemental_frames = self._sample_video_frames(
                    path,
                    duration_seconds,
                    1,
                    sample_times=[sample_times[-1]],
                )
                for frame in supplemental_frames:
                    sampled_frames.append(
                        SampledVideoFrame(
                            image=frame.image,
                            timestamp=frame.timestamp,
                            label=f"supplemental final frame from {label}",
                        )
                    )
        content: list[dict[str, Any]] = [{"type": "text", "text": self._prompt(settings, task)}]
        for index, frame in enumerate(sampled_frames, start=1):
            if isinstance(frame, SampledVideoFrame):
                image_bytes = frame.image
                label = frame.label or f"frame {index}"
                if frame.timestamp is not None:
                    label = f"{label} at {frame.timestamp:.2f}s"
                content.append({"type": "text", "text": label})
            else:
                image_bytes = frame
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}",
                        "detail": "high",
                    },
                }
            )
        payload = {
            "model": settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict robot demonstration evaluator. "
                        "Return only JSON with keys success, score, and reason."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        base_url = settings.api_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "api-key": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urlrequest.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise VlmEvaluationError(f"OpenAI-compatible VLM request failed: {detail}") from error
        except (OSError, URLError, json.JSONDecodeError) as error:
            raise VlmEvaluationError(f"OpenAI-compatible VLM request failed: {error}") from error
        content_text = result["choices"][0]["message"]["content"]
        return self._parse_evaluation(content_text, raw_response=result)

    def _sample_video_frames(
        self,
        video_path: Path,
        duration_seconds: float,
        sample_frames: int,
        sample_times: list[float] | None = None,
    ) -> list[SampledVideoFrame]:
        ffmpeg = self._ffmpeg_path()
        if not ffmpeg:
            raise VlmEvaluationError("ffmpeg is required to sample frames for OpenAI-compatible VLM scoring.")
        with tempfile.TemporaryDirectory() as directory:
            if sample_times:
                sampled = []
                labels = self._keyframe_labels(len(sample_times))
                for index, timestamp in enumerate(sample_times):
                    output = Path(directory) / f"frame-{index:03d}.jpg"
                    command = [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        str(output),
                    ]
                    subprocess.run(command, check=True)
                    if output.is_file():
                        sampled.append(
                            SampledVideoFrame(
                                image=output.read_bytes(),
                                timestamp=timestamp,
                                label=labels[index],
                            )
                        )
                return sampled

            output_pattern = Path(directory) / "frame-%03d.jpg"
            fps = max(sample_frames / max(duration_seconds, 1.0), 0.1)
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps}",
                "-frames:v",
                str(sample_frames),
                str(output_pattern),
            ]
            subprocess.run(command, check=True)
            frames = sorted(Path(directory).glob("frame-*.jpg"))
            return [SampledVideoFrame(image=frame.read_bytes()) for frame in frames]

    def _select_video_path(self, root: Path, video_files: dict[str, str]) -> Path | None:
        candidates = [(key, root / path) for key, path in video_files.items()]
        existing = [(key, path) for key, path in candidates if path.is_file()]
        if not existing:
            return None

        def priority(item: tuple[str, Path]) -> tuple[int, str]:
            key = item[0].lower()
            if "cam_high" in key or "high" in key:
                return (0, key)
            if "main" in key or "front" in key or "overhead" in key:
                return (1, key)
            if "wrist" not in key:
                return (2, key)
            return (3, key)

        return sorted(existing, key=priority)[0][1]

    def _supplemental_video_paths(
        self,
        root: Path,
        video_files: dict[str, str],
        primary_path: Path,
    ) -> list[tuple[str, Path]]:
        candidates = [
            (key, root / path)
            for key, path in video_files.items()
            if root / path != primary_path and (root / path).is_file()
        ]

        def priority(item: tuple[str, Path]) -> tuple[int, str]:
            key = item[0].lower()
            if "cam_low" in key or "low" in key:
                return (0, key)
            if "wrist" not in key:
                return (1, key)
            return (2, key)

        return sorted(candidates, key=priority)[:1]

    def _keyframe_times(self, trajectory_frames: list[Any], sample_frames: int) -> list[float]:
        if not trajectory_frames or sample_frames <= 0:
            return []
        timestamps = [float(frame.timestamp) for frame in trajectory_frames]
        final_time = timestamps[-1]
        if sample_frames == 1:
            return [final_time]

        gripper_times = self._last_gripper_event_times(trajectory_frames)
        anchors = [timestamps[0], *gripper_times, final_time]
        deduped = []
        for timestamp in anchors:
            if not any(abs(timestamp - existing) < 0.05 for existing in deduped):
                deduped.append(timestamp)
        if len(deduped) > sample_frames:
            keep_middle = deduped[1:-1][-max(sample_frames - 2, 0) :]
            deduped = [deduped[0], *keep_middle, deduped[-1]]
        while len(deduped) < sample_frames:
            ratio = len(deduped) / max(sample_frames - 1, 1)
            candidate = timestamps[0] + ratio * (final_time - timestamps[0])
            if not any(abs(candidate - existing) < 0.05 for existing in deduped):
                deduped.insert(-1, candidate)
            else:
                break
        return sorted(deduped)

    def _last_gripper_event_times(self, trajectory_frames: list[Any]) -> list[float]:
        first_values = trajectory_frames[0].observation_state or trajectory_frames[0].action
        if not first_values:
            return []
        gripper_indices = [index for index in range(6, len(first_values), 7)]
        event_times = []
        for gripper_index in gripper_indices:
            values = []
            for frame in trajectory_frames:
                source = frame.observation_state or frame.action
                if len(source) <= gripper_index:
                    break
                values.append(float(source[gripper_index]))
            if len(values) < 2:
                continue
            diffs = [abs(values[index] - values[index - 1]) for index in range(1, len(values))]
            threshold = max(0.003, self._robust_change_threshold(diffs))
            event_indexes = [index + 1 for index, diff in enumerate(diffs) if diff >= threshold]
            if event_indexes:
                event_times.append(float(trajectory_frames[event_indexes[-1]].timestamp))
        return event_times

    def _robust_change_threshold(self, values: list[float]) -> float:
        ordered = sorted(values)
        median_value = ordered[len(ordered) // 2]
        deviations = sorted(abs(value - median_value) for value in values)
        mad = deviations[len(deviations) // 2]
        return median_value + 6 * 1.4826 * mad

    def _keyframe_labels(self, count: int) -> list[str]:
        if count == 1:
            return ["final frame"]
        if count == 2:
            return ["start frame", "final frame"]
        if count == 3:
            return ["start frame", "last gripper event", "final frame"]
        return [
            "start frame",
            *[f"gripper event frame {index}" for index in range(1, count - 1)],
            "final frame",
        ]

    def _ffmpeg_path(self) -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        imageio_ffmpeg = _import_imageio_ffmpeg()
        if imageio_ffmpeg is None:
            return None
        return str(imageio_ffmpeg.get_ffmpeg_exe())

    def _prompt(self, settings: VlmSettings, task: str) -> str:
        prompt = settings.prompt.strip()
        if "{task}" in prompt:
            return prompt.replace("{task}", task)
        return (
            f"{prompt}\n\nTask description: {task}\n"
            "Judge whether the task was accomplished. Return JSON like "
            '{"success": true, "score": 1.0, "reason": "brief explanation"}.'
        )

    def _parse_evaluation(self, text: str, raw_response: dict | None = None) -> VlmEvaluation:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise VlmEvaluationError(f"VLM returned non-JSON content: {text[:160]}")
            parsed = json.loads(text[start : end + 1])
        score_value = parsed.get("score")
        if isinstance(score_value, bool) or score_value is None:
            score = 1.0 if bool(parsed.get("success")) else 0.0
        else:
            score = max(0.0, min(1.0, float(score_value)))
        success = bool(parsed.get("success", score >= 0.5))
        reason = str(parsed.get("reason", "")).strip()
        return VlmEvaluation(success=success, score=score, reason=reason, raw_response=raw_response)
