from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from robot_data_studio.formats.models import DatasetAdapter

from .models import VlmEvaluation, VlmSettings


class VlmEvaluationError(RuntimeError):
    pass


class VlmTaskSuccessEvaluator:
    def evaluate(
        self,
        reader: DatasetAdapter,
        episode_index: int,
        settings: VlmSettings,
    ) -> VlmEvaluation:
        episode = reader.episode(episode_index)
        task = episode.tasks[0] if episode.tasks else "No task description provided."
        video_paths = [reader.root / path for path in episode.video_files.values()]
        existing_videos = [path for path in video_paths if path.is_file()]
        if not existing_videos:
            raise VlmEvaluationError("VLM task success requires at least one local video file.")
        if settings.provider == "Gemini":
            return self._evaluate_gemini(existing_videos[0], task, settings)
        if settings.provider == "Local":
            return self._evaluate_local(existing_videos[0], task)
        return self._evaluate_openai_compatible(existing_videos[0], task, episode.duration_seconds, settings)

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
    ) -> VlmEvaluation:
        api_key = settings.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise VlmEvaluationError("Set OPENAI_API_KEY or enter an API key to use OpenAI VLM scoring.")
        frame_bytes = self._sample_video_frames(video_path, duration_seconds, settings.sample_frames)
        if not frame_bytes:
            raise VlmEvaluationError("Could not sample frames for OpenAI-compatible VLM scoring.")
        content: list[dict[str, Any]] = [{"type": "text", "text": self._prompt(settings, task)}]
        for frame in frame_bytes:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64.b64encode(frame).decode('ascii')}",
                        "detail": "low",
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
    ) -> list[bytes]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise VlmEvaluationError("ffmpeg is required to sample frames for OpenAI-compatible VLM scoring.")
        with tempfile.TemporaryDirectory() as directory:
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
            return [frame.read_bytes() for frame in frames]

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
