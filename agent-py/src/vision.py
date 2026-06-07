"""USB camera capture + Qwen-2.5-VL visual analysis.

`Camera` wraps an OpenCV VideoCapture on the USB webcam and grabs a small burst
of frames on demand. `QwenVision` downsamples + JPEG-encodes 1-3 frames and asks
an OpenAI-compatible Qwen-2.5-VL endpoint for a dense, factual, text-only scene
description focused on electrical components. Everything degrades gracefully:
a missing camera or a failed inference yields an empty summary and the pipeline
continues from speech + Moss alone.
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

from config import CameraConfig, VisionConfig

logger = logging.getLogger("jarvis.vision")


class Camera:
    """On-demand burst capture from a USB camera (OpenCV)."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        cap = cv2.VideoCapture(self._config.index)
        if not cap.isOpened():
            logger.warning("Could not open camera index %d", self._config.index)
            self._cap = None
            return False
        self._cap = cap
        # Warm up the sensor so the first real frame isn't black/auto-exposing.
        for _ in range(max(0, self._config.warmup_frames)):
            cap.read()
        logger.info("Camera opened (index %d)", self._config.index)
        return True

    def grab(self, count: int) -> list[np.ndarray]:
        """Grab up to `count` recent BGR frames; [] if the camera is unavailable."""
        if not self.open() or self._cap is None:
            return []
        frames: list[np.ndarray] = []
        for _ in range(max(1, count)):
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                frames.append(frame)
        return frames

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


_VISION_SYSTEM_PROMPT = (
    "You are a precise machine-vision module for a wearable engineering "
    "assistant. You receive 1-3 frames from a camera worn by an electronics "
    "engineer looking at a workbench, breadboard, or PCB.\n\n"
    "Produce a DENSE, FACTUAL, TEXT-ONLY description of exactly what is visible. "
    "Be specific and exhaustive about:\n"
    "- ICs/chips: read and transcribe EVERY part number / silkscreen label "
    "verbatim (e.g. 'LM358', 'ATMEGA328P-PU', 'NE555').\n"
    "- Resistors, capacitors, diodes, transistors, crystals, connectors, headers.\n"
    "- LEDs: how many, their colours, and which appear lit vs unlit.\n"
    "- Displays, switches, potentiometers, buttons, jumpers.\n"
    "- Any printed text, pin labels, or board markings, transcribed exactly.\n"
    "- Wiring and spatial layout: what connects to what, relative positions.\n\n"
    "Rules: Report only what you actually see. Do NOT speculate about function or "
    "give advice. If a label is partially legible, transcribe what you can and "
    "mark it uncertain. Output 2-6 plain-text sentences, no markdown, no lists."
)


def _frame_to_data_url(frame: np.ndarray, cfg: VisionConfig) -> str | None:
    """Downsample + JPEG-encode a BGR frame to a base64 data URL, or None."""
    try:
        h, w = frame.shape[:2]
        if w < 8 or h < 8:
            return None
        longest = max(w, h)
        if longest > cfg.frame_max_edge:
            scale = cfg.frame_max_edge / float(longest)
            frame = cv2.resize(frame, (max(8, int(w * scale)), max(8, int(h * scale))))
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]
        )
        if not ok or buf is None or len(buf) < 256:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        logger.exception("Failed to encode frame; skipping it")
        return None


class QwenVision:
    """OpenAI-compatible Qwen-2.5-VL client for scene description."""

    def __init__(self, config: VisionConfig) -> None:
        self._cfg = config
        self._client = None
        if config.base_url and config.api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_s,
            )
            logger.info("Qwen vision ready: %s", config.model)
        else:
            logger.warning(
                "Qwen not configured (QWEN_BASE_URL/QWEN_API_KEY); vision skipped."
            )

    @property
    def ready(self) -> bool:
        return self._client is not None

    async def describe(self, frames: list[np.ndarray], user_request: str = "") -> str:
        """Return a dense factual scene summary, or "" on any failure/no input."""
        if not self.ready or not frames:
            return ""

        # Keep at most max_frames, spread across the captured burst.
        n = max(1, min(self._cfg.max_frames, 3, len(frames)))
        step = max(1, len(frames) // n)
        chosen = frames[::step][:n]

        data_urls = [u for u in (_frame_to_data_url(f, self._cfg) for f in chosen) if u]
        if not data_urls:
            logger.warning("All captured frames were unusable; skipping vision")
            return ""

        focus = (
            f'\n\nThe engineer just asked: "{user_request.strip()}". Pay extra '
            "attention to anything relevant, but only describe what is visible."
            if user_request.strip()
            else ""
        )
        content = [
            {"type": "text", "text": "Describe these frames." + focus},
            *({"type": "image_url", "image_url": {"url": u}} for u in data_urls),
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self._cfg.model,
                messages=[
                    {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.0,
                max_tokens=400,
                # Qwen3-VL reasoning toggle (off = faster, direct description).
                extra_body={"enable_thinking": self._cfg.enable_thinking},
            )
            summary = (resp.choices[0].message.content or "").strip()
            logger.info(
                "Qwen summary: %d frame(s), %d chars", len(data_urls), len(summary)
            )
            return summary
        except Exception:
            logger.exception("Qwen vision inference failed; proceeding without vision")
            return ""
