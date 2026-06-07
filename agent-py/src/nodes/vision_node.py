"""Vision node — captures USB camera frames and describes them with Qwen-2.5-VL."""

from __future__ import annotations

import asyncio
import logging

from bus import Bus
from config import CameraConfig, VisionConfig
from topics import SVC_DESCRIBE, TOPIC_VISION
from vision import Camera, QwenVision

logger = logging.getLogger("jarvis.node.vision")


class VisionNode:
    def __init__(self, bus: Bus, camera: CameraConfig, vision: VisionConfig) -> None:
        self._bus = bus
        self._cfg = camera
        self._camera = Camera(camera)
        self._vision = QwenVision(vision)
        bus.register_service(SVC_DESCRIBE, self._describe)

    async def _describe(self, request) -> str:
        transcript = ""
        if isinstance(request, dict):
            transcript = request.get("transcript", "")
        elif isinstance(request, str):
            transcript = request
        # Camera I/O is blocking; keep it off the event loop.
        frames = await asyncio.to_thread(self._camera.grab, self._cfg.capture_frames)
        summary = await self._vision.describe(frames, transcript)
        await self._bus.publish(TOPIC_VISION, summary)
        return summary

    def stop(self) -> None:
        self._camera.close()
