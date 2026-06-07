"""Speech-to-text via LiveKit Inference (streaming).

LiveKit Inference STT only supports the streaming API, and it must be fed in
real time — batch-pushing a whole clip only finalizes the first segment. So the
STT node (nodes/stt_node.py) pushes mic frames live while the user speaks. This
module is a thin wrapper that opens a stream and builds audio frames.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from livekit import rtc
from livekit.agents import inference

from config import SAMPLE_RATE, STTConfig

logger = logging.getLogger("jarvis.stt")


class SpeechToText:
    def __init__(self, config: STTConfig) -> None:
        self._stt = inference.STT(model=config.model, language=config.language)
        logger.info("STT ready: %s (%s)", config.model, config.language)

    def new_stream(self):
        """Open a fresh streaming recognition session."""
        return self._stt.stream()

    @staticmethod
    def frame(block: np.ndarray, sample_rate: int = SAMPLE_RATE) -> rtc.AudioFrame:
        """Wrap an int16 mono block as an rtc.AudioFrame for push_frame."""
        return rtc.AudioFrame(
            data=block.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=int(block.size),
        )

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._stt.aclose()
