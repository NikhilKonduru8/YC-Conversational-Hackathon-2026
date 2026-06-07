"""Wake node — detects "Hey Jarvis" on the shared mic stream.

Provides one service: audio/wait_wake, which blocks until the wake word fires
(openWakeWord) and publishes /jarvis/wake. The microphone (AudioInput) is shared
with the STT node and owned by the launch (jarvis.py).
"""

from __future__ import annotations

import logging

from audio_io import AudioInput
from bus import Bus
from config import WakeConfig
from topics import SVC_WAIT_WAKE, TOPIC_WAKE
from wake import WakeWord

logger = logging.getLogger("jarvis.node.wake")


class WakeNode:
    def __init__(self, bus: Bus, audio_input: AudioInput, wake: WakeConfig) -> None:
        self._bus = bus
        self._input = audio_input
        self._wake = WakeWord(wake)
        bus.register_service(SVC_WAIT_WAKE, self._wait_wake)

    async def _wait_wake(self, _request=None) -> float:
        self._wake.reset()
        while True:
            block = await self._input.read_block()
            triggered, score = self._wake.detect(block)
            if triggered:
                logger.info("Wake word detected (score %.2f)", score)
                await self._bus.publish(TOPIC_WAKE, {"score": score})
                return score
