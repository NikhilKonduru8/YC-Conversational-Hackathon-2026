"""TTS node — speaks response sentences and owns the speaker.

Subscribes to /jarvis/response_chunk and speaks each sentence (LiveKit TTS →
USB speaker). Resets its interrupt flag when the orchestrator enters THINKING
(a new answer is starting), and stops immediately on /jarvis/interrupt.
"""

from __future__ import annotations

import logging

from audio_io import AudioOutput
from bus import Bus
from config import AudioConfig, TTSConfig
from state import State
from topics import TOPIC_INTERRUPT, TOPIC_RESPONSE_CHUNK, TOPIC_STATE
from tts import TextToSpeech

logger = logging.getLogger("jarvis.node.tts")


class TTSNode:
    def __init__(self, bus: Bus, tts: TTSConfig, audio: AudioConfig) -> None:
        self._output = AudioOutput(audio)
        self._tts = TextToSpeech(tts, self._output)
        bus.subscribe(TOPIC_RESPONSE_CHUNK, self._on_chunk)
        bus.subscribe(TOPIC_INTERRUPT, self._on_interrupt)
        bus.subscribe(TOPIC_STATE, self._on_state)

    async def _on_chunk(self, sentence: str) -> None:
        await self._tts.speak(sentence)

    def _on_interrupt(self, _message=None) -> None:
        logger.info("Interrupt: clearing TTS playout")
        self._output.interrupt()

    def _on_state(self, message) -> None:
        # A new turn is about to be answered: clear any prior interrupt flag.
        state = message.get("state") if isinstance(message, dict) else message
        if state == State.THINKING:
            self._output.begin()

    async def aclose(self) -> None:
        self._output.interrupt()
        await self._tts.aclose()
        self._output.close()
