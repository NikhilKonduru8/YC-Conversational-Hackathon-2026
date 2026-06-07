"""STT node — streams the mic to LiveKit Inference STT in real time.

`audio/listen` reads mic blocks while the user speaks, pushing each to the STT
stream live (so transcription happens during speech, not after), uses webrtcvad
to detect end-of-speech, then gathers the final transcript. This is both more
accurate (no batch-push truncation) and lower latency than record-then-transcribe.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import webrtcvad

from audio_io import AudioInput
from bus import Bus
from config import BLOCK_SIZE, SAMPLE_RATE, VAD_FRAME, AudioConfig, STTConfig
from stt import SpeechToText
from topics import SVC_LISTEN, TOPIC_TRANSCRIPT

logger = logging.getLogger("jarvis.node.stt")

_BLOCK_MS = int(BLOCK_SIZE / SAMPLE_RATE * 1000)
# After end-of-speech, wait at most this long for the final segment to flush.
_FINAL_WAIT_S = 1.2


class STTNode:
    def __init__(
        self,
        bus: Bus,
        audio_input: AudioInput,
        stt: STTConfig,
        audio: AudioConfig,
    ) -> None:
        self._bus = bus
        self._input = audio_input
        self._cfg = audio
        self._stt = SpeechToText(stt)
        self._vad = webrtcvad.Vad(audio.vad_aggressiveness)
        bus.register_service(SVC_LISTEN, self._listen)

    def _is_voiced(self, block) -> bool:
        for i in range(0, BLOCK_SIZE, VAD_FRAME):
            frame = block[i : i + VAD_FRAME]
            if frame.size < VAD_FRAME:
                break
            if self._vad.is_speech(frame.tobytes(), SAMPLE_RATE):
                return True
        return False

    async def _listen(self, _request=None) -> str:
        self._input.flush()
        logger.info("Listening (speak now)...")
        stream = self._stt.new_stream()
        parts: list[str] = []

        async def collect() -> None:
            from livekit.agents import stt as lk_stt

            with contextlib.suppress(Exception):
                async for event in stream:
                    if (
                        event.type == lk_stt.SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives
                    ):
                        text = (event.alternatives[0].text or "").strip()
                        if text:
                            parts.append(text)

        collector = asyncio.create_task(collect())

        speech_ms = silence_ms = total_ms = 0
        max_ms = int(self._cfg.max_utterance_s * 1000)
        try:
            while total_ms < max_ms:
                block = await self._input.read_block()
                # Feed STT live so it transcribes as the user speaks.
                with contextlib.suppress(Exception):
                    stream.push_frame(self._stt.frame(block))
                total_ms += _BLOCK_MS
                if self._is_voiced(block):
                    speech_ms += _BLOCK_MS
                    silence_ms = 0
                else:
                    silence_ms += _BLOCK_MS
                if (
                    speech_ms >= self._cfg.min_speech_ms
                    and silence_ms >= self._cfg.silence_ms
                ):
                    break
        finally:
            with contextlib.suppress(Exception):
                stream.end_input()

        if speech_ms < self._cfg.min_speech_ms:
            await self._stop(collector, stream)
            logger.info("Too little speech detected; ignoring.")
            return ""

        # Speech already streamed live; just wait briefly for the last final.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(collector, timeout=_FINAL_WAIT_S)
        await self._stop(collector, stream)

        text = " ".join(parts).strip()
        logger.info("Heard: %r", text)
        if text:
            await self._bus.publish(TOPIC_TRANSCRIPT, text)
        return text

    @staticmethod
    async def _stop(collector: asyncio.Task, stream) -> None:
        if not collector.done():
            collector.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await collector
        with contextlib.suppress(Exception):
            await stream.aclose()

    async def aclose(self) -> None:
        await self._stt.aclose()
