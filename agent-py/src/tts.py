"""Text-to-speech via LiveKit Inference, streamed to the USB speaker.

Sentences are synthesized and played as they arrive from the reasoning stream,
so the user hears the first sentence while the rest is still generating. Playback
checks the AudioOutput interrupt flag between frames so a barge-in stops it fast.
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
from livekit.agents import inference

from audio_io import AudioOutput
from config import TTSConfig

logger = logging.getLogger("jarvis.tts")


class TextToSpeech:
    def __init__(self, config: TTSConfig, output: AudioOutput) -> None:
        # speed (Cartesia control) makes the voice talk faster without changing
        # pitch — "fast" ≈ 1.3x. Passed as provider extra_kwargs.
        self._tts = inference.TTS(
            model=config.model,
            voice=config.voice,
            extra_kwargs={"speed": config.speed},
        )
        self._output = output
        logger.info(
            "TTS ready: %s (voice %s, speed %s)",
            config.model,
            config.voice,
            config.speed,
        )

    async def speak(self, text: str) -> bool:
        """Synthesize + play one chunk of text. Returns False if interrupted."""
        text = (text or "").strip()
        if not text:
            return True
        if self._output.interrupted:
            return False

        stream = self._tts.synthesize(text)
        try:
            async for synth in stream:
                if self._output.interrupted:
                    return False
                frame = synth.frame
                samples = np.frombuffer(bytes(frame.data), dtype=np.int16)
                await self._output.play(samples, frame.sample_rate)
        except Exception:
            logger.exception("TTS synthesis/playback failed")
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
        return not self._output.interrupted

    async def aclose(self) -> None:
        with contextlib.suppress(Exception):
            await self._tts.aclose()
