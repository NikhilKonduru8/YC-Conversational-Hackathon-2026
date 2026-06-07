"""USB microphone capture + USB speaker playback (sounddevice / PortAudio).

Pi audio devices often don't support the rates we want (openWakeWord + the VAD
need 16 kHz mono in; the TTS emits ~24 kHz out), and PortAudio/ALSA won't
resample for you — it just errors with "Invalid sample rate". So this module
captures/plays at a rate the *device* supports and resamples in software:

  * AudioInput  — captures at the mic's native rate, resamples down to 16 kHz,
    and hands out fixed 80 ms (BLOCK_SIZE) int16 mono blocks.
  * AudioOutput — plays TTS frames, resampling to the speaker's supported rate
    if it can't take the frame's native rate. Interruptible for barge-in.

On the Pi you need PortAudio: `sudo apt install libportaudio2`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import queue
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from config import BLOCK_SIZE, SAMPLE_RATE, AudioConfig

logger = logging.getLogger("jarvis.audio")


def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample int16 mono audio between rates (no-op if rates match)."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    g = math.gcd(int(src_rate), int(dst_rate))
    up, down = dst_rate // g, src_rate // g
    out = resample_poly(samples.astype(np.float32), up, down)
    return np.clip(out, -32768, 32767).astype(np.int16)


def _fit(samples: np.ndarray, length: int) -> np.ndarray:
    """Pad or trim to exactly `length` samples (resampling can be off-by-a-few)."""
    if samples.size == length:
        return samples
    if samples.size > length:
        return samples[:length]
    return np.concatenate([samples, np.zeros(length - samples.size, dtype=np.int16)])


class AudioInput:
    """Always-on mic stream yielding 80 ms (BLOCK_SIZE) int16 mono blocks at 16 kHz."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream: sd.InputStream | None = None
        self._capture_rate = SAMPLE_RATE
        self._capture_block = BLOCK_SIZE

    def _select_capture_rate(self) -> int:
        device = self._config.input_device
        # Prefer 16 kHz (no resampling) if the device actually supports it.
        try:
            sd.check_input_settings(
                device=device, samplerate=SAMPLE_RATE, channels=1, dtype="int16"
            )
            return SAMPLE_RATE
        except Exception:
            pass
        try:
            info = sd.query_devices(device, "input")
            return int(info.get("default_samplerate") or 48000)
        except Exception:
            return 48000

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("input stream status: %s", status)
        block = indata[:, 0].copy()
        if self._capture_rate != SAMPLE_RATE:
            block = _resample_int16(block, self._capture_rate, SAMPLE_RATE)
        block = _fit(block, BLOCK_SIZE)
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            with contextlib.suppress(queue.Empty, queue.Full):
                self._queue.get_nowait()
                self._queue.put_nowait(block)

    def start(self) -> None:
        if self._stream is not None:
            return
        self._capture_rate = self._select_capture_rate()
        # 80 ms worth of native samples; resampled back to BLOCK_SIZE downstream.
        self._capture_block = max(
            1, round(BLOCK_SIZE * self._capture_rate / SAMPLE_RATE)
        )
        self._stream = sd.InputStream(
            samplerate=self._capture_rate,
            blocksize=self._capture_block,
            channels=1,
            dtype="int16",
            device=self._config.input_device,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Mic stream started (device=%s, capturing %d Hz -> 16000 Hz)",
            self._config.input_device,
            self._capture_rate,
        )

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    async def read_block(self) -> np.ndarray:
        return await asyncio.to_thread(self._queue.get)

    def stop(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None


class AudioOutput:
    """Plays int16 frames to the speaker (resampling if needed); interruptible."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._stream: sd.OutputStream | None = None
        self._src_rate: int | None = None  # TTS frame rate we're set up for
        self._out_rate: int | None = None  # actual device rate
        self._out_channels = 1
        self._out_dtype = "int16"
        self._out_device: object = None
        self._broken = False  # all output configs failed; give up quietly
        self._interrupt = threading.Event()

    def _ensure_stream(self, src_rate: int) -> None:
        if self._stream is not None and self._src_rate == src_rate:
            return
        if self._broken:
            return
        self.close()
        self._src_rate = src_rate

        # Pi audio devices are picky about format/channels/rate (HDMI usually
        # wants stereo, and the raw device may reject mono or int16). Probe a
        # range of combinations — and fall back to the default device — until
        # one opens. The first that works is cached for the rest of the session.
        device = self._config.output_device
        devices = [device, None] if device is not None else [None]
        rates = _dedup([src_rate, 48000, 44100])
        last_err: Exception | None = None
        for dev in _dedup(devices):
            for rate in rates:
                for channels in (2, 1):
                    for dtype in ("int16", "float32"):
                        try:
                            stream = sd.OutputStream(
                                samplerate=rate,
                                channels=channels,
                                dtype=dtype,
                                device=dev,
                            )
                            stream.start()
                        except Exception as exc:
                            last_err = exc
                            continue
                        self._stream = stream
                        self._out_rate = rate
                        self._out_channels = channels
                        self._out_dtype = dtype
                        self._out_device = dev
                        logger.info(
                            "Speaker open: device=%s rate=%d ch=%d dtype=%s",
                            dev,
                            rate,
                            channels,
                            dtype,
                        )
                        return

        self._broken = True
        logger.error(
            "Could not open any speaker output (last error: %s). "
            "Audio out disabled for this session; check AUDIO_OUTPUT_DEVICE.",
            last_err,
        )

    def interrupt(self) -> None:
        self._interrupt.set()

    def begin(self) -> None:
        self._interrupt.clear()

    @property
    def interrupted(self) -> bool:
        return self._interrupt.is_set()

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._interrupt.is_set() or self._broken:
            return
        await asyncio.to_thread(self._play_sync, samples, sample_rate)

    def _play_sync(self, samples: np.ndarray, sample_rate: int) -> None:
        if self._interrupt.is_set():
            return
        self._ensure_stream(sample_rate)
        if self._stream is None:
            return
        out = samples
        if self._out_rate != sample_rate:
            out = _resample_int16(out, sample_rate, self._out_rate)
        if self._out_dtype == "float32":
            out = out.astype(np.float32) / 32768.0
        if self._out_channels == 2:
            out = np.repeat(out.reshape(-1, 1), 2, axis=1)  # mono -> stereo
        with contextlib.suppress(Exception):
            self._stream.write(out)

    def close(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None
            self._src_rate = None
            self._out_rate = None


def _dedup(items: list) -> list:
    seen: list = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
