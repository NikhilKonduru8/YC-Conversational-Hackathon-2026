"""Central configuration for the standalone Jarvis device.

All tunables come from environment variables (loaded from ``.env.local`` by
``jarvis.py``). Every value has a safe default so the app imports on a dev
laptop before any keys are set; live model calls obviously still need real keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_device(name: str):
    """Audio device index (int) or name (str) or None for the system default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_speed(name: str, default):
    """TTS speed: a float multiplier or a string ('slow'/'normal'/'fast')."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        return raw.lower()


# Moss index name (kept in sync with create_index.py).
KNOWLEDGE_INDEX = _env("MOSS_INDEX_NAME", "knowledge") or "knowledge"

# Audio runs at 16 kHz mono — required by openWakeWord and webrtcvad.
SAMPLE_RATE = 16000
# 80 ms blocks (= 4 x 20 ms VAD frames). openWakeWord expects 1280-sample chunks.
BLOCK_SIZE = 1280
VAD_FRAME = 320  # 20 ms


@dataclass(frozen=True)
class WakeConfig:
    model: str = field(default_factory=lambda: _env("WAKE_MODEL", "hey_jarvis"))
    threshold: float = field(default_factory=lambda: _env_float("WAKE_THRESHOLD", 0.5))
    framework: str = field(
        default_factory=lambda: _env("WAKE_INFERENCE_FRAMEWORK", "onnx") or "onnx"
    )
    # Saying any of these quits the app ("Okay, done" on the OLED, then exit).
    exit_phrases: list[str] = field(
        default_factory=lambda: _env_list(
            "JARVIS_EXIT_PHRASES",
            [
                "jarvis exit",
                "exit jarvis",
                "jarvis quit",
                "quit jarvis",
                "jarvis goodbye",
                "goodbye jarvis",
                "jarvis shut down",
            ],
        )
    )


@dataclass(frozen=True)
class AudioConfig:
    input_device: object = field(
        default_factory=lambda: _env_device("AUDIO_INPUT_DEVICE")
    )
    output_device: object = field(
        default_factory=lambda: _env_device("AUDIO_OUTPUT_DEVICE")
    )
    # End an utterance after this much trailing silence (lower = snappier).
    silence_ms: int = field(
        default_factory=lambda: _env_int("UTTERANCE_SILENCE_MS", 450)
    )
    # Hard cap on a single utterance.
    max_utterance_s: float = field(
        default_factory=lambda: _env_float("UTTERANCE_MAX_S", 12.0)
    )
    # Require this much speech before we accept an utterance (filters coughs).
    min_speech_ms: int = field(
        default_factory=lambda: _env_int("UTTERANCE_MIN_MS", 300)
    )
    vad_aggressiveness: int = field(
        default_factory=lambda: _env_int("VAD_AGGRESSIVENESS", 2)
    )
    # Barge-in: say "Hey Jarvis" while it's talking to interrupt and start a new
    # request. Safe because TTS never says the wake phrase. Disable with BARGE_IN=0.
    barge_in: bool = field(default_factory=lambda: _env_bool("BARGE_IN", True))


@dataclass(frozen=True)
class STTConfig:
    model: str = field(
        default_factory=lambda: (
            _env("STT_MODEL", "deepgram/nova-3") or "deepgram/nova-3"
        )
    )
    language: str = field(default_factory=lambda: _env("STT_LANGUAGE", "en") or "en")


@dataclass(frozen=True)
class TTSConfig:
    model: str = field(
        default_factory=lambda: (
            _env("TTS_MODEL", "cartesia/sonic-3") or "cartesia/sonic-3"
        )
    )
    voice: str = field(
        default_factory=lambda: _env(
            "TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        )
    )
    # Speaking rate: "fast" (≈1.3x, default), "normal", "slow", or a float.
    speed: object = field(default_factory=lambda: _env_speed("TTS_SPEED", "fast"))


@dataclass(frozen=True)
class CameraConfig:
    index: int = field(default_factory=lambda: _env_int("CAMERA_INDEX", 0))
    # Frames to read (and discard the older ones) per capture, to let the sensor
    # settle and give the vision model a small temporal spread.
    capture_frames: int = field(default_factory=lambda: _env_int("CAMERA_FRAMES", 3))
    warmup_frames: int = field(default_factory=lambda: _env_int("CAMERA_WARMUP", 3))


@dataclass(frozen=True)
class VisionConfig:
    # For Qwen3-VL on DashScope this is the workspace MaaS endpoint, e.g.
    # https://<workspace-id>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
    base_url: str = field(default_factory=lambda: _env("QWEN_BASE_URL"))
    api_key: str = field(default_factory=lambda: _env("QWEN_API_KEY"))
    model: str = field(
        default_factory=lambda: _env("QWEN_MODEL", "qwen3-vl-flash") or "qwen3-vl-flash"
    )
    max_frames: int = field(default_factory=lambda: _env_int("QWEN_MAX_FRAMES", 3))
    frame_max_edge: int = field(
        default_factory=lambda: _env_int("QWEN_FRAME_MAX_EDGE", 768)
    )
    jpeg_quality: int = field(default_factory=lambda: _env_int("QWEN_JPEG_QUALITY", 70))
    timeout_s: float = field(default_factory=lambda: _env_float("QWEN_TIMEOUT_S", 20.0))
    # Qwen3-VL "thinking" — keep off for fast, direct scene descriptions.
    enable_thinking: bool = field(
        default_factory=lambda: _env_bool("QWEN_ENABLE_THINKING", False)
    )


@dataclass(frozen=True)
class ReasoningConfig:
    """MiniMax reasoning engine, called directly (OpenAI-compatible API)."""

    base_url: str = field(
        default_factory=lambda: (
            _env("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
            or "https://api.minimax.io/v1"
        )
    )
    api_key: str = field(default_factory=lambda: _env("MINIMAX_API_KEY"))
    # MiniMax-M3 supports disabling thinking (fast, direct answers). M2.x models
    # always think (slow + leak <think> tags), so M3 is the default here.
    model: str = field(
        default_factory=lambda: _env("MINIMAX_MODEL", "MiniMax-M3") or "MiniMax-M3"
    )
    temperature: float = field(
        default_factory=lambda: _env_float("MINIMAX_TEMPERATURE", 0.3)
    )
    max_tokens: int = field(default_factory=lambda: _env_int("MINIMAX_MAX_TOKENS", 256))
    timeout_s: float = field(
        default_factory=lambda: _env_float("MINIMAX_TIMEOUT_S", 30.0)
    )
    # Disable model "thinking" for low latency + clean output (works on M3;
    # harmlessly ignored by models that don't support it).
    disable_thinking: bool = field(
        default_factory=lambda: _env_bool("MINIMAX_DISABLE_THINKING", True)
    )


@dataclass(frozen=True)
class OledConfig:
    mode: str = field(default_factory=lambda: _env("JARVIS_OLED", "auto").lower())
    i2c_port: int = field(default_factory=lambda: _env_int("JARVIS_OLED_I2C_PORT", 1))
    i2c_address: int = field(
        default_factory=lambda: int(_env("JARVIS_OLED_ADDR", "0x3C"), 16)
    )
    width: int = field(default_factory=lambda: _env_int("JARVIS_OLED_WIDTH", 128))
    height: int = field(default_factory=lambda: _env_int("JARVIS_OLED_HEIGHT", 64))


@dataclass(frozen=True)
class MossConfig:
    project_id: str = field(default_factory=lambda: _env("MOSS_PROJECT_ID"))
    project_key: str = field(default_factory=lambda: _env("MOSS_PROJECT_KEY"))
    index: str = field(default_factory=lambda: KNOWLEDGE_INDEX)
    model_id: str = field(
        default_factory=lambda: _env("MOSS_MODEL_ID", "moss-minilm") or "moss-minilm"
    )
    top_k: int = field(default_factory=lambda: _env_int("MOSS_TOP_K", 4))
    # Directory of component JSON files indexed for semantic search. Sized for
    # ~100 MB of data, so the indexer streams + batches (see create_index.py).
    data_dir: str = field(
        default_factory=lambda: (
            _env("JARVIS_DATA_DIR", str(AGENT_DIR / "data")) or str(AGENT_DIR / "data")
        )
    )
    # How many chunks to push to Moss per batch when (re)building the index.
    index_batch_size: int = field(
        default_factory=lambda: _env_int("MOSS_INDEX_BATCH_SIZE", 256)
    )


@dataclass(frozen=True)
class Config:
    wake: WakeConfig = field(default_factory=WakeConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    oled: OledConfig = field(default_factory=OledConfig)
    moss: MossConfig = field(default_factory=MossConfig)

    @property
    def vision_ready(self) -> bool:
        return bool(self.vision.base_url and self.vision.api_key)

    @property
    def reasoning_ready(self) -> bool:
        return bool(self.reasoning.base_url and self.reasoning.api_key)

    @property
    def moss_ready(self) -> bool:
        return bool(self.moss.project_id and self.moss.project_key)


def load_config() -> Config:
    return Config()
