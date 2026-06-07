"""Unit tests for camera frame encoding + Qwen guards (no real camera/API)."""

import numpy as np

from config import VisionConfig
from vision import QwenVision, _frame_to_data_url


def _frame(h, w):
    # A non-uniform BGR frame so JPEG encodes to a plausible size.
    rng = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    return rng


def test_frame_to_data_url_encodes_valid_frame():
    url = _frame_to_data_url(_frame(120, 160), VisionConfig())
    assert isinstance(url, str)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 256


def test_frame_to_data_url_downsamples_large_frame():
    cfg = VisionConfig()  # frame_max_edge default 768
    big = _frame(1080, 1920)
    url = _frame_to_data_url(big, cfg)
    assert url is not None  # still encodes after resize


def test_frame_to_data_url_rejects_degenerate_frame():
    assert _frame_to_data_url(_frame(2, 2), VisionConfig()) is None


def test_qwen_not_configured_is_not_ready():
    v = QwenVision(VisionConfig())  # no base_url/api_key
    assert v.ready is False


async def test_qwen_describe_returns_empty_without_config():
    v = QwenVision(VisionConfig())
    summary = await v.describe([_frame(120, 160)], "what is this")
    assert summary == ""
