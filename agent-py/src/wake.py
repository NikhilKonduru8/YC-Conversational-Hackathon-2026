"""Local "Hey Jarvis" wake-word detection via openWakeWord.

Runs fully on-device (no API key). Feed it 80 ms int16 blocks; it returns the
current confidence for the wake model. On the first run it downloads the small
pretrained models (melspectrogram, embedding, and the hey_jarvis classifier).
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np

from config import WakeConfig

logger = logging.getLogger("jarvis.wake")


class WakeWord:
    def __init__(self, config: WakeConfig) -> None:
        self._config = config
        self._model = None
        self._load()

    def _load(self) -> None:
        from openwakeword import utils as ow_utils
        from openwakeword.model import Model

        # Idempotent; downloads the bundled models on first run only.
        try:
            ow_utils.download_models()
        except Exception:
            logger.debug("openWakeWord model download skipped/failed (already cached?)")

        self._model = Model(
            wakeword_models=[self._config.model],
            inference_framework=self._config.framework,
        )
        logger.info(
            "Wake word ready: '%s' (threshold %.2f, %s)",
            self._config.model,
            self._config.threshold,
            self._config.framework,
        )

    def reset(self) -> None:
        """Clear the model's internal audio buffer between activations."""
        if self._model is not None:
            with contextlib.suppress(Exception):
                self._model.reset()

    def detect(self, block: np.ndarray) -> tuple[bool, float]:
        """Feed an 80 ms int16 block; return (triggered, score)."""
        if self._model is None:
            return False, 0.0
        scores = self._model.predict(block)
        score = float(scores.get(self._config.model, 0.0))
        return score >= self._config.threshold, score
