"""OLED I2C status display with a safe no-op fallback.

On the Raspberry Pi (with `uv sync --group pi` and an SSD1306 wired over I2C)
this shows the current stage. On a dev laptop the hardware libs aren't present,
so it transparently degrades to logging. The app calls `show()` either way.

Wiring (SSD1306): VCC->3V3, GND->GND, SDA->GPIO2, SCL->GPIO3. Enable I2C with
`sudo raspi-config`. Default address 0x3C.
"""

from __future__ import annotations

import logging
import threading

from config import OledConfig
from state import State, oled_detail, oled_title

logger = logging.getLogger("jarvis.display")


class Display:
    def __init__(self, config: OledConfig) -> None:
        self._config = config
        self._device = None
        self._lock = threading.Lock()
        self._enabled = False
        self._image_mod = None
        self._draw_mod = None
        self._font_big = None
        self._font_small = None

        if config.mode in ("0", "off", "false", "none"):
            logger.info("OLED disabled (JARVIS_OLED=%s)", config.mode)
            return

        try:
            self._init_hardware()
            self._enabled = True
            logger.info(
                "OLED ready (%dx%d @ I2C 0x%02x)",
                config.width,
                config.height,
                config.i2c_address,
            )
        except Exception as exc:
            if config.mode in ("1", "on", "true"):
                raise
            logger.info(
                "OLED not available (%s); continuing without it. "
                "On the Pi: uv sync --group pi",
                exc.__class__.__name__,
            )

    def _init_hardware(self) -> None:
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306
        from PIL import Image, ImageDraw, ImageFont

        serial = i2c(port=self._config.i2c_port, address=self._config.i2c_address)
        self._device = ssd1306(
            serial, width=self._config.width, height=self._config.height
        )
        self._image_mod = Image
        self._draw_mod = ImageDraw
        try:
            self._font_big = ImageFont.load_default(size=20)
            self._font_small = ImageFont.load_default(size=11)
        except TypeError:
            # Older Pillow: load_default() takes no size arg.
            self._font_big = ImageFont.load_default()
            self._font_small = ImageFont.load_default()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def show_state(self, state: State, detail: str | None = None) -> None:
        self.show(
            oled_title(state), detail if detail is not None else oled_detail(state)
        )

    def show(self, title: str, detail: str = "") -> None:
        if not self._enabled or self._device is None:
            logger.debug("OLED: %s | %s", title, detail)
            return
        try:
            with self._lock:
                image = self._image_mod.new("1", self._device.size)
                draw = self._draw_mod.Draw(image)
                draw.text((2, 2), title, font=self._font_big, fill=255)
                if detail:
                    y = 30
                    for line in _wrap(detail, 24)[:3]:
                        draw.text((2, y), line, font=self._font_small, fill=255)
                        y += 12
                self._device.display(image)
        except Exception:
            logger.exception("OLED render failed")

    def clear(self) -> None:
        if not self._enabled or self._device is None:
            return
        try:
            with self._lock:
                self._device.clear()
        except Exception:
            logger.exception("OLED clear failed")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
