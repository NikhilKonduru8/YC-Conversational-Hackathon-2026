"""Display node — mirrors /jarvis/state onto the OLED. Subscriber only."""

from __future__ import annotations

import logging

from bus import Bus
from config import OledConfig
from display import Display
from state import State
from topics import TOPIC_STATE

logger = logging.getLogger("jarvis.node.display")


class DisplayNode:
    def __init__(self, bus: Bus, config: OledConfig) -> None:
        self._display = Display(config)
        bus.subscribe(TOPIC_STATE, self._on_state)

    def _on_state(self, message) -> None:
        if isinstance(message, dict):
            state = message.get("state")
            detail = message.get("detail")
        else:
            state = message
            detail = None
        if isinstance(state, State):
            self._display.show_state(state, detail)

    def clear(self) -> None:
        self._display.clear()
