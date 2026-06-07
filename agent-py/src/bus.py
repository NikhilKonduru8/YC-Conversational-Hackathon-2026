"""A tiny in-process message bus modelled on ROS topics + services.

This gives the project a ROS-style architecture (decoupled nodes that talk over
named channels) without requiring a ROS 2 install. Each pipeline stage is a node
that publishes/subscribes to **topics** and/or answers **services**:

  * topics   — fan-out, fire-and-forget (e.g. /jarvis/state, /jarvis/transcript)
  * services — request/response, one handler (e.g. stt/transcribe)

`publish` awaits every subscriber, which gives natural back-pressure (e.g. the
reasoning node publishing a sentence blocks until the TTS node has spoken it).

Porting to real ROS 2 later means swapping these calls for rclpy
publishers/subscribers and service clients — the node logic stays the same.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("jarvis.bus")


class Bus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._services: dict[str, Callable] = {}

    # --- topics (pub/sub) ------------------------------------------------- #
    def subscribe(self, topic: str, callback: Callable) -> None:
        self._subscribers[topic].append(callback)

    async def publish(self, topic: str, message: Any = None) -> None:
        for callback in list(self._subscribers.get(topic, ())):
            try:
                result = callback(message)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Subscriber for %s failed", topic)

    # --- services (request/response) -------------------------------------- #
    def register_service(self, name: str, handler: Callable) -> None:
        if name in self._services:
            logger.warning("Service %s re-registered", name)
        self._services[name] = handler

    async def call(self, name: str, request: Any = None) -> Any:
        handler = self._services.get(name)
        if handler is None:
            raise KeyError(f"No service registered for '{name}'")
        result = handler(request)
        return await result if inspect.isawaitable(result) else result

    def has_service(self, name: str) -> bool:
        return name in self._services
