"""Retrieval node — fuses transcript + vision into a Moss query and grounding."""

from __future__ import annotations

import logging

from moss import MossClient

from bus import Bus
from config import MossConfig
from moss_context import ContextCompiler
from topics import SVC_COMPILE, TOPIC_CONTEXT

logger = logging.getLogger("jarvis.node.retrieval")


class RetrievalNode:
    def __init__(self, bus: Bus, config: MossConfig) -> None:
        self._bus = bus
        self._cfg = config
        self._moss = MossClient(config.project_id, config.project_key)
        self._compiler = ContextCompiler(self._moss, config.index, top_k=config.top_k)
        bus.register_service(SVC_COMPILE, self._compile)

    async def setup(self) -> None:
        if not (self._cfg.project_id and self._cfg.project_key):
            logger.warning("Moss not configured; spec retrieval disabled.")
            return
        try:
            logger.info(
                "Loading Moss index '%s' (first run downloads the model — "
                "can take a minute)...",
                self._cfg.index,
            )
            await self._moss.load_index(self._cfg.index)
            logger.info("Loaded Moss index '%s'", self._cfg.index)
        except Exception:
            logger.exception("Failed to preload Moss index; will retry on first query")

    async def _compile(self, request) -> str:
        transcript = request.get("transcript", "") if isinstance(request, dict) else ""
        vision = request.get("vision", "") if isinstance(request, dict) else ""
        compiled = await self._compiler.compile(transcript, vision)
        logger.info(
            "Moss query: %s (%d matches)", compiled.moss_query, len(compiled.matches)
        )
        await self._bus.publish(
            TOPIC_CONTEXT,
            {"query": compiled.moss_query, "matches": compiled.matches},
        )
        return compiled.grounding_prompt()
