"""Jarvis — standalone wearable engineering assistant (ROS-style node graph).

This file is the "launch": it builds the message bus, instantiates every node,
and runs the orchestrator. Each node owns one stage and talks only over the bus
(topics + services), so the graph is decoupled and ports to ROS 2 cleanly.

    uv run src/jarvis.py

Graph:
    [wake] --/wake--> [orchestrator] --services--> [stt] [vision] [retrieval]
                              |                              |
                              +--state--> [display]          +--/context
    [reasoning] --/response_chunk--> [tts] --> speaker
    everyone with a screen subscribes to /jarvis/state.

Degrades gracefully: no camera / no vision-Moss-MiniMax keys / a model hiccup
just trims that stage and continues. OLED no-ops off-Pi.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit.agents.utils import http_context  # noqa: E402

from audio_io import AudioInput  # noqa: E402
from bus import Bus  # noqa: E402
from config import load_config  # noqa: E402
from nodes.display_node import DisplayNode  # noqa: E402
from nodes.orchestrator_node import OrchestratorNode  # noqa: E402
from nodes.reasoning_node import ReasoningNode  # noqa: E402
from nodes.retrieval_node import RetrievalNode  # noqa: E402
from nodes.stt_node import STTNode  # noqa: E402
from nodes.tts_node import TTSNode  # noqa: E402
from nodes.vision_node import VisionNode  # noqa: E402
from nodes.wake_node import WakeNode  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis")


class Jarvis:
    """Composition root: wires the nodes onto a shared bus."""

    def __init__(self) -> None:
        self.cfg = load_config()
        self.bus = Bus()

        # One mic stream shared by wake detection + STT.
        self.audio_in = AudioInput(self.cfg.audio)

        # Nodes register their services/subscriptions on construction.
        self.wake = WakeNode(self.bus, self.audio_in, self.cfg.wake)
        self.stt = STTNode(self.bus, self.audio_in, self.cfg.stt, self.cfg.audio)
        self.vision = VisionNode(self.bus, self.cfg.camera, self.cfg.vision)
        self.retrieval = RetrievalNode(self.bus, self.cfg.moss)
        self.reasoning = ReasoningNode(self.bus, self.cfg.reasoning)
        self.tts = TTSNode(self.bus, self.cfg.tts, self.cfg.audio)
        self.display = DisplayNode(self.bus, self.cfg.oled)
        self.orchestrator = OrchestratorNode(
            self.bus, self.cfg.audio, self.cfg.wake.exit_phrases
        )

    async def run(self) -> None:
        # LiveKit Inference STT/TTS need a shared aiohttp session that the agent
        # worker normally provides. Running standalone, we open it ourselves.
        async with http_context.open():
            await self.retrieval.setup()
            self.audio_in.start()
            try:
                await self.orchestrator.run()
            except (KeyboardInterrupt, asyncio.CancelledError):
                logger.info("Shutting down…")
            finally:
                await self.shutdown()

    async def shutdown(self) -> None:
        self.audio_in.stop()
        self.vision.stop()
        await self.stt.aclose()
        await self.tts.aclose()
        self.display.clear()


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(Jarvis().run())


if __name__ == "__main__":
    main()
