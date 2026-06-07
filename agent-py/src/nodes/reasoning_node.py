"""Reasoning node — MiniMax streamed as spoken sentences.

Publishes each complete sentence to /jarvis/response_chunk as it's generated, so
the TTS node can start speaking the first sentence while the rest streams in.
Because `bus.publish` awaits subscribers, publishing a sentence blocks until the
TTS node has finished speaking it — natural back-pressure with no queue.
"""

from __future__ import annotations

import logging

from bus import Bus
from config import ReasoningConfig
from reasoning import Reasoner
from topics import SVC_RESPOND, TOPIC_RESPONSE, TOPIC_RESPONSE_CHUNK

logger = logging.getLogger("jarvis.node.reasoning")

_SENTENCE_ENDINGS = ".!?"


def _pop_sentence(buffer: str) -> tuple[str | None, str]:
    """Pop the earliest complete sentence from `buffer` for low-latency TTS."""
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_ENDINGS and (
            i == len(buffer) - 1 or buffer[i + 1] in " \n\t"
        ):
            return buffer[: i + 1].strip(), buffer[i + 1 :]
    return None, buffer


# Keep the last few turns so follow-ups have context ("the yellow LED", "every
# 3 seconds"). Short, since each turn also carries fresh vision + Moss grounding.
_MAX_HISTORY_MESSAGES = 8


class ReasoningNode:
    def __init__(self, bus: Bus, config: ReasoningConfig) -> None:
        self._bus = bus
        self._reasoner = Reasoner(config)
        # Rolling conversation history: [{role, content}, ...] (user/assistant).
        self._history: list[dict] = []
        bus.register_service(SVC_RESPOND, self._respond)

    async def _respond(self, request) -> str:
        if isinstance(request, dict):
            grounding = request.get("grounding", "")
            user_text = request.get("user_text", "")
        else:  # backward-compatible: a bare grounding string
            grounding, user_text = request or "", ""

        buffer = ""
        spoken: list[str] = []
        async for token in self._reasoner.stream(grounding, self._history):
            buffer += token
            sentence, buffer = _pop_sentence(buffer)
            while sentence is not None:
                spoken.append(sentence)
                await self._bus.publish(TOPIC_RESPONSE_CHUNK, sentence)
                sentence, buffer = _pop_sentence(buffer)
        if buffer.strip():
            spoken.append(buffer.strip())
            await self._bus.publish(TOPIC_RESPONSE_CHUNK, buffer.strip())

        text = " ".join(spoken)
        await self._bus.publish(TOPIC_RESPONSE, text)

        # Remember this turn (the user's actual words + our answer), not the big
        # grounding block — keeps history small but conversational.
        if user_text and text:
            self._history.append({"role": "user", "content": user_text})
            self._history.append({"role": "assistant", "content": text})
            if len(self._history) > _MAX_HISTORY_MESSAGES:
                self._history = self._history[-_MAX_HISTORY_MESSAGES:]
        return text
