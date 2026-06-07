"""Orchestrator node — the state machine that drives one turn at a time.

It calls the other nodes' services in sequence and publishes /jarvis/state so the
display (and TTS) react. It does not import other nodes — only the bus — so the
graph stays decoupled (and ports to a ROS 2 node cleanly).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re

from bus import Bus
from config import AudioConfig
from state import State
from topics import (
    SVC_COMPILE,
    SVC_DESCRIBE,
    SVC_LISTEN,
    SVC_RESPOND,
    SVC_WAIT_WAKE,
    TOPIC_INTERRUPT,
    TOPIC_STATE,
)

logger = logging.getLogger("jarvis.node.orchestrator")

# Single-word transcripts that also mean "quit" (e.g. after a barge-in).
_EXIT_EXACT = {"exit", "quit", "goodbye"}
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (for phrase matching)."""
    return _WS.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


class _ExitRequested(Exception):  # noqa: N818 - control-flow signal, not an error
    """Raised internally when the user says an exit phrase."""


class OrchestratorNode:
    def __init__(
        self, bus: Bus, audio: AudioConfig, exit_phrases: list[str] | None = None
    ) -> None:
        self._bus = bus
        self._barge_in = audio.barge_in
        self._exit_phrases = [_normalize(p) for p in (exit_phrases or []) if p.strip()]

    async def set_state(self, state: State, detail: str = "") -> None:
        await self._bus.publish(TOPIC_STATE, {"state": state, "detail": detail})

    def _is_exit(self, transcript: str) -> bool:
        norm = _normalize(transcript)
        if not norm:
            return False
        return norm in _EXIT_EXACT or any(p in norm for p in self._exit_phrases)

    async def run(self) -> None:
        await self.set_state(State.SLEEPING)
        logger.info('Jarvis ready — say "Hey Jarvis".')
        # When a turn ends via barge-in, skip the wake word and record the user's
        # follow-up immediately (they already said "Hey Jarvis" to interrupt).
        barged = False
        try:
            while True:
                barged = await self._turn(skip_wake=barged)
        except _ExitRequested:
            logger.info("Exit phrase heard — goodbye.")

    async def _turn(self, skip_wake: bool = False) -> bool:
        """Run one turn. Returns True if the user barged in (skip wake next)."""
        if not skip_wake:
            await self._bus.call(SVC_WAIT_WAKE)
        await self.set_state(State.LISTENING)

        # Snapshot the workspace the instant the user starts talking and run
        # Qwen-VL in the background, so vision overlaps listening/transcription
        # instead of adding its ~2-3 s to the critical path.
        vision_task = asyncio.create_task(self._describe_workspace())

        # Stream mic -> STT live and get the transcript when the user stops.
        transcript = await self._bus.call(SVC_LISTEN)
        if not transcript:
            logger.info("No speech understood; back to sleep.")
            await self._cancel(vision_task)  # don't waste the Qwen call
            await self.set_state(State.SLEEPING)
            return False

        # Exit phrase ("Jarvis exit") — show "Okay, done" and quit.
        if self._is_exit(transcript):
            await self._cancel(vision_task)
            logger.info("Exit phrase: %r", transcript)
            await self.set_state(State.DONE)
            await asyncio.sleep(2.0)  # let the OLED message be seen
            raise _ExitRequested

        await self.set_state(State.LISTENING, transcript[:28])

        # Vision was kicked off at wake; by now it's usually already done.
        await self.set_state(State.VISION)
        vision = await vision_task

        await self.set_state(State.MOSS)
        grounding = await self._bus.call(
            SVC_COMPILE, {"transcript": transcript, "vision": vision}
        )

        # THINKING resets the TTS interrupt flag (see TTSNode); SPEAKING is shown
        # as soon as the first sentence is synthesized.
        await self.set_state(State.THINKING)
        await self.set_state(State.SPEAKING)
        barged = await self._respond(grounding, transcript)

        if not barged:
            await self.set_state(State.SLEEPING)
        return barged

    async def _describe_workspace(self) -> str:
        """Run the vision service; never raise (degrade to no visual context)."""
        try:
            return await self._bus.call(SVC_DESCRIBE, {})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Vision capture failed; continuing without it")
            return ""

    async def _cancel(self, task: asyncio.Task) -> None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _respond(self, grounding: str, user_text: str) -> bool:
        """Stream the answer. Returns True if the user barged in with the wake
        word during playback (so the caller skips the wake word next turn)."""
        request = {"grounding": grounding, "user_text": user_text}
        respond = asyncio.create_task(self._bus.call(SVC_RESPOND, request))
        if not self._barge_in:
            await respond
            return False

        # Race the answer against a fresh wake word; if the user barges in,
        # stop playback + generation immediately and signal a follow-up turn.
        barge = asyncio.create_task(self._bus.call(SVC_WAIT_WAKE))
        done, _pending = await asyncio.wait(
            {respond, barge}, return_when=asyncio.FIRST_COMPLETED
        )
        if barge in done and respond not in done:
            logger.info("Barge-in: stopping current answer.")
            await self._bus.publish(TOPIC_INTERRUPT)
            await self._cancel(respond)  # abort generation so the next turn is snappy
            return True

        barge.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await barge
        await respond
        return False
