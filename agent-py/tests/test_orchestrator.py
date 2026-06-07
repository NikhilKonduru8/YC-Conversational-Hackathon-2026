"""Integration tests for the orchestrator turn over a real bus with stubs."""

import asyncio
from dataclasses import replace

from bus import Bus
from config import AudioConfig
from nodes.orchestrator_node import OrchestratorNode
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

# barge-in races a wake-word against the answer; disable it for the linear
# pipeline tests so the instant stub wake doesn't look like a barge.
_NO_BARGE = replace(AudioConfig(), barge_in=False)


def _wire(bus: Bus, transcript: str, calls: dict):
    bus.register_service(SVC_WAIT_WAKE, lambda _=None: 0.9)
    bus.register_service(SVC_LISTEN, lambda _=None: transcript)

    def describe(req):
        calls["describe"] = req
        return "An LM358 op-amp is on the breadboard."

    def compile_ctx(req):
        calls["compile"] = req
        return "GROUNDING BLOCK"

    def respond(req):
        calls["respond"] = req
        return "Five volts."

    bus.register_service(SVC_DESCRIBE, describe)
    bus.register_service(SVC_COMPILE, compile_ctx)
    bus.register_service(SVC_RESPOND, respond)


async def test_turn_runs_full_pipeline_in_order():
    bus = Bus()
    calls: dict = {}
    states: list[State] = []
    bus.subscribe(TOPIC_STATE, lambda m: states.append(m["state"]))
    _wire(bus, "what is the max voltage", calls)

    orch = OrchestratorNode(bus, _NO_BARGE)
    barged = await orch._turn()

    assert barged is False
    assert states == [
        State.LISTENING,
        State.LISTENING,
        State.VISION,
        State.MOSS,
        State.THINKING,
        State.SPEAKING,
        State.SLEEPING,
    ]
    assert calls["describe"] == {}
    assert calls["compile"] == {
        "transcript": "what is the max voltage",
        "vision": "An LM358 op-amp is on the breadboard.",
    }
    # The reasoning service gets the grounding AND the raw user text (for memory).
    assert calls["respond"] == {
        "grounding": "GROUNDING BLOCK",
        "user_text": "what is the max voltage",
    }


async def test_empty_transcript_short_circuits_to_sleep():
    bus = Bus()
    calls: dict = {}
    states: list[State] = []
    bus.subscribe(TOPIC_STATE, lambda m: states.append(m["state"]))
    _wire(bus, "", calls)  # STT returns nothing

    orch = OrchestratorNode(bus, _NO_BARGE)
    barged = await orch._turn()

    assert barged is False
    assert "describe" not in calls
    assert "respond" not in calls
    assert states == [State.LISTENING, State.SLEEPING]


async def test_barge_in_interrupts_and_signals_followup():
    bus = Bus()
    interrupts: list = []
    bus.subscribe(TOPIC_INTERRUPT, lambda _m: interrupts.append(1))
    bus.register_service(SVC_WAIT_WAKE, lambda _=None: 0.9)  # wake fires immediately
    bus.register_service(SVC_LISTEN, lambda _=None: "blink the led")
    bus.register_service(SVC_DESCRIBE, lambda req: "")
    bus.register_service(SVC_COMPILE, lambda req: "GROUNDING")

    async def slow_respond(req):
        await asyncio.sleep(0.5)  # still "speaking" when the wake word fires
        return "answer"

    bus.register_service(SVC_RESPOND, slow_respond)

    orch = OrchestratorNode(bus, AudioConfig())  # barge_in on by default
    barged = await orch._turn()

    # The wake word during playback interrupted and requests a follow-up turn.
    assert barged is True
    assert interrupts == [1]


async def test_exit_phrase_quits_and_shows_done():
    bus = Bus()
    states: list[State] = []
    bus.subscribe(TOPIC_STATE, lambda m: states.append(m["state"]))
    bus.register_service(SVC_WAIT_WAKE, lambda _=None: 0.9)
    bus.register_service(SVC_LISTEN, lambda _=None: "Jarvis exit.")
    bus.register_service(SVC_DESCRIBE, lambda req: "")
    bus.register_service(SVC_COMPILE, lambda req: "G")
    bus.register_service(SVC_RESPOND, lambda req: "x")

    orch = OrchestratorNode(
        bus, replace(AudioConfig(), barge_in=False), ["jarvis exit", "exit jarvis"]
    )
    # run() should catch the internal exit and return cleanly.
    await orch.run()

    assert State.DONE in states
    # No answer was produced for the exit turn.
    assert State.SPEAKING not in states
