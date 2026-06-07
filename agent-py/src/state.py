"""Conversation states + the labels shown on the OLED."""

from __future__ import annotations

from enum import StrEnum


class State(StrEnum):
    SLEEPING = "sleeping"  # waiting for the wake word
    LISTENING = "listening"  # recording the user's request
    VISION = "vision"  # Qwen-2.5-VL analyzing the camera
    MOSS = "moss"  # querying the Moss component index
    THINKING = "thinking"  # MiniMax reasoning
    SPEAKING = "speaking"  # TTS playing the answer
    DONE = "done"  # exit phrase heard; shutting down


# Two lines for the OLED: a big title + a small detail line.
OLED_TITLE: dict[State, str] = {
    State.SLEEPING: "Jarvis",
    State.LISTENING: "Listening",
    State.VISION: "Looking",
    State.MOSS: "Searching",
    State.THINKING: "Thinking",
    State.SPEAKING: "Speaking",
    State.DONE: "Okay, done",
}

OLED_DETAIL: dict[State, str] = {
    State.SLEEPING: 'Say "Hey Jarvis"',
    State.LISTENING: "go ahead...",
    State.VISION: "analyzing view",
    State.MOSS: "component specs",
    State.THINKING: "reasoning",
    State.SPEAKING: "",
    State.DONE: "goodbye",
}


def oled_title(state: State) -> str:
    return OLED_TITLE.get(state, state.value.title())


def oled_detail(state: State) -> str:
    return OLED_DETAIL.get(state, "")
