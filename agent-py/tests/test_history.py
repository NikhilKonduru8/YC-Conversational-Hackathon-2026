"""The reasoning node should carry conversation history across turns."""

from bus import Bus
from config import ReasoningConfig
from nodes.reasoning_node import ReasoningNode


class _FakeReasoner:
    def __init__(self):
        self.seen_histories = []

    async def stream(self, grounding, history=None):
        self.seen_histories.append(list(history or []))
        for token in ["Use a ", "resistor."]:
            yield token


async def test_history_accumulates_and_is_passed_forward():
    node = ReasoningNode(Bus(), ReasoningConfig())
    node._reasoner = _FakeReasoner()

    await node._respond({"grounding": "g1", "user_text": "blink the yellow LED"})
    await node._respond({"grounding": "g2", "user_text": "every three seconds"})

    # First turn saw empty history; second turn saw the first turn's exchange.
    assert node._reasoner.seen_histories[0] == []
    assert node._reasoner.seen_histories[1] == [
        {"role": "user", "content": "blink the yellow LED"},
        {"role": "assistant", "content": "Use a resistor."},
    ]


async def test_history_is_trimmed():
    node = ReasoningNode(Bus(), ReasoningConfig())
    node._reasoner = _FakeReasoner()
    for i in range(10):
        await node._respond({"grounding": "g", "user_text": f"q{i}"})
    # 2 messages per turn, capped at 8.
    assert len(node._history) <= 8
    assert node._history[-2] == {"role": "user", "content": "q9"}
