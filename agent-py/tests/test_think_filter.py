"""Tests for stripping <think> reasoning out of streamed MiniMax output."""

from reasoning import _ThinkFilter


def _run(filter_obj, chunks):
    out = "".join(filter_obj.feed(c) for c in chunks)
    return out + filter_obj.flush()


def test_no_think_passthrough():
    assert _run(_ThinkFilter(), ["Five ", "volts."]) == "Five volts."


def test_strips_think_block():
    f = _ThinkFilter()
    out = _run(f, ["<think>let me reason</think>The answer is 5V."])
    assert out == "The answer is 5V."


def test_strips_think_split_across_chunks():
    f = _ThinkFilter()
    out = _run(
        f, ["<thi", "nk>reasoning here", " more</thi", "nk>Use a 220 ohm resistor."]
    )
    assert out == "Use a 220 ohm resistor."


def test_leading_text_then_think():
    f = _ThinkFilter()
    out = _run(f, ["Answer: <think>x</think> done"])
    assert "Answer:" in out and "done" in out and "x" not in out


def test_unclosed_think_is_dropped():
    f = _ThinkFilter()
    out = _run(f, ["<think>still thinking and never closed"])
    assert out == ""
