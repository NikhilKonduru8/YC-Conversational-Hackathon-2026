"""Unit tests for the unified context compiler (Prompt 3)."""

import pytest

from moss_context import (
    CompiledContext,
    ContextCompiler,
    condense_query,
    extract_components,
)


def test_extract_components_finds_part_numbers():
    text = (
        "An 8-pin DIP chip labeled LM358 sits next to an NE555 timer and an "
        "ATMEGA328P-PU microcontroller. Two red LEDs are lit."
    )
    parts = extract_components(text)
    assert "LM358" in parts
    assert "NE555" in parts
    assert "ATMEGA328P-PU" in parts


def test_extract_components_ignores_plain_words_and_numbers():
    parts = extract_components("There are 3 resistors and a capacitor on the board.")
    assert parts == []


def test_condense_query_combines_vision_and_speech():
    query = condense_query(
        user_request="what is the max voltage?",
        vision_summary="A chip labeled LM358 is visible on the breadboard.",
    )
    assert "LM358" in query
    # Intent expansion maps "max voltage" -> abs-max terminology.
    assert "absolute maximum ratings" in query
    assert "max voltage" in query


def test_condense_query_falls_back_to_request():
    assert condense_query("tell me about this", "") == "tell me about this"


def test_condense_query_handles_empty_inputs():
    # Never returns empty (Moss needs a non-empty query).
    assert condense_query("", "").strip() != ""


def test_grounding_prompt_includes_all_three_sources():
    ctx = CompiledContext(
        user_request="what is the max voltage",
        vision_summary="LM358 op-amp on a breadboard.",
        moss_query="LM358 absolute maximum ratings",
        matches=[{"text": "Absolute maximum supply voltage: 32 V.", "score": 0.91}],
    )
    prompt = ctx.grounding_prompt()
    assert "what is the max voltage" in prompt
    assert "LM358 op-amp on a breadboard." in prompt
    assert "32 V" in prompt
    assert "relevance 0.91" in prompt


def test_grounding_prompt_handles_no_matches_and_no_vision():
    ctx = CompiledContext(
        user_request="what is this",
        vision_summary="",
        moss_query="what is this",
        matches=[],
    )
    prompt = ctx.grounding_prompt()
    assert "No visual context available." in prompt
    assert "No matching specifications" in prompt


# --- ContextCompiler with a fake Moss client -------------------------------- #
class _FakeDoc:
    def __init__(self, text, score=None, metadata=None):
        self.text = text
        self.score = score
        self.metadata = metadata


class _FakeResult:
    def __init__(self, docs, time_taken_ms=9.0):
        self.docs = docs
        self.time_taken_ms = time_taken_ms


class _FakeMoss:
    def __init__(self, result):
        self._result = result
        self.queries = []

    async def query(self, index, query, options=None):
        self.queries.append((index, query, options))
        return self._result


@pytest.mark.asyncio
async def test_compiler_queries_moss_and_builds_context():
    moss = _FakeMoss(
        _FakeResult(
            [
                _FakeDoc(
                    "Supply voltage max 32 V.",
                    score=0.88,
                    metadata={"part_number": "LM358"},
                )
            ]
        )
    )
    compiler = ContextCompiler(moss, "knowledge", top_k=4)
    compiled = await compiler.compile(
        "what's the max voltage", "A chip labeled LM358 is on the bench."
    )

    assert len(moss.queries) == 1
    index, query, options = moss.queries[0]
    assert index == "knowledge"
    assert "LM358" in query
    assert options.top_k == 4

    assert compiled.matches[0]["text"] == "Supply voltage max 32 V."
    assert compiled.matches[0]["score"] == 0.88
    assert "32 V" in compiled.grounding_prompt()


@pytest.mark.asyncio
async def test_compiler_survives_moss_failure():
    class _BrokenMoss:
        async def query(self, *a, **k):
            raise RuntimeError("moss down")

    compiler = ContextCompiler(_BrokenMoss(), "knowledge")
    compiled = await compiler.compile("max voltage", "LM358 visible")
    # Degrades gracefully: no matches, but still a usable grounding prompt.
    assert compiled.matches == []
    assert "No matching specifications" in compiled.grounding_prompt()
