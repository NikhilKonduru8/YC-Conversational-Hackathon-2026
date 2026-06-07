"""Unified context compiler: STT + vision -> Moss query -> grounded context.

This is the heart of Prompt 3. On every active turn it:

1. Takes the raw user transcription and the Qwen-2.5-VL visual summary.
2. Extracts component identifiers (part numbers) seen on the bench and expands
   the user's intent into spec terminology, producing one condensed query
   (e.g. vision sees "LM358" + user asks "max voltage" ->
   "LM358 absolute maximum ratings supply voltage ...").
3. Runs a low-latency similarity query against the Moss `knowledge` index.
4. Formats everything into a single grounding block ready for MiniMax.
"""

from __future__ import annotations

import contextlib
import logging
import re
from dataclasses import dataclass, field

from moss import MossClient, QueryOptions

logger = logging.getLogger("agent.context")

# Part-number-like tokens: must contain at least one letter AND one digit, be
# uppercase-ish, length >= 3. Matches LM358, NE555, 2N2222, 1N4148,
# ATMEGA328P-PU, STM32F103, ESP32. Avoids plain words and bare numbers.
_PART_RE = re.compile(
    r"\b(?=[A-Z0-9-]*[0-9])(?=[A-Z0-9-]*[A-Z])[A-Z0-9][A-Z0-9-]{2,}\b"
)

# Map casual intent phrasing -> datasheet terminology to sharpen recall.
_INTENT_SYNONYMS: dict[str, str] = {
    "max voltage": "absolute maximum ratings supply voltage",
    "maximum voltage": "absolute maximum ratings supply voltage",
    "max current": "absolute maximum ratings output current",
    "maximum current": "absolute maximum ratings output current",
    "operating voltage": "recommended operating conditions supply voltage",
    "pinout": "pin configuration pinout",
    "pin": "pin configuration pinout",
    "pins": "pin configuration pinout",
    "datasheet": "datasheet electrical specifications",
    "power": "power dissipation supply current",
    "gain": "gain bandwidth open loop gain",
    "temperature": "operating temperature range",
    "package": "package type footprint",
}


def extract_components(text: str, limit: int = 4) -> list[str]:
    """Pull likely component part numbers out of a visual summary."""
    if not text:
        return []
    seen: list[str] = []
    for match in _PART_RE.findall(text):
        token = match.strip("-")
        if len(token) < 3:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    return seen


def condense_query(user_request: str, vision_summary: str) -> str:
    """Build one condensed Moss query from speech + vision."""
    components = extract_components(vision_summary)
    request = (user_request or "").strip()

    expansions: list[str] = []
    lowered = request.lower()
    for phrase, expansion in _INTENT_SYNONYMS.items():
        if phrase in lowered:
            expansions.append(expansion)

    parts = [p for p in [" ".join(components), request, " ".join(expansions)] if p]
    query = " ".join(parts).strip()
    # Fall back to the raw request (or a generic prompt) if nothing else.
    return query or request or "component electrical specifications"


@dataclass
class CompiledContext:
    user_request: str
    vision_summary: str
    moss_query: str
    matches: list[dict] = field(default_factory=list)
    time_taken_ms: float | None = None

    @property
    def knowledge_block(self) -> str:
        if not self.matches:
            return "No matching specifications were found in the index."
        lines = []
        for i, m in enumerate(self.matches, start=1):
            text = (m.get("text") or "").strip()
            if not text:
                continue
            score = m.get("score")
            tag = f" (relevance {score:.2f})" if isinstance(score, (int, float)) else ""
            lines.append(f"[{i}]{tag} {text}")
        return "\n\n".join(lines) if lines else "No matching specifications were found."

    def grounding_prompt(self) -> str:
        """The single combined block handed to the MiniMax reasoning layer."""
        vision = self.vision_summary.strip() or "No visual context available."
        return (
            "USER REQUEST (transcribed speech):\n"
            f'"{self.user_request.strip()}"\n\n'
            "VISUAL CONTEXT (live camera, analyzed by Qwen-2.5-VL):\n"
            f"{vision}\n\n"
            "RETRIEVED TECHNICAL SPECIFICATIONS "
            f'(Moss semantic index, query: "{self.moss_query}"):\n'
            f"{self.knowledge_block}"
        )


class ContextCompiler:
    """Owns the Moss client and turns a turn's inputs into grounded context."""

    def __init__(
        self, moss: MossClient, knowledge_index: str, *, top_k: int = 4
    ) -> None:
        self._moss = moss
        self._index = knowledge_index
        self._top_k = top_k

    async def compile(self, user_request: str, vision_summary: str) -> CompiledContext:
        query = condense_query(user_request, vision_summary)
        matches: list[dict] = []
        time_taken_ms: float | None = None

        try:
            result = await self._moss.query(
                self._index, query, QueryOptions(top_k=self._top_k)
            )
            time_taken_ms = getattr(result, "time_taken_ms", None)
            for doc in getattr(result, "docs", None) or []:
                entry: dict = {"text": (getattr(doc, "text", "") or "").strip()}
                score = getattr(doc, "score", None)
                if score is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        entry["score"] = float(score)
                metadata = getattr(doc, "metadata", None)
                if metadata:
                    entry["metadata"] = metadata
                if entry["text"]:
                    matches.append(entry)
            logger.info(
                "Moss query '%s' -> %d match(es) in %sms",
                query,
                len(matches),
                time_taken_ms,
            )
        except Exception:
            logger.exception("Moss query failed; proceeding with no retrieved specs")

        return CompiledContext(
            user_request=user_request,
            vision_summary=vision_summary,
            moss_query=query,
            matches=matches,
            time_taken_ms=time_taken_ms,
        )
