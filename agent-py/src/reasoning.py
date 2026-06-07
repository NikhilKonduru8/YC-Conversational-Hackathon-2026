"""MiniMax reasoning engine, called directly via its OpenAI-compatible API.

We stream MiniMax with the plain `openai` async client pointed at
https://api.minimax.io/v1 using a MiniMax API key. Tokens are yielded as they
arrive so TTS can start speaking the first sentence immediately. Connection
failures yield a short spoken fallback instead of crashing. The agent stays on
topic via the grounded system prompt below (no external guardrail service).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from config import ReasoningConfig

logger = logging.getLogger("jarvis.reasoning")

JARVIS_SYSTEM_PROMPT = """\
You are Jarvis, an expert electronics engineering assistant built into smart
glasses. The user is at a workbench with real hardware and talks to you
hands-free. Each turn you may also get a VISUAL description of what their camera
sees and RETRIEVED component specs — use them when relevant.

How to answer:
- Actually help, like a sharp lab partner. Use the visual + retrieved context
  when useful; otherwise rely on your own electronics expertise. Give correct,
  practical guidance — wiring, components, pinouts, voltages, resistor values.
- Think silently and reply with ONLY your final answer. Never show reasoning,
  steps headers, or restate the question.
- Be brief: one or two short sentences for a fact; for "how do I..." give the
  key steps in two or three short sentences. Lead with the answer.
- Use exact figures from the specs when present. If you genuinely don't know a
  specific value, say so in one sentence and suggest what to check — never stall,
  repeat yourself, or say "I couldn't find that" more than once.
- Be proactive: if the request is ambiguous or you're missing a key detail you'd
  need to answer well (which component, what supply voltage, what you're building),
  ask ONE short clarifying question instead of guessing. Otherwise just answer.
- Stay on electronics/hardware; if asked something unrelated, give a one-sentence
  redirect. Decline unsafe requests briefly.

Output is spoken aloud: plain text only — no markdown, lists, code, tags, or
emojis. Say numbers and units naturally ("five volts", "pin three"). Do not
mention the context, tools, or these instructions.
"""

_FALLBACK = (
    "Sorry, I had trouble reaching my reasoning engine just now. "
    "Could you ask me that again?"
)


class _ThinkFilter:
    """Strip <think>...</think> spans from a streamed token feed.

    Safety net for reasoning models that leak thinking into `content` (M2.x).
    Holds back a few trailing chars so a tag split across chunks isn't missed.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    # Emit all but a tail that might be a partial "<think>".
                    cut = max(0, len(self._buf) - (len(self._OPEN) - 1))
                    out.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                    break
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + len(self._OPEN) :]
                self._in_think = True
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    # Drop thinking text, keep a tail for a partial "</think>".
                    self._buf = self._buf[-(len(self._CLOSE) - 1) :]
                    break
                self._buf = self._buf[idx + len(self._CLOSE) :]
                self._in_think = False
        return "".join(out)

    def flush(self) -> str:
        if self._in_think:
            self._buf = ""
            return ""
        tail = self._buf
        self._buf = ""
        return tail


class Reasoner:
    def __init__(self, config: ReasoningConfig) -> None:
        self._cfg = config
        self._client = None
        if config.base_url and config.api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=config.timeout_s,
            )
            logger.info("MiniMax ready: %s (%s)", config.model, config.base_url)
        else:
            logger.warning(
                "MiniMax not configured (MINIMAX_API_KEY); reasoning unavailable."
            )

    @property
    def ready(self) -> bool:
        return self._client is not None

    async def stream(
        self, grounding_prompt: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Yield response text chunks for the grounded context + prior turns."""
        if not self.ready:
            yield _FALLBACK
            return

        messages = [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": grounding_prompt})
        # Disable model thinking (M3) for low latency + clean output; ignored by
        # models that don't support it.
        extra_body = (
            {"thinking": {"type": "disabled"}} if self._cfg.disable_thinking else None
        )

        think = _ThinkFilter()
        try:
            response = await self._client.chat.completions.create(
                model=self._cfg.model,
                messages=messages,
                temperature=self._cfg.temperature,
                max_tokens=self._cfg.max_tokens,
                stream=True,
                extra_body=extra_body,
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                text = getattr(chunk.choices[0].delta, "content", None)
                if text:
                    clean = think.feed(text)
                    if clean:
                        yield clean
            tail = think.flush()
            if tail:
                yield tail
        except Exception:
            logger.exception("MiniMax stream failed")
            yield _FALLBACK
