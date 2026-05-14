"""Shared OpenAI (GPT-5.5) helper used by all four subagents and the synthesis step.

We use the raw `openai` async client rather than any wrapping agent SDK here
because each subagent is a single-shot rendering call — they don't need tool
use, message threading, or session state. The orchestrator fans out N of these
in `asyncio.gather` for the 4× speedup.
"""

from __future__ import annotations

import os
import re

DEFAULT_MODEL = os.environ.get("DD_MODEL", "gpt-5.5")
FAST_MODEL = os.environ.get("DD_MODEL_FAST", "gpt-5.5-mini")


async def render_section(
    *,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 4000,
) -> str:
    """One-shot OpenAI chat completion. Returns the text. Raises on any failure."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def load_prompt(path: str) -> str:
    """Load a prompt file. Path is relative to the dd_agent package root."""
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    return (here / path).read_text(encoding="utf-8")


_CITATION_RE = re.compile(r"\[(\d+)\]")


def rewrite_citations(section_text: str, mapping: dict[int, int]) -> str:
    """Rewrite [n] markers in a section from local subagent numbering to global
    orchestrator numbering. `mapping[local] = global`."""
    def repl(m: re.Match) -> str:
        local = int(m.group(1))
        target = mapping.get(local)
        return f"[{target}]" if target else m.group(0)
    return _CITATION_RE.sub(repl, section_text)
