"""Shared LLM helper — shells out to the OpenAI `codex` CLI.

We use the `codex exec --json` non-interactive entry point so the agent
inherits the user's existing ChatGPT login (no API key required). This is the
mechanism used by all 4 subagents, the synthesis call, the ingestion
extractor, and the photo-corpus trait scorer.

The CLI is invoked as:

    codex exec --skip-git-repo-check --json -m <model> [--image FILE]... -

with the prompt streamed in on stdin. We parse the JSONL events and return
the text of the final `item.completed` agent_message.

Configure via env:
    DD_MODEL        default: gpt-5.5
    DD_MODEL_FAST   default: gpt-5.5-mini   (used by extractor + photo traits)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil

def _default_model() -> str:
    return os.environ.get("DD_MODEL", "gpt-5.5")


def _fast_model() -> str:
    return os.environ.get("DD_MODEL_FAST", "gpt-5.5-mini")


def _codex_bin() -> str:
    return os.environ.get("DD_CODEX_BIN", "codex")


# Module-level constants for callers that want a static default. These are
# resolved once at import time but every helper re-reads env so monkeypatched
# tests Just Work.
DEFAULT_MODEL = _default_model()
FAST_MODEL = _fast_model()
CODEX_BIN = _codex_bin()


class CodexUnavailableError(RuntimeError):
    pass


class CodexError(RuntimeError):
    pass


def codex_path() -> str:
    """Return absolute path to codex, or raise. Re-reads env each call."""
    binary = _codex_bin()
    path = shutil.which(binary)
    if not path:
        raise CodexUnavailableError(
            f"codex CLI not found on PATH (looked for `{binary}`). "
            "Install it: `npm install -g @openai/codex` and run `codex login`."
        )
    return path


async def codex_exec(
    prompt: str,
    *,
    model: str | None = None,
    images: list[str] | None = None,
    timeout: float = 300.0,
) -> str:
    """Run `codex exec` with the prompt and return the assistant's final text.

    The prompt is sent on stdin. Multiple images can be attached via --image.
    """
    bin_path = codex_path()
    args = [bin_path, "exec", "--skip-git-repo-check", "--json"]
    if model:
        args += ["-m", model]
    for img in images or []:
        args += ["--image", img]
    args.append("-")  # read prompt from stdin

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise CodexError(f"codex exec timed out after {timeout}s")

    if proc.returncode != 0:
        raise CodexError(
            f"codex exec exited {proc.returncode}: "
            f"{stderr.decode('utf-8', errors='replace')[:500]}"
        )
    return _last_agent_message(stdout.decode("utf-8", errors="replace"))


def _last_agent_message(jsonl: str) -> str:
    """Parse codex --json output and return the text of the final agent message."""
    last_text = ""
    for line in jsonl.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "item.completed":
            item = evt.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                last_text = item["text"]
    return last_text


async def render_section(
    *,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 4000,  # noqa: ARG001 — kept for API compat; codex chooses
) -> str:
    """Subagent entry point. System + user are joined with a clear delimiter
    since `codex exec` is single-prompt (no role separation on the CLI)."""
    combined = (
        "<system_instructions>\n"
        f"{system}\n"
        "</system_instructions>\n\n"
        "<task>\n"
        f"{user}\n"
        "</task>"
    )
    return await codex_exec(combined, model=model or DEFAULT_MODEL)


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
