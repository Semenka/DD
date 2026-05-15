"""Auto-delivery: shell out to `openclaw message send` after a pipeline finishes.

The MCP `submit_deal` accepts an optional `deliver_to` dict. If set, the
orchestrator calls `deliver()` after `save_report`. dd-agent thus fires the
Telegram delivery itself — OpenClaw's main agent only needs to call
`submit_deal` once per deal and end its turn. No polling required.

Why this fixes the parallel-submit bug: previously OpenClaw's main agent
submitted deal A and deal B, returned both deal_ids in two turns, then
nothing kept it alive to poll for completion. Only the deal whose status
the main agent happened to ask about later got delivered. With dd-agent
firing delivery itself, every completed deal lands.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("dd_agent.delivery")


def _openclaw_media_dir() -> Path:
    """Return the directory OpenClaw allows for media attachments. Defaults
    to `~/.openclaw/workspace/dd-reports/` because OpenClaw's LocalMediaAccessError
    blocks paths outside its workspace. Override via DD_OPENCLAW_MEDIA_DIR."""
    override = os.environ.get("DD_OPENCLAW_MEDIA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openclaw" / "workspace" / "dd-reports"


@dataclass(frozen=True)
class DeliverTo:
    """Where to send the report when the pipeline finishes."""
    channel: str               # "telegram" | "whatsapp" | "imessage" | etc.
    account: str               # OpenClaw account id, e.g. "cosmo" or "default"
    target: str                # chat id / phone number / handle
    format: str = "html"       # "html" | "pdf" | "markdown" (which file to attach)
    summary_line: str | None = None  # message body; if None, extracted from report

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DeliverTo | None":
        if not d:
            return None
        if not (d.get("channel") and d.get("target")):
            return None
        return cls(
            channel=str(d["channel"]),
            account=str(d.get("account", "default")),
            target=str(d["target"]),
            format=str(d.get("format", "html")),
            summary_line=d.get("summary_line"),
        )


def _openclaw_bin() -> str | None:
    return shutil.which("openclaw")


async def deliver(
    *,
    deliver_to: DeliverTo,
    deal_id: str,
    company: str | None,
    markdown_path: str | None,
    html_path: str | None,
    pdf_path: str | None,
    one_line_bet: str | None = None,
) -> dict[str, Any]:
    """Fire `openclaw message send` with the chosen format as an attachment.

    Returns a dict with `ok` and either `message_id` or `error`. Never raises;
    the orchestrator catches delivery failures separately so a delivery error
    doesn't break the pipeline.
    """
    binary = _openclaw_bin()
    if not binary:
        return {"ok": False, "error": "openclaw CLI not on PATH; install or set $PATH"}

    file_path = _pick_attachment(deliver_to.format, html_path, pdf_path, markdown_path)
    if not file_path:
        return {"ok": False, "error": f"no {deliver_to.format} report available to send"}

    # OpenClaw rejects media paths outside its workspace ("LocalMediaAccessError").
    # Copy the file into an allowed directory + use a human-friendly filename.
    try:
        attach_path = _stage_for_openclaw(file_path, deal_id, company, deliver_to.format)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"could not stage attachment: {exc}"}

    summary = deliver_to.summary_line or one_line_bet or (
        f"DD report for {company or 'this deal'} (deal_id {deal_id})"
    )
    summary = summary.strip()[:1024]

    args = [
        binary, "message", "send",
        "--channel", deliver_to.channel,
        "--account", deliver_to.account,
        "--target", deliver_to.target,
        "--message", summary,
        "--media", str(attach_path),
        "--json",
    ]
    if deliver_to.channel == "telegram":
        args.append("--force-document")

    log.info("delivering deal %s via %s/%s to %s (format=%s, file=%s)",
             deal_id, deliver_to.channel, deliver_to.account,
             deliver_to.target, deliver_to.format, file_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
    except asyncio.TimeoutError:
        return {"ok": False, "error": "openclaw message send timed out after 120s"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"openclaw message send raised: {exc}"}

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"openclaw exit {proc.returncode}: "
                     f"{stderr.decode('utf-8', errors='replace')[:500]}",
        }

    # The --json output is a single JSON object on stdout.
    import json
    try:
        data = json.loads(stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {"ok": True, "raw": stdout.decode("utf-8", errors="replace")[:500]}
    return {"ok": data.get("ok", True), "result": data}


def _stage_for_openclaw(
    source_path: str, deal_id: str, company: str | None, fmt: str,
) -> Path:
    """Copy the report into a directory OpenClaw can access, with a friendly
    filename like `Revoy-DD-deal_id.html`. Returns the staged path."""
    dest_dir = _openclaw_media_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_company = "".join(c if c.isalnum() else "_" for c in (company or "deal")).strip("_") or "deal"
    ext_map = {"html": "html", "pdf": "pdf", "markdown": "md"}
    ext = ext_map.get(fmt.lower(), Path(source_path).suffix.lstrip(".") or "bin")
    dest = dest_dir / f"{safe_company}-DD-{deal_id}.{ext}"
    shutil.copyfile(source_path, dest)
    return dest


def _pick_attachment(
    fmt: str, html_path: str | None, pdf_path: str | None, markdown_path: str | None
) -> str | None:
    """Choose the attachment to send. Falls through to whatever IS available."""
    fmt = (fmt or "html").lower()
    if fmt == "html" and html_path:
        return html_path
    if fmt == "pdf" and pdf_path:
        return pdf_path
    if fmt == "markdown" and markdown_path:
        return markdown_path
    # Sensible fall-throughs
    return html_path or pdf_path or markdown_path


_ONE_LINE_BET_RE = None


def extract_one_line_bet(markdown: str) -> str | None:
    """Pull the first `### 1-line bet` text from the report markdown so we can
    use it as the Telegram message body. Looks across all section variants."""
    import re
    global _ONE_LINE_BET_RE
    if _ONE_LINE_BET_RE is None:
        _ONE_LINE_BET_RE = re.compile(
            r"###\s*1[\s\-]?line\s+bet\s*\n+(.+?)(?=\n##|\n###|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
    m = _ONE_LINE_BET_RE.search(markdown or "")
    if not m:
        return None
    line = m.group(1).strip()
    # Strip leading markdown emphasis and trailing punctuation noise.
    line = line.strip("*_ \t\n")
    # First sentence only (avoid grabbing the whole subsection).
    line = line.split("\n", 1)[0].strip()
    return line or None
