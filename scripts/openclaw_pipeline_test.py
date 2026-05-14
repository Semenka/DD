"""Drive dd-agent end-to-end through MCP exactly the way OpenClaw will.

OpenClaw, when it needs a tool, reads its registered `mcp.servers.dd-agent`
config, spawns the subprocess (`/Users/andrey/DD/.venv/bin/dd-agent serve`),
and speaks the MCP JSON-RPC stdio protocol against it. This script
reproduces that exact lifecycle so we can verify the pipeline works
without depending on OpenClaw having model credentials configured.

Flow:
  1. Read OpenClaw's stored config for the dd-agent MCP server
  2. Spawn that command with that env (just like OpenClaw does)
  3. MCP initialize + tools/list
  4. call submit_deal with the sample memo + company URL
  5. poll get_report_status every 5s
  6. call get_report and write markdown + html to disk
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"
MCP_NAME = "dd-agent"


def load_mcp_config(name: str) -> dict:
    data = json.loads(OPENCLAW_CONFIG.read_text())
    servers = data.get("mcp", {}).get("servers", {})
    cfg = servers.get(name)
    if not cfg:
        raise SystemExit(f"MCP server '{name}' not found in {OPENCLAW_CONFIG}")
    return cfg


class MCPClient:
    """Tiny MCP JSON-RPC client over stdio. Drop-in equivalent of what
    OpenClaw's gateway does internally."""

    def __init__(self, command: str, args: list[str], env: dict[str, str]):
        merged_env = {**os.environ, **env}
        self.proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=merged_env,
        )
        self._next_id = 1

    def _send(self, obj: dict) -> None:
        line = json.dumps(obj)
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _recv(self, timeout: float = 60.0) -> dict | None:
        # Simple blocking readline; sufficient because the server replies
        # within milliseconds for status calls and seconds for tool calls.
        end = time.time() + timeout
        while time.time() < end:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    return None
                continue
            return json.loads(line)
        return None

    def request(self, method: str, params: dict | None = None, timeout: float = 60.0) -> dict:
        rid = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        while True:
            msg = self._recv(timeout=timeout)
            if msg is None:
                raise RuntimeError(f"server died waiting for {method}")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"{method} error: {msg['error']}")
                return msg.get("result", {})

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def call_tool(self, name: str, arguments: dict, timeout: float = 600.0) -> dict:
        result = self.request("tools/call",
                              {"name": name, "arguments": arguments},
                              timeout=timeout)
        # FastMCP wraps tool results in `content`/`structuredContent`
        if "structuredContent" in result:
            return result["structuredContent"]
        if "content" in result:
            for block in result["content"]:
                if block.get("type") == "text":
                    try:
                        return json.loads(block["text"])
                    except json.JSONDecodeError:
                        return {"_raw": block["text"]}
        return result

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default="examples/sample_deal/memo.md")
    ap.add_argument("--deck", default=None)
    ap.add_argument("--url", default="https://linear.app")
    ap.add_argument("--founders", nargs="*", default=None)
    ap.add_argument("--out-md", default="report.md")
    ap.add_argument("--out-html", default="report.html")
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--max-wait", type=float, default=600.0)
    args = ap.parse_args()

    cfg = load_mcp_config(MCP_NAME)
    print(f"[openclaw] dd-agent MCP server registered at: {cfg['command']}")
    client = MCPClient(cfg["command"], cfg.get("args", []), cfg.get("env", {}))

    try:
        # 1. Initialize handshake
        init = client.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "openclaw-pipeline-test", "version": "0.1"},
        }, timeout=10)
        print(f"[mcp] initialize ok — server: {init.get('serverInfo', {}).get('name')}")
        client.notify("notifications/initialized")

        # 2. tools/list
        tools = client.request("tools/list", {}, timeout=10).get("tools", [])
        print(f"[mcp] tools: {[t['name'] for t in tools]}")

        # 3. submit_deal — exactly what OpenClaw would call
        memo_path = Path(args.memo)
        memo_text = memo_path.read_text() if memo_path.exists() else None
        deck_path_arg = str(Path(args.deck).resolve()) if args.deck else None
        submit_args = {
            "memo_text": memo_text,
            "deck_path": deck_path_arg,
            "company_url": args.url,
            "founder_names": args.founders,
        }
        print(f"[mcp] submit_deal: memo={memo_path.name}, deck={deck_path_arg or 'none'}, url={args.url}")
        result = client.call_tool("submit_deal", submit_args, timeout=60)
        deal_id = result.get("deal_id")
        if not deal_id:
            print(f"[mcp] FAIL: submit_deal returned no deal_id: {result}")
            return 2
        print(f"[mcp] submit_deal returned deal_id={deal_id}, status={result.get('status')}")

        # 4. Poll get_report_status
        start = time.time()
        last_phase = ""
        while time.time() - start < args.max_wait:
            status = client.call_tool("get_report_status", {"deal_id": deal_id}, timeout=15)
            phase = status.get("phase", "?")
            if phase != last_phase:
                print(f"[mcp] status: {status.get('status')} | {phase} | {status.get('progress_pct')}%")
                last_phase = phase
            if status.get("status") in ("done", "failed"):
                break
            time.sleep(args.poll_interval)
        else:
            print(f"[mcp] FAIL: timed out after {args.max_wait}s")
            return 3

        if status.get("status") != "done":
            print(f"[mcp] FAIL: pipeline failed: {status.get('error')}")
            return 4

        # 5. Retrieve final report
        report = client.call_tool("get_report", {"deal_id": deal_id}, timeout=15)
        md = report.get("markdown") or ""
        html = report.get("html") or ""
        citations = report.get("citations", [])

        Path(args.out_md).write_text(md, encoding="utf-8")
        Path(args.out_html).write_text(html, encoding="utf-8")
        elapsed = time.time() - start
        print(f"[mcp] done in {elapsed:.1f}s — wrote {args.out_md} ({len(md)} bytes), "
              f"{args.out_html} ({len(html)} bytes), {len(citations)} citations")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
