"""Submit N deals in parallel via the dd-agent MCP server and verify each
auto-delivery fires. This is the regression test for the bug where parallel
submits previously produced only one delivery.

Usage:
    python scripts/parallel_submit_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

DEALS = [
    {
        "label": "Revoy",
        "memo_path": str(ROOT / "examples/sample_deal/revoy_memo.md"),
        "company_url": "https://revoy.com",
    },
    {
        "label": "Sellara",
        "memo_path": str(ROOT / "examples/sample_deal/sellara_memo.md"),
        "company_url": None,
    },
]

TELEGRAM = {
    "channel": "telegram",
    "account": "cosmo",
    "target": "148594943",
    "format": "html",
}


class MCP:
    def __init__(self, log_file: str):
        self.proc = subprocess.Popen(
            [str(ROOT / ".venv/bin/dd-agent"), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(log_file, "w"),
            text=True, bufsize=1, env={**os.environ},
        )
        self._id = 0

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"dd-agent died (exit {self.proc.poll()})")
        return json.loads(line)

    def request(self, method: str, params: dict) -> dict:
        self._id += 1
        rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            m = self._recv()
            if m and m.get("id") == rid:
                return m

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def tool(self, name: str, args: dict) -> dict:
        r = self.request("tools/call", {"name": name, "arguments": args})
        return r["result"]["structuredContent"]

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()
        self.proc.wait(timeout=5)


def main() -> int:
    for db in ["data/deals.db", "data/deals.db-shm", "data/deals.db-wal"]:
        Path(db).unlink(missing_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)

    log_file = "/tmp/dd-parallel.log"
    mcp = MCP(log_file)
    try:
        mcp.request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {}, "clientInfo": {"name": "parallel-test", "version": "0"},
        })
        mcp.notify("notifications/initialized")

        # Submit both deals back-to-back
        deal_ids = []
        for d in DEALS:
            result = mcp.tool("submit_deal", {
                "memo_path": d["memo_path"],
                "company_url": d["company_url"],
                "deliver_to": {
                    **TELEGRAM,
                    "summary_line": (
                        f"{d['label']} DD (parallel-submit test) — "
                        f"verifying both deals deliver independently."
                    ),
                },
            })
            deal_ids.append((d["label"], result["deal_id"]))
            print(f"submitted {d['label']}: deal_id={result['deal_id']}, "
                  f"auto_delivery={result.get('auto_delivery')}", flush=True)

        print(f"\n{len(deal_ids)} deals queued — pipelines run in parallel inside "
              f"dd-agent. Polling status…", flush=True)

        # Track each deal's last reported phase
        states: dict[str, dict] = {did: {} for _, did in deal_ids}
        start = time.time()
        deadline = start + 720  # 12 min
        while time.time() < deadline:
            all_done = True
            for label, did in deal_ids:
                s = mcp.tool("get_report_status", {"deal_id": did})
                prev = states[did].get("phase")
                if s.get("phase") != prev:
                    print(f"  [{int(time.time()-start)}s] {label} ({did}): "
                          f"{s.get('status'):8} | {s.get('phase'):35} | {s.get('progress_pct')}%",
                          flush=True)
                states[did] = s
                if s.get("status") not in ("done", "failed"):
                    all_done = False
            if all_done:
                break
            time.sleep(8)

        print(f"\n{'='*60}\nFinal status:", flush=True)
        success = 0
        for label, did in deal_ids:
            s = states[did]
            ok = s.get("status") == "done"
            print(f"  {'✓' if ok else '✗'} {label} ({did}): "
                  f"{s.get('status')} — {s.get('error') or 'no error'}")
            if ok:
                success += 1

        # Give dd-agent's delivery subprocesses time to complete
        print("\nWaiting 30s for delivery subprocesses to finish…", flush=True)
        time.sleep(30)

        # Inspect stderr log for delivery results
        log = Path(log_file).read_text()
        delivered = log.count("delivered")
        attempted = log.count("delivering deal")
        print(f"\nDelivery log: {attempted} delivery attempts, "
              f"{delivered} success/failure callbacks logged", flush=True)
        for line in log.splitlines():
            if "delivering" in line or "delivered" in line or "delivery failed" in line:
                print(f"  {line[-120:]}")

        print(f"\nVerdict: {success}/{len(deal_ids)} deals completed, "
              f"{attempted}/{len(deal_ids)} delivery calls fired")
        return 0 if success == len(deal_ids) else 4
    finally:
        mcp.close()


if __name__ == "__main__":
    sys.exit(main())
