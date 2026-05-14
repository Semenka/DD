"""Refresh NOTICE_CO_COOKIE in .env from the user's logged-in browser.

Strategy in order:
  1. **browser_cookie3 (Safari/Chrome/Brave/Firefox/Arc)** — automatic if the
     OS sandbox lets us read the cookie DB. On macOS, Safari's binarycookies
     file lives inside Safari's container and is blocked by sandbox; you'd
     need Full Disk Access for Terminal/Python. Chrome on macOS uses
     OS-keychain encryption — usually works.
  2. **Manual paste** — if no browser is readable, print a 1-line Web
     Inspector snippet for the user to paste into Safari's dev console on
     notice.co. The script reads the resulting cookie string from stdin
     and writes it to .env.

Usage:
    python scripts/refresh_notice_cookie.py
    python scripts/refresh_notice_cookie.py --browser chrome
    python scripts/refresh_notice_cookie.py --paste     # force manual mode
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOMAIN = "notice.co"
ENV_KEY = "NOTICE_CO_COOKIE"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

MANUAL_INSTRUCTIONS = f"""
Couldn't read your browser's cookies automatically (macOS Safari sandbox blocks
this without Full Disk Access). Here's a 30-second manual path that always works:

1. In Safari, open https://{DOMAIN} and confirm you're logged in.
2. Develop menu → Show Web Inspector (or Cmd+Opt+I).
3. Console tab. Paste this exactly and hit Return:

       copy(document.cookie); console.log("copied " + document.cookie.length + " chars to clipboard");

4. Switch back to this terminal. Paste the clipboard contents below, then Enter.
   (If you don't see the Develop menu, enable it in Safari → Settings →
   Advanced → "Show features for web developers".)

Paste cookie string > """


def try_browser_cookie(browser: str | None) -> str | None:
    """Try each supported browser. Returns "k=v; k=v" cookie header, or None."""
    try:
        import browser_cookie3 as bc
    except ImportError:
        print("browser_cookie3 not installed. `pip install browser_cookie3`", file=sys.stderr)
        return None

    funcs = []
    if browser:
        if hasattr(bc, browser):
            funcs.append((browser, getattr(bc, browser)))
    else:
        for name in ("chrome", "brave", "edge", "arc", "vivaldi", "safari", "firefox"):
            if hasattr(bc, name):
                funcs.append((name, getattr(bc, name)))

    for name, fn in funcs:
        try:
            cj = fn(domain_name=DOMAIN)
        except Exception as exc:
            print(f"  - {name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            continue
        cookies = list(cj)
        if cookies:
            cookie_str = "; ".join(f"{c.name}={c.value}" for c in cookies)
            print(f"  ✓ {name}: extracted {len(cookies)} cookies "
                  f"({len(cookie_str)} chars)", file=sys.stderr)
            return cookie_str
        else:
            print(f"  - {name}: no cookies found for {DOMAIN}", file=sys.stderr)
    return None


def manual_paste() -> str | None:
    print(MANUAL_INSTRUCTIONS, end="", flush=True)
    try:
        line = input().strip()
    except EOFError:
        return None
    if not line:
        return None
    # Strip surrounding quotes if user pasted with them.
    if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
        line = line[1:-1]
    return line


def write_env(cookie: str) -> None:
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text().splitlines()
    found = False
    for i, l in enumerate(lines):
        if l.startswith(f"{ENV_KEY}="):
            lines[i] = f'{ENV_KEY}="{cookie}"'
            found = True
            break
    if not found:
        lines.append(f'{ENV_KEY}="{cookie}"')
    ENV_PATH.write_text("\n".join(lines) + "\n")
    # Brief sanity: parse # cookies
    n_pairs = len(re.findall(r"[^=;]+=", cookie))
    print(f"✓ wrote {ENV_KEY} to {ENV_PATH} ({n_pairs} cookie pairs, "
          f"{len(cookie)} chars total)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", choices=[
        "chrome", "brave", "edge", "arc", "vivaldi", "safari", "firefox",
    ], help="Try a specific browser only.")
    ap.add_argument("--paste", action="store_true",
                    help="Skip automatic extraction and prompt for manual paste.")
    args = ap.parse_args()

    cookie: str | None = None
    if not args.paste:
        print(f"Attempting browser cookie extraction for {DOMAIN}…", file=sys.stderr)
        cookie = try_browser_cookie(args.browser)

    if not cookie:
        cookie = manual_paste()

    if not cookie:
        print("\nNo cookie obtained. Aborting.", file=sys.stderr)
        return 2

    if "=" not in cookie:
        print(f"Refusing to write a cookie string with no '=' (got: {cookie[:60]!r})",
              file=sys.stderr)
        return 3

    write_env(cookie)
    print("\nThe dd-agent notice.co adapter will pick this up on its next run "
          "(it reads NOTICE_CO_COOKIE from the environment via python-dotenv).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
