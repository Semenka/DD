"""GitHub adapter — founder commit history and repo signal.

Used in founder DD to show "shipped artifacts" — number of public repos,
total stars, language breakdown, recent commit cadence. No auth needed for
public read; rate-limited to 60 req/hr unauthenticated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx

_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "DD-Agent"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


@dataclass
class FounderGithub:
    handle: str
    name: str | None = None
    bio: str | None = None
    public_repos: int = 0
    followers: int = 0
    total_stars: int = 0
    top_repos: list[dict] = field(default_factory=list)
    contributions_last_year: int | None = None
    created_at: str | None = None


async def fetch_founder(handle: str) -> FounderGithub | None:
    """Pull user profile + top repos for a GitHub handle."""
    if not handle:
        return None
    handle = handle.lstrip("@")
    async with httpx.AsyncClient(timeout=15.0, headers=_headers()) as client:
        try:
            r = await client.get(f"{_API}/users/{handle}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            user = r.json()
            rr = await client.get(
                f"{_API}/users/{handle}/repos",
                params={"sort": "updated", "per_page": 30, "type": "owner"},
            )
            rr.raise_for_status()
            repos = rr.json()
        except Exception:
            return None

    top = sorted(repos, key=lambda x: x.get("stargazers_count", 0), reverse=True)[:5]
    return FounderGithub(
        handle=handle,
        name=user.get("name"),
        bio=user.get("bio"),
        public_repos=user.get("public_repos", 0),
        followers=user.get("followers", 0),
        total_stars=sum(r.get("stargazers_count", 0) for r in repos),
        top_repos=[
            {
                "name": r.get("name"),
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language"),
                "description": r.get("description"),
                "url": r.get("html_url"),
                "pushed_at": r.get("pushed_at"),
            }
            for r in top
        ],
        created_at=user.get("created_at"),
    )
