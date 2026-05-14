"""DealContext — the typed object that flows from ingestion → orchestrator → subagents.

Subagents read from DealContext and never re-parse raw inputs. Anything a subagent
needs from the deck/memo/site must be normalized here first.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Founder:
    name: str
    role: str | None = None                # "CEO", "CTO", etc.
    linkedin_url: str | None = None
    twitter_handle: str | None = None
    github_handle: str | None = None
    photo_url: str | None = None
    bio: str | None = None                 # short bio from deck or website
    prior_companies: list[str] = field(default_factory=list)


@dataclass
class Investor:
    name: str
    type: str | None = None                # "vc" | "angel" | "strategic" | "accelerator"
    round: str | None = None               # "seed", "series_a", etc.
    is_lead: bool = False


@dataclass
class Metrics:
    arr_usd: float | None = None
    mrr_usd: float | None = None
    growth_rate_yoy: float | None = None   # 0.0–10.0 (e.g. 2.5 = 250%)
    burn_usd_monthly: float | None = None
    runway_months: float | None = None
    gross_margin: float | None = None
    customer_count: int | None = None
    nps: float | None = None
    churn_monthly: float | None = None


@dataclass
class DealContext:
    """Everything subsequent stages need to know about the deal."""

    deal_id: str
    company_name: str
    one_liner: str | None = None
    sector: str | None = None              # "vertical_saas", "infra", "ai_devtools", etc.
    stage: str | None = None               # "seed", "series_a", etc.
    founded_year: int | None = None
    hq_location: str | None = None
    website: str | None = None

    founders: list[Founder] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    existing_investors: list[Investor] = field(default_factory=list)

    ask_amount_usd: float | None = None
    ask_valuation_usd: float | None = None
    pre_money_usd: float | None = None
    round_type: str | None = None

    raw_memo: str | None = None
    raw_deck_text: str | None = None
    raw_website_text: str | None = None

    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def headline(self) -> str:
        bits = [self.company_name]
        if self.sector:
            bits.append(f"({self.sector})")
        if self.stage:
            bits.append(f"— {self.stage}")
        return " ".join(bits)
