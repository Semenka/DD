"""v10 Quality Gate — score a DD report before it ships.

A 14-deal performance audit found a 50% failure rate and, worse, that
garbage reports shipped silently: the "PK", "Round Seed", and
"Investment" deals were all marked `done` and delivered even though they
analyzed a section header instead of a real company. Nothing in the
pipeline checked whether a report was *good* before delivery.

This module is that check. Two layers:

  - **Layer A — deterministic** (`_deterministic_score`): instant, no LLM.
    Seven weighted boolean checks over the assembled markdown + DealContext
    + extras. This layer alone catches every historical garbage report
    (PK scores ~2/10 — fails identity + competitors + founders). Because
    it's deterministic it's fully test-anchorable and is the real
    wrong-company backstop.

  - **Layer B — LLM rubric** (`_llm_score`, best-effort): one cheap Gemini
    call scoring factual grounding / decisiveness / evidence density /
    narrative quality and naming the weakest sections. Falls back to the
    deterministic score on ANY error — never blocks shipping.

Final score = min(deterministic, llm) when both exist, else whichever is
available. The gate decision (ship / retry / flag) lives in the
orchestrator; this module only scores.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..context import DealContext

log = logging.getLogger("dd_agent.quality")


# Section-header tokens that must never be a company name. Mirrors
# ingestion.normalize._SECTION_HEADERS but kept local so quality.py has no
# import-time dependency on the ingestion package (avoids cycles).
_BAD_COMPANY_TOKENS = frozenset({
    "pk", "terms", "round seed", "investment", "memo", "our story", "deck",
    "pitch", "summary", "overview", "confidential", "disclaimer", "tbd",
    "unknown", "company", "deal", "round", "series", "seed",
})

# Controlled-vocab recommendation verdicts the synthesis is supposed to emit.
_RECOMMENDATION_TOKENS = (
    "pass", "lean in", "lead", "revisit",
)


@dataclass
class CheckResult:
    name: str
    weight: int
    passed: bool
    detail: str = ""


@dataclass
class QualityReport:
    score: float                       # 0-10 final
    passed: bool                       # score >= pass_threshold
    deterministic_score: float
    llm_score: float | None = None
    checks: list[CheckResult] = field(default_factory=list)
    weakest_sections: list[dict] = field(default_factory=list)  # [{section, why}]
    verdict: str = ""
    failed_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 1),
            "passed": self.passed,
            "deterministic_score": round(self.deterministic_score, 1),
            "llm_score": round(self.llm_score, 1) if self.llm_score is not None else None,
            "verdict": self.verdict,
            "weakest_sections": self.weakest_sections,
            "failed_checks": self.failed_checks,
            "checks": [
                {"name": c.name, "weight": c.weight, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


# ---------- public API ------------------------------------------------------


PASS_THRESHOLD = 7.0
RETRY_FLOOR = 4.0   # below this → flag with LOW CONFIDENCE; between → retry


async def score_report(
    *,
    ctx: "DealContext",
    markdown: str,
    merged: dict | None = None,
    extras: dict | None = None,
    use_llm: bool = True,
) -> QualityReport:
    """Score an assembled report. `merged`/`extras` are the orchestrator's
    intermediate dicts (citations, photo_analyses, funding_rounds, …); both
    optional so the gate degrades gracefully if a caller only has markdown."""
    merged = merged or {}
    extras = extras if extras is not None else (merged.get("extras") or {})

    det_score, checks = _deterministic_score(ctx, markdown, merged, extras)
    failed = [c.name for c in checks if not c.passed]

    llm_score: float | None = None
    weakest: list[dict] = []
    verdict = ""
    if use_llm:
        try:
            llm_score, weakest, verdict = await _llm_score(ctx, markdown)
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM quality rubric failed (%s) — using deterministic only", exc)

    if llm_score is not None:
        final = min(det_score, llm_score)
    else:
        final = det_score

    if not verdict:
        verdict = _auto_verdict(final, failed)

    return QualityReport(
        score=final,
        passed=final >= PASS_THRESHOLD,
        deterministic_score=det_score,
        llm_score=llm_score,
        checks=checks,
        weakest_sections=weakest,
        verdict=verdict,
        failed_checks=failed,
    )


# ---------- Layer A: deterministic ------------------------------------------


def _deterministic_score(
    ctx: "DealContext", markdown: str, merged: dict, extras: dict,
) -> tuple[float, list[CheckResult]]:
    md = markdown or ""
    main_body = md.split("<details", 1)[0]  # exclude appendix from leak check

    checks: list[CheckResult] = [
        _check_company_identity(ctx),
        _check_four_pillars(md),
        _check_competitors(merged, md),
        _check_founders(ctx, extras),
        _check_recommendation(md),
        _check_citations(merged, md),
        _check_no_leaks(main_body),
    ]
    total_w = sum(c.weight for c in checks)
    earned = sum(c.weight for c in checks if c.passed)
    score = (earned / total_w * 10.0) if total_w else 0.0
    return score, checks


def _check_company_identity(ctx: "DealContext") -> CheckResult:
    """Weight 3 — the single most important check. The company name must
    not be a section-header token, and must appear ≥3× in the source body."""
    name = (ctx.company_name or "").strip()
    if not name:
        return CheckResult("company_identity", 3, False, "no company name")
    if name.lower() in _BAD_COMPANY_TOKENS:
        return CheckResult("company_identity", 3, False,
                           f"company name {name!r} is a section-header token")
    body = (ctx.raw_memo or "") + "\n" + (ctx.raw_deck_text or "")
    if body.strip():
        pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
        n = len(pattern.findall(body))
        if n < 3:
            return CheckResult("company_identity", 3, False,
                               f"{name!r} appears only {n}× in source body")
    return CheckResult("company_identity", 3, True, f"{name!r} confirmed")


def _check_four_pillars(md: str) -> CheckResult:
    """Weight 2 — the Exec Summary must carry all four Bessemer pillars."""
    pillars = ("**Founders.**", "**Co-investors.**",
               "**Growth metrics.**", "**Competitive position.**")
    missing = [p for p in pillars if p not in md]
    if missing:
        return CheckResult("exec_summary_pillars", 2, False,
                           f"missing pillars: {', '.join(missing)}")
    return CheckResult("exec_summary_pillars", 2, True, "all 4 pillars present")


def _check_competitors(merged: dict, md: str) -> CheckResult:
    """Weight 1 — Market section must name ≥3 competitors. Heuristic: count
    distinct capitalized multi-word names in the Market section, plus look
    for a competitor table / segmentation."""
    market = merged.get("market") or _extract_section(md, "Market")
    if not market:
        return CheckResult("competitors_named", 1, False, "no Market section")
    # Count table rows + bulleted competitor mentions as a proxy.
    table_rows = market.count("\n|")
    # Distinct Capitalized-Word company-ish tokens (2+ chars, not sentence start noise)
    caps = set(re.findall(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)?)\b", market))
    signal = max(table_rows - 1, 0) + (1 if len(caps) >= 6 else 0)
    if signal >= 3 or len(caps) >= 8:
        return CheckResult("competitors_named", 1, True,
                           f"{len(caps)} capitalized names, {table_rows} table rows")
    return CheckResult("competitors_named", 1, False,
                       f"weak competitor signal ({len(caps)} names, {table_rows} rows)")


def _check_founders(ctx: "DealContext", extras: dict) -> CheckResult:
    """Weight 1 — at least one founder is quantified OR has a resolved photo."""
    if not ctx.founders:
        return CheckResult("founders_substantive", 1, False, "no founders extracted")
    has_photo = any(getattr(f, "photo_url", None) for f in ctx.founders)
    photo_analyses = extras.get("photo_analyses") or []
    has_photo = has_photo or any(
        isinstance(p, dict) and p.get("available") for p in photo_analyses
    )
    has_bio = any(
        (getattr(f, "bio", None) or "") or getattr(f, "prior_companies", None)
        for f in ctx.founders
    )
    if has_photo or has_bio:
        why = "photo resolved" if has_photo else "bio/prior-companies present"
        return CheckResult("founders_substantive", 1, True, why)
    return CheckResult("founders_substantive", 1, False,
                       "founders named but no photo, bio, or prior companies")


def _check_recommendation(md: str) -> CheckResult:
    """Weight 1 — the synthesis must end with a decisive recommendation from
    the controlled vocabulary."""
    rec = _extract_section(md, "Recommendation")
    blob = (rec or md).lower()
    if any(tok in blob for tok in _RECOMMENDATION_TOKENS):
        return CheckResult("decisive_recommendation", 1, True, "verdict present")
    return CheckResult("decisive_recommendation", 1, False,
                       "no controlled-vocab recommendation found")


def _check_citations(merged: dict, md: str) -> CheckResult:
    """Weight 1 — citation density ≥ 8."""
    book = merged.get("citations")
    n = 0
    if book is not None and hasattr(book, "citations"):
        n = len(book.citations)
    if n == 0:
        n = len(set(re.findall(r"\[(\d+)\]", md)))
    if n >= 8:
        return CheckResult("citation_density", 1, True, f"{n} citations")
    return CheckResult("citation_density", 1, False, f"only {n} citations")


def _check_no_leaks(main_body: str) -> CheckResult:
    """Weight 1 — no 'unknown' / placeholder em-dash leaks in the main body."""
    leaks = []
    if re.search(r"\bunknown\b", main_body, re.IGNORECASE):
        leaks.append("'unknown'")
    for pat in (": —", "— |", "| —", "—\n"):
        if pat in main_body:
            leaks.append(repr(pat))
    if leaks:
        return CheckResult("no_placeholder_leaks", 1, False,
                           f"leaks: {', '.join(leaks)}")
    return CheckResult("no_placeholder_leaks", 1, True, "clean")


# ---------- Layer B: LLM rubric ---------------------------------------------


async def _llm_score(ctx: "DealContext", markdown: str) -> tuple[float, list[dict], str]:
    """Best-effort Gemini rubric. Returns (score_0_10, weakest_sections,
    verdict). Raises on any failure so the caller falls back to deterministic."""
    from pathlib import Path

    rubric_path = Path(__file__).parent / "quality_rubric.md"
    rubric = rubric_path.read_text(encoding="utf-8") if rubric_path.exists() else (
        "Score this VC investment memo 0-10 on factual grounding, decisiveness, "
        "evidence density, and narrative quality."
    )

    # Truncate the report to keep the prompt cheap.
    snippet = markdown[:9000]
    prompt = (
        f"{rubric}\n\n"
        f"Company under review: {ctx.company_name}\n\n"
        f"--- REPORT (truncated) ---\n{snippet}\n--- END ---\n\n"
        "Output ONLY a JSON object, no prose, of the exact shape:\n"
        '{"score": <0-10 number>, "verdict": "<=12 words", '
        '"weakest_sections": [{"section": "<name>", "why": "<one sentence>"}]}'
    )

    from ..modules._llm import _gemini_render
    # 2500-token budget: gemini-3.5-flash spends a large share on internal
    # "thinking" tokens, so a tight 600 budget truncated the JSON mid-output.
    raw = await _gemini_render(prompt, max_tokens=2500)
    data = _extract_json(raw)
    if data and "score" in data:
        score = max(0.0, min(10.0, float(data["score"])))
        weakest = data.get("weakest_sections") or []
        if not isinstance(weakest, list):
            weakest = []
        verdict = str(data.get("verdict") or "").strip()
        return score, weakest, verdict

    # Resilient fallback: the response may be valid but truncated before the
    # closing brace (thinking-token overrun). Pull the fields we can with
    # targeted regexes so a truncated-but-present score still counts.
    score_m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
    if score_m:
        score = max(0.0, min(10.0, float(score_m.group(1))))
        verdict_m = re.search(r'"verdict"\s*:\s*"([^"]*)"', raw)
        verdict = verdict_m.group(1).strip() if verdict_m else ""
        # weakest_sections best-effort: collect any {"section": "...","why":"..."}
        weakest = [
            {"section": s, "why": w}
            for s, w in re.findall(
                r'"section"\s*:\s*"([^"]+)"\s*,\s*"why"\s*:\s*"([^"]*)"', raw,
            )
        ]
        return score, weakest, verdict

    raise ValueError(f"rubric returned no parseable score: {raw[:200]!r}")


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response (handles code fences)."""
    import json
    if not text:
        return None
    # Strip markdown code fences.
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------- helpers ---------------------------------------------------------


def _extract_section(md: str, header: str) -> str:
    """Return the text under a `## {header}` or `### {header}` heading up to
    the next heading of the same-or-higher level. Best-effort."""
    if not md:
        return ""
    lines = md.splitlines()
    out: list[str] = []
    capture = False
    capture_level = 0
    for line in lines:
        m = re.match(r"^(#{2,4})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip().lower()
            if not capture and header.lower() in title:
                capture = True
                capture_level = level
                continue
            elif capture and level <= capture_level:
                break
        if capture:
            out.append(line)
    return "\n".join(out).strip()


def _auto_verdict(score: float, failed: list[str]) -> str:
    if score >= PASS_THRESHOLD:
        return f"Ship-ready ({score:.0f}/10)."
    if score >= RETRY_FLOOR:
        worst = failed[0] if failed else "narrative depth"
        return f"Borderline ({score:.0f}/10) — weakest: {worst}."
    top = ", ".join(failed[:2]) if failed else "multiple checks"
    return f"LOW CONFIDENCE ({score:.0f}/10) — failed: {top}."
