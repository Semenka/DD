"""Company-identity verification.

After `normalize()` returns a `DealContext`, we verify the extracted
`company_name` actually corresponds to a company that appears in the memo
body or the source filename. This catches the entire class of failures
where the LLM/heuristic grabs a section header ("PK", "TERMS", "Round Seed",
"Our Story") or a generic word ("Investment") instead of the real company.

The Rivian regression: a `.docx` file was opened as binary, its ZIP magic
bytes `PK\\x03\\x04` were extracted as the company name, and a 14-minute
pipeline ran on a deal labeled "PK" instead of "Rivian". The .docx
dispatcher fixes the source problem; this verifier is the catch-all
backstop for the next failure that surfaces.

Three rescue strategies (in order):
  1. filename-derived candidate — parse `Rivan_Investment_Memo.docx` → "Rivan"
  2. frequent-noun candidate    — pick the most-frequent capitalized noun
  3. grounded LLM confirmation  — ask Perplexity/Gemini "what company is this?"

If all three fail to produce a consistent answer, the verifier returns
`VerifyResult(verified=False, ...)` and the orchestrator should refuse to
run the pipeline.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("dd_agent.identity")


# Stop words for the frequent-noun heuristic — exclude common prose words
# that happen to be capitalized in memo headings.
_STOP_WORDS = frozenset({
    "The", "A", "An", "And", "Or", "But", "If", "Then", "So", "Of", "In",
    "On", "At", "To", "For", "With", "By", "From", "As", "Is", "Are", "Was",
    "Were", "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
    "Will", "Would", "Should", "Could", "Can", "May", "Might", "Must",
    "I", "You", "He", "She", "It", "We", "They", "This", "That", "These",
    "Those", "There", "Here", "Where", "When", "Why", "How", "What", "Who",
    # Common memo prose words
    "Company", "Founder", "Founders", "Team", "Product", "Market", "Customer",
    "Customers", "Investor", "Investors", "Revenue", "ARR", "Growth", "Round",
    "Series", "Seed", "Pre", "Stage", "Investment", "Memo", "Deal", "Pitch",
    "Deck", "Note", "Notes", "Section", "Page", "Summary", "Overview",
    "Highlights", "Confidential", "Disclaimer", "Legal", "Terms", "Story",
    "Mission", "Vision", "Values", "Why", "Now", "Use", "Funds", "Cap",
    "Table", "PK", "TBD", "Unknown",
})


@dataclass
class VerifyResult:
    verified: bool
    company_name: str | None        # the verified or corrected name
    original_name: str | None       # what normalize() originally returned
    source: str                     # "memo-frequency" | "filename" | "grounded" | "refused"
    notes: str | None = None        # explanation for status logs


# ---------- public API ------------------------------------------------------


async def verify_company_identity(
    *,
    extracted_name: str | None,
    raw_memo: str,
    raw_deck: str | None = None,
    source_filename: str | None = None,
    min_occurrences: int = 3,
) -> VerifyResult:
    """Verify the extracted company name and recover via rescue strategies
    when it fails."""
    body = (raw_memo or "") + "\n" + (raw_deck or "")

    # Strategy 0: extracted_name appears in body ≥N times → trust it
    if extracted_name and _passes_body_check(extracted_name, body, min_occurrences):
        return VerifyResult(
            verified=True,
            company_name=extracted_name,
            original_name=extracted_name,
            source="memo-frequency",
            notes=f"'{extracted_name}' appears in memo body ≥{min_occurrences} times",
        )

    # Strategy 1: filename-derived candidate
    fname_candidate = _from_filename(source_filename)
    if fname_candidate and _passes_body_check(fname_candidate, body, 1):
        return VerifyResult(
            verified=True,
            company_name=fname_candidate,
            original_name=extracted_name,
            source="filename",
            notes=f"filename suggests {fname_candidate!r}; appears in body",
        )

    # Strategy 2: most-frequent capitalized noun in the body
    freq_candidate = _most_frequent_capitalized(body, min_count=min_occurrences)
    if freq_candidate:
        return VerifyResult(
            verified=True,
            company_name=freq_candidate,
            original_name=extracted_name,
            source="memo-frequency",
            notes=f"most-frequent capitalized noun in body: {freq_candidate!r}",
        )

    # Strategy 3: grounded LLM rescue (best-effort)
    grounded_candidate = await _grounded_rescue(body, fname_candidate)
    if grounded_candidate and _passes_body_check(grounded_candidate, body, 1):
        return VerifyResult(
            verified=True,
            company_name=grounded_candidate,
            original_name=extracted_name,
            source="grounded",
            notes=f"grounded LLM resolved to {grounded_candidate!r}",
        )

    # All strategies failed → refuse pipeline
    return VerifyResult(
        verified=False,
        company_name=None,
        original_name=extracted_name,
        source="refused",
        notes=(
            "could not confirm company identity. Rename the file to include "
            "the company name, or add a 'Company: X' line to the memo."
        ),
    )


# ---------- internals -------------------------------------------------------


def _passes_body_check(name: str, body: str, min_count: int) -> bool:
    """Case-insensitive count of whole-word occurrences."""
    if not name or not body:
        return False
    # Reject obvious section-header bare matches
    if name.strip().lower() in {s.lower() for s in _STOP_WORDS}:
        return False
    pattern = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
    return len(pattern.findall(body)) >= min_count


def _from_filename(filename: str | None) -> str | None:
    """Parse a path or filename like `Rivan_Investment_Memo.docx` or
    `Acme-DD-Series-A.pdf` and return the most company-name-shaped token."""
    if not filename:
        return None
    stem = Path(filename).stem  # strip .docx / .pdf
    # Split on common separators
    tokens = re.split(r"[_\-\s]+", stem)
    # Filter out tokens that match stop words or look like deal-meta words
    candidates = [t for t in tokens if t and t.lower() not in
                  {s.lower() for s in _STOP_WORDS}]
    if not candidates:
        return None
    # Prefer the first non-meta token (filenames typically lead with company)
    return candidates[0]


_CAP_WORD_RE = re.compile(r"\b([A-Z][a-z][a-zA-Z]{1,30})\b")


def _most_frequent_capitalized(body: str, min_count: int = 3) -> str | None:
    """Find capitalized words that occur ≥min_count times in the body and
    aren't stop words. Picks the highest-frequency candidate."""
    if not body:
        return None
    words = _CAP_WORD_RE.findall(body)
    # Filter stop words
    filtered = [w for w in words if w not in _STOP_WORDS]
    if not filtered:
        return None
    counts = Counter(filtered)
    most_common = counts.most_common(5)
    for word, count in most_common:
        if count >= min_count:
            return word
    return None


async def _grounded_rescue(body: str, hint: str | None) -> str | None:
    """Ask Perplexity/Gemini: 'what company is this memo about?'

    Best-effort — returns None on any failure. Truncates body to ~3000 chars
    to keep the prompt cheap."""
    if not body or len(body.strip()) < 200:
        return None
    try:
        from ..data_sources.search import ask_grounded
    except ImportError:
        return None

    hint_clause = f" (filename suggests '{hint}')" if hint else ""
    snippet = body[:3000]
    prompt = (
        f"What single company is the following memo about?{hint_clause}\n\n"
        f"Output ONLY the company's name as a single word or short phrase, "
        f"no preamble, no quotes, no explanation. If you cannot tell, output "
        f"the literal string 'UNCLEAR'.\n\n"
        f"--- MEMO ---\n{snippet}\n--- END ---"
    )
    try:
        ans = await ask_grounded(prompt, max_sources=2, max_tokens=50)
    except Exception:
        return None
    if ans is None:
        return None
    name = (ans.text or "").strip().strip("\"'.,;:").strip()
    if not name or name.upper() == "UNCLEAR" or len(name) > 60:
        return None
    return name
