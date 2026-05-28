"""SQLite-backed state for deal records.

A deal moves through phases: queued → ingesting → running → done | failed.
Reports (markdown + html) are stored as TEXT in the deal row when done.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite


class DealStatus(str, Enum):
    QUEUED = "queued"
    INGESTING = "ingesting"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class DealRecord:
    deal_id: str
    company_name: str | None
    status: DealStatus
    phase: str                   # human-readable e.g. "running founders subagent"
    progress_pct: int            # 0-100
    created_at: float
    updated_at: float
    error: str | None = None
    context_json: str | None = None
    report_markdown: str | None = None
    report_html: str | None = None
    report_pdf_path: str | None = None
    citations_json: str | None = None
    quality_score: float | None = None      # v10: 0-10 quality-gate score
    quality_notes: str | None = None         # v10: gate verdict / failed checks

    def to_summary(self) -> dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "company": self.company_name,
            "status": self.status.value,
            "phase": self.phase,
            "progress_pct": self.progress_pct,
            "created_at": self.created_at,
            "error": self.error,
            "quality_score": self.quality_score,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS deals (
    deal_id TEXT PRIMARY KEY,
    company_name TEXT,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    progress_pct INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    error TEXT,
    context_json TEXT,
    report_markdown TEXT,
    report_html TEXT,
    report_pdf_path TEXT,
    citations_json TEXT,
    quality_score REAL,
    quality_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_deals_status ON deals(status);
CREATE INDEX IF NOT EXISTS idx_deals_created_at ON deals(created_at);
"""

# Run light-touch migrations so existing DBs pick up new columns. Each runs
# inside a try/except so "duplicate column" on an already-migrated DB is a
# harmless no-op.
_MIGRATIONS = [
    "ALTER TABLE deals ADD COLUMN report_pdf_path TEXT",
    "ALTER TABLE deals ADD COLUMN quality_score REAL",   # v10
    "ALTER TABLE deals ADD COLUMN quality_notes TEXT",   # v10
]


class DealStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = str(db_path or os.environ.get("DD_DB_PATH", "./data/deals.db"))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            # Apply each migration; tolerate "duplicate column" errors when the
            # column already exists.
            for stmt in _MIGRATIONS:
                try:
                    await db.execute(stmt)
                except Exception:
                    pass
            await db.commit()

    async def create(self, company_name: str | None = None) -> DealRecord:
        now = time.time()
        record = DealRecord(
            deal_id=uuid.uuid4().hex[:12],
            company_name=company_name,
            status=DealStatus.QUEUED,
            phase="queued",
            progress_pct=0,
            created_at=now,
            updated_at=now,
        )
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO deals "
                "(deal_id, company_name, status, phase, progress_pct, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.deal_id,
                    record.company_name,
                    record.status.value,
                    record.phase,
                    record.progress_pct,
                    record.created_at,
                    record.updated_at,
                ),
            )
            await db.commit()
        return record

    async def update_status(
        self,
        deal_id: str,
        status: DealStatus | None = None,
        phase: str | None = None,
        progress_pct: int | None = None,
        company_name: str | None = None,
        error: str | None = None,
    ) -> None:
        sets: list[str] = ["updated_at = ?"]
        vals: list[Any] = [time.time()]
        if status is not None:
            sets.append("status = ?"); vals.append(status.value)
        if phase is not None:
            sets.append("phase = ?"); vals.append(phase)
        if progress_pct is not None:
            sets.append("progress_pct = ?"); vals.append(progress_pct)
        if company_name is not None:
            sets.append("company_name = ?"); vals.append(company_name)
        if error is not None:
            sets.append("error = ?"); vals.append(error)
        vals.append(deal_id)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"UPDATE deals SET {', '.join(sets)} WHERE deal_id = ?", vals)
            await db.commit()

    async def save_context(self, deal_id: str, context: dict[str, Any]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE deals SET context_json = ?, updated_at = ? WHERE deal_id = ?",
                (json.dumps(context, default=str), time.time(), deal_id),
            )
            await db.commit()

    async def save_report(
        self,
        deal_id: str,
        markdown: str,
        html: str,
        citations: list[dict[str, Any]],
        pdf_path: str | None = None,
        quality_score: float | None = None,
        quality_notes: str | None = None,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE deals SET report_markdown = ?, report_html = ?, "
                "report_pdf_path = ?, citations_json = ?, quality_score = ?, "
                "quality_notes = ?, status = ?, phase = ?, "
                "progress_pct = ?, updated_at = ? WHERE deal_id = ?",
                (
                    markdown,
                    html,
                    pdf_path,
                    json.dumps(citations),
                    quality_score,
                    quality_notes,
                    DealStatus.DONE.value,
                    "done",
                    100,
                    time.time(),
                    deal_id,
                ),
            )
            await db.commit()

    async def get(self, deal_id: str) -> DealRecord | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM deals WHERE deal_id = ?", (deal_id,))
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def list_recent(self, limit: int = 50) -> list[DealRecord]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM deals ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            rows = await cur.fetchall()
        return [_row_to_record(r) for r in rows]


def _row_to_record(row: aiosqlite.Row) -> DealRecord:
    return DealRecord(
        deal_id=row["deal_id"],
        company_name=row["company_name"],
        status=DealStatus(row["status"]),
        phase=row["phase"],
        progress_pct=row["progress_pct"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        error=row["error"],
        context_json=row["context_json"],
        report_markdown=row["report_markdown"],
        report_html=row["report_html"],
        report_pdf_path=row["report_pdf_path"] if "report_pdf_path" in row.keys() else None,
        citations_json=row["citations_json"],
        quality_score=row["quality_score"] if "quality_score" in row.keys() else None,
        quality_notes=row["quality_notes"] if "quality_notes" in row.keys() else None,
    )
