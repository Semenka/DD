"""SQLite-backed deal state — create, update, save report, list."""

from pathlib import Path

import pytest

from dd_agent.state import DealStatus, DealStore


@pytest.fixture
async def store(tmp_path: Path) -> DealStore:
    s = DealStore(db_path=tmp_path / "deals.db")
    await s.init()
    return s


async def test_create_and_get(store: DealStore):
    rec = await store.create(company_name="Acme")
    assert rec.deal_id
    assert rec.status == DealStatus.QUEUED
    assert rec.company_name == "Acme"

    fetched = await store.get(rec.deal_id)
    assert fetched is not None
    assert fetched.company_name == "Acme"
    assert fetched.status == DealStatus.QUEUED


async def test_update_status(store: DealStore):
    rec = await store.create()
    await store.update_status(rec.deal_id, status=DealStatus.RUNNING,
                              phase="running market subagent", progress_pct=42,
                              company_name="Beta")
    fetched = await store.get(rec.deal_id)
    assert fetched.status == DealStatus.RUNNING
    assert fetched.phase == "running market subagent"
    assert fetched.progress_pct == 42
    assert fetched.company_name == "Beta"


async def test_save_report_marks_done(store: DealStore):
    rec = await store.create()
    await store.save_report(
        rec.deal_id, markdown="# Hi", html="<p>hi</p>",
        citations=[{"n": 1, "title": "x"}],
    )
    fetched = await store.get(rec.deal_id)
    assert fetched.status == DealStatus.DONE
    assert fetched.report_markdown == "# Hi"
    assert fetched.progress_pct == 100


async def test_list_recent_ordering(store: DealStore):
    a = await store.create("A")
    b = await store.create("B")
    c = await store.create("C")
    listing = await store.list_recent(limit=10)
    ids = [r.deal_id for r in listing]
    # most recent first
    assert ids[0] == c.deal_id
    assert ids[-1] == a.deal_id
