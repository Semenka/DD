"""v8 founder-photo cascade — mock each network tier and assert the
cascade falls through correctly. The cascade is best-effort: every tier
either returns image bytes or None, and `resolve_founder_photo`
short-circuits on the first hit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from dd_agent.context import DealContext, Founder
from dd_agent.data_sources import founder_photo


# ---------- helpers ---------------------------------------------------------


def _mk_ctx(*, deal_id="abc", company="Rivian", website="rivian.com",
            founders=None) -> DealContext:
    # Note: explicit None gets the default; an explicit empty list is honored.
    return DealContext(
        deal_id=deal_id,
        company_name=company,
        website=website,
        founders=[Founder(name="RJ Scaringe")] if founders is None else founders,
    )


# A 1x1 PNG (too small — the persistence layer rejects it, but for cascade
# logic tests we monkey-patch _fetch_image and the deck strategy.)
_SMALL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63f80f0000010001000051f80abc0000000049454e"
    "44ae426082"
)

# Larger fake-image bytes so the persistence layer's >1500 byte check passes.
_FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 2000  # enough bytes to pass size check


# ---------- short-circuit on already-populated photo_url -------------------


@pytest.mark.asyncio
async def test_skips_when_photo_url_already_set(tmp_path: Path):
    f = Founder(name="X Y", photo_url="https://existing.example/photo.jpg")
    ctx = _mk_ctx(founders=[f])
    result = await founder_photo.resolve_founder_photo(
        founder=f, ctx=ctx, save_dir=tmp_path,
    )
    assert result == "https://existing.example/photo.jpg"


# ---------- Wikipedia tier --------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_resolves_via_wikipedia(tmp_path: Path):
    """If Wikipedia returns image bytes, no later tier is tried."""
    f = Founder(name="RJ Scaringe")
    ctx = _mk_ctx(founders=[f])

    with patch(
        "dd_agent.data_sources.founder_photo._from_deck_slides",
        new=AsyncMock(return_value=None),
    ), patch(
        "dd_agent.data_sources.founder_photo._from_wikipedia",
        new=AsyncMock(return_value=_FAKE_JPEG),
    ), patch(
        "dd_agent.data_sources.founder_photo._from_company_team",
        new=AsyncMock(return_value=None),
    ) as mock_team:
        path = await founder_photo.resolve_founder_photo(
            founder=f, ctx=ctx, save_dir=tmp_path,
        )
    assert path is not None
    # Wikipedia hit → company /team never called
    mock_team.assert_not_called()
    # Founder photo_url was mutated to local path
    assert f.photo_url == path
    assert Path(path).exists()


# ---------- LinkedIn og:image placeholder rejection ------------------------


@pytest.mark.asyncio
async def test_linkedin_rejects_placeholder_urls(tmp_path: Path):
    """LinkedIn often returns the generic 'in' logo for unauth requests.
    Our helper rejects those known placeholder URLs."""
    f = Founder(
        name="No One Famous",
        linkedin_url="https://www.linkedin.com/in/no-one-famous",
    )
    ctx = _mk_ctx(founders=[f])

    placeholder_html = (
        '<html><head>'
        '<meta property="og:image" '
        'content="https://static.licdn.com/sc/h/anonymous_default.jpg">'
        '</head></html>'
    )

    class FakeResp:
        status_code = 200
        text = placeholder_html

    async def fake_get(*args, **kwargs):
        return FakeResp()

    import httpx
    async with httpx.AsyncClient() as client:
        client.get = fake_get  # type: ignore[assignment]
        result = await founder_photo._from_linkedin_og(client, f)
    assert result is None


@pytest.mark.asyncio
async def test_linkedin_accepts_real_og_image(tmp_path: Path):
    f = Founder(
        name="Someone Real",
        linkedin_url="https://www.linkedin.com/in/real",
    )

    good_html = (
        '<html><head>'
        '<meta property="og:image" '
        'content="https://media.licdn.com/dms/image/real.jpg">'
        '</head></html>'
    )

    class FakeResp:
        status_code = 200
        text = good_html

    async def fake_get(self, url, *args, **kwargs):
        # First call: profile page. Second call: image fetch.
        if "linkedin.com" in url:
            return FakeResp()
        # Image fetch
        r = type("R", (), {})()
        r.status_code = 200
        r.content = _FAKE_JPEG
        r.headers = {"content-type": "image/jpeg"}
        return r

    import httpx
    with patch.object(httpx.AsyncClient, "get", new=fake_get):
        async with httpx.AsyncClient() as client:
            result = await founder_photo._from_linkedin_og(client, f)
    assert result == _FAKE_JPEG


# ---------- full cascade exhaustion ----------------------------------------


@pytest.mark.asyncio
async def test_cascade_returns_none_when_all_tiers_fail(tmp_path: Path):
    f = Founder(name="Nobody")
    ctx = _mk_ctx(founders=[f])

    with patch.multiple(
        "dd_agent.data_sources.founder_photo",
        # v8 + v9 tiers — pre-discovery + 7 strategies
        _discover_linkedin_url=AsyncMock(return_value=None),
        _discover_company_website=AsyncMock(return_value=None),
        _from_deck_slides=AsyncMock(return_value=None),
        _from_wikipedia=AsyncMock(return_value=None),
        _from_company_team=AsyncMock(return_value=None),
        _from_linkedin_og=AsyncMock(return_value=None),
        _from_web_image_search=AsyncMock(return_value=None),
        _from_grounded=AsyncMock(return_value=None),
        _from_clipping=AsyncMock(return_value=None),
    ):
        result = await founder_photo.resolve_founder_photo(
            founder=f, ctx=ctx, save_dir=tmp_path,
        )
    assert result is None
    assert f.photo_url is None


# ---------- parallel resolve_all -------------------------------------------


@pytest.mark.asyncio
async def test_resolve_all_runs_per_founder(tmp_path: Path):
    ctx = _mk_ctx(founders=[
        Founder(name="Founder A"),
        Founder(name="Founder B"),
    ])

    with patch(
        "dd_agent.data_sources.founder_photo.resolve_founder_photo",
        new=AsyncMock(side_effect=["pathA", "pathB"]),
    ):
        results = await founder_photo.resolve_all_founder_photos(ctx=ctx)

    assert results == {"Founder A": "pathA", "Founder B": "pathB"}


@pytest.mark.asyncio
async def test_resolve_all_handles_individual_exception(tmp_path: Path):
    ctx = _mk_ctx(founders=[
        Founder(name="Founder A"),
        Founder(name="Founder B"),
    ])

    async def fake_resolve(*, founder, **_kwargs):
        if founder.name == "Founder A":
            raise RuntimeError("network exploded")
        return "pathB"

    with patch(
        "dd_agent.data_sources.founder_photo.resolve_founder_photo",
        new=fake_resolve,
    ):
        results = await founder_photo.resolve_all_founder_photos(ctx=ctx)

    # Exception in one founder doesn't break the others
    assert results["Founder A"] is None
    assert results["Founder B"] == "pathB"


@pytest.mark.asyncio
async def test_resolve_all_empty_founders():
    ctx = _mk_ctx(founders=[])
    results = await founder_photo.resolve_all_founder_photos(ctx=ctx)
    assert results == {}


# ---------- persistence ----------------------------------------------------


def test_persist_writes_jpeg(tmp_path: Path):
    # The helper goes through PIL for normalization; use a real PNG via PIL
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not installed")
    import io
    img = Image.new("RGB", (400, 400), color=(120, 60, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    out = founder_photo._persist(buf.getvalue(), tmp_path, "Some Founder",
                                  source="wikipedia")
    assert out is not None
    assert Path(out).exists()
    assert Path(out).name.endswith("_via_wikipedia.jpg")
