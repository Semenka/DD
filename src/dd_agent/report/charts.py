"""v8 chart helpers — replace prose tables and dense data blocks with
visual elements so the memo lands at Bessemer/Sequoia 5-page length.

Two backends:
  - **Inline SVG** for simple "1D" charts (percentile rulers, horizontal
    trait bars, cohort donuts). Zero deps beyond stdlib; embeds directly
    in the markdown via the renderer's raw-HTML pass-through.
  - **matplotlib → PNG → base64 data URL** for "2D" charts (DCF heatmap,
    funding-rounds timeline, ARR trajectory). Renders inside `<img class="dd-chart">`.

Every helper returns either a non-empty string (chart present) or `""`
(no data → template skips the section via `{% if charts.x %}`). No
exceptions propagate up; chart rendering is strictly best-effort.

All SVG output is namespaced (`xmlns="http://www.w3.org/2000/svg"`) so
it renders correctly when extracted from the HTML and viewed standalone.

All matplotlib output uses the Agg backend (no display required); the
import is deferred to render time so unit-tests don't pay the cost.
"""

from __future__ import annotations

import base64
import html as html_escape
import io
import logging
import math
from typing import Iterable

log = logging.getLogger("dd_agent.charts")


# --------------------------------------------------------------------- helpers


def _esc(text: str) -> str:
    """SVG-safe escape — SVG and HTML share the same special-char set."""
    return html_escape.escape(str(text), quote=True)


def _png_to_data_url(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


def _matplotlib():
    """Lazy import so module-import cost stays cheap. Returns the pyplot
    module configured with the Agg backend (headless)."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------- SVG: percentile ruler


def svg_percentile_ruler(
    *,
    percentile: float | None,
    label: str = "",
    p25: float = 25.0,
    p50: float = 50.0,
    p75: float = 75.0,
    width: int = 520,
    height: int = 70,
) -> str:
    """A horizontal 0-100 axis with p25/p50/p75 ticks and a tall vertical
    marker at the deal's percentile. Used for comp-distribution percentile
    visualization (e.g. 'growth at 78th percentile vs public SaaS').

    Returns an SVG string (renders inline in the report) or '' when
    `percentile` is None or out of range."""
    if percentile is None:
        return ""
    try:
        pct = float(percentile)
    except (TypeError, ValueError):
        return ""
    if not (0 <= pct <= 100):
        return ""

    pad_x = 26
    track_y = height - 28
    track_left = pad_x
    track_right = width - pad_x
    track_len = track_right - track_left

    def x_at(p: float) -> float:
        return track_left + (p / 100.0) * track_len

    # Marker position
    marker_x = x_at(pct)

    # Build ticks for 0, 25, 50, 75, 100 with band coloring (cool→warm)
    bands = [
        (0, 25, "#e2e8f0"),
        (25, 50, "#cbd5e1"),
        (50, 75, "#fbcfe8"),
        (75, 100, "#fda4af"),
    ]
    band_svg = "".join(
        f'<rect x="{x_at(lo):.1f}" y="{track_y - 8}" '
        f'width="{x_at(hi) - x_at(lo):.1f}" height="16" fill="{color}" />'
        for lo, hi, color in bands
    )

    # Quartile reference ticks
    ticks_svg = ""
    for v, txt in ((p25, "p25"), (p50, "p50"), (p75, "p75")):
        tx = x_at(v)
        ticks_svg += (
            f'<line x1="{tx:.1f}" y1="{track_y - 12}" x2="{tx:.1f}" y2="{track_y + 12}" '
            f'stroke="#64748b" stroke-width="1" />'
            f'<text x="{tx:.1f}" y="{track_y + 26}" font-size="9" fill="#64748b" '
            f'text-anchor="middle" font-family="ui-sans-serif,system-ui">{txt}</text>'
        )

    marker_color = "#b14b1e"  # accent
    label_text = _esc(label) if label else ""

    return (
        f'<svg class="dd-chart" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px" role="img" aria-label="percentile ruler">'
        f'<title>{label_text}: {pct:.0f}th percentile</title>'
        # Title line
        f'<text x="{pad_x}" y="16" font-size="11" font-weight="600" '
        f'fill="#1a1a1a" font-family="ui-sans-serif,system-ui">{label_text}</text>'
        f'<text x="{width - pad_x}" y="16" font-size="11" font-weight="600" '
        f'fill="{marker_color}" text-anchor="end" '
        f'font-family="ui-sans-serif,system-ui">{pct:.0f}th percentile</text>'
        # Bands
        f'{band_svg}'
        # Ticks
        f'{ticks_svg}'
        # Deal marker
        f'<line x1="{marker_x:.1f}" y1="{track_y - 18}" '
        f'x2="{marker_x:.1f}" y2="{track_y + 18}" '
        f'stroke="{marker_color}" stroke-width="2.5" />'
        f'<polygon points="{marker_x - 5:.1f},{track_y - 20} '
        f'{marker_x + 5:.1f},{track_y - 20} {marker_x:.1f},{track_y - 13}" '
        f'fill="{marker_color}" />'
        f'</svg>'
    )


# --------------------------------------------------------------------- SVG: trait bars


def svg_trait_bars(
    *,
    trait_percentiles: dict[str, float] | None,
    trait_scores: dict[str, float] | None = None,
    width: int = 520,
    bar_height: int = 18,
    gap: int = 8,
) -> str:
    """Horizontal bar chart of founder trait percentiles vs the unicorn
    corpus. Renders one row per trait. Bars are color-graded by direction
    (>50: warm accent; <50: cool gray) and labeled with the percentile.

    Returns '' when `trait_percentiles` is empty or None."""
    if not trait_percentiles:
        return ""
    # Stable order across founders
    order = ("resilience", "intensity", "warmth", "presentation_polish", "energy")
    rows = [(t, trait_percentiles[t]) for t in order if t in trait_percentiles]
    if not rows:
        return ""

    label_col = 130
    val_col = 40
    bar_left = label_col + 8
    bar_right = width - val_col - 8
    bar_len = bar_right - bar_left
    height = len(rows) * (bar_height + gap) + 20

    body = ""
    for i, (trait, pct) in enumerate(rows):
        try:
            pct_f = float(pct)
        except (TypeError, ValueError):
            continue
        pct_f = max(0.0, min(100.0, pct_f))
        y = 14 + i * (bar_height + gap)
        # Highlight bar in accent color if >65, neutral otherwise
        if pct_f >= 65:
            fill = "#b14b1e"
        elif pct_f <= 35:
            fill = "#64748b"
        else:
            fill = "#94a3b8"
        score_label = ""
        if trait_scores and trait in trait_scores:
            try:
                score_label = f" ({float(trait_scores[trait]):.1f}/5)"
            except (TypeError, ValueError):
                pass
        body += (
            f'<text x="{label_col}" y="{y + bar_height - 5}" font-size="10.5" '
            f'fill="#1a1a1a" text-anchor="end" '
            f'font-family="ui-sans-serif,system-ui">{_esc(trait)}{_esc(score_label)}</text>'
            # background track
            f'<rect x="{bar_left}" y="{y}" width="{bar_len}" height="{bar_height}" '
            f'fill="#f1f5f9" rx="2" />'
            # filled bar
            f'<rect x="{bar_left}" y="{y}" width="{bar_len * pct_f / 100:.1f}" '
            f'height="{bar_height}" fill="{fill}" rx="2" />'
            # p50 reference line
            f'<line x1="{bar_left + bar_len / 2:.1f}" y1="{y - 2}" '
            f'x2="{bar_left + bar_len / 2:.1f}" y2="{y + bar_height + 2}" '
            f'stroke="#cbd5e1" stroke-width="1" stroke-dasharray="2,2" />'
            # percentile text
            f'<text x="{bar_right + 8}" y="{y + bar_height - 5}" font-size="10.5" '
            f'fill="#1a1a1a" font-weight="600" '
            f'font-family="ui-sans-serif,system-ui">{pct_f:.0f}th</text>'
        )

    return (
        f'<svg class="dd-chart" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="100%" '
        f'style="max-width:{width}px" role="img" aria-label="founder trait percentiles">'
        f'<title>Founder trait percentiles vs unicorn corpus</title>'
        f'{body}'
        f'</svg>'
    )


# --------------------------------------------------------------------- SVG: cohort donut


def svg_cohort_donut(
    *,
    cohort_breakdown: dict[str, int] | None,
    size: int = 220,
) -> str:
    """Donut chart of nearest-N match cohort distribution. Returns '' when
    breakdown is empty."""
    if not cohort_breakdown:
        return ""
    items = [(c, int(n)) for c, n in cohort_breakdown.items() if n > 0]
    if not items:
        return ""
    total = sum(n for _, n in items)
    if total <= 0:
        return ""

    cx, cy = size / 2, size / 2
    r_outer = size * 0.40
    r_inner = size * 0.25
    palette = ["#b14b1e", "#7c3aed", "#0284c7", "#16a34a", "#eab308", "#475569"]

    arcs = ""
    legend_y = size - len(items) * 14 - 6
    legend = ""
    angle_start = -math.pi / 2  # start at top
    for i, (cohort, n) in enumerate(items):
        frac = n / total
        angle_end = angle_start + frac * 2 * math.pi
        large_arc = 1 if frac > 0.5 else 0
        x1 = cx + r_outer * math.cos(angle_start)
        y1 = cy + r_outer * math.sin(angle_start)
        x2 = cx + r_outer * math.cos(angle_end)
        y2 = cy + r_outer * math.sin(angle_end)
        ix1 = cx + r_inner * math.cos(angle_end)
        iy1 = cy + r_inner * math.sin(angle_end)
        ix2 = cx + r_inner * math.cos(angle_start)
        iy2 = cy + r_inner * math.sin(angle_start)
        color = palette[i % len(palette)]
        arcs += (
            f'<path d="M {x1:.1f} {y1:.1f} A {r_outer:.1f} {r_outer:.1f} 0 '
            f'{large_arc} 1 {x2:.1f} {y2:.1f} L {ix1:.1f} {iy1:.1f} '
            f'A {r_inner:.1f} {r_inner:.1f} 0 {large_arc} 0 {ix2:.1f} {iy2:.1f} Z" '
            f'fill="{color}" />'
        )
        legend += (
            f'<rect x="{size + 12}" y="{legend_y + i * 14 - 9}" '
            f'width="10" height="10" fill="{color}" rx="1" />'
            f'<text x="{size + 28}" y="{legend_y + i * 14}" font-size="10.5" '
            f'fill="#1a1a1a" font-family="ui-sans-serif,system-ui">'
            f'{_esc(cohort)} ({n})</text>'
        )
        angle_start = angle_end

    canvas_w = size + 200
    return (
        f'<svg class="dd-chart" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas_w} {size}" width="100%" '
        f'style="max-width:{canvas_w}px" role="img" aria-label="cohort breakdown">'
        f'<title>Closest-match cohort breakdown (n={total})</title>'
        f'{arcs}'
        f'<text x="{cx}" y="{cy + 5}" font-size="14" font-weight="600" '
        f'fill="#1a1a1a" text-anchor="middle" '
        f'font-family="ui-sans-serif,system-ui">n={total}</text>'
        f'{legend}'
        f'</svg>'
    )


# --------------------------------------------------------------------- matplotlib: DCF heatmap


def png_dcf_heatmap(*, sweep: list[dict] | None) -> str:
    """Reverse-DCF heatmap: rows = terminal FCF margin, cols = years to
    terminal, cell = required annual growth (%). Color graded
    green→yellow→red so 'how stretched is this valuation' is visible at a
    glance. Returns an `<img>` HTML element (data URL src) or ''."""
    if not sweep:
        return ""
    try:
        plt = _matplotlib()
        import numpy as np
    except Exception:
        return ""

    # Build 2D matrix keyed by (margin, years).
    margins = sorted({float(r["fcf_margin"]) for r in sweep
                      if r.get("fcf_margin") is not None})
    years_list = sorted({int(r["years_to_terminal"]) for r in sweep
                         if r.get("years_to_terminal") is not None})
    if not margins or not years_list:
        return ""

    grid = np.full((len(margins), len(years_list)), np.nan)
    for r in sweep:
        try:
            m = float(r["fcf_margin"])
            y = int(r["years_to_terminal"])
            g = r.get("required_growth_yoy")
            if g is None:
                continue
            i = margins.index(m)
            j = years_list.index(y)
            grid[i, j] = float(g) * 100.0  # percent
        except (KeyError, ValueError, TypeError):
            continue
    if np.isnan(grid).all():
        return ""

    fig, ax = plt.subplots(figsize=(5.2, 2.5), dpi=140)
    cmap = plt.get_cmap("RdYlGn_r")  # green low, red high (high growth = stretched)
    vmin, vmax = 10, 100
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(years_list)), [f"{y}y" for y in years_list], fontsize=9)
    ax.set_yticks(range(len(margins)), [f"{int(m*100)}%" for m in margins], fontsize=9)
    ax.set_xlabel("Years to terminal", fontsize=9)
    ax.set_ylabel("Terminal FCF margin", fontsize=9)
    ax.set_title("Required annual revenue growth (%) to justify ask", fontsize=10, pad=8)
    for i in range(len(margins)):
        for j in range(len(years_list)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="#1a1a1a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    png = buf.getvalue()
    if not png:
        return ""
    return (
        f'<figure class="chart">'
        f'<img class="dd-chart" alt="Reverse-DCF required-growth heatmap" '
        f'src="{_png_to_data_url(png)}" />'
        f'<figcaption>Required annual revenue growth to justify the ask, '
        f'across terminal FCF margins and horizons. Green = pedestrian; '
        f'red = unprecedented in public-SaaS history.</figcaption>'
        f'</figure>'
    )


# --------------------------------------------------------------------- matplotlib: funding timeline


def png_funding_timeline(*, rounds: list[dict] | None) -> str:
    """Horizontal timeline of funding rounds. X axis = date, bubble size
    scales with amount_usd, label is 'Round · $XM · lead'. Returns
    `<img>` element or ''."""
    if not rounds:
        return ""
    # Pull (date_year, amount, round_label, lead) tuples.
    pts: list[tuple[float, float, str, str]] = []
    for r in rounds:
        d = (r.get("date") or "").strip()
        if not d:
            continue
        # Parse year (date may be YYYY, YYYY-MM, YYYY-MM-DD)
        try:
            year = float(d[:4])
            month = float(d[5:7]) if len(d) >= 7 else 6.0
            x = year + (month - 1) / 12.0
        except (ValueError, TypeError):
            continue
        amt = r.get("amount_usd")
        if amt is None:
            continue
        try:
            amt_f = float(amt)
        except (TypeError, ValueError):
            continue
        if amt_f <= 0:
            continue
        rt = r.get("round_type") or "round"
        lead_list = r.get("lead_investors") or []
        lead = lead_list[0] if lead_list else ""
        pts.append((x, amt_f, str(rt), str(lead)))
    if not pts:
        return ""

    try:
        plt = _matplotlib()
    except Exception:
        return ""

    pts.sort(key=lambda t: t[0])
    xs = [p[0] for p in pts]
    ys = [1.0] * len(pts)
    amts = [p[1] for p in pts]
    max_amt = max(amts)
    sizes = [max(160.0, 1800.0 * (a / max_amt)) for a in amts]

    fig, ax = plt.subplots(figsize=(5.2, 1.7), dpi=140)
    ax.scatter(xs, ys, s=sizes, c="#b14b1e", alpha=0.62, edgecolors="#1a1a1a", linewidths=0.6)
    for (x, amt, rt, lead), y in zip(pts, ys):
        amt_label = f"${amt/1e6:.0f}M" if amt < 1e9 else f"${amt/1e9:.1f}B"
        lead_clause = f" · {lead}" if lead else ""
        ax.annotate(
            f"{rt}\n{amt_label}{lead_clause}",
            xy=(x, y), xytext=(0, 22 if (xs.index(x) % 2 == 0) else -34),
            textcoords="offset points",
            ha="center", fontsize=8, color="#1a1a1a",
        )
    ax.set_yticks([])
    ax.set_xlabel("Year", fontsize=9)
    ax.set_title("Funding history", fontsize=10, pad=6)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylim(0.0, 2.0)
    # Pad x-range a bit
    pad = 0.5 if len(xs) > 1 else 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    png = buf.getvalue()
    if not png:
        return ""
    return (
        f'<figure class="chart">'
        f'<img class="dd-chart" alt="Funding-rounds timeline" '
        f'src="{_png_to_data_url(png)}" />'
        f'<figcaption>Funding history. Bubble area scales with round size.</figcaption>'
        f'</figure>'
    )


# --------------------------------------------------------------------- matplotlib: ARR trajectory


def png_arr_trajectory(*, points: Iterable[tuple[str, float]] | None) -> str:
    """Optional ARR-over-time line plot. `points` is an iterable of
    (date_label, arr_usd). Renders only when ≥ 2 points. Returns '' on
    insufficient data."""
    if not points:
        return ""
    pts = list(points)
    if len(pts) < 2:
        return ""
    try:
        plt = _matplotlib()
    except Exception:
        return ""

    xs = list(range(len(pts)))
    ys = [float(p[1]) for p in pts]
    labels = [str(p[0]) for p in pts]

    fig, ax = plt.subplots(figsize=(5.0, 2.2), dpi=140)
    ax.plot(xs, ys, "-o", color="#b14b1e", linewidth=2.0, markersize=6)
    ax.set_xticks(xs, labels, fontsize=8, rotation=0)
    ax.set_ylabel("ARR (USD)", fontsize=9)
    ax.set_title("ARR trajectory", fontsize=10, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Format y ticks as $XM / $XB
    def _fmt(v, _pos):
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        if v >= 1e6:
            return f"${v/1e6:.0f}M"
        return f"${v:.0f}"
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    png = buf.getvalue()
    if not png:
        return ""
    return (
        f'<figure class="chart">'
        f'<img class="dd-chart" alt="ARR trajectory" '
        f'src="{_png_to_data_url(png)}" />'
        f'</figure>'
    )


# --------------------------------------------------------------------- top-level orchestrator helper


def build_chart_bundle(
    *,
    extras: dict | None,
    metrics_points: Iterable[tuple[str, float]] | None = None,
) -> dict:
    """Convenience: build the full `charts` dict the template expects,
    given the orchestrator's `extras` payload. Best-effort — any individual
    chart that errors silently maps to '' so the template skips it."""
    extras = extras or {}
    charts: dict = {}

    # Comp-distribution percentile ruler (Market section)
    rd = extras.get("reverse_dcf") or {}
    growth_pct = rd.get("growth_percentile_vs_public")
    if growth_pct is not None:
        try:
            charts["market_comp_ruler"] = svg_percentile_ruler(
                percentile=float(growth_pct),
                label="Required-growth percentile vs public SaaS",
            )
        except Exception as exc:
            log.warning("market_comp_ruler render failed: %s", exc)

    # DCF heatmap (Traction section)
    try:
        charts["dcf_heatmap"] = png_dcf_heatmap(sweep=extras.get("sweep"))
    except Exception as exc:
        log.warning("dcf_heatmap render failed: %s", exc)
        charts["dcf_heatmap"] = ""

    # Funding-rounds timeline (Co-investors section)
    try:
        charts["funding_timeline"] = png_funding_timeline(
            rounds=extras.get("funding_rounds"),
        )
    except Exception as exc:
        log.warning("funding_timeline render failed: %s", exc)
        charts["funding_timeline"] = ""

    # Per-founder trait bars (Founders section)
    trait_bars: dict[str, str] = {}
    for p in (extras.get("photo_analyses") or []):
        if not isinstance(p, dict) or not p.get("available"):
            continue
        name = p.get("founder_name") or ""
        if not name:
            continue
        try:
            chart = svg_trait_bars(
                trait_percentiles=p.get("trait_percentiles"),
                trait_scores=p.get("trait_scores"),
            )
        except Exception as exc:
            log.warning("trait_bars render failed for %s: %s", name, exc)
            chart = ""
        if chart:
            trait_bars[name] = chart
    if trait_bars:
        charts["trait_bars_by_founder"] = trait_bars

    # ARR trajectory if a points series was supplied
    try:
        charts["arr_trajectory"] = png_arr_trajectory(points=metrics_points)
    except Exception as exc:
        log.warning("arr_trajectory render failed: %s", exc)
        charts["arr_trajectory"] = ""

    return charts
