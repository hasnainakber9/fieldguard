"""Dependency-free inline SVG charts for the self-contained HTML report.

Colours come from CSS custom properties defined in the report template, so the charts follow the light and
dark themes. Nothing here loads external resources.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

SEVERITY_ORDER = ("critical", "warning", "info")
SEVERITY_VAR = {"critical": "var(--sev-critical)", "warning": "var(--sev-warning)", "info": "var(--sev-info)"}


def stacked_hbar(
    rows: Sequence[tuple[str, Mapping[str, float]]], *, width: int = 680, label_width: int = 240, bar_height: int = 18
) -> str:
    """Horizontal stacked bars: one row per label, segments by severity."""
    if not rows:
        return "<p class='muted'>No issues recorded.</p>"
    totals = [sum(float(v.get(s, 0)) for s in SEVERITY_ORDER) for _, v in rows]
    scale_max = max(max(totals), 1.0)
    plot_w = width - label_width - 60
    gap = 6
    height = len(rows) * (bar_height + gap) + 26
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Issues by check and severity' class='chart'>"]
    y = 14
    for (label, counts), total in zip(rows, totals, strict=True):
        parts.append(
            f"<text x='{label_width - 8}' y='{y + bar_height - 5}' text-anchor='end' class='axis'>{escape(label)}</text>"
        )
        x = float(label_width)
        for sev in SEVERITY_ORDER:
            val = float(counts.get(sev, 0))
            if val <= 0:
                continue
            w = plot_w * val / scale_max
            parts.append(
                f"<rect x='{x:.1f}' y='{y}' width='{w:.1f}' height='{bar_height}' fill='{SEVERITY_VAR[sev]}'>"
                f"<title>{escape(label)}: {int(val)} {sev}</title></rect>"
            )
            x += w
        parts.append(f"<text x='{x + 6:.1f}' y='{y + bar_height - 5}' class='value'>{int(total)}</text>")
        y += bar_height + gap
    legend_y = height - 6
    lx = label_width
    for sev in SEVERITY_ORDER:
        parts.append(f"<rect x='{lx}' y='{legend_y - 10}' width='12' height='12' fill='{SEVERITY_VAR[sev]}'/>")
        parts.append(f"<text x='{lx + 18}' y='{legend_y}' class='axis'>{sev}</text>")
        lx += 90
    parts.append("</svg>")
    return "".join(parts)


def bar_scores(
    rows: Sequence[tuple[str, float, bool]],
    *,
    width: int = 680,
    label_width: int = 160,
    bar_height: int = 14,
    max_value: float = 100.0,
) -> str:
    """Horizontal bars for 0-100 scores; flagged rows use the critical colour, others the neutral accent."""
    if not rows:
        return "<p class='muted'>No enumerators scored.</p>"
    plot_w = width - label_width - 50
    gap = 6
    height = len(rows) * (bar_height + gap) + 8
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Enumerator risk scores' class='chart'>"]
    y = 4
    for label, score, flagged in rows:
        val = max(0.0, min(float(score), max_value))
        w = plot_w * val / max_value
        colour = "var(--sev-critical)" if flagged else "var(--accent)"
        parts.append(
            f"<text x='{label_width - 8}' y='{y + bar_height - 3}' text-anchor='end' class='axis'>{escape(label)}</text>"
        )
        parts.append(f"<rect x='{label_width}' y='{y}' width='{plot_w:.1f}' height='{bar_height}' class='track'/>")
        parts.append(f"<rect x='{label_width}' y='{y}' width='{w:.1f}' height='{bar_height}' fill='{colour}'/>")
        parts.append(
            f"<text x='{label_width + plot_w + 6:.1f}' y='{y + bar_height - 3}' class='value'>{score:.0f}</text>"
        )
        y += bar_height + gap
    parts.append("</svg>")
    return "".join(parts)


def sparkline_counts(values: Sequence[float], *, width: int = 320, height: int = 48) -> str:
    """Compact column chart for small count series (e.g. interviews per day)."""
    if not values:
        return ""
    vmax = max(max(values), 1.0)
    n = len(values)
    bw = width / n
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Counts' class='spark'>"]
    for i, v in enumerate(values):
        h = (height - 4) * float(v) / vmax
        parts.append(
            f"<rect x='{i * bw + 1:.1f}' y='{height - h:.1f}' width='{max(bw - 2, 1):.1f}' height='{h:.1f}' fill='var(--accent)'/>"
        )
    parts.append("</svg>")
    return "".join(parts)
