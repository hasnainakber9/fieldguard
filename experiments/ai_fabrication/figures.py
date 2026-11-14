"""Figures for the report, from results/test/ (held-out seeds, content-only condition).

    uv run python experiments/ai_fabrication/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import FROZEN, group_name  # noqa: E402
from common import RESULTS  # noqa: E402

BLUE, ORANGE, MUTED, INK, INK2, GRID, AXIS = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SHORT = {
    "Rule-based (FieldGuard generator)": "Rule-based\nfabricator",
    "claude-opus-5-5-cc / S0_naive": "AI, S0\nnaive",
    "claude-opus-5-5-cc / S1_informed": "AI, S1\ninformed",
    "claude-opus-5-5-cc / S2_careful": "AI, S2\ncareful",
}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.family": "DejaVu Sans"})


def load() -> pd.DataFrame:
    c = pd.read_csv(RESULTS / "test" / "scorecards.csv")
    c["group"] = c.apply(group_name, axis=1)
    return c[c["paradata"] == "content_only"].copy()


def fig_recall(c: pd.DataFrame, out: Path) -> None:
    z = c[[f"z_{s}" for s in FROZEN["signals"]]].abs()
    c["new_flag"] = z.ge(FROZEN["threshold"]).any(axis=1) & c["n"].ge(FROZEN["min_n"])
    groups = list(SHORT)
    fab = c[c["is_fabricator"]]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    y = np.arange(len(groups))
    h = 0.34
    for k, (col, color, label) in enumerate(
        [("flagged", BLUE, "FieldGuard as released"), ("new_flag", ORANGE, "New answer-level signals (frozen)")]
    ):
        vals, texts = [], []
        for g in groups:
            f = fab[fab["group"] == g]
            hit = int(f[col].astype(bool).sum())
            vals.append(hit / len(f))
            texts.append(f"{hit}/{len(f)}")
        yy = y + (k - 0.5) * h
        ax.barh(yy, vals, height=h - 0.04, color=color, label=label)
        for v, t, yi in zip(vals, texts, yy, strict=True):
            ax.text(v + 0.015, yi, t, va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [SHORT[g] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Fabricating enumerators flagged (held-out surveys, answers only)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_signals(c: pd.DataFrame, out: Path) -> None:
    labels = {"grid_sd": "Spread of ratings\nwithin an interview", "spread": "Distance to the most\nsimilar household",
              "typicality": "Distance from the\nmedian household"}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.8), dpi=200, sharey=True)
    rng = np.random.default_rng(0)
    # Honest enumerators appear in every condition's run; show each once (from the S0 runs).
    honest_once = ~c["is_fabricator"] & (c["strategy"] == "S0_naive")
    cats = [("Honest", honest_once, MUTED, "o", "none"),
            ("Rule-based fabricator", c["is_fabricator"] & (c["fabricator"] != "language model"), BLUE, "s", BLUE),
            ("AI fabricator", c["is_fabricator"] & (c["fabricator"] == "language model"), ORANGE, "D", ORANGE)]
    for ax, s in zip(axes, FROZEN["signals"], strict=True):
        for i, (name, mask, color, marker, face) in enumerate(cats):
            v = c.loc[mask, f"z_{s}"].to_numpy(dtype=float)
            x = i + rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(x, v, s=10, marker=marker, facecolors=face, edgecolors=color, linewidths=0.7, label=name)
        for t in (-FROZEN["threshold"], FROZEN["threshold"]):
            ax.axhline(t, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.axhline(0, color=GRID, linewidth=0.6, zorder=0)
        ax.set_title(labels[s], fontsize=8.5, color=INK)
        ax.set_xticks([0, 1, 2], ["Honest", "Rule-\nbased", "AI"], fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Robust z-score vs. peers")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    c = load()
    figs = Path(__file__).resolve().parent / "report"
    figs.mkdir(exist_ok=True)
    fig_recall(c, figs / "fig1_recall.png")
    fig_signals(c, figs / "fig2_signals.png")
    print("figures written to", figs)


if __name__ == "__main__":
    main()
