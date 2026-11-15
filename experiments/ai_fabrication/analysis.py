"""Tables and figures from results/<split>/ (written by evaluate.py).

    uv run python experiments/ai_fabrication/analysis.py --split dev     # look, choose, then freeze FROZEN below
    uv run python experiments/ai_fabrication/analysis.py --split test    # score the frozen choice once

Counts are small (3 fabricators per survey), so results are reported as counts with Wilson 95% intervals, and the
ROC-AUC per survey is averaged over surveys.
"""
 
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS  # noqa: E402
from signatures import SIGNALS  # noqa: E402

# Fixed on the development seeds before the held-out seeds were scored (see results/dev/analysis.md).
FROZEN: dict = {
    "signals": ["grid_sd", "spread", "typicality"],
    "threshold": 3.5,
    "min_n": 8,
    "chosen_on": "dev seeds 101-102, 15 Nov 2026, before any held-out result was computed",
    "dev_result": "content-only: AI fabricators 13/18, rule-based 5/6, honest 0/216 enumerator-runs",
}

LABEL = {
    ("rule-based", "-"): "Rule-based (FieldGuard generator)",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def group_name(row: pd.Series) -> str:
    if row["fabricator"].startswith("rule"):
        return "Rule-based (FieldGuard generator)"
    return f"{row['model']} / {row['strategy']}"


def fieldguard_table(cards: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grp, paradata), g in cards.groupby(["group", "paradata"]):
        fab, hon = g[g["is_fabricator"]], g[~g["is_fabricator"]]
        tp, fp = int(fab["flagged"].sum()), int(hon["flagged"].sum())
        aucs = []
        for _, s in g.groupby("seed"):
            y, sc = s["is_fabricator"].astype(int), pd.to_numeric(s["risk_score"], errors="coerce")
            if y.nunique() == 2 and sc.notna().all():
                aucs.append(roc_auc_score(y, sc))
        lo, hi = wilson(tp, len(fab))
        rows.append(
            {
                "fabricator": grp,
                "paradata": paradata,
                "surveys": g["seed"].nunique(),
                "flagged": f"{tp}/{len(fab)}",
                "recall": tp / len(fab) if len(fab) else np.nan,
                "recall_95ci": f"{lo:.2f}-{hi:.2f}",
                "honest_flagged": f"{fp}/{len(hon)}",
                "auc_risk_mean": float(np.mean(aucs)) if aucs else np.nan,
                "auc_risk_min": float(np.min(aucs)) if aucs else np.nan,
            }
        )
    return pd.DataFrame(rows)


def checks_table(fired: pd.DataFrame, cards: pd.DataFrame, paradata: str) -> pd.DataFrame:
    """Share of fabricators and of honest enumerators with at least one warning or critical issue, per check."""
    c = cards[cards["paradata"] == paradata]
    f = fired[fired["paradata"] == paradata]
    out = {}
    for grp, g in c.groupby("group"):
        runs = set(g["run_id"])
        ff = f[f["run_id"].isin(runs)]
        key = set(zip(ff["run_id"], ff["enumerator"].astype(str), ff["check_id"], strict=True))
        fabs = g[g["is_fabricator"]][["run_id", "enumerator"]].astype(str).itertuples(index=False)
        hons = g[~g["is_fabricator"]][["run_id", "enumerator"]].astype(str).itertuples(index=False)
        fabs, hons = list(fabs), list(hons)
        for check in sorted(f["check_id"].unique()):
            out.setdefault(check, {})[f"{grp} | fabricators"] = np.mean([(r, e, check) in key for r, e in fabs])
            out[check][f"{grp} | honest"] = np.mean([(r, e, check) in key for r, e in hons])
    return pd.DataFrame(out).T.sort_index()


def signals_table(cards: pd.DataFrame, threshold: float, min_n: int) -> pd.DataFrame:
    rows = []
    for (grp, paradata), g in cards.groupby(["group", "paradata"]):
        y = g["is_fabricator"].astype(int)
        for s in SIGNALS:
            z = pd.to_numeric(g[f"z_{s}"], errors="coerce").abs()
            ok = z.notna()
            auc = roc_auc_score(y[ok], z[ok]) if y[ok].nunique() == 2 else np.nan
            hit = z.ge(threshold) & g["n"].ge(min_n)
            rows.append(
                {
                    "fabricator": grp,
                    "paradata": paradata,
                    "signal": s,
                    "auc_abs_z": auc,
                    "fabricators_flagged": f"{int(hit[y == 1].sum())}/{int((y == 1).sum())}",
                    "honest_flagged": f"{int(hit[y == 0].sum())}/{int((y == 0).sum())}",
                }
            )
    return pd.DataFrame(rows)


def combined_table(cards: pd.DataFrame, signals: list[str], threshold: float, min_n: int) -> pd.DataFrame:
    rows = []
    for (grp, paradata), g in cards.groupby(["group", "paradata"]):
        z = g[[f"z_{s}" for s in signals]].apply(pd.to_numeric, errors="coerce").abs()
        new = z.ge(threshold).any(axis=1) & g["n"].ge(min_n)
        both = new | g["flagged"].astype(bool)
        fab = g["is_fabricator"].astype(bool)
        for name, flag in (("FieldGuard", g["flagged"].astype(bool)), ("new signals", new), ("either", both)):
            k, n = int(flag[fab].sum()), int(fab.sum())
            lo, hi = wilson(k, n)
            rows.append(
                {
                    "fabricator": grp,
                    "paradata": paradata,
                    "rule": name,
                    "fabricators_flagged": f"{k}/{n}",
                    "recall": k / n if n else np.nan,
                    "recall_95ci": f"{lo:.2f}-{hi:.2f}",
                    "honest_flagged": f"{int(flag[~fab].sum())}/{int((~fab).sum())}",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--threshold", type=float, default=None, help="dev only: threshold to explore")
    args = ap.parse_args()
    d = RESULTS / args.split
    cards = pd.read_csv(d / "scorecards.csv")
    fired = pd.read_csv(d / "checks_fired.csv")
    cards["group"] = cards.apply(group_name, axis=1)
    cards["enumerator"] = cards["enumerator"].astype(str)
    fired["enumerator"] = fired["enumerator"].astype(str)
    fg = fieldguard_table(cards)
    fg.to_csv(d / "table_fieldguard.csv", index=False)
    print("FieldGuard as released\n", fg.round(2).to_string(index=False), "\n")
    for paradata in ("content_only", "with_paradata"):
        ct = checks_table(fired, cards, paradata)
        ct.to_csv(d / f"table_checks_{paradata}.csv")
        print(f"Checks firing ({paradata})\n", ct.round(2).to_string(), "\n")
    threshold = args.threshold or FROZEN["threshold"] or 3.5
    st = signals_table(cards, threshold, FROZEN["min_n"])
    st.to_csv(d / "table_signals.csv", index=False)
    print(f"Candidate signals at |z| >= {threshold}\n", st.round(2).to_string(index=False), "\n")
    if FROZEN["signals"]:
        cb = combined_table(cards, FROZEN["signals"], FROZEN["threshold"], FROZEN["min_n"])
        cb.to_csv(d / "table_combined.csv", index=False)
        print("Frozen rule\n", cb.round(2).to_string(index=False))
    (d / "frozen.json").write_text(json.dumps(FROZEN, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
