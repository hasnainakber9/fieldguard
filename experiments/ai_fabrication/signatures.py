"""Candidate enumerator-level signals aimed at model-fabricated interviews (new work, sprint of 13-15 Nov 2026).

Each signal is one number per enumerator, computed only from the answers (never paradata), and compared with the
enumerator's peers in the same survey by a robust z-score (median and MAD across enumerators). Every comparison is
two-sided: fabricated data can be too regular or too irregular. The motivation is the synthetic-respondent
literature: model output tends to compress scales, homogenise answers and drift towards the typical household.

    round_share        share of positive money answers with at most two significant digits (people round)
    wb_alpha, svc_alpha  Cronbach's alpha of the wellbeing and services grids (reversed items recoded from the pooled
                       data), i.e. how strongly the items hang together within the enumerator's interviews
    grid_sd            mean within-interview standard deviation across the 14 rating items
    food_ratio_sd      standard deviation of log(food spending / income)
    typicality         mean distance of an interview from the pooled median household (robust z units)
    spread             mean distance from each interview to its nearest neighbour among the enumerator's others

Signals are computed on the development seeds first; which of them to keep, and the threshold, are fixed there
before the held-out seeds are scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONEY = ["liv/monthly_income", "liv/business_income", "liv/livestock_value", "exp/food_exp", "exp/nonfood_exp"]
WB = [f"wb/wb_{i}" for i in range(1, 9)]
SVC = [f"svc/svc_{i}" for i in range(1, 7)]
PROFILE = [
    "resp/age",
    "hh/hh_size",
    "hh/n_children",
    "liv/monthly_income",
    "exp/food_exp",
    "exp/nonfood_exp",
    "water/minutes_to_water",
    "assets/n_phones",
    *WB,
    *SVC,
]
LOGGED = {"liv/monthly_income", "exp/food_exp", "exp/nonfood_exp", "water/minutes_to_water"}
DK = (-99.0, -98.0)
SIGNALS = ["round_share", "wb_alpha", "svc_alpha", "grid_sd", "food_ratio_sd", "typicality", "spread"]


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    return out.mask(out.isin(DK))


def _sig_digits(x: float) -> int:
    s = f"{abs(int(round(x)))}".rstrip("0")
    return max(1, len(s))


def _alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    if len(items) < 5 or k < 2:
        return float("nan")
    var_items = items.var(ddof=1).sum()
    var_total = items.sum(axis=1).var(ddof=1)
    if not np.isfinite(var_total) or var_total <= 0:
        return float("nan")
    return float(k / (k - 1) * (1 - var_items / var_total))


def _recode(grid: pd.DataFrame) -> pd.DataFrame:
    """Flip items whose pooled first-component loading is negative (the reverse-worded ones)."""
    g = grid.dropna()
    if len(g) < 10:
        return grid
    z = (g - g.mean()) / g.std(ddof=0).replace(0, np.nan)
    _, _, vt = np.linalg.svd(z.fillna(0).to_numpy(), full_matrices=False)
    load = vt[0]
    if (load > 0).sum() < (load < 0).sum():
        load = -load
    out = grid.copy()
    for col, sign in zip(grid.columns, load, strict=True):
        if sign < 0:
            out[col] = 6 - out[col]
    return out


def enumerator_signals(df: pd.DataFrame, enum_col: str = "enum_id") -> pd.DataFrame:
    money = _num(df, MONEY)
    wb = _recode(_num(df, WB))
    svc = _recode(_num(df, SVC))
    grid = pd.concat([_num(df, WB), _num(df, SVC)], axis=1)
    prof = _num(df, PROFILE)
    for c in LOGGED:
        prof[c] = np.log1p(prof[c].clip(lower=0))
    med, mad = prof.median(), (prof - prof.median()).abs().median() * 1.4826
    z = (prof - med) / mad.replace(0, np.nan)
    z = z.clip(-6, 6)
    rows = []
    for e, idx in df.groupby(enum_col).groups.items():
        m = money.loc[idx].to_numpy().ravel()
        m = m[np.isfinite(m) & (m > 0)]
        food = pd.to_numeric(df.loc[idx, "exp/food_exp"], errors="coerce")
        inc = pd.to_numeric(df.loc[idx, "liv/monthly_income"], errors="coerce")
        ok = (food > 0) & (inc > 0)
        zi = z.loc[idx]
        dist = np.sqrt((zi**2).mean(axis=1))
        zf = zi.fillna(0).to_numpy()
        if len(zf) > 1:
            d = np.sqrt(((zf[:, None, :] - zf[None, :, :]) ** 2).mean(axis=2))
            np.fill_diagonal(d, np.inf)
            spread = float(np.median(d.min(axis=1)))
        else:
            spread = float("nan")
        rows.append(
            {
                "enumerator": e,
                "n": len(idx),
                "round_share": float(np.mean([_sig_digits(v) <= 2 for v in m])) if len(m) else float("nan"),
                "wb_alpha": _alpha(wb.loc[idx]),
                "svc_alpha": _alpha(svc.loc[idx]),
                "grid_sd": float(grid.loc[idx].std(axis=1, ddof=0).mean()),
                "food_ratio_sd": float(np.log(food[ok] / inc[ok]).std(ddof=1)) if ok.sum() > 2 else float("nan"),
                "typicality": float(dist.mean()),
                "spread": spread,
            }
        )
    return pd.DataFrame(rows)


def robust_z(sig: pd.DataFrame, cols: list[str] = SIGNALS) -> pd.DataFrame:
    out = sig[["enumerator", "n"]].copy()
    for c in cols:
        x = sig[c]
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        out[c] = (x - med) / mad if mad and np.isfinite(mad) else np.nan
    return out


def flags(z: pd.DataFrame, cols: list[str], threshold: float, min_n: int = 8) -> pd.Series:
    """True for enumerators with at least min_n interviews and any |z| >= threshold among cols."""
    big = z[cols].abs().ge(threshold).any(axis=1)
    return big & z["n"].ge(min_n)
