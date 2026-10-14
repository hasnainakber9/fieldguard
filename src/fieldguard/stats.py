"""Statistical helpers (robust z-scores, Benford, ICC, haversine)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st

EARTH_RADIUS_M = 6_371_008.8

#: Expected first-digit proportions under Benford's law, digits 1..9.
BENFORD_FIRST = np.log10(1 + 1 / np.arange(1, 10))

#: Nigrini (2012) MAD conformity thresholds for the first-digit test.
NIGRINI_FIRST_DIGIT = ((0.006, "close conformity"), (0.012, "acceptable conformity"), (0.015, "marginally acceptable"))


def mad(x: np.ndarray | pd.Series) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.median(np.abs(a - np.median(a))))


def robust_z(x: pd.Series) -> pd.Series:
    """Modified z-score 0.6745 (x - median) / MAD (Iglewicz & Hoaglin 1993).

    Falls back to the mean absolute deviation (x 1.2533) when MAD is zero.
    """
    v = x.astype("float64")
    med = v.median()
    m = mad(v)
    if np.isfinite(m) and m > 0:
        return 0.6745 * (v - med) / m
    meanad = float((v - v.mean()).abs().mean())
    if not np.isfinite(meanad) or meanad == 0:
        return pd.Series(0.0, index=v.index).where(v.notna())
    return (v - med) / (1.2533 * meanad)


def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def first_digits(x: pd.Series | np.ndarray, min_value: float = 10.0) -> np.ndarray:
    a = np.abs(np.asarray(x, dtype=float))
    a = a[np.isfinite(a) & (a >= max(min_value, 1e-12))]
    if a.size == 0:
        return np.array([], dtype=int)
    exp = np.floor(np.log10(a))
    d = np.floor(a / 10**exp).astype(int)
    d = np.clip(d, 1, 9)
    return d


def benford_mad(digits: np.ndarray, expected: np.ndarray | None = None) -> float:
    if digits.size == 0:
        return float("nan")
    exp = BENFORD_FIRST if expected is None else expected
    observed = np.bincount(digits, minlength=10)[1:10] / digits.size
    return float(np.mean(np.abs(observed - exp)))


def benford_chi2(digits: np.ndarray, expected: np.ndarray | None = None) -> tuple[float, float]:
    if digits.size == 0:
        return float("nan"), float("nan")
    exp = BENFORD_FIRST if expected is None else expected
    observed = np.bincount(digits, minlength=10)[1:10]
    expected_counts = exp / exp.sum() * digits.size
    chi2, pval = st.chisquare(observed, expected_counts)
    return float(chi2), float(pval)


def nigrini_class(value: float) -> str:
    if not np.isfinite(value):
        return "insufficient data"
    for limit, label in NIGRINI_FIRST_DIGIT:
        if value <= limit:
            return label
    return "nonconformity"


def icc1(values: pd.Series, groups: pd.Series) -> tuple[float, float, int]:
    """One-way random-effects ICC(1) via ANOVA with unequal group sizes; returns (icc, p_value, k)."""
    df = pd.DataFrame({"y": values.astype("float64"), "g": groups}).dropna()
    counts = df.groupby("g")["y"].count()
    counts = counts[counts >= 2]
    df = df[df["g"].isin(counts.index)]
    k = len(counts)
    n = len(df)
    if k < 2 or n <= k:
        return float("nan"), float("nan"), k
    grand = df["y"].mean()
    means = df.groupby("g")["y"].mean()
    ssb = float((counts * (means - grand) ** 2).sum())
    ssw = float(((df["y"] - df["g"].map(means)) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - float((counts**2).sum()) / n) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        return float("nan"), float("nan"), k
    icc = (msb - msw) / denom
    pval = float(st.f.sf(msb / msw, k - 1, n - k)) if msw > 0 else 0.0
    return float(icc), pval, k


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def two_proportion_z(x1: float, n1: float, x2: float, n2: float) -> float:
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0
    return float((x1 / n1 - x2 / n2) / se)
