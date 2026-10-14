"""Statistical checks: outliers, DK/refusal rates, digit preference (heaping), Benford's law."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats as st

from fieldguard.checks.context import CheckContext, normalize_series
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.models import Level, Severity, empty_issues
from fieldguard.stats import (
    BENFORD_FIRST,
    benford_chi2,
    benford_mad,
    first_digits,
    iqr_bounds,
    nigrini_class,
    robust_z,
    two_proportion_z,
)


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)


def _numeric_targets(ctx: CheckContext, names: list[str] | None, check_id: str) -> tuple[list[str], pd.DataFrame]:
    names = names if names is not None else ctx.config.variables.numeric
    found, missing = ctx.cols(names)
    return found, ctx.missing_columns_issue(check_id, missing)


# ----------------------------------------------------------------------------- outliers
class OutlierParams(CheckParams):
    variables: list[str] | None = Field(None, description="Default: variables.numeric")
    method: str = Field("mad", pattern="^(mad|iqr)$")
    threshold: float = Field(3.5, gt=0, description="|modified z| for MAD; ignored for IQR")
    iqr_k: float = Field(3.0, gt=0)
    log_transform: list[str] = Field(default_factory=list, description="Skewed variables analysed on log(1+x)")
    within_enumerator: bool = True
    min_n: int = Field(10, ge=3)
    severity: Severity = Severity.WARNING


def _outlier_mask(x: pd.Series, p: OutlierParams) -> tuple[pd.Series, pd.Series]:
    if x.notna().sum() < p.min_n:
        return pd.Series(False, index=x.index), pd.Series(np.nan, index=x.index)
    if p.method == "mad":
        z = robust_z(x)
        return (z.abs() > p.threshold).fillna(False), z
    lo, hi = iqr_bounds(x.dropna(), p.iqr_k)
    return ((x < lo) | (x > hi)).fillna(False), pd.Series(np.nan, index=x.index)


def _grouped_outlier_mask(x: pd.Series, groups: pd.Series, p: OutlierParams) -> tuple[pd.Series, pd.Series]:
    """Vectorised per-group version of :func:`_outlier_mask` (groups with fewer than ``min_n`` values are not judged)."""
    valid = x.notna()
    counts = valid.groupby(groups).transform("sum")
    enough = valid & (counts >= p.min_n)
    xv = x.where(enough)
    if p.method == "mad":
        med = xv.groupby(groups).transform("median")
        mad = (xv - med).abs().groupby(groups).transform("median")
        meanad = (xv - xv.groupby(groups).transform("mean")).abs().groupby(groups).transform("mean")
        with np.errstate(divide="ignore", invalid="ignore"):
            z_mad = 0.6745 * (xv - med) / mad
            z_fallback = (xv - med) / (1.2533 * meanad)
        z = z_mad.where(mad > 0, z_fallback)
        z = z.where((mad > 0) | (meanad > 0), 0.0).where(enough)
        return ((z.abs() > p.threshold).fillna(False) & enough), z
    q1 = xv.groupby(groups).quantile(0.25)
    q3 = xv.groupby(groups).quantile(0.75)
    lo = groups.map(q1 - p.iqr_k * (q3 - q1)).astype("float64")
    hi = groups.map(q3 + p.iqr_k * (q3 - q1)).astype("float64")
    mask = ((xv < lo) | (xv > hi)).fillna(False) & enough
    return mask, pd.Series(np.nan, index=x.index)


@register_check(
    "outliers",
    title="Univariate outliers",
    description="Modified z-score (MAD) or IQR fences, overall and within each enumerator's own interviews.",
    category="consistency",
    params=OutlierParams,
    references=("Iglewicz & Hoaglin (1993)", "IPA ipacheck: ipacheckoutliers"),
)
def outliers(ctx: CheckContext, p: OutlierParams) -> pd.DataFrame:
    found, frames0 = _numeric_targets(ctx, p.variables, "outliers")
    frames = [frames0]
    log_cols, _ = ctx.cols(p.log_transform)
    any_out = pd.Series(False, index=ctx.data.index)
    for col in found:
        x = ctx.num(col)
        xt = np.log1p(x.clip(lower=0)) if col in log_cols else x
        mask, z = _outlier_mask(xt, p)
        any_out |= mask
        label = "log-scale " if col in log_cols else ""
        msg = f"Outlier ({label}{p.method.upper()})" + (
            " |z|>" + f"{p.threshold:g}" if p.method == "mad" else f" beyond {p.iqr_k:g}xIQR"
        )
        frames.append(ctx.record_issues("outliers", mask, p.severity, msg, variable=col, value=ctx.data[col], metric=z))
        if p.within_enumerator and ctx.has_enumerators:
            within, wz = _grouped_outlier_mask(xt, ctx.enumerator, p)
            within = within & ~mask
            frames.append(
                ctx.record_issues(
                    "outliers",
                    within,
                    Severity.INFO,
                    "Unusual for this enumerator's own interviews",
                    variable=col,
                    value=ctx.data[col],
                    metric=wz,
                )
            )
    if ctx.n and found:
        ctx.add_metric_series("outliers", "outlier_rate", any_out.groupby(ctx.enumerator).mean())
    return _concat(frames)


# ----------------------------------------------------------------------------- nonresponse
class NonresponseParams(CheckParams):
    variables: list[str] | None = Field(None, description="Default: variables.nonresponse or all answer columns")
    z_threshold: float = Field(3.0, gt=0, description="Two-proportion z vs other enumerators")
    min_ratio: float = Field(2.0, ge=1, description="Also require rate >= min_ratio x others' rate (or <= 1/min_ratio)")
    min_answers: int = Field(50, ge=1)
    flag_low: bool = Field(True, description="Flag enumerators with suspiciously few DK/refusals")
    variable_rate_threshold: float = Field(0.2, ge=0, le=1)


@register_check(
    "nonresponse",
    title="Don't know / refused rates by enumerator",
    description="Enumerators whose DK or refusal rate differs strongly from their peers (too high or suspiciously low).",
    category="enumerator",
    params=NonresponseParams,
    references=("IPA ipacheck: ipacheckspecify/ipachecknomiss", "J-PAL data quality checks"),
)
def nonresponse(ctx: CheckContext, p: NonresponseParams) -> pd.DataFrame:
    names = p.variables or ctx.config.variables.nonresponse
    if names:
        cols, missing = ctx.cols(names)
        frames = [ctx.missing_columns_issue("nonresponse", missing)]
    else:
        cols, frames = ctx.answer_columns(), []
    if not cols or ctx.n == 0:
        return _concat(frames)
    dk_codes, ref_codes = ctx.special_codes("dont_know"), ctx.special_codes("refused")
    dk = pd.DataFrame(index=ctx.data.index)
    ref = pd.DataFrame(index=ctx.data.index)
    answered = ctx.data[cols].notna()
    for c in cols:
        norm = normalize_series(ctx.data[c])
        dk[c] = norm.isin(dk_codes)
        ref[c] = norm.isin(ref_codes)
    per_row = pd.DataFrame(
        {"dk": dk.sum(axis=1), "ref": ref.sum(axis=1), "n": answered.sum(axis=1), "enum": ctx.enumerator}
    )
    g = per_row.groupby("enum")[["dk", "ref", "n"]].sum()
    tot = g.sum()
    rows = []
    rates = (g["dk"] + g["ref"]) / g["n"].replace(0, np.nan)
    ctx.add_metric_series("nonresponse", "dk_rate", g["dk"] / g["n"].replace(0, np.nan))
    ctx.add_metric_series("nonresponse", "refusal_rate", g["ref"] / g["n"].replace(0, np.nan))
    if len(g) >= 3:
        rz = robust_z(rates)
        ctx.add_metric_series("nonresponse", "nonresponse_abs_z", rz.abs())
    for enum, r in g.iterrows():
        if r["n"] < p.min_answers or not ctx.has_enumerators:
            continue
        for kind, label in (("dk", "don't know"), ("ref", "refusal")):
            x1, n1 = r[kind], r["n"]
            x2, n2 = tot[kind] - x1, tot["n"] - n1
            if n2 <= 0:
                continue
            z = two_proportion_z(x1, n1, x2, n2)
            rate, other = x1 / n1, x2 / n2
            if z > p.z_threshold and rate >= p.min_ratio * other:
                rows.append(
                    ctx.issue(
                        "nonresponse",
                        Severity.WARNING,
                        f"High {label} rate {rate:.1%} vs {other:.1%} for other enumerators (z={z:.1f})",
                        enumerator=str(enum),
                        variable=label,
                        value=f"{rate:.4f}",
                        metric=z,
                    )
                )
            elif p.flag_low and z < -p.z_threshold and rate * p.min_ratio <= other:
                rows.append(
                    ctx.issue(
                        "nonresponse",
                        Severity.INFO,
                        f"Suspiciously low {label} rate {rate:.1%} vs {other:.1%} for other enumerators (z={z:.1f})",
                        enumerator=str(enum),
                        variable=label,
                        value=f"{rate:.4f}",
                        metric=z,
                    )
                )
    var_rates = (dk.sum() + ref.sum()) / answered.sum().replace(0, np.nan)
    for c, rate in var_rates.items():
        if pd.notna(rate) and rate >= p.variable_rate_threshold and answered[c].sum() >= 10:
            rows.append(
                ctx.issue(
                    "nonresponse",
                    Severity.INFO,
                    f"{rate:.0%} of answers to {c} are don't know/refused - check question wording/translation",
                    level=Level.VARIABLE,
                    variable=str(c),
                    metric=float(rate),
                )
            )
    frames.append(ctx.frame(rows))
    return _concat(frames)


# ----------------------------------------------------------------------------- heaping
class HeapingParams(CheckParams):
    variables: list[str] | None = Field(None, description="Default: variables.numeric")
    reference: str = Field("pooled", pattern="^(pooled|uniform)$")
    min_n: int = Field(20, ge=5)
    ratio_threshold: float = Field(1.5, gt=1)
    p_threshold: float = Field(0.01, gt=0, lt=1)


def heaped(x: pd.Series) -> pd.Series:
    """True when a value (>=10) is a 'round' number: its 2nd significant digit is 0/5 and the rest zeros.

    For 10-99 this is the classical terminal-digit 0/5 test (Whipple-style); for 2500 or 3000 it captures
    rounding of amounts. Expected share under no preference ~ 0.2.
    """
    v = np.round(np.abs(x.astype("float64")))
    ok = v >= 10
    nd = np.floor(np.log10(np.where(ok, v, 10))) + 1
    unit = 5 * 10 ** np.maximum(nd - 2, 0)
    res = pd.Series(np.where(ok, (v % unit) == 0, np.nan), index=x.index)
    return res.where(ok & x.notna())


@register_check(
    "heaping",
    title="Digit preference / heaping",
    description="Share of 'round' values (terminal 0/5; round leading pairs like 2500) by enumerator vs peers or uniform.",
    category="fabrication",
    params=HeapingParams,
    references=("Shryock & Siegel: Whipple's index", "Bredl et al. (2012)"),
)
def heaping(ctx: CheckContext, p: HeapingParams) -> pd.DataFrame:
    found, frame0 = _numeric_targets(ctx, p.variables, "heaping")
    frames = [frame0]
    if not found or ctx.n == 0:
        return _concat(frames)
    rows = []
    obs_tot: dict[str, float] = {}
    exp_tot: dict[str, float] = {}
    var_tot: dict[str, float] = {}
    for col in found:
        h = heaped(ctx.num(col))
        valid = h.notna()
        if valid.sum() < p.min_n:
            continue
        overall = float(h[valid].mean())
        if overall > 0.2 * p.ratio_threshold * 1.5 and valid.sum() >= 50:
            rows.append(
                ctx.issue(
                    "heaping",
                    Severity.INFO,
                    f"{overall:.0%} of {col} values are round numbers (expected ~20% without digit preference)",
                    level=Level.VARIABLE,
                    variable=col,
                    metric=overall / 0.2,
                )
            )
        df = pd.DataFrame({"h": h[valid], "e": ctx.enumerator[valid]})
        g = df.groupby("e")["h"].agg(["sum", "count"])
        for enum, r in g.iterrows():
            others_n = g["count"].sum() - r["count"]
            ref = 0.2 if p.reference == "uniform" or others_n < p.min_n else (g["sum"].sum() - r["sum"]) / others_n
            ref = float(np.clip(ref, 0.01, 0.99))
            obs_tot[enum] = obs_tot.get(enum, 0.0) + r["sum"]
            exp_tot[enum] = exp_tot.get(enum, 0.0) + r["count"] * ref
            var_tot[enum] = var_tot.get(enum, 0.0) + r["count"] * ref * (1 - ref)
            if r["count"] < p.min_n:
                continue
            share = r["sum"] / r["count"]
            ratio = share / ref
            pval = float(st.binomtest(int(r["sum"]), int(r["count"]), ref, alternative="greater").pvalue)
            if ratio >= p.ratio_threshold and pval < p.p_threshold:
                rows.append(
                    ctx.issue(
                        "heaping",
                        Severity.INFO,
                        f"{share:.0%} round values in {col} vs {ref:.0%} expected (ratio {ratio:.2f}, p={pval:.3g})",
                        enumerator=str(enum),
                        variable=col,
                        value=f"{share:.4f}",
                        metric=ratio,
                    )
                )
    for enum, obs in obs_tot.items():
        exp = exp_tot[enum]
        if exp <= 0:
            continue
        ratio = obs / exp
        ctx.add_metric("heaping", enum, "heaping_ratio", ratio)
        z = (obs - exp) / np.sqrt(var_tot[enum]) if var_tot[enum] > 0 else 0.0
        if ratio >= p.ratio_threshold and z > st.norm.isf(p.p_threshold) and ctx.has_enumerators:
            rows.append(
                ctx.issue(
                    "heaping",
                    Severity.WARNING,
                    f"Digit preference across numeric answers: {ratio:.2f}x the expected share of round values (z={z:.1f})",
                    enumerator=str(enum),
                    variable="(all numeric)",
                    metric=ratio,
                )
            )
    frames.append(ctx.frame(rows))
    return _concat(frames)


# ----------------------------------------------------------------------------- Benford
class BenfordParams(CheckParams):
    variables: list[str] | None = Field(None, description="Amount-type variables spanning orders of magnitude")
    min_n: int = Field(50, ge=10, description="Minimum leading digits per enumerator")
    min_value: float = Field(10.0, ge=0)
    reference: str = Field("benford", pattern="^(benford|pooled)$")
    excess_threshold: float = Field(
        1.5, gt=1, description="Flag when MAD > threshold x MAD expected by chance at this n"
    )
    p_threshold: float = Field(0.01, gt=0, lt=1)


def expected_mad(n: int, expected: np.ndarray = BENFORD_FIRST) -> float:
    """Approximate MAD expected from sampling noise alone for n digits (normal approximation)."""
    if n <= 0:
        return float("nan")
    return float(np.mean(np.sqrt(2 * expected * (1 - expected) / (np.pi * n))))


@register_check(
    "benford",
    title="Benford's law (first digit) by enumerator",
    description="First-digit distribution per enumerator; MAD with Nigrini thresholds, adjusted for sample size.",
    category="fabrication",
    params=BenfordParams,
    references=("Nigrini (2012)", "Judge & Schechter (2009)", "Schräpler (2011)"),
)
def benford(ctx: CheckContext, p: BenfordParams) -> pd.DataFrame:
    found, frame0 = _numeric_targets(ctx, p.variables, "benford")
    frames = [frame0]
    if not found or ctx.n == 0:
        return _concat(frames)
    rows = []
    digit_frames = []
    for col in found:
        x = ctx.num(col)
        mask = x.abs() >= max(p.min_value, 1e-12)
        d = first_digits(x[mask], p.min_value)
        digit_frames.append(pd.DataFrame({"d": d, "e": ctx.enumerator[mask].to_numpy()}))
        if d.size >= 300:
            m = benford_mad(d)
            cls = nigrini_class(m)
            rows.append(
                ctx.issue(
                    "benford",
                    Severity.INFO,
                    f"{col}: first-digit MAD {m:.4f} ({cls}, n={d.size}) for the whole dataset",
                    level=Level.VARIABLE,
                    variable=col,
                    metric=m,
                )
            )
    digits = pd.concat(digit_frames, ignore_index=True)
    if digits.empty:
        return _concat(frames)
    all_counts = np.bincount(digits["d"], minlength=10)[1:10].astype(float)
    for enum, sub in digits.groupby("e"):
        d = sub["d"].to_numpy()
        n = d.size
        if n < p.min_n:
            continue
        if p.reference == "pooled":
            others = all_counts - np.bincount(d, minlength=10)[1:10]
            expected = (others + 0.5) / (others + 0.5).sum()
        else:
            expected = BENFORD_FIRST
        m = benford_mad(d, expected)
        exp_m = expected_mad(n, expected)
        excess = m / exp_m if exp_m > 0 else float("nan")
        _, pval = benford_chi2(d, expected)
        ctx.add_metric("benford", str(enum), "benford_mad", m)
        ctx.add_metric("benford", str(enum), "benford_excess", excess)
        if not ctx.has_enumerators:
            continue
        if excess >= p.excess_threshold and pval < p.p_threshold:
            sev = Severity.WARNING if m > 0.015 else Severity.INFO
            rows.append(
                ctx.issue(
                    "benford",
                    sev,
                    f"First digits deviate from {p.reference}: MAD {m:.4f} ({nigrini_class(m)}), "
                    f"{excess:.1f}x the MAD expected by chance at n={n}, chi2 p={pval:.2g}",
                    enumerator=str(enum),
                    variable=", ".join(found),
                    value=str(n),
                    metric=m,
                )
            )
    frames.append(ctx.frame(rows))
    return _concat(frames)
