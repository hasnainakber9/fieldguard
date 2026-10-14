"""Respondent- and enumerator-level pattern checks: straight-lining and enumerator effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats as st

from fieldguard.checks.context import CheckContext
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.models import Level, Severity, empty_issues
from fieldguard.stats import icc1, two_proportion_z


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)


# ----------------------------------------------------------------------------- straight-lining
class StraightliningParams(CheckParams):
    sd_threshold: float = Field(0.5, ge=0, description="Within-respondent SD below this = low variance")
    enumerator_share_threshold: float = Field(0.25, ge=0, le=1)
    z_threshold: float = Field(3.0, gt=0)
    min_n: int = Field(5, ge=1)


def _grid_matrix(ctx: CheckContext, cols: list[str]) -> pd.DataFrame:
    nums = pd.DataFrame({c: ctx.num(c) for c in cols})
    numeric_ok = all(nums[c].notna().sum() >= 0.8 * (ctx.data[c].notna() & ~ctx.special_mask(c)).sum() for c in cols)
    if numeric_ok:
        return nums
    # text Likert labels: code consistently across items
    texts = pd.DataFrame({c: ctx.text(c).mask(ctx.special_mask(c)) for c in cols})
    levels = sorted(pd.unique(texts.stack().dropna()))
    mapping = {v: i for i, v in enumerate(levels)}
    return texts.apply(lambda s: s.map(mapping)).astype("float64")


@register_check(
    "straightlining",
    title="Straight-lining and low variance in grids",
    description="Respondents giving identical (or near-identical) answers across a Likert/grid block, and enumerators with many such interviews.",
    category="fabrication",
    params=StraightliningParams,
    references=("Kemper & Menold (2014)", "Blasius & Thiessen (2015)"),
)
def straightlining(ctx: CheckContext, p: StraightliningParams) -> pd.DataFrame:
    specs = ctx.config.variables.grid_specs()
    if not specs or ctx.n == 0:
        return empty_issues(with_row=True)
    frames = []
    straight_counts = pd.Series(0.0, index=ctx.data.index)
    eligible_counts = pd.Series(0.0, index=ctx.data.index)
    sds = []
    for name, spec in specs.items():
        cols, missing = ctx.cols(spec.columns)
        frames.append(ctx.missing_columns_issue("straightlining", missing))
        if len(cols) < spec.min_items:
            continue
        m = _grid_matrix(ctx, cols)
        n_items = m.notna().sum(axis=1)
        eligible = n_items >= spec.min_items
        straight = eligible & (m.nunique(axis=1, dropna=True) == 1)
        sd = m.std(axis=1, ddof=0).where(eligible)
        low = eligible & ~straight & (sd < p.sd_threshold)
        straight_counts += straight.astype(float)
        eligible_counts += eligible.astype(float)
        sds.append(sd.rename(name))
        first_val = m.bfill(axis=1).iloc[:, 0]
        frames.append(
            ctx.record_issues(
                "straightlining",
                straight,
                Severity.INFO,
                f"Same answer to all {len(cols)} items of grid '{name}'",
                variable=name,
                value=first_val,
                metric=sd,
            )
        )
        frames.append(
            ctx.record_issues(
                "straightlining",
                low,
                Severity.INFO,
                f"Very low variation across grid '{name}' (SD < {p.sd_threshold:g})",
                variable=name,
                metric=sd,
            )
        )
    if eligible_counts.sum() == 0:
        return _concat(frames)
    g = pd.DataFrame({"s": straight_counts, "n": eligible_counts, "e": ctx.enumerator}).groupby("e")[["s", "n"]].sum()
    share = g["s"] / g["n"].replace(0, np.nan)
    ctx.add_metric_series("straightlining", "straightline_share", share)
    if sds:
        sd_all = pd.concat(sds, axis=1).mean(axis=1)
        overall = float(sd_all.mean())
        if overall > 0:
            ctx.add_metric_series("straightlining", "grid_sd_ratio", sd_all.groupby(ctx.enumerator).mean() / overall)
    rows = []
    if ctx.has_enumerators:
        tot_s, tot_n = g["s"].sum(), g["n"].sum()
        for enum, r in g.iterrows():
            if r["n"] < p.min_n:
                continue
            z = two_proportion_z(r["s"], r["n"], tot_s - r["s"], tot_n - r["n"])
            if share[enum] >= p.enumerator_share_threshold and z > p.z_threshold:
                rows.append(
                    ctx.issue(
                        "straightlining",
                        Severity.WARNING,
                        f"{share[enum]:.0%} of grid answers straight-lined vs {(tot_s - r['s']) / max(tot_n - r['n'], 1):.0%} for others (z={z:.1f})",
                        enumerator=str(enum),
                        metric=float(share[enum]),
                    )
                )
    frames.append(ctx.frame(rows))
    return _concat(frames)


# ----------------------------------------------------------------------------- enumerator effects
class EnumeratorEffectParams(CheckParams):
    variables: list[str] | None = Field(None, description="Default: variables.key, else variables.numeric")
    icc_threshold: float = Field(0.10, ge=0, le=1)
    p_threshold: float = Field(0.01, gt=0, lt=1)
    mean_z_threshold: float = Field(4.0, gt=0)
    variance_ratio_threshold: float = Field(0.5, gt=0, lt=1)
    min_n: int = Field(8, ge=3)
    adjust_for_cluster: str = Field("auto", pattern="^(auto|yes|no)$")
    cramers_v_threshold: float = Field(0.3, gt=0, le=1)


def _cramers_v(x: pd.Series, g: pd.Series) -> float:
    tab = pd.crosstab(x, g)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return float("nan")
    chi2 = st.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1))))


@register_check(
    "enumerator_effects",
    title="Enumerator effects (ICC, mean and variance deviations)",
    description="Share of variance explained by enumerators (ICC), enumerators with atypical means, and enumerators with implausibly low answer variance.",
    category="enumerator",
    params=EnumeratorEffectParams,
    references=("West & Blom (2017)", "Schäfer et al. (2005)", "Bredl et al. (2012)"),
)
def enumerator_effects(ctx: CheckContext, p: EnumeratorEffectParams) -> pd.DataFrame:
    names = p.variables or ctx.config.variables.key or ctx.config.variables.numeric
    cols, missing = ctx.cols(names)
    frames = [ctx.missing_columns_issue("enumerator_effects", missing)]
    if not cols or ctx.n == 0 or not ctx.has_enumerators:
        return _concat(frames)
    groups = ctx.enumerator
    adjust = False
    if ctx.cluster is not None and p.adjust_for_cluster != "no":
        per_cluster = pd.DataFrame({"c": ctx.cluster, "e": groups}).dropna().groupby("c")["e"].nunique()
        adjust = p.adjust_for_cluster == "yes" or (len(per_cluster) > 0 and per_cluster.median() >= 2)
    rows = []
    zs: dict[str, list[float]] = {}
    vratios: dict[str, list[float]] = {}
    for col in cols:
        raw = ctx.data[col]
        y = ctx.num(col)
        nonnull = (raw.notna() & ~ctx.special_mask(col)).sum()
        if nonnull == 0:
            continue
        if y.notna().sum() < 0.9 * nonnull:
            v = _cramers_v(ctx.text(col).mask(ctx.special_mask(col)), groups)
            if np.isfinite(v) and v >= p.cramers_v_threshold:
                rows.append(
                    ctx.issue(
                        "enumerator_effects",
                        Severity.INFO,
                        f"Answers to {col} strongly associated with enumerator (Cramér's V={v:.2f})",
                        level=Level.VARIABLE,
                        variable=col,
                        metric=v,
                    )
                )
            continue
        if adjust:
            y = y - y.groupby(ctx.cluster).transform("mean")
        icc, pval, k = icc1(y, groups)
        if np.isfinite(icc) and icc >= p.icc_threshold and pval < p.p_threshold:
            rows.append(
                ctx.issue(
                    "enumerator_effects",
                    Severity.WARNING,
                    f"Enumerators explain {icc:.0%} of the variance in {col} (ICC={icc:.2f}, k={k}, p={pval:.2g}"
                    + (", cluster-adjusted)" if adjust else ")"),
                    level=Level.VARIABLE,
                    variable=col,
                    metric=icc,
                )
            )
        frame = pd.DataFrame({"y": y, "e": groups}).dropna()
        stats_ = frame.groupby("e")["y"].agg(["mean", "std", "count", "sum"])
        sq = frame.assign(y2=frame["y"] ** 2).groupby("e")["y2"].sum()
        tot_n, tot_s, tot_sq = stats_["count"].sum(), stats_["sum"].sum(), sq.sum()
        for enum, r in stats_.iterrows():
            n_o = tot_n - r["count"]
            if r["count"] < p.min_n or n_o < p.min_n:
                continue
            m_o = (tot_s - r["sum"]) / n_o
            var_o = max((tot_sq - sq[enum]) / n_o - m_o**2, 0.0) * n_o / max(n_o - 1, 1)
            sd_o = np.sqrt(var_o)
            if sd_o <= 0:
                continue
            z = (r["mean"] - m_o) / (sd_o * np.sqrt(1 / r["count"] + 1 / n_o))
            zs.setdefault(str(enum), []).append(abs(float(z)))
            vratios.setdefault(str(enum), []).append(float(r["std"] / sd_o) if pd.notna(r["std"]) else np.nan)
            if abs(z) >= p.mean_z_threshold:
                rows.append(
                    ctx.issue(
                        "enumerator_effects",
                        Severity.INFO,
                        f"Mean {col} = {r['mean']:.3g} vs {m_o:.3g} for other enumerators (z={z:.1f})",
                        enumerator=str(enum),
                        variable=col,
                        value=f"{r['mean']:.6g}",
                        metric=float(z),
                    )
                )
    for enum, vals in zs.items():
        ctx.add_metric("enumerator_effects", enum, "mean_deviation", float(np.mean(vals)))
    for enum, vals in vratios.items():
        arr = np.array([v for v in vals if np.isfinite(v)])
        if arr.size == 0:
            continue
        med = float(np.median(arr))
        ctx.add_metric("enumerator_effects", enum, "variance_ratio", med)
        if med < p.variance_ratio_threshold and arr.size >= 2:
            rows.append(
                ctx.issue(
                    "enumerator_effects",
                    Severity.WARNING,
                    f"Answers vary much less than other enumerators' (median SD ratio {med:.2f} across {arr.size} variables)",
                    enumerator=enum,
                    metric=med,
                )
            )
    frames.append(ctx.frame(rows))
    return _concat(frames)
