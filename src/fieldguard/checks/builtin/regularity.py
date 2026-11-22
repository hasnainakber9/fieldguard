"""Answer regularity: enumerators whose interviews are far more regular (or more varied) than their peers'.

Three enumerator-level signals, computed from the answers only (never paradata) and each compared with the other
enumerators in the same survey by a robust z-score, ``(x - median) / (1.4826 * MAD)`` across enumerators:

``grid_sd``
    Mean within-interview standard deviation of the rating items (the numeric items of every grid in
    ``variables.grids``, pooled). Low when every rating in an interview is about the same.
``spread``
    Median distance from each of the enumerator's interviews to the most similar other interview by the same
    enumerator, in the household profile below. Low when the enumerator's households look alike.
``typicality``
    Mean distance of the enumerator's interviews from the survey's median household. Low when every household is
    close to "typical".

The household profile is a set of numeric answers, each standardised against all interviews by median and MAD
(after ``log(1 + x)`` for skewed amounts) and clipped at +-6; distances are root-mean-square over the profile. A
missing answer counts as the median in ``spread`` and is left out of ``typicality``.

The signals, the threshold ``|z| >= 3.5`` and the minimum of 8 interviews were fixed on development surveys of a
synthetic study of model-written (AI) fabrication before held-out surveys were scored once (branch
``sprint/ai-fabrication``, ``experiments/ai_fabrication/``, frozen in commit b0ea6d1). On the held-out surveys,
with honest-looking paradata, they flagged 25 of 25 AI fabricators, every one in the too-regular direction, and 7
of 9 rule-based fabricators, and 2 of 90 honest enumerators on surveys with no fabrication. The honest interviews
in that study were simulated, so these rates are not evidence about real surveys.

Both directions are reported, as in the study, but they are not equal. A too-regular result (negative z) is a
warning and is what the enumerator risk score counts (metric ``regularity_excess``). A too-varied result
(positive z) is information only: it found no fabricator in the study's development surveys or in fresh generator
surveys, and it is what honest enumerators with an unusual or mixed area or few interviews produce (``spread``
grows as an enumerator's number of interviews falls). Honest reasons for a too-regular result include a very uniform
assignment area and a much larger workload than peers. A flag is a lead for a back-check, not proof of fabrication.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.spatial.distance import cdist

from fieldguard.checks.context import MISSING_ENUMERATOR, CheckContext
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.models import Severity, empty_issues

CHECK_ID = "answer_regularity"
Signal = Literal["grid_sd", "spread", "typicality"]
SIGNALS: tuple[str, ...] = ("grid_sd", "spread", "typicality")

#: Consistency constant that makes the MAD estimate the standard deviation of a normal distribution.
MAD_SCALE = 1.4826
#: Standardised profile values are clipped at this many robust z units, so one extreme answer cannot dominate.
PROFILE_CLIP = 6.0
#: Upper bound on the cells of one block of pairwise distances (keeps memory flat for very large workloads).
_BLOCK_CELLS = 2_000_000

#: Plain-language reading of each signal in each direction, for supervisors.
_EXPLAIN: dict[str, dict[str, str]] = {
    "grid_sd": {
        "low": "Ratings within this enumerator's interviews vary much less than their peers'",
        "high": "Ratings within this enumerator's interviews vary much more than their peers'",
    },
    "spread": {
        "low": "This enumerator's households are much more alike than their peers' households",
        "high": "This enumerator's households differ from one another much more than their peers' households",
    },
    "typicality": {
        "low": "This enumerator's households are much closer to the survey's typical household than their peers'",
        "high": "This enumerator's households are much further from the survey's typical household than their peers'",
    },
}
_MEASURE: dict[str, str] = {
    "grid_sd": "average spread of the ratings within an interview",
    "spread": "median distance to the most similar other household in their interviews",
    "typicality": "average distance from the typical household",
}
_HONEST_HIGH = "Often honest, for example an unusual or mixed area, or fewer interviews than peers."
_LEAD = "A lead for a back-check, not proof of fabrication."


class AnswerRegularityParams(CheckParams):
    signals: list[Signal] = Field(
        default_factory=lambda: list(SIGNALS), min_length=1, description="Signals to compute (default: all three)"
    )
    threshold: float = Field(3.5, gt=0, description="Report an enumerator whose robust z against peers reaches this")
    min_n: int = Field(8, ge=2, description="Enumerators with fewer interviews are neither compared nor flagged")
    min_enumerators: int = Field(
        15, ge=3, description="Enumerators with at least min_n interviews needed before peers are compared"
    )
    variables: list[str] | None = Field(
        None,
        description="Household profile for spread and typicality. Default: variables.numeric without questions "
        "that skip logic or a conditional `required` makes depend on other answers, plus every grid item.",
    )
    log_transform: list[str] | None = Field(
        None, description="Profile variables compared on log(1 + x). Default: checks.outliers.log_transform."
    )


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)


def _note(ctx: CheckContext, message: str) -> pd.DataFrame:
    return ctx.dataset_issue(CHECK_ID, Severity.INFO, message)


def _is_numeric(ctx: CheckContext, col: str) -> bool:
    """True when at least 80% of the answers (special codes aside) are numbers; an empty column counts as numeric."""
    answered = int((ctx.data[col].notna() & ~ctx.special_mask(col)).sum())
    return answered == 0 or ctx.num(col).notna().sum() >= 0.8 * answered


def _rating_items(ctx: CheckContext) -> tuple[pd.DataFrame, list[str], list[pd.DataFrame]]:
    """Numeric rating items of every grid, pooled.

    Returns (items for ``grid_sd``, all numeric item columns, notes). In the first frame an interview's items of a
    grid count only when it answered at least that grid's ``min_items`` of them, as in the straight-lining check.
    """
    blocks: list[pd.DataFrame] = []
    columns: list[str] = []
    frames: list[pd.DataFrame] = []
    for name, spec in ctx.config.variables.grid_specs().items():
        cols, missing = ctx.cols(spec.columns)
        frames.append(ctx.missing_columns_issue(CHECK_ID, missing))
        cols = [c for c in dict.fromkeys(cols) if c not in columns]
        if not cols:
            continue
        text = [c for c in cols if not _is_numeric(ctx, c)]
        if text:
            frames.append(
                _note(ctx, f"Grid '{name}' has answers that are not numbers ({', '.join(text)}); it is left out.")
            )
            continue
        nums = pd.DataFrame({c: ctx.num(c) for c in cols})
        columns.extend(cols)
        if len(cols) >= spec.min_items:
            blocks.append(nums.where(nums.notna().sum(axis=1) >= spec.min_items, axis=0))
    items = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=ctx.data.index)
    return items, columns, frames


def _skip_dependent(ctx: CheckContext) -> set[str]:
    """Columns asked of only some respondents (named in skip logic or in a conditional `required`)."""
    v = ctx.config.variables
    names = {n for rule in v.skip_logic for n in (*rule.then_empty, *rule.then_required)}
    names |= {r.variable for r in v.required_specs() if r.when}
    found, _ = ctx.cols(sorted(names))
    return set(found)


def _profile_columns(
    ctx: CheckContext, p: AnswerRegularityParams, grid_columns: list[str]
) -> tuple[list[str], set[str], pd.DataFrame]:
    """(profile columns, those compared on the log scale, note about configured columns that are missing)."""
    missing: list[str] = []
    if p.variables is not None:
        cols, missing = ctx.cols(p.variables)
    else:
        numeric, _ = ctx.cols(ctx.config.variables.numeric)
        skip = _skip_dependent(ctx)
        cols = [c for c in numeric if c not in skip] + grid_columns
    cols = list(dict.fromkeys(cols))
    log_names = p.log_transform if p.log_transform is not None else ctx.config.params_for("outliers").log_transform
    log_cols, _ = ctx.cols(log_names)
    return cols, set(log_cols) & set(cols), ctx.missing_columns_issue(CHECK_ID, missing)


def _standardised_profile(ctx: CheckContext, cols: list[str], log_cols: set[str]) -> pd.DataFrame:
    """One column per answer: robust z against all interviews (median and scaled MAD), clipped at +-6."""
    prof = pd.DataFrame({c: ctx.num(c) for c in cols}, index=ctx.data.index)
    for c in cols:
        if c in log_cols:
            prof[c] = np.log1p(prof[c].clip(lower=0))
    med = prof.median()
    mad = (prof - med).abs().median() * MAD_SCALE
    z = (prof - med) / mad.replace(0, np.nan)
    return z.clip(-PROFILE_CLIP, PROFILE_CLIP)


def _median_nearest_distance(zf: np.ndarray) -> float:
    """Median over rows of the RMS distance to the nearest other row (``nan`` with fewer than two rows)."""
    n, d = zf.shape
    if n < 2 or d == 0:
        return float("nan")
    step = max(1, _BLOCK_CELLS // n)
    best = np.empty(n)
    for s in range(0, n, step):
        block = cdist(zf[s : s + step], zf, "sqeuclidean")
        rows = np.arange(s, min(s + step, n))
        block[np.arange(len(rows)), rows] = np.inf  # not the interview itself
        best[rows] = block.min(axis=1)
    return float(np.median(np.sqrt(best / d)))


def peer_z(values: pd.Series) -> tuple[pd.Series, float] | None:
    """Robust z of each value against all values, ``(x - median) / (1.4826 * MAD)``, and the median.

    ``None`` when the MAD is zero (at least half of the values are equal), so there is no spread to compare with.
    """
    x = values.dropna().astype("float64")
    if x.empty:
        return None
    med = float(x.median())
    mad = float((x - med).abs().median()) * MAD_SCALE
    if not np.isfinite(mad) or mad <= 1e-9 * max(1.0, abs(med)):
        return None
    return (x - med) / mad, med


@register_check(
    CHECK_ID,
    title="Answer regularity against peers",
    description="Enumerators whose interviews are far more regular, or more varied, than other enumerators': the "
    "spread of ratings within interviews, how alike their households are, and how close they sit to the typical "
    "household. Too regular is a warning; too varied is information. Each flag is a lead for a back-check, not "
    "proof of fabrication.",
    category="fabrication",
    params=AnswerRegularityParams,
    references=(
        "FieldGuard AI-fabrication study (2026), branch sprint/ai-fabrication, signals frozen in commit b0ea6d1",
        "Iglewicz & Hoaglin (1993)",
    ),
)
def answer_regularity(ctx: CheckContext, p: AnswerRegularityParams) -> pd.DataFrame:
    if ctx.n == 0 or not ctx.has_enumerators:
        return empty_issues(with_row=True)
    wanted = [s for s in SIGNALS if s in p.signals]
    items, grid_columns, frames = _rating_items(ctx)
    profile_cols: list[str] = []
    log_cols: set[str] = set()
    if "spread" in wanted or "typicality" in wanted:
        profile_cols, log_cols, missing = _profile_columns(ctx, p, grid_columns)
        frames.append(missing)
    groups = {e: idx for e, idx in ctx.enumerator_groups(min_n=1).items() if e != MISSING_ENUMERATOR}
    eligible = sorted(e for e, idx in groups.items() if len(idx) >= p.min_n)
    if len(eligible) < p.min_enumerators:
        frames.append(
            _note(
                ctx,
                f"Peer comparison skipped: {len(eligible)} enumerator(s) have at least {p.min_n} interviews and "
                f"{p.min_enumerators} are needed.",
            )
        )
        return _concat(frames)

    raw: dict[str, pd.Series] = {}
    if "grid_sd" in wanted:
        if items.shape[1] >= 2:
            raw["grid_sd"] = items.std(axis=1, ddof=0).groupby(ctx.enumerator).mean()
        else:
            frames.append(_note(ctx, "grid_sd skipped: no numeric rating grid in variables.grids."))
    if "spread" in wanted or "typicality" in wanted:
        if not profile_cols:
            frames.append(_note(ctx, "spread and typicality skipped: no household profile variables found."))
        else:
            z = _standardised_profile(ctx, profile_cols, log_cols)
            if "typicality" in wanted:
                raw["typicality"] = np.sqrt((z**2).mean(axis=1)).groupby(ctx.enumerator).mean()
            if "spread" in wanted:
                zf = z.fillna(0.0).to_numpy()
                raw["spread"] = pd.Series({e: _median_nearest_distance(zf[groups[e]]) for e in eligible})

    excess: dict[str, float] = {}
    rows = []
    for signal in wanted:
        if signal not in raw:
            continue
        values = raw[signal].reindex(eligible)
        if values.notna().sum() < p.min_enumerators:
            frames.append(
                _note(ctx, f"{signal} skipped: fewer than {p.min_enumerators} enumerators have a value to compare.")
            )
            continue
        scored = peer_z(values)
        if scored is None:
            frames.append(
                _note(
                    ctx,
                    f"{signal} skipped: at least half of the enumerators have the same value, so there is no spread "
                    "among peers to compare with.",
                )
            )
            continue
        zs, med = scored
        ctx.add_metric_series(CHECK_ID, f"regularity_{signal}_z", zs)
        for enum, z_value in zs.items():
            zv = float(z_value)
            excess[str(enum)] = max(excess.get(str(enum), 0.0), max(-zv, 0.0) / p.threshold)
            if abs(zv) < p.threshold:
                continue
            low = zv < 0
            value = float(values[enum])
            rows.append(
                ctx.issue(
                    CHECK_ID,
                    Severity.WARNING if low else Severity.INFO,
                    f"{_EXPLAIN[signal]['low' if low else 'high']} ({_MEASURE[signal]} {value:.2f} vs a peer "
                    f"median of {med:.2f}; z={zv:.1f}). " + ("" if low else _HONEST_HIGH + " ") + _LEAD,
                    enumerator=str(enum),
                    variable=signal,
                    value=f"{value:.4g}",
                    metric=zv,
                )
            )
    for enum, ex in excess.items():
        ctx.add_metric(CHECK_ID, enum, "regularity_excess", ex)
    frames.append(ctx.frame(rows))
    return _concat(frames)
