"""Integrity checks: duplicate IDs, near-duplicate submissions, missing required fields."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import Field

from fieldguard.checks.context import CheckContext
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.expressions import Expression, ExpressionError
from fieldguard.models import Level, Severity, empty_issues


# ----------------------------------------------------------------------------- duplicate IDs
class DuplicateIdParams(CheckParams):
    id_columns: list[str] | None = Field(
        None, description="Columns forming the respondent key (default: respondent_id role)."
    )
    check_submission_ids: bool = True
    severity: Severity = Severity.CRITICAL


@register_check(
    "duplicate_ids",
    title="Duplicate IDs",
    description="Respondent IDs (or composite keys) that occur more than once, and duplicated submission UUIDs.",
    category="integrity",
    params=DuplicateIdParams,
    references=("J-PAL data quality checks", "IPA ipacheck: ipacheckids"),
)
def duplicate_ids(ctx: CheckContext, p: DuplicateIdParams) -> pd.DataFrame:
    frames = []
    if ctx.n == 0:
        return empty_issues(with_row=True)
    key: pd.Series | None = None
    label = "respondent_id"
    if p.id_columns:
        found, missing = ctx.cols(p.id_columns)
        frames.append(ctx.missing_columns_issue("duplicate_ids", missing))
        if found and not missing:
            parts = [ctx.text(c).fillna("") for c in found]
            key = parts[0]
            for part in parts[1:]:
                key = key + "|" + part
            key = key.mask(pd.concat([ctx.data[c].isna() for c in found], axis=1).all(axis=1))
            label = "+".join(found)
    elif ctx.respondent_id is not None:
        key = ctx.respondent_id
        label = ctx.roles["respondent_id"] or "respondent_id"
    if key is not None:
        dup = key.notna() & key.duplicated(keep=False)
        if dup.any():
            counts = key[dup].map(key[dup].value_counts())
            msg = (
                "ID "
                + key.astype("str")
                + " appears "
                + counts.reindex(key.index).fillna(0).astype(int).astype(str)
                + " times"
            )
            frames.append(
                ctx.record_issues(
                    "duplicate_ids", dup, p.severity, msg, variable=label, value=key, metric=counts.reindex(key.index)
                )
            )
        missing_key = key.isna()
        if missing_key.any():
            frames.append(
                ctx.record_issues(
                    "duplicate_ids", missing_key, Severity.WARNING, "Respondent ID is missing", variable=label
                )
            )
        share = dup.groupby(ctx.enumerator).mean()
        ctx.add_metric_series("duplicate_ids", "duplicate_id_share", share)
    if p.check_submission_ids and ctx.roles.get("submission_id"):
        sid = ctx.record_id
        dup_s = sid.duplicated(keep=False)
        if dup_s.any():
            frames.append(
                ctx.record_issues(
                    "duplicate_ids",
                    dup_s,
                    Severity.CRITICAL,
                    "Submission UUID is duplicated (same submission exported/merged twice?)",
                    variable=ctx.roles["submission_id"],
                    value=sid,
                )
            )
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)


# ----------------------------------------------------------------------------- near duplicates
class NearDuplicateParams(CheckParams):
    threshold: float = Field(0.95, gt=0, le=1, description="Share of matching answers to flag a pair.")
    min_compared: int = Field(10, ge=1, description="Minimum answers both submissions have non-missing.")
    scope: str = Field("enumerator", pattern="^(enumerator|all)$")
    max_group_size: int = Field(3000, ge=10)
    severity: Severity = Severity.WARNING


def _pairwise_match(codes: np.ndarray, valid: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Best match share and partner index for each row (excluding itself)."""
    n = codes.shape[0]
    best = np.zeros(n)
    partner = np.full(n, -1)
    for s in range(0, n, chunk):
        a, va = codes[s : s + chunk, None, :], valid[s : s + chunk, None, :]
        both = va & valid[None, :, :]
        eq = (a == codes[None, :, :]) & both
        n_both = both.sum(axis=2)
        share = np.where(n_both > 0, eq.sum(axis=2) / np.maximum(n_both, 1), 0.0)
        rows = np.arange(s, min(s + chunk, n))
        share[np.arange(len(rows)), rows] = -1  # exclude self
        j = share.argmax(axis=1)
        best[rows] = share[np.arange(len(rows)), j]
        partner[rows] = j
        # store n_both of best partner via share>... compute below
    return best, partner


@register_check(
    "near_duplicates",
    title="Near-duplicate submissions",
    description="Pairs of submissions whose answers match on a very high share of questions (copied interviews).",
    category="fabrication",
    params=NearDuplicateParams,
    references=("Kuriakose & Robbins (2016)", "Simmons et al. (2016)"),
)
def near_duplicates(ctx: CheckContext, p: NearDuplicateParams) -> pd.DataFrame:
    cols = ctx.answer_columns()
    if ctx.n < 2 or len(cols) < p.min_compared:
        return empty_issues(with_row=True)
    codes = np.empty((ctx.n, len(cols)), dtype=np.int64)
    valid = np.empty((ctx.n, len(cols)), dtype=bool)
    for j, c in enumerate(cols):
        s = ctx.data[c]
        codes[:, j], _ = pd.factorize(s.astype("str").where(s.notna()), use_na_sentinel=True)
        valid[:, j] = s.notna().to_numpy()
    groups = ctx.enumerator_groups(min_n=2) if p.scope == "enumerator" else {"all": np.arange(ctx.n)}
    flagged_rows: list[int] = []
    partners: list[str] = []
    shares: list[float] = []
    is_dup = np.zeros(ctx.n, dtype=bool)
    for _, idx in groups.items():
        if len(idx) > p.max_group_size:
            idx = idx[: p.max_group_size]
        best, partner = _pairwise_match(codes[idx], valid[idx])
        n_both = (valid[idx] & valid[idx][partner]).sum(axis=1)
        hit = (best >= p.threshold) & (n_both >= p.min_compared)
        for k in np.flatnonzero(hit):
            row = int(idx[k])
            is_dup[row] = True
            flagged_rows.append(row)
            partners.append(str(ctx.record_id.iloc[int(idx[partner[k]])]))
            shares.append(float(best[k]))
    share = pd.Series(is_dup, index=ctx.data.index).groupby(ctx.enumerator).mean()
    ctx.add_metric_series("near_duplicates", "near_duplicate_share", share)
    if not flagged_rows:
        return empty_issues(with_row=True)
    msgs = [f"{s:.0%} of answers identical to submission {pid}" for s, pid in zip(shares, partners, strict=True)]
    severities = [Severity.CRITICAL if s >= 0.999 else p.severity for s in shares]
    out = ctx.record_issues("near_duplicates", flagged_rows, p.severity, msgs, value=partners, metric=shares)
    out["severity"] = [str(s) for s in severities]
    return out


# ----------------------------------------------------------------------------- missing required
class MissingParams(CheckParams):
    flag_all_missing_columns: bool = True
    answered_ratio_metric: bool = True


@register_check(
    "missing_required",
    title="Missing required fields",
    description="Required (optionally conditional) variables left empty, and configured columns that are entirely empty.",
    category="integrity",
    params=MissingParams,
    references=("DIME Wiki: High Frequency Checks",),
)
def missing_required(ctx: CheckContext, p: MissingParams) -> pd.DataFrame:
    frames = []
    specs = ctx.config.variables.required_specs()
    missing_any = pd.Series(False, index=ctx.data.index)
    absent: list[str] = []
    for spec in specs:
        col = ctx.col(spec.variable)
        if col is None:
            absent.append(spec.variable)
            continue
        empty = ctx.data[col].isna() | (ctx.text(col).fillna("") == "")
        if spec.when:
            try:
                empty &= Expression(spec.when).evaluate(ctx.data, ctx.resolver)
            except ExpressionError as exc:
                frames.append(
                    ctx.dataset_issue("missing_required", Severity.WARNING, f"Rule for {spec.variable} skipped: {exc}")
                )
                continue
        missing_any |= empty
        if empty.any():
            msg = f"Required field {spec.variable} is empty" + (f" (when {spec.when})" if spec.when else "")
            frames.append(ctx.record_issues("missing_required", empty, spec.severity, msg, variable=col))
    frames.append(ctx.missing_columns_issue("missing_required", absent))
    if ctx.n:
        ctx.add_metric_series("missing_required", "missing_required_rate", missing_any.groupby(ctx.enumerator).mean())
    # completeness: answered items per interview relative to the overall median
    cols = ctx.answer_columns()
    if p.answered_ratio_metric and cols and ctx.n:
        answered = ctx.data[cols].notna().sum(axis=1).astype("float64")
        med = float(answered.median())
        if med > 0:
            ratio = answered.groupby(ctx.enumerator).median() / med
            ctx.add_metric_series("missing_required", "answered_ratio", ratio)
    if p.flag_all_missing_columns and ctx.n:
        configured = set(ctx.config.variables.key) | set(ctx.config.variables.numeric) | {s.variable for s in specs}
        found, _ = ctx.cols(sorted(configured))
        rows = [
            ctx.issue(
                "missing_required",
                Severity.WARNING,
                f"Column {c} is empty in every record",
                level=Level.VARIABLE,
                variable=c,
            )
            for c in found
            if ctx.data[c].isna().all()
        ]
        frames.append(ctx.frame(rows))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)
