"""Consistency checks: range/constraint violations and skip-logic rules from the config."""

from __future__ import annotations

import pandas as pd

from fieldguard.checks.context import CheckContext, normalize_code, normalize_series
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.expressions import Expression, ExpressionError
from fieldguard.models import Severity, empty_issues


class ConstraintParams(CheckParams):
    pass


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


@register_check(
    "constraints",
    title="Range and constraint violations",
    description="Hard (min/max/allowed) and soft ranges per variable plus cross-variable constraint expressions.",
    category="consistency",
    params=ConstraintParams,
    references=("IPA ipacheck: ipacheckoutliers/ipacheckconstraints", "DIME Wiki: High Frequency Checks"),
)
def constraints(ctx: CheckContext, p: ConstraintParams) -> pd.DataFrame:
    frames = []
    violated_any = pd.Series(False, index=ctx.data.index)
    absent: list[str] = []
    for i, rule in enumerate(ctx.config.variables.constraints):
        rid = rule.id or (f"{rule.variable}_range" if rule.variable and not rule.expression else f"constraint_{i + 1}")
        try:
            scope = (
                Expression(rule.when).evaluate(ctx.data, ctx.resolver)
                if rule.when
                else pd.Series(True, index=ctx.data.index)
            )
        except ExpressionError as exc:
            frames.append(ctx.dataset_issue("constraints", Severity.WARNING, f"Constraint {rid} skipped: {exc}"))
            continue
        if rule.expression:
            try:
                ok = Expression(rule.expression).evaluate(ctx.data, ctx.resolver)
            except ExpressionError as exc:
                frames.append(ctx.dataset_issue("constraints", Severity.WARNING, f"Constraint {rid} skipped: {exc}"))
                continue
            # only judge rows where every referenced variable is answered (not DK/refused)
            names = Expression(rule.expression).column_names()
            answered = pd.Series(True, index=ctx.data.index)
            for name in names:
                col = ctx.col(name)
                if col is not None:
                    answered &= ctx.data[col].notna() & ~ctx.special_mask(col)
            bad = scope & answered & ~ok
            violated_any |= bad
            msg = rule.message or f"Constraint violated: {rule.expression}"
            frames.append(
                ctx.record_issues("constraints", bad, rule.severity, f"[{rid}] {msg}", variable=", ".join(names))
            )
            continue
        assert rule.variable is not None
        col = ctx.col(rule.variable)
        if col is None:
            absent.append(rule.variable)
            continue
        raw = ctx.data[col]
        if rule.allowed is not None:
            allowed = {normalize_code(a) for a in rule.allowed}
            special = ctx.special_mask(col)
            bad = scope & raw.notna() & ~special & ~normalize_series(raw).isin(allowed)
            violated_any |= bad
            msg = rule.message or f"Value not in allowed set {sorted(allowed)}"
            frames.append(
                ctx.record_issues("constraints", bad, rule.severity, f"[{rid}] {msg}", variable=col, value=raw)
            )
        x = ctx.num(col)
        non_numeric = scope & raw.notna() & x.isna() & ~ctx.special_mask(col)
        if (rule.min is not None or rule.max is not None) and non_numeric.any():
            frames.append(
                ctx.record_issues(
                    "constraints", non_numeric, Severity.WARNING, f"[{rid}] Non-numeric value", variable=col, value=raw
                )
            )
        if rule.min is not None or rule.max is not None:
            lo = rule.min if rule.min is not None else float("-inf")
            hi = rule.max if rule.max is not None else float("inf")
            bad = scope & x.notna() & ((x < lo) | (x > hi))
            violated_any |= bad
            msg = rule.message or f"Outside hard range [{_fmt(rule.min)}, {_fmt(rule.max)}]"
            frames.append(
                ctx.record_issues(
                    "constraints", bad, rule.severity, f"[{rid}] {msg}", variable=col, value=raw, metric=x
                )
            )
        if rule.soft_min is not None or rule.soft_max is not None:
            lo = rule.soft_min if rule.soft_min is not None else float("-inf")
            hi = rule.soft_max if rule.soft_max is not None else float("inf")
            hard_lo = rule.min if rule.min is not None else float("-inf")
            hard_hi = rule.max if rule.max is not None else float("inf")
            soft = scope & x.notna() & ((x < lo) | (x > hi)) & (x >= hard_lo) & (x <= hard_hi)
            msg = f"Outside expected range [{_fmt(rule.soft_min)}, {_fmt(rule.soft_max)}] - please confirm"
            frames.append(
                ctx.record_issues(
                    "constraints", soft, Severity.INFO, f"[{rid}] {msg}", variable=col, value=raw, metric=x
                )
            )
    frames.append(ctx.missing_columns_issue("constraints", absent))
    if ctx.n and ctx.config.variables.constraints:
        ctx.add_metric_series("constraints", "constraint_violation_rate", violated_any.groupby(ctx.enumerator).mean())
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)


class SkipLogicParams(CheckParams):
    pass


@register_check(
    "skip_logic",
    title="Skip-logic consistency",
    description="Rules like `if q5 == 'no' then q6 must be empty` (or required), in Python or XLSForm syntax.",
    category="consistency",
    params=SkipLogicParams,
    references=("DIME Wiki: High Frequency Checks", "J-PAL data quality checks"),
)
def skip_logic(ctx: CheckContext, p: SkipLogicParams) -> pd.DataFrame:
    frames = []
    violated_any = pd.Series(False, index=ctx.data.index)
    for i, rule in enumerate(ctx.config.variables.skip_logic):
        rid = rule.id or f"skip_{i + 1}"
        try:
            cond = Expression(rule.if_).evaluate(ctx.data, ctx.resolver)
        except ExpressionError as exc:
            frames.append(ctx.dataset_issue("skip_logic", Severity.WARNING, f"Skip rule {rid} skipped: {exc}"))
            continue
        absent = []
        for name in rule.then_empty:
            col = ctx.col(name)
            if col is None:
                absent.append(name)
                continue
            filled = ctx.data[col].notna() & (ctx.text(col).fillna("") != "")
            bad = cond & filled
            violated_any |= bad
            msg = rule.message or f"{name} should be empty when {rule.if_}"
            frames.append(
                ctx.record_issues("skip_logic", bad, rule.severity, f"[{rid}] {msg}", variable=col, value=ctx.data[col])
            )
        for name in rule.then_required:
            col = ctx.col(name)
            if col is None:
                absent.append(name)
                continue
            empty = ctx.data[col].isna() | (ctx.text(col).fillna("") == "")
            bad = cond & empty
            violated_any |= bad
            msg = rule.message or f"{name} should be answered when {rule.if_}"
            frames.append(ctx.record_issues("skip_logic", bad, rule.severity, f"[{rid}] {msg}", variable=col))
        if rule.then:
            try:
                ok = Expression(rule.then).evaluate(ctx.data, ctx.resolver)
            except ExpressionError as exc:
                frames.append(ctx.dataset_issue("skip_logic", Severity.WARNING, f"Skip rule {rid} skipped: {exc}"))
                continue
            bad = cond & ~ok
            violated_any |= bad
            msg = rule.message or f"When {rule.if_}, expected {rule.then}"
            frames.append(ctx.record_issues("skip_logic", bad, rule.severity, f"[{rid}] {msg}"))
        frames.append(ctx.missing_columns_issue("skip_logic", absent))
    if ctx.n and ctx.config.variables.skip_logic:
        ctx.add_metric_series("skip_logic", "skip_violation_rate", violated_any.groupby(ctx.enumerator).mean())
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)
