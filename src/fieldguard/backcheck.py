"""Back-checks: stratified re-interview sampling and comparison of re-interview answers.

Sampling (``select_sample``): for every enumerator, a target share of their interviews is drawn. Part of the
sample is uniform at random (``random_share``); the rest is drawn with probability proportional to a risk weight
(interviews with record-level flags weigh more, and interviews by enumerators flagged by the scorecard are
oversampled). Every enumerator gets at least ``min_per_enumerator`` back-checks where they have that many interviews.

Comparison (``compare_backchecks``): each back-checked variable is compared with the original answer using a
rule for its kind (categorical, text similarity, numeric tolerance, date window). Variables are grouped into three
types (the scheme used in many back-check protocols):

* type 1: factual identity and process variables that should match exactly;
* type 2: administrative or factual variables where small differences are tolerated;
* type 3: key outcomes that are expected to vary (for example income), judged with wide tolerance.

Error rates are computed per variable and pooled per enumerator and type. A rate above the configured threshold
(default 10%, a common convention, not a validated cut-off) is a *lead for supervision*, not evidence of falsification.
The field sample is not shared with enumerators; contact columns appear only in the field-team sample file.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fieldguard.config import BackcheckVariable, FieldGuardConfig
from fieldguard.io.columns import ColumnResolver
from fieldguard.models import ISSUE_COLUMNS, Level, Severity, coerce_issues

SAMPLE_COLUMNS = [
    "record_id",
    "respondent_id",
    "enumerator",
    "cluster",
    "reason",
    "record_flags",
    "enumerator_flagged",
    "backchecker",
]


@dataclass
class BackcheckReport:
    comparisons: pd.DataFrame
    variable_summary: pd.DataFrame
    enumerator_summary: pd.DataFrame
    issues: pd.DataFrame
    coverage: dict[str, Any] = field(default_factory=dict)


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip() == ""


def _key(v: Any) -> str | None:
    if _is_missing(v):
        return None
    text = str(v).strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    return text


def _norm_text(v: Any) -> str | None:
    if _is_missing(v):
        return None
    return str(v).strip().lower()


def _as_float(v: Any) -> float:
    if _is_missing(v):
        return float("nan")
    try:
        return float(str(v).strip())
    except ValueError:
        return float("nan")


def _as_date(v: Any) -> pd.Timestamp | None:
    if _is_missing(v):
        return None
    ts = pd.to_datetime(str(v).strip(), errors="coerce")
    if pd.isna(ts):
        return None
    out = pd.Timestamp(ts)
    return out.tz_localize(None) if out.tzinfo is not None else out


def compare_pair(
    original: Any, backcheck: Any, spec: BackcheckVariable, *, missing_as_value: bool = True
) -> tuple[bool | None, float | None]:
    """Compare one pair of answers. Returns (match, absolute_difference); match None means not comparable."""
    ignore = {_norm_text(x) for x in spec.ignore_values}
    if spec.kind in ("categorical", "text", "date"):
        if _norm_text(original) in ignore or _norm_text(backcheck) in ignore:
            return None, None
    else:
        if _key(original) in {_key(x) for x in spec.ignore_values} or _key(backcheck) in {
            _key(x) for x in spec.ignore_values
        }:
            return None, None

    if spec.kind == "date":
        da, db = _as_date(original), _as_date(backcheck)
        if da is None and db is None:
            return True, 0.0
        if da is None or db is None:
            return (False, None) if missing_as_value else (None, None)
        days = abs((da - db).days)
        return days <= spec.date_tolerance_days, float(days)

    if spec.kind == "numeric":
        fa, fb = _as_float(original), _as_float(backcheck)
        if math.isnan(fa) and math.isnan(fb):
            return True, 0.0
        if math.isnan(fa) or math.isnan(fb):
            return (False, None) if missing_as_value else (None, None)
        diff = abs(fa - fb)
        tol = max(spec.abs_tolerance, spec.rel_tolerance * abs(fa))
        return diff <= tol + 1e-12, diff

    na, nb = _norm_text(original), _norm_text(backcheck)
    if na is None and nb is None:
        return True, 0.0
    if na is None or nb is None:
        return (False, None) if missing_as_value else (None, None)
    if spec.kind == "text":
        ratio = difflib.SequenceMatcher(None, na, nb).ratio()
        return ratio >= spec.text_similarity, float(1.0 - ratio)
    return (na == nb), (None if na == nb else 1.0)


def select_sample(
    ctx: Any,
    issues: pd.DataFrame,
    scorecard: pd.DataFrame,
    cfg: FieldGuardConfig,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Stratified back-check sample from a completed run (``ctx`` is ``RunResult.context``)."""
    bc = cfg.backcheck
    seed = bc.seed if seed is None else seed
    n = ctx.n
    if n == 0 or not ctx.has_enumerators:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    rec_issues = (
        issues[
            (issues["level"] == Level.RECORD.value)
            & issues["severity"].isin([Severity.WARNING.value, Severity.CRITICAL.value])
        ]
        if not issues.empty
        else issues
    )
    flags_by_id = rec_issues.groupby("record_id").size() if not rec_issues.empty else pd.Series(dtype="float64")
    record_flags = ctx.record_id.astype(str).map(flags_by_id).fillna(0).astype(int).to_numpy()
    flagged_enum = set()
    if not scorecard.empty and "flagged" in scorecard.columns:
        flagged_enum = set(scorecard.loc[scorecard["flagged"].astype(bool), "enumerator"].astype(str))
    respondent = ctx.respondent_id if ctx.respondent_id is not None else ctx.record_id
    excluded = {str(x) for x in bc.exclude_ids}
    rows: list[pd.DataFrame] = []
    groups = ctx.enumerator_groups(min_n=1)
    for g_index, (enum, idx) in enumerate(sorted(groups.items())):
        if enum in {"(missing)", "(unknown)"}:
            continue
        idx = np.array(
            [i for i in idx if str(ctx.record_id.iloc[i]) not in excluded and str(respondent.iloc[i]) not in excluded]
        )
        if idx.size == 0:
            continue
        target = min(idx.size, max(bc.min_per_enumerator, math.ceil(bc.sample_rate * idx.size)))
        if enum in flagged_enum:
            target = min(idx.size, max(target, math.ceil(bc.sample_rate * idx.size * (1 + bc.risk_oversample))))
        rng = np.random.default_rng([seed, g_index])
        n_random = min(target, round(bc.random_share * target))
        random_pick = rng.choice(idx, size=n_random, replace=False) if n_random else np.array([], dtype=int)
        remaining = np.setdiff1d(idx, random_pick)
        n_risk = min(target - n_random, remaining.size)
        weights = np.where(record_flags[remaining] > 0, bc.flagged_record_weight, 1.0)
        if enum in flagged_enum:
            weights = weights * (1.0 + bc.risk_oversample)
        risk_pick = (
            rng.choice(remaining, size=n_risk, replace=False, p=weights / weights.sum())
            if n_risk and remaining.size
            else np.array([], dtype=int)
        )
        for reason, picked in (("random", random_pick), ("risk", risk_pick)):
            if len(picked) == 0:
                continue
            frame = pd.DataFrame(
                {
                    "record_id": ctx.record_id.iloc[picked].to_numpy(),
                    "respondent_id": respondent.iloc[picked].to_numpy(),
                    "enumerator": enum,
                    "cluster": ctx.cluster.iloc[picked].to_numpy() if ctx.cluster is not None else None,
                    "reason": reason,
                    "record_flags": record_flags[picked],
                    "enumerator_flagged": enum in flagged_enum,
                    "backchecker": None,
                }
            )
            for col in bc.contact_columns:
                real = ctx.col(col)
                if real is not None:
                    frame[col] = ctx.data[real].iloc[picked].to_numpy()
            rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["enumerator", "reason", "record_id"]).reset_index(drop=True)


def _resolve_pair(name: str, resolver: ColumnResolver) -> str | None:
    return resolver.resolve(name)


def compare_backchecks(
    survey: pd.DataFrame,
    enumerator: pd.Series,
    backcheck: pd.DataFrame,
    cfg: FieldGuardConfig,
    *,
    id_column: str,
    run_id: str = "backcheck",
) -> BackcheckReport:
    """Compare re-interview answers with the original survey (both frames already read as text)."""
    bc_cfg = cfg.backcheck
    orig_resolver = ColumnResolver(survey.columns)
    bc_resolver = ColumnResolver(backcheck.columns)
    bc_id = bc_resolver.resolve(bc_cfg.id_column or id_column) or bc_resolver.resolve(id_column)
    orig_id = orig_resolver.resolve(id_column)
    issues: list[dict[str, Any]] = []
    if orig_id is None or bc_id is None:
        issues.append(
            _issue(
                run_id,
                "backcheck",
                Severity.WARNING,
                "No matching ID column in survey and back-check data",
                level=Level.DATASET,
            )
        )
        return _empty_report(issues, run_id)

    survey_keys = survey[orig_id].map(_key)
    first_pos: dict[str, int] = {}
    for pos, k in enumerate(survey_keys):
        if k is not None:
            first_pos.setdefault(k, pos)
    bc_keys = backcheck[bc_id].map(_key)
    dup_mask = bc_keys.duplicated(keep="first") & bc_keys.notna()
    if dup_mask.any():
        issues.append(
            _issue(
                run_id,
                "backcheck",
                Severity.WARNING,
                f"{int(dup_mask.sum())} duplicate back-check record(s) ignored",
                level=Level.DATASET,
            )
        )
    unmatched = int(sum(1 for k in bc_keys[~dup_mask] if k is not None and k not in first_pos))
    if unmatched:
        issues.append(
            _issue(
                run_id,
                "backcheck",
                Severity.WARNING,
                f"{unmatched} back-check record(s) do not match any survey record",
                level=Level.DATASET,
            )
        )

    variables = bc_cfg.variables
    records: list[dict[str, Any]] = []
    for pos_bc, key in enumerate(bc_keys):
        if key is None or key not in first_pos or dup_mask.iloc[pos_bc]:
            continue
        orig_pos = first_pos[key]
        enum = str(enumerator.iloc[orig_pos])
        for spec in variables:
            o_col = _resolve_pair(spec.name, orig_resolver)
            b_col = _resolve_pair(spec.backcheck_name or spec.name, bc_resolver)
            if o_col is None or b_col is None:
                continue
            match, diff = compare_pair(
                survey[o_col].iloc[orig_pos],
                backcheck[b_col].iloc[pos_bc],
                spec,
                missing_as_value=bc_cfg.missing_as_value,
            )
            records.append(
                {
                    "key": key,
                    "enumerator": enum,
                    "variable": spec.name,
                    "type": spec.type,
                    "kind": spec.kind,
                    "original": _clean(survey[o_col].iloc[orig_pos]),
                    "backcheck": _clean(backcheck[b_col].iloc[pos_bc]),
                    "compared": match is not None,
                    "match": bool(match) if match is not None else None,
                    "abs_diff": diff,
                }
            )
    missing_vars = [
        v.name
        for v in variables
        if orig_resolver.resolve(v.name) is None or bc_resolver.resolve(v.backcheck_name or v.name) is None
    ]
    if missing_vars:
        issues.append(
            _issue(
                run_id,
                "backcheck",
                Severity.WARNING,
                f"Back-check variable(s) not found in both files: {', '.join(missing_vars)}",
                level=Level.DATASET,
            )
        )

    comp = pd.DataFrame(records)
    if comp.empty:
        return _empty_report(issues, run_id, comparisons=comp)
    compared = comp[comp["compared"]].copy()
    compared["mismatch"] = ~compared["match"].astype(bool)

    var_rows = []
    for spec in variables:
        sub = compared[compared["variable"] == spec.name]
        n_c, n_m = len(sub), int(sub["mismatch"].sum())
        rate = n_m / n_c if n_c else float("nan")
        thr = bc_cfg.threshold(spec.type)
        flagged = bool(n_c >= bc_cfg.min_compared and rate > thr)
        var_rows.append(
            {
                "variable": spec.name,
                "type": spec.type,
                "kind": spec.kind,
                "n_compared": n_c,
                "n_mismatch": n_m,
                "error_rate": rate,
                "threshold": thr,
                "flagged": flagged,
            }
        )
        if flagged:
            issues.append(
                _issue(
                    run_id,
                    "backcheck",
                    Severity.WARNING if spec.type == 1 else Severity.INFO,
                    f"Back-check disagreement on {spec.name}: {rate:.0%} of {n_c} re-interviews differ (type {spec.type} threshold {thr:.0%})",
                    level=Level.VARIABLE,
                    variable=spec.name,
                    value=f"{rate:.4f}",
                    metric=rate,
                )
            )
    variable_summary = pd.DataFrame(var_rows)

    enum_rows = []
    for enum, sub in compared.groupby("enumerator"):
        for t in sorted({v.type for v in variables}):
            ts = sub[sub["type"] == t]
            n_c, n_m = len(ts), int(ts["mismatch"].sum())
            rate = n_m / n_c if n_c else float("nan")
            thr = bc_cfg.threshold(t)
            flagged = bool(n_c >= bc_cfg.min_compared and rate > thr)
            enum_rows.append(
                {
                    "enumerator": enum,
                    "type": t,
                    "n_compared": n_c,
                    "n_mismatch": n_m,
                    "error_rate": rate,
                    "threshold": thr,
                    "flagged": flagged,
                    "n_records": int(ts["key"].nunique()),
                }
            )
            if flagged:
                sev = Severity.WARNING if t == 1 else Severity.INFO
                issues.append(
                    _issue(
                        run_id,
                        "backcheck",
                        sev,
                        f"Type {t} back-check error rate {rate:.0%} ({n_m}/{n_c} comparisons) above {thr:.0%}: lead for supervisory review, not proof",
                        level=Level.ENUMERATOR,
                        enumerator=enum,
                        variable=f"type {t}",
                        value=f"{rate:.4f}",
                        metric=rate,
                    )
                )
    enumerator_summary = pd.DataFrame(enum_rows)
    n_enum_all = int(enumerator.nunique(dropna=True))
    back_enum = int(compared["enumerator"].nunique()) if not compared.empty else 0
    coverage = {
        "n_backchecked_records": int(comp["key"].nunique()),
        "n_enumerators": n_enum_all,
        "n_enumerators_backchecked": back_enum,
        "share_enumerators_covered": back_enum / n_enum_all if n_enum_all else float("nan"),
    }
    comparisons = comp.drop(columns=["compared"])
    issue_df = coerce_issues(pd.DataFrame(issues)) if issues else pd.DataFrame(columns=ISSUE_COLUMNS)
    return BackcheckReport(comparisons, variable_summary, enumerator_summary, issue_df, coverage)


def _clean(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v if isinstance(v, (int, float, str)) else str(v)


def _issue(
    run_id: str,
    check_id: str,
    severity: Severity,
    message: str,
    *,
    level: Level,
    enumerator: str | None = None,
    variable: str | None = None,
    value: Any = None,
    metric: float | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "check_id": check_id,
        "severity": str(severity),
        "level": str(level),
        "record_id": None,
        "respondent_id": None,
        "enumerator": enumerator,
        "survey_date": None,
        "variable": variable,
        "value": None if value is None else str(value),
        "metric": metric,
        "message": message,
    }


def _empty_report(
    issues: list[dict[str, Any]], run_id: str, comparisons: pd.DataFrame | None = None
) -> BackcheckReport:
    issue_df = coerce_issues(pd.DataFrame(issues)) if issues else pd.DataFrame(columns=ISSUE_COLUMNS)
    comp = (
        comparisons
        if comparisons is not None
        else pd.DataFrame(
            columns=["key", "enumerator", "variable", "type", "kind", "original", "backcheck", "match", "abs_diff"]
        )
    )
    return BackcheckReport(comp, pd.DataFrame(), pd.DataFrame(), issue_df, {"n_backchecked_records": 0})


__all__ = ["SAMPLE_COLUMNS", "BackcheckReport", "compare_backchecks", "compare_pair", "select_sample"]
