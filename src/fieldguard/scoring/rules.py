"""Transparent rule-based enumerator risk points.

Each indicator earns ``weight x points_per_weight x s`` points, where ``s`` ramps linearly from 0 at ``lo``
to 1 at ``hi`` (``lo > hi`` means lower values are riskier). The rule score is the sum, capped at 100.
Indicators whose metric is unavailable (e.g. no GPS) simply contribute nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fieldguard.config import ScoringConfig, ScoringRule

DEFAULT_RULES: list[ScoringRule] = [
    ScoringRule(metric="short_duration_share", lo=0.10, hi=0.40, weight=2.0, label="Short interviews"),
    ScoringRule(metric="median_duration_ratio", lo=0.85, hi=0.55, weight=2.0, label="Fast median duration"),
    ScoringRule(metric="straightline_share", lo=0.10, hi=0.40, weight=1.5, label="Straight-lining"),
    ScoringRule(metric="grid_sd_ratio", lo=0.85, hi=0.50, weight=1.0, label="Low within-grid variance"),
    ScoringRule(metric="variance_ratio", lo=0.80, hi=0.45, weight=1.5, label="Low answer variance"),
    ScoringRule(metric="benford_excess", lo=1.5, hi=3.0, weight=1.0, label="Benford deviation"),
    ScoringRule(metric="heaping_ratio", lo=1.25, hi=2.0, weight=1.0, label="Digit preference"),
    ScoringRule(metric="gps_duplicate_share", lo=0.05, hi=0.30, weight=2.0, label="Same-spot GPS"),
    ScoringRule(metric="gps_dispersion_ratio", lo=0.50, hi=0.15, weight=1.5, label="Concentrated GPS"),
    ScoringRule(metric="gps_outside_share", lo=0.10, hi=0.50, weight=1.0, label="GPS outside cluster"),
    ScoringRule(metric="overlap_share", lo=0.02, hi=0.20, weight=1.5, label="Overlapping interviews"),
    ScoringRule(metric="offhours_share", lo=0.05, hi=0.30, weight=1.0, label="Off-hours interviews"),
    ScoringRule(metric="nonresponse_abs_z", lo=2.0, hi=5.0, weight=0.5, label="Unusual DK/refusal rate"),
    ScoringRule(metric="near_duplicate_share", lo=0.02, hi=0.20, weight=2.0, label="Near-duplicate submissions"),
    ScoringRule(metric="answered_ratio", lo=0.92, hi=0.75, weight=1.0, label="Fewer questions answered"),
    ScoringRule(metric="mean_deviation", lo=2.0, hi=4.0, weight=0.5, label="Atypical answer means"),
    ScoringRule(metric="duplicate_id_share", lo=0.0, hi=0.10, weight=0.5, label="Duplicate IDs"),
    ScoringRule(metric="missing_required_rate", lo=0.02, hi=0.10, weight=0.5, label="Missing required fields"),
    ScoringRule(metric="constraint_violation_rate", lo=0.02, hi=0.10, weight=0.5, label="Constraint violations"),
    ScoringRule(metric="skip_violation_rate", lo=0.02, hi=0.10, weight=0.5, label="Skip-logic violations"),
    ScoringRule(metric="outlier_rate", lo=0.05, hi=0.15, weight=0.5, label="Outliers"),
]


def effective_rules(cfg: ScoringConfig) -> list[ScoringRule]:
    rules = list(cfg.rules) if cfg.rules is not None else [r.model_copy() for r in DEFAULT_RULES]
    by_metric = {r.metric: r for r in rules}
    for metric, override in cfg.rule_overrides.items():
        if metric in by_metric:
            by_metric[metric] = by_metric[metric].model_copy(update=override)
        else:
            by_metric[metric] = ScoringRule(metric=metric, **override)
    return list(by_metric.values())


def ramp(values: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi == lo:
        return (values >= hi).astype(float).where(values.notna())
    return ((values - lo) / (hi - lo)).clip(0, 1)


def rule_scores(features: pd.DataFrame, cfg: ScoringConfig) -> tuple[pd.Series, pd.DataFrame]:
    """Return (score 0-100 per enumerator, points per indicator)."""
    points = pd.DataFrame(index=features.index)
    for rule in effective_rules(cfg):
        if rule.metric not in features.columns:
            continue
        s = ramp(features[rule.metric].astype("float64"), rule.lo, rule.hi)
        points[rule.label or rule.metric] = (s * rule.weight * cfg.points_per_weight).fillna(0.0)
    score = points.sum(axis=1).clip(upper=100.0) if not points.empty else pd.Series(0.0, index=features.index)
    return score.astype("float64"), points


def top_reasons(points: pd.DataFrame, k: int = 3, min_points: float = 3.0) -> pd.Series:
    def fmt(row: pd.Series) -> str:
        r = row[row >= min_points].sort_values(ascending=False).head(k)
        return "; ".join(f"{name} ({val:.0f})" for name, val in r.items())

    if points.empty:
        return pd.Series("", index=points.index)
    return points.apply(fmt, axis=1).replace({np.nan: ""})
