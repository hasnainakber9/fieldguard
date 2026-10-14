"""Enumerator scoring: transparent rule points, the Isolation Forest / LOF anomaly score and the scorecard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fieldguard.checks.context import build_context
from fieldguard.config import ScoringConfig, ScoringRule
from fieldguard.models import empty_metrics
from fieldguard.scoring.anomaly import ANOMALY_FEATURES, anomaly_scores, robust_standardize
from fieldguard.scoring.rules import DEFAULT_RULES, effective_rules, ramp, rule_scores, top_reasons
from fieldguard.scoring.scorecard import build_scorecard, feature_table
from tests.conftest import make_cfg, survey_frame


def test_ramp_is_linear_and_reversible():
    s = pd.Series([0.0, 0.5, 1.0, np.nan])
    assert ramp(s, 0.0, 1.0).tolist()[:3] == [0.0, 0.5, 1.0]
    rev = ramp(pd.Series([0.9, 0.5, 0.1]), 0.9, 0.1)  # lo > hi: lower values are riskier
    assert rev.tolist() == [0.0, 0.5, 1.0]
    step = ramp(pd.Series([0.2, 0.7]), 0.5, 0.5)
    assert step.tolist() == [0.0, 1.0]


def test_rule_scores_cap_at_100_and_ignore_missing_metrics():
    feats = pd.DataFrame({"short_duration_share": [0.0, 0.6], "dk_rate": [0.5, np.nan]}, index=["E1", "E2"])
    cfg = ScoringConfig()
    score, points = rule_scores(feats, cfg)
    assert score.loc["E2"] > score.loc["E1"]
    assert (score <= 100).all()
    assert "Short interviews" in points.columns
    assert points.loc["E1", "Short interviews"] == 0.0


def test_rule_override_and_new_metric_are_merged():
    cfg = ScoringConfig(
        rule_overrides={"short_duration_share": {"hi": 0.9}, "custom_metric": {"lo": 0.0, "hi": 1.0, "weight": 2.0}}
    )
    rules = {r.metric: r for r in effective_rules(cfg)}
    assert rules["short_duration_share"].hi == 0.9
    assert rules["custom_metric"].weight == 2.0
    assert len(rules) == len(DEFAULT_RULES) + 1


def test_explicit_rule_set_replaces_defaults():
    cfg = ScoringConfig(rules=[ScoringRule(metric="dk_rate", lo=0.0, hi=0.2, weight=1.0, label="DK")])
    assert [r.metric for r in effective_rules(cfg)] == ["dk_rate"]


def test_top_reasons_formats_and_filters_small_points():
    points = pd.DataFrame({"A": [12.0, 1.0], "B": [30.0, 0.0], "C": [5.0, 0.0]}, index=["E1", "E2"])
    reasons = top_reasons(points, k=2, min_points=3.0)
    assert reasons["E1"] == "B (30); A (12)"
    assert reasons["E2"] == ""


def test_anomaly_scores_are_deterministic_and_bounded():
    rng = np.random.default_rng(1)
    feats = pd.DataFrame(
        rng.normal(0, 1, (12, len(ANOMALY_FEATURES))), columns=ANOMALY_FEATURES, index=[f"E{i}" for i in range(12)]
    )
    feats.iloc[0] = 8.0  # an obvious outlier row
    a = anomaly_scores(feats, "isolation_forest", random_state=7)
    b = anomaly_scores(feats, "isolation_forest", random_state=7)
    assert np.allclose(a, b)
    assert ((a >= 0) & (a <= 100)).all()
    assert a.iloc[0] == a.max()


def test_anomaly_lof_and_guards():
    rng = np.random.default_rng(2)
    feats = pd.DataFrame(rng.normal(0, 1, (10, 4)), columns=ANOMALY_FEATURES[:4], index=list("abcdefghij"))
    lof = anomaly_scores(feats, "lof")
    assert lof.notna().all()
    assert anomaly_scores(feats, "none").isna().all()
    assert anomaly_scores(feats.iloc[:2], "isolation_forest").isna().all()
    assert anomaly_scores(pd.DataFrame({"unrelated": range(10)}), "isolation_forest").isna().all()


def test_robust_standardize_drops_constant_and_sparse_columns():
    df = pd.DataFrame(
        {"const": [1.0] * 6, "sparse": [1.0, np.nan, np.nan, np.nan, np.nan, 3.0], "ok": [1, 2, 3, 4, 5, 60]}
    )
    out = robust_standardize(df)
    assert "const" not in out.columns and "sparse" not in out.columns
    assert out["ok"].abs().max() <= 10


def test_scorecard_without_enumerators_is_empty():
    cfg = make_cfg(roles={"submission_id": "_uuid"})
    df = survey_frame(3, ("E1",))
    ctx = build_context(df, cfg, run_id="t")
    card, risk = build_scorecard(ctx, empty_metrics(), pd.DataFrame(), cfg.scoring)
    assert card.empty and risk.empty


def test_scorecard_flags_enumerator_with_strong_rule_signal():
    df = survey_frame(6, ("E1", "E2"))
    cfg = make_cfg(scoring={"flag_threshold": 10, "anomaly_method": "none"})
    ctx = build_context(df, cfg, run_id="t")
    metrics = pd.DataFrame(
        [
            {"check_id": "duration", "enumerator": "E1", "metric": "short_duration_share", "value": 0.9},
            {"check_id": "duration", "enumerator": "E1", "metric": "median_duration_ratio", "value": 0.2},
            {"check_id": "duration", "enumerator": "E2", "metric": "short_duration_share", "value": 0.0},
            {"check_id": "duration", "enumerator": "E2", "metric": "median_duration_ratio", "value": 1.0},
        ]
    )
    card, risk = build_scorecard(ctx, metrics, pd.DataFrame(columns=["severity", "enumerator", "level"]), cfg.scoring)
    assert card.iloc[0]["enumerator"] == "E1"
    assert bool(card.iloc[0]["flagged"]) is True
    assert risk["check_id"].tolist() == ["enumerator_risk"]
    assert "Prioritise for supervision" in risk["message"].iloc[0]


def test_feature_table_pivots_metrics():
    m = pd.DataFrame({"check_id": ["a", "a"], "enumerator": ["E1", "E1"], "metric": ["x", "y"], "value": [1.0, 2.0]})
    table = feature_table(m)
    assert table.loc["E1", "x"] == 1.0 and table.loc["E1", "y"] == 2.0
    assert feature_table(empty_metrics()).empty
