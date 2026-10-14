"""Enumerator scorecard: features, transparent rule score, anomaly score and combined risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fieldguard.checks.context import UNKNOWN_ENUMERATOR, CheckContext
from fieldguard.config import ScoringConfig
from fieldguard.models import Level, Severity, coerce_issues, empty_issues
from fieldguard.scoring.anomaly import anomaly_scores
from fieldguard.scoring.rules import rule_scores, top_reasons

SCORECARD_BASE_COLUMNS = [
    "enumerator",
    "n_interviews",
    "n_clusters",
    "first_date",
    "last_date",
    "issues_critical",
    "issues_warning",
    "issues_info",
    "rule_score",
    "anomaly_score",
    "risk_score",
    "risk_rank",
    "flagged",
    "top_reasons",
]


def feature_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return metrics.pivot_table(index="enumerator", columns="metric", values="value", aggfunc="mean")


def build_scorecard(
    ctx: CheckContext, metrics: pd.DataFrame, issues: pd.DataFrame, cfg: ScoringConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (scorecard, enumerator_risk issues)."""
    if not ctx.has_enumerators or ctx.n == 0:
        return pd.DataFrame(columns=SCORECARD_BASE_COLUMNS), empty_issues()
    enums = ctx.enumerator[ctx.enumerator != UNKNOWN_ENUMERATOR]
    base = pd.DataFrame({"n_interviews": enums.value_counts()})
    base.index = base.index.astype(str)
    base.index.name = "enumerator"
    if ctx.cluster is not None:
        base["n_clusters"] = ctx.cluster.groupby(ctx.enumerator).nunique().reindex(base.index)
    else:
        base["n_clusters"] = np.nan
    if ctx.survey_date is not None:
        dates = pd.DataFrame({"d": ctx.survey_date, "e": ctx.enumerator}).dropna()
        base["first_date"] = dates.groupby("e")["d"].min().reindex(base.index)
        base["last_date"] = dates.groupby("e")["d"].max().reindex(base.index)
    else:
        base["first_date"] = None
        base["last_date"] = None
    for sev in ("critical", "warning", "info"):
        sub = issues[(issues["severity"] == sev) & issues["enumerator"].notna()] if not issues.empty else issues
        counts = sub.groupby("enumerator").size() if not sub.empty else pd.Series(dtype="float64")
        base[f"issues_{sev}"] = counts.reindex(base.index).fillna(0).astype(int)

    feats = feature_table(metrics).reindex(base.index)
    rule, points = (
        rule_scores(feats, cfg)
        if not feats.empty
        else (pd.Series(0.0, index=base.index), pd.DataFrame(index=base.index))
    )
    base["rule_score"] = rule.reindex(base.index).fillna(0.0).round(1)
    eligible = base["n_interviews"] >= cfg.min_interviews
    anomaly = pd.Series(np.nan, index=base.index)
    if cfg.anomaly_method != "none" and eligible.sum() >= cfg.min_enumerators_for_anomaly and not feats.empty:
        anomaly.loc[eligible] = anomaly_scores(feats[eligible], cfg.anomaly_method, cfg.random_state)
    base["anomaly_score"] = anomaly.round(1)
    a = cfg.anomaly_weight
    risk = base["rule_score"] * (1 - a) + anomaly.fillna(base["rule_score"]) * a
    base["risk_score"] = risk.round(1)
    base["risk_rank"] = base["risk_score"].rank(ascending=False, method="min").astype(int)
    base["flagged"] = (base["risk_score"] >= cfg.flag_threshold) & eligible
    base["top_reasons"] = top_reasons(points.reindex(base.index)) if not points.empty else ""
    card = base.reset_index().rename(columns={"index": "enumerator"})
    card = pd.concat([card, feats.reset_index(drop=True).add_prefix("m_")], axis=1)
    card = card.sort_values(["risk_score", "enumerator"], ascending=[False, True]).reset_index(drop=True)

    rows = []
    for r in card.itertuples(index=False):
        if not r.flagged:
            continue
        sev = Severity.CRITICAL if r.risk_score >= min(100.0, 2 * cfg.flag_threshold) else Severity.WARNING
        anom = "n/a" if pd.isna(r.anomaly_score) else f"{r.anomaly_score:.0f}"
        rows.append(
            {
                "check_id": "enumerator_risk",
                "severity": str(sev),
                "level": Level.ENUMERATOR.value,
                "enumerator": r.enumerator,
                "variable": None,
                "value": f"{r.risk_score:.1f}",
                "metric": float(r.risk_score),
                "message": f"Risk score {r.risk_score:.0f}/100 (rules {r.rule_score:.0f}, anomaly {anom}); "
                f"signals: {r.top_reasons or 'multivariate pattern'}. Prioritise for supervision/back-checks.",
            }
        )
    risk_issues = coerce_issues(pd.DataFrame(rows)) if rows else empty_issues()
    return card, risk_issues
