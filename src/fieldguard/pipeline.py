"""Run all registered checks, score enumerators and assemble a RunResult."""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from fieldguard import __version__
from fieldguard.checks.context import CheckContext, build_context
from fieldguard.checks.registry import get_registry
from fieldguard.config import FieldGuardConfig
from fieldguard.models import ISSUE_COLUMNS, ROW_COLUMN, SEVERITY_RANK, Level, Severity, coerce_issues, empty_issues
from fieldguard.scoring.scorecard import build_scorecard

log = logging.getLogger(__name__)

SYNTHETIC_COLUMN = "_synthetic"


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    return f"fg-{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


@dataclass
class RunResult:
    run_id: str
    project: str
    started_at: datetime
    finished_at: datetime
    issues: pd.DataFrame
    scorecard: pd.DataFrame
    metrics: pd.DataFrame
    check_status: pd.DataFrame
    n_records: int
    n_new_records: int
    n_enumerators: int
    synthetic: bool
    platform: str
    data_source: str
    config: FieldGuardConfig
    mode: str = "full"
    new_record_ids: list[str] = field(default_factory=list)
    context: CheckContext | None = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        sev = self.issues["severity"].value_counts().to_dict() if not self.issues.empty else {}
        flagged = int(self.scorecard["flagged"].sum()) if "flagged" in self.scorecard else 0
        return {
            "run_id": self.run_id,
            "project": self.project,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "mode": self.mode,
            "fieldguard_version": __version__,
            "platform": self.platform,
            "data_source": self.data_source,
            "synthetic": self.synthetic,
            "n_records": self.n_records,
            "n_new_records": self.n_new_records,
            "n_enumerators": self.n_enumerators,
            "n_issues": len(self.issues),
            "issues_by_severity": {k: int(sev.get(k, 0)) for k in ("critical", "warning", "info")},
            "n_flagged_enumerators": flagged,
            "checks": self.check_status.to_dict(orient="records"),
        }


def finalize_issues(issues: pd.DataFrame, ctx: CheckContext) -> pd.DataFrame:
    """Fill record_id / respondent / enumerator / date from the row reference and drop helper columns."""
    if issues.empty:
        return empty_issues()
    out = coerce_issues(issues)
    if ROW_COLUMN in out.columns:
        rows = out[ROW_COLUMN]
        has = rows.notna()
        if has.any():
            pos = rows[has].astype(int).to_numpy()

            def fill(col: str, source: pd.Series | None) -> None:
                if source is None:
                    return
                vals = pd.Series(source.iloc[pos].to_numpy(), index=out.index[has]).astype("object")
                current = out.loc[has, col]
                out.loc[has, col] = current.where(current.notna(), vals)

            fill("record_id", ctx.record_id)
            fill("respondent_id", ctx.respondent_id)
            fill("enumerator", ctx.enumerator if ctx.has_enumerators else None)
            fill("survey_date", ctx.survey_date)
        out = out.drop(columns=[ROW_COLUMN])
    out["run_id"] = ctx.run_id
    return coerce_issues(out)[ISSUE_COLUMNS]


def sort_issues(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return issues
    rank = issues["severity"].map(SEVERITY_RANK).fillna(0)
    order = issues.assign(_rank=-rank).sort_values(
        ["_rank", "check_id", "enumerator", "record_id"], na_position="last", kind="stable"
    )
    return order.drop(columns="_rank").reset_index(drop=True)


def run_checks(
    data: pd.DataFrame,
    config: FieldGuardConfig,
    *,
    run_id: str | None = None,
    only: Iterable[str] | None = None,
    skip: Iterable[str] | None = None,
    strict: bool = False,
    seen_ids: set[str] | None = None,
    data_source: str = "file",
) -> RunResult:
    """Run every enabled check on ``data`` and score enumerators.

    ``seen_ids`` enables incremental mode: checks still see the full data (for context such as duplicates
    and distributions) but record-level issues are only reported for submissions not seen before.
    """
    started = datetime.now(UTC)
    run_id = run_id or new_run_id(started)
    ctx = build_context(data, config, run_id)
    only_set = set(only) if only else None
    skip_set = set(skip or [])
    frames: list[pd.DataFrame] = []
    status = []
    for check_id, spec in get_registry().items():
        params = config.params_for(check_id)
        if (only_set is not None and check_id not in only_set) or check_id in skip_set or not params.enabled:
            status.append({"check_id": check_id, "status": "skipped", "n_issues": 0, "seconds": 0.0, "message": ""})
            continue
        t0 = time.perf_counter()
        try:
            out = spec.func(ctx, params)
            frames.append(out)
            status.append(
                {
                    "check_id": check_id,
                    "status": "ok",
                    "n_issues": len(out),
                    "seconds": round(time.perf_counter() - t0, 3),
                    "message": "",
                }
            )
        except Exception as exc:
            if strict:
                raise
            log.exception("check %s failed", check_id)
            frames.append(
                ctx.dataset_issue(check_id, Severity.INFO, f"Check failed and was skipped: {type(exc).__name__}: {exc}")
            )
            status.append(
                {
                    "check_id": check_id,
                    "status": "error",
                    "n_issues": 0,
                    "seconds": round(time.perf_counter() - t0, 3),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
    for note in ctx.setup_notes:
        frames.append(ctx.dataset_issue("setup", Severity.WARNING, f"Configuration: {note}"))
    if ctx.n == 0:
        frames.append(ctx.dataset_issue("setup", Severity.INFO, "No records in the data"))
    if not ctx.has_enumerators:
        frames.append(
            ctx.dataset_issue(
                "setup", Severity.INFO, "No enumerator role configured: enumerator-level checks and scoring are limited"
            )
        )
    frames = [f for f in frames if f is not None and not f.empty]
    issues = finalize_issues(pd.concat(frames, ignore_index=True), ctx) if frames else empty_issues()
    metrics = ctx.metrics_frame()

    new_ids: list[str] = list(ctx.record_id.astype(str))
    mode = "full"
    if seen_ids is not None:
        mode = "incremental"
        is_new = ~ctx.record_id.astype(str).isin(seen_ids)
        new_ids = list(ctx.record_id[is_new].astype(str))
        if not issues.empty:
            rec = issues["level"] == Level.RECORD.value
            issues = issues[~rec | issues["record_id"].isin(set(new_ids))].reset_index(drop=True)

    if config.scoring.enabled:
        scorecard, risk_issues = build_scorecard(ctx, metrics, issues, config.scoring)
        if not risk_issues.empty:
            risk_issues["run_id"] = run_id
            issues = pd.concat([issues, risk_issues[ISSUE_COLUMNS]], ignore_index=True)
    else:
        scorecard = pd.DataFrame()
    issues = sort_issues(coerce_issues(issues)[ISSUE_COLUMNS])
    synthetic = bool(config.synthetic or SYNTHETIC_COLUMN in ctx.data.columns)
    n_enum = int(ctx.enumerator.nunique()) if ctx.has_enumerators else 0
    return RunResult(
        run_id=run_id,
        project=config.project.slug or "project",
        started_at=started,
        finished_at=datetime.now(UTC),
        issues=issues,
        scorecard=scorecard,
        metrics=metrics,
        check_status=pd.DataFrame(status),
        n_records=ctx.n,
        n_new_records=len(new_ids),
        n_enumerators=n_enum,
        synthetic=synthetic,
        platform=ctx.platform,
        data_source=data_source,
        config=config,
        mode=mode,
        new_record_ids=new_ids,
        context=ctx,
    )


def issues_by_check(issues: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if issues.empty:
        return {}
    return {cid: g.reset_index(drop=True) for cid, g in issues.groupby("check_id", sort=True)}


def severity_counts(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return pd.DataFrame(columns=["check_id", "critical", "warning", "info", "total"])
    tab = issues.pivot_table(index="check_id", columns="severity", values="run_id", aggfunc="count", fill_value=0)
    for s in ("critical", "warning", "info"):
        if s not in tab.columns:
            tab[s] = 0
    tab = tab[["critical", "warning", "info"]].astype(int)
    tab["total"] = tab.sum(axis=1)
    return tab.sort_values("total", ascending=False).reset_index()


__all__ = ["RunResult", "finalize_issues", "issues_by_check", "new_run_id", "run_checks", "severity_counts"]
