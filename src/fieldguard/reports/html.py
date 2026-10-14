"""Self-contained HTML report (Jinja2 template plus inline SVG charts). No scripts, fonts or images are loaded."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from fieldguard import __version__
from fieldguard.checks.registry import get_registry
from fieldguard.config import FieldGuardConfig, dump_config
from fieldguard.models import SEVERITY_RANK
from fieldguard.privacy import redact_issues, redact_scorecard
from fieldguard.reports.svg import bar_scores, stacked_hbar

if TYPE_CHECKING:
    from fieldguard.pipeline import RunResult

TEMPLATE_DIR = Path(__file__).parent / "templates"
SEVERITIES = ("critical", "warning", "info")


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _scorecard_rows(card: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if card.empty:
        return rows
    for r in card.head(limit).itertuples(index=False):
        anomaly = getattr(r, "anomaly_score", float("nan"))
        rows.append(
            {
                "enumerator": _clean(r.enumerator),
                "n_interviews": int(r.n_interviews),
                "risk_score": float(r.risk_score),
                "rule_score": float(r.rule_score),
                "anomaly_text": "n/a" if pd.isna(anomaly) else f"{float(anomaly):.1f}",
                "flagged": bool(r.flagged),
                "top_reasons": _clean(r.top_reasons),
            }
        )
    return rows


def _issue_blocks(issues: pd.DataFrame, max_rows: int) -> list[dict[str, Any]]:
    registry = get_registry()
    blocks: list[dict[str, Any]] = []
    if issues.empty:
        return blocks
    ordered = issues.assign(_rank=issues["severity"].map(SEVERITY_RANK).fillna(0))
    order = (
        ordered.groupby("check_id")["_rank"]
        .max()
        .reset_index()
        .assign(_n=ordered.groupby("check_id").size().to_numpy())
        .sort_values(["_rank", "_n"], ascending=False)["check_id"]
        .tolist()
    )
    for check_id in order:
        sub = issues[issues["check_id"] == check_id]
        spec = registry.get(check_id)
        rows = [
            {
                "severity": _clean(r.severity),
                "enumerator": _clean(r.enumerator),
                "record_id": _clean(r.record_id),
                "variable": _clean(r.variable),
                "message": _clean(r.message),
            }
            for r in sub.head(max_rows).itertuples(index=False)
        ]
        blocks.append(
            {
                "check_id": check_id,
                "title": spec.title if spec else check_id,
                "description": spec.description if spec else "Dataset-level notes and configuration messages.",
                "references": "; ".join(spec.references) if spec else "",
                "total": int(len(sub)),
                "rows": rows,
            }
        )
    return blocks


def build_context(
    result: RunResult, cfg: FieldGuardConfig, *, max_per_check: int | None = None, salt: str | None = None
) -> dict[str, Any]:
    from fieldguard.synthetic import SYNTHETIC_LABEL

    limit = cfg.outputs.max_issues_per_check_html if max_per_check is None else max_per_check
    issues = redact_issues(result.issues, cfg, salt)
    card = redact_scorecard(result.scorecard, cfg, salt) if not result.scorecard.empty else result.scorecard
    counts = {s: int((issues["severity"] == s).sum()) for s in SEVERITIES}
    if not card.empty and "risk_score" in card.columns:
        chart_rows = [
            (_clean(r.enumerator), float(r.risk_score), bool(r.flagged)) for r in card.head(40).itertuples(index=False)
        ]
        risk_chart = bar_scores(chart_rows)
        n_flagged = int(card["flagged"].sum())
    else:
        risk_chart = "<p class='muted'>No enumerator scores: configure the enumerator role and scoring.</p>"
        n_flagged = 0
    per_check: list[tuple[str, dict[str, int]]] = []
    if not issues.empty:
        table = issues.pivot_table(index="check_id", columns="severity", values="run_id", aggfunc="count", fill_value=0)
        for check_id in table.index:
            per_check.append((str(check_id), {s: int(table.loc[check_id].get(s, 0)) for s in SEVERITIES}))
        per_check.sort(key=lambda item: -sum(item[1].values()))
    issue_chart = stacked_hbar(per_check)
    config_hash = hashlib.sha256(dump_config(cfg).encode("utf-8")).hexdigest()[:12]
    return {
        "version": __version__,
        "project": result.project,
        "run_id": result.run_id,
        "mode": result.mode,
        "finished_at": result.finished_at.strftime("%Y-%m-%d %H:%M UTC"),
        "synthetic": result.synthetic,
        "synthetic_label": SYNTHETIC_LABEL,
        "n_records": result.n_records,
        "n_new_records": result.n_new_records,
        "n_enumerators": result.n_enumerators,
        "n_issues": int(len(issues)),
        "sev": counts,
        "n_flagged": n_flagged,
        "flag_threshold": f"{cfg.scoring.flag_threshold:g}",
        "platform": result.platform,
        "data_source": result.data_source,
        "risk_chart": risk_chart,
        "scorecard": _scorecard_rows(card, 60),
        "issue_chart": issue_chart,
        "check_status": result.check_status.to_dict(orient="records"),
        "max_per_check": limit,
        "blocks": _issue_blocks(issues, limit),
        "config_hash": config_hash,
    }


def render_html(
    result: RunResult, cfg: FieldGuardConfig, *, max_per_check: int | None = None, salt: str | None = None
) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError as exc:  # pragma: no cover - depends on the 'html' extra
        raise ImportError("The HTML report needs Jinja2: pip install 'fieldguard[html]'") from exc
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("j2", "html")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("report.html.j2")
    return template.render(**build_context(result, cfg, max_per_check=max_per_check, salt=salt))


def write_html(result: RunResult, cfg: FieldGuardConfig, path: Path, **kwargs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_html(result, cfg, **kwargs), encoding="utf-8")
    tmp.replace(path)
    return path


__all__ = ["build_context", "render_html", "write_html"]
