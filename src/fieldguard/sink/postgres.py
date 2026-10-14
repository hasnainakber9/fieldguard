"""Optional PostgreSQL sink: schema ``fieldguard`` with runs, issues, scorecards and seen records.

* The URL comes from the environment variable named by ``sink.database_url_env`` (default FIELDGUARD_DATABASE_URL).
* Reachability is probed first; when the database is unreachable or the optional extra is missing the sink is
  skipped with a message that names only the exception type, never the URL or its password.
* Only redacted tables are written (see :mod:`fieldguard.privacy`).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fieldguard.config import FieldGuardConfig
from fieldguard.privacy import redact_issues, redact_scorecard

if TYPE_CHECKING:
    from fieldguard.pipeline import RunResult

_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass(frozen=True)
class SinkStatus:
    available: bool
    message: str


def database_url(env_name: str = "FIELDGUARD_DATABASE_URL") -> str | None:
    value = os.environ.get(env_name, "").strip()
    return value or None


def redact_url(url: str) -> str:
    """Scheme, user and host only; the password is replaced by ***."""
    try:
        from sqlalchemy.engine import make_url
    except ImportError:
        return "<database URL (password hidden)>"
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:  # noqa: BLE001 - never echo a URL that failed to parse
        return "<unparseable database URL>"


def _schema(cfg: FieldGuardConfig) -> str:
    name = cfg.sink.schema_name
    if not _SCHEMA.match(name):
        raise ValueError(f"sink.schema_name {name!r} must be a lower-case SQL identifier")
    return name


def check_reachable(url: str, timeout: float = 3.0) -> SinkStatus:
    """Try a trivial query. Never includes the URL or driver messages (which can echo host details)."""
    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        return SinkStatus(False, "PostgreSQL sink skipped: install the extra with pip install 'fieldguard[postgres]'")
    engine = None
    try:
        engine = create_engine(url, connect_args={"connect_timeout": max(1, int(timeout))})
        with engine.connect() as conn:
            conn.execute(text("select 1"))
        return SinkStatus(True, "PostgreSQL reachable")
    except Exception as exc:  # noqa: BLE001 - any failure means "skip the sink"
        return SinkStatus(False, f"PostgreSQL sink skipped: database unreachable ({type(exc).__name__})")
    finally:
        if engine is not None:
            engine.dispose()


def _ddl(schema: str) -> list[str]:
    return [
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""CREATE TABLE IF NOT EXISTS {schema}.runs (
            run_id text PRIMARY KEY, project text NOT NULL, mode text, started_at timestamptz,
            finished_at timestamptz, synthetic boolean NOT NULL DEFAULT false, n_records integer,
            n_new_records integer, n_enumerators integer, n_issues integer, fieldguard_version text,
            summary_json text)""",
        f"""CREATE TABLE IF NOT EXISTS {schema}.issues (
            id bigserial PRIMARY KEY, run_id text NOT NULL REFERENCES {schema}.runs(run_id) ON DELETE CASCADE,
            check_id text, severity text, level text, record_id text, respondent_id text, enumerator text,
            survey_date text, variable text, value text, metric double precision, message text)""",
        f"CREATE INDEX IF NOT EXISTS issues_run_idx ON {schema}.issues (run_id)",
        f"CREATE INDEX IF NOT EXISTS issues_check_idx ON {schema}.issues (check_id, severity)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.scorecards (
            run_id text NOT NULL REFERENCES {schema}.runs(run_id) ON DELETE CASCADE, enumerator text NOT NULL,
            n_interviews integer, rule_score double precision, anomaly_score double precision,
            risk_score double precision, risk_rank integer, flagged boolean, issues_critical integer,
            issues_warning integer, issues_info integer, top_reasons text, PRIMARY KEY (run_id, enumerator))""",
        f"""CREATE TABLE IF NOT EXISTS {schema}.seen_records (
            project text NOT NULL, record_id text NOT NULL, first_run_id text,
            PRIMARY KEY (project, record_id))""",
    ]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def write_run(result: RunResult, cfg: FieldGuardConfig, url: str, salt: str | None = None) -> dict[str, int]:
    """Insert one run with its issues and scorecard. Returns row counts. Assumes the database is reachable."""
    from sqlalchemy import create_engine, text

    schema = _schema(cfg)
    issues = redact_issues(result.issues, cfg, salt)
    card = redact_scorecard(result.scorecard, cfg, salt) if not result.scorecard.empty else result.scorecard
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            for stmt in _ddl(schema):
                conn.execute(text(stmt))
            conn.execute(
                text(
                    f"""INSERT INTO {schema}.runs (run_id, project, mode, started_at, finished_at, synthetic, n_records,
                    n_new_records, n_enumerators, n_issues, fieldguard_version, summary_json)
                    VALUES (:run_id, :project, :mode, :started_at, :finished_at, :synthetic, :n_records,
                    :n_new_records, :n_enumerators, :n_issues, :version, :summary)
                    ON CONFLICT (run_id) DO NOTHING"""
                ),
                {
                    "run_id": result.run_id,
                    "project": result.project,
                    "mode": result.mode,
                    "started_at": result.started_at,
                    "finished_at": result.finished_at,
                    "synthetic": bool(result.synthetic),
                    "n_records": int(result.n_records),
                    "n_new_records": int(result.n_new_records),
                    "n_enumerators": int(result.n_enumerators),
                    "n_issues": int(len(issues)),
                    "version": _version(),
                    "summary": _json_summary(result),
                },
            )
            rows_i = [{k: _clean(v) for k, v in r.items()} for r in issues.to_dict(orient="records")]
            if rows_i:
                conn.execute(
                    text(
                        f"""INSERT INTO {schema}.issues (run_id, check_id, severity, level, record_id, respondent_id,
                        enumerator, survey_date, variable, value, metric, message)
                        VALUES (:run_id, :check_id, :severity, :level, :record_id, :respondent_id, :enumerator,
                        :survey_date, :variable, :value, :metric, :message)"""
                    ),
                    [{**r, "run_id": result.run_id} for r in rows_i],
                )
            rows_s = []
            if not card.empty:
                for r in card.to_dict(orient="records"):
                    rows_s.append(
                        {
                            "run_id": result.run_id,
                            "enumerator": str(r["enumerator"]),
                            "n_interviews": _clean(r.get("n_interviews")),
                            "rule_score": _clean(r.get("rule_score")),
                            "anomaly_score": _clean(r.get("anomaly_score")),
                            "risk_score": _clean(r.get("risk_score")),
                            "risk_rank": _clean(r.get("risk_rank")),
                            "flagged": bool(r.get("flagged", False)),
                            "issues_critical": _clean(r.get("issues_critical")),
                            "issues_warning": _clean(r.get("issues_warning")),
                            "issues_info": _clean(r.get("issues_info")),
                            "top_reasons": _clean(r.get("top_reasons")),
                        }
                    )
            if rows_s:
                conn.execute(
                    text(
                        f"""INSERT INTO {schema}.scorecards (run_id, enumerator, n_interviews, rule_score, anomaly_score,
                        risk_score, risk_rank, flagged, issues_critical, issues_warning, issues_info, top_reasons)
                        VALUES (:run_id, :enumerator, :n_interviews, :rule_score, :anomaly_score, :risk_score,
                        :risk_rank, :flagged, :issues_critical, :issues_warning, :issues_info, :top_reasons)
                        ON CONFLICT (run_id, enumerator) DO NOTHING"""
                    ),
                    rows_s,
                )
    finally:
        engine.dispose()
    return {"runs": 1, "issues": len(rows_i), "scorecards": len(rows_s)}


def _version() -> str:
    from fieldguard import __version__

    return __version__


def _json_summary(result: RunResult) -> str:
    import json

    return json.dumps(result.summary(), default=str, ensure_ascii=False)


__all__ = ["SinkStatus", "check_reachable", "database_url", "redact_url", "write_run"]
