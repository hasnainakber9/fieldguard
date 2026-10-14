"""Write a run's results to disk: issues (CSV, Parquet), workbook (Excel), HTML report and a JSON summary.

Every file is redacted (see :mod:`fieldguard.privacy`). Runs on synthetic data write files whose names start
with ``synthetic_`` and carry the synthetic label in the summary, the workbook and the HTML banner.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from fieldguard import __version__
from fieldguard.config import FieldGuardConfig
from fieldguard.privacy import needs_salt, redact_issues, redact_metrics, redact_scorecard, resolve_salt

if TYPE_CHECKING:
    from fieldguard.pipeline import RunResult

log = logging.getLogger(__name__)

ALL_FORMATS = ("csv", "parquet", "excel", "html")
EXCEL_SHEET_MAX = 31
ETHICS_NOTE = (
    "Flags are leads for supervisory review, not proof of misconduct. They must not trigger automated sanctions. "
    "Names, phone numbers and precise GPS are redacted in this file."
)


def resolve_output_dir(cfg: FieldGuardConfig, run_id: str, override: str | Path | None = None) -> Path:
    """Output folder: explicit override, then ``FIELDGUARD_REPORTS_DIR/<project>/<run>``, then ``outputs.dir``."""
    if override:
        return Path(override)
    env_root = os.environ.get("FIELDGUARD_REPORTS_DIR", "").strip()
    if env_root:
        return Path(env_root) / (cfg.project.slug or "project") / run_id
    template = cfg.outputs.dir.format(project=cfg.project.slug or "project", run_id=run_id)
    return cfg.resolve_path(template)


def _prefix(result: RunResult) -> str:
    return "synthetic_" if result.synthetic else ""


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return path


def write_parquet(df: pd.DataFrame, path: Path, metadata: dict[str, str] | None = None) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the 'parquet' extra
        raise ImportError("Parquet output needs pyarrow: pip install 'fieldguard[parquet]'") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df.astype(object).where(df.notna(), None), preserve_index=False)
    if metadata:
        table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def _readme_rows(result: RunResult, cfg: FieldGuardConfig, n_sheet_truncated: list[str]) -> pd.DataFrame:
    from fieldguard.synthetic import SYNTHETIC_LABEL

    rows: list[tuple[str, str]] = [
        ("Project", result.project),
        ("Run id", result.run_id),
        ("Mode", result.mode),
        ("FieldGuard version", __version__),
        ("Started (UTC)", result.started_at.isoformat(timespec="seconds")),
        ("Finished (UTC)", result.finished_at.isoformat(timespec="seconds")),
        ("Platform", result.platform),
        ("Data source", result.data_source),
        ("Records", str(result.n_records)),
        ("New records in this run", str(result.n_new_records)),
        ("Enumerators", str(result.n_enumerators)),
        ("Issues", str(len(result.issues))),
        ("Synthetic", "YES - " + SYNTHETIC_LABEL if result.synthetic else "no"),
        ("Ethics", ETHICS_NOTE),
        ("Config hash", _config_hash(cfg)),
    ]
    if n_sheet_truncated:
        rows.append(
            ("Truncated sheets", ", ".join(n_sheet_truncated) + f" (limit {cfg.outputs.max_rows_per_sheet} rows)")
        )
    return pd.DataFrame(rows, columns=["field", "value"])


def _config_hash(cfg: FieldGuardConfig) -> str:
    import hashlib

    from fieldguard.config import dump_config

    return hashlib.sha256(dump_config(cfg).encode("utf-8")).hexdigest()[:12]


def _sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:EXCEL_SHEET_MAX] or "sheet"
    cand, i = base, 2
    while cand.lower() in used:
        suffix = f"_{i}"
        cand = base[: EXCEL_SHEET_MAX - len(suffix)] + suffix
        i += 1
    used.add(cand.lower())
    return cand


def write_workbook(
    result: RunResult, cfg: FieldGuardConfig, path: Path, issues: pd.DataFrame, card: pd.DataFrame, status: pd.DataFrame
) -> Path:
    """Excel workbook: README, scorecard, one sheet per check, check status."""
    try:
        import xlsxwriter  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the 'excel' extra
        raise ImportError("Excel output needs xlsxwriter: pip install 'fieldguard[excel]'") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    limit = cfg.outputs.max_rows_per_sheet
    truncated: list[str] = []
    used: set[str] = {"readme"}
    with pd.ExcelWriter(tmp, engine="xlsxwriter") as writer:
        book = writer.book
        head = book.add_format({"bold": True, "bg_color": "#ECECE6", "border": 1, "text_wrap": True, "valign": "top"})
        sheets: list[tuple[str, pd.DataFrame]] = [("scorecard", card)]
        # One sheet per check that ran, including checks with no issues (empty sheet with the issue columns).
        by_check = {str(k): sub for k, sub in issues.groupby("check_id", sort=True)} if not issues.empty else {}
        ran = [str(c) for c in status["check_id"]] if "check_id" in status.columns else []
        for check_id in sorted(set(ran) | set(by_check)):
            sub = by_check.get(check_id, issues.iloc[0:0])
            sheets.append((check_id, sub.drop(columns=["run_id"], errors="ignore")))
        sheets.append(("check_status", status))
        for name, df in sheets:
            sheet = _sheet_name(name, used)
            if len(df) > limit:
                truncated.append(sheet)
                df = df.head(limit)
            df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            for j, col in enumerate(df.columns):
                ws.write(0, j, str(col), head)
                lens = df[col].astype(str).str.len()
                q90 = float(lens.quantile(0.9)) if len(df) and lens.notna().any() else 0.0
                width = min(max(10, len(str(col)) + 2, int(q90)), 60)
                ws.set_column(j, j, width)
            ws.freeze_panes(1, 0)
            if len(df.columns):
                ws.autofilter(0, 0, max(len(df), 1), len(df.columns) - 1)
        readme = _readme_rows(result, cfg, truncated)
        readme.to_excel(writer, sheet_name=_sheet_name("README", set()), index=False)
        ws = writer.sheets["README"]
        ws.set_column(0, 0, 26)
        ws.set_column(1, 1, 110, book.add_format({"text_wrap": True}))
        if result.synthetic:
            ws.write(len(readme) + 2, 0, "SYNTHETIC DATA", book.add_format({"bold": True, "font_color": "#B3261E"}))
    tmp.replace(path)
    return path


def write_run_outputs(
    result: RunResult,
    cfg: FieldGuardConfig,
    out_dir: Path,
    formats: Iterable[str] | None = None,
    *,
    salt: str | None = None,
) -> dict[str, Path]:
    """Write all requested formats into ``out_dir``. Returns a mapping of artefact name to path."""
    from fieldguard.reports.html import write_html

    fmts = set(formats) if formats is not None else set(cfg.outputs.formats)
    unknown = fmts - set(ALL_FORMATS)
    if unknown:
        raise ValueError(f"unknown output format(s): {sorted(unknown)}; choose from {ALL_FORMATS}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = _prefix(result)
    salt = salt or (resolve_salt(cfg) if needs_salt(cfg) else None)
    issues = redact_issues(result.issues, cfg, salt)
    card = redact_scorecard(result.scorecard, cfg, salt) if not result.scorecard.empty else result.scorecard
    metrics = redact_metrics(result.metrics, cfg, salt)
    status = result.check_status
    written: dict[str, Path] = {}
    meta = {
        "fieldguard_version": __version__,
        "run_id": result.run_id,
        "synthetic": str(result.synthetic).lower(),
        "note": ETHICS_NOTE,
    }

    if "csv" in fmts:
        written["issues_csv"] = write_csv(issues, out_dir / f"{pre}issues.csv")
        written["scorecard_csv"] = write_csv(card, out_dir / f"{pre}scorecard.csv")
        written["metrics_csv"] = write_csv(metrics, out_dir / f"{pre}metrics.csv")
        written["check_status_csv"] = write_csv(status, out_dir / f"{pre}check_status.csv")
    if "parquet" in fmts:
        try:
            written["issues_parquet"] = write_parquet(issues, out_dir / f"{pre}issues.parquet", meta)
            written["scorecard_parquet"] = write_parquet(card.astype(object), out_dir / f"{pre}scorecard.parquet", meta)
        except ImportError as exc:  # the parquet extra is optional; the other formats still get written
            log.warning("Parquet outputs skipped: %s", exc)
    if "excel" in fmts:
        written["workbook"] = write_workbook(
            result, cfg, out_dir / f"{pre}fieldguard_workbook.xlsx", issues, card, status
        )
    if "html" in fmts:
        written["html_report"] = write_html(result, cfg, out_dir / f"{pre}report.html", salt=salt)
    summary = result.summary()
    summary["synthetic_label"] = "SYNTHETIC - generated data, not a real survey" if result.synthetic else ""
    summary["config_hash"] = _config_hash(cfg)
    written["summary_json"] = _atomic_write_text(
        out_dir / f"{pre}run_summary.json", json.dumps(summary, indent=2, default=str, ensure_ascii=False)
    )
    return written


__all__ = [
    "ALL_FORMATS",
    "ETHICS_NOTE",
    "resolve_output_dir",
    "write_csv",
    "write_parquet",
    "write_run_outputs",
    "write_workbook",
]
