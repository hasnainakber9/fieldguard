"""Core data model: the tidy issues table shared by every check, scorer and sink."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd


class Severity(StrEnum):
    """How urgently a flag needs supervisory review (not a verdict of misconduct)."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


SEVERITY_RANK: dict[str, int] = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


class Level(StrEnum):
    """Unit an issue refers to."""

    RECORD = "record"
    ENUMERATOR = "enumerator"
    VARIABLE = "variable"
    DATASET = "dataset"


#: Column order of the tidy issues table.
ISSUE_COLUMNS: list[str] = [
    "run_id",
    "check_id",
    "severity",
    "level",
    "record_id",
    "respondent_id",
    "enumerator",
    "survey_date",
    "variable",
    "value",
    "metric",
    "message",
]

_STRING_COLUMNS = [c for c in ISSUE_COLUMNS if c != "metric"]

#: Internal helper column (row position in the survey frame) used before finalisation.
ROW_COLUMN = "_row"

METRIC_COLUMNS: list[str] = ["check_id", "enumerator", "metric", "value"]


def empty_issues(with_row: bool = False) -> pd.DataFrame:
    """Return an empty issues table with the canonical columns and dtypes."""
    data: dict[str, pd.Series] = {c: pd.Series(dtype="object") for c in _STRING_COLUMNS}
    data["metric"] = pd.Series(dtype="float64")
    df = pd.DataFrame(data)[ISSUE_COLUMNS]
    if with_row:
        df[ROW_COLUMN] = pd.Series(dtype="float64")
    return df


def coerce_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an issues frame to canonical columns, order and dtypes."""
    out = df.copy()
    for col in ISSUE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    keep = ISSUE_COLUMNS + ([ROW_COLUMN] if ROW_COLUMN in out.columns else [])
    out = out[keep]
    out["metric"] = pd.to_numeric(out["metric"], errors="coerce").astype("float64")
    for col in _STRING_COLUMNS:
        s = out[col].astype("object")
        present = s.notna().to_numpy()
        values = np.full(len(s), None, dtype=object)
        if present.any():
            values[present] = [str(v) for v in s.to_numpy(dtype=object)[present]]
        out[col] = pd.Series(values, index=s.index, dtype="object")
    return out.reset_index(drop=True)


def empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "check_id": pd.Series(dtype="object"),
            "enumerator": pd.Series(dtype="object"),
            "metric": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )
