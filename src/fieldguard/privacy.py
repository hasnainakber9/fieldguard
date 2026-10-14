"""Privacy helpers for shared outputs.

Shared reports never carry names, phone numbers or precise GPS coordinates:

* columns listed in ``privacy.pii_columns`` are never quoted in issue values;
* phone-like digit runs are replaced in free text;
* coordinates in GPS-related issues are rounded to ``privacy.gps_decimals_in_reports`` decimals;
* respondent IDs and enumerator names can be pseudonymised with a salted hash
  (salt from the environment variable named in ``privacy.hash_salt_env``, never written to disk).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
from collections.abc import Iterable

import pandas as pd

from fieldguard.config import FieldGuardConfig
from fieldguard.models import ISSUE_COLUMNS

log = logging.getLogger(__name__)

REDACTED = "[redacted]"
PHONE_REDACTED = "[phone]"

#: A run of 9 to 15 digits (optionally grouped with spaces or dashes), e.g. 0712345678 or +254 712 345 678.
#: Dates such as 2025-03-05 contain only 8 digits and are not matched.
_PHONE = re.compile(r"(?<![\w.])\+?\d(?:[ -]?\d){8,14}(?![\w.])")
#: Decimal coordinates with at least four decimals.
_DECIMAL = re.compile(r"(?<![\w.])-?\d{1,3}\.\d{4,}(?![\w.])")
_GPS_HINTS = ("gps", "lat", "lon", "geopoint", "location", "coord")


def _last_segment(name: str) -> str:
    return re.split(r"[/\-]", name.strip())[-1].lower()


def is_pii_variable(variable: str | None, pii_names: set[str]) -> bool:
    if not variable:
        return False
    tokens = {_last_segment(t) for t in re.split(r",\s*", str(variable)) if t.strip()}
    return bool(tokens & pii_names)


def is_gps_variable(variable: str | None, check_id: str | None = None) -> bool:
    if check_id == "gps":
        return True
    low = (variable or "").lower()
    return any(h in low for h in _GPS_HINTS)


def redact_phone_numbers(text: str) -> str:
    return _PHONE.sub(PHONE_REDACTED, text)


def coarsen_coordinates(text: str, decimals: int) -> str:
    def repl(m: re.Match[str]) -> str:
        return f"{float(m.group(0)):.{decimals}f}"

    return _DECIMAL.sub(repl, text)


def pseudonym(value: str, salt: str, prefix: str = "id") -> str:
    digest = hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def needs_salt(cfg: FieldGuardConfig) -> bool:
    return bool(cfg.privacy.hash_respondent_ids or cfg.privacy.pseudonymize_enumerators)


def resolve_salt(cfg: FieldGuardConfig) -> str:
    """Salt for pseudonyms: from the environment (stable across runs) or random for this run only."""
    salt = os.environ.get(cfg.privacy.hash_salt_env, "")
    if not salt:
        log.warning("%s is not set: pseudonyms will not be reproducible across runs", cfg.privacy.hash_salt_env)
        salt = secrets.token_hex(16)
    return salt


def _pii_names(cfg: FieldGuardConfig) -> set[str]:
    return {_last_segment(c) for c in cfg.privacy.pii_columns}


def _map_text(series: pd.Series, fn) -> pd.Series:  # type: ignore[no-untyped-def]
    out = series.astype("object")
    mask = out.notna()
    out = out.copy()
    out[mask] = [fn(str(v)) for v in out[mask]]
    return out


def redact_issues(issues: pd.DataFrame, cfg: FieldGuardConfig, salt: str | None = None) -> pd.DataFrame:
    """Return a copy of the issues table that is safe to share (see module docstring)."""
    out = issues.copy()
    for col in ISSUE_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out.astype({c: "object" for c in ISSUE_COLUMNS if c != "metric"})
    if out.empty:
        return out[ISSUE_COLUMNS]
    pii = _pii_names(cfg)
    priv = cfg.privacy
    if priv.redact_pii_in_reports:
        is_pii = out["variable"].map(lambda v: is_pii_variable(v, pii)).astype(bool)
        out.loc[is_pii, "value"] = REDACTED
        out["message"] = _map_text(out["message"], redact_phone_numbers)
        out["value"] = _map_text(out["value"], redact_phone_numbers)
    gps_rows = out.apply(lambda r: is_gps_variable(r["variable"], r["check_id"]), axis=1).astype(bool)
    if gps_rows.any() and priv.gps_decimals_in_reports is not None:
        d = int(priv.gps_decimals_in_reports)
        for col in ("message", "value"):
            out.loc[gps_rows, col] = _map_text(out.loc[gps_rows, col], lambda t: coarsen_coordinates(t, d))
    if priv.hash_respondent_ids or priv.pseudonymize_enumerators:
        salt = salt or resolve_salt(cfg)
        if priv.hash_respondent_ids:
            for col in ("record_id", "respondent_id"):
                out[col] = out[col].map(lambda v: None if pd.isna(v) else pseudonym(str(v), salt, "resp"))
        if priv.pseudonymize_enumerators:
            out["enumerator"] = out["enumerator"].map(lambda v: None if pd.isna(v) else pseudonym(str(v), salt, "enum"))
    return out[ISSUE_COLUMNS].reset_index(drop=True)


def redact_scorecard(card: pd.DataFrame, cfg: FieldGuardConfig, salt: str | None = None) -> pd.DataFrame:
    out = card.copy()
    if out.empty:
        return out
    if cfg.privacy.pseudonymize_enumerators:
        salt = salt or resolve_salt(cfg)
        out["enumerator"] = out["enumerator"].map(lambda v: pseudonym(str(v), salt, "enum"))
    if "top_reasons" in out.columns and cfg.privacy.redact_pii_in_reports:
        out["top_reasons"] = _map_text(out["top_reasons"], redact_phone_numbers)
    return out


def redact_metrics(metrics: pd.DataFrame, cfg: FieldGuardConfig, salt: str | None = None) -> pd.DataFrame:
    out = metrics.copy()
    if out.empty or not cfg.privacy.pseudonymize_enumerators:
        return out
    salt = salt or resolve_salt(cfg)
    out["enumerator"] = out["enumerator"].map(lambda v: pseudonym(str(v), salt, "enum"))
    return out


def drop_columns(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    """Drop PII columns from a survey frame before it is written to a shared file."""
    keep = [c for c in df.columns if _last_segment(c) not in {_last_segment(n) for n in names}]
    return df[keep]


__all__ = [
    "PHONE_REDACTED",
    "REDACTED",
    "coarsen_coordinates",
    "drop_columns",
    "is_gps_variable",
    "is_pii_variable",
    "pseudonym",
    "redact_issues",
    "redact_metrics",
    "redact_phone_numbers",
    "redact_scorecard",
    "needs_salt",
    "resolve_salt",
]
