"""Timezone-aware timestamp parsing for Kobo/ODK/SurveyCTO exports.

* Kobo/ODK ``start``/``end`` carry an offset (``2025-03-01T09:12:00.000+03:00``).
* Kobo ``_submission_time`` is naive UTC; ODK ``SubmissionDate`` ends in ``Z``.
* SurveyCTO exports naive local times such as ``Mar 2, 2025 11:23:45 AM``.

Everything is converted to the project timezone. Naive values are localised to ``naive_tz``.
"""

from __future__ import annotations

import warnings

import pandas as pd


def _localize(ts: pd.Series, tz: str, naive_tz: str) -> pd.Series:
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(naive_tz, ambiguous="NaT", nonexistent="shift_forward")
    return ts.dt.tz_convert(tz)


def parse_timestamps(values: pd.Series, tz: str = "UTC", naive_tz: str | None = None) -> pd.Series:
    """Parse a column of timestamps into tz-aware datetimes in ``tz`` (unparseable -> NaT)."""
    naive_tz = naive_tz or tz
    s = pd.Series(values, copy=False)
    if s.empty:
        return pd.Series(pd.DatetimeIndex([], tz=tz), index=s.index, dtype=f"datetime64[us, {tz}]")
    if pd.api.types.is_datetime64_any_dtype(s):
        return _localize(s, tz, naive_tz)
    text = s.astype("str").where(s.notna()).str.strip()
    text = text.mask(text == "")
    if text.isna().all():
        return pd.Series(pd.NaT, index=s.index, dtype=f"datetime64[us, {tz}]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
            if parsed.dtype == object:  # mixed offsets -> python objects
                raise ValueError("mixed offsets")
            return _localize(parsed, tz, naive_tz)
        except (ValueError, TypeError):
            pass
    # Mixed aware/naive or mixed offsets: parse element-wise.
    out: list[pd.Timestamp] = []
    for v in text:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append(pd.NaT)
            continue
        try:
            ts = pd.Timestamp(v)
        except (ValueError, TypeError):
            out.append(pd.NaT)
            continue
        if ts is pd.NaT:
            out.append(pd.NaT)
            continue
        ts = ts.tz_localize(naive_tz) if ts.tzinfo is None else ts
        out.append(ts.tz_convert(tz))
    return pd.Series(pd.to_datetime(pd.Series(out, dtype="object"), utc=True), index=s.index).dt.tz_convert(tz)


def to_local_dates(ts: pd.Series) -> pd.Series:
    """ISO date strings (YYYY-MM-DD) of tz-aware timestamps, NaN where missing."""
    out = ts.dt.strftime("%Y-%m-%d")
    return out.astype("object").where(ts.notna(), None)


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse a 'today'-style date column to ISO strings."""
    s = pd.Series(values, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(s.astype("str").where(s.notna()), errors="coerce", format="mixed")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    out = parsed.dt.strftime("%Y-%m-%d")
    return out.astype("object").where(parsed.notna(), None)
