"""CheckContext: the prepared survey frame plus resolved roles, shared by all checks."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fieldguard.config import FieldGuardConfig
from fieldguard.io.columns import AmbiguousColumnError, ColumnResolver
from fieldguard.io.platforms import GENERIC, PLATFORMS, detect_gps_columns, detect_platform, first_present
from fieldguard.io.timestamps import parse_dates, parse_timestamps, to_local_dates
from fieldguard.models import ROW_COLUMN, Level, Severity, coerce_issues, empty_issues

log = logging.getLogger(__name__)

MISSING_ENUMERATOR = "(missing)"
UNKNOWN_ENUMERATOR = "(unknown)"


def normalize_code(value: Any) -> str:
    """Canonical string for special-code matching: -99, '-99', -99.0 -> '-99'; text lower-cased."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else repr(float(value))
    text = str(value).strip().lower()
    if re.fullmatch(r"[-+]?\d+(\.0+)?", text):
        return str(int(float(text)))
    return text


def normalize_series(s: pd.Series) -> pd.Series:
    text = s.astype("str").where(s.notna(), "").str.strip().str.lower()
    # '-99.0' -> '-99'
    return text.str.replace(r"^([-+]?\d+)\.0+$", r"\1", regex=True)


@dataclass
class CheckContext:
    data: pd.DataFrame
    config: FieldGuardConfig
    run_id: str
    platform: str
    resolver: ColumnResolver
    roles: dict[str, str | None]
    record_id: pd.Series
    enumerator: pd.Series
    respondent_id: pd.Series | None = None
    cluster: pd.Series | None = None
    team: pd.Series | None = None
    start: pd.Series | None = None
    end: pd.Series | None = None
    submission_time: pd.Series | None = None
    survey_date: pd.Series | None = None
    duration_min: pd.Series | None = None
    gps: pd.DataFrame | None = None
    meta_columns: set[str] = field(default_factory=set)
    setup_notes: list[str] = field(default_factory=list)
    _metrics: list[dict[str, Any]] = field(default_factory=list)
    _num_cache: dict[str, pd.Series] = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def has_enumerators(self) -> bool:
        return self.roles.get("enumerator_id") is not None

    @property
    def tz(self) -> str:
        return self.config.project.timezone

    def col(self, name: str | None) -> str | None:
        if name is None:
            return None
        try:
            return self.resolver.resolve(name)
        except AmbiguousColumnError:
            return None

    def cols(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        return self.resolver.resolve_many(names)

    def answer_columns(self) -> list[str]:
        """Substantive answer columns (metadata, roles, GPS, PII and excluded columns removed)."""
        excluded, _ = self.cols(self.config.variables.exclude)
        pii, _ = self.cols(self.config.privacy.pii_columns)
        skip = self.meta_columns | set(excluded) | set(pii)
        out = []
        for c in self.data.columns:
            if c in skip or c.startswith("_") or c.endswith("_URL"):
                continue
            low = c.lower()
            if low.split("/")[-1] in {"start", "end", "today", "deviceid", "audit"}:
                continue
            out.append(c)
        return out

    # ------------------------------------------------------------------ values
    def special_codes(self, kind: str) -> set[str]:
        codes = getattr(self.config.variables.special_codes, kind)
        return {normalize_code(c) for c in codes}

    def special_mask(self, column: str, kinds: Sequence[str] = ("dont_know", "refused", "not_applicable")) -> pd.Series:
        codes: set[str] = set()
        for k in kinds:
            codes |= self.special_codes(k)
        return normalize_series(self.data[column]).isin(codes) & self.data[column].notna()

    def num(self, column: str) -> pd.Series:
        """Numeric view of a column with special codes (DK/refused/NA) masked as NaN."""
        if column not in self._num_cache:
            s = self.data[column]
            values = pd.to_numeric(s.astype("str").where(s.notna()), errors="coerce").astype("float64")
            values = values.mask(self.special_mask(column))
            self._num_cache[column] = values
        return self._num_cache[column]

    def text(self, column: str) -> pd.Series:
        s = self.data[column]
        return s.astype("str").where(s.notna()).str.strip()

    # ------------------------------------------------------------------ metrics
    def add_metric(self, check_id: str, enumerator: str, metric: str, value: float) -> None:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        self._metrics.append(
            {"check_id": check_id, "enumerator": str(enumerator), "metric": metric, "value": float(value)}
        )

    def add_metric_series(self, check_id: str, metric: str, values: pd.Series) -> None:
        for enum, value in values.items():
            if pd.notna(value):
                self.add_metric(check_id, str(enum), metric, float(value))

    def metrics_frame(self) -> pd.DataFrame:
        from fieldguard.models import empty_metrics

        if not self._metrics:
            return empty_metrics()
        return pd.DataFrame(self._metrics)

    # ------------------------------------------------------------------ issue builders
    def record_issues(
        self,
        check_id: str,
        rows: pd.Series | pd.Index | Sequence[int] | np.ndarray,
        severity: Severity | str,
        message: str | pd.Series,
        variable: str | pd.Series | None = None,
        value: Any = None,
        metric: Any = None,
    ) -> pd.DataFrame:
        """Build record-level issues for the given rows (boolean mask or row positions)."""
        if isinstance(rows, pd.Series) and pd.api.types.is_bool_dtype(rows):
            idx = np.flatnonzero(rows.fillna(False).to_numpy(dtype=bool))
        else:
            idx = np.asarray(list(rows), dtype=int)
        if len(idx) == 0:
            return empty_issues(with_row=True)

        def pick(v: Any) -> Any:
            # pd.Series -> aligned with the full survey frame; list/ndarray -> one value per issue.
            if isinstance(v, pd.Series):
                return v.reindex(self.data.index).iloc[idx].to_numpy()
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=object)
                if len(arr) != len(idx):
                    raise ValueError("per-issue values must have one entry per flagged row")
                return arr
            return v

        frame = pd.DataFrame(
            {
                "check_id": check_id,
                "severity": str(severity),
                "level": Level.RECORD.value,
                "variable": pick(variable),
                "value": pick(value),
                "metric": pick(metric),
                "message": pick(message),
                ROW_COLUMN: idx.astype("float64"),
            },
            index=range(len(idx)),
        )
        return coerce_issues(frame)

    def issue(
        self,
        check_id: str,
        severity: Severity | str,
        message: str,
        *,
        level: Level | str = Level.ENUMERATOR,
        enumerator: str | None = None,
        variable: str | None = None,
        value: Any = None,
        metric: float | None = None,
        record_id: str | None = None,
        row: int | None = None,
    ) -> dict[str, Any]:
        return {
            "check_id": check_id,
            "severity": str(severity),
            "level": str(level),
            "enumerator": enumerator,
            "variable": variable,
            "value": None if value is None else str(value),
            "metric": metric,
            "message": message,
            "record_id": record_id,
            ROW_COLUMN: float(row) if row is not None else np.nan,
        }

    def frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not rows:
            return empty_issues(with_row=True)
        return coerce_issues(pd.DataFrame(rows))

    def dataset_issue(
        self, check_id: str, severity: Severity | str, message: str, variable: str | None = None
    ) -> pd.DataFrame:
        return self.frame([self.issue(check_id, severity, message, level=Level.DATASET, variable=variable)])

    def missing_columns_issue(self, check_id: str, names: Sequence[str]) -> pd.DataFrame:
        if not names:
            return empty_issues(with_row=True)
        return self.dataset_issue(
            check_id,
            Severity.WARNING,
            f"Configured column(s) not found in data and skipped: {', '.join(map(str, names))}",
            variable=", ".join(map(str, names)),
        )

    def enumerator_groups(self, min_n: int = 1) -> dict[str, np.ndarray]:
        groups = {}
        for enum, idx in self.enumerator.groupby(self.enumerator, sort=True).indices.items():
            if len(idx) >= min_n:
                groups[str(enum)] = idx
        return groups


# ---------------------------------------------------------------------- construction
def _role_column(
    ctx_resolver: ColumnResolver,
    configured: str | None,
    defaults: Sequence[str],
    columns: set[str],
    role: str,
    notes: list[str],
) -> str | None:
    if configured:
        try:
            col = ctx_resolver.resolve(configured)
        except AmbiguousColumnError as exc:
            notes.append(f"role {role}: {exc}")
            return None
        if col is None:
            notes.append(f"role {role}: column {configured!r} not found")
        return col
    return first_present(defaults, columns)


def _clean_ids(s: pd.Series) -> pd.Series:
    t = s.astype("str").where(s.notna()).str.strip()
    return t.mask(t == "")


def build_context(data: pd.DataFrame, config: FieldGuardConfig, run_id: str = "adhoc") -> CheckContext:
    """Resolve roles, parse timestamps/GPS and return a ready-to-use CheckContext."""
    df = data.reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    columns = set(df.columns)
    resolver = ColumnResolver(df.columns)
    notes: list[str] = []
    platform = config.project.platform
    if platform == "auto":
        platform = detect_platform(df.columns)
    conv = PLATFORMS.get(platform, GENERIC)
    r = config.roles

    roles: dict[str, str | None] = {
        "submission_id": _role_column(resolver, r.submission_id, conv.submission_id, columns, "submission_id", notes),
        "respondent_id": _role_column(resolver, r.respondent_id, (), columns, "respondent_id", notes),
        "enumerator_id": _role_column(resolver, r.enumerator_id, (), columns, "enumerator_id", notes),
        "cluster": _role_column(resolver, r.cluster, (), columns, "cluster", notes),
        "team": _role_column(resolver, r.team, (), columns, "team", notes),
        "start": _role_column(resolver, r.start, conv.start, columns, "start", notes),
        "end": _role_column(resolver, r.end, conv.end, columns, "end", notes),
        "submission_time": _role_column(
            resolver, r.submission_time, conv.submission_time, columns, "submission_time", notes
        ),
        "survey_date": _role_column(resolver, r.survey_date, conv.survey_date, columns, "survey_date", notes),
        "duration": _role_column(resolver, r.duration, conv.duration, columns, "duration", notes),
        "deviceid": _role_column(resolver, r.deviceid, conv.deviceid, columns, "deviceid", notes),
    }
    n = len(df)
    tz = config.project.timezone
    naive_tz = config.project.naive_timestamps_timezone or tz
    sub_tz = config.project.submission_time_timezone or conv.naive_submission_tz or tz

    if roles["submission_id"]:
        record_id = _clean_ids(df[roles["submission_id"]])
        record_id = record_id.fillna(pd.Series([f"row-{i + 1}" for i in range(n)], index=df.index))
    else:
        record_id = pd.Series([f"row-{i + 1}" for i in range(n)], index=df.index, dtype="object")
    record_id = record_id.astype("object")

    if roles["enumerator_id"]:
        enumerator = _clean_ids(df[roles["enumerator_id"]]).fillna(MISSING_ENUMERATOR).astype("object")
    else:
        enumerator = pd.Series(UNKNOWN_ENUMERATOR, index=df.index, dtype="object")

    def opt_ids(role: str) -> pd.Series | None:
        col = roles[role]
        return _clean_ids(df[col]).astype("object") if col else None

    start = parse_timestamps(df[roles["start"]], tz, naive_tz) if roles["start"] else None
    end = parse_timestamps(df[roles["end"]], tz, naive_tz) if roles["end"] else None
    sub = parse_timestamps(df[roles["submission_time"]], tz, sub_tz) if roles["submission_time"] else None

    survey_date: pd.Series | None = None
    if roles["survey_date"]:
        survey_date = parse_dates(df[roles["survey_date"]])
    if start is not None:
        from_start = to_local_dates(start)
        survey_date = from_start if survey_date is None else survey_date.where(survey_date.notna(), from_start)
    if survey_date is None and sub is not None:
        survey_date = to_local_dates(sub)

    duration: pd.Series | None = None
    if roles["duration"]:
        raw = pd.to_numeric(df[roles["duration"]].astype("str").where(df[roles["duration"]].notna()), errors="coerce")
        duration = raw / 60.0 if r.duration_unit == "seconds" else raw.astype("float64")
    elif start is not None and end is not None:
        duration = (end - start).dt.total_seconds() / 60.0

    gps, gps_cols = _build_gps(df, resolver, config, notes)

    meta = set(conv.metadata) | set(GENERIC.metadata) | {c for c in roles.values() if c} | gps_cols
    return CheckContext(
        data=df,
        config=config,
        run_id=run_id,
        platform=platform,
        resolver=resolver,
        roles=roles,
        record_id=record_id,
        enumerator=enumerator,
        respondent_id=opt_ids("respondent_id"),
        cluster=opt_ids("cluster"),
        team=opt_ids("team"),
        start=start,
        end=end,
        submission_time=sub,
        survey_date=survey_date,
        duration_min=duration,
        gps=gps,
        meta_columns=meta,
        setup_notes=notes,
    )


def _build_gps(
    df: pd.DataFrame, resolver: ColumnResolver, config: FieldGuardConfig, notes: list[str]
) -> tuple[pd.DataFrame | None, set[str]]:
    g = config.roles.gps

    def num(col: str | None) -> pd.Series:
        if col is None:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        s = df[col]
        return pd.to_numeric(s.astype("str").where(s.notna()), errors="coerce").astype("float64")

    if g is not None and g.point:
        col = resolver.resolve(g.point)
        if col is None:
            notes.append(f"gps.point column {g.point!r} not found")
            return None, set()
        parts = df[col].astype("str").where(df[col].notna()).str.split(r"\s+", expand=True, regex=True)
        parts = parts.reindex(columns=range(4))
        frame = pd.DataFrame(
            {
                "lat": pd.to_numeric(parts[0], errors="coerce"),
                "lon": pd.to_numeric(parts[1], errors="coerce"),
                "accuracy": pd.to_numeric(parts[3], errors="coerce"),
            }
        ).astype("float64")
        return _valid_coords(frame), {col}
    if g is not None and (g.lat or g.lon):
        lat, lon = resolver.resolve(g.lat), resolver.resolve(g.lon)
        acc = resolver.resolve(g.accuracy) if g.accuracy else None
        if lat is None or lon is None:
            notes.append(f"gps lat/lon columns not found ({g.lat!r}, {g.lon!r})")
            return None, set()
        frame = pd.DataFrame({"lat": num(lat), "lon": num(lon), "accuracy": num(acc)})
        return _valid_coords(frame), {c for c in (lat, lon, acc) if c}
    detected = detect_gps_columns(df.columns)
    if detected:
        frame = pd.DataFrame(
            {"lat": num(detected["lat"]), "lon": num(detected["lon"]), "accuracy": num(detected["accuracy"])}
        )
        cols = {c for c in detected.values() if c}
        base = detected["lat"]
        # include sibling columns (altitude, raw point) in metadata
        prefixes = (base.rsplit("_", 1)[0] + "_", base.rsplit("-", 1)[0] + "-")
        cols |= {c for c in df.columns if c.startswith(prefixes)}
        field_name = base[1:].rsplit("_", 1)[0] if base.startswith("_") else base.rsplit("-", 1)[0]
        raw_point = resolver.resolve(field_name) if field_name else None
        if raw_point:
            cols.add(raw_point)
        return _valid_coords(frame), cols
    return None, set()


def _valid_coords(frame: pd.DataFrame) -> pd.DataFrame | None:
    bad = (frame["lat"].abs() > 90) | (frame["lon"].abs() > 180) | ((frame["lat"] == 0) & (frame["lon"] == 0))
    if bad.any():
        frame = frame.copy()
        frame.loc[bad, :] = np.nan
    if frame["lat"].notna().sum() == 0:
        return None
    return frame
