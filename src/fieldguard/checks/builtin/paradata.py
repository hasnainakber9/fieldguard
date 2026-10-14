"""Paradata checks: interview duration, timestamp anomalies and GPS."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.spatial.distance import pdist

from fieldguard.checks.context import CheckContext
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.models import Severity, empty_issues
from fieldguard.stats import EARTH_RADIUS_M, haversine_m


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else empty_issues(with_row=True)


# ----------------------------------------------------------------------------- duration
class DurationParams(CheckParams):
    short_ratio: float = Field(0.5, gt=0, lt=1, description="Flag interviews shorter than ratio x form median")
    min_minutes: float = Field(5.0, ge=0, description="Interviews shorter than this are critical")
    long_ratio: float = Field(4.0, gt=1)
    enumerator_median_ratio: float = Field(0.7, gt=0, lt=1)
    min_n: int = Field(5, ge=1)


@register_check(
    "duration",
    title="Interview duration",
    description="Interviews much shorter than the form median, overall and as an enumerator pattern.",
    category="paradata",
    params=DurationParams,
    references=("Murphy et al. (2004)", "J-PAL data quality checks", "IPA ipacheck: ipacheckduration"),
)
def duration(ctx: CheckContext, p: DurationParams) -> pd.DataFrame:
    if ctx.duration_min is None or ctx.n == 0:
        return empty_issues(with_row=True)
    d = ctx.duration_min
    pos = d.where(d > 0)
    if pos.notna().sum() == 0:
        return empty_issues(with_row=True)
    med = float(pos.median())
    frames = []
    very_short = pos < p.min_minutes
    short = (pos < p.short_ratio * med) & ~very_short
    long_ = pos > p.long_ratio * med
    ratio = pos / med
    frames.append(
        ctx.record_issues(
            "duration",
            very_short,
            Severity.CRITICAL,
            "Interview took " + d.round(1).astype(str) + f" min (< {p.min_minutes:g} min)",
            variable="duration_min",
            value=d.round(2),
            metric=ratio,
        )
    )
    frames.append(
        ctx.record_issues(
            "duration",
            short,
            Severity.WARNING,
            "Interview took "
            + d.round(1).astype(str)
            + f" min, less than {p.short_ratio:.0%} of the median ({med:.1f} min)",
            variable="duration_min",
            value=d.round(2),
            metric=ratio,
        )
    )
    frames.append(
        ctx.record_issues(
            "duration",
            long_,
            Severity.INFO,
            "Interview took "
            + d.round(1).astype(str)
            + f" min, more than {p.long_ratio:g}x the median (paused/edited later?)",
            variable="duration_min",
            value=d.round(2),
            metric=ratio,
        )
    )
    g = pd.DataFrame(
        {"r": ratio, "s": (very_short | short).astype(float).where(pos.notna()), "e": ctx.enumerator}
    ).groupby("e")
    med_ratio = g["r"].median()
    share = g["s"].mean()
    counts = g["r"].count()
    ctx.add_metric_series("duration", "median_duration_ratio", med_ratio)
    ctx.add_metric_series("duration", "short_duration_share", share)
    rows = []
    if ctx.has_enumerators:
        for enum, mr in med_ratio.items():
            if counts[enum] >= p.min_n and mr <= p.enumerator_median_ratio:
                rows.append(
                    ctx.issue(
                        "duration",
                        Severity.WARNING,
                        f"Median interview {mr:.0%} of the form median; {share[enum]:.0%} of interviews short",
                        enumerator=str(enum),
                        variable="duration_min",
                        value=f"{mr * med:.1f}",
                        metric=float(mr),
                    )
                )
    frames.append(ctx.frame(rows))
    return _concat(frames)


# ----------------------------------------------------------------------------- timestamps
class TimestampParams(CheckParams):
    working_hours: tuple[int, int] = Field((6, 21), description="Allowed local start hours [from, to)")
    max_per_day: int = Field(8, ge=1)
    overlap_tolerance_minutes: float = Field(1.0, ge=0)
    future_tolerance_hours: float = Field(24.0, ge=0)
    submission_before_end_tolerance_minutes: float = Field(5.0, ge=0)


@register_check(
    "timestamps",
    title="Timestamp anomalies",
    description="End before start, implausible hours, overlapping interviews, too many per day, out-of-fieldwork dates, clock problems.",
    category="paradata",
    params=TimestampParams,
    references=("DIME Wiki: High Frequency Checks", "AAPOR (2003)"),
)
def timestamps(ctx: CheckContext, p: TimestampParams) -> pd.DataFrame:
    frames = []
    if ctx.n == 0:
        return empty_issues(with_row=True)
    start, end, sub = ctx.start, ctx.end, ctx.submission_time
    if start is not None and end is not None:
        neg = (end < start).fillna(False)
        frames.append(
            ctx.record_issues(
                "timestamps",
                neg,
                Severity.CRITICAL,
                "End time is before start time",
                variable="end",
                value=end.astype(str),
            )
        )
    if start is not None:
        hours = start.dt.hour + start.dt.minute / 60.0
        lo, hi = p.working_hours
        off = (start.notna() & ((hours < lo) | (hours >= hi))).fillna(False)
        frames.append(
            ctx.record_issues(
                "timestamps",
                off,
                Severity.WARNING,
                f"Interview started outside working hours ({lo:02d}:00-{hi:02d}:00 local)",
                variable="start",
                value=start.dt.strftime("%Y-%m-%d %H:%M %z"),
            )
        )
        ctx.add_metric_series(
            "timestamps", "offhours_share", off.astype(float).where(start.notna()).groupby(ctx.enumerator).mean()
        )
        # overlapping interviews by the same enumerator
        if end is not None and ctx.has_enumerators:
            order = pd.DataFrame({"s": start, "e": end, "enum": ctx.enumerator}).dropna(subset=["s", "e"])
            order = order.sort_values(["enum", "s"])
            prev_end = order.groupby("enum")["e"].shift(1)
            prev_end_max = prev_end.groupby(order["enum"]).cummax() if not prev_end.empty else prev_end
            tol = pd.Timedelta(minutes=p.overlap_tolerance_minutes)
            ov = (prev_end_max.notna()) & (order["s"] < prev_end_max - tol)
            overlap = pd.Series(False, index=ctx.data.index)
            overlap.loc[ov[ov].index] = True
            frames.append(
                ctx.record_issues(
                    "timestamps",
                    overlap,
                    Severity.WARNING,
                    "Interview overlaps a previous interview by the same enumerator",
                    variable="start",
                    value=start.dt.strftime("%Y-%m-%d %H:%M"),
                )
            )
            ctx.add_metric_series(
                "timestamps", "overlap_share", overlap.astype(float).where(start.notna()).groupby(ctx.enumerator).mean()
            )
    if ctx.survey_date is not None and ctx.has_enumerators:
        per_day = pd.DataFrame({"d": ctx.survey_date, "e": ctx.enumerator}).dropna().groupby(["e", "d"]).size()
        if not per_day.empty:
            ctx.add_metric_series("timestamps", "max_per_day", per_day.groupby(level=0).max())
            rows = [
                ctx.issue(
                    "timestamps",
                    Severity.WARNING,
                    f"{count} interviews on {day} (max {p.max_per_day} per day)",
                    enumerator=str(enum),
                    variable="survey_date",
                    value=day,
                    metric=float(count),
                )
                for (enum, day), count in per_day.items()
                if count > p.max_per_day
            ]
            frames.append(ctx.frame(rows))
    fw = ctx.config.project.fieldwork
    if ctx.survey_date is not None and (fw.start_date or fw.end_date):
        sd = pd.to_datetime(ctx.survey_date, errors="coerce")
        if fw.start_date:
            before = (sd < pd.Timestamp(fw.start_date)).fillna(False)
            frames.append(
                ctx.record_issues(
                    "timestamps",
                    before,
                    Severity.WARNING,
                    f"Interview dated before fieldwork start ({fw.start_date})",
                    variable="survey_date",
                    value=ctx.survey_date,
                )
            )
        if fw.end_date:
            after = (sd > pd.Timestamp(fw.end_date)).fillna(False)
            frames.append(
                ctx.record_issues(
                    "timestamps",
                    after,
                    Severity.INFO,
                    f"Interview dated after planned fieldwork end ({fw.end_date})",
                    variable="survey_date",
                    value=ctx.survey_date,
                )
            )
    now = pd.Timestamp(datetime.now(UTC))
    ref = start if start is not None else end
    if ref is not None:
        fut = (ref > now + pd.Timedelta(hours=p.future_tolerance_hours)).fillna(False)
        frames.append(
            ctx.record_issues(
                "timestamps",
                fut,
                Severity.CRITICAL,
                "Timestamp is in the future (device clock wrong?)",
                variable="start",
                value=ref.astype(str),
            )
        )
    if sub is not None and end is not None:
        tol = pd.Timedelta(minutes=p.submission_before_end_tolerance_minutes)
        early = (sub < end - tol).fillna(False)
        frames.append(
            ctx.record_issues(
                "timestamps",
                early,
                Severity.WARNING,
                "Submitted to the server before the interview ended (device clock manipulated?)",
                variable="submission_time",
                value=sub.astype(str),
            )
        )
    today_col = ctx.roles.get("survey_date")
    if today_col and start is not None:
        today = pd.to_datetime(ctx.text(today_col), errors="coerce", format="mixed")
        sdate = pd.to_datetime(start.dt.strftime("%Y-%m-%d"), errors="coerce")
        mismatch = ((today - sdate).abs() > pd.Timedelta(days=1)).fillna(False)
        frames.append(
            ctx.record_issues(
                "timestamps",
                mismatch,
                Severity.INFO,
                "`today` differs from the start date by more than a day",
                variable=today_col,
                value=ctx.data[today_col],
            )
        )
    return _concat(frames)


# ----------------------------------------------------------------------------- GPS
class GPSParams(CheckParams):
    cluster_radius_m: float = Field(3000.0, gt=0)
    centroids_file: str | None = Field(None, description="CSV with cluster,lat,lon[,radius_m]")
    max_accuracy_m: float = Field(50.0, gt=0)
    duplicate_decimals: int = Field(5, ge=3, le=8, description="Coordinates equal at this precision are identical")
    min_distance_m: float = Field(10.0, ge=0, description="An enumerator's interviews closer than this are 'same spot'")
    dispersion_ratio_threshold: float = Field(0.2, gt=0, lt=1)
    min_n: int = Field(5, ge=2)
    max_points_baseline: int = Field(400, ge=20)


def _median_pairwise(lat: np.ndarray, lon: np.ndarray, rng: np.random.Generator, cap: int) -> float:
    if len(lat) < 2:
        return float("nan")
    if len(lat) > cap:
        sel = rng.choice(len(lat), cap, replace=False)
        lat, lon = lat[sel], lon[sel]
    pts = np.radians(np.column_stack([lat, lon]))

    def hav(u: np.ndarray, v: np.ndarray) -> float:
        a = np.sin((v[0] - u[0]) / 2) ** 2 + np.cos(u[0]) * np.cos(v[0]) * np.sin((v[1] - u[1]) / 2) ** 2
        return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(min(max(a, 0.0), 1.0)))

    if len(lat) <= 60:
        return float(np.median(pdist(pts, hav)))
    # equirectangular approximation for larger sets (fast; fine at village scale)
    x = pts[:, 1] * np.cos(pts[:, 0].mean()) * EARTH_RADIUS_M
    y = pts[:, 0] * EARTH_RADIUS_M
    return float(np.median(pdist(np.column_stack([x, y]))))


@register_check(
    "gps",
    title="GPS checks",
    description="Points outside the cluster radius, identical or same-spot coordinates, low accuracy, and enumerators whose interviews are implausibly concentrated.",
    category="paradata",
    params=GPSParams,
    references=("DIME Wiki: High Frequency Checks", "IPA ipacheck: ipacheckgpsdistance"),
)
def gps(ctx: CheckContext, p: GPSParams) -> pd.DataFrame:
    if ctx.gps is None or ctx.n == 0:
        return empty_issues(with_row=True)
    frames = []
    lat, lon, acc = ctx.gps["lat"], ctx.gps["lon"], ctx.gps["accuracy"]
    has = lat.notna() & lon.notna()
    frames.append(ctx.record_issues("gps", ~has, Severity.INFO, "No GPS coordinates recorded", variable="gps"))
    low = (acc > p.max_accuracy_m).fillna(False) & has
    frames.append(
        ctx.record_issues(
            "gps",
            low,
            Severity.INFO,
            f"GPS accuracy worse than {p.max_accuracy_m:g} m",
            variable="gps_accuracy",
            value=acc,
            metric=acc,
        )
    )
    ctx.add_metric_series("gps", "gps_low_accuracy_share", low.astype(float).where(has).groupby(ctx.enumerator).mean())

    # identical coordinates across submissions
    key = lat.round(p.duplicate_decimals).astype(str) + "," + lon.round(p.duplicate_decimals).astype(str)
    ident = has & key.where(has).duplicated(keep=False)
    frames.append(
        ctx.record_issues(
            "gps", ident, Severity.WARNING, "Identical GPS coordinates to another submission", variable="gps", value=key
        )
    )

    # same-spot interviews by one enumerator (within min_distance_m)
    same_spot = pd.Series(False, index=ctx.data.index)
    if ctx.has_enumerators and p.min_distance_m > 0:
        for _, idx in ctx.enumerator_groups(min_n=2).items():
            sub = idx[has.iloc[idx].to_numpy()]
            if len(sub) < 2:
                continue
            la, lo_ = lat.iloc[sub].to_numpy(), lon.iloc[sub].to_numpy()
            dist = haversine_m(la[:, None], lo_[:, None], la[None, :], lo_[None, :])
            np.fill_diagonal(dist, np.inf)
            close = dist.min(axis=1) <= p.min_distance_m
            same_spot.iloc[sub[close]] = True
        only_spot = same_spot & ~ident
        frames.append(
            ctx.record_issues(
                "gps",
                only_spot,
                Severity.INFO,
                f"Within {p.min_distance_m:g} m of another interview by the same enumerator",
                variable="gps",
            )
        )
    ctx.add_metric_series(
        "gps", "gps_duplicate_share", (same_spot | ident).astype(float).where(has).groupby(ctx.enumerator).mean()
    )

    # cluster centroids and radius
    clusters = ctx.cluster if ctx.cluster is not None else pd.Series("all", index=ctx.data.index, dtype="object")
    cent = pd.DataFrame({"lat": lat, "lon": lon, "c": clusters})[has].groupby("c")[["lat", "lon"]].median()
    cent["radius_m"] = p.cluster_radius_m
    if p.centroids_file:
        path = ctx.config.resolve_path(p.centroids_file)
        if path.exists():
            ref = pd.read_csv(path, dtype={"cluster": str})
            ref = ref.set_index("cluster")
            for c in ref.index:
                cent.loc[c, ["lat", "lon"]] = ref.loc[c, ["lat", "lon"]].astype(float).to_numpy()
                if "radius_m" in ref.columns and pd.notna(ref.loc[c, "radius_m"]):
                    cent.loc[c, "radius_m"] = float(ref.loc[c, "radius_m"])
        else:
            frames.append(ctx.dataset_issue("gps", Severity.WARNING, f"Centroids file not found: {path}"))
    if ctx.cluster is not None and not cent.empty:
        c_lat = clusters.map(cent["lat"]).astype("float64")
        c_lon = clusters.map(cent["lon"]).astype("float64")
        radius = clusters.map(cent["radius_m"]).astype("float64")
        dist = pd.Series(
            haversine_m(lat.to_numpy(), lon.to_numpy(), c_lat.to_numpy(), c_lon.to_numpy()), index=ctx.data.index
        )
        outside = (has & (dist > radius)).fillna(False)
        frames.append(
            ctx.record_issues(
                "gps",
                outside,
                Severity.WARNING,
                (dist / 1000).round(2).astype(str)
                + " km from the cluster centre (radius "
                + (radius / 1000).round(2).astype(str)
                + " km)",
                variable="gps",
                value=clusters,
                metric=dist,
            )
        )
        ctx.add_metric_series(
            "gps", "gps_outside_share", outside.astype(float).where(has).groupby(ctx.enumerator).mean()
        )

    # dispersion: median distance between an enumerator's households vs other enumerators in the same clusters
    if ctx.has_enumerators:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"lat": lat, "lon": lon, "c": clusters, "e": ctx.enumerator})[has]
        baseline: dict[tuple[str, str], float] = {}
        rows = []
        for enum, sub in frame.groupby("e"):
            ratios, weights, own_d, base_d = [], [], [], []
            for c, pts in sub.groupby("c"):
                if len(pts) < 3:
                    continue
                others = frame[(frame["c"] == c) & (frame["e"] != enum)]
                if len(others) < 3:
                    continue
                key2 = (str(enum), str(c))
                if key2 not in baseline:
                    baseline[key2] = _median_pairwise(
                        others["lat"].to_numpy(), others["lon"].to_numpy(), rng, p.max_points_baseline
                    )
                b = baseline[key2]
                own = _median_pairwise(pts["lat"].to_numpy(), pts["lon"].to_numpy(), rng, p.max_points_baseline)
                if b and np.isfinite(b) and b > 0:
                    ratios.append(own / b)
                    weights.append(len(pts))
                    own_d.append(own)
                    base_d.append(b)
            if not ratios:
                continue
            ratio = float(np.average(ratios, weights=weights))
            ctx.add_metric("gps", str(enum), "gps_dispersion_ratio", ratio)
            if ratio < p.dispersion_ratio_threshold and sum(weights) >= p.min_n:
                rows.append(
                    ctx.issue(
                        "gps",
                        Severity.WARNING,
                        f"Interview locations concentrated: median {np.average(own_d, weights=weights):.0f} m between households vs "
                        f"{np.average(base_d, weights=weights):.0f} m for other enumerators in the same clusters",
                        enumerator=str(enum),
                        variable="gps",
                        metric=ratio,
                    )
                )
        frames.append(ctx.frame(rows))
    return _concat(frames)
