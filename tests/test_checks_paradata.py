"""Paradata checks: interview duration, timestamp anomalies (with timezones) and GPS."""

from __future__ import annotations

import pandas as pd

from tests.conftest import make_cfg, run_check, survey_frame


def with_durations(df: pd.DataFrame, minutes: list[float]) -> pd.DataFrame:
    out = df.copy()
    start = pd.to_datetime(out["start"], utc=True)
    out["end"] = [(s + pd.Timedelta(minutes=m)).isoformat() for s, m in zip(start, minutes, strict=True)]
    return out


# ----------------------------------------------------------------------------- duration
def test_very_short_critical_and_short_warning_and_long_info():
    df = survey_frame(10, ("E1",))
    minutes = [50.0] * len(df)
    minutes[0] = 2.0  # below the 5-minute floor
    minutes[1] = 18.0  # below half the median
    minutes[2] = 400.0  # far above the median
    out = run_check("duration", with_durations(df, minutes))
    crit = out[(out["severity"] == "critical") & (out["check_id"] == "duration")]
    warn = out[(out["severity"] == "warning") & (out["check_id"] == "duration") & (out["level"] == "record")]
    info = out[(out["severity"] == "info") & (out["check_id"] == "duration") & (out["level"] == "record")]
    assert crit["record_id"].tolist() == ["uuid-0001"]
    assert "uuid-0002" in set(warn["record_id"])
    assert "uuid-0003" in set(info["record_id"])


def test_enumerator_with_fast_interviews_gets_enumerator_flag():
    df = survey_frame(10, ("E1", "E2"))
    minutes = [50.0] * len(df)
    for i in range(10):
        minutes[i] = 12.0  # all of E1 is fast
    out = run_check("duration", with_durations(df, minutes))
    enum_flags = out[(out["level"] == "enumerator") & (out["enumerator"] == "E1")]
    assert len(enum_flags) == 1


def test_duration_without_timestamps_is_empty():
    df = survey_frame(4, ("E1",)).drop(columns=["start", "end"])
    cfg = make_cfg(roles={"submission_id": "_uuid", "enumerator_id": "enum"})
    assert run_check("duration", df, cfg).empty


# ----------------------------------------------------------------------------- timestamps
def test_end_before_start_is_critical():
    df = survey_frame(3, ("E1",))
    df.loc[1, "end"] = (pd.Timestamp(df.loc[1, "start"]) - pd.Timedelta(minutes=10)).isoformat()
    out = run_check("timestamps", df)
    crit = out[out["message"].str.startswith("End time is before start")]
    assert crit["record_id"].tolist() == ["uuid-0002"]
    assert (crit["severity"] == "critical").all()


def test_overlapping_interviews_by_same_enumerator():
    df = survey_frame(3, ("E1", "E2"))
    df.loc[1, "start"] = "2026-03-02T09:10:00+03:00"
    df.loc[1, "end"] = "2026-03-02T10:30:00+03:00"
    df.loc[0, "start"] = "2026-03-02T09:00:00+03:00"
    df.loc[0, "end"] = "2026-03-02T10:00:00+03:00"
    out = run_check("timestamps", df)
    overlap = out[out["message"].str.startswith("Interview overlaps")]
    assert "uuid-0002" in set(overlap["record_id"])
    assert "uuid-0005" not in set(overlap["record_id"])


def test_timezone_offsets_are_compared_as_instants():
    df = survey_frame(2, ("E1",))
    df.loc[0, ["start", "end"]] = ["2026-03-02T09:00:00+03:00", "2026-03-02T09:30:00+03:00"]
    df.loc[1, ["start", "end"]] = ["2026-03-02T06:05:00Z", "2026-03-02T06:40:00Z"]  # 09:05 Nairobi: inside the first
    out = run_check("timestamps", df)
    assert "uuid-0002" in set(out[out["message"].str.startswith("Interview overlaps")]["record_id"])


def test_off_hours_start_and_too_many_per_day():
    df = survey_frame(3, ("E1",))
    df.loc[0, "start"] = "2026-03-02T03:15:00+03:00"
    df.loc[0, "end"] = "2026-03-02T03:50:00+03:00"
    df["today"] = "2026-03-02"
    out = run_check("timestamps", df, make_cfg(), {"max_per_day": 2})
    assert out["message"].str.contains("outside working hours").any()
    assert out["message"].str.contains("interviews on 2026-03-02").any()


def test_before_fieldwork_start_and_future_timestamps():
    df = survey_frame(2, ("E1",))
    df.loc[0, "start"] = "2030-01-05T09:00:00+03:00"
    df.loc[0, "end"] = "2030-01-05T09:40:00+03:00"
    cfg_early = make_cfg(project={"name": "x", "timezone": "Africa/Nairobi", "fieldwork": {"start_date": "2026-03-10"}})
    out = run_check("timestamps", df, cfg_early)
    assert out["message"].str.contains("future").any()
    assert out["message"].str.contains("before fieldwork start").any()


def test_submitted_before_end_flags_clock_problem():
    df = survey_frame(2, ("E1",))
    df["_submission_time"] = "2026-03-02T05:00:00"
    out = run_check("timestamps", df)
    assert out["message"].str.contains("before the interview ended").any()


def test_timestamps_with_missing_values_do_not_crash():
    df = survey_frame(3, ("E1",))
    df["start"] = None
    df["end"] = None
    out = run_check("timestamps", df)
    assert not (out["severity"] == "critical").any()


# ----------------------------------------------------------------------------- GPS
def test_gps_outside_cluster_radius_and_low_accuracy():
    df = survey_frame(4, ("E1", "E2"))
    df.loc[0, "_gps_latitude"] = "-0.100000"  # about 33 km north of the cluster
    df.loc[1, "_gps_precision"] = "80.0"
    out = run_check("gps", df, make_cfg(), {"cluster_radius_m": 3000})
    far = out[out["message"].str.contains("from the cluster centre")]
    assert far["record_id"].tolist() == ["uuid-0001"]
    assert (far["severity"] == "warning").all()
    low = out[out["message"].str.startswith("GPS accuracy worse")]
    assert "uuid-0002" in set(low["record_id"])


def test_identical_coordinates_across_submissions_are_flagged():
    df = survey_frame(3, ("E1", "E2"))
    df.loc[3, ["_gps_latitude", "_gps_longitude"]] = df.loc[0, ["_gps_latitude", "_gps_longitude"]].to_numpy()
    out = run_check("gps", df)
    ident = out[out["message"].str.startswith("Identical GPS")]
    assert set(ident["record_id"]) == {"uuid-0001", "uuid-0004"}


def test_same_spot_interviews_by_one_enumerator():
    df = survey_frame(4, ("E1", "E2"))
    df.loc[1, ["_gps_latitude", "_gps_longitude"]] = [
        float(df.loc[0, "_gps_latitude"]) + 0.00003,
        float(df.loc[0, "_gps_longitude"]),
    ]
    out = run_check("gps", df, make_cfg(), {"min_distance_m": 10})
    spot = out[out["message"].str.startswith("Within 10 m")]
    assert {"uuid-0001", "uuid-0002"} <= set(spot["record_id"])


def test_gps_absent_or_all_missing_returns_empty():
    df = survey_frame(3, ("E1",))
    df[["_gps_latitude", "_gps_longitude"]] = None
    assert run_check("gps", df).empty
    no_gps = survey_frame(3, ("E1",)).drop(columns=["_gps_latitude", "_gps_longitude", "_gps_precision"])
    assert run_check("gps", no_gps).empty


def test_paradata_checks_on_empty_frame():
    empty = survey_frame(2, ("E1",)).iloc[0:0]
    for check in ("duration", "timestamps", "gps"):
        assert run_check(check, empty).empty
