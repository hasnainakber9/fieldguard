"""Readers, column resolution, platform conventions, flattening and timestamp parsing."""

from __future__ import annotations

import pandas as pd
import pytest

from fieldguard.io.columns import AmbiguousColumnError, ColumnResolver
from fieldguard.io.flatten import flatten_record, records_to_frame
from fieldguard.io.platforms import detect_gps_columns, detect_platform
from fieldguard.io.readers import read_table, sniff_delimiter
from fieldguard.io.timestamps import parse_dates, parse_timestamps, to_local_dates


def test_csv_semicolon_keeps_leading_zeros_and_missing_as_nan(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text("_uuid;hhid;age;note\nu1;00123;34;\nu2;00456;NA;ok\n", encoding="utf-8")
    df = read_table(path)
    assert list(df["hhid"]) == ["00123", "00456"]
    assert df["age"].isna().tolist() == [False, True]
    assert df["note"].isna().tolist() == [True, False]


def test_sniff_delimiter_prefers_header_separator():
    assert sniff_delimiter("a,b,c\n1,2,3") == ","
    assert sniff_delimiter("a;b;c\n1;2;3") == ";"
    assert sniff_delimiter("a\tb\n1\t2") == "\t"
    assert sniff_delimiter("") == ","


def test_xlsx_round_trip_reads_text(tmp_path):
    pytest.importorskip("openpyxl")
    path = tmp_path / "export.xlsx"
    pd.DataFrame({"hhid": ["00123", "00456"], "age": [34, 51]}).to_excel(path, index=False)
    df = read_table(path)
    assert df["hhid"].tolist() == ["00123", "00456"]
    assert df["age"].tolist() == ["34", "51"]


def test_unsupported_suffix_and_missing_file(tmp_path):
    bad = tmp_path / "export.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_table(bad)
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.csv")


def test_flatten_groups_geopoints_and_repeats():
    record = {
        "_uuid": "u1",
        "grp": {"q1": "a", "inner": {"q2": 3}},
        "gps": {"type": "Point", "coordinates": [36.9, -0.4], "properties": {"accuracy": 5.0}},
        "rep": [{"x": 1}, {"x": 2}],
        "empty_rep": [],
    }
    flat = flatten_record(record)
    assert flat["grp/q1"] == "a"
    assert flat["grp/inner/q2"] == 3
    assert flat["gps-Latitude"] == -0.4
    assert flat["gps-Longitude"] == 36.9
    assert flat["gps-Accuracy"] == 5.0
    assert flat["rep"] == '[{"x": 1}, {"x": 2}]'
    assert flat["empty_rep"] is None


def test_records_to_frame_empty_and_text_values():
    assert records_to_frame([]).empty
    df = records_to_frame([{"a": 1, "b": None}, {"a": 2, "b": "x"}])
    assert df["a"].tolist() == ["1", "2"]
    assert pd.isna(df.loc[0, "b"])


def test_column_resolver_slash_dash_and_ambiguity():
    res = ColumnResolver(["grp/q1", "other/q1", "age", "grp-q9"])
    assert res.resolve("age") == "age"
    assert res.resolve("grp/q1") == "grp/q1"
    assert res.resolve("q9") == "grp-q9"  # dash-separated tail
    with pytest.raises(AmbiguousColumnError):
        res.resolve("q1")
    cols, missing = res.resolve_many(["age", "nope"])
    assert cols == ["age"] and missing == ["nope"]


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["_uuid", "_submission_time", "start"], "kobo"),
        (["__id", "KEY", "start"], "odk"),
        (["KEY", "starttime", "endtime", "formdef_version"], "surveycto"),
        (["hhid", "age"], "generic"),
    ],
)
def test_platform_detection(columns, expected):
    assert detect_platform(columns) == expected


def test_gps_columns_detected_for_kobo_and_odk_styles():
    kobo = detect_gps_columns(["_gps_latitude", "_gps_longitude", "_gps_altitude", "_gps_precision"])
    assert kobo == {"lat": "_gps_latitude", "lon": "_gps_longitude", "accuracy": "_gps_precision"}
    odk = detect_gps_columns(["gps-Latitude", "gps-Longitude", "gps-Accuracy"])
    assert odk["lat"] == "gps-Latitude" and odk["accuracy"] == "gps-Accuracy"
    assert detect_gps_columns(["a", "b"]) is None


def test_timestamps_offsets_and_naive_localisation():
    raw = pd.Series(["2026-03-02T09:00:00.000+03:00", "2026-03-02T06:00:00Z", "2026-03-02 09:00:00"])
    parsed = parse_timestamps(raw, tz="Africa/Nairobi", naive_tz="Africa/Nairobi")
    assert str(parsed.dt.tz) == "Africa/Nairobi"
    assert parsed.iloc[0] == parsed.iloc[1]  # same instant after conversion
    assert parsed.iloc[2] == parsed.iloc[0]  # naive values are read in naive_tz


def test_timestamps_empty_all_missing_and_garbage():
    empty = parse_timestamps(pd.Series([], dtype=object), tz="UTC")
    assert empty.empty
    allnan = parse_timestamps(pd.Series([None, None], dtype=object), tz="UTC")
    assert allnan.isna().all()
    garbage = parse_timestamps(pd.Series(["not a time", "2026-03-02T09:00:00+03:00"]), tz="UTC")
    assert pd.isna(garbage.iloc[0]) and not pd.isna(garbage.iloc[1])


def test_to_local_dates_and_parse_dates():
    ts = parse_timestamps(pd.Series(["2026-03-02T23:30:00-05:00"]), tz="UTC")
    assert to_local_dates(ts).tolist() == ["2026-03-03"]
    dates = parse_dates(pd.Series(["2026-03-02", "garbage", None]))
    assert dates.tolist() == ["2026-03-02", None, None]
