"""Integrity checks: duplicate IDs, near-duplicate submissions and missing required fields."""

from __future__ import annotations

import numpy as np

from tests.conftest import make_cfg, run_check, survey_frame


def test_duplicate_respondent_ids_are_critical_and_listed_per_record():
    df = survey_frame(3, ("E1", "E2"))
    df.loc[3, "hhid"] = df.loc[0, "hhid"]
    out = run_check("duplicate_ids", df)
    dup = out[out["check_id"] == "duplicate_ids"]
    assert set(dup["record_id"]) == {"uuid-0001", "uuid-0004"}
    assert (dup["severity"] == "critical").all()


def test_duplicated_submission_uuid_is_flagged():
    df = survey_frame(2, ("E1",))
    df.loc[1, "_uuid"] = df.loc[0, "_uuid"]
    out = run_check("duplicate_ids", df)
    assert out["message"].str.contains("Submission UUID is duplicated").any()


def test_all_missing_ids_do_not_crash_and_are_reported():
    df = survey_frame(2, ("E1",))
    df["hhid"] = np.nan
    out = run_check("duplicate_ids", df)
    assert out["message"].str.contains("Respondent ID is missing").all()


def test_near_duplicates_detected_within_enumerator_only():
    df = survey_frame(4, ("E1", "E2"))
    cols = ["age", "income", "village"]
    df.loc[1, cols] = df.loc[0, cols].to_numpy()
    cfg = make_cfg(variables={"numeric": ["age", "income"], "exclude": ["start", "end", "today"]})
    out = run_check("near_duplicates", df, cfg, {"min_compared": 2, "threshold": 0.9})
    flagged = set(out["record_id"])
    assert "uuid-0002" in flagged
    assert "uuid-0005" not in flagged


def test_missing_required_flags_empty_fields_and_all_nan_columns():
    df = survey_frame(3, ("E1",))
    df.loc[0, "hhid"] = None
    df["consent"] = None
    cfg = make_cfg(variables={"required": ["hhid", "consent"], "numeric": ["age"]})
    out = run_check("missing_required", df, cfg)
    assert (out["check_id"] == "missing_required").any()
    assert out["message"].str.contains("Required field hhid is empty").any()
    assert out["message"].str.contains("Column consent is empty in every record").any()


def test_conditional_required_only_where_condition_holds():
    df = survey_frame(3, ("E1",))
    df["has_business"] = ["yes", "no", "yes"]
    df["business_income"] = [None, None, "500"]
    cfg = make_cfg(variables={"required": [{"variable": "business_income", "when": "has_business == 'yes'"}]})
    out = run_check("missing_required", df, cfg)
    assert out["record_id"].tolist() == ["uuid-0001"]


def test_missing_configured_column_is_reported_not_fatal():
    df = survey_frame(2, ("E1",))
    cfg = make_cfg(variables={"required": ["does_not_exist"]})
    out = run_check("missing_required", df, cfg)
    assert out["message"].str.contains("not found in data").any()


def test_integrity_checks_on_empty_frame_return_no_issues():
    empty = survey_frame(2, ("E1",)).iloc[0:0]
    for check in ("duplicate_ids", "near_duplicates", "missing_required"):
        assert run_check(check, empty).empty
