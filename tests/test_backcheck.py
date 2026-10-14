"""Back-checks: comparison rules by variable kind, stratified sampling and comparison with enumerator-level flags."""

from __future__ import annotations

import pandas as pd
import pytest

from fieldguard.backcheck import compare_backchecks, compare_pair, select_sample
from fieldguard.config import BackcheckVariable
from fieldguard.pipeline import run_checks
from tests.conftest import make_cfg, survey_frame


@pytest.mark.parametrize(
    ("a", "b", "spec", "expected"),
    [
        ("male", "Male ", BackcheckVariable(name="sex", kind="categorical"), True),
        ("male", "female", BackcheckVariable(name="sex", kind="categorical"), False),
        ("Kitchen Garden", "kitchen garden", BackcheckVariable(name="crop", kind="text"), True),
        ("maize", "mahogany", BackcheckVariable(name="crop", kind="text", text_similarity=0.9), False),
        ("100", "104", BackcheckVariable(name="n", kind="numeric", abs_tolerance=5), True),
        ("100", "120", BackcheckVariable(name="n", kind="numeric", abs_tolerance=5), False),
        ("1000", "1200", BackcheckVariable(name="inc", kind="numeric", rel_tolerance=0.25), True),
        ("1000", "1300", BackcheckVariable(name="inc", kind="numeric", rel_tolerance=0.25), False),
        ("2026-03-02", "2026-03-04", BackcheckVariable(name="d", kind="date", date_tolerance_days=2), True),
        ("2026-03-02", "2026-03-06", BackcheckVariable(name="d", kind="date", date_tolerance_days=2), False),
    ],
)
def test_compare_pair_by_kind(a, b, spec, expected):
    match, _ = compare_pair(a, b, spec)
    assert match is expected


def test_missing_values_are_a_match_when_both_missing_and_mismatch_or_skip_otherwise():
    spec = BackcheckVariable(name="x", kind="numeric")
    assert compare_pair(None, None, spec)[0] is True
    assert compare_pair("12", None, spec, missing_as_value=True)[0] is False
    assert compare_pair("12", None, spec, missing_as_value=False)[0] is None


def test_ignored_values_are_not_compared():
    spec = BackcheckVariable(name="x", kind="numeric", ignore_values=[-99])
    assert compare_pair("-99", "5", spec)[0] is None


def backcheck_fixture():
    rows = []
    survey = survey_frame(10, ("E1", "E2", "E3"))
    survey["hh_size"] = ["4"] * len(survey)
    survey["water_source"] = ["piped"] * len(survey)
    cfg = make_cfg(
        roles={"submission_id": "_uuid", "respondent_id": "hhid", "enumerator_id": "enum", "cluster": "village"},
        backcheck={
            "sample_rate": 0.2,
            "min_per_enumerator": 2,
            "min_compared": 3,
            "variables": [
                {"name": "hh_size", "type": 1, "kind": "numeric"},
                {"name": "water_source", "type": 1, "kind": "categorical"},
            ],
        },
    )
    # re-interviews: E1's answers are all different (type 1 failure); E2 and E3 agree
    for _, r in survey.iterrows():
        bad = r["enum"] == "E1"
        rows.append({"hhid": r["hhid"], "hh_size": "9" if bad else "4", "water_source": "river" if bad else "piped"})
    return survey, cfg, pd.DataFrame(rows)


def test_compare_flags_enumerator_with_type1_disagreement():
    survey, cfg, bc = backcheck_fixture()
    enum = survey["enum"]
    report = compare_backchecks(survey, enum, bc, cfg, id_column="hhid")
    flagged = report.enumerator_summary[report.enumerator_summary["flagged"]]
    assert set(flagged["enumerator"]) == {"E1"}
    assert (flagged["type"] == 1).all()
    assert report.coverage["n_enumerators_backchecked"] == 3
    assert report.variable_summary["flagged"].all()
    warn = report.issues[report.issues["enumerator"] == "E1"]
    assert warn["message"].str.contains("lead for supervisory review").all()


def test_duplicate_and_unmatched_backcheck_records_are_reported():
    survey, cfg, bc = backcheck_fixture()
    extra = pd.concat(
        [bc, bc.iloc[[0]], pd.DataFrame([{"hhid": "ZZZ", "hh_size": "4", "water_source": "piped"}])], ignore_index=True
    )
    report = compare_backchecks(survey, survey["enum"], extra, cfg, id_column="hhid")
    msgs = " ".join(report.issues["message"])
    assert "duplicate back-check record" in msgs
    assert "do not match any survey record" in msgs


def test_missing_id_column_is_a_dataset_issue():
    survey, cfg, bc = backcheck_fixture()
    report = compare_backchecks(survey, survey["enum"], bc.rename(columns={"hhid": "other"}), cfg, id_column="nope")
    assert report.comparisons.empty
    assert report.issues["message"].str.contains("No matching ID column").any()


def test_missing_variable_is_reported_and_comparison_still_runs():
    survey, cfg, bc = backcheck_fixture()
    bc = bc.drop(columns=["water_source"])
    report = compare_backchecks(survey, survey["enum"], bc, cfg, id_column="hhid")
    assert report.issues["message"].str.contains("not found in both files").any()
    summary = report.variable_summary.set_index("variable")
    assert summary.loc["hh_size", "n_compared"] == len(survey)
    assert summary.loc["water_source", "n_compared"] == 0


def test_sample_is_stratified_by_enumerator_and_deterministic(small_synthetic):
    survey, _ = small_synthetic
    cfg = make_cfg(
        roles={
            "submission_id": "_uuid",
            "respondent_id": "hhid",
            "enumerator_id": "enum_id",
            "cluster": "village",
            "survey_date": "today",
        },
        backcheck={"sample_rate": 0.1, "min_per_enumerator": 2, "random_share": 0.5, "seed": 99},
    )
    result = run_checks(survey.data, cfg, run_id="bc")
    a = select_sample(result.context, result.issues, result.scorecard, cfg)
    b = select_sample(result.context, result.issues, result.scorecard, cfg)
    assert a.equals(b)
    per_enum = a.groupby("enumerator").size()
    assert (per_enum >= 2).all()
    assert set(a["reason"]) <= {"random", "risk"}
    assert a["record_id"].is_unique


def test_sample_excludes_listed_ids_and_includes_contact_columns():
    survey, cfg, bc = backcheck_fixture()
    cfg = make_cfg(
        roles={"submission_id": "_uuid", "respondent_id": "hhid", "enumerator_id": "enum", "cluster": "village"},
        backcheck={
            "sample_rate": 1.0,
            "min_per_enumerator": 1,
            "exclude_ids": ["H0001"],
            "contact_columns": ["hh_size"],
        },
    )
    result = run_checks(survey, cfg, run_id="bc2")
    sample = select_sample(result.context, result.issues, result.scorecard, cfg)
    assert "H0001" not in set(sample["respondent_id"])
    assert "hh_size" in sample.columns
    assert len(sample) == len(survey) - 1


def test_sample_on_empty_data_is_empty_with_schema():
    survey, cfg, _ = backcheck_fixture()
    result = run_checks(survey.iloc[0:0], cfg, run_id="bc3")
    sample = select_sample(result.context, result.issues, result.scorecard, cfg)
    assert sample.empty
    assert "record_id" in sample.columns
