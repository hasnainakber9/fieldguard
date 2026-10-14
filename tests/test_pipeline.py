"""End-to-end checks on the pipeline: empty and all-NaN data, configuration notes, strict mode, incremental mode
and detection on a small SYNTHETIC survey."""

from __future__ import annotations

import numpy as np
import pytest

import fieldguard.checks.registry as registry
import fieldguard.pipeline as pipeline
from fieldguard.checks.registry import CheckParams, CheckSpec
from fieldguard.models import ISSUE_COLUMNS
from fieldguard.pipeline import run_checks
from tests.conftest import make_cfg, survey_frame


def test_empty_data_returns_issues_table_with_notice():
    df = survey_frame(2, ("E1",)).iloc[0:0]
    result = run_checks(df, make_cfg())
    assert result.n_records == 0
    assert list(result.issues.columns) == ISSUE_COLUMNS
    assert result.issues["message"].str.contains("No records in the data").any()
    assert (result.check_status["status"] != "error").all()


def test_all_nan_frame_runs_without_errors():
    df = survey_frame(4, ("E1", "E2")).astype("object")
    df[:] = None
    result = run_checks(df, make_cfg())
    assert (result.check_status["status"] != "error").all()
    assert result.n_records == len(df)


def test_missing_configured_columns_are_noted_as_configuration_warnings():
    df = survey_frame(3, ("E1",))
    cfg = make_cfg(roles={"submission_id": "_uuid", "enumerator_id": "enum", "respondent_id": "no_such_column"})
    result = run_checks(df, cfg)
    notes = result.issues[(result.issues["check_id"] == "setup") & (result.issues["severity"] == "warning")]
    assert notes["message"].str.contains("no_such_column").any()


def test_clean_data_has_no_record_level_critical_issues():
    result = run_checks(survey_frame(6, ("E1", "E2", "E3")), make_cfg())
    crit_records = result.issues[(result.issues["severity"] == "critical") & (result.issues["level"] == "record")]
    assert crit_records.empty


class _Boom(CheckParams):
    pass


def _failing_registry(monkeypatch):
    real = registry.get_registry()

    def boom(ctx, params):  # noqa: ANN001 - check signature
        raise RuntimeError("simulated failure")

    spec = CheckSpec("boom_check", boom, _Boom, "Boom", "Always fails (test only)", "test")
    extended = {**real, "boom_check": spec}
    monkeypatch.setattr(registry, "get_registry", lambda: extended)
    monkeypatch.setattr(pipeline, "get_registry", lambda: extended)
    return extended


def test_check_failure_is_recorded_not_fatal(monkeypatch):
    _failing_registry(monkeypatch)
    result = run_checks(survey_frame(3, ("E1",)), make_cfg())
    status = result.check_status.set_index("check_id")["status"]
    assert status["boom_check"] == "error"
    assert result.issues["message"].str.contains("Check failed and was skipped").any()


def test_strict_mode_raises_on_check_failure(monkeypatch):
    _failing_registry(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated failure"):
        run_checks(survey_frame(3, ("E1",)), make_cfg(), strict=True)


def test_only_and_skip_select_checks():
    result = run_checks(survey_frame(3, ("E1",)), make_cfg(), only=["duplicate_ids"])
    ran = set(result.check_status.loc[result.check_status["status"] == "ok", "check_id"])
    assert ran == {"duplicate_ids"}
    result2 = run_checks(survey_frame(3, ("E1",)), make_cfg(), skip=["gps", "timestamps"])
    skipped = set(result2.check_status.loc[result2.check_status["status"] == "skipped", "check_id"])
    assert {"gps", "timestamps"} <= skipped


def test_incremental_mode_reports_record_issues_only_for_new_submissions():
    df = survey_frame(4, ("E1",))
    df.loc[1, "hhid"] = df.loc[0, "hhid"]  # duplicate respondent ID across two submissions
    full = run_checks(df, make_cfg())
    dup_full = set(full.issues.loc[full.issues["check_id"] == "duplicate_ids", "record_id"])
    assert {"uuid-0001", "uuid-0002"} <= dup_full
    seen = {"uuid-0001", "uuid-0002", "uuid-0003", "uuid-0004"}
    inc = run_checks(df, make_cfg(), seen_ids=seen)
    assert inc.mode == "incremental" and inc.n_new_records == 0
    assert not (inc.issues["level"] == "record").any()
    partial = run_checks(df, make_cfg(), seen_ids={"uuid-0003", "uuid-0004"})
    assert set(partial.new_record_ids) == {"uuid-0001", "uuid-0002"}
    assert set(partial.issues.loc[partial.issues["level"] == "record", "record_id"]) <= {"uuid-0001", "uuid-0002"}


def test_scoring_disabled_gives_empty_scorecard():
    cfg = make_cfg(scoring={"enabled": False})
    result = run_checks(survey_frame(3, ("E1", "E2")), cfg)
    assert result.scorecard.empty


def test_synthetic_flag_propagates_and_summary_is_serialisable(synthetic_cfg, small_synthetic):
    survey, _ = small_synthetic
    result = run_checks(survey.data, synthetic_cfg)
    assert result.synthetic is True
    summary = result.summary()
    assert summary["synthetic"] is True and summary["n_records"] == len(survey.data)
    assert set(summary["issues_by_severity"]) == {"critical", "warning", "info"}


def test_detection_on_small_synthetic_survey_finds_fabricators(synthetic_cfg, small_synthetic):
    survey, _ = small_synthetic
    result = run_checks(survey.data, synthetic_cfg)
    truth = set(survey.labels.loc[survey.labels["is_fabricator"], "enumerator"])
    flagged = set(result.scorecard.loc[result.scorecard["flagged"].astype(bool), "enumerator"])
    assert truth, "the fixture must contain fabricating enumerators"
    assert flagged & truth, f"no fabricator flagged: flagged={flagged}, truth={truth}"
    assert np.isfinite(result.scorecard["risk_score"]).all()
