"""Reports and privacy: redaction of names, phone numbers and GPS; pseudonyms; CSV, Parquet, Excel and HTML outputs;
synthetic labelling; output folder resolution."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from fieldguard.models import ISSUE_COLUMNS
from fieldguard.outputs import ALL_FORMATS, resolve_output_dir, write_run_outputs
from fieldguard.pipeline import run_checks
from fieldguard.privacy import (
    coarsen_coordinates,
    is_gps_variable,
    is_pii_variable,
    pseudonym,
    redact_issues,
    redact_phone_numbers,
)
from fieldguard.reports.html import render_html
from fieldguard.reports.svg import bar_scores, stacked_hbar
from tests.conftest import make_cfg, survey_frame


def issue_row(**kw) -> dict:
    base = dict.fromkeys(ISSUE_COLUMNS)
    base.update({"run_id": "r", "check_id": "x", "severity": "warning", "level": "record", "metric": None})
    base.update(kw)
    return base


def test_phone_numbers_redacted_but_dates_kept():
    assert redact_phone_numbers("call 0712345678 today") == "call [phone] today"
    assert redact_phone_numbers("+254 712 345 678") == "[phone]"
    assert redact_phone_numbers("interview on 2025-03-05") == "interview on 2025-03-05"


def test_coordinates_coarsened_to_requested_decimals():
    assert coarsen_coordinates("at -0.401234567, 36.950123", 3) == "at -0.401, 36.950"
    assert (
        coarsen_coordinates("ratio 0.12345", 3) == "ratio 0.123"
    )  # any 4+ decimal number is treated as coordinate precision


def test_pii_and_gps_variable_detection():
    assert is_pii_variable("resp/name", {"name"})
    assert is_pii_variable("group/phone, age", {"phone"})
    assert not is_pii_variable("named_area", {"name"})
    assert is_gps_variable("_gps_latitude") and is_gps_variable(None, "gps")
    assert not is_gps_variable("income")


def test_pseudonyms_are_salted_and_stable():
    assert pseudonym("E001", "s1", "enum") == pseudonym("E001", "s1", "enum")
    assert pseudonym("E001", "s1", "enum") != pseudonym("E001", "s2", "enum")
    assert pseudonym("E001", "s1", "enum").startswith("enum-")


def test_redact_issues_removes_pii_values_and_phone_numbers():
    cfg = make_cfg(privacy={"pii_columns": ["name", "phone"], "redact_pii_in_reports": True})
    issues = pd.DataFrame(
        [
            issue_row(variable="resp/name", value="Jane Doe", message="Name missing"),
            issue_row(variable="income", value="5000", message="Caller 0712345678 gave 5000"),
        ]
    )
    out = redact_issues(issues, cfg)
    assert out.loc[0, "value"] == "[redacted]"
    assert "0712345678" not in out.loc[1, "message"]
    assert "[phone]" in out.loc[1, "message"]


def test_redact_issues_coarsens_gps_messages():
    cfg = make_cfg(privacy={"gps_decimals_in_reports": 3})
    issues = pd.DataFrame(
        [issue_row(check_id="gps", variable="gps", value="-0.401234,36.950987", message="at -0.401234, 36.950987")]
    )
    out = redact_issues(issues, cfg)
    assert out.loc[0, "value"] == "-0.401,36.951"
    assert "-0.401234" not in out.loc[0, "message"]


def test_redact_issues_pseudonymises_ids_when_enabled(monkeypatch):
    monkeypatch.setenv("FIELDGUARD_HASH_SALT", "unit-test-salt")
    cfg = make_cfg(privacy={"hash_respondent_ids": True, "pseudonymize_enumerators": True})
    issues = pd.DataFrame([issue_row(record_id="uuid-1", respondent_id="H1", enumerator="E001", message="m")])
    out = redact_issues(issues, cfg)
    assert out.loc[0, "record_id"].startswith("resp-")
    assert out.loc[0, "enumerator"].startswith("enum-")
    assert "E001" not in out.loc[0, "enumerator"]


def test_redact_issues_keeps_schema_for_empty_input():
    out = redact_issues(pd.DataFrame(), make_cfg())
    assert list(out.columns) == ISSUE_COLUMNS and out.empty


@pytest.fixture
def run_result(small_synthetic, synthetic_cfg):  # type: ignore[no-untyped-def]
    survey, _ = small_synthetic
    return run_checks(survey.data, synthetic_cfg, run_id="fg-test-run"), synthetic_cfg


def test_write_all_formats_with_synthetic_prefix(tmp_path, run_result):
    result, cfg = run_result
    written = write_run_outputs(result, cfg, tmp_path, formats=list(ALL_FORMATS))
    names = {p.name for p in written.values()}
    assert "synthetic_issues.csv" in names
    assert "synthetic_report.html" in names
    assert "synthetic_fieldguard_workbook.xlsx" in names
    assert "synthetic_issues.parquet" in names
    assert "synthetic_run_summary.json" in names
    summary = json.loads(written["summary_json"].read_text(encoding="utf-8"))
    assert summary["synthetic"] is True
    assert "SYNTHETIC" in summary["synthetic_label"]


def test_csv_issues_are_redacted_and_schema_stable(tmp_path, run_result):
    result, cfg = run_result
    written = write_run_outputs(result, cfg, tmp_path, formats=["csv"])
    text = written["issues_csv"].read_text(encoding="utf-8")
    assert "Synthetic Person" not in text
    assert not re.search(r"\b07\d{8}\b", text)
    header = text.splitlines()[0].split(",")
    assert header == ISSUE_COLUMNS


def test_html_report_is_self_contained_and_labelled(tmp_path, run_result):
    result, cfg = run_result
    html = render_html(result, cfg)
    assert "SYNTHETIC DATA" in html
    assert "<script" not in html
    assert not re.search(r'(src|href)="https?://', html)
    assert "Flags are leads" in html or "lead for supervisory review" in html
    assert "Synthetic Person" not in html


def test_workbook_has_readme_scorecard_and_check_sheets(tmp_path, run_result):
    pytest.importorskip("openpyxl")
    from openpyxl import load_workbook

    result, cfg = run_result
    written = write_run_outputs(result, cfg, tmp_path, formats=["excel"])
    wb = load_workbook(written["workbook"], read_only=True)
    names = wb.sheetnames
    assert "README" in names and "scorecard" in names and "check_status" in names
    assert all(len(n) <= 31 for n in names)
    status_ids = [row[0] for row in wb["check_status"].iter_rows(min_row=2, values_only=True) if row[0]]
    assert status_ids and set(status_ids) <= set(names), "every check that ran needs its own sheet"
    readme = list(wb["README"].iter_rows(values_only=True))
    assert any(v and "Synthetic" in str(v) for row in readme for v in row)


def test_parquet_metadata_marks_synthetic(tmp_path, run_result):
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")
    result, cfg = run_result
    written = write_run_outputs(result, cfg, tmp_path, formats=["parquet"])
    meta = pyarrow_parquet.read_schema(written["issues_parquet"]).metadata
    assert meta[b"synthetic"] == b"true"
    frame = pd.read_parquet(written["issues_parquet"])
    assert list(frame.columns) == ISSUE_COLUMNS


def test_missing_pyarrow_skips_only_parquet(tmp_path, run_result, monkeypatch):
    import fieldguard.outputs as outputs

    def no_pyarrow(*args, **kwargs):
        raise ImportError("Parquet output needs pyarrow: pip install 'fieldguard[parquet]'")

    monkeypatch.setattr(outputs, "write_parquet", no_pyarrow)
    result, cfg = run_result
    written = write_run_outputs(result, cfg, tmp_path, formats=["csv", "parquet"])
    assert "issues_csv" in written and "issues_parquet" not in written


def test_unknown_output_format_is_rejected(tmp_path, run_result):
    result, cfg = run_result
    with pytest.raises(ValueError, match="unknown output format"):
        write_run_outputs(result, cfg, tmp_path, formats=["pdf"])


def test_output_folder_resolution(monkeypatch, tmp_path):
    cfg = make_cfg(project={"name": "Pilot Survey"})
    monkeypatch.delenv("FIELDGUARD_REPORTS_DIR", raising=False)
    assert resolve_output_dir(cfg, "r1", tmp_path / "x") == tmp_path / "x"
    default = resolve_output_dir(cfg, "r1")
    assert default.parts[-2:] == ("pilot-survey", "r1")
    monkeypatch.setenv("FIELDGUARD_REPORTS_DIR", str(tmp_path / "reports"))
    assert resolve_output_dir(cfg, "r2") == tmp_path / "reports" / "pilot-survey" / "r2"


def test_svg_labels_are_escaped_and_empty_charts_are_safe():
    chart = stacked_hbar([("<script>x</script>", {"critical": 2, "warning": 1})])
    assert "<script>" not in chart and "&lt;script&gt;" in chart
    assert "No issues" in stacked_hbar([])
    scores = bar_scores([("E1", 120.0, True), ("E2", -5.0, False)])
    assert "var(--sev-critical)" in scores
    assert "No enumerators" in bar_scores([])


def test_render_html_handles_result_without_scores(tmp_path):
    df = survey_frame(4, ("E1",))
    cfg = make_cfg(roles={"submission_id": "_uuid"}, scoring={"enabled": False})
    result = run_checks(df, cfg, run_id="fg-noscore")
    html = render_html(result, cfg)
    assert "No enumerator scores" in html
    assert "<html" in html and "FieldGuard data-quality report" in html
