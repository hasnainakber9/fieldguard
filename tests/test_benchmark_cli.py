"""Benchmark metrics, a small benchmark run, the command-line interface and the Grafana dashboard JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from fieldguard.benchmark import evaluate_scorecard, run_benchmark
from fieldguard.cli import app

ROOT = Path(__file__).resolve().parents[1]


def card(flags: dict[str, bool], risk: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"enumerator": e, "flagged": flags[e], "risk_score": risk[e], "rule_score": risk[e], "anomaly_score": risk[e]}
        for e in flags
    ]
    return pd.DataFrame(rows)


def labels(truth: dict[str, bool]) -> pd.DataFrame:
    return pd.DataFrame({"enumerator": list(truth), "is_fabricator": list(truth.values())})


def test_precision_recall_f1_on_known_confusion_matrix():
    truth = {"E1": True, "E2": True, "E3": False, "E4": False}
    flags = {"E1": True, "E2": False, "E3": True, "E4": False}
    risk = {"E1": 90.0, "E2": 20.0, "E3": 70.0, "E4": 10.0}
    m = evaluate_scorecard(card(flags, risk), labels(truth))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["roc_auc_risk_score"] == pytest.approx(0.75)


def test_f1_is_zero_when_every_flag_is_wrong_and_a_fabricator_is_missed():
    truth = {"E1": True, "E2": False}
    flags = {"E1": False, "E2": True}
    m = evaluate_scorecard(card(flags, {"E1": 1.0, "E2": 2.0}), labels(truth))
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)
    assert m["precision"] == pytest.approx(0.0) and m["recall"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.0)


def test_auc_undefined_with_one_class_and_no_flags_gives_nan():
    truth = {"E1": False, "E2": False}
    flags = {"E1": False, "E2": False}
    m = evaluate_scorecard(card(flags, {"E1": 1.0, "E2": 2.0}), labels(truth))
    assert math.isnan(m["roc_auc_risk_score"])
    assert math.isnan(m["precision"]) and math.isnan(m["recall"])


def test_empty_inputs_are_handled():
    m = evaluate_scorecard(pd.DataFrame(), pd.DataFrame())
    assert m["n_enumerators"] == 0


def test_small_benchmark_writes_synthetic_labelled_outputs(tmp_path):
    runs = run_benchmark(tmp_path, rates=(0.2,), seeds=(3,), n_enumerators=8, interviews=10)
    assert len(runs) == 1
    for name in ("synthetic_benchmark_runs.csv", "synthetic_benchmark_summary.csv", "synthetic_benchmark.json"):
        assert (tmp_path / name).exists()
    payload = json.loads((tmp_path / "synthetic_benchmark.json").read_text(encoding="utf-8"))
    assert "SYNTHETIC" in payload["label"]
    assert payload["summary"][0]["contamination"] == 0.2


# ----------------------------------------------------------------------------- CLI
@pytest.fixture
def cli(monkeypatch):
    monkeypatch.delenv("FIELDGUARD_REPORTS_DIR", raising=False)
    monkeypatch.delenv("FIELDGUARD_DATABASE_URL", raising=False)
    return CliRunner()


def test_help_lists_the_commands(cli):
    result = cli.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "demo", "backcheck", "benchmark", "fetch", "version"):
        assert cmd in result.output


def test_version_command(cli):
    result = cli.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == "0.1.0"


def test_demo_writes_labelled_outputs(cli, tmp_path):
    out = tmp_path / "demo"
    result = cli.invoke(app, ["demo", "--out", str(out), "--enumerators", "6", "--interviews", "8", "--seed", "5"])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert (out / "synthetic_report.html").exists()
    assert (out / "synthetic_answer_key_enumerators.csv").exists()
    assert (out / "synthetic_backcheck_coverage.json").exists()


def test_run_on_demo_export_with_saved_config(cli, tmp_path):
    out = tmp_path / "demo"
    assert (
        cli.invoke(
            app, ["demo", "--out", str(out), "--enumerators", "6", "--interviews", "8", "--no-backcheck"]
        ).exit_code
        == 0
    )
    res = cli.invoke(
        app,
        ["run", "--config", str(out / "synthetic_config.yaml"), "--out", str(tmp_path / "rerun"), "--formats", "csv"],
    )
    assert res.exit_code == 0, res.output
    assert "SYNTHETIC DATA" in res.output
    assert (tmp_path / "rerun" / "synthetic_issues.csv").exists()


def test_run_without_data_file_fails_cleanly(cli):
    result = cli.invoke(app, ["run", "--config", str(ROOT / "examples" / "example_project.yaml"), "--no-sink"])
    assert result.exit_code == 2
    assert "data file not found" in result.output or "not found" in result.output


def test_run_incremental_state_round_trip(cli, tmp_path):
    out = tmp_path / "demo"
    assert (
        cli.invoke(
            app, ["demo", "--out", str(out), "--enumerators", "6", "--interviews", "8", "--no-backcheck"]
        ).exit_code
        == 0
    )
    cfg_text = (
        (out / "synthetic_config.yaml")
        .read_text(encoding="utf-8")
        .replace("state_dir: .fieldguard_state", f"state_dir: {tmp_path / 'state'}")
    )
    cfg_path = out / "incremental.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    first = cli.invoke(
        app, ["run", "--config", str(cfg_path), "--incremental", "--formats", "csv", "--out", str(tmp_path / "r1")]
    )
    second = cli.invoke(
        app, ["run", "--config", str(cfg_path), "--incremental", "--formats", "csv", "--out", str(tmp_path / "r2")]
    )
    assert first.exit_code == 0 and second.exit_code == 0, second.output
    assert "incremental state: file" in second.output
    assert "new 0" in second.output


def test_backcheck_select_writes_field_only_sample(cli, tmp_path):
    out = tmp_path / "demo"
    assert (
        cli.invoke(
            app, ["demo", "--out", str(out), "--enumerators", "6", "--interviews", "8", "--no-backcheck"]
        ).exit_code
        == 0
    )
    res = cli.invoke(app, ["backcheck", "--config", str(out / "synthetic_config.yaml"), "--out", str(tmp_path / "bc")])
    assert res.exit_code == 0, res.output
    assert "FIELD ONLY" in res.output
    assert (tmp_path / "bc" / "synthetic_backcheck_sample_FIELD_ONLY.csv").exists()


def test_fetch_without_source_is_refused(cli, tmp_path):
    res = cli.invoke(
        app, ["fetch", "--config", str(ROOT / "examples" / "example_project.yaml"), "--out", str(tmp_path / "raw.csv")]
    )
    assert res.exit_code == 2
    assert "data.source is not set" in res.output


def test_fetch_missing_credentials_names_variable_only(cli, tmp_path, monkeypatch):
    monkeypatch.delenv("KOBO_TOKEN", raising=False)
    cfg = tmp_path / "kobo.yaml"
    cfg.write_text("data:\n  source:\n    type: kobo\n    asset_uid: aXYZ\n", encoding="utf-8")
    res = cli.invoke(app, ["fetch", "--config", str(cfg), "--out", str(tmp_path / "raw.csv")])
    assert res.exit_code == 2
    assert "KOBO_TOKEN" in res.output


def test_benchmark_quick_cli(cli, tmp_path):
    res = cli.invoke(app, ["benchmark", "--quick", "--enumerators", "6", "--out", str(tmp_path / "bm")])
    assert res.exit_code == 0, res.output
    assert "SYNTHETIC BENCHMARK" in res.output
    assert (tmp_path / "bm" / "synthetic_benchmark.json").exists()


def test_module_entry_point_exists():
    import fieldguard.__main__ as entry  # noqa: F401

    assert callable(entry.main)


# ----------------------------------------------------------------------------- Grafana
def test_grafana_dashboard_uses_datasource_variable_and_fieldguard_schema():
    dash = json.loads((ROOT / "integrations" / "grafana" / "fieldguard.json").read_text(encoding="utf-8"))
    names = [v["name"] for v in dash["templating"]["list"]]
    assert "DS_POSTGRES" in names and "project" in names
    sql = [t["rawSql"] for p in dash["panels"] for t in p.get("targets", [])]
    assert sql and all("fieldguard." in s for s in sql)
    text = json.dumps(dash)
    assert "password" not in text.lower() and "postgresql://" not in text
    assert "not proof of misconduct" in text
