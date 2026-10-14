"""Config validation: pydantic models, YAML loading, path resolution and clear error messages."""

from __future__ import annotations

import pytest

from fieldguard.config import ConfigError, config_from_dict, dump_config, load_config
from tests.conftest import EXAMPLE_YAML, base_config


def test_example_project_validates_and_resolves_rules():
    cfg = load_config(EXAMPLE_YAML)
    assert cfg.project.platform == "kobo"
    assert cfg.variables.required_specs()[0].variable == "hhid"
    assert any(r.variable == "business_income" and r.when for r in cfg.variables.required_specs())
    assert cfg.params_for("outliers").method == "mad"
    assert cfg.scoring.rule_overrides["short_duration_share"]["hi"] == 0.45


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="extra|not permitted"):
        config_from_dict({"proejct": {"name": "typo"}})


def test_unknown_check_id_is_rejected_with_available_list():
    with pytest.raises(ConfigError, match="unknown check id"):
        config_from_dict({"checks": {"no_such_check": {}}})


def test_invalid_check_parameter_is_rejected():
    with pytest.raises(ConfigError, match="outliers"):
        config_from_dict({"checks": {"outliers": {"method": "zscore"}}})


def test_bad_timezone_is_rejected():
    with pytest.raises(ConfigError, match="timezone"):
        config_from_dict({"project": {"timezone": "Mars/Olympus"}})


def test_fieldwork_end_before_start_is_rejected():
    with pytest.raises(ConfigError, match="end_date"):
        config_from_dict({"project": {"fieldwork": {"start_date": "2026-05-01", "end_date": "2026-04-01"}}})


def test_constraint_needs_a_target():
    with pytest.raises(ConfigError, match="needs"):
        config_from_dict({"variables": {"constraints": [{"min": 1}]}})


def test_constraint_on_variable_needs_a_bound():
    with pytest.raises(ConfigError, match="min/max"):
        config_from_dict({"variables": {"constraints": [{"variable": "age"}]}})


def test_skip_rule_needs_a_consequence():
    with pytest.raises(ConfigError, match="then_empty"):
        config_from_dict({"variables": {"skip_logic": [{"if": "a == 'yes'"}]}})


def test_scorecard_flag_threshold_range_is_checked():
    with pytest.raises(ConfigError, match="flag_threshold"):
        config_from_dict({"scoring": {"flag_threshold": 250}})


def test_backcheck_rate_must_be_positive():
    with pytest.raises(ConfigError, match="sample_rate"):
        config_from_dict({"backcheck": {"sample_rate": 0}})


def test_empty_config_uses_defaults():
    cfg = config_from_dict(None)
    assert cfg.project.platform == "auto"
    assert cfg.outputs.formats == ["csv", "parquet", "excel", "html"]
    assert cfg.synthetic is False


def test_dump_and_reload_round_trip():
    cfg = config_from_dict(base_config())
    text = dump_config(cfg)
    again = config_from_dict(__import__("yaml").safe_load(text))
    assert again.roles.respondent_id == "hhid"
    assert again.variables.numeric == ["age", "income"]


def test_relative_paths_resolve_against_config_folder(tmp_path):
    cfg_file = tmp_path / "proj" / "config.yaml"
    cfg_file.parent.mkdir()
    cfg_file.write_text("data:\n  path: raw/export.csv\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.resolve_path(cfg.data.path) == (tmp_path / "proj" / "raw" / "export.csv").resolve()


def test_missing_config_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("project: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
    listy = tmp_path / "list.yaml"
    listy.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(listy)
