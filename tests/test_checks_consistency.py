"""Consistency checks: range and constraint violations, and skip logic."""

from __future__ import annotations

import pandas as pd

from tests.conftest import make_cfg, run_check


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_uuid": ["a", "b", "c", "d", "e"],
            "hhid": ["H1", "H2", "H3", "H4", "H5"],
            "enum": ["E1"] * 5,
            "age": ["34", "12", "-99", "95", "not a number"],
            "sex": ["male", "female", "other", "female", "male"],
            "hh_size": ["4", "2", "3", "3", "5"],
            "n_children": ["2", "3", "1", "0", "1"],
            "has_business": ["yes", "no", "no", "yes", "yes"],
            "business_income": ["500", "300", None, None, "100"],
            "owns_land": ["no", "yes", "no", "yes", "no"],
            "land_size": ["2", "1", None, "3", "4"],
        }
    )


def test_hard_range_violation_and_special_codes_ignored():
    cfg = make_cfg(variables={"constraints": [{"variable": "age", "min": 18, "max": 100}]})
    out = run_check("constraints", frame(), cfg)
    flagged = set(out[out["message"].str.contains("Outside hard range")]["record_id"])
    assert flagged == {"b"}  # -99 is a special code and must not be treated as an age
    assert out["message"].str.contains("Non-numeric value").any()


def test_soft_range_is_informational_and_only_inside_hard_range():
    cfg = make_cfg(variables={"constraints": [{"variable": "age", "min": 18, "max": 100, "soft_max": 85}]})
    out = run_check("constraints", frame(), cfg)
    soft = out[out["message"].str.contains("expected range")]
    assert soft["record_id"].tolist() == ["d"]
    assert (soft["severity"] == "info").all()


def test_allowed_values_list():
    cfg = make_cfg(variables={"constraints": [{"variable": "sex", "allowed": ["male", "female"]}]})
    out = run_check("constraints", frame(), cfg)
    assert out["record_id"].tolist() == ["c"]


def test_cross_variable_expression_only_where_answered():
    cfg = make_cfg(variables={"constraints": [{"id": "kids", "expression": "n_children <= hh_size"}]})
    out = run_check("constraints", frame(), cfg)
    assert out["record_id"].tolist() == ["b"]
    assert out["message"].str.contains("[kids]", regex=False).all()


def test_constraint_when_scope_limits_rows():
    cfg = make_cfg(
        variables={
            "constraints": [{"variable": "land_size", "min": 2, "when": "owns_land == 'yes'", "severity": "critical"}]
        }
    )
    out = run_check("constraints", frame(), cfg)
    assert out["record_id"].tolist() == ["b"]
    assert (out["severity"] == "critical").all()


def test_constraint_on_missing_column_is_skipped_with_note():
    cfg = make_cfg(variables={"constraints": [{"variable": "nope", "min": 0}]})
    out = run_check("constraints", frame(), cfg)
    assert out["message"].str.contains("not found in data").any()


def test_skip_logic_then_empty_then_required_then_expression():
    cfg = make_cfg(
        variables={
            "skip_logic": [
                {"id": "biz_empty", "if": "has_business == 'no'", "then_empty": ["business_income"]},
                {"id": "biz_req", "if": "${has_business} = 'yes'", "then_required": ["business_income"]},
                {"id": "land_expr", "if": "owns_land == 'no'", "then": "empty(land_size)"},
            ]
        }
    )
    out = run_check("skip_logic", frame(), cfg)
    biz_empty = out[out["message"].str.startswith("[biz_empty]")]
    assert set(biz_empty["record_id"]) == {"b"}
    biz_req = out[out["message"].str.startswith("[biz_req]")]
    assert set(biz_req["record_id"]) == {"d"}
    land = out[out["message"].str.startswith("[land_expr]")]
    assert set(land["record_id"]) == {"a", "e"}  # owns_land == 'no' but land_size is filled


def test_skip_rule_with_unknown_column_is_reported():
    cfg = make_cfg(variables={"skip_logic": [{"id": "x", "if": "has_business == 'no'", "then_empty": ["ghost"]}]})
    out = run_check("skip_logic", frame(), cfg)
    assert out["message"].str.contains("not found in data").any()


def test_skip_logic_bad_expression_is_reported_not_fatal():
    cfg = make_cfg(
        variables={"skip_logic": [{"id": "bad", "if": "has_business.upper() == 'x'", "then_empty": ["land_size"]}]}
    )
    out = run_check("skip_logic", frame(), cfg)
    assert out["message"].str.contains("skipped").any()


def test_consistency_checks_on_empty_frame():
    empty = frame().iloc[0:0]
    cfg = make_cfg(
        variables={
            "constraints": [{"variable": "age", "min": 18}],
            "skip_logic": [{"if": "a == 'x'", "then_empty": ["b"]}],
        }
    )
    assert run_check("constraints", empty, cfg).empty
    # configuration problems (unknown column) are still reported; no record-level issues on empty data
    skip = run_check("skip_logic", empty, cfg)
    assert (skip["level"] != "record").all()
