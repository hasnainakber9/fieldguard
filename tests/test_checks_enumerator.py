"""Respondent- and enumerator-level checks: straight-lining, low variance, enumerator effects (ICC, means, variance)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.conftest import make_cfg, run_check, survey_frame

GRID = ["wb_1", "wb_2", "wb_3", "wb_4"]


def grid_frame() -> pd.DataFrame:
    df = survey_frame(8, ("E1", "E2"))
    rng = np.random.default_rng(2)
    for c in GRID:
        df[c] = [str(v) for v in rng.integers(1, 6, len(df))]
    e1 = df["enum"] == "E1"
    for c in GRID:
        df.loc[e1, c] = "4"  # straight-lined for every E1 interview
    df.loc[df.index[0], GRID] = ["3", "3", "3", "4"]  # very low variance, not straight-lined
    return df


def grid_cfg():
    return make_cfg(variables={"grids": {"wellbeing": {"columns": GRID, "min_items": 4}}, "numeric": ["age"]})


def test_straightlining_flagged_per_record_for_identical_grid_answers():
    out = run_check("straightlining", grid_frame(), grid_cfg())
    straight = out[out["message"].str.startswith("Same answer to all")]
    assert set(straight["record_id"]) == {f"uuid-{i:04d}" for i in range(2, 9)}  # row 1 is the low-variance case
    assert (straight["severity"] == "info").all()


def test_low_variance_grid_is_reported_but_not_as_straight_lined():
    out = run_check("straightlining", grid_frame(), grid_cfg())
    low = out[out["message"].str.startswith("Very low variation")]
    assert "uuid-0001" in set(low["record_id"])


def test_straightlining_enumerator_share_when_pattern_is_concentrated():
    df = grid_frame()
    out = run_check(
        "straightlining", df, grid_cfg(), {"min_n": 5, "z_threshold": 2.0, "enumerator_share_threshold": 0.5}
    )
    enum_level = out[out["level"] == "enumerator"]
    assert enum_level["enumerator"].tolist() == ["E1"]


def test_text_likert_grid_is_coded_and_checked():
    df = grid_frame()
    for c in GRID:
        df[c] = (
            df[c]
            .map({"4": "agree", "3": "neutral", "5": "strongly agree", "1": "disagree", "2": "disagree"})
            .fillna("neutral")
        )
    e1 = df["enum"] == "E1"
    df.loc[e1, GRID] = "agree"
    out = run_check("straightlining", df, grid_cfg())
    assert set(out[out["message"].str.startswith("Same answer")]["record_id"]) >= {f"uuid-{i:04d}" for i in range(1, 9)}


def test_no_grid_configured_returns_nothing():
    assert run_check("straightlining", grid_frame(), make_cfg()).empty


def test_enumerator_effect_icc_flags_large_between_enumerator_variance():
    df = survey_frame(20, ("E1", "E2", "E3"))
    rng = np.random.default_rng(4)
    levels = {"E1": 3000, "E2": 6000, "E3": 9000}
    df["income"] = [str(levels[e] + int(rng.normal(0, 300))) for e in df["enum"]]
    out = run_check("enumerator_effects", df, make_cfg(), {"variables": ["income"]})
    warn = out[out["message"].str.startswith("Enumerators explain")]
    assert len(warn) == 1
    assert "ICC=" in warn["message"].iloc[0]


def test_enumerator_effect_is_quiet_without_group_structure():
    df = survey_frame(20, ("E1", "E2", "E3"))
    rng = np.random.default_rng(8)
    df["income"] = [str(int(v)) for v in rng.normal(6000, 800, len(df))]
    out = run_check("enumerator_effects", df, make_cfg(), {"variables": ["income"]})
    assert not out["message"].str.startswith("Enumerators explain").any()


def test_categorical_association_uses_cramers_v():
    df = survey_frame(20, ("E1", "E2", "E3"))
    df["sex"] = [
        "male" if e == "E1" else "female" if e == "E2" else ("male" if i % 2 else "female")
        for i, e in enumerate(df["enum"])
    ]
    out = run_check("enumerator_effects", df, make_cfg(), {"variables": ["sex"]})
    assert out["message"].str.contains("Cramér's V").any()


def test_enumerator_effects_without_enumerator_role_is_empty():
    df = survey_frame(5, ("E1", "E2"))
    cfg = make_cfg(roles={"submission_id": "_uuid"})
    assert run_check("enumerator_effects", df, cfg, {"variables": ["income"]}).empty
