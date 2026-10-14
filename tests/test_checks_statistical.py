"""Statistical checks: outliers (MAD, IQR, within enumerator, vectorised equivalence), don't-know rates,
heaping and Benford's law."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fieldguard.checks.builtin.statistical import OutlierParams, _grouped_outlier_mask, _outlier_mask
from tests.conftest import make_cfg, run_check, survey_frame


def test_mad_outlier_flagged_for_the_extreme_value():
    df = survey_frame(10, ("E1", "E2", "E3"))
    df.loc[5, "income"] = "900000"
    out = run_check("outliers", df, make_cfg(), {"variables": ["income"]})
    overall = out[(out["variable"] == "income") & out["message"].str.startswith("Outlier (MAD)")]
    assert overall["record_id"].tolist() == ["uuid-0006"]


def test_iqr_method_flags_the_same_extreme_value():
    df = survey_frame(10, ("E1", "E2", "E3"))
    df.loc[5, "income"] = "900000"
    out = run_check("outliers", df, make_cfg(), {"variables": ["income"], "method": "iqr"})
    assert "uuid-0006" in set(out["record_id"])


def test_within_enumerator_outlier_is_info_and_only_for_that_group():
    df = survey_frame(12, ("E1", "E2"))
    df["age"] = "30"
    df.loc[df["enum"] == "E1", "age"] = [str(v) for v in [30, 31, 32, 30, 31, 32, 30, 31, 32, 30, 31, 45]]
    df.loc[df["enum"] == "E2", "age"] = [str(v) for v in [60, 61, 62, 60, 61, 62, 60, 61, 62, 60, 61, 62]]
    out = run_check("outliers", df, make_cfg(), {"variables": ["age"], "threshold": 3.5})
    within = out[out["message"].str.contains("enumerator's own")]
    assert within["record_id"].tolist() == ["uuid-0012"]
    assert (within["severity"] == "info").all()


def test_log_transform_and_missing_variable_note():
    df = survey_frame(10, ("E1", "E2"))
    df.loc[0, "income"] = "1000000"
    out = run_check("outliers", df, make_cfg(), {"variables": ["income", "ghost"], "log_transform": ["income"]})
    assert out["message"].str.contains("log-scale").any()
    assert out["message"].str.contains("not found in data").any()


def test_grouped_mask_matches_per_group_loop_including_edge_groups():
    rng = np.random.default_rng(11)
    groups = pd.Series(rng.choice(["a", "b", "c", "d"], 300), name="e")
    x = pd.Series(np.exp(rng.normal(9, 1, 300)))
    x[rng.integers(0, 300, 25)] = np.nan
    x[groups == "d"] = 5.0  # constant group: MAD and mean deviation are zero
    x[groups == "c"] = np.nan  # empty group after NaNs
    x.iloc[10] = 9e9
    for method in ("mad", "iqr"):
        p = OutlierParams(method=method, min_n=10)
        loop_mask = pd.Series(False, index=x.index)
        loop_z = pd.Series(np.nan, index=x.index)
        for _, idx in groups.groupby(groups).indices.items():
            m, z = _outlier_mask(x.iloc[idx], p)
            loop_mask.iloc[idx] = m.to_numpy()
            loop_z.iloc[idx] = z.to_numpy()
        vec_mask, vec_z = _grouped_outlier_mask(x, groups, p)
        assert np.array_equal(loop_mask.to_numpy(), vec_mask.to_numpy()), method
        if method == "mad":
            both = loop_z.notna() & vec_z.notna()
            assert np.allclose(loop_z[both], vec_z[both])


def test_outliers_on_all_nan_column_do_not_crash():
    df = survey_frame(4, ("E1",))
    df["income"] = None
    out = run_check("outliers", df, make_cfg(), {"variables": ["income"]})
    assert out.empty or not out["message"].str.contains("Outlier").any()


def test_high_dont_know_rate_flags_the_enumerator():
    df = survey_frame(40, ("E1", "E2", "E3"))
    df.loc[df["enum"] == "E3", "income"] = "-99"
    out = run_check("nonresponse", df, make_cfg(), {"variables": ["age", "income"]})
    hit = out[out["message"].str.startswith("High don't know")]
    assert hit["enumerator"].tolist() == ["E3"]


def test_heaping_flags_enumerator_with_round_numbers():
    df = survey_frame(40, ("E1", "E2", "E3"))
    rng = np.random.default_rng(5)
    e1 = df["enum"] == "E1"
    df.loc[e1, "income"] = [str(int(v) * 500) for v in rng.integers(6, 19, e1.sum())]
    out = run_check("heaping", df, make_cfg(), {"variables": ["income"]})
    flagged = out[(out["enumerator"] == "E1") & out["message"].str.contains("round")]
    assert len(flagged) >= 1


def test_benford_flags_enumerator_with_first_digit_nine():
    df = survey_frame(120, ("E1", "E2", "E3"))
    rng = np.random.default_rng(9)
    e1 = df["enum"] == "E1"
    df.loc[e1, "income"] = [str(v) for v in rng.integers(9000, 9999, e1.sum())]
    others = ~e1
    df.loc[others, "income"] = [str(int(v)) for v in np.exp(rng.uniform(np.log(10), np.log(99999), others.sum()))]
    out = run_check("benford", df, make_cfg(), {"variables": ["income"], "min_n": 50})
    flagged = out[out["message"].str.startswith("First digits deviate")]
    assert "E1" in set(flagged["enumerator"])
    assert not {"E2", "E3"} & set(flagged["enumerator"])


def test_statistical_checks_on_empty_frame():
    empty = survey_frame(2, ("E1",)).iloc[0:0]
    for check in ("outliers", "nonresponse", "heaping", "benford"):
        assert run_check(check, empty).empty
