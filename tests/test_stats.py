"""Statistical helpers: robust z, IQR, Benford, ICC, haversine and two-proportion z."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fieldguard import stats


def test_robust_z_matches_modified_z_and_handles_constant_series():
    x = pd.Series([10.0, 11.0, 12.0, 11.5, 10.5, 100.0])
    z = stats.robust_z(x)
    assert z.iloc[-1] > 3.5
    assert abs(z.iloc[0]) < 3.5
    const = stats.robust_z(pd.Series([5.0, 5.0, 5.0]))
    assert (const == 0).all()


def test_robust_z_mean_absolute_fallback_when_mad_is_zero():
    x = pd.Series([5.0, 5.0, 5.0, 5.0, 9.0])
    z = stats.robust_z(x)
    assert np.isfinite(z).all()
    assert z.iloc[-1] > 0


def test_iqr_bounds_and_k():
    lo, hi = stats.iqr_bounds(pd.Series(range(1, 101)), k=1.5)
    assert lo < 1 and hi > 100
    lo3, hi3 = stats.iqr_bounds(pd.Series(range(1, 101)), k=3.0)
    assert lo3 < lo and hi3 > hi


def test_first_digits_ignores_small_and_negative_values():
    d = stats.first_digits(pd.Series([1234.0, -560.0, 9.0, 0.0, np.nan]), min_value=10.0)
    assert d.tolist() == [1, 5]


def test_benford_mad_zero_for_exact_benford_distribution():
    digits = np.concatenate(
        [np.full(int(round(p * 10000)), d) for d, p in zip(range(1, 10), stats.BENFORD_FIRST, strict=True)]
    )
    assert stats.benford_mad(digits) < 0.001
    assert np.isnan(stats.benford_mad(np.array([], dtype=int)))


def test_benford_chi2_flags_skewed_digits():
    skew = np.full(500, 9)
    _, p = stats.benford_chi2(skew)
    assert p < 0.001


def test_nigrini_classes():
    assert stats.nigrini_class(0.004) == "close conformity"
    assert stats.nigrini_class(0.010) == "acceptable conformity"
    assert stats.nigrini_class(0.014) == "marginally acceptable"
    assert stats.nigrini_class(0.03) == "nonconformity"
    assert stats.nigrini_class(float("nan")) == "insufficient data"


def test_icc_near_zero_without_group_effect_and_high_with_one():
    rng = np.random.default_rng(3)
    groups = pd.Series(np.repeat(["a", "b", "c", "d"], 40))
    noise = pd.Series(rng.normal(0, 1, len(groups)))
    icc0, p0, k = stats.icc1(noise, groups)
    assert k == 4 and abs(icc0) < 0.15 and p0 > 0.01
    effect = groups.map({"a": 0.0, "b": 3.0, "c": 6.0, "d": 9.0}) + rng.normal(0, 0.5, len(groups))
    icc1, p1, _ = stats.icc1(pd.Series(effect), groups)
    assert icc1 > 0.8 and p1 < 1e-6


def test_icc_undefined_with_one_group():
    icc, p, k = stats.icc1(pd.Series([1.0, 2.0, 3.0, 4.0]), pd.Series(["a"] * 4))
    assert np.isnan(icc) and k < 2


def test_haversine_one_degree_latitude():
    d = stats.haversine_m(np.array([0.0]), np.array([36.0]), np.array([1.0]), np.array([36.0]))
    assert 110_000 < float(d[0]) < 112_000


def test_two_proportion_z_direction_and_degenerate_inputs():
    assert stats.two_proportion_z(30, 100, 10, 100) > 0
    assert stats.two_proportion_z(10, 100, 30, 100) < 0
    assert stats.two_proportion_z(0, 100, 0, 100) == 0.0
    assert np.isnan(stats.two_proportion_z(1, 0, 1, 10))
