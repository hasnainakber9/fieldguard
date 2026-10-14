"""Multivariate enumerator-level anomaly score (Isolation Forest or LOF on aggregated features)."""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Metrics fed to the anomaly model (enumerator-level aggregates emitted by the checks).
ANOMALY_FEATURES = [
    "short_duration_share",
    "median_duration_ratio",
    "straightline_share",
    "grid_sd_ratio",
    "variance_ratio",
    "benford_excess",
    "heaping_ratio",
    "gps_duplicate_share",
    "gps_dispersion_ratio",
    "gps_outside_share",
    "overlap_share",
    "offhours_share",
    "dk_rate",
    "refusal_rate",
    "near_duplicate_share",
    "answered_ratio",
    "mean_deviation",
    "outlier_rate",
    "max_per_day",
]


def robust_standardize(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    for col in features.columns:
        x = features[col].astype("float64")
        if x.notna().sum() < 3:
            continue
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        scale = mad if mad > 1e-9 else x.std(ddof=0)
        if not np.isfinite(scale) or scale <= 1e-9:
            continue
        out[col] = ((x - med) / scale).clip(-10, 10).fillna(0.0)
    return out


def anomaly_scores(features: pd.DataFrame, method: str = "isolation_forest", random_state: int = 42) -> pd.Series:
    """Return an anomaly score in [0, 100] per enumerator (NaN when it cannot be computed).

    Both models run single-job on purpose: scikit-learn ensembles spawned worker processes that cost
    several seconds on tiny (per-enumerator) data, and single-job runs are deterministic for a given seed.
    Isolation Forest: ``score_samples`` gives s in (0, 1] (~0.5 typical, -> 1 anomalous, Liu et al. 2008);
    mapped to 0-100 via (s - 0.5) / 0.25. LOF: (LOF - 1) / 1.5, clipped.
    """
    cols = [c for c in ANOMALY_FEATURES if c in features.columns]
    x = robust_standardize(features[cols]) if cols else pd.DataFrame(index=features.index)
    if x.shape[1] == 0 or len(x) < 3 or method == "none":
        return pd.Series(np.nan, index=features.index)
    values = x.to_numpy()
    if method == "lof":
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=max(2, min(20, len(x) - 1)), n_jobs=1)
        lof.fit(values)
        raw = -lof.negative_outlier_factor_
        score = np.clip((raw - 1.0) / 1.5, 0, 1) * 100
    else:
        from sklearn.ensemble import IsolationForest

        model = IsolationForest(n_estimators=200, random_state=random_state, n_jobs=1, max_samples="auto")
        model.fit(values)
        raw = -model.score_samples(values)
        score = np.clip((raw - 0.5) / 0.25, 0, 1) * 100
    return pd.Series(score, index=features.index, dtype="float64")
