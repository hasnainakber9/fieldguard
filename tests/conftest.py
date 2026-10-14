"""Shared fixtures and helpers. Nothing here writes outside pytest's tmp_path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fieldguard.checks.context import build_context
from fieldguard.checks.registry import get_check
from fieldguard.config import FieldGuardConfig, config_from_dict
from fieldguard.pipeline import finalize_issues
from fieldguard.synthetic import SyntheticSpec, generate_survey, synthetic_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = ROOT / "examples" / "example_project.yaml"


def base_config(**overrides: Any) -> dict[str, Any]:
    """A small, valid config used by most check tests (Kobo-style column names)."""
    cfg: dict[str, Any] = {
        "project": {"name": "Unit test project", "platform": "kobo", "timezone": "Africa/Nairobi"},
        "roles": {
            "submission_id": "_uuid",
            "respondent_id": "hhid",
            "enumerator_id": "enum",
            "cluster": "village",
            "start": "start",
            "end": "end",
            "survey_date": "today",
            "gps": {"lat": "_gps_latitude", "lon": "_gps_longitude", "accuracy": "_gps_precision"},
        },
        "variables": {"numeric": ["age", "income"], "key": ["income"]},
    }
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def make_cfg(**overrides: Any) -> FieldGuardConfig:
    return config_from_dict(base_config(**overrides))


def run_check(
    check_id: str, df: pd.DataFrame, cfg: FieldGuardConfig | None = None, params: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Run one registered check on ``df`` and return its finalised issues table."""
    cfg = cfg or make_cfg()
    ctx = build_context(df, cfg, run_id="test")
    spec = get_check(check_id)
    p = spec.params_model.model_validate(params or {})
    out = spec.func(ctx, p)
    return finalize_issues(out, ctx)


def survey_frame(n_per_enum: int = 6, enumerators: tuple[str, ...] = ("E1", "E2", "E3")) -> pd.DataFrame:
    """Simple, clean survey frame (no issues by construction)."""
    rng = np.random.default_rng(0)
    rows = []
    k = 0
    for e in enumerators:
        for i in range(n_per_enum):
            k += 1
            start = pd.Timestamp("2026-03-02 09:00", tz="Africa/Nairobi") + pd.Timedelta(hours=2 * i + k * 0.01)
            end = start + pd.Timedelta(minutes=float(rng.uniform(40, 60)))
            rows.append(
                {
                    "_uuid": f"uuid-{k:04d}",
                    "hhid": f"H{k:04d}",
                    "enum": e,
                    "village": "V1" if i % 2 == 0 else "V2",
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "today": start.strftime("%Y-%m-%d"),
                    "age": str(int(rng.integers(20, 70))),
                    "income": str(int(rng.integers(3000, 9000))),
                    "_gps_latitude": f"{-0.40 + rng.normal(0, 0.002):.6f}",
                    "_gps_longitude": f"{36.95 + rng.normal(0, 0.002):.6f}",
                    "_gps_precision": f"{rng.uniform(3, 9):.1f}",
                }
            )
    return pd.DataFrame(rows).astype("object")


@pytest.fixture(scope="session")
def small_synthetic():  # type: ignore[no-untyped-def]
    """Small SYNTHETIC survey with 20% fabricating enumerators (test data only)."""
    spec = SyntheticSpec(n_enumerators=10, interviews_per_enumerator=18, contamination=0.2, seed=7)
    survey = generate_survey(spec)
    return survey, spec


@pytest.fixture
def synthetic_cfg(small_synthetic):  # type: ignore[no-untyped-def]
    _, spec = small_synthetic
    return config_from_dict(synthetic_config(spec=spec))
