"""answer_regularity: rating spread, household spread and typicality against peers (SYNTHETIC data only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fieldguard.checks.builtin.regularity import SIGNALS, _skip_dependent, peer_z
from fieldguard.checks.context import build_context
from fieldguard.checks.registry import get_check
from fieldguard.config import ConfigError, config_from_dict
from fieldguard.pipeline import finalize_issues, run_checks
from fieldguard.synthetic import SVC_ITEMS, WB_ITEMS, SyntheticSpec, generate_survey, synthetic_config

RATINGS = WB_ITEMS + SVC_ITEMS
HOUSEHOLD = [
    "resp/age",
    "hh/hh_size",
    "hh/n_children",
    "liv/monthly_income",
    "exp/food_exp",
    "exp/nonfood_exp",
    "water/minutes_to_water",
    "assets/n_phones",
]


def survey(seed: int, n_enumerators: int = 30, interviews: int = 25, contamination: float = 0.0):
    spec = SyntheticSpec(
        n_enumerators=n_enumerators, interviews_per_enumerator=interviews, contamination=contamination, seed=seed
    )
    return generate_survey(spec).data.copy(), spec


def config(spec: SyntheticSpec, **params):
    raw = synthetic_config(spec=spec)
    raw["checks"]["answer_regularity"].update(params)
    return config_from_dict(raw)


def regularity(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run only answer_regularity with the parameters in ``cfg``; return (issues, metrics)."""
    ctx = build_context(df, cfg, run_id="test")
    out = get_check("answer_regularity").func(ctx, cfg.params_for("answer_regularity"))
    return finalize_issues(out, ctx), ctx.metrics_frame()


def reported(issues: pd.DataFrame) -> pd.DataFrame:
    """Enumerator-level results in either direction."""
    return issues[issues["level"] == "enumerator"]


def too_regular(issues: pd.DataFrame) -> pd.DataFrame:
    return issues[(issues["level"] == "enumerator") & (issues["severity"] == "warning")]


def notes(issues: pd.DataFrame) -> pd.Series:
    return issues.loc[issues["level"] == "dataset", "message"]


def excess(metrics: pd.DataFrame) -> pd.Series:
    return metrics[metrics["metric"] == "regularity_excess"].set_index("enumerator")["value"]


def make_too_regular(df: pd.DataFrame, enumerator: str, seed: int = 0) -> pd.DataFrame:
    """Rewrite one enumerator's answers the way a too-regular fabricator would: households close to the survey
    median, ratings of 3 or 4 throughout."""
    rng = np.random.default_rng(seed)
    rows = df.index[df["enum_id"] == enumerator]
    for col in HOUSEHOLD:
        med = pd.to_numeric(df[col], errors="coerce").median()
        values = med * np.exp(rng.normal(0, 0.05, len(rows)))
        df.loc[rows, col] = [str(int(round(v))) for v in values]
    for item in RATINGS:
        df.loc[rows, item] = [str(v) for v in rng.choice([3, 4], len(rows), p=[0.6, 0.4])]
    return df


# ----------------------------------------------------------------------------- behaviour
def test_honest_surveys_rarely_flag():
    n_reported = n_scored = n_compared = 0
    for seed in (11, 12, 13, 14, 15):
        df, spec = survey(seed)
        issues, metrics = regularity(df, config(spec))
        n_compared += len(excess(metrics))
        n_reported += reported(issues)["enumerator"].nunique()
        n_scored += int((excess(metrics) >= 1).sum())
    assert n_compared == 150
    assert n_reported <= 3, f"{n_reported} of {n_compared} honest enumerators reported"
    assert n_scored <= n_reported


def test_too_regular_enumerator_is_flagged_with_a_plain_explanation():
    df, spec = survey(21)
    df = make_too_regular(df, "E007")
    issues, metrics = regularity(df, config(spec))
    mine = too_regular(issues)[lambda d: d["enumerator"] == "E007"]
    assert set(mine["variable"]) == set(SIGNALS)
    assert (mine["metric"] <= -3.5).all()  # below peers on every signal
    grid = mine.loc[mine["variable"] == "grid_sd", "message"].iloc[0]
    assert grid.startswith("Ratings within this enumerator's interviews vary much less than their peers'")
    assert mine["message"].str.endswith("A lead for a back-check, not proof of fabrication.").all()
    assert excess(metrics)["E007"] >= 1.0


def test_too_varied_enumerator_is_information_and_not_scored():
    df, spec = survey(31)
    rows = df.index[df["enum_id"] == "E004"]
    rng = np.random.default_rng(3)
    for item in RATINGS:
        df.loc[rows, item] = [str(v) for v in rng.choice([1, 5], len(rows))]
    issues, metrics = regularity(df, config(spec))
    mine = reported(issues)[lambda d: d["enumerator"] == "E004"]
    grid = mine[mine["variable"] == "grid_sd"]
    assert len(grid) == 1 and grid["severity"].iloc[0] == "info" and grid["metric"].iloc[0] >= 3.5
    assert "vary much more than their peers'" in grid["message"].iloc[0]
    assert "Often honest" in grid["message"].iloc[0]
    assert (mine["severity"] == "info").all()
    assert excess(metrics)["E004"] < 1.0


def test_too_regular_enumerator_is_flagged_in_the_scorecard_and_others_keep_their_scores():
    df, spec = survey(22, n_enumerators=16, interviews=15)
    df = make_too_regular(df, "E003")
    on = run_checks(df, config(spec))
    off = run_checks(df, config(spec, enabled=False))
    card = on.scorecard.set_index("enumerator")
    assert bool(card.loc["E003", "flagged"]) is True
    assert "Too-regular answers vs peers (40)" in card.loc["E003", "top_reasons"]
    scored = set(too_regular(on.issues[on.issues["check_id"] == "answer_regularity"])["enumerator"])
    others = card.index.difference(sorted(scored))
    before = off.scorecard.set_index("enumerator")["risk_score"]
    assert np.allclose(card.loc[others, "risk_score"], before.loc[others])
    assert on.check_status.set_index("check_id").loc["answer_regularity", "status"] == "ok"


def test_small_n_enumerator_is_not_compared_unless_min_n_is_lowered():
    df, spec = survey(23)
    keep = df.index[df["enum_id"] == "E005"][:6]
    df = df[(df["enum_id"] != "E005") | df.index.isin(keep)].reset_index(drop=True)
    df = make_too_regular(df, "E005")
    issues, metrics = regularity(df, config(spec))
    assert "E005" not in set(issues["enumerator"].dropna())
    assert "E005" not in set(metrics["enumerator"])
    issues, _ = regularity(df, config(spec, min_n=5))
    assert "E005" in set(too_regular(issues)["enumerator"])


def test_zero_mad_skips_the_signal_with_a_note():
    df, spec = survey(24)
    df.loc[:, RATINGS] = "3"  # every interview rates everything 3: no rating spread for any enumerator
    issues, metrics = regularity(df, config(spec))
    assert "grid_sd" not in set(reported(issues)["variable"])
    assert notes(issues).str.contains("grid_sd skipped: at least half of the enumerators have the same value").any()
    assert "regularity_grid_sd_z" not in set(metrics["metric"])
    assert {"regularity_spread_z", "regularity_typicality_z"} <= set(metrics["metric"])
    assert np.isfinite(metrics["value"]).all()


def test_peer_z_uses_median_and_scaled_mad_and_refuses_zero_mad():
    z, med = peer_z(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, np.nan], index=list("abcdef")))
    assert med == 3.0
    assert z["e"] == pytest.approx(97 / 1.4826)
    assert z["c"] == 0.0 and "f" not in z.index
    assert peer_z(pd.Series([1.0, 1.0, 1.0, 2.0, 5.0])) is None
    assert peer_z(pd.Series([np.nan, np.nan])) is None


# ----------------------------------------------------------------------------- configuration
def test_parameters_validate_and_override():
    cfg = config_from_dict(
        {"checks": {"answer_regularity": {"threshold": 4.0, "min_n": 10, "min_enumerators": 8, "signals": ["grid_sd"]}}}
    )
    p = cfg.params_for("answer_regularity")
    assert (p.threshold, p.min_n, p.min_enumerators, p.signals) == (4.0, 10, 8, ["grid_sd"])
    d = config_from_dict({}).params_for("answer_regularity")
    assert (d.threshold, d.min_n, d.min_enumerators, d.signals) == (3.5, 8, 15, list(SIGNALS))
    for bad in ({"signals": ["round_share"]}, {"signals": []}, {"threshold": 0}, {"min_n": 1}, {"direction": "low"}):
        with pytest.raises(ConfigError):
            config_from_dict({"checks": {"answer_regularity": bad}})


def test_overrides_change_what_is_flagged():
    df, spec = survey(21)
    df = make_too_regular(df, "E007")
    issues, metrics = regularity(df, config(spec, threshold=1000.0))
    assert reported(issues).empty
    assert (excess(metrics) < 1).all()
    issues, _ = regularity(df, config(spec, signals=["typicality"]))
    assert set(reported(issues)["variable"]) == {"typicality"}
    issues, metrics = regularity(df, config(spec, min_enumerators=31))
    assert reported(issues).empty and metrics.empty
    assert notes(issues).str.startswith("Peer comparison skipped: 30 enumerator(s)").any()


def test_generic_default_profile_leaves_out_skip_dependent_questions():
    df, spec = survey(25, n_enumerators=12, interviews=10)
    raw = synthetic_config(spec=spec)
    del raw["checks"]["answer_regularity"]
    ctx = build_context(df, config_from_dict(raw), run_id="test")
    assert _skip_dependent(ctx) == {"liv/business_income", "land/land_size"}


# ----------------------------------------------------------------------------- fidelity to the study
def reference_z(df: pd.DataFrame) -> pd.DataFrame:
    """Independent transcription of the study's signals (experiments/ai_fabrication/signatures.py, commit b0ea6d1)."""
    profile = HOUSEHOLD + RATINGS
    logged = ["liv/monthly_income", "exp/food_exp", "exp/nonfood_exp", "water/minutes_to_water"]

    def num(cols: list[str]) -> pd.DataFrame:
        out = df[cols].apply(pd.to_numeric, errors="coerce")
        return out.mask(out.isin((-99.0, -98.0)))

    grid, prof = num(RATINGS), num(profile)
    for c in logged:
        prof[c] = np.log1p(prof[c].clip(lower=0))
    med = prof.median()
    z = ((prof - med) / ((prof - med).abs().median() * 1.4826).replace(0, np.nan)).clip(-6, 6)
    rows = {}
    for e, idx in df.groupby("enum_id").groups.items():
        zi = z.loc[idx]
        zf = zi.fillna(0).to_numpy()
        d = np.sqrt(((zf[:, None, :] - zf[None, :, :]) ** 2).mean(axis=2))
        np.fill_diagonal(d, np.inf)
        rows[e] = {
            "grid_sd": grid.loc[idx].std(axis=1, ddof=0).mean(),
            "spread": float(np.median(d.min(axis=1))),
            "typicality": float(np.sqrt((zi**2).mean(axis=1)).mean()),
        }
    sig = pd.DataFrame(rows).T
    return (sig - sig.median()) / ((sig - sig.median()).abs().median() * 1.4826)


def test_matches_the_studys_signal_definitions():
    df, spec = survey(26, contamination=0.2)
    _, metrics = regularity(df, config(spec))
    mine = metrics[metrics["metric"].str.endswith("_z")].pivot(index="enumerator", columns="metric", values="value")
    ref = reference_z(df)
    for signal in SIGNALS:
        diff = (mine[f"regularity_{signal}_z"] - ref[signal]).abs()
        assert diff.max() < 1e-9, signal


# ----------------------------------------------------------------------------- edge cases
def test_empty_data_and_missing_enumerator_role_are_quiet():
    df, spec = survey(27, n_enumerators=16, interviews=10)
    issues, _ = regularity(df.iloc[0:0], config(spec))
    assert issues.empty
    raw = synthetic_config(spec=spec)
    del raw["roles"]["enumerator_id"]
    issues, _ = regularity(df, config_from_dict(raw))
    assert issues.empty


def test_too_few_enumerators_skips_the_comparison_with_a_note():
    df, spec = survey(28, n_enumerators=6, interviews=12)
    issues, metrics = regularity(df, config(spec))
    assert reported(issues).empty and metrics.empty
    assert notes(issues).str.startswith("Peer comparison skipped").any()


def test_all_nan_answers_and_missing_profile_columns_do_not_fail():
    df, spec = survey(29, n_enumerators=16, interviews=10)
    df.loc[:, RATINGS + HOUSEHOLD] = None
    issues, metrics = regularity(df, config(spec, variables=["age", "no_such_column"]))
    assert reported(issues).empty and metrics.empty
    warn = issues[(issues["level"] == "dataset") & (issues["severity"] == "warning")]
    assert warn["message"].str.contains("no_such_column").any()
    assert notes(issues).str.contains("skipped").any()


def test_text_rating_grid_is_left_out_with_a_note():
    df, spec = survey(30, n_enumerators=16, interviews=10)
    labels = {"1": "never", "2": "rarely", "3": "sometimes", "4": "often", "5": "always"}
    for item in WB_ITEMS:
        df[item] = df[item].map(labels)
    issues, _ = regularity(df, config(spec, variables=["age", "hh_size"]))
    assert notes(issues).str.contains("Grid 'wellbeing' has answers that are not numbers").any()
