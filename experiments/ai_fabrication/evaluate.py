"""Stage C: inject the model-fabricated interviews, run FieldGuard, and score it against the answer key.

    uv run python experiments/ai_fabrication/evaluate.py --split dev
    uv run python experiments/ai_fabrication/evaluate.py --split test

Two paradata conditions are run for every kind of fabricator:

* ``content_only``: the fabricated interviews keep the honest interview's timestamps and GPS. This isolates what
  the answers alone give away (a fabricator who also fakes paradata well, or one who visits but makes up answers).
* ``with_paradata``: the fabricated interviews also carry the usual traces of curbstoning, as FieldGuard's
  generator models them: interviews a quarter to half as long, entered in evening batches, GPS points at home.

The comparison fabricator is FieldGuard's own rule-based generator restricted to the matching patterns
(content patterns only, or all ten). Every survey has 30 enumerators, 3 of them fabricating.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEV_SEEDS,
    FIELDS,
    GENERATED,
    INTERVIEWS,
    N_ENUMERATORS,
    RESULTS,
    STRATEGIES,
    TEST_SEEDS,
    load_json,
    payload_path,
)
from make_prompts import base_survey  # noqa: E402
from signatures import enumerator_signals, robust_z  # noqa: E402

from fieldguard.benchmark import evaluate_scorecard  # noqa: E402
from fieldguard.config import config_from_dict  # noqa: E402
from fieldguard.pipeline import run_checks  # noqa: E402
from fieldguard.synthetic import ALL_PATTERNS, SyntheticSpec, generate_survey, synthetic_config  # noqa: E402

CONTENT_PATTERNS = tuple(p for p in ALL_PATTERNS if p not in ("short_duration", "timing", "gps_clustering"))
PARADATA = ("content_only", "with_paradata")
COL = {k: c for k, c, _kind, _q in FIELDS}
KIND = {k: kind for k, _c, kind, _q in FIELDS}


def _cell(v) -> object:
    """A model's value as an export cell: integers without '.0', missing as NaN (like fieldguard.synthetic)."""
    if v is None:
        return np.nan
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def inject(df: pd.DataFrame, payload: dict, records: list[dict]) -> pd.DataFrame:
    """Replace the content of the payload's interviews with the model's records, in order."""
    out = df.copy()
    idx = out.index[out["hhid"].isin(payload["hhids"])]
    order = {h: i for i, h in enumerate(payload["hhids"])}
    idx = sorted(idx, key=lambda i: order[out.at[i, "hhid"]])
    for i, rec in zip(idx, records, strict=False):
        for key, col in COL.items():
            out.at[i, col] = _cell(rec.get(key))
    return out


def fake_paradata(df: pd.DataFrame, payload: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Short interviews in evening batches with GPS points at home, as in fieldguard.synthetic's patterns."""
    out = df.copy()
    rows = out.index[out["hhid"].isin(payload["hhids"])]
    rows = sorted(rows, key=lambda i: out.at[i, "start"])
    first = rows[0]
    lat0, lon0 = float(out.at[first, "_gps_latitude"]), float(out.at[first, "_gps_longitude"])
    home = (lat0 + rng.normal(0, 300) / 111_320.0, lon0 + rng.normal(0, 300) / (111_320.0 * np.cos(np.radians(lat0))))
    evening: datetime | None = None
    prev: tuple[float, float] | None = None
    for i in rows:
        start = datetime.fromisoformat(out.at[i, "start"])
        end = datetime.fromisoformat(out.at[i, "end"])
        minutes = (end - start).total_seconds() / 60 * rng.uniform(0.25, 0.5)
        if evening is None or evening.date() != start.date() or rng.random() < 0.25:
            evening = datetime.combine(start.date(), datetime.min.time(), start.tzinfo) + timedelta(
                hours=float(rng.uniform(19.5, 21.5))
            )
        s, e = evening, evening + timedelta(minutes=minutes)
        evening = e + timedelta(minutes=float(rng.uniform(-12, 2)))
        if prev is not None and rng.random() < 0.3:
            lat, lon = prev
        else:
            lat = home[0] + rng.normal(0, 4) / 111_320.0
            lon = home[1] + rng.normal(0, 4) / (111_320.0 * np.cos(np.radians(home[0])))
        prev = (lat, lon)
        acc = float(rng.uniform(3, 8))
        out.at[i, "start"] = s.isoformat(timespec="milliseconds")
        out.at[i, "end"] = e.isoformat(timespec="milliseconds")
        out.at[i, "today"] = s.date().isoformat()
        sub = e + timedelta(hours=float(rng.uniform(0.2, 8)))
        out.at[i, "_submission_time"] = sub.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
        alt = str(out.at[i, "gps"]).split()[2] if len(str(out.at[i, "gps"]).split()) > 2 else "1600.0"
        out.at[i, "gps"] = f"{lat:.6f} {lon:.6f} {alt} {acc:.1f}"
        out.at[i, "_gps_latitude"] = str(round(lat, 6))
        out.at[i, "_gps_longitude"] = str(round(lon, 6))
        out.at[i, "_gps_precision"] = str(round(acc, 1))
    return out


def run(df: pd.DataFrame, fabricators: list[str], seed: int, run_id: str) -> tuple[dict, pd.DataFrame]:
    spec = SyntheticSpec(n_enumerators=N_ENUMERATORS, interviews_per_enumerator=INTERVIEWS, seed=seed)
    cfg = config_from_dict(synthetic_config(spec=spec))
    result = run_checks(df, cfg, run_id=run_id, data_source="synthetic generator + model fabrication")
    enumerators = sorted(df["enum_id"].unique())
    labels = pd.DataFrame({"enumerator": enumerators, "is_fabricator": [e in fabricators for e in enumerators]})
    metrics = evaluate_scorecard(result.scorecard, labels)
    issues = result.issues
    fired = (
        issues[issues["severity"].isin(["warning", "critical"]) & issues["enumerator"].notna()]
        .groupby(["check_id", "enumerator"])
        .size()
        .rename("n")
        .reset_index()
    )
    fired["is_fabricator"] = fired["enumerator"].isin(fabricators)
    card = result.scorecard.assign(is_fabricator=lambda c: c["enumerator"].astype(str).isin(fabricators))
    z = robust_z(enumerator_signals(df)).rename(columns=lambda c: c if c in ("enumerator", "n") else f"z_{c}")
    card = card.merge(z, on="enumerator", how="left")
    return metrics, fired.assign(run_id=run_id), card.assign(run_id=run_id)


def ai_conditions() -> list[tuple[str, str]]:
    out = []
    if GENERATED.exists():
        for model_dir in sorted(p for p in GENERATED.iterdir() if p.is_dir()):
            for strategy in STRATEGIES:
                if (model_dir / strategy).exists():
                    out.append((model_dir.name, strategy))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    seeds = DEV_SEEDS if args.split == "dev" else TEST_SEEDS
    rows, fired_all, cards = [], [], []
    for seed in seeds:
        base = base_survey(seed)
        payloads = [load_json(p) for p in sorted(payload_path(seed, "x").parent.glob(f"seed{seed}_*.json"))]
        fabs = [p["enumerator"] for p in payloads]
        # comparison: FieldGuard's rule-based fabricators (own generator draw, 3 of 30 enumerators)
        for paradata in PARADATA:
            patterns = CONTENT_PATTERNS if paradata == "content_only" else ALL_PATTERNS
            spec = SyntheticSpec(
                n_enumerators=N_ENUMERATORS,
                interviews_per_enumerator=INTERVIEWS,
                contamination=0.10,
                patterns=patterns,
                seed=seed,
            )
            sv = generate_survey(spec)
            classic_fabs = sv.labels.loc[sv.labels["is_fabricator"], "enumerator"].tolist()
            m, f, c = run(sv.data, classic_fabs, seed, f"{args.split}-s{seed}-classic-{paradata}")
            rows.append({"seed": seed, "fabricator": "rule-based (FieldGuard generator)", "model": "-",
                         "strategy": "-", "paradata": paradata, **m})
            fired_all.append(f.assign(fabricator="rule-based", model="-", strategy="-", paradata=paradata, seed=seed))
            cards.append(c.assign(fabricator="rule-based", model="-", strategy="-", paradata=paradata, seed=seed))
        for model, strategy in ai_conditions():
            gens = {}
            for p in payloads:
                g = GENERATED / model / strategy / f"seed{seed}_{p['enumerator']}.json"
                if g.exists():
                    gens[p["enumerator"]] = load_json(g)
            if len(gens) < len(payloads):
                continue
            # A refused request leaves that enumerator's interviews genuine: in this run they did not fabricate.
            done = [p for p in payloads if gens[p["enumerator"]]["records"]]
            fabs_run = [p["enumerator"] for p in done]
            for paradata in PARADATA:
                rng = np.random.default_rng(seed * 7 + len(model) + len(strategy))
                df = base
                for p in done:
                    df = inject(df, p, gens[p["enumerator"]]["records"])
                    if paradata == "with_paradata":
                        df = fake_paradata(df, p, rng)
                m, f, c = run(df, fabs_run, seed, f"{args.split}-s{seed}-{model}-{strategy}-{paradata}")
                n_ret = sum(len(gens[e]["records"]) for e in fabs)
                n_req = sum(p["n_records"] for p in payloads)
                m["refused"] = sum(bool(gens[e].get("refused")) for e in fabs)
                rows.append({"seed": seed, "fabricator": "language model", "model": model, "strategy": strategy,
                             "paradata": paradata, "records_returned": n_ret, "records_requested": n_req, **m})
                fired_all.append(f.assign(fabricator="language model", model=model, strategy=strategy,
                                          paradata=paradata, seed=seed))
                cards.append(c.assign(fabricator="language model", model=model, strategy=strategy,
                                      paradata=paradata, seed=seed))
        print(f"seed {seed} done", flush=True)
    out = RESULTS / args.split
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "runs.csv", index=False)
    pd.concat(fired_all).to_csv(out / "checks_fired.csv", index=False)
    pd.concat(cards).to_csv(out / "scorecards.csv", index=False)
    keys = ["fabricator", "model", "strategy", "paradata"]
    summary = runs.groupby(keys)[["recall", "precision", "roc_auc_risk_score", "roc_auc_rule_score", "fp"]].mean()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
