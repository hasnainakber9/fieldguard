"""Stage A: honest base surveys, the AI-fabricating enumerators, and one prompt payload per fabricator.

    uv run python experiments/ai_fabrication/make_prompts.py

For each seed, ``fieldguard.synthetic`` generates a survey with no fabrication. Three enumerators are then chosen
to fabricate a share of their interviews (60% to 100%, as in FieldGuard's benchmark) with a language model. The
payload holds which interviews they replace and, for the informed prompts, summary statistics of the *other*
enumerators' records in the same two villages: what a fabricator could learn from colleagues or a past round.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEV_SEEDS,
    FIELDS,
    INTERVIEWS,
    N_AI_FABRICATORS,
    N_ENUMERATORS,
    TEST_SEEDS,
    payload_path,
    write_json,
)

from fieldguard.synthetic import SyntheticSpec, generate_survey  # noqa: E402

DK = (-99.0, -98.0)
STAT_VARS = [k for k, _c, kind, _q in FIELDS if kind in ("int", "num", "num_or_null") and not k.startswith(("wb", "svc"))]


def base_survey(seed: int) -> pd.DataFrame:
    spec = SyntheticSpec(n_enumerators=N_ENUMERATORS, interviews_per_enumerator=INTERVIEWS, contamination=0.0, seed=seed)
    return generate_survey(spec).data


def village_stats(df: pd.DataFrame) -> dict:
    col = {k: c for k, c, _kind, _q in FIELDS}
    num = {k: pd.to_numeric(df[col[k]], errors="coerce") for k in STAT_VARS}
    stats: dict = {"n": int(len(df)), "numeric": {}, "yes_share": {}, "likert_mean": {}}
    dk_n = dk_total = 0
    for k, s in num.items():
        if k in ("business_income", "monthly_income", "livestock_value", "nonfood_exp", "land_size", "minutes_to_water"):
            valid = s.dropna()
            dk_total += len(valid)
            dk_n += int(valid.isin(DK).sum())
        v = s[~s.isin(DK)].dropna()
        if v.empty:
            continue
        stats["numeric"][k] = {
            "median": float(v.median()),
            "q25": float(v.quantile(0.25)),
            "q75": float(v.quantile(0.75)),
            "min": float(v.min()),
            "max": float(v.max()),
        }
    for k in ("has_business", "owns_land"):
        stats["yes_share"][k] = float((df[col[k]] == "yes").mean())
    stats["water"] = df[col["water_source"]].value_counts(normalize=True).round(3).to_dict()
    for k, c, kind, _q in FIELDS:
        if kind == "likert":
            stats["likert_mean"][k] = float(pd.to_numeric(df[c], errors="coerce").mean())
    stats["dk_share"] = dk_n / dk_total if dk_total else 0.0
    return stats


def payloads_for(seed: int) -> list[dict]:
    df = base_survey(seed)
    rng = np.random.default_rng(seed + 10_000)
    enumerators = sorted(df["enum_id"].unique())
    fabricators = sorted(rng.choice(enumerators, size=N_AI_FABRICATORS, replace=False).tolist())
    out = []
    for e in fabricators:
        own = df[df["enum_id"] == e]
        share = float(rng.uniform(0.6, 1.0))
        k = max(1, int(round(share * len(own))))
        picked = own.iloc[np.sort(rng.choice(len(own), size=k, replace=False))]
        order = picked["village"].value_counts().index.tolist()  # the busier village first
        picked = pd.concat([picked[picked["village"] == v] for v in order])
        villages = [(v, int((picked["village"] == v).sum())) for v in order]
        peers = df[(df["enum_id"] != e) & (df["village"].isin(order))]
        out.append(
            {
                "seed": seed,
                "split": "dev" if seed in DEV_SEEDS else "test",
                "enumerator": e,
                "n_interviews": int(len(own)),
                "n_records": int(len(picked)),
                "fabricated_share": len(picked) / len(own),
                "villages": villages,
                "hhids": picked["hhid"].tolist(),
                "village_stats": village_stats(peers),
                "label": "SYNTHETIC - fieldguard.synthetic base survey; no real survey data",
            }
        )
    return out


def main() -> None:
    n = 0
    for seed in (*DEV_SEEDS, *TEST_SEEDS):
        for p in payloads_for(seed):
            write_json(payload_path(seed, p["enumerator"]), p)
            n += 1
            print(f"seed {seed} {p['enumerator']}: {p['n_records']} of {p['n_interviews']} interviews, {p['villages']}")
    print(f"{n} payloads written")


if __name__ == "__main__":
    main()
