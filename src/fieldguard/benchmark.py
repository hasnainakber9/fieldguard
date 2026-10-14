"""Benchmark and evaluation on SYNTHETIC data only.

``run_benchmark`` generates surveys with fabricating enumerators at several contamination rates (share of
enumerators who fabricate), runs the full pipeline and reports precision, recall and ROC-AUC of the flags against
the known answer key. The results describe FieldGuard on the generator's assumptions. They are not evidence of
performance on real surveys, where fabrication patterns differ and labels are uncertain.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fieldguard.config import config_from_dict
from fieldguard.outputs import write_csv
from fieldguard.pipeline import run_checks
from fieldguard.synthetic import SYNTHETIC_LABEL, SyntheticSpec, generate_survey, synthetic_config

DEFAULT_RATES: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3)
SCORES: tuple[str, ...] = ("risk_score", "rule_score", "anomaly_score")


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    if y.size < 2 or len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def evaluate_scorecard(card: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    """Precision, recall, F1 of the ``flagged`` enumerators and ROC-AUC of each score, against the answer key."""
    if card.empty or labels.empty:
        return {
            "n_enumerators": 0,
            "n_fabricators": 0,
            "n_flagged": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
    truth = (
        labels.assign(enumerator=labels["enumerator"].astype(str)).set_index("enumerator")["is_fabricator"].astype(bool)
    )
    c = card.assign(enumerator=card["enumerator"].astype(str)).set_index("enumerator")
    common = c.index.intersection(truth.index)
    y = truth.loc[common].to_numpy(dtype=bool)
    flagged = c.loc[common, "flagged"].astype(bool).to_numpy()
    tp = int((y & flagged).sum())
    fp = int((~y & flagged).sum())
    fn = int((y & ~flagged).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    denom = 2 * tp + fp + fn
    f1 = 2 * tp / denom if denom else float("nan")
    out: dict[str, Any] = {
        "n_enumerators": int(len(common)),
        "n_fabricators": int(y.sum()),
        "n_flagged": int(flagged.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    for name in SCORES:
        if name not in c.columns:
            out[f"roc_auc_{name}"] = float("nan")
            continue
        s = pd.to_numeric(c.loc[common, name], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(s)
        out[f"roc_auc_{name}"] = _auc(y[ok], s[ok]) if ok.any() else float("nan")
    return out


def run_benchmark(
    out_dir: Path,
    *,
    rates: Sequence[float] = DEFAULT_RATES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    n_enumerators: int = 30,
    interviews: int = 25,
) -> pd.DataFrame:
    """Run the pipeline on synthetic surveys over ``rates`` x ``seeds`` and write CSV and JSON summaries."""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rate in rates:
        for seed in seeds:
            spec = SyntheticSpec(
                n_enumerators=n_enumerators,
                interviews_per_enumerator=interviews,
                contamination=float(rate),
                seed=int(seed),
            )
            survey = generate_survey(spec)
            cfg = config_from_dict(synthetic_config(spec=spec))
            result = run_checks(
                survey.data,
                cfg,
                run_id=f"bench-c{int(round(rate * 100)):02d}-s{seed}",
                data_source="synthetic generator",
            )
            metrics = evaluate_scorecard(result.scorecard, survey.labels)
            rows.append(
                {
                    "contamination": float(rate),
                    "seed": int(seed),
                    "n_records": result.n_records,
                    "n_issues": len(result.issues),
                    **metrics,
                }
            )
    runs = pd.DataFrame(rows)
    numeric = [c for c in runs.columns if c not in ("contamination", "seed")]
    summary = runs.groupby("contamination")[numeric].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    write_csv(runs, out_dir / "synthetic_benchmark_runs.csv")
    write_csv(summary, out_dir / "synthetic_benchmark_summary.csv")
    payload = {
        "label": SYNTHETIC_LABEL,
        "rates": [float(r) for r in rates],
        "seeds": [int(s) for s in seeds],
        "n_enumerators": n_enumerators,
        "interviews_per_enumerator": interviews,
        "summary": json.loads(summary.to_json(orient="records")),
        "note": "Synthetic data from fieldguard.synthetic; describes the generator's assumptions, not real surveys.",
    }
    path = out_dir / "synthetic_benchmark.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    tmp.replace(path)
    return runs


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return str(value)


__all__ = ["DEFAULT_RATES", "DEFAULT_SEEDS", "evaluate_scorecard", "run_benchmark"]
