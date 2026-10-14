"""Orchestration shared by the CLI and the tests: loading data, back-check runs and the synthetic demo.

Every SYNTHETIC output is prefixed ``synthetic_`` and the demo writes its answer key next to the data so the
evaluation is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fieldguard.backcheck import BackcheckReport, compare_backchecks, select_sample
from fieldguard.benchmark import evaluate_scorecard
from fieldguard.config import FieldGuardConfig, config_from_dict, dump_config
from fieldguard.io.readers import read_from_config, read_table
from fieldguard.outputs import write_csv, write_run_outputs
from fieldguard.pipeline import RunResult, run_checks
from fieldguard.synthetic import SyntheticSpec, generate_backcheck, generate_survey, synthetic_config


def load_survey(cfg: FieldGuardConfig, data_path: Path | None = None) -> pd.DataFrame:
    """Read the survey export: ``data_path`` if given, else ``data.path`` from the config."""
    if data_path is not None:
        return read_table(data_path, sheet=cfg.data.sheet, delimiter=cfg.data.delimiter, encoding=cfg.data.encoding)
    if not cfg.data.path:
        raise ValueError("data.path is not set: pass --data, or download with `fieldguard fetch` first")
    return read_from_config(cfg.data, cfg.base_dir)


def write_backcheck_outputs(
    report: BackcheckReport, sample: pd.DataFrame | None, out_dir: Path, *, synthetic: bool
) -> dict[str, Path]:
    """Write back-check artefacts. The sample is for the field team only and is not a shared report."""
    pre = "synthetic_" if synthetic else ""
    files: dict[str, Path] = {}
    if sample is not None:
        files["backcheck_sample"] = write_csv(sample, out_dir / f"{pre}backcheck_sample_FIELD_ONLY.csv")
    files["backcheck_comparisons"] = write_csv(report.comparisons, out_dir / f"{pre}backcheck_comparisons.csv")
    files["backcheck_variables"] = write_csv(report.variable_summary, out_dir / f"{pre}backcheck_variables.csv")
    files["backcheck_enumerators"] = write_csv(report.enumerator_summary, out_dir / f"{pre}backcheck_enumerators.csv")
    files["backcheck_issues"] = write_csv(report.issues, out_dir / f"{pre}backcheck_issues.csv")
    coverage = out_dir / f"{pre}backcheck_coverage.json"
    coverage.parent.mkdir(parents=True, exist_ok=True)
    coverage.write_text(json.dumps(report.coverage, indent=2, default=float), encoding="utf-8")
    files["backcheck_coverage"] = coverage
    return files


def run_backcheck(
    result: RunResult,
    cfg: FieldGuardConfig,
    out_dir: Path,
    *,
    compare: pd.DataFrame | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Select a back-check sample (no ``compare``) or compare returned re-interviews with the survey."""
    if result.context is None:
        raise ValueError("run result has no check context")
    out_dir.mkdir(parents=True, exist_ok=True)
    if compare is None:
        sample = select_sample(result.context, result.issues, result.scorecard, cfg, seed=seed)
        pre = "synthetic_" if result.synthetic else ""
        path = write_csv(sample, out_dir / f"{pre}backcheck_sample_FIELD_ONLY.csv")
        return {"mode": "select", "n_sample": int(len(sample)), "files": {"backcheck_sample": path}}
    if not cfg.roles.respondent_id:
        raise ValueError("roles.respondent_id must be set to match re-interviews to the survey")
    report = compare_backchecks(
        result.context.data,
        result.context.enumerator,
        compare,
        cfg,
        id_column=cfg.roles.respondent_id,
        run_id=result.run_id,
    )
    files = write_backcheck_outputs(report, None, out_dir, synthetic=result.synthetic)
    return {
        "mode": "compare",
        "n_compared_records": report.coverage.get("n_backchecked_records", 0),
        "files": files,
        "report": report,
    }


def run_demo(
    out_dir: Path,
    *,
    n_enumerators: int = 30,
    interviews: int = 25,
    contamination: float = 0.15,
    seed: int = 42,
    formats: list[str] | None = None,
    backcheck: bool = True,
) -> dict[str, Any]:
    """End-to-end run on generated SYNTHETIC data, including a back-check round trip and a check against the key."""
    spec = SyntheticSpec(
        n_enumerators=n_enumerators, interviews_per_enumerator=interviews, contamination=contamination, seed=seed
    )
    survey = generate_survey(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(survey.data, out_dir / "synthetic_survey.csv")
    write_csv(survey.labels, out_dir / "synthetic_answer_key_enumerators.csv")
    write_csv(survey.truth, out_dir / "synthetic_answer_key_records.csv")
    cfg = config_from_dict(synthetic_config(spec=spec, data_path="synthetic_survey.csv"), base_dir=out_dir)
    result = run_checks(survey.data, cfg, data_source="synthetic generator (in memory)")
    files = write_run_outputs(result, cfg, out_dir, formats=formats)
    config_path = out_dir / "synthetic_config.yaml"
    config_path.write_text(dump_config(cfg), encoding="utf-8")
    files["config"] = config_path
    evaluation = evaluate_scorecard(result.scorecard, survey.labels)
    info: dict[str, Any] = {"run_id": result.run_id, "out_dir": out_dir, "evaluation": evaluation, "files": files}
    if backcheck:
        sample = select_sample(result.context, result.issues, result.scorecard, cfg)
        hhids = sample["respondent_id"].astype(str).tolist() if not sample.empty else []
        reinterviews = generate_backcheck(survey, hhids, seed=seed + 1)
        write_csv(reinterviews, out_dir / "synthetic_backcheck_reinterviews.csv")
        report = compare_backchecks(
            result.context.data,
            result.context.enumerator,
            reinterviews,
            cfg,
            id_column=cfg.roles.respondent_id or "hhid",
            run_id=result.run_id,
        )
        info["files"].update(write_backcheck_outputs(report, sample, out_dir, synthetic=True))
        info["backcheck"] = {
            "n_sample": int(len(sample)),
            "n_reinterviews": int(len(reinterviews)),
            "coverage": report.coverage,
        }
    return info


__all__ = ["load_survey", "run_backcheck", "run_demo", "write_backcheck_outputs"]
