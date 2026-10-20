"""Command-line interface: ``fieldguard run | demo | backcheck | benchmark | fetch | version``.

Flags are leads for supervisory review, not proof of misconduct. Credentials are read from environment variables
only and are never printed. Database URLs are shown without passwords.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from fieldguard import __version__

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,  # locals can hold the database URL with its password
    no_args_is_help=True,
    help="FieldGuard: survey quality checks and fabrication detection for KoboToolbox, ODK and SurveyCTO exports. "
    "Flags are leads for supervisory review, not proof of misconduct.",
)

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Project YAML configuration.", exists=True, dir_okay=False)
]
OutOpt = Annotated[
    Path | None, typer.Option("--out", "-o", help="Output folder (overrides FIELDGUARD_REPORTS_DIR and outputs.dir).")
]
FormatsOpt = Annotated[
    str | None, typer.Option("--formats", help="Comma-separated: csv,parquet,excel,html (default: config).")
]


def _setup() -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


@app.callback()
def _root() -> None:
    _setup()


def _formats(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [f.strip() for f in value.split(",") if f.strip()]


def _fail(message: str) -> None:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


@app.command()
def run(
    config: ConfigOpt,
    out: OutOpt = None,
    formats: FormatsOpt = None,
    data: Annotated[
        Path | None, typer.Option("--data", help="Survey file (CSV, TSV, XLSX or Parquet) overriding data.path.")
    ] = None,
    incremental: Annotated[
        bool, typer.Option("--incremental", help="Report record-level issues only for submissions not seen before.")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Stop at the first failing check instead of recording it.")
    ] = False,
    no_sink: Annotated[bool, typer.Option("--no-sink", help="Do not write to PostgreSQL even if configured.")] = False,
) -> None:
    """Run every enabled check on a survey export and write the reports."""
    from fieldguard.config import load_config
    from fieldguard.outputs import resolve_output_dir, write_run_outputs
    from fieldguard.pipeline import run_checks
    from fieldguard.privacy import resolve_salt
    from fieldguard.sink import (
        check_reachable,
        database_url,
        load_seen_file,
        load_seen_postgres,
        redact_url,
        save_seen_file,
        save_seen_postgres,
        write_run,
    )
    from fieldguard.workflows import load_survey

    cfg = load_config(config)
    try:
        frame = load_survey(cfg, data)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        _fail(str(exc))
    url = database_url(cfg.sink.database_url_env)
    sink_wanted = (cfg.sink.postgres or url is not None) and not no_sink
    sink_ok = False
    sink_msg = "PostgreSQL sink not configured"
    if sink_wanted and url:
        status = check_reachable(url)
        sink_ok = status.available
        sink_msg = status.message if not sink_ok else f"PostgreSQL sink ready at {redact_url(url)}"
    elif sink_wanted:
        sink_msg = "PostgreSQL sink skipped: FIELDGUARD_DATABASE_URL is not set"
    seen: set[str] | None = None
    state_backend = "none"
    if incremental:
        use_pg = sink_ok and cfg.incremental.state in ("auto", "postgres")
        if cfg.incremental.state == "postgres" and not sink_ok:
            typer.secho(
                "warning: incremental.state is postgres but the database is unavailable; using the file state",
                fg=typer.colors.YELLOW,
                err=True,
            )
        if use_pg and url:
            seen = load_seen_postgres(url, cfg)
            state_backend = "postgres"
        else:
            seen = load_seen_file(cfg)
            state_backend = "file"
    result = run_checks(frame, cfg, seen_ids=seen, strict=strict, data_source=str(data or cfg.data.path or "config"))
    out_dir = resolve_output_dir(cfg, result.run_id, out)
    written = write_run_outputs(result, cfg, out_dir, formats=_formats(formats))
    if sink_ok and url:
        counts = write_run(
            result,
            cfg,
            url,
            salt=resolve_salt(cfg) if cfg.privacy.hash_respondent_ids or cfg.privacy.pseudonymize_enumerators else None,
        )
        sink_msg = f"PostgreSQL sink: wrote {counts['issues']} issues and {counts['scorecards']} scorecard rows"
    if incremental:
        all_ids = [str(x) for x in result.context.record_id] if result.context is not None else []
        if state_backend == "postgres" and url:
            save_seen_postgres(url, cfg, all_ids, result.run_id)
        else:
            save_seen_file(cfg, all_ids, result.run_id)
    typer.echo(f"run {result.run_id} ({result.mode}) for project {result.project}")
    if result.synthetic:
        typer.secho(
            "SYNTHETIC DATA: outputs are labelled synthetic_ and do not describe a real survey.", fg=typer.colors.YELLOW
        )
    typer.echo(
        f"records {result.n_records} (new {result.n_new_records}), enumerators {result.n_enumerators}, issues {len(result.issues)}"
    )
    flagged = int(result.scorecard["flagged"].sum()) if not result.scorecard.empty else 0
    typer.echo(f"flagged enumerators: {flagged} (leads for supervisory review, not proof of misconduct)")
    if incremental:
        typer.echo(f"incremental state: {state_backend}")
    typer.echo(sink_msg)
    typer.echo(f"outputs in {out_dir}")
    for name, path in written.items():
        typer.echo(f"  {name}: {path.name}")


@app.command()
def demo(
    out: OutOpt = None,
    n_enumerators: Annotated[int, typer.Option("--enumerators", min=2)] = 30,
    interviews: Annotated[int, typer.Option("--interviews", min=5)] = 25,
    contamination: Annotated[float, typer.Option("--contamination", min=0.0, max=0.99)] = 0.15,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    formats: FormatsOpt = None,
    backcheck: Annotated[bool, typer.Option("--backcheck/--no-backcheck")] = True,
) -> None:
    """Generate SYNTHETIC data with injected fabrication, run the checks and score them against the answer key."""
    from fieldguard.workflows import run_demo

    folder = out or (
        Path(os.environ["FIELDGUARD_REPORTS_DIR"]) / "demo"
        if os.environ.get("FIELDGUARD_REPORTS_DIR")
        else Path("fieldguard_output") / "demo"
    )
    info = run_demo(
        folder,
        n_enumerators=n_enumerators,
        interviews=interviews,
        contamination=contamination,
        seed=seed,
        formats=_formats(formats),
        backcheck=backcheck,
    )
    ev = info["evaluation"]
    typer.secho(
        "SYNTHETIC DEMO: generated data with injected fabrication (not real survey data).", fg=typer.colors.YELLOW
    )
    typer.echo(f"run {info['run_id']} written to {info['out_dir']}")
    typer.echo(
        f"against the synthetic answer key: flagged {ev['n_flagged']} of {ev['n_enumerators']} enumerators; "
        f"{ev['tp']} true positives, {ev['fp']} false positives, {ev['fn']} missed; "
        f"precision {ev['precision']:.2f}, recall {ev['recall']:.2f}, ROC-AUC (risk) {ev['roc_auc_risk_score']:.2f}"
    )
    if "backcheck" in info:
        bc = info["backcheck"]
        typer.echo(f"back-check round trip: sample {bc['n_sample']}, re-interviews {bc['n_reinterviews']} (synthetic)")


@app.command()
def backcheck(
    config: ConfigOpt,
    out: OutOpt = None,
    compare: Annotated[
        Path | None,
        typer.Option("--compare", help="Back-check file to compare with the survey. Omit to draw a sample."),
    ] = None,
    data: Annotated[Path | None, typer.Option("--data", help="Survey file overriding data.path.")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Draw a stratified back-check sample (default) or compare re-interviews with the survey (--compare)."""
    from fieldguard.config import load_config
    from fieldguard.io.readers import read_table
    from fieldguard.outputs import resolve_output_dir
    from fieldguard.pipeline import run_checks
    from fieldguard.workflows import load_survey, run_backcheck

    cfg = load_config(config)
    try:
        frame = load_survey(cfg, data)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        _fail(str(exc))
    result = run_checks(frame, cfg, data_source=str(data or cfg.data.path or "config"))
    bc_frame = None
    if compare is not None:
        try:
            bc_frame = read_table(compare)
        except (ValueError, FileNotFoundError) as exc:
            _fail(str(exc))
    out_dir = resolve_output_dir(cfg, result.run_id, out)
    info = run_backcheck(result, cfg, out_dir, compare=bc_frame, seed=seed)
    if info["mode"] == "select":
        typer.echo(f"back-check sample: {info['n_sample']} interviews (FIELD ONLY, not for sharing with enumerators)")
    else:
        report = info["report"]
        typer.echo(f"back-check comparison: {info['n_compared_records']} re-interviews matched")
        typer.echo(
            f"enumerator-level leads: {int((report.enumerator_summary['flagged'].astype(bool)).sum()) if not report.enumerator_summary.empty else 0}"
        )
    for name, path in info["files"].items():
        typer.echo(f"  {name}: {path.name}")


@app.command()
def benchmark(
    out: OutOpt = None,
    quick: Annotated[bool, typer.Option("--quick", help="One seed and two rates (for a fast check).")] = False,
    enumerators: Annotated[int, typer.Option("--enumerators", min=4)] = 30,
) -> None:
    """Precision, recall and ROC-AUC at several contamination rates on SYNTHETIC data; saved to the reports folder."""
    from fieldguard.benchmark import DEFAULT_RATES, DEFAULT_SEEDS, run_benchmark

    folder = out or (
        Path(os.environ["FIELDGUARD_REPORTS_DIR"]) / "benchmark"
        if os.environ.get("FIELDGUARD_REPORTS_DIR")
        else Path("fieldguard_output") / "benchmark"
    )
    rates = (0.10, 0.30) if quick else DEFAULT_RATES
    seeds = (1,) if quick else DEFAULT_SEEDS
    runs = run_benchmark(folder, rates=rates, seeds=seeds, n_enumerators=enumerators)
    typer.secho("SYNTHETIC BENCHMARK: generator assumptions, not evidence about real surveys.", fg=typer.colors.YELLOW)
    for rate, sub in runs.groupby("contamination"):
        typer.echo(
            f"contamination {rate:.2f}: precision {sub['precision'].mean():.2f}, recall {sub['recall'].mean():.2f}, "
            f"ROC-AUC {sub['roc_auc_risk_score'].mean():.2f} (mean over {len(sub)} seed(s))"
        )
    typer.echo(f"written to {folder}")


@app.command()
def fetch(
    config: ConfigOpt,
    out: Annotated[Path, typer.Option("--out", "-o", help="Raw CSV to write (D: raw folder).")],
    since: Annotated[
        str | None, typer.Option("--since", help="SurveyCTO only: pull submissions after this date value.")
    ] = None,
    page_size: Annotated[int, typer.Option("--page-size", min=1, max=1000)] = 1000,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", help="Page cache for resumable downloads.")] = None,
) -> None:
    """Download submissions from KoboToolbox, ODK Central or SurveyCTO into a CSV (credentials from environment)."""
    from fieldguard.config import load_config
    from fieldguard.connectors import (
        ConnectorError,
        MissingCredentialError,
        PageCache,
        fetch_kobo,
        fetch_odk,
        fetch_surveycto,
        kobo_settings_from_env,
        odk_settings_from_env,
        surveycto_settings_from_env,
        write_raw_csv,
    )

    cfg = load_config(config)
    if cfg.data.source is None:
        _fail("data.source is not set in the config (type: kobo, odk or surveycto)")
    src = cfg.data.source
    cache = PageCache(cache_dir or out.with_name(out.stem + "_pages"))
    try:
        if src.type == "kobo":
            records = fetch_kobo(kobo_settings_from_env(src), page_size=page_size, cache=cache)
        elif src.type == "odk":
            records = fetch_odk(odk_settings_from_env(src), page_size=page_size, cache=cache)
        else:
            records = fetch_surveycto(surveycto_settings_from_env(src), since=since, cache=cache)
    except MissingCredentialError as exc:
        _fail(str(exc))
    except ConnectorError as exc:
        _fail(str(exc))
    write_raw_csv(records, out)
    typer.echo(f"downloaded {len(records)} submission(s) from {src.type}; wrote {out}")
    typer.echo("credentials were read from environment variables and are not shown")


@app.command()
def version() -> None:
    """Print the FieldGuard version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
