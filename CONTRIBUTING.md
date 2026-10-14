# Contributing to FieldGuard

Thank you for considering a contribution. FieldGuard supports supervisors and evaluators, so changes should be
careful about accuracy, privacy and the way flags are worded. Please read the ethics rules below before you start.

## Ground rules

1. **Flags are leads.** Wording in code, messages, reports and docs must not imply misconduct. A check reports
   what the data looks like. It does not prove that an interview was fabricated, and it must not trigger any
   automated sanction.
2. **No real survey data in the repository.** Use the synthetic generator (`fieldguard.synthetic`) for tests and
   demos. Synthetic outputs must carry the `SYNTHETIC` label and the `synthetic_` file prefix. Real exports belong
   outside the repository (see `DATA_SOURCES.md`).
3. **No credentials in code, tests or docs.** Read secrets from environment variables. Tests use fake values, and
   no test may contact a real server. Connector tests use mocked HTTP.
4. **Privacy in shared outputs.** Names, phone numbers and precise GPS must stay out of shared files. Any new output
   must go through `fieldguard.privacy` (redaction and rounding) before it is written.
5. **Keep the docs accurate.** If you change a check, a default or an output, update `README.md` (including the
   check catalogue and configuration reference) and `DATA_SOURCES.md` in the same change.

## Setup

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
uv run ruff check --fix .
uv run ruff format .
```

The PostgreSQL tests are marked `postgres` and skip unless `FIELDGUARD_DATABASE_URL` points at a reachable server.
Do not put a real database URL or password in a test file. If you need a live database, set the variable in your
shell only.

Please keep the test suite green and ruff clean before you open a change. Type hints are expected on public
functions, and configuration must be modelled with pydantic (`src/fieldguard/config.py`).

## Project layout

| Path | Role |
|---|---|
| `src/fieldguard/config.py` | Pydantic models for the project file. Unknown keys are rejected. |
| `src/fieldguard/io/` | Readers (CSV, TSV, XLSX, Parquet), platform detection, column resolution, timestamps, API flattening |
| `src/fieldguard/checks/` | Check context (`context.py`), registry (`registry.py`) and the built-in checks in `builtin/` |
| `src/fieldguard/scoring/` | Rule-based score, Isolation Forest / LOF anomaly score, enumerator scorecard |
| `src/fieldguard/backcheck.py` | Back-check sampling and comparison |
| `src/fieldguard/pipeline.py` | Runs the registered checks, scores enumerators, builds the run result |
| `src/fieldguard/outputs.py`, `reports/` | CSV, Parquet, Excel and HTML outputs |
| `src/fieldguard/sink/` | Optional PostgreSQL sink and incremental state |
| `src/fieldguard/connectors/` | KoboToolbox, ODK Central and SurveyCTO API clients (`fetch`) |
| `src/fieldguard/synthetic.py`, `benchmark.py` | Synthetic generator, demo and benchmark |
| `src/fieldguard/cli.py` | Command-line interface (`python -m fieldguard`) |

## Adding a check

A check is a function that receives a `CheckContext` and a parameter model, and returns a tidy issues table.
Built-in checks live in `src/fieldguard/checks/builtin/`. A new check should follow the pattern of an existing one:

```python
from pydantic import Field

from fieldguard.checks.context import CheckContext
from fieldguard.checks.registry import CheckParams, register_check
from fieldguard.models import Severity, empty_issues


class ExampleParams(CheckParams):
    threshold: float = Field(3.0, gt=0, description="Flag values above this")
    severity: Severity = Severity.WARNING


@register_check(
    "example_check",
    title="Example check",
    description="One sentence on what the check flags, in neutral language.",
    category="consistency",
    params=ExampleParams,
    references=("Author (year)",),
)
def example_check(ctx: CheckContext, p: ExampleParams):
    found, missing = ctx.cols(ctx.config.variables.numeric)
    if not found:
        return ctx.missing_columns_issue("example_check", missing)
    ...  # build a boolean mask per row and call ctx.record_issues(...)
```

Guidance:

* Import the module from `checks/builtin/__init__.py` so that it registers on import.
* Use the context helpers (`cols`, `num`, `text`, `special_mask`, `record_issues`, `add_metric_series`,
  `enumerator_groups`, `missing_columns_issue`) rather than reading `ctx.data` directly, so special codes,
  missing columns and enumerator grouping behave the same way in every check.
* Vectorise across rows and groups. Avoid Python loops over submissions, and avoid process pools.
  Scikit-learn ensembles must use `n_jobs=1`.
* Return an empty issues table (`empty_issues`) when there is no data. Do not raise on missing columns, all-NaN
  columns or empty input. Record a note instead.
* Give each issue a severity that means "how urgently a supervisor should look", not "how bad the interviewer is".
* Add a reference for the method, or say plainly that the threshold is a convention.
* Add tests in `tests/` for: a clean dataset (no critical issues), a planted problem, missing columns, empty input,
  all-NaN columns, and timezone handling where timestamps are involved.
* Document the check in the catalogue table in `README.md`, with its defaults and its reference.

External check packages can register through the `fieldguard.checks` entry-point group.

## Adding an output or a connector

* Outputs should be written through `fieldguard.outputs` so that redaction, the `synthetic_` prefix and the
  atomic write (temporary file, then rename) apply.
* Connectors read credentials only from the environment, never print them, scrub them from error messages, and
  are tested with mocked HTTP. Add the variable names to `.env.example` with blank values.

## Pull requests

* Describe what changed and why, and which check, default or output it affects.
* State which tests you ran and their result, and confirm that ruff is clean.
* If the change affects a published number (for example the benchmark table in the README), regenerate the
  output from the synthetic data and say so.
* Keep changes focused. Do not reformat unrelated files.

## Licence

FieldGuard is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`). By contributing you agree that
your contribution is licensed the same way.
