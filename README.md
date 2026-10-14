# FieldGuard

Survey quality checks and fabrication detection ("high-frequency checks") for survey exports from
KoboToolbox, ODK Central and SurveyCTO. FieldGuard is a Python library and command-line tool for monitoring
and evaluation teams. It reads a survey export, runs a registry of checks, scores enumerators, and writes
issue tables, an Excel workbook, an HTML report and an optional PostgreSQL copy.

**FieldGuard flags leads for supervisory review. A flag is not proof of misconduct, and FieldGuard takes no
automated action against anyone.**

Status: alpha (v0.1.0). Licence: Apache-2.0 (see `LICENSE` and `NOTICE`).

> **No real survey data is included or was used in testing.** Demo and benchmark data come from the synthetic
> generator in `fieldguard.synthetic`, their outputs are labelled `SYNTHETIC`, and they describe the generator's
> assumptions rather than real fieldwork. Tests build small synthetic frames of their own. The benchmark numbers below are therefore not evidence of performance
> on real surveys. See [DATA_SOURCES.md](DATA_SOURCES.md).

## Contents

- [Quickstart](#quickstart)
- [Inputs and conventions](#inputs-and-conventions)
- [Example project file](#example-project-file)
- [Configuration reference](#configuration-reference)
- [Check catalogue](#check-catalogue)
- [Enumerator scoring](#enumerator-scoring)
- [Back-checks](#back-checks)
- [Outputs](#outputs)
- [PostgreSQL sink and daily incremental runs](#postgresql-sink-and-daily-incremental-runs)
- [Grafana dashboard](#grafana-dashboard)
- [Connectors](#connectors)
- [Privacy and ethics](#privacy-and-ethics)
- [Benchmark on synthetic data](#benchmark-on-synthetic-data)
- [Limitations](#limitations)
- [References](#references)
- [Development](#development)

## Quickstart

Requirements: Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Run every command from the project folder.

```bash
uv sync                                   # project venv with all extras (dev group includes fieldguard[all])

uv run python -m fieldguard version
uv run python -m fieldguard demo --out fieldguard_output/demo              # SYNTHETIC data, answer key included
uv run python -m fieldguard benchmark --out fieldguard_output/benchmark    # SYNTHETIC; add --quick for a fast check

uv run python -m fieldguard run --config my_project.yaml --out outputs/    # your export (data.path in the YAML)
uv run python -m fieldguard backcheck --config my_project.yaml             # draw a back-check sample (field only)
uv run python -m fieldguard backcheck --config my_project.yaml --compare reinterviews.csv   # compare re-interviews
```

`uv run fieldguard ...` works too (console script). If `--out` is omitted, `demo` and `benchmark` write to
`$FIELDGUARD_REPORTS_DIR/demo` (or `.../benchmark`) when that variable is set, otherwise to `fieldguard_output/`.
The CLI does not read `.env` files: export the variables in your shell.

Other commands: `run --incremental` (only record-level issues for submissions not seen before), `run --strict`
(stop at the first failing check), `run --no-sink`, `run --data FILE` (override `data.path`), and
`fetch --config FILE --out raw.csv` (download from a platform API; see [Connectors](#connectors)).

## Inputs and conventions

* **Files:** `.csv`, `.tsv`, `.txt`, `.xlsx`/`.xlsm` and `.parquet`. The delimiter is detected from the header
  (`;` is common in Kobo exports). Values are read as text so IDs such as `00123` keep their zeros.
* **Platform:** `project.platform` is `auto` (default), `kobo`, `odk`, `surveycto` or `generic`. Auto-detection
  uses the usual columns: `_uuid` / `_submission_time` / `_gps_latitude` (Kobo), `KEY` with `starttime`/`endtime`
  or `formdef_version` (SurveyCTO), and `__id` / `SubmissionDate` / `meta-instanceID` (ODK).
* **Group paths:** columns named with a group prefix (for example `resp/name`, `group/q1`) are used as exported.
* **GPS:** split columns (`_gps_latitude`, `_gps_longitude`, `_gps_precision`; or ODK/SurveyCTO `gps-Latitude`
  style) or a single geopoint column (`roles.gps.point`).
* **Timestamps:** `start` and `end` values are interpreted in `project.timezone` (or
  `project.naive_timestamps_timezone` when they carry no offset). Submission times follow the platform convention.
* **Special codes:** the defaults are `-99` / `dk` (don't know), `-98` / `refused`, `-97` (not applicable). Add
  project codes under `variables.special_codes`.

## Example project file

A short, valid project file (checked against the configuration model). The full annotated version is in
[examples/example_project.yaml](examples/example_project.yaml).

```yaml
project:
  name: Household survey pilot
  platform: kobo
  timezone: Africa/Nairobi
  fieldwork:
    start_date: 2026-03-02
    end_date: 2026-05-29

data:
  path: raw/household_export.csv        # relative paths resolve against this file

roles:
  submission_id: _uuid
  respondent_id: hhid
  enumerator_id: enum_id
  cluster: village
  start: start
  end: end
  survey_date: today
  deviceid: deviceid
  gps:
    lat: _gps_latitude
    lon: _gps_longitude
    accuracy: _gps_precision

variables:
  key: [monthly_income, food_exp]
  numeric: [age, hh_size, monthly_income, food_exp]
  required: [hhid, enum_id, consent]
  constraints:
    - {variable: age, min: 18, max: 100, soft_max: 85, severity: critical}
    - {id: children_le_size, expression: "n_children <= hh_size", message: "More children than household members"}
  skip_logic:
    - {id: business_skip, if: "has_business == 'no'", then_empty: [business_income]}

checks:
  outliers:
    method: mad
    threshold: 3.5
  gps:
    cluster_radius_m: 2500
    max_accuracy_m: 50

scoring:
  flag_threshold: 30

backcheck:
  sample_rate: 0.10
  min_per_enumerator: 2
```

Unknown keys are rejected, so typos fail when the file is loaded.

## Configuration reference

Every section is optional except that `data.path` (or `--data`) must be given for `run`. Defaults are shown.

| Section | Key fields (default) | Purpose |
|---|---|---|
| `project` | `name`, `platform` (`auto`), `timezone` (`UTC`), `naive_timestamps_timezone`, `submission_time_timezone`, `fieldwork.start_date`, `fieldwork.end_date` | Identity, fieldwork timezone and window. Interviews dated outside the window are flagged. |
| `data` | `path`, `sheet` (0), `delimiter` (detected), `encoding` (`utf-8-sig`), `source` (`type`: kobo/odk/surveycto, `asset_uid`, `server_url`, `project_id`, `form_id`, `page_size`) | Where the export is, and the optional remote source for `fetch`. Credentials are never stored here. |
| `roles` | `submission_id`, `respondent_id`, `enumerator_id`, `cluster`, `team`, `start`, `end`, `submission_time`, `survey_date`, `duration` (`duration_unit`: `seconds`), `deviceid`, `gps.lat`, `gps.lon`, `gps.accuracy`, `gps.point` | Maps the export's columns to the roles the checks need. |
| `variables` | `key` (enumerator effects and ICC), `numeric`, `required` (names or `{variable, when, severity}`), `special_codes`, `grids` (`columns`, `min_items`), `constraints`, `skip_logic`, `nonresponse`, `exclude` | What to check and how. |
| `variables.constraints[]` | `variable` or `expression`; `min`, `max`, `soft_min`, `soft_max`, `allowed`, `when`, `severity` (warning), `message` | Hard ranges, soft ranges, allowed values, or a cross-variable expression that must hold. |
| `variables.skip_logic[]` | `id`, `if`, `then_empty` or `then_required` or `then`, `severity`, `message` | Skip-logic consistency. `if` accepts Python or XLSForm-style conditions. |
| `checks` | Per-check parameters keyed by check id (see the catalogue). Omitting a check uses its defaults. `enabled: false` turns one off. | Thresholds and options. Unknown parameters are rejected. |
| `scoring` | `flag_threshold` (30), `anomaly_method` (`isolation_forest`; `lof` or `none`), `anomaly_weight` (0.25), `min_enumerators_for_anomaly` (6), `min_interviews` (3), `random_state` (42), `rule_overrides` | Enumerator risk score. Models run single-job so results are reproducible. |
| `backcheck` | `backchecker_column`, `id_column`, `variables[]` (`name`, `type` 1-3, `kind`, tolerances), `type1_threshold`/`type2_threshold`/`type3_threshold` (0.10), `sample_rate` (0.10), `min_per_enumerator` (2), `random_share` (0.5), `risk_oversample` (1.0), `contact_columns`, `seed` | Back-check sampling and comparison. |
| `privacy` | `pii_columns`, `redact_pii_in_reports` (true), `gps_decimals_in_reports` (3), `hash_respondent_ids` (false), `pseudonymize_enumerators` (false), `hash_salt_env` (`FIELDGUARD_HASH_SALT`) | What is kept out of shared outputs. |
| `outputs` | `dir`, `formats` (`csv`, `parquet`, `excel`, `html`), `max_rows_per_sheet`, `max_issues_per_check_html` | Output folder and formats. `--formats` on the CLI overrides this. |
| `sink` | `postgres` (false), `database_url_env` (`FIELDGUARD_DATABASE_URL`), `schema_name` (`fieldguard`) | Optional PostgreSQL copy. |
| `incremental` | `enabled`, `state` (`auto`, `file`, `postgres`), `state_dir` (`.fieldguard_state`) | Daily incremental state. |
| `synthetic` | `false` | Set by the demo and benchmark. Every output is labelled when true. |

## Check catalogue

Fourteen checks are registered. Each produces rows in one tidy issues table with the columns `run_id`,
`check_id`, `severity`, `level`, `record_id`, `respondent_id`, `enumerator`, `survey_date`, `variable`, `value`,
`metric`, `message`. Severities are `info`, `warning` and `critical`. Thresholds below are defaults; each is set
under `checks.<id>` in the project file. The list can be printed with the registry (`fieldguard.checks.registry`).

| Check id | Category | What it flags | Main defaults | Basis |
|---|---|---|---|---|
| `constraints` | consistency | Values outside hard or soft ranges, disallowed values, and cross-variable expressions that fail | from `variables.constraints` | IPA ipacheck; DIME Wiki |
| `skip_logic` | consistency | Answers that should be empty or required given the skip condition | from `variables.skip_logic` | DIME Wiki; J-PAL |
| `duplicate_ids` | integrity | Respondent IDs (or composite keys) that occur more than once, and duplicated submission UUIDs | `check_submission_ids: true` | J-PAL; IPA ipacheck |
| `near_duplicates` | fabrication | Pairs of interviews by the same enumerator that agree on at least 95% of compared questions | `threshold` 0.95, `min_compared` 10, `scope` `enumerator` | Kuriakose and Robbins (2016); Simmons et al. (2016) |
| `missing_required` | integrity | Required variables left empty (optionally only when a condition holds), and configured columns that are entirely empty | `flag_all_missing_columns: true` | DIME Wiki |
| `duration` | paradata | Interviews shorter than half the form median (and at least 5 minutes), and enumerators whose median duration is far below peers | `short_ratio` 0.5, `min_minutes` 5, `enumerator_median_ratio` 0.7 | Murphy et al. (2004); J-PAL; IPA ipacheck |
| `timestamps` | paradata | End before start, interviews outside working hours, overlapping interviews, too many interviews per day, dates outside the fieldwork window, timestamps in the future | `working_hours` 6-21, `max_per_day` 8, `overlap_tolerance_minutes` 1 | DIME Wiki; AAPOR (2003) |
| `gps` | paradata | Points more than `cluster_radius_m` from the cluster centre, identical coordinates across submissions, same-spot interviews by one enumerator, low accuracy, and concentrated GPS | `cluster_radius_m` 3000, `max_accuracy_m` 50, `min_distance_m` 10, `dispersion_ratio_threshold` 0.2 | DIME Wiki; IPA ipacheck |
| `outliers` | consistency | Modified z-score (MAD) or IQR fences, overall (warning) and within each enumerator's own interviews (info) | `method` mad, `threshold` 3.5, `iqr_k` 3.0, `min_n` 10 | Iglewicz and Hoaglin (1993); IPA ipacheck |
| `nonresponse` | enumerator | Enumerators whose don't-know or refusal rate differs strongly from peers, in either direction | `z_threshold` 3.0, `min_ratio` 2.0, `min_answers` 50 | IPA ipacheck; J-PAL |
| `straightlining` | fabrication | Identical or near-identical answers across a grid; enumerators with many such interviews | `sd_threshold` 0.5, `enumerator_share_threshold` 0.25, `min_n` 5 | Kemper and Menold (2014); Blasius and Thiessen (2015) |
| `heaping` | fabrication | Excess of values ending in 0 or 5, and round leading pairs (for example 2500), by enumerator against pooled data | `ratio_threshold` 1.5, `p_threshold` 0.01, `min_n` 20 | Shryock and Siegel (Whipple's index); Bredl et al. (2012) |
| `benford` | fabrication | First-digit distribution by enumerator, judged with Nigrini's MAD bands (0.006 close, 0.012 acceptable, 0.015 marginal, above that nonconformity) and a chi-square test | `min_n` 50, `min_value` 10, `excess_threshold` 1.5 | Nigrini (2012); Judge and Schechter (2009); Schräpler (2011) |
| `enumerator_effects` | enumerator | Intra-class correlation (ICC(1)) of key variables by enumerator, enumerators with atypical means, implausibly low variance, and categorical associations (Cramér's V) | `icc_threshold` 0.1, `p_threshold` 0.01, `mean_z_threshold` 4, `variance_ratio_threshold` 0.5, `min_n` 8 | West and Blom (2017); Schäfer et al. (2005); Bredl et al. (2012) |

The check status (ran, skipped, failed, issue count, seconds) is written to `check_status`. A failing check is
recorded and does not stop the run unless `--strict` is given. A check whose columns are missing reports which
columns it needed.

Severity is a triage label. The rationale for each check is in its docstring in `src/fieldguard/checks/builtin/`.

## Enumerator scoring

Each enumerator gets a **rule score** (0-100) and an **anomaly score** (0-100), combined into a **risk score**.

* **Rule score:** each indicator has a linear ramp from a "no concern" value to a "strong concern" value, and a
  weight. The points are summed and capped at 100, and the top three contributing indicators are reported as
  reasons. The ramps and weights are documented in `src/fieldguard/scoring/rules.py` and can be overridden under
  `scoring.rule_overrides`. They are judgement-based and have not been calibrated on real data.
* **Anomaly score:** an Isolation Forest (or LOF, or none) on robust-standardised enumerator-level features,
  single job with a fixed seed. It needs at least `min_enumerators_for_anomaly` eligible enumerators; otherwise the
  anomaly score is blank and the risk score equals the rule score. Isolation Forest scores are mapped to 0-100 as
  `(s - 0.5) / 0.25`.
* **Risk score:** `rule * (1 - anomaly_weight) + anomaly * anomaly_weight`. An enumerator is **flagged** when the
  risk score reaches `flag_threshold` (30) and they have at least `min_interviews` interviews.

Flags go to supervisors as leads. The scorecard shows the reasons and the underlying metrics so a reviewer can
see why an enumerator was flagged.

## Back-checks

`fieldguard backcheck` works in two modes.

1. **Select** (no `--compare`): draws a stratified sample. For each enumerator, `sample_rate` of their interviews
   are drawn (at least `min_per_enumerator` where they have that many). Half the sample (`random_share`) is
   uniform, and half is weighted by risk, with flagged enumerators oversampled. The sample is written as
   `backcheck_sample_FIELD_ONLY.csv`. It can contain contact columns and must not be shared with enumerators.
2. **Compare** (`--compare reinterviews.csv`): joins re-interviews to the survey on `roles.respondent_id`. Each
   variable is compared by its kind (categorical, text similarity, numeric tolerance, date window). Variables
   are grouped into three types: type 1 (factual identity and process, exact match), type 2 (administrative or
   factual, small differences tolerated), and type 3 (key outcomes expected to vary, wide tolerance). Error rates
   are reported per variable, per enumerator and per type.

The 10% threshold for each type is a common convention, not a validated cut-off. A rate above it is a lead for
supervision. The default sample rate of 10% is the lower end of the 10-20% range in the DIME Wiki's *Monitoring
Data Quality* guidance, which also recommends back-checking every enumerator (the default `min_per_enumerator` is 2).

## Outputs

`run` writes into one folder per run. The folder is `--out` if given; otherwise `$FIELDGUARD_REPORTS_DIR/<project>/<run_id>`
if that variable is set; otherwise `outputs/<project>/<run_id>` relative to the project file:

| File | Content |
|---|---|
| `issues.csv`, `issues.parquet` | The tidy issues table (redacted) |
| `scorecard.csv`, `scorecard.parquet` | One row per enumerator: scores, flags, reasons and metrics |
| `metrics.csv` | Enumerator-level metrics used by the scores |
| `check_status.csv` | Each check's status, issue count, runtime and message |
| `fieldguard_workbook.xlsx` | README sheet, scorecard, one sheet per check (empty if clean), check status |
| `report.html` | Self-contained HTML report with inline SVG charts (no external resources) |
| `run_summary.json` | Run metadata, counts, check status and the synthetic label |
| `backcheck_*` | Back-check sample (field only), comparisons, variable and enumerator summaries, coverage |

Files from synthetic data start with `synthetic_` and carry the label in the summary, workbook and HTML banner.
Formats are selected with `--formats csv,parquet,excel,html`. If `pyarrow` is not installed, Parquet is skipped
with a warning and the other formats are still written.

## PostgreSQL sink and daily incremental runs

* **Sink:** set `FIELDGUARD_DATABASE_URL` (SQLAlchemy URL with the psycopg driver) and `sink.postgres: true` (or
  just set the variable). FieldGuard creates the `fieldguard` schema with tables `runs`, `issues`, `scorecards`
  and `seen_records`, and inserts one run per call.
* **Unreachable database:** the sink is skipped with a one-line message that names only the exception type. The
  URL and password are never printed, and the run still writes its files. Live-database tests are marked
  `postgres` and skip unless the variable points to a reachable server.
* **Daily incremental mode:** `fieldguard run --config ... --incremental` (for example from a daily scheduler).
  Checks still see the whole dataset, so enumerator-level results stay comparable. Only record-level issues for
  submissions not seen before are reported, and the seen IDs are kept in the database when it is reachable
  (`incremental.state: auto`) or in `.fieldguard_state/` otherwise.

Tested in this session: the file-state incremental path (a second run on the same data reported 0 new records)
and the skip behaviour with an unreachable database. The live PostgreSQL round trip was **not** run (no
database was available), so it is verified only by code review and a skipped test.

## Grafana dashboard

`integrations/grafana/fieldguard.json` is a dashboard for the PostgreSQL schema above. It has a Postgres
datasource variable (`DS_POSTGRES`), a `project` variable and an `include_synthetic` switch. Panels show flagged
enumerators, critical issues, records and enumerators in the latest run, issues by severity over runs, flagged
enumerators over runs, the enumerator risk table, issues by check and the run list. Every table and
alias-qualified column in the panel SQL was checked against the DDL in `sink/postgres.py`. The dashboard has **not**
been loaded in a running Grafana instance in this session. Provisioning the datasource is left to the coordinator.

## Connectors

Credentials come only from environment variables, never from YAML, and are never printed or stored. The
variables are listed in `.env.example`.

| Platform | Variables | Notes |
|---|---|---|
| KoboToolbox API v2 | `KOBO_SERVER_URL` (default `https://kf.kobotoolbox.org`), `KOBO_TOKEN`, `KOBO_ASSET_UID` | The asset UID may also be set as `data.source.asset_uid`. |
| ODK Central (OData) | `ODK_CENTRAL_URL`, `ODK_CENTRAL_PROJECT_ID`, `ODK_CENTRAL_FORM_ID`, and either `ODK_CENTRAL_TOKEN` or `ODK_CENTRAL_EMAIL` with `ODK_CENTRAL_PASSWORD` | A read-only web user is preferable. |
| SurveyCTO server API | `SURVEYCTO_SERVER` (server name only), `SURVEYCTO_FORM_ID`, `SURVEYCTO_USERNAME`, `SURVEYCTO_PASSWORD` | `--since` limits the pull by date. |

`fieldguard fetch --config project.yaml --out raw/export.csv` downloads submissions into a CSV that `run` can read.
Downloads are paged and resumable through a page cache. All three connectors are tested with mocked HTTP only.
No account was created and no live server was contacted while building this release.

## Privacy and ethics

* **Leads, not findings.** Every output says so. FieldGuard takes no sanction and sends no report to anyone. The risk
  rank only orders the review queue. Use the flags to decide where to look: back-checks, supervision, a conversation
  with the team.
* **Redaction in shared outputs.** Columns listed in `privacy.pii_columns` are not quoted in issue values, and
  phone-like digit runs in free text are replaced with `[phone]`. GPS coordinates in issues are rounded to
  `gps_decimals_in_reports` (3 decimals, about 110 m). Respondent IDs and enumerator names can be replaced with
  salted hashes (`hash_respondent_ids`, `pseudonymize_enumerators`, salt from `FIELDGUARD_HASH_SALT`).
  Testing on synthetic data confirmed that the synthetic names, phone numbers and raw coordinates do not appear in
  the shared issue, scorecard, workbook, report or summary files. This is not a guarantee for your data: check
  your own outputs and the `pii_columns` list before sharing.
* **Field-only files.** The back-check sample (`*_FIELD_ONLY.csv`) is for the field team. Contact columns appear
  only there.
* **Credentials.** Read from the environment, never printed. Database URLs are shown without passwords.

## Benchmark on synthetic data

`fieldguard benchmark` generates surveys in which some enumerators fabricate (copied, straight-lined,
digit-heaped, short or GPS-clustered interviews, among other patterns), runs the checks and scores the flags
against the answer key. It covers contamination rates of 5%, 10%, 20% and 30% of 30 enumerators, with three
seeds each, and writes per-run and summary tables and a JSON file under the output folder.

Results from the run recorded in this release (`C:/ComputationalLab/reports/fieldguard/benchmark/`; synthetic; 30
enumerators and about 25 interviews each; 3 seeds per rate; mean over seeds, with the standard deviation in brackets
where it is not zero):

| Contamination | Fabricators (mean) | Precision | Recall | F1 | ROC-AUC, risk score | ROC-AUC, rule score | ROC-AUC, anomaly score |
|---|---|---|---|---|---|---|---|
| 5% | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10% | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20% | 6 | 1.00 | 0.83 (0.17) | 0.90 | 1.00 | 1.00 | 0.94 |
| 30% | 9 | 1.00 | 0.82 (0.13) | 0.89 | 1.00 | 1.00 | 0.93 |

Precision and recall are computed from the flagged enumerators against the answer key. Recall below 1.00 means
some fabricators were not flagged. No honest enumerator was flagged in any run.

These numbers reflect the generator's assumptions. Each fabricating enumerator applies 2 to 4 of the 10 modelled
patterns to 60% to 100% of their interviews, so the signal is strong by design. Perfect separation is expected
from that design and should not be read as performance on real surveys, where fabrication is subtler and the
answer key is uncertain. The benchmark is a regression test for the pipeline, not a validation study.

## Limitations

* **No real data validation.** No real survey export was available. No threshold has been calibrated on real
  fieldwork, and the rule weights are judgement.
* **Synthetic results are optimistic.** See the benchmark section.
* **Small samples.** Enumerator-level statistics (Benford, heaping, ICC, Isolation Forest) are unstable with few
  interviews or enumerators. Check `min_n` and `min_enumerators_for_anomaly` for your design.
* **Platform coverage.** Export conventions are handled for the column names described in the inputs section.
  Unusual form designs (repeat groups, custom naming) may need `roles` and `variables` settings.
* **Not tested here:** the live PostgreSQL round trip, the Grafana dashboard in a running Grafana, the Docker
  image (Docker was off), and live calls to the three platform APIs.

## References

Sources marked (checked) were confirmed in this session against the publisher, repository or a citing paper. The
others are the citations used in the code and should be verified before you cite them in a publication.

* J-PAL, *Data quality checks* (checked): https://www.povertyactionlab.org/resource/data-quality-checks
* IPA, *ipacheck* high-frequency checks template, Stata (checked): https://github.com/PovertyAction/high-frequency-checks
* IPA Knowledge Hub, *High Frequency Checks* (checked): https://data.poverty-action.org/data-quality/high-frequency-checks.html
* DIME Analytics, *High Frequency Checks*, DIME Wiki (checked): https://dimewiki.worldbank.org/High_Frequency_Checks
* DIME Analytics, *Back Checks* and *Monitoring Data Quality*, DIME Wiki (checked for the 10-20% guidance):
  https://dimewiki.worldbank.org/Back_Checks
* Nigrini, M. J. (2012). *Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud Detection*. Wiley.
* Judge, G., and Schechter, L. (2009). Detecting problems in survey data using Benford's law. *Journal of Human Resources*.
* Schräpler, J.-P. (2011). Benford's law as an instrument for fraud detection in surveys using the data of the
  Socio-Economic Panel (SOEP).
* Schäfer, C., Schräpler, J.-P., Müller, K.-R., and Wagner, G. G. (2005). Automatic identification of faked and
  fraudulent interviews in the German SOEP. *Schmollers Jahrbuch* (checked).
* Bredl, S., Winker, P., and Kötschau, K. (2012). A statistical approach to detect interviewer falsification of
  survey data. *Survey Methodology*, 38(1) (checked).
* Kuriakose, N., and Robbins, M. (2016). Don't get duped: Fraud through duplication in public opinion surveys.
  *Statistical Journal of the IAOS* (existence checked through a citing paper).
* Simmons et al. (2016). Evaluating a new proposal for detecting data falsification in surveys. *Statistical
  Journal of the IAOS* (title and journal checked; author list not checked).
* Blasius, J., and Thiessen, V. (2015). Should we trust survey data? Assessing response simplification and data
  fabrication. *Social Science Research*.
* Kemper, C. J., and Menold, N. (2014). Nuisance or remedy? The utility of stylistic responding as an indicator of
  data fabrication in surveys. *Methodology*.
* West, B. T., and Blom, A. G. (2017). Explaining interviewer effects: a research synthesis. *Journal of Survey
  Statistics and Methodology*.
* Iglewicz, B., and Hoaglin, D. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
* Murphy, J., et al. (2004). A system for detecting interviewer falsification. AAPOR proceedings.
* AAPOR (2003). *Interviewer Falsification in Survey Research: Current Best Methods for Prevention, Detection and
  Repair of Its Effects*.
* Shryock, H., and Siegel, J. (*Methods and Materials of Demography*), for Whipple's index.

## Development

```bash
uv sync
uv run pytest -q                    # the live Postgres test skips unless FIELDGUARD_DATABASE_URL points at a reachable server
uv run ruff check --fix .
uv run ruff format .
```

Tests live in `tests/`. They cover every check module with edge cases (missing columns, timezones, empty data,
all-NaN columns), configuration validation, mocked connectors, report generation, the sink (skipped when
unreachable) and a small benchmark.
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a check.

Licence: Apache-2.0. See `LICENSE` and `NOTICE`.
