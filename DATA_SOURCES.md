# Data sources

This file lists every data source FieldGuard depends on, what is and is not in the repository, and where the
outputs of this release are stored. It was last checked on 2026-10-31.

## Summary

* **Real survey data:** none. The repository ships no survey exports, and none were used for development,
  tests, the demo or the benchmark.
* **Synthetic data:** produced by `src/fieldguard/synthetic.py` (seeded, labelled `SYNTHETIC`). It is the only
  input to the demo, the benchmark, and the tests that need survey-shaped data.
* **External services:** the KoboToolbox, ODK Central and SurveyCTO connectors can download a real project, but
  they were tested with mocked HTTP only. No account was used and no live server was contacted.
* **Reference documents:** cited for method and practice. Only their links are included. No text was copied.

## 1. Repository contents

| Path | What it is | Origin |
|---|---|---|
| `examples/example_project.yaml` | Annotated example project file for a household survey | Written for this project. It refers to `raw/household_export.csv`, which is not included. |
| `integrations/grafana/fieldguard.json` | Grafana dashboard for the PostgreSQL schema | Written for this project. Imported into a throwaway Grafana 13.2 container through the import API; every panel query ran against a database written by the sink (synthetic data only). Not reviewed visually in a browser. |
| `.env.example` | Names of environment variables; all values are blank or placeholders | Written for this project. |
| `tests/` | Pytest suite. Survey-shaped data is generated inside the tests (`tests/conftest.py`, `fieldguard.synthetic`). | Written for this project. No binary fixtures. |

## 2. Synthetic generator (`fieldguard.synthetic`)

The generator mimics a KoboToolbox-style CSV export of a household survey: group paths such as `wb/wb_1`,
`_uuid`, `_submission_time`, `start` and `end` with UTC offsets, and split GPS columns. It is used by:

* `fieldguard demo` (writes the raw synthetic survey, the answer keys and a field-only back-check sample);
* `fieldguard benchmark` (many surveys across contamination rates and seeds);
* the test suite.

Default assumptions (`SyntheticSpec`):

| Parameter | Default |
|---|---|
| Enumerators | 30 |
| Interviews per enumerator | 25 on average (realised counts vary by enumerator) |
| Clusters (villages) | 15 |
| Fieldwork start | 2025-03-03 |
| Timezone | Africa/Nairobi |
| Fabricating share of enumerators | `contamination` (demo 0.15; benchmark 0.05, 0.10, 0.20, 0.30) |
| Patterns per fabricator | 2 to 4 of 10 (`short_duration`, `straightlining`, `low_variance`, `benford`, `heaping`, `gps_clustering`, `timing`, `skip_shortcut`, `low_dk`, `near_duplicate`) |
| Share of a fabricator's interviews affected | 60% to 100% |
| Random seed | 42 for the demo; 1, 2, 3 for the benchmark |

Honest enumerators have realistic variation in speed, DK propensity, working hours, rounding habits and village
differences. The fabrication model is a plausible generator, not a description of real fieldwork. Results on
it do not transfer to real surveys without testing on real data.

Every synthetic row carries `_synthetic`. Every output file from synthetic data starts with `synthetic_` and
carries the label in the summary, workbook and HTML report.

The defaults of the `answer_regularity` check (its three signals, the threshold of 3.5 and the minimum of 8
interviews) come from a synthetic-only study of model-written interviews, kept on the branch
`sprint/ai-fabrication` (`experiments/ai_fabrication/`). Its honest interviews come from this generator and its
fabricated ones were written by a language model for the generator's synthetic villages; no real survey data was
used. `synthetic_config()` sets that study's household profile for the check, so the demo and the benchmark use
the signals exactly as they were defined there.

## 3. Real survey data: search and status

* Local folders were searched by file name: `D:/ComputationalLab/data/` contains geospatial and other project data
  (for example Pakistan administrative boundaries, a relative wealth index file and a building-footprint dataset
  index). None of it is a household survey export with enumerator IDs and submission timestamps, so none was used.
  Some files in another project's folder have names that refer to a household education survey (ASER); they were
  not opened or used.
* A web search for an openly licensed survey export with enumerator IDs and timestamps found no candidate that
  could be used quickly. Most public microdata releases do not include enumerator IDs, submission times or GPS at
  the level the checks need, and several require registration or accepting terms, which this project does not do.
  Nothing was downloaded.
* If a real export becomes available, it should be placed outside the repository (for example under
  `D:/ComputationalLab/data/fieldguard/raw/`), and the project file should point to it. Real data must not be
  committed.

## 4. External services

| Service | Used for | Data retrieved in this release |
|---|---|---|
| KoboToolbox API v2 | `fieldguard fetch` (`type: kobo`) | None |
| ODK Central OData | `fieldguard fetch` (`type: odk`) | None |
| SurveyCTO server API | `fieldguard fetch` (`type: surveycto`) | None |
| PostgreSQL (optional sink) | `FIELDGUARD_DATABASE_URL` | None. The sink only writes. The live tests (`-m postgres`) and incremental runs wrote synthetic demo results to throwaway local databases, which were dropped afterwards. |

Credentials are read from environment variables and are never stored in files or printed.

## 5. Reference documents

Methods and practice were checked against these sources. Only links and short paraphrases are used.

* J-PAL, *Data quality checks*: https://www.povertyactionlab.org/resource/data-quality-checks
* IPA, *ipacheck* high-frequency checks template: https://github.com/PovertyAction/high-frequency-checks
* IPA Knowledge Hub, *High Frequency Checks*: https://data.poverty-action.org/data-quality/high-frequency-checks.html
* DIME Analytics, *High Frequency Checks* and *Back Checks*, DIME Wiki: https://dimewiki.worldbank.org/High_Frequency_Checks and https://dimewiki.worldbank.org/Back_Checks

The remaining references in `README.md` are cited from the literature. Each was checked in October 2026 against
the publisher, Crossref or the organisation's own page.

## 6. Outputs of this release

| Location | Content | Shared? |
|---|---|---|
| `D:/ComputationalLab/reports/fieldguard/demo/` | Issues, scorecard, metrics, check status, workbook, HTML report, run summary, configuration and back-check summaries from the demo (all `synthetic_`) | Yes. Contains no raw survey rows, no answer keys and no field-only sample. |
| `D:/ComputationalLab/reports/fieldguard/benchmark/` | Benchmark runs, summary and JSON (`synthetic_`) | Yes |
| `D:/ComputationalLab/data/fieldguard/demo/` | Raw synthetic survey, answer keys (enumerator and record level), re-interviews, the field-only back-check sample (`*_FIELD_ONLY.csv`, which contains contact columns) and a full copy of the demo outputs | No. Field and data use only. |
| `D:/ComputationalLab/data/fieldguard/run_check*/` | Outputs of test runs of `fieldguard run` on the synthetic demo data (file-state incremental runs and the unreachable-sink check) | No |
