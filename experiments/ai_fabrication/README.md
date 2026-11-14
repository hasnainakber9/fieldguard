# Can survey fabrication checks catch AI-fabricated interviews?

Study for the AI x Epistemics Research Sprint (Apart Research, 13-15 November 2026), Track 2: Trust Infrastructure.
Author: Syed Hasnain Akber (KSBL).

**Question.** Household surveys feed development and humanitarian decisions, and their main defence against
interviewers who invent interviews ("curbstoning") is a set of statistical checks run while fieldwork is going on.
Those checks were designed for human fabricators, who are lazy in characteristic ways (copying, straight-lining,
round numbers, short evening interviews at home). A language model removes the labour. Do the checks still catch an
interviewer who asks a chat assistant to fill in the interviews, and can new answer-level signals close the gap?

**Everything here is synthetic.** The "genuine" interviews come from FieldGuard's own generator
(`fieldguard.synthetic`); the fabricated ones were written by language models for that synthetic village. No real
survey data or personal data is used anywhere.

## What existed before the sprint, and what is new

| | Status |
|---|---|
| FieldGuard v0.1.0: the package under `src/`, its 14 checks, enumerator scoring, synthetic generator and benchmark | **Preexisting** (written before 13 November 2026; unchanged by this study) |
| Everything in `experiments/ai_fabrication/` | **New, written during the sprint (13-15 November 2026)** |

New files: `common.py` (questionnaire, prompts, seed plan), `make_prompts.py`, `generate.py` and `handoff.py` (model
calls), `evaluate.py` (injection, FieldGuard runs, scoring), `signatures.py` (candidate answer-level signals),
`analysis.py` (tables and figures), `data/` (prompt payloads and every model answer, verbatim), `results/`, `report/`.

## Design

* 5 surveys (seeds): 2 for development, 3 held out (`common.py`, fixed before any model call). Each has 30
  enumerators and about 25 interviews per enumerator; 3 enumerators per survey fabricate 60-100% of their interviews.
* Three prompting strategies, each written as an ordinary request for "realistic" interviews, with no purpose stated:
  `S0_naive` (questionnaire only), `S1_informed` (plus summary statistics of colleagues' interviews in the same
  villages), `S2_careful` (plus a request to make the records look like genuine field data).
* Two paradata conditions: `content_only` (the fabricated interviews keep honest timestamps and GPS) and
  `with_paradata` (short evening interviews entered at home, as in FieldGuard's generator).
* Comparison: FieldGuard's own rule-based fabricators, restricted to the matching patterns.
* Answer key: which enumerators fabricated. Scores: FieldGuard's flag (risk score of 30 or more), the ROC-AUC of its
  risk score, and which checks fire for fabricators and for honest enumerators.

## Reproduce

```
uv sync --all-extras
uv run python experiments/ai_fabrication/make_prompts.py
# model answers are in data/generated/ (verbatim); to make new ones:
#   generate.py --model <model>   (Anthropic API or a local Ollama model), or handoff.py export/ingest
uv run python experiments/ai_fabrication/evaluate.py --split dev
uv run python experiments/ai_fabrication/evaluate.py --split test
uv run python experiments/ai_fabrication/analysis.py
```

The 14 November runs used Claude Opus 5.5 through fresh Claude Code subagents (one per request, with no context but
the request file, written to a neutral folder so nothing about this study reached the model; see `handoff.py`),
because no API key was available that weekend.

## Responsible use

The prompts are ordinary requests for synthetic data; they are published so the study can be checked, and they add
nothing a fabricator could not type unaided. The candidate signals are published as leads for supervisors, like
FieldGuard's other checks: a flag is not proof of misconduct.
