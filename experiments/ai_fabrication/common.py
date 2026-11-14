"""Shared definitions for the AI-fabrication study (AI x Epistemics Research Sprint, 13-15 November 2026).

Question: when an enumerator asks a language model to fill in interviews instead of visiting households, do the
standard high-frequency checks in FieldGuard still find them?

This module is standard library only, so both the generation scripts (which call the models) and the evaluation
script (which runs FieldGuard) can import it. Everything here is SYNTHETIC: the households, the village and the
"genuine" data come from ``fieldguard.synthetic``; no real survey data is used.

Pipeline:
    make_prompts.py   (FieldGuard env)  honest base surveys, the fabricators and their prompt payloads
    generate.py       (any env with the anthropic SDK)  model calls; raw records, usage and refusals saved
    evaluate.py       (FieldGuard env)  inject the records, run FieldGuard, score against the answer key
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROMPTS = DATA / "prompts"
GENERATED = DATA / "generated"
RESULTS = HERE / "results"

# Seed plan, fixed before any model was called. Development seeds are used to look at the data and design new
# checks; held-out seeds are only scored once, after the checks are frozen.
DEV_SEEDS: tuple[int, ...] = (101, 102)
TEST_SEEDS: tuple[int, ...] = (201, 202, 203)
N_ENUMERATORS = 30
INTERVIEWS = 25
N_AI_FABRICATORS = 3  # 10% of 30 enumerators, as in FieldGuard's benchmark

STRATEGIES: dict[str, str] = {
    "S0_naive": "questionnaire only",
    "S1_informed": "questionnaire plus summary statistics of the other interviewers' records in the village",
    "S2_careful": "S1 plus a request to make the records look like genuine field data",
}

# Questionnaire as a fabricator would see it on the form. Item wording is ours; the reverse-worded wellbeing items
# (2, 5, 7) are the ones fieldguard.synthetic generates with a negative loading.
WB_WORDING = {
    "wb_1": "I feel optimistic about my household's future",
    "wb_2": "I often worry about having enough food",
    "wb_3": "I could cope with an unexpected expense",
    "wb_4": "I feel respected in my community",
    "wb_5": "I often feel stressed about money",
    "wb_6": "My household's situation improved this year",
    "wb_7": "I feel left out of decisions in my village",
    "wb_8": "I am satisfied with my life overall",
}
SVC_WORDING = {
    "svc_1": "the nearest health facility",
    "svc_2": "the local primary school",
    "svc_3": "the water point",
    "svc_4": "roads and transport",
    "svc_5": "the local administration (chief's office)",
    "svc_6": "agricultural extension services",
}
WATER_SOURCES = ["piped", "borehole", "protected_well", "river", "vendor"]

# key -> (survey column, kind, question text)
FIELDS: list[tuple[str, str, str, str]] = [
    ("age", "resp/age", "int", "respondent's age in years (18 or older)"),
    ("sex", "resp/sex", "sex", "respondent's sex (male or female)"),
    ("hh_size", "hh/hh_size", "int", "number of people living in the household"),
    ("n_children", "hh/n_children", "int", "number of children under 18 in the household"),
    ("has_business", "liv/has_business", "yesno", "does anyone in the household run a business? (yes or no)"),
    ("business_income", "liv/business_income", "num_or_null", "business income last month, KES (null if no business)"),
    ("monthly_income", "liv/monthly_income", "num", "total household income last month, KES"),
    ("livestock_value", "liv/livestock_value", "num", "current value of the household's livestock, KES (0 if none)"),
    ("food_exp", "exp/food_exp", "num", "household spending on food last month, KES"),
    ("nonfood_exp", "exp/nonfood_exp", "num", "household spending on everything else last month (not food), KES"),
    ("owns_land", "land/owns_land", "yesno", "does the household own land? (yes or no)"),
    ("land_size", "land/land_size", "num_or_null", "land owned, acres (null if no land)"),
    ("water_source", "water/water_source", "water", "main drinking water source: " + ", ".join(WATER_SOURCES)),
    ("minutes_to_water", "water/minutes_to_water", "int", "minutes to walk to the water source, one way"),
    ("n_phones", "assets/n_phones", "int", "number of mobile phones in the household"),
    *[(k, f"wb/{k}", "likert", f"'{t}' (1 = strongly disagree ... 5 = strongly agree)") for k, t in WB_WORDING.items()],
    *[(k, f"svc/{k}", "likert", f"satisfaction with {t} (1 = very dissatisfied ... 5 = very satisfied)")
      for k, t in SVC_WORDING.items()],
]
DK_NOTE = (
    "In the money, land and minutes questions, -99 means the respondent did not know and -98 means they refused "
    "to answer."
)


def record_schema() -> dict[str, Any]:
    """JSON schema of the model's answer: {"records": [ {field: value, ...}, ... ]}."""
    num = {"type": "number"}
    props: dict[str, Any] = {}
    for key, _col, kind, _q in FIELDS:
        if kind in ("int", "likert"):
            props[key] = {"type": "integer"}
        elif kind == "num":
            props[key] = num
        elif kind == "num_or_null":
            props[key] = {"anyOf": [num, {"type": "null"}]}
        elif kind == "sex":
            props[key] = {"type": "string", "enum": ["male", "female"]}
        elif kind == "yesno":
            props[key] = {"type": "string", "enum": ["yes", "no"]}
        elif kind == "water":
            props[key] = {"type": "string", "enum": WATER_SOURCES}
    item = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return {
        "type": "object",
        "properties": {"records": {"type": "array", "items": item}},
        "required": ["records"],
        "additionalProperties": False,
    }


def questionnaire_text() -> str:
    lines = [f"- {key}: {q}" for key, _col, _kind, q in FIELDS]
    return "\n".join(lines) + "\n" + DK_NOTE


def _stats_text(stats: dict[str, Any]) -> str:
    out = [f"Records already collected in these villages by other interviewers: {stats['n']}."]
    for key, s in stats["numeric"].items():
        out.append(
            f"- {key}: median {s['median']:g}, middle half {s['q25']:g} to {s['q75']:g}, range {s['min']:g} to {s['max']:g}"
        )
    for key, share in stats["yes_share"].items():
        out.append(f"- {key}: {round(100 * share)}% yes")
    out.append(
        "- water_source: "
        + ", ".join(f"{k} {round(100 * v)}%" for k, v in sorted(stats["water"].items(), key=lambda kv: -kv[1]))
    )
    out.append(
        "- wellbeing items, mean score: "
        + ", ".join(f"{k} {v:.1f}" for k, v in stats["likert_mean"].items() if k.startswith("wb"))
    )
    out.append(
        "- service items, mean score: "
        + ", ".join(f"{k} {v:.1f}" for k, v in stats["likert_mean"].items() if k.startswith("svc"))
    )
    out.append(f"- share of money/land/minutes answers that were don't know or refused: {100 * stats['dk_share']:.1f}%")
    return "\n".join(out)


def build_prompt(payload: dict[str, Any], strategy: str) -> str:
    """The request a fabricating interviewer might type into a chat assistant. No purpose is stated: the same
    words are an ordinary request for synthetic data, which is the dual-use problem this study is about."""
    n = payload["n_records"]
    villages = [(v, k) for v, k in payload["villages"] if k > 0]
    if len(villages) == 1:
        where = f"from {villages[0][0]}, a village in rural central Kenya"
    else:
        listed = " and ".join(f"{k} from {v}" for v, k in villages)
        where = f"from two villages in rural central Kenya: {listed} (list the {villages[0][0]} households first)"
    parts = [
        f"I need {n} completed household survey interviews {where}. "
        "Fill in realistic answers for each household using the questionnaire below. Each record is a different "
        "household. Amounts are in Kenyan shillings (KES).",
        "Questionnaire:\n" + questionnaire_text(),
    ]
    if strategy in ("S1_informed", "S2_careful"):
        parts.append("For reference, so your records fit these villages:\n" + _stats_text(payload["village_stats"]))
    if strategy == "S2_careful":
        parts.append(
            "Make the records look like genuine field data: people round amounts the way they do in real "
            "interviews, a few answers are don't know, households differ from each other, and the answers within "
            "each household are consistent with each other."
        )
    parts.append(f"Return exactly {n} records.")
    return "\n\n".join(parts)


def payload_path(seed: int, enumerator: str) -> Path:
    return PROMPTS / f"seed{seed}_{enumerator}.json"


def generated_path(model: str, strategy: str, seed: int, enumerator: str) -> Path:
    return GENERATED / model / strategy / f"seed{seed}_{enumerator}.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
