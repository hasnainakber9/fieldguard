"""Stage B: ask a language model to fabricate the interviews in each payload, and save exactly what came back.

    # Claude (needs an Anthropic credential; in ComputationalLab, agents/.env and the lab's monthly spend guard)
    D:/ComputationalLab/agents/.venv/Scripts/python.exe experiments/ai_fabrication/generate.py --model claude-opus-5-5
    # A local open-weights model through Ollama (free, slow on a laptop CPU)
    ... generate.py --model qwen3.5:9b --strategies S0_naive --split test

Outputs go to data/generated/<model>/<strategy>/seed<seed>_<enumerator>.json with the prompt, the parsed records,
the stop reason, any refusal, the model that actually answered and the token usage. Existing outputs are kept, so
an interrupted run resumes where it stopped. Nothing here contains real people's data: the prompts describe a
SYNTHETIC village.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROMPTS, STRATEGIES, build_prompt, generated_path, load_json, record_schema, write_json  # noqa: E402

AGENTS = Path(r"D:\ComputationalLab\agents")
PURPOSE = "sprint-ai-fabrication"
OLLAMA = "http://127.0.0.1:11434/api/chat"


def _claude(model: str, prompt: str) -> dict:
    import anthropic

    sys.path.insert(0, str(AGENTS / "src"))
    from agents_lab import spend_guard
    from dotenv import load_dotenv

    load_dotenv(AGENTS / ".env")
    spend_guard.check(PURPOSE, reserve_usd=0.60)
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model,
        max_tokens=32000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": record_schema()}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    est = spend_guard.record(msg.model, msg.usage, PURPOSE)
    iterations = [getattr(i, "type", "") for i in (getattr(msg.usage, "iterations", None) or [])]
    text = next((b.text for b in msg.content if b.type == "text"), "")
    refusal = None
    if msg.stop_reason == "refusal":
        d = getattr(msg, "stop_details", None)
        refusal = {"category": getattr(d, "category", None), "explanation": getattr(d, "explanation", None)}
    return {
        "served_by": msg.model,
        "fallback_ran": "fallback_message" in iterations,
        "stop_reason": msg.stop_reason,
        "refusal": refusal,
        "text": text,
        "usage": {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        },
        "est_usd": est,
        "request_id": getattr(msg, "_request_id", None),
    }


CSV_NOTE = (
    "Give your answer as CSV only: a header line with exactly these column names, then one line per household: "
    "{keys}. Leave a cell empty where the questionnaire says null. No other text."
)


def _csv_records(text: str) -> list[dict]:
    """Parse a CSV answer into records with the JSON route's types (numbers as numbers, empty as None)."""
    import csv
    import io

    body = text.strip().removeprefix("```csv").removeprefix("```").removesuffix("```").strip()
    rows = list(csv.DictReader(io.StringIO(body)))
    out = []
    for row in rows:
        rec = {}
        for k, v in row.items():
            if k is None:
                continue
            v = (v or "").strip()
            if v == "" or v.lower() == "null":
                rec[k.strip()] = None
                continue
            try:
                f = float(v.replace(",", ""))
                rec[k.strip()] = int(f) if f.is_integer() else f
            except ValueError:
                rec[k.strip()] = v
        out.append(rec)
    return out


def _ollama(model: str, prompt: str) -> dict:
    keys = ", ".join(record_schema()["properties"]["records"]["items"]["properties"])
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + "\n\n" + CSV_NOTE.format(keys=keys)}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": 8192, "num_predict": 6000, "temperature": 0.8},
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=4 * 3600) as resp:  # nosec B310 -- fixed local Ollama URL
        out = json.loads(resp.read())
    return {
        "served_by": model,
        "fallback_ran": False,
        "stop_reason": out.get("done_reason"),
        "refusal": None,
        "text": out.get("message", {}).get("content", ""),
        "usage": {"input_tokens": out.get("prompt_eval_count"), "output_tokens": out.get("eval_count")},
        "est_usd": 0.0,
        "seconds": (out.get("total_duration") or 0) / 1e9,
    }


def generate(model: str, payload: dict, strategy: str) -> dict:
    prompt = build_prompt(payload, strategy)
    t0 = time.perf_counter()
    is_claude = model.startswith("claude")
    raw = _claude(model, prompt) if is_claude else _ollama(model, prompt)
    try:
        if not raw["text"]:
            records = []
        elif is_claude:
            records = json.loads(raw["text"])["records"]
        else:
            records = _csv_records(raw["text"])
        parse_error = None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        records, parse_error = [], f"{type(exc).__name__}: {exc}"
    return {
        "model": model,
        "strategy": strategy,
        "seed": payload["seed"],
        "split": payload["split"],
        "enumerator": payload["enumerator"],
        "n_requested": payload["n_records"],
        "n_returned": len(records),
        "records": records,
        "parse_error": parse_error,
        "prompt": prompt,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "label": "SYNTHETIC - model output for a synthetic village; not real survey data",
        **{k: v for k, v in raw.items() if k != "text"},
        "answer_format": "json schema" if is_claude else "csv (prompt ends with CSV_NOTE)",
        "raw_text": raw["text"] if (parse_error or not is_claude) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new calls (0: no limit)")
    args = ap.parse_args()
    done = 0
    for path in sorted(PROMPTS.glob("seed*.json")):
        payload = load_json(path)
        if args.split != "all" and payload["split"] != args.split:
            continue
        for strategy in args.strategies:
            out = generated_path(args.model.replace(":", "_"), strategy, payload["seed"], payload["enumerator"])
            if out.exists():
                continue
            if args.limit and done >= args.limit:
                return
            result = generate(args.model, payload, strategy)
            write_json(out, result)
            done += 1
            print(
                f"{strategy} seed {payload['seed']} {payload['enumerator']}: {result['n_returned']}/"
                f"{result['n_requested']} records, stop {result['stop_reason']}, "
                f"{result['usage']['output_tokens']} out tokens, US${result['est_usd']:.3f}, {result['wall_seconds']}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
