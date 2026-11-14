"""Stage B, alternative route: hand the requests to an assistant session and ingest its answers.

    uv run python experiments/ai_fabrication/handoff.py export --split dev   # writes requests/<id>.txt
    (each request is answered by a fresh assistant session that sees only that file and writes answers/<id>.json)
    uv run python experiments/ai_fabrication/handoff.py ingest --model claude-opus-5-5-cc

Used when no API key is at hand: the 14 November 2026 runs used Claude Opus 5.5 through fresh Claude Code
subagents (one per request, no other context, no tools besides writing its answer). The requests live in a
neutral folder with numbered names so nothing about this study reaches the model; ``manifest.json`` maps the
numbers back to (strategy, seed, enumerator). Ingested files have the same shape as generate.py's output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PROMPTS, STRATEGIES, build_prompt, generated_path, load_json, record_schema, write_json  # noqa: E402

HANDOFF = Path(r"D:\ComputationalLab\data\fg-sprint")
FORMAT_NOTE = (
    "Give your answer as one JSON object of the form {{\"records\": [ ... ]}}, one object per household, each with "
    "exactly these keys: {keys}. Use null only where the questionnaire says null. Numbers as numbers, no comments."
)


def export(split: str, strategies: list[str]) -> None:
    manifest_path = HANDOFF / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    keys = ", ".join(record_schema()["properties"]["records"]["items"]["properties"])
    n = 0
    for path in sorted(PROMPTS.glob("seed*.json")):
        p = load_json(path)
        if split != "all" and p["split"] != split:
            continue
        for strategy in strategies:
            if any(v == [strategy, p["seed"], p["enumerator"]] for v in manifest.values()):
                continue
            rid = f"r{len(manifest) + 1:03d}"
            manifest[rid] = [strategy, p["seed"], p["enumerator"]]
            text = build_prompt(p, strategy) + "\n\n" + FORMAT_NOTE.format(keys=keys)
            (HANDOFF / "requests").mkdir(parents=True, exist_ok=True)
            (HANDOFF / "requests" / f"{rid}.txt").write_text(text, encoding="utf-8", newline="\n")
            n += 1
    write_json(manifest_path, manifest)
    print(f"{n} requests exported; manifest has {len(manifest)}")


def ingest(model: str) -> None:
    manifest = load_json(HANDOFF / "manifest.json")
    n = 0
    for rid, (strategy, seed, enumerator) in sorted(manifest.items()):
        ans = HANDOFF / "answers" / f"{rid}.json"
        refusal = HANDOFF / "answers" / f"{rid}.refusal.json"
        out = generated_path(model, strategy, seed, enumerator)
        if out.exists() or not (ans.exists() or refusal.exists()):
            continue
        payload = load_json(PROMPTS / f"seed{seed}_{enumerator}.json")
        refused = None
        if refusal.exists():
            refused, ans = load_json(refusal), refusal
            text, records, parse_error = "", [], None
        else:
            text = ans.read_text(encoding="utf-8")
            try:
                obj = json.loads(text)
                records = obj["records"]
                parse_error = None
                if obj.get("declined"):  # the assistant wrote its refusal into the answer file
                    refused = {"refused": True, "reply_summary": str(obj["declined"])}
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                records, parse_error = [], f"{type(exc).__name__}: {exc}"
        write_json(
            out,
            {
                "model": model,
                "strategy": strategy,
                "seed": seed,
                "split": payload["split"],
                "enumerator": enumerator,
                "n_requested": payload["n_records"],
                "n_returned": len(records),
                "records": records,
                "parse_error": parse_error,
                "refused": bool(refused),
                "refusal": refused,
                "prompt": (HANDOFF / "requests" / f"{rid}.txt").read_text(encoding="utf-8"),
                "request_id": rid,
                "at": datetime.fromtimestamp(ans.stat().st_mtime, UTC).isoformat(timespec="seconds"),
                "served_by": "Claude Opus 5.5 via a fresh Claude Code subagent (no API key route)",
                "label": "SYNTHETIC - model output for a synthetic village; not real survey data",
                "raw_text": text if parse_error else None,
            },
        )
        n += 1
        print(f"{rid} {strategy} seed {seed} {enumerator}: {len(records)}/{payload['n_records']} records"
              + (f" PARSE ERROR {parse_error}" if parse_error else ""))
    print(f"{n} answers ingested")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ex.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ing = sub.add_parser("ingest")
    ing.add_argument("--model", required=True)
    args = ap.parse_args()
    if args.cmd == "export":
        export(args.split, args.strategies)
    else:
        ingest(args.model)


if __name__ == "__main__":
    main()
