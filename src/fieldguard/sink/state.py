"""Incremental state: which submissions were already reported, kept in a JSON file or in the Postgres sink.

Daily incremental mode (``fieldguard run --incremental``): checks still see the whole dataset, so enumerator-level
statistics stay comparable, but record-level issues are reported only for submissions not seen before. After a
successful run every submission in the data is recorded as seen. Failed runs record nothing, so the next run
reports those submissions again.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from fieldguard.config import FieldGuardConfig

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def state_file(cfg: FieldGuardConfig) -> Path:
    folder = cfg.resolve_path(cfg.incremental.state_dir)
    slug = _SAFE.sub("-", cfg.project.slug or "project").strip("-") or "project"
    return folder / f"{slug}.seen.json"


def load_seen_file(cfg: FieldGuardConfig) -> set[str]:
    path = state_file(cfg)
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(x) for x in data.get("seen", [])}


def save_seen_file(cfg: FieldGuardConfig, record_ids: Iterable[str], run_id: str) -> Path:
    path = state_file(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"project": cfg.project.slug, "last_run_id": run_id, "seen": sorted({str(x) for x in record_ids})}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_seen_postgres(url: str, cfg: FieldGuardConfig) -> set[str]:
    from sqlalchemy import create_engine, text

    schema = cfg.sink.schema_name
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT record_id FROM {schema}.seen_records WHERE project = :p"), {"p": cfg.project.slug}
            )
            return {str(r[0]) for r in rows}
    finally:
        engine.dispose()


def save_seen_postgres(url: str, cfg: FieldGuardConfig, record_ids: Iterable[str], run_id: str) -> int:
    from sqlalchemy import create_engine, text

    schema = cfg.sink.schema_name
    rows = [{"p": cfg.project.slug, "r": str(x), "run": run_id} for x in sorted({str(x) for x in record_ids})]
    if not rows:
        return 0
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {schema}.seen_records (project, record_id, first_run_id) VALUES (:p, :r, :run) "
                    "ON CONFLICT (project, record_id) DO NOTHING"
                ),
                rows,
            )
    finally:
        engine.dispose()
    return len(rows)


__all__ = ["load_seen_file", "load_seen_postgres", "save_seen_file", "save_seen_postgres", "state_file"]
