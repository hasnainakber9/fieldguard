"""Resolve config variable names to export columns.

Kobo exports group paths as ``group/q1``, ODK Central CSV as ``group-q1``; SurveyCTO uses bare names.
A config may refer to ``q1`` and FieldGuard finds the unique column whose last path segment matches.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SPLIT = re.compile(r"[/]")


class AmbiguousColumnError(KeyError):
    pass


class ColumnResolver:
    def __init__(self, columns: Iterable[str]):
        self.columns: list[str] = [str(c) for c in columns]
        self._exact = set(self.columns)
        self._lower: dict[str, list[str]] = {}
        self._slash_tail: dict[str, list[str]] = {}
        self._dash_tail: dict[str, list[str]] = {}
        for col in self.columns:
            self._lower.setdefault(col.lower(), []).append(col)
            if "/" in col:
                self._slash_tail.setdefault(_SPLIT.split(col)[-1], []).append(col)
            if "-" in col:
                self._dash_tail.setdefault(col.rsplit("-", 1)[-1], []).append(col)

    def resolve(self, name: str | None) -> str | None:
        """Return the matching column or None. Raises AmbiguousColumnError for >1 candidate."""
        if name is None:
            return None
        if name in self._exact:
            return name
        for table, key in (
            (self._slash_tail, name),
            (self._dash_tail, name),
            (self._lower, name.lower()),
        ):
            hits = table.get(key, [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise AmbiguousColumnError(f"{name!r} matches several columns {hits}; use the full column name")
        # also allow config to use 'group/q1' when the export flattened groups to 'group-q1' or 'q1'
        if "/" in name:
            alt = name.replace("/", "-")
            if alt in self._exact:
                return alt
            return self.resolve(name.split("/")[-1]) if name.split("/")[-1] != name else None
        return None

    def require(self, name: str) -> str:
        col = self.resolve(name)
        if col is None:
            raise KeyError(f"column {name!r} not found in data")
        return col

    def resolve_many(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        """Resolve names; returns (found_columns, missing_names)."""
        found: list[str] = []
        missing: list[str] = []
        for n in names:
            try:
                col = self.resolve(n)
            except AmbiguousColumnError:
                col = None
            if col is None:
                missing.append(n)
            elif col not in found:
                found.append(col)
        return found, missing
