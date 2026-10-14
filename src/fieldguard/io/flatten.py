"""Flatten API JSON submissions (Kobo v2, ODK OData, SurveyCTO) into export-like frames."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def _is_geojson_point(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("type") == "Point" and isinstance(value.get("coordinates"), list)


def flatten_record(record: Mapping[str, Any], sep: str = "/") -> dict[str, Any]:
    """Flatten nested dicts with ``sep``; GeoJSON points become ``name-Latitude`` etc. (ODK CSV style).

    Repeat groups (lists of dicts) are kept as JSON strings so the main table stays one row per submission.
    """
    out: dict[str, Any] = {}

    def walk(obj: Mapping[str, Any], prefix: str) -> None:
        for key, value in obj.items():
            name = f"{prefix}{sep}{key}" if prefix else str(key)
            if _is_geojson_point(value):
                coords = value["coordinates"] + [None, None, None]
                out[f"{name}-Longitude"] = coords[0]
                out[f"{name}-Latitude"] = coords[1]
                out[f"{name}-Altitude"] = coords[2]
                props = value.get("properties") or {}
                out[f"{name}-Accuracy"] = props.get("accuracy")
            elif isinstance(value, Mapping):
                walk(value, name)
            elif isinstance(value, list):
                out[name] = json.dumps(value, ensure_ascii=False, default=str) if value else None
            else:
                out[name] = value

    walk(record, "")
    return out


def records_to_frame(records: Iterable[Mapping[str, Any]], sep: str = "/") -> pd.DataFrame:
    rows = [flatten_record(r, sep=sep) for r in records]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Text frame, consistent with file exports.
    return df.astype("str").mask(df.isna())
