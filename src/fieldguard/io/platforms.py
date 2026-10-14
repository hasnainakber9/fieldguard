"""Export conventions of KoboToolbox, ODK Central and SurveyCTO."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformConventions:
    name: str
    submission_id: tuple[str, ...]
    submission_time: tuple[str, ...]
    start: tuple[str, ...]
    end: tuple[str, ...]
    survey_date: tuple[str, ...]
    deviceid: tuple[str, ...]
    duration: tuple[str, ...] = ()
    naive_submission_tz: str | None = "UTC"
    metadata: frozenset[str] = field(default_factory=frozenset)


_COMMON_META = {
    "start",
    "end",
    "today",
    "deviceid",
    "DeviceID",
    "subscriberid",
    "simserial",
    "simid",
    "phonenumber",
    "devicephonenum",
    "username",
    "email",
    "audit",
    "audit_URL",
    "instanceID",
    "instanceName",
    "meta/instanceID",
    "meta/instanceName",
    "meta/deprecatedID",
    "meta-instanceID",
    "meta-instanceName",
    "formdef_version",
    "__version__",
}

KOBO = PlatformConventions(
    name="kobo",
    submission_id=("_uuid", "meta/instanceID", "instanceID"),
    submission_time=("_submission_time",),
    start=("start",),
    end=("end",),
    survey_date=("today",),
    deviceid=("deviceid",),
    naive_submission_tz="UTC",
    metadata=frozenset(
        _COMMON_META
        | {
            "_id",
            "_uuid",
            "_submission_time",
            "_validation_status",
            "_notes",
            "_status",
            "_submitted_by",
            "_tags",
            "_index",
            "_version_",
            "_xform_id_string",
            "_attachments",
            "_geolocation",
            "_parent_table_name",
            "_parent_index",
            "__version__",
        }
    ),
)

ODK = PlatformConventions(
    name="odk",
    submission_id=("KEY", "instanceID", "meta-instanceID", "meta/instanceID", "__id"),
    submission_time=("SubmissionDate", "__system/submissionDate"),
    start=("start",),
    end=("end",),
    survey_date=("today",),
    deviceid=("deviceid", "DeviceID", "__system/deviceId"),
    naive_submission_tz="UTC",
    metadata=frozenset(
        _COMMON_META
        | {
            "KEY",
            "SubmissionDate",
            "SubmitterID",
            "SubmitterName",
            "AttachmentsPresent",
            "AttachmentsExpected",
            "Status",
            "ReviewState",
            "Edits",
            "FormVersion",
            "__id",
        }
    ),
)

SURVEYCTO = PlatformConventions(
    name="surveycto",
    submission_id=("KEY", "instanceID"),
    submission_time=("SubmissionDate", "CompletionDate"),
    start=("starttime", "start"),
    end=("endtime", "end"),
    survey_date=("today",),
    deviceid=("deviceid",),
    duration=("duration",),
    naive_submission_tz=None,  # SurveyCTO exports server-local naive times -> project timezone
    metadata=frozenset(
        _COMMON_META
        | {
            "KEY",
            "SubmissionDate",
            "CompletionDate",
            "starttime",
            "endtime",
            "duration",
            "caseid",
            "review_status",
            "review_comments",
            "review_corrections",
            "review_quality",
            "formdef_version",
        }
    ),
)

GENERIC = PlatformConventions(
    name="generic",
    submission_id=("_uuid", "KEY", "instanceID", "submission_id", "id"),
    submission_time=("_submission_time", "SubmissionDate", "submission_time"),
    start=("start", "starttime", "start_time"),
    end=("end", "endtime", "end_time"),
    survey_date=("today", "date", "survey_date"),
    deviceid=("deviceid", "DeviceID"),
    duration=("duration",),
    metadata=frozenset(_COMMON_META),
)

PLATFORMS: dict[str, PlatformConventions] = {p.name: p for p in (KOBO, ODK, SURVEYCTO, GENERIC)}


def detect_platform(columns: Iterable[str]) -> str:
    cols = set(columns)
    if {"_uuid", "_submission_time"} & cols or any(c.startswith("_") and c.endswith("_latitude") for c in cols):
        return "kobo"
    if "__id" in cols or "__system/submissionDate" in cols:
        return "odk"
    if "KEY" in cols and ({"formdef_version", "starttime", "endtime"} & cols):
        return "surveycto"
    if "KEY" in cols or "SubmissionDate" in cols or "meta-instanceID" in cols:
        return "odk"
    return "generic"


_KOBO_GPS = re.compile(r"^_(?P<base>.+)_(?P<part>latitude|longitude|altitude|precision)$")
_DASH_GPS = re.compile(r"^(?P<base>.+)-(?P<part>Latitude|Longitude|Altitude|Accuracy)$")


def detect_gps_columns(columns: Iterable[str]) -> dict[str, str | None] | None:
    """Find split geopoint columns (Kobo ``_gps_latitude`` or ODK/SurveyCTO ``gps-Latitude``)."""
    groups: dict[str, dict[str, str]] = {}
    for col in columns:
        for pattern in (_KOBO_GPS, _DASH_GPS):
            m = pattern.match(col)
            if m:
                part = m.group("part").lower()
                part = "accuracy" if part == "precision" else part
                groups.setdefault(m.group("base"), {})[part] = col
    complete = {b: g for b, g in groups.items() if "latitude" in g and "longitude" in g}
    if len(complete) != 1:
        return None
    g = next(iter(complete.values()))
    return {"lat": g["latitude"], "lon": g["longitude"], "accuracy": g.get("accuracy")}


def first_present(candidates: Iterable[str], columns: set[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None
