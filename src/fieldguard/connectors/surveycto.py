"""SurveyCTO server API connector (wide JSON export of one form).

Endpoint: ``GET https://{server}.surveycto.com/api/v2/forms/data/wide/json/{formId}?date=D`` with HTTP Basic
authentication. ``date=0`` returns everything; a later value returns submissions since that point (a timestamp in
seconds or milliseconds, or the CompletionDate string). The response is a JSON array. The user needs a role that
may download data and the server setting "Allow server API access". The date boundary can be inclusive or exclusive
depending on the release, so overlapping rows are removed by submission key downstream.

Environment: SURVEYCTO_SERVER (name only, e.g. "myorg"), SURVEYCTO_FORM_ID, SURVEYCTO_USERNAME, SURVEYCTO_PASSWORD.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from fieldguard.config import SourceConfig
from fieldguard.connectors.base import ConnectorError, PageCache, env_required, get_json

_SERVER_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,61}$")


@dataclass(frozen=True)
class SurveyCtoSettings:
    server: str
    form_id: str
    username: str = field(repr=False)
    password: str = field(repr=False)

    @property
    def base_url(self) -> str:
        return f"https://{self.server}.surveycto.com"

    @property
    def secrets(self) -> tuple[str, ...]:
        return (self.username, self.password)


def settings_from_env(source: SourceConfig | None = None, env: Mapping[str, str] | None = None) -> SurveyCtoSettings:
    server = (source.server_url if source and source.server_url else None) or env_required("SURVEYCTO_SERVER", env)
    server = str(server).strip().lower()
    if server.startswith("http") or "." in server or "/" in server:
        raise ConnectorError("SURVEYCTO_SERVER must be the server name only (for example 'myorg'), not a URL")
    if not _SERVER_NAME.match(server):
        raise ConnectorError("SURVEYCTO_SERVER contains characters that are not allowed in a server name")
    form = (source.form_id if source and source.form_id else None) or env_required("SURVEYCTO_FORM_ID", env)
    return SurveyCtoSettings(
        server=server,
        form_id=str(form),
        username=env_required("SURVEYCTO_USERNAME", env),
        password=env_required("SURVEYCTO_PASSWORD", env),
    )


def fetch_surveycto(
    settings: SurveyCtoSettings,
    *,
    since: str | int | None = None,
    client: httpx.Client | None = None,
    cache: PageCache | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Download the form's submissions (all of them, or those after ``since``). One request, cached as one page."""
    owns_client = client is None
    client = client or httpx.Client(timeout=120.0, follow_redirects=False)
    url = f"{settings.base_url}/api/v2/forms/data/wide/json/{quote(settings.form_id, safe='')}"
    date_value = str(since) if since not in (None, "") else "0"
    key = f"surveycto-{settings.server}-{settings.form_id}-date{date_value}"
    try:
        data = cache.get(key) if cache else None
        if data is None:
            data = get_json(
                client,
                url,
                params={"date": date_value},
                headers={"Accept": "application/json"},
                auth=httpx.BasicAuth(settings.username, settings.password),
                secrets=settings.secrets,
                sleep=sleep,
            )
            if cache:
                cache.put(key, data)
    finally:
        if owns_client:
            client.close()
    if not isinstance(data, list):
        raise ConnectorError("unexpected SurveyCTO response: expected a JSON array of submissions")
    return data


__all__ = ["SurveyCtoSettings", "fetch_surveycto", "settings_from_env"]
