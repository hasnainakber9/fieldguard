"""KoboToolbox API v2 connector (submissions of one asset).

Endpoint: ``GET {server}/api/v2/assets/{uid}/data/?format=json&limit=N&start=S`` with the header
``Authorization: Token <KOBO_TOKEN>``. Pages follow the ``next`` link. The per-request limit is kept at or below
1000, the maximum that KoboToolbox documents in its community announcements for the data endpoint.
Environment: KOBO_SERVER_URL (default https://kf.kobotoolbox.org), KOBO_TOKEN, KOBO_ASSET_UID.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from fieldguard.config import SourceConfig
from fieldguard.connectors.base import ConnectorError, PageCache, env_optional, env_required, get_json, same_host

DEFAULT_SERVER = "https://kf.kobotoolbox.org"
MAX_PAGE = 1000


@dataclass(frozen=True)
class KoboSettings:
    server_url: str
    asset_uid: str
    token: str = field(repr=False)

    @property
    def secrets(self) -> tuple[str, ...]:
        return (self.token,)


def settings_from_env(source: SourceConfig | None = None, env: Mapping[str, str] | None = None) -> KoboSettings:
    server = (source.server_url if source and source.server_url else None) or env_optional(
        "KOBO_SERVER_URL", DEFAULT_SERVER, env
    )
    uid = (source.asset_uid if source and source.asset_uid else None) or env_required("KOBO_ASSET_UID", env)
    token = env_required("KOBO_TOKEN", env)
    return KoboSettings(server_url=str(server).rstrip("/"), asset_uid=str(uid), token=token)


def fetch_kobo(
    settings: KoboSettings,
    *,
    page_size: int = MAX_PAGE,
    client: httpx.Client | None = None,
    cache: PageCache | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Download every submission of the asset. Pages already in ``cache`` are reused (resumable)."""
    page_size = max(1, min(int(page_size), MAX_PAGE))
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=False)
    headers = {"Authorization": f"Token {settings.token}", "Accept": "application/json"}
    base = f"{settings.server_url}/api/v2/assets/{settings.asset_uid}/data/"
    url: str | None = base
    params: dict[str, Any] | None = {"format": "json", "limit": page_size, "start": 0}
    records: list[dict[str, Any]] = []
    page = 0
    try:
        while url:
            page += 1
            key = f"kobo-{settings.asset_uid}-p{page:05d}"
            data = cache.get(key) if cache else None
            if data is None:
                data = get_json(client, url, params=params, headers=headers, secrets=settings.secrets, sleep=sleep)
                if cache:
                    cache.put(key, data)
            if not isinstance(data, dict) or "results" not in data:
                raise ConnectorError("unexpected KoboToolbox response: no 'results' key")
            records.extend(data["results"])
            nxt = data.get("next")
            if nxt and not same_host(nxt, settings.server_url):
                raise ConnectorError("KoboToolbox 'next' link points to another host; refusing to send the token there")
            url = nxt
            params = None
            if not data["results"]:
                break
    finally:
        if owns_client:
            client.close()
    return records


__all__ = ["DEFAULT_SERVER", "MAX_PAGE", "KoboSettings", "fetch_kobo", "settings_from_env"]
