"""Shared connector helpers: credentials from the environment, retried GETs, resumable page cache, secret scrubbing.

Rules enforced here:

* credentials are read only from environment variables (never from YAML, never printed, never stored);
* errors are scrubbed of every secret that was in use before they are raised or logged;
* pagination links that point to another host are refused, so credentials are never sent elsewhere;
* each page is cached as JSON, so an interrupted download resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd

from fieldguard.io.flatten import records_to_frame

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=20.0)


class ConnectorError(RuntimeError):
    """A download failed. The message never contains credentials."""


class MissingCredentialError(ConnectorError):
    """A required environment variable is not set (the message names the variable, not its value)."""


def env_required(name: str, env: Mapping[str, str] | None = None) -> str:
    value = (env if env is not None else os.environ).get(name, "").strip()
    if not value:
        raise MissingCredentialError(f"environment variable {name} is not set (see .env.example)")
    return value


def env_optional(name: str, default: str | None = None, env: Mapping[str, str] | None = None) -> str | None:
    value = (env if env is not None else os.environ).get(name, "").strip()
    return value or default


def scrub(text: str, secrets: Iterable[str]) -> str:
    out = text
    for s in secrets:
        if s:
            out = out.replace(s, "***")
    return out


def safe_url(url: httpx.URL | str) -> str:
    """URL without its query string (query values can carry tokens)."""
    text = str(url)
    return text.split("?", 1)[0]


def same_host(url: str, base: str) -> bool:
    a, b = urlparse(url), urlparse(base)
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=False, transport=transport)


class PageCache:
    """One JSON file per page; a missing file means the page still has to be downloaded."""

    def __init__(self, root: Path | None):
        self.root = root

    def _path(self, key: str) -> Path | None:
        if self.root is None:
            return None
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
        return self.root / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, payload: Any) -> None:
        path = self._path(key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


def get_json(
    client: httpx.Client,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    auth: httpx.Auth | None = None,
    secrets: Iterable[str] = (),
    retries: int = 4,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """GET a JSON document with retries on rate limits and server errors. Raises ConnectorError on failure."""
    secret_list = [s for s in secrets if s]
    last: str = ""
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, params=dict(params) if params else None, headers=dict(headers or {}), auth=auth)
        except httpx.HTTPError as exc:
            last = f"network error: {type(exc).__name__}"
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise ConnectorError(f"response from {safe_url(resp.request.url)} is not JSON") from exc
            last = f"HTTP {resp.status_code} from {safe_url(resp.request.url)}"
            if resp.status_code not in RETRY_STATUS:
                raise ConnectorError(scrub(last, secret_list))
        if attempt < retries:
            sleep(backoff * (2**attempt))
    raise ConnectorError(scrub(f"{last} after {retries + 1} attempts", secret_list))


def write_raw_csv(records: list[dict[str, Any]], path: Path) -> Path:
    """Flatten downloaded records to the export layout (group/q1 paths) and write a UTF-8 CSV."""
    frame: pd.DataFrame = records_to_frame(records, sep="/")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    return path


__all__ = [
    "ConnectorError",
    "MissingCredentialError",
    "PageCache",
    "env_optional",
    "env_required",
    "get_json",
    "make_client",
    "safe_url",
    "same_host",
    "scrub",
    "write_raw_csv",
]
