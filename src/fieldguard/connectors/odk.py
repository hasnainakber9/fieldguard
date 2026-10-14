"""ODK Central OData connector (form submissions with nested groups expanded).

Endpoint: ``GET {url}/v1/projects/{projectId}/forms/{xmlFormId}.svc/Submissions?$top=N&$expand=*`` and the
``@odata.nextLink`` cursor for further pages. Central documents Bearer sessions (``POST /v1/sessions`` returns a
token) and HTTP Basic credentials; Basic is documented as discouraged and only over HTTPS. This connector uses
``Authorization: Bearer`` when ``ODK_CENTRAL_TOKEN`` is set and HTTP Basic otherwise.

Environment: ODK_CENTRAL_URL, ODK_CENTRAL_PROJECT_ID, ODK_CENTRAL_FORM_ID, ODK_CENTRAL_TOKEN or
ODK_CENTRAL_EMAIL + ODK_CENTRAL_PASSWORD (a read-only web user is recommended).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from fieldguard.config import SourceConfig
from fieldguard.connectors.base import ConnectorError, PageCache, env_optional, env_required, get_json, same_host

ODATA_PAGE = 1000


@dataclass(frozen=True)
class OdkSettings:
    base_url: str
    project_id: str
    form_id: str
    token: str | None = field(default=None, repr=False)
    email: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.token, self.password) if s)

    def auth_parts(self) -> tuple[dict[str, str], httpx.Auth | None]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}, None
        return {"Accept": "application/json"}, httpx.BasicAuth(self.email or "", self.password or "")


def settings_from_env(source: SourceConfig | None = None, env: Mapping[str, str] | None = None) -> OdkSettings:
    base = (source.server_url if source and source.server_url else None) or env_required("ODK_CENTRAL_URL", env)
    project = (source.project_id if source and source.project_id is not None else None) or env_required(
        "ODK_CENTRAL_PROJECT_ID", env
    )
    form = (source.form_id if source and source.form_id else None) or env_required("ODK_CENTRAL_FORM_ID", env)
    token = env_optional("ODK_CENTRAL_TOKEN", None, env)
    email = env_optional("ODK_CENTRAL_EMAIL", None, env)
    password = env_optional("ODK_CENTRAL_PASSWORD", None, env)
    if not token and not (email and password):
        raise ConnectorError("set ODK_CENTRAL_TOKEN, or ODK_CENTRAL_EMAIL and ODK_CENTRAL_PASSWORD")
    return OdkSettings(
        base_url=str(base).rstrip("/"),
        project_id=str(project),
        form_id=str(form),
        token=token,
        email=email,
        password=password,
    )


def fetch_odk(
    settings: OdkSettings,
    *,
    page_size: int = ODATA_PAGE,
    expand: bool = True,
    client: httpx.Client | None = None,
    cache: PageCache | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Download all submissions of the form, following ``@odata.nextLink`` (resumable through ``cache``)."""
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=False)
    headers, auth = settings.auth_parts()
    form_seg = quote(settings.form_id, safe="")
    url: str | None = (
        f"{settings.base_url}/v1/projects/{quote(settings.project_id, safe='')}/forms/{form_seg}.svc/Submissions"
    )
    params: dict[str, Any] | None = {"$top": min(int(page_size), ODATA_PAGE)}
    if expand:
        params["$expand"] = "*"
    records: list[dict[str, Any]] = []
    page = 0
    try:
        while url:
            page += 1
            key = f"odk-{settings.project_id}-{settings.form_id}-p{page:05d}"
            data = cache.get(key) if cache else None
            if data is None:
                data = get_json(
                    client, url, params=params, headers=headers, auth=auth, secrets=settings.secrets, sleep=sleep
                )
                if cache:
                    cache.put(key, data)
            if not isinstance(data, dict) or "value" not in data:
                raise ConnectorError("unexpected ODK Central response: no 'value' key")
            records.extend(data["value"])
            nxt = data.get("@odata.nextLink")
            if nxt:
                nxt = urljoin(settings.base_url + "/", nxt)
                if not same_host(nxt, settings.base_url):
                    raise ConnectorError(
                        "ODK Central nextLink points to another host; refusing to send credentials there"
                    )
            url = nxt
            params = None
            if not data["value"]:
                break
    finally:
        if owns_client:
            client.close()
    return records


__all__ = ["ODATA_PAGE", "OdkSettings", "fetch_odk", "settings_from_env"]
