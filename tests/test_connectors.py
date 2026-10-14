"""Connectors with MOCKED HTTP only (httpx.MockTransport). No network access, no real credentials."""

from __future__ import annotations

import json

import httpx
import pytest

from fieldguard.config import config_from_dict
from fieldguard.connectors.base import (
    ConnectorError,
    MissingCredentialError,
    PageCache,
    env_required,
    scrub,
    write_raw_csv,
)
from fieldguard.connectors.kobo import fetch_kobo
from fieldguard.connectors.kobo import settings_from_env as kobo_env
from fieldguard.connectors.odk import OdkSettings, fetch_odk
from fieldguard.connectors.odk import settings_from_env as odk_env
from fieldguard.connectors.surveycto import SurveyCtoSettings, fetch_surveycto
from fieldguard.connectors.surveycto import settings_from_env as scto_env

FAKE_KOBO_TOKEN = "kobo-test-token-123"  # not a real credential
FAKE_ODK_TOKEN = "odk-test-token-456"
FAKE_PASSWORD = "scto-test-password-789"


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def no_sleep(_seconds: float) -> None:
    return None


# ----------------------------------------------------------------------------- base helpers
def test_env_required_names_the_variable_but_never_prints_a_value(monkeypatch):
    with pytest.raises(MissingCredentialError) as info:
        env_required("SOME_TOKEN", env={})
    assert "SOME_TOKEN" in str(info.value)
    assert env_required("SOME_TOKEN", env={"SOME_TOKEN": "  v  "}) == "v"


def test_scrub_removes_every_secret():
    text = f"failed with {FAKE_KOBO_TOKEN} and {FAKE_PASSWORD}"
    out = scrub(text, [FAKE_KOBO_TOKEN, FAKE_PASSWORD, ""])
    assert FAKE_KOBO_TOKEN not in out and FAKE_PASSWORD not in out


def test_page_cache_round_trip_and_key_sanitising(tmp_path):
    cache = PageCache(tmp_path)
    assert cache.get("a/b:c") is None
    cache.put("a/b:c", {"results": [1]})
    assert cache.get("a/b:c") == {"results": [1]}
    assert not any(p.name.startswith("a/") for p in tmp_path.iterdir())
    assert PageCache(None).get("x") is None
    PageCache(None).put("x", {})  # no-op without a folder


def test_write_raw_csv_uses_group_paths(tmp_path):
    out = write_raw_csv(
        [{"_uuid": "u1", "grp": {"q1": "a"}}, {"_uuid": "u2", "grp": {"q1": "b"}}], tmp_path / "raw.csv"
    )
    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "_uuid,grp/q1"
    assert text[1] == "u1,a"


def test_get_json_refuses_error_message_with_token_in_query(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "no"})

    from fieldguard.connectors.base import get_json

    client = client_for(handler)
    with pytest.raises(ConnectorError) as info:
        get_json(client, "https://example.org/x", params={"token": FAKE_KOBO_TOKEN}, secrets=[FAKE_KOBO_TOKEN])
    assert FAKE_KOBO_TOKEN not in str(info.value)
    assert "HTTP 401" in str(info.value)


# ----------------------------------------------------------------------------- KoboToolbox
def test_kobo_settings_from_env_strips_slash_and_hides_token():
    s = kobo_env(
        env={"KOBO_TOKEN": FAKE_KOBO_TOKEN, "KOBO_ASSET_UID": "aXYZ", "KOBO_SERVER_URL": "https://kf.example.org/"}
    )
    assert s.server_url == "https://kf.example.org"
    assert FAKE_KOBO_TOKEN not in repr(s)


def test_kobo_missing_token_is_a_clear_error():
    with pytest.raises(MissingCredentialError, match="KOBO_TOKEN"):
        kobo_env(env={"KOBO_ASSET_UID": "aXYZ"})


def test_kobo_pagination_uses_token_header_and_next_links():
    seen: list[httpx.Request] = []
    base = "https://kf.example.org/api/v2/assets/aXYZ/data/"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params.get("start") in (None, "0"):
            return httpx.Response(
                200,
                json={
                    "count": 3,
                    "next": f"{base}?format=json&limit=2&start=2",
                    "previous": None,
                    "results": [{"_uuid": "a"}, {"_uuid": "b"}],
                },
            )
        return httpx.Response(200, json={"count": 3, "next": None, "previous": None, "results": [{"_uuid": "c"}]})

    settings = kobo_env(
        env={"KOBO_TOKEN": FAKE_KOBO_TOKEN, "KOBO_ASSET_UID": "aXYZ", "KOBO_SERVER_URL": "https://kf.example.org"}
    )
    records = fetch_kobo(settings, page_size=2, client=client_for(handler), sleep=no_sleep)
    assert [r["_uuid"] for r in records] == ["a", "b", "c"]
    assert all(r.headers["Authorization"] == f"Token {FAKE_KOBO_TOKEN}" for r in seen)
    assert seen[0].url.params["limit"] == "2"


def test_kobo_page_size_is_capped_at_documented_maximum():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"next": None, "results": [{"_uuid": "z"}]})

    fetch_kobo(OneOff(), page_size=5000, client=client_for(handler), sleep=no_sleep)
    assert seen[0].url.params["limit"] == "1000"


class OneOff:
    server_url = "https://kf.example.org"
    asset_uid = "aXYZ"
    token = FAKE_KOBO_TOKEN
    secrets = (FAKE_KOBO_TOKEN,)


def test_kobo_retries_server_errors_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"next": None, "results": [{"_uuid": "ok"}]})

    records = fetch_kobo(OneOff(), client=client_for(handler), sleep=sleeps.append)
    assert records == [{"_uuid": "ok"}]
    assert sleeps == [1.0]


def test_kobo_unauthorised_error_does_not_echo_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad token"})

    with pytest.raises(ConnectorError) as info:
        fetch_kobo(OneOff(), client=client_for(handler), sleep=no_sleep)
    assert FAKE_KOBO_TOKEN not in str(info.value)
    assert "HTTP 401" in str(info.value)


def test_kobo_refuses_next_link_to_another_host():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"next": "https://evil.example.net/steal", "results": [{"_uuid": "a"}]})

    with pytest.raises(ConnectorError, match="another host"):
        fetch_kobo(OneOff(), client=client_for(handler), sleep=no_sleep)


def test_kobo_resumes_from_page_cache(tmp_path):
    cache = PageCache(tmp_path)
    cache.put("kobo-aXYZ-p00001", {"next": None, "results": [{"_uuid": "cached"}]})

    def handler(request: httpx.Request) -> httpx.Response:  # must not be called
        raise AssertionError("network used although the page is cached")

    records = fetch_kobo(OneOff(), client=client_for(handler), cache=cache, sleep=no_sleep)
    assert records == [{"_uuid": "cached"}]


def test_kobo_unexpected_payload_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"detail": "not a list"})

    with pytest.raises(ConnectorError, match="results"):
        fetch_kobo(OneOff(), client=client_for(handler), sleep=no_sleep)


# ----------------------------------------------------------------------------- ODK Central
def test_odk_settings_need_a_credential_and_prefer_bearer():
    with pytest.raises(ConnectorError, match="ODK_CENTRAL_TOKEN"):
        odk_env(
            env={
                "ODK_CENTRAL_URL": "https://central.example.org",
                "ODK_CENTRAL_PROJECT_ID": "1",
                "ODK_CENTRAL_FORM_ID": "f",
            }
        )
    s = odk_env(
        env={
            "ODK_CENTRAL_URL": "https://central.example.org/",
            "ODK_CENTRAL_PROJECT_ID": "1",
            "ODK_CENTRAL_FORM_ID": "f",
            "ODK_CENTRAL_TOKEN": FAKE_ODK_TOKEN,
        }
    )
    headers, auth = s.auth_parts()
    assert headers["Authorization"] == f"Bearer {FAKE_ODK_TOKEN}" and auth is None
    assert FAKE_ODK_TOKEN not in repr(s)


def test_odk_basic_auth_when_no_token():
    s = OdkSettings(
        base_url="https://central.example.org",
        project_id="1",
        form_id="f",
        email="u@example.org",
        password=FAKE_PASSWORD,
    )
    headers, auth = s.auth_parts()
    assert "Authorization" not in headers
    assert isinstance(auth, httpx.BasicAuth)


def test_odk_odata_pagination_and_query_options():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "skiptoken" not in str(request.url):
            return httpx.Response(
                200,
                json={
                    "value": [{"__id": "1", "meta": {"instanceID": "uuid:1"}}],
                    "@odata.nextLink": "https://central.example.org/v1/projects/1/forms/My%20Form.svc/Submissions?$skiptoken=abc",
                },
            )
        return httpx.Response(200, json={"value": [{"__id": "2"}]})

    settings = OdkSettings(
        base_url="https://central.example.org", project_id="1", form_id="My Form", token=FAKE_ODK_TOKEN
    )
    records = fetch_odk(settings, page_size=50, client=client_for(handler), sleep=no_sleep)
    assert [r["__id"] for r in records] == ["1", "2"]
    first = seen[0]
    assert "/v1/projects/1/forms/My%20Form.svc/Submissions" in str(first.url)
    assert first.url.params["$top"] == "50" and first.url.params["$expand"] == "*"
    assert first.headers["Authorization"] == f"Bearer {FAKE_ODK_TOKEN}"


def test_odk_refuses_next_link_to_another_host():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"value": [{"__id": "1"}], "@odata.nextLink": "https://other.example.net/next"})

    settings = OdkSettings(base_url="https://central.example.org", project_id="1", form_id="f", token=FAKE_ODK_TOKEN)
    with pytest.raises(ConnectorError, match="another host"):
        fetch_odk(settings, client=client_for(handler), sleep=no_sleep)


def test_odk_rejects_payload_without_value_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "nope"})

    settings = OdkSettings(base_url="https://central.example.org", project_id="1", form_id="f", token=FAKE_ODK_TOKEN)
    with pytest.raises(ConnectorError, match="value"):
        fetch_odk(settings, client=client_for(handler), sleep=no_sleep)


# ----------------------------------------------------------------------------- SurveyCTO
def test_surveycto_server_must_be_a_name_not_a_url():
    with pytest.raises(ConnectorError, match="server name only"):
        scto_env(
            env={
                "SURVEYCTO_SERVER": "https://myorg.surveycto.com",
                "SURVEYCTO_FORM_ID": "f",
                "SURVEYCTO_USERNAME": "u",
                "SURVEYCTO_PASSWORD": FAKE_PASSWORD,
            }
        )
    with pytest.raises(ConnectorError, match="not allowed"):
        scto_env(
            env={
                "SURVEYCTO_SERVER": "my_org!",
                "SURVEYCTO_FORM_ID": "f",
                "SURVEYCTO_USERNAME": "u",
                "SURVEYCTO_PASSWORD": FAKE_PASSWORD,
            }
        )
    s = scto_env(
        env={
            "SURVEYCTO_SERVER": "MyOrg",
            "SURVEYCTO_FORM_ID": "hh_survey",
            "SURVEYCTO_USERNAME": "u",
            "SURVEYCTO_PASSWORD": FAKE_PASSWORD,
        }
    )
    assert s.server == "myorg" and FAKE_PASSWORD not in repr(s)


def test_surveycto_wide_json_url_date_zero_and_basic_auth():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"KEY": "uuid:1", "hhid": "H1"}])

    s = SurveyCtoSettings(server="myorg", form_id="hh_survey", username="api_user", password=FAKE_PASSWORD)
    rows = fetch_surveycto(s, client=client_for(handler), sleep=no_sleep)
    assert rows == [{"KEY": "uuid:1", "hhid": "H1"}]
    req = seen[0]
    assert req.url.path == "/api/v2/forms/data/wide/json/hh_survey"
    assert req.url.params["date"] == "0"
    assert req.headers["Authorization"].startswith("Basic ")


def test_surveycto_incremental_date_parameter_and_bad_payload():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"not": "a list"})

    s = SurveyCtoSettings(server="myorg", form_id="hh_survey", username="u", password=FAKE_PASSWORD)
    with pytest.raises(ConnectorError, match="JSON array"):
        fetch_surveycto(s, since=1767225600000, client=client_for(handler), sleep=no_sleep)
    assert seen[0].url.params["date"] == "1767225600000"


def test_surveycto_error_message_has_no_password():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    s = SurveyCtoSettings(server="myorg", form_id="hh_survey", username="u", password=FAKE_PASSWORD)
    with pytest.raises(ConnectorError) as info:
        fetch_surveycto(s, client=client_for(handler), sleep=no_sleep)
    assert FAKE_PASSWORD not in str(info.value)


def test_config_source_block_validates():
    cfg = config_from_dict({"data": {"source": {"type": "kobo", "asset_uid": "aXYZ", "page_size": 500}}})
    assert cfg.data.source.type == "kobo" and cfg.data.source.page_size == 500


def test_json_fixture_is_serialisable():
    assert json.dumps({"ok": True})
