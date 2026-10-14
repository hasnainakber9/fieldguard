"""PostgreSQL sink and incremental state. The live-database tests are marked ``postgres`` and skip unless
FIELDGUARD_DATABASE_URL points at a reachable server. Nothing here prints or stores a password."""

from __future__ import annotations

import os

import pytest

from fieldguard.config import config_from_dict
from fieldguard.sink import (
    check_reachable,
    database_url,
    load_seen_file,
    redact_url,
    save_seen_file,
    state_file,
)
from tests.conftest import make_cfg

SECRET = "s3cr3t-pass-value"
CLOSED_URL = f"postgresql+psycopg://lab:{SECRET}@127.0.0.1:1/lab"


def test_redact_url_never_shows_the_password():
    shown = redact_url(CLOSED_URL)
    assert SECRET not in shown
    assert "lab" in shown and "127.0.0.1" in shown


def test_unparseable_url_is_masked_entirely():
    assert SECRET not in redact_url(f"not a url {SECRET}")


def test_unreachable_database_is_skipped_without_leaking_details():
    status = check_reachable(CLOSED_URL, timeout=1.0)
    assert status.available is False
    assert SECRET not in status.message
    assert "127.0.0.1" not in status.message
    assert status.message.startswith("PostgreSQL sink skipped")


def test_database_url_comes_from_environment(monkeypatch):
    monkeypatch.delenv("FIELDGUARD_DATABASE_URL", raising=False)
    assert database_url() is None
    monkeypatch.setenv("FIELDGUARD_DATABASE_URL", "  ")
    assert database_url() is None
    monkeypatch.setenv("FIELDGUARD_DATABASE_URL", CLOSED_URL)
    assert database_url() == CLOSED_URL


def test_file_state_round_trip_and_sanitised_name(tmp_path):
    cfg = config_from_dict(
        {"project": {"name": "Pilot / Survey #1"}, "incremental": {"state_dir": "state"}}, base_dir=tmp_path
    )
    assert load_seen_file(cfg) == set()
    path = save_seen_file(cfg, ["b", "a", "a"], run_id="fg-1")
    assert path == state_file(cfg)
    assert "/" not in path.name and "#" not in path.name
    assert load_seen_file(cfg) == {"a", "b"}


def test_state_file_lives_under_state_dir(tmp_path):
    cfg = make_cfg()
    cfg._base_dir = tmp_path  # private attribute, set for the test only
    assert state_file(cfg).parent == tmp_path / ".fieldguard_state"


@pytest.mark.postgres
@pytest.mark.skipif(not os.environ.get("FIELDGUARD_DATABASE_URL"), reason="FIELDGUARD_DATABASE_URL not set")
def test_live_postgres_round_trip(small_synthetic):  # pragma: no cover - needs a database
    from fieldguard.config import config_from_dict as cfd
    from fieldguard.pipeline import run_checks
    from fieldguard.sink import write_run
    from fieldguard.synthetic import synthetic_config

    url = os.environ["FIELDGUARD_DATABASE_URL"]
    status = check_reachable(url)
    if not status.available:
        pytest.skip(status.message)
    survey, spec = small_synthetic
    cfg = cfd(synthetic_config(spec=spec))
    result = run_checks(survey.data, cfg, run_id="pytest-live")
    counts = write_run(result, cfg, url)
    assert counts["runs"] == 1
