"""Optional persistence: PostgreSQL sink (schema ``fieldguard``) and incremental state."""

from fieldguard.sink.postgres import SinkStatus, check_reachable, database_url, redact_url, write_run
from fieldguard.sink.state import (
    load_seen_file,
    load_seen_postgres,
    save_seen_file,
    save_seen_postgres,
    state_file,
)

__all__ = [
    "SinkStatus",
    "check_reachable",
    "database_url",
    "load_seen_file",
    "load_seen_postgres",
    "redact_url",
    "save_seen_file",
    "save_seen_postgres",
    "state_file",
    "write_run",
]
