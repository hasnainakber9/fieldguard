"""Built-in checks; importing this package registers them."""

from fieldguard.checks.builtin import consistency, enumerator, integrity, paradata, regularity, statistical

__all__ = ["consistency", "enumerator", "integrity", "paradata", "regularity", "statistical"]
