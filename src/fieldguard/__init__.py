"""FieldGuard: survey data-quality checks and fabrication detection for KoboToolbox, ODK and SurveyCTO.

Flags are leads for supervisory review, not proof of misconduct.
"""

from __future__ import annotations

__version__ = "0.1.0"

from fieldguard.config import FieldGuardConfig, config_from_dict, load_config  # noqa: E402
from fieldguard.models import ISSUE_COLUMNS, Severity  # noqa: E402

__all__ = [
    "ISSUE_COLUMNS",
    "FieldGuardConfig",
    "Severity",
    "__version__",
    "config_from_dict",
    "load_config",
    "run_checks",
]


def __getattr__(name: str):  # lazy import keeps `import fieldguard` light
    if name == "run_checks":
        from fieldguard.pipeline import run_checks

        return run_checks
    raise AttributeError(name)
