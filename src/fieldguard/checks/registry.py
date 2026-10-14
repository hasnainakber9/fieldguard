"""Pluggable check registry.

A check is a function ``(ctx: CheckContext, params: <ParamsModel>) -> pd.DataFrame`` returning a tidy
issues table (see :mod:`fieldguard.models`). Register it with :func:`register_check`; third-party packages
can expose checks through the ``fieldguard.checks`` entry-point group.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import pandas as pd

    from fieldguard.checks.context import CheckContext

log = logging.getLogger(__name__)


class CheckParams(BaseModel):
    """Base class for check parameters (all checks accept ``enabled``)."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = True


CheckFunc = Callable[["CheckContext", Any], "pd.DataFrame"]


@dataclass(frozen=True)
class CheckSpec:
    id: str
    func: CheckFunc
    params_model: type[CheckParams]
    title: str
    description: str
    category: str
    references: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, CheckSpec] = {}
_LOADED = False


def register_check(
    check_id: str,
    *,
    title: str,
    description: str,
    category: str,
    params: type[CheckParams] = CheckParams,
    references: tuple[str, ...] = (),
) -> Callable[[CheckFunc], CheckFunc]:
    def decorator(func: CheckFunc) -> CheckFunc:
        if check_id in _REGISTRY and _REGISTRY[check_id].func is not func:
            raise ValueError(f"check id {check_id!r} is already registered")
        _REGISTRY[check_id] = CheckSpec(check_id, func, params, title, description, category, references)
        return func

    return decorator


def _load_builtin() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import fieldguard.checks.builtin  # noqa: F401  (registers built-in checks)

    try:
        for ep in entry_points(group="fieldguard.checks"):
            try:
                ep.load()
            except Exception as exc:  # pragma: no cover - plugin errors are environment specific
                log.warning("could not load FieldGuard plugin %s: %s", ep.name, exc)
    except Exception as exc:  # pragma: no cover
        log.debug("entry point discovery failed: %s", exc)


def get_registry() -> dict[str, CheckSpec]:
    _load_builtin()
    return dict(_REGISTRY)


def get_check(check_id: str) -> CheckSpec:
    reg = get_registry()
    if check_id not in reg:
        raise KeyError(f"unknown check {check_id!r}; available: {sorted(reg)}")
    return reg[check_id]
