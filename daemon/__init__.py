"""Orchestrator daemon — routing, orchestration, entry.

Heavy imports are lazy so ``python -m sensus.daemon.main`` does not preload ``main``
via this package (avoids runpy double-import warnings).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Orchestrator",
    "OrchestratorTurnResult",
    "RouteResult",
    "classify_intent",
    "load_featherless_env",
    "load_orchestrator_env",
    "parse_assistant_output",
]

_LAZY_MAIN = frozenset({
    "Orchestrator",
    "OrchestratorTurnResult",
    "load_orchestrator_env",
    "parse_assistant_output",
})
_LAZY_ROUTER = frozenset({"RouteResult", "classify_intent", "load_featherless_env"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAIN:
        from sensus.daemon import main as _m

        return getattr(_m, name)
    if name in _LAZY_ROUTER:
        from sensus.daemon import router as _r

        return getattr(_r, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
