"""Sensus agents — concrete actuators behind the orchestrator's tool calls.

Submodules (``browser``, ``shortcuts``, ``desktop``, ``shell``, ``coding``) are
imported directly by the callers that need them, e.g.::

    from sensus.agents import browser

This package intentionally does no lazy import or ``__getattr__`` magic — that
pattern previously caused infinite recursion when a submodule attribute was
looked up before the import system had registered it on the package.
"""
