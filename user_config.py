"""
Per-user configuration (~/.config/sensus/config.json).

Written by the first-run wizard; merged into ``os.environ`` for keys that are not already set
(so ``.env`` / the shell can still override). Call :func:`invalidate_credential_caches` after
updating credentials so ``lru_cache`` loaders pick up new values.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

_CONFIG_NAME = "config.json"
_DIR_NAME = "sensus"

# JSON keys (snake_case) → environment names
_ENV_MAP: tuple[tuple[str, str], ...] = (
    ("featherless_ai_api_key", "FEATHERLESS_AI_API_KEY"),
    ("featherless_base_url", "FEATHERLESS_BASE_URL"),
    ("ibm_stt_api_key", "IBM_STT_API_KEY"),
    ("ibm_stt_url", "IBM_STT_URL"),
    ("ibm_tts_api_key", "IBM_TTS_API_KEY"),
    ("ibm_tts_url", "IBM_TTS_URL"),
)


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if base:
        return Path(base) / _DIR_NAME
    return Path.home() / ".config" / _DIR_NAME


def config_path() -> Path:
    return config_dir() / _CONFIG_NAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_user_config_into_environ() -> None:
    """Apply ``~/.config/sensus/config.json`` to the environment (do not replace existing vars)."""
    path = config_path()
    if not path.is_file():
        return
    data = _read_json(path)
    for jk, ek in _ENV_MAP:
        v = data.get(jk)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if not (os.environ.get(ek) or "").strip():
            os.environ[ek] = s


def read_user_config() -> dict[str, Any]:
    """Return merged defaults + file contents (file wins)."""
    path = config_path()
    return dict(_read_json(path)) if path.is_file() else {}


def save_user_config(
    data: Mapping[str, Any],
    *,
    merge: bool = True,
) -> Path:
    """Write ``config.json``, optionally merging with existing file."""
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / _CONFIG_NAME
    prev: dict[str, Any] = {}
    if merge and path.is_file():
        prev = _read_json(path)
    merged: dict[str, Any] = {**prev, **dict(data)}
    path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def invalidate_credential_caches() -> None:
    """Clear ``lru_cache`` loaders after env-changing updates."""
    try:
        from sensus.daemon.main import load_orchestrator_env
        from sensus.daemon.router import load_featherless_env
        from sensus.voice.tts import load_tts_env

        load_orchestrator_env.cache_clear()
        load_featherless_env.cache_clear()
        load_tts_env.cache_clear()
    except Exception:
        pass


def credentials_present_in_environ() -> bool:
    """Return True when Featherless + IBM STT + IBM TTS env vars are all non-empty."""
    load_user_config_into_environ()
    f = (os.environ.get("FEATHERLESS_AI_API_KEY") or "").strip()
    stt_k = (os.environ.get("IBM_STT_API_KEY") or "").strip()
    stt_u = (os.environ.get("IBM_STT_URL") or "").strip()
    tts_k = (os.environ.get("IBM_TTS_API_KEY") or "").strip()
    tts_u = (os.environ.get("IBM_TTS_URL") or "").strip()
    return bool(f and stt_k and stt_u and tts_k and tts_u)


def is_setup_complete() -> bool:
    """
    Setup is complete when ``setup_complete`` is true in config, or when all required
    credentials are already in the environment (e.g. from ``.env``).
    """
    path = config_path()
    if path.is_file():
        data = _read_json(path)
        if data.get("setup_complete") is True:
            return True
    return credentials_present_in_environ()


def mark_setup_complete() -> None:
    save_user_config({"setup_complete": True}, merge=True)
    invalidate_credential_caches()
