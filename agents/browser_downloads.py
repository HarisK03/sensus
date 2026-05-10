"""Shared helpers for saving files triggered by browser automation.

Playwright emits a :class:`playwright.async_api.Download` after a navigation or
click that returns ``Content-Disposition: attachment`` (or similar). Callers wrap
those actions in ``page.expect_download()`` and pass the resulting object here.

Other agents (e.g. a future shell downloader) can reuse
``resolve_browser_download_dir`` so everything lands in the same folder the user
configures once via the environment.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional


def default_download_timeout_ms() -> int:
    """Upper bound for how long to wait for a download after click/navigate."""
    raw = os.environ.get("SENSUS_BROWSER_DOWNLOAD_TIMEOUT_MS", "120000").strip()
    try:
        v = int(raw)
    except ValueError:
        v = 120_000
    return max(3_000, min(v, 600_000))


def resolve_browser_download_dir() -> Path:
    """Directory for ``Download.save_as``; created if missing.

    ``SENSUS_BROWSER_DOWNLOAD_DIR`` overrides the default. Relative paths are
    resolved against the current working directory (shell session), then the
    path is expanded for ``~``.
    """
    custom = os.environ.get("SENSUS_BROWSER_DOWNLOAD_DIR", "").strip()
    if custom:
        base = Path(custom).expanduser()
        if not base.is_absolute():
            base = Path.cwd() / base
    else:
        base = Path.home() / "Downloads"
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def pick_non_colliding_path(directory: Path, filename: str) -> Path:
    """If ``directory/filename`` exists, append `` (1)``, `` (2)``, … before ext."""
    directory = directory.resolve()
    name = filename.strip() or "download"
    dest = directory / name
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for i in range(1, 1000):
        candidate = directory / f"{stem} ({i}){suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}-{int(time.time() * 1000)}{suffix}"


async def persist_playwright_download(
    download: Any,
    directory: Optional[Path] = None,
) -> Path:
    """Write Playwright's temporary download into ``directory``; return final path."""
    dest_dir = directory or resolve_browser_download_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    suggested = (download.suggested_filename or "").strip() or "download"
    dest = pick_non_colliding_path(dest_dir, suggested)
    await download.save_as(str(dest))
    return dest.resolve()
