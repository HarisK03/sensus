"""Small shared helpers for Linux desktop agents (subprocess + PATH)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def run(
    argv: list[str],
    *,
    timeout: float = 30,
    check: bool = False,
) -> subprocess.CompletedProcess:
    env = {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
        env=env,
    )
