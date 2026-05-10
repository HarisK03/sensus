"""
Run shell commands from orchestrator ``tool_call`` payloads.

Uses ``shlex.split`` for string commands (no shell interpolation). When the model
omits ``details.command``, common voice phrases (list/install from Downloads) are
inferred for demos.

Optional: ``SENSUS_SHELL_ALLOW_PIPES=1`` to allow ``|``, ``;``, ``$(``, newlines.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from sensus.agents.linux_exec import is_linux, run

_DANGEROUS = re.compile(r"([;&|`]|\$\(|\n|\r)")

# GNU ``ls -l`` / ``ls -la``: mode, links, user, group, size, Mon DD time_or_year, name...
_LS_LONG_LINE = re.compile(
    r"^[\-bcdl][\-rwx]{9}\s+\d+\s+\S+\s+\S+\s+\d+\s+"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+"
    r"(?:\d{4}|\d{1,2}:\d{2})\s+(.+)$"
)

_STOP = frozenset(
    """
    the and for with that this from can please your you are was were have has
    install file files download downloads into about want need will would
    could should there here into app application package deb
    """.split(),
)


def _normalize_argv_paths(argv: list[str]) -> list[str]:
    """Expand env vars (``$HOME``, ``$USER``) and ``~`` — subprocess has no shell interpolation."""
    out: list[str] = []
    for a in argv:
        a = os.path.expandvars(a)
        a = os.path.expanduser(a)
        out.append(a)
    return out


def _truncate_for_voice(name: str, *, max_chars: int = 36) -> str:
    """Shorten long filenames so TTS does not read 80-character package names aloud."""
    n = (name or "").strip()
    if not n or len(n) <= max_chars:
        return n
    ext = Path(n).suffix
    base = n[: len(n) - len(ext)] if ext else n
    # Single ellipsis before extension avoids ``...`` + ``.deb`` reading as many dots.
    ell = "\u2026"
    if ext:
        reserve = len(ext) + len(ell)
        keep = max(6, max_chars - reserve)
        chunk = base[:keep].rstrip(" \t(")
        return f"{chunk}{ell}{ext}"
    keep = max(6, max_chars - len(ell))
    out = base[:keep].rstrip(" \t(") + ell
    return out[:max_chars]


def _parse_ls_long_filename(line: str) -> Optional[str]:
    m = _LS_LONG_LINE.match(line.rstrip())
    if not m:
        return None
    return m.group(1).strip()


def _voice_listing_from_stdout(stdout: str, argv: list[str]) -> Optional[str]:
    """
    Turn raw ``ls`` output into a short spoken summary (names only, truncated).

    Returns ``None`` when stdout does not look like a directory listing.
    """
    raw = (stdout or "").strip()
    if not raw:
        return None

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    names: list[str] = []
    long_hits = 0

    for line in lines:
        if line.startswith("total "):
            continue
        fn = _parse_ls_long_filename(line)
        if fn is not None:
            long_hits += 1
            if fn not in (".", ".."):
                names.append(fn)
        elif not line.startswith("-"):
            # Might still be ls long format with weird locale — skip parsing errors.
            pass

    # Plain ``ls`` (no ``-l``): one name per line.
    if not names and argv and argv[0] == "ls":
        candidate_lines = [ln for ln in lines if not ln.startswith("total ")]
        if candidate_lines and all(_parse_ls_long_filename(ln) is None for ln in candidate_lines):
            names = [ln.strip() for ln in candidate_lines if ln.strip() not in (".", "..")]

    if not names:
        if long_hits > 0:
            # Listing parsed but only . / ..
            return "That folder only contains the usual entries — nothing else to list."
        return None

    short = [_truncate_for_voice(n) for n in names]
    max_names = int(os.environ.get("SENSUS_SHELL_VOICE_LIST_MAX", "12"))
    extra = max(0, len(short) - max_names)
    shown = short[:max_names]

    if len(shown) == 1:
        body = f"one item: {shown[0]}."
    elif len(shown) == 2:
        body = f"two items: {shown[0]}, and {shown[1]}."
    else:
        joined = ", ".join(shown[:-1])
        body = f"{len(names)} items: {joined}, and {shown[-1]}."
    if extra:
        body += f" Plus {extra} more — say if you want the full list."
    return body


def _execute_argv(argv: list[str], *, timeout: float) -> str:
    try:
        r = run(argv, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"The command timed out after {int(timeout)} seconds."

    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    code = r.returncode
    parts: list[str] = []
    if out:
        voice_list = None
        if code == 0 and argv and argv[0] == "ls":
            voice_list = _voice_listing_from_stdout(out, argv)
        if voice_list:
            parts.append(voice_list)
        else:
            parts.append(out[:2000])
    if err and code != 0:
        parts.append(f"(stderr) {err[:1500]}")
    summary = "\n".join(parts) if parts else "(no output)"
    status = "succeeded" if code == 0 else f"exited with code {code}"
    return f"The command {status}. {summary}"[:4000]


def _pick_deb_for_voice(transcript: str, action: str) -> Optional[Path]:
    """Choose a .deb in Downloads — match spoken keywords to filename, else newest."""
    d = Path.home() / "Downloads"
    if not d.is_dir():
        return None
    debs = sorted(d.glob("*.deb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not debs:
        return None
    blob = f"{transcript} {action}".lower()
    words = [
        w
        for w in re.findall(r"[a-z0-9]{3,}", blob)
        if w not in _STOP
    ]
    for p in debs:
        stem = p.stem.lower()
        if any(w in stem for w in words):
            return p
    return debs[0]


def _infer_argv(transcript: str, payload: dict) -> Optional[list[str]]:
    """Build argv for obvious demo phrases when the LLM omitted ``command``."""
    t = (transcript or "").strip()
    action = str(payload.get("action") or "").strip()
    blob = f"{t} {action}".lower()
    home = Path.home()
    down = home / "Downloads"

    wants_list = bool(
        re.search(
            r"\b(list|show|what(?:'s|\s+is)?|see|display)\b",
            blob,
            re.I,
        )
    ) and ("download" in blob or str(down).lower() in blob.replace("~", str(home)))
    if wants_list and "install" not in blob:
        if not down.is_dir():
            return None
        return ["ls", "-la", str(down)]

    wants_install = bool(re.search(r"\binstall\b", blob, re.I)) and (
        "download" in blob or "deb" in blob
    )
    if wants_install:
        deb = _pick_deb_for_voice(t, action)
        if not deb:
            return None
        path = str(deb.resolve())
        if shutil.which("pkexec"):
            return ["pkexec", "apt", "install", "-y", path]
        if shutil.which("sudo"):
            return ["sudo", "apt", "install", "-y", path]
        return ["apt", "install", "-y", path]

    return None


def run_shell_action(transcript: str, payload: dict) -> Optional[str]:
    """
    Run ``details.command`` (or ``details.cmd``), or infer argv from voice + action.

    Without a runnable command, returns ``None``.
    """
    if not is_linux():
        return "Shell integration needs Linux."

    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    cmd = str(details.get("command") or details.get("cmd") or "").strip()

    argv: Optional[list[str]] = None
    if cmd:
        if len(cmd) > 800:
            return "That command is too long to run automatically."
        allow = os.environ.get("SENSUS_SHELL_ALLOW_PIPES", "").strip() == "1"
        if not allow and _DANGEROUS.search(cmd):
            return (
                "That command uses shell operators I block by default. "
                "Set SENSUS_SHELL_ALLOW_PIPES=1 if you really need that."
            )
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            return f"I couldn't parse that command: {e}."
    else:
        argv = _infer_argv(transcript, payload)

    if not argv:
        return None

    argv = _normalize_argv_paths(argv)

    base_timeout = float(os.environ.get("SENSUS_SHELL_TIMEOUT_SEC", "30"))
    if any(x in argv for x in ("apt", "dpkg", "pkexec")):
        base_timeout = max(base_timeout, 300.0)

    return _execute_argv(argv, timeout=base_timeout)


def main() -> None:
    import logging
    import sys

    logging.basicConfig(level=logging.INFO)
    line = " ".join(sys.argv[1:]).strip() or "list my downloads"
    print(
        run_shell_action(
            line,
            {"agent": "shell", "action": "list downloads", "details": {}},
        )
    )


if __name__ == "__main__":
    main()
