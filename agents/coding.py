"""
Voice-driven coding: generate Python with Featherless, save under a workspace dir,
open in VS Code (``code`` / ``codium``), run in a new terminal.

Environment:
  SENSUS_CODE_WORKDIR — output folder (default: ~/sensus-generated)
  FEATHERLESS_CODING_MODEL — optional override (default: same as orchestrator)
  FEATHERLESS_CODING_MAX_TOKENS — default 4096
  SENSUS_CODE_OPEN_EDITOR — 1 (default) or 0 to skip VS Code
  SENSUS_CODE_RUN_TERMINAL — 1 (default) or 0 to skip launching the script
  SENSUS_CODE_AUTO_PIP — 1 (default): ``pip install --user`` for detected imports
    from a small allowlist (pygame, Pillow, numpy, …). Set 0 to disable.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from sensus.agents.linux_exec import is_linux, which

logger = logging.getLogger(__name__)

_SYSTEM = """You write runnable Python 3 for Ubuntu.

Output format:
- Exactly ONE markdown fenced block: ```python ... ``` and nothing outside it (no prose before or after).

Correctness and literals (critical):
- If the user names a specific word, string, name, number, filename, or constant, it MUST appear **verbatim**
  in your code (same spelling and casing). Never substitute a different example (e.g. if they say the secret
  word is "judge", use SECRET_WORD = "judge" — not "apple", not a synonym).
- Put user-specified secrets and tunables in clearly named ALL_CAPS constants at the top of the file so they
  are easy to see and edit.
- Do not invent a "sample" value when the user already gave the real one.

Libraries:
- Prefer the Python standard library when it fits.
- If the user asks for **pygame**, a **GUI**, **graphical window**, **sprites**, or similar: use **pygame**
  (``import pygame``). Build a normal event loop; call ``pygame.quit()`` on exit. Window size reasonable (e.g. 480x640).
- If they ask for tkinter only, use tkinter (stdlib).
- Other third-party packages only when clearly needed; list them in a comment line:
  ``# pip: pygame`` or ``# pip: Pillow`` at the top.

Structure:
- The script must run with: python3 script.py
- Include ``if __name__ == "__main__":`` for the entry point.
- Keep the program self-contained; for CLI games use clear printed prompts when not using pygame.

Implement the full user request — do not leave TODO placeholders for parts they asked for."""


# Top-level import name → pip package name (only these get auto-installed when SENSUS_CODE_AUTO_PIP=1)
_PIP_FOR_IMPORT: dict[str, str] = {
    "pygame": "pygame",
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "requests": "requests",
}


def _load_llm() -> tuple[str, str, str, float]:
    from sensus.daemon.main import load_orchestrator_env

    return load_orchestrator_env()


def _extract_python(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    m = re.search(r"```python\s*([\s\S]*?)\s*```", text, re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return None


def _slug_filename(request: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (request or "script").lower()).strip("_")
    s = s[:48] or "script"
    return f"{s}_{int(time.time())}.py"


def _extract_literal_hints(blob: str) -> list[str]:
    """
    Pull likely verbatim strings from free-form speech (general, not game-specific).
    De-dupe; keep order.
    """
    seen: set[str] = set()
    out: list[str] = []

    def add(s: str) -> None:
        s = s.strip()
        if len(s) < 2 or len(s) > 80:
            return
        low = s.lower()
        if low in seen:
            return
        seen.add(low)
        out.append(s)

    # Double- and single-quoted phrases in the request
    for m in re.finditer(r'"([^"\\]{1,64})"', blob):
        add(m.group(1))
    for m in re.finditer(r"'([^'\\]{1,64})'", blob):
        add(m.group(1))

    low = blob.lower()
    # "word judge" / "word is judge" / "secret word judge"
    for m in re.finditer(
        r"\b(?:secret\s+word|the\s+word|hard-?code(?:d)?\s+word|word)\s+(?:is|to\s+be|=|:)\s*['\"]?([a-z]{3,20})['\"]?",
        low,
        re.I,
    ):
        add(m.group(1))
    for m in re.finditer(
        r"\b(?:hard-?code(?:d)?|fix(?:ed)?)\s+(?:the\s+)?(?:word|answer|string)\s+['\"]?([a-z]{3,20})['\"]?",
        low,
        re.I,
    ):
        add(m.group(1))
    # "hardcode word judge" / "secret word judge" (no "is")
    for m in re.finditer(
        r"\bhard-?code(?:d)?\s+word\s+([a-z]{3,20})\b",
        low,
        re.I,
    ):
        add(m.group(1))
    for m in re.finditer(
        r"\b(?:secret|the)\s+word\s+([a-z]{3,20})\b",
        low,
        re.I,
    ):
        add(m.group(1))

    return out


def _verbatim_constraints_section(transcript: str, hints: str) -> str:
    blob = f"{transcript}\n{hints}".strip()
    literals = _extract_literal_hints(blob)
    lines = [
        "## Verbatim requirements (must appear exactly in code — no substitutions):",
    ]
    if literals:
        for lit in literals[:24]:
            lines.append(f'- Use the literal string/name "{lit}" where the user specified it (exact spelling).')
    else:
        lines.append(
            "- Re-read the user message: every named word, number, or string they gave must appear verbatim "
            "in constants or logic — do not replace with a different example."
        )
    return "\n".join(lines)


def _imports_from_ast(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = (alias.name or "").split(".")[0]
                if base:
                    names.add(base)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                if base:
                    names.add(base)
    return names


def _pip_install_allowlisted(code: str) -> list[str]:
    """Return human-readable list of packages we attempted to install."""
    if os.environ.get("SENSUS_CODE_AUTO_PIP", "1").strip() == "0":
        return []
    py = which("python3") or which("python")
    if not py:
        return []
    imports = _imports_from_ast(code)
    attempted: list[str] = []
    for imp, pkg in _PIP_FOR_IMPORT.items():
        if imp not in imports:
            continue
        try:
            r = subprocess.run(
                [py, "-m", "pip", "install", "--user", "-q", pkg],
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ},
            )
            if r.returncode == 0:
                attempted.append(pkg)
                logger.info("[CODING] pip install ok: %s", pkg)
            else:
                logger.warning("[CODING] pip install %s failed: %s", pkg, r.stderr[:300])
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("[CODING] pip install %s: %s", pkg, e)
    return attempted


def _open_editor(path: Path) -> bool:
    if os.environ.get("SENSUS_CODE_OPEN_EDITOR", "1").strip() == "0":
        return False
    for exe in ("code", "codium", "cursor"):
        c = shutil.which(exe)
        if c:
            try:
                subprocess.Popen(
                    [c, str(path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
                )
                return True
            except OSError as e:
                logger.warning("[CODING] editor %s: %s", exe, e)
    return False


def _run_in_terminal(script: Path) -> bool:
    if os.environ.get("SENSUS_CODE_RUN_TERMINAL", "1").strip() == "0":
        return False
    py = which("python3") or which("python")
    if not py:
        return False
    inner = (
        f"cd {shlex.quote(str(script.parent))} && "
        f"{shlex.quote(py)} {shlex.quote(script.name)}; "
        r'echo ""; read -r -p "Press Enter to close…" _'
    )
    term = shutil.which("gnome-terminal")
    if term:
        try:
            subprocess.Popen(
                [term, "--", "bash", "-lc", inner],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
            )
            return True
        except OSError as e:
            logger.warning("[CODING] gnome-terminal: %s", e)
    xte = which("x-terminal-emulator")
    if xte:
        try:
            subprocess.Popen(
                [xte, "-e", "bash", "-lc", inner],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
            )
            return True
        except OSError as e:
            logger.warning("[CODING] x-terminal-emulator: %s", e)
    return False


def run_coding_task(transcript: str, payload: dict) -> Optional[str]:
    """
    Generate Python from the user request, save, optionally open in editor and run in a terminal.

    ``details`` may include: ``filename`` (optional stem), extra ``hints`` string.
    """
    if not is_linux():
        return "Coding tasks need Linux for now."

    text = (transcript or "").strip()
    if not text:
        return None

    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    hints = str(details.get("hints") or details.get("notes") or "").strip()
    name_hint = str(details.get("filename") or details.get("name") or "").strip()

    work = Path(
        os.environ.get("SENSUS_CODE_WORKDIR", str(Path.home() / "sensus-generated"))
    ).expanduser().resolve()
    try:
        work.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"I couldn't create the workspace folder: {e}."

    try:
        k, b, m, t = _load_llm()
    except Exception as e:
        logger.exception("[CODING] env load failed")
        return f"I can't reach the AI service: {e}."

    model = os.environ.get("FEATHERLESS_CODING_MODEL", "").strip() or m
    max_tokens = int(os.environ.get("FEATHERLESS_CODING_MAX_TOKENS", "4096"))

    constraint_block = _verbatim_constraints_section(text, hints)
    user_msg = (
        f"User request:\n{text}\n\n"
        f"{constraint_block}\n"
    )
    if hints:
        user_msg += f"\nAdditional notes from orchestrator:\n{hints}\n"

    from openai import OpenAI

    client = OpenAI(base_url=b, api_key=k, timeout=max(t, 120.0))

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("[CODING] LLM call failed")
        return f"The code generator failed: {e}."

    code = _extract_python(raw)
    if not code:
        return "The model didn't return a Python file I could parse. Try again with a shorter request."

    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Generated code had a syntax error: {e}. Try asking again."

    fname = _slug_filename(name_hint or text)
    out = (work / fname).resolve()
    try:
        out.relative_to(work)
    except ValueError:
        return "Unsafe path; refusing to write outside the workspace."
    try:
        out.write_text(code, encoding="utf-8")
    except OSError as e:
        return f"I couldn't save the file: {e}."

    pip_done = _pip_install_allowlisted(code)

    opened = _open_editor(out)
    ran = _run_in_terminal(out)

    parts = [f"I saved {out.name} under {work}."]
    if pip_done:
        parts.append(f"Installed: {', '.join(pip_done)}.")
    if opened:
        parts.append("Opened it in your editor.")
    else:
        parts.append(
            "Couldn't find code or codium on PATH — install VS Code's shell command or open the file yourself."
        )
    if ran:
        parts.append("Started it in a new terminal.")
    else:
        parts.append(f"Run it with: python3 {out}")

    return " ".join(parts)


def main() -> None:
    import logging as lg
    import sys

    lg.basicConfig(level=lg.INFO)
    line = " ".join(sys.argv[1:]).strip() or "a tiny wordle clone with secret word judge"
    print(
        run_coding_task(
            line,
            {"agent": "coding", "action": "implement", "details": {}},
        )
    )


if __name__ == "__main__":
    main()
