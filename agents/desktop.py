"""
Native desktop control: window focus (wmctrl, xdotool), list open windows.

``wmctrl -a`` only matches a substring of the window title; we parse ``wmctrl -l``
and activate by window id (``wmctrl -ia``) using the best title match. When
``wmctrl -l`` is empty (typical on **Wayland**), we use **GNOME Shell** D-Bus
``org.gnome.Shell.Eval`` (via ``busctl`` or ``gdbus``) to list titles and focus
windows. Set ``SENSUS_GNOME_SHELL_EVAL=0`` to disable. Further fallbacks:
``xdotool`` (including WM_CLASS for browsers), then the distro executable on
PATH (e.g. ``firefox``), and only then ``gtk-launch`` — the latter often opens a
new URL/tab for Firefox.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import Optional

from sensus.agents.linux_exec import is_linux, run, which

logger = logging.getLogger(__name__)

_RE_FOCUS = re.compile(
    r"(?:^|\b)(?:focus|switch to|bring up|show)\s+(?:the\s+|my\s+)?(.+)$",
    re.I,
)

_RE_LIST_WINDOWS = re.compile(
    r"(?:what(?:'s|\s+is)?|which|list|show)\s+(?:(?:open|active|are)\s+)?"
    r"(?:the\s+|my\s+)?(?:windows?|apps?|applications?|programs?)\b|"
    r"\bwhat(?:'s|\s+is)\s+open\b|"
    r"\bwhat\s+do\s+i\s+have\s+open\b|"
    r"\bwhat\s+windows?\s+are\s+open\b",
    re.I,
)


def _strip_leading_filler(text: str) -> str:
    return re.sub(
        r"(?i)^(let'?s|can you|could you|please|i want to|i need to)\s+",
        "",
        text.strip(),
    ).strip()


def _normalize_focus_target(raw: str) -> str:
    t = raw.strip().rstrip(".!?")
    t = re.sub(
        r"\s+(app|application|window|browser|program)\s*$",
        "",
        t,
        flags=re.I,
    ).strip()
    return t


def _wmctrl_list_windows() -> list[tuple[str, str]]:
    """Return [(hex_window_id, title), ...] from ``wmctrl -l``."""
    w = which("wmctrl")
    if not w:
        return []
    r = run([w, "-l"], timeout=5)
    if r.returncode != 0:
        logger.debug("[DESKTOP] wmctrl -l failed: %s", r.stderr)
        return []
    out: list[tuple[str, str]] = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("0x"):
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        wid, title = parts[0], parts[3]
        out.append((wid, title))
    return out


def _unified_window_rows() -> list[tuple[str, str]]:
    """
    Prefer wmctrl (X11). If empty, use GNOME Shell Eval on Wayland — same
    ``(id, title)`` shape with synthetic ``gnome:N`` ids.
    """
    w = _wmctrl_list_windows()
    if w:
        return w
    try:
        from sensus.agents import gnome_shell

        titles = gnome_shell.list_window_titles()
        if titles:
            return [(f"gnome:{i}", t) for i, t in enumerate(titles)]
    except Exception:
        logger.debug("[DESKTOP] gnome_shell list failed", exc_info=True)
    return []


def _score_title_match(query: str, title: str) -> float:
    q = re.sub(r"\s+", " ", query.lower()).strip()
    t = title.lower()
    if not q:
        return 0.0
    score = 0.0
    if q in t:
        score += 25.0
    q_tokens = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 1]
    for w in q_tokens:
        if w in t:
            score += 6.0
    # Prefer strong match on first token (e.g. "firefox" in "Mozilla Firefox — …")
    if q_tokens and q_tokens[0] in t:
        score += 8.0
    return score


def _best_window_match(query: str) -> Optional[tuple[str, str]]:
    windows = _unified_window_rows()
    if not windows:
        return None
    best: Optional[tuple[float, str, str]] = None
    for wid, title in windows:
        s = _score_title_match(query, title)
        if s > 0 and (best is None or s > best[0]):
            best = (s, wid, title)
    if best and best[0] >= 6.0:
        return best[1], best[2]
    return None


def _wmctrl_activate_id(window_id: str) -> bool:
    w = which("wmctrl")
    if not w:
        return False
    r = run([w, "-ia", window_id], timeout=5)
    return r.returncode == 0


def _wmctrl_activate_substr(substr: str) -> bool:
    w = which("wmctrl")
    if not w:
        return False
    r = run([w, "-a", substr[:128]], timeout=5)
    return r.returncode == 0


def _xdotool_activate_name(name: str) -> bool:
    """Raise a window whose title contains ``name`` (visible windows only)."""
    x = which("xdotool")
    if not x:
        return False
    r = run(
        [x, "search", "--onlyvisible", "--limit", "20", "--name", name[:128]],
        timeout=5,
    )
    if r.returncode != 0 or not (r.stdout or "").strip():
        return False
    for line in reversed((r.stdout or "").splitlines()):
        wid = line.strip()
        if not wid.isdigit():
            continue
        r2 = run([x, "windowactivate", wid], timeout=5)
        if r2.returncode == 0:
            return True
    return False


def _xdotool_activate_class(wm_class: str) -> bool:
    """Focus a window by WM_CLASS (second component is often e.g. Navigator for Firefox)."""
    x = which("xdotool")
    if not x:
        return False
    r = run(
        [x, "search", "--onlyvisible", "--limit", "20", "--class", wm_class[:128]],
        timeout=5,
    )
    if r.returncode != 0 or not (r.stdout or "").strip():
        return False
    for line in reversed((r.stdout or "").splitlines()):
        wid = line.strip()
        if not wid.isdigit():
            continue
        r2 = run([x, "windowactivate", wid], timeout=5)
        if r2.returncode == 0:
            return True
    return False


def _popen_which(exe: str) -> bool:
    """Start ``exe`` from PATH detached — uses the distro wrapper (Firefox attaches to running instance)."""
    path = shutil.which(exe)
    if not path:
        return False
    try:
        subprocess.Popen(
            [path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
        )
    except OSError:
        return False
    return True


def _try_browser_style_raise(entry: object) -> Optional[str]:
    """
    Focus Chromium-family / Firefox via xdotool WM_CLASS or ``firefox``/``chrome``
    on PATH — avoids ``gtk-launch`` opening a new tab when possible.
    """
    did = str(getattr(entry, "desktop_id", "")).lower()
    nl = str(getattr(entry, "name", "")).lower()

    classes: tuple[str, ...] = ()
    exes: tuple[str, ...] = ()

    if "firefox" in did or "firefox" in nl:
        classes = ("Navigator", "firefox", "Firefox")
        exes = ("firefox",)
    elif "google-chrome" in did or "chrome-google" in did or (
        "chrome" in nl and "chromium" not in nl
    ):
        classes = ("Google-chrome", "google-chrome", "Chrome", "chrome")
        exes = ("google-chrome-stable", "google-chrome", "chrome")
    elif "chromium" in did or "chromium" in nl:
        classes = ("Chromium", "chromium")
        exes = ("chromium", "chromium-browser")
    else:
        return None

    for c in classes:
        if _xdotool_activate_class(c):
            return f"Focused {getattr(entry, 'name', 'app')}."
    for e in exes:
        if _popen_which(e):
            return f"Raised {getattr(entry, 'name', 'app')}."
    return None


def _try_gtk_raise_installed_app(query: str) -> Optional[str]:
    """
    Raise a resolved .desktop app: browsers get xdotool/PATH first; others use gtk-launch.
    """
    try:
        from sensus.agents.shortcuts import _gtk_launch, resolve_installed_app
    except Exception:
        logger.debug("[DESKTOP] shortcuts import failed for gtk raise", exc_info=True)
        return None
    entry = resolve_installed_app(query)
    if not entry:
        return None
    alt = _try_browser_style_raise(entry)
    if alt:
        return alt
    if _gtk_launch(entry.desktop_id):
        return f"Raised {entry.name}."
    return None


def _focus_window(query: str) -> tuple[bool, str]:
    """
    Try to focus a window. Returns (ok, message_fragment for logging).
    """
    q = _normalize_focus_target(query)
    if not q:
        return False, ""

    if _wmctrl_activate_substr(q):
        return True, q

    hit = _best_window_match(q)
    if hit:
        wid, title = hit
        if wid.startswith("gnome:"):
            try:
                from sensus.agents import gnome_shell

                if gnome_shell.activate_title_substring(title):
                    return True, title
            except Exception:
                logger.debug("[DESKTOP] gnome activate by title failed", exc_info=True)
        elif _wmctrl_activate_id(wid):
            return True, title

    try:
        from sensus.agents import gnome_shell

        if gnome_shell.activate_title_substring(q):
            return True, q
    except Exception:
        logger.debug("[DESKTOP] gnome activate substring failed", exc_info=True)

    if _xdotool_activate_name(q):
        return True, q

    # Retry xdotool with first significant token (e.g. "Google" from "Google Chrome")
    tokens = [w for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) > 2]
    for tok in tokens[:3]:
        if tok.lower() != q.lower() and _xdotool_activate_name(tok):
            return True, tok

    return False, q


def _guess_running_gui_apps() -> list[str]:
    """When wmctrl is empty (Wayland), report common apps if their main process exists."""
    checks = (
        ("Firefox", "firefox"),
        ("Chrome", "chrome"),
        ("Spotify", "spotify"),
        ("VS Code", "code"),
        ("Terminal", "gnome-terminal"),
        ("Files", "nautilus"),
    )
    found: list[str] = []
    for label, exe in checks:
        r = run(["pgrep", "-x", exe], timeout=2)
        if r.returncode == 0:
            found.append(label)
    return found


def _list_windows_summary() -> Optional[str]:
    rows = _unified_window_rows()
    if not rows:
        if not which("wmctrl"):
            return (
                "wmctrl is not installed. On GNOME Wayland, install busctl (systemd) or gdbus "
                "(libglib2.0-bin) so I can list windows via GNOME Shell, or: sudo apt install wmctrl "
                "and use an Xorg session."
            )
        guess = _guess_running_gui_apps()
        if guess:
            return (
                "I couldn't list window titles (wmctrl is empty and GNOME Shell Eval did not return any). "
                f"These processes look related to GUI apps: {', '.join(guess)}. "
                "Say switch to Firefox to raise one. On GNOME, ensure busctl or gdbus is installed; "
                "or try Ubuntu on Xorg with wmctrl."
            )
        return (
            "No windows found: wmctrl list is empty and GNOME Shell did not report titles "
            "(non-GNOME Wayland, or Eval unavailable). Say switch to and an app name to raise it, "
            "or use Ubuntu on Xorg."
        )
    lines = [title for _, title in rows]
    # Keep TTS-friendly length
    tail = lines[:12]
    more = len(lines) - len(tail)
    msg = "Open windows: " + "; ".join(tail)
    if more > 0:
        msg += f", and {more} more."
    return msg


def _xdotool(args: list[str]) -> bool:
    x = which("xdotool")
    if not x:
        logger.warning("[DESKTOP] xdotool not on PATH")
        return False
    r = run([x, *args], timeout=30)
    return r.returncode == 0


def run_desktop_action(transcript: str, payload: dict) -> Optional[str]:
    """
    Handle tool_call payloads and spoken patterns (focus/switch window, list windows).

    ``details`` may include: window | application | app | target, type_text | text,
    keys | key, click [x, y] in pixel coordinates.
    """
    if not is_linux():
        return None

    text = (transcript or "").strip()
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    action = str(payload.get("action") or "").strip().lower()
    blob = f"{text} {action}"

    if _RE_LIST_WINDOWS.search(_strip_leading_filler(text)) or "list open window" in blob:
        listed = _list_windows_summary()
        return listed

    for key in ("window", "application", "app", "target"):
        window = str(details.get(key) or "").strip()
        if window:
            ok, label = _focus_window(window)
            if ok:
                return f"Focused {label}."
            raised = _try_gtk_raise_installed_app(window)
            if raised:
                return raised

    type_text = str(details.get("type_text") or details.get("text") or "").strip()
    if type_text:
        if _xdotool(["type", "--delay", "15", "--", type_text]):
            return "Typed that."
        return None

    keys = str(details.get("keys") or details.get("key") or "").strip()
    if keys:
        chord = keys if "+" in keys else keys.replace(" ", "+")
        if _xdotool(["key", "--clearmodifiers", chord]):
            return "Sent that key."
        return None

    click = details.get("click")
    if isinstance(click, (list, tuple)) and len(click) == 2:
        try:
            x, y = int(click[0]), int(click[1])
        except (TypeError, ValueError):
            x, y = -1, -1
        if x >= 0 and y >= 0:
            if _xdotool(["mousemove", str(x), str(y), "click", "1"]):
                return "Clicked there."

    m = _RE_FOCUS.search(_strip_leading_filler(text))
    if m:
        target = m.group(1).strip()
        ok, label = _focus_window(target)
        if ok:
            return f"Focused {label}."
        raised = _try_gtk_raise_installed_app(target)
        if raised:
            return raised

    if re.search(
        r"(?i)\b(list|show)\s+(open\s+)?(windows?|apps?)\b",
        action,
    ):
        return _list_windows_summary()

    return None


def main() -> None:
    import sys

    logging.basicConfig(level=logging.INFO)
    line = " ".join(sys.argv[1:]).strip() or "switch to Firefox"
    print(run_desktop_action(line, {"agent": "desktop", "action": "", "details": {}}))


if __name__ == "__main__":
    main()
