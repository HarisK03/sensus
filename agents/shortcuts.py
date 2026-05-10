"""
Ubuntu desktop shortcuts — fast local actions (no LLM) where possible, plus
vision-guided xdotool steps for arbitrary apps.

- Volume: pactl (level, mute, step up/down)
- Brightness: brightnessctl (level, step up/down)
- Wi‑Fi / wireless: nmcli radio wifi
- Bluetooth: bluetoothctl power (rfkill fallback)
- Airplane / flight mode: nmcli networking + bluetooth
- Media / Spotify transport: playerctl or MPRIS dbus-send
- Play a search in Spotify: MPRIS OpenUri spotify:search:… (no Web API)
- Open installed apps: resolve *.desktop entries, gtk-launch
- Fallback: screenshot + vision model returns one JSON step (click %, keys, type, done)

Environment:
  SENSUS_SHORTCUTS_VISION_STEPS — max vision loop iterations (default 4)
  SENSUS_SPOTIFY_SEARCH_KEYS — xdotool key tokens after OpenUri (default: Return)
  SENSUS_SPOTIFY_SEARCH_WAIT — seconds to wait for search UI (default: 2.2)
  SENSUS_VOLUME_STEP_PCT — pactl relative step for up/down (default 10)
  SENSUS_BRIGHTNESS_STEP_PCT — brightnessctl relative step (default 10)
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sensus.agents.linux_exec import is_linux as _is_linux
from sensus.agents.linux_exec import run as _run
from sensus.agents.linux_exec import which as _which

logger = logging.getLogger(__name__)

DESKTOP_DIRS = (
    Path("/usr/share/applications"),
    Path.home() / ".local/share/applications",
    Path("/var/lib/snapd/desktop/applications"),
)


@dataclass(frozen=True)
class DesktopEntry:
    desktop_id: str
    name: str
    exec_hint: str
    keywords: str


def _strip_json_fence(text: str) -> str:
    t = (text or "").strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _iter_desktop_entries() -> list[DesktopEntry]:
    out: list[DesktopEntry] = []
    seen: set[str] = set()
    for base in DESKTOP_DIRS:
        if not base.is_dir():
            continue
        for path in base.glob("*.desktop"):
            if not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if re.search(r"(?m)^Hidden\s*=\s*true\s*$", raw):
                continue
            if re.search(r"(?m)^NoDisplay\s*=\s*true\s*$", raw):
                continue
            name = ""
            exec_hint = ""
            keywords = ""
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("Name=") and not name and "[" not in line[:5]:
                    name = line[5:].strip()
                elif line.startswith("Name[") and not name:
                    name = line.split("=", 1)[-1].strip()
                elif line.startswith("Exec=") and not exec_hint:
                    exec_hint = line[5:].strip()
                elif line.startswith("Keywords=") and not keywords:
                    keywords = line[9:].strip()
            if not name or not exec_hint:
                continue
            did = path.name
            if did in seen:
                continue
            seen.add(did)
            out.append(
                DesktopEntry(
                    desktop_id=did.replace(".desktop", ""),
                    name=name,
                    exec_hint=exec_hint,
                    keywords=keywords,
                )
            )
    return out


_desktop_cache: Optional[list[DesktopEntry]] = None


def _all_desktops() -> list[DesktopEntry]:
    global _desktop_cache
    if _desktop_cache is None:
        _desktop_cache = _iter_desktop_entries()
    return _desktop_cache


def _score_desktop(query: str, e: DesktopEntry) -> float:
    q = re.sub(r"\s+", " ", query.lower()).strip()
    if not q:
        return 0.0
    blob = f"{e.name} {e.keywords} {e.exec_hint}".lower()
    score = 0.0
    if q in blob:
        score += 10.0
    for word in re.findall(r"\w+", q):
        if len(word) < 2:
            continue
        if word in blob:
            score += 2.0
    if q.split()[0] in e.desktop_id.lower():
        score += 3.0
    return score


def resolve_installed_app(user_phrase: str) -> Optional[DesktopEntry]:
    """Best-effort match against installed .desktop files."""
    phrase = (user_phrase or "").strip()
    if not phrase:
        return None
    # common speech → package names
    aliases = {
        "file explorer": "files",
        "files app": "files",
        "file manager": "files",
        "explorer": "nautilus",
    }
    phrase_l = phrase.lower()
    for a, b in aliases.items():
        if a in phrase_l:
            phrase = b
            break
    best: Optional[tuple[float, DesktopEntry]] = None
    for e in _all_desktops():
        s = _score_desktop(phrase, e)
        if s > 0 and (best is None or s > best[0]):
            best = (s, e)
    if best and best[0] >= 2.0:
        return best[1]
    return None


def _gtk_launch(desktop_id: str) -> bool:
    """Start or raise an app via its .desktop id. Never blocks on Snap wrappers."""
    if not _which("gtk-launch"):
        logger.warning("[SHORTCUTS] gtk-launch not found; install libgtk-3-bin")
        return False
    try:
        subprocess.Popen(
            ["gtk-launch", desktop_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
        )
    except OSError as e:
        logger.warning("[SHORTCUTS] gtk-launch %s failed: %s", desktop_id, e)
        return False
    return True


def _wmctrl_activate_substr(title_substr: str) -> bool:
    if not _which("wmctrl"):
        return False
    r = _run(["wmctrl", "-a", title_substr], timeout=5)
    return r.returncode == 0


def _xdotool(args: list[str]) -> bool:
    x = _which("xdotool")
    if not x:
        logger.warning("[SHORTCUTS] xdotool not on PATH")
        return False
    r = _run([x, *args], timeout=30)
    if r.returncode != 0:
        logger.debug("[SHORTCUTS] xdotool %s stderr=%s", args, r.stderr)
        return False
    return True


def _display_geometry() -> tuple[int, int]:
    if not _which("xdotool"):
        return (1920, 1080)
    r = _run(["xdotool", "getdisplaygeometry"], timeout=5)
    if r.returncode != 0:
        return (1920, 1080)
    parts = (r.stdout or "").strip().split()
    if len(parts) >= 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return (1920, 1080)


def _click_percent(x_pct: float, y_pct: float) -> None:
    w, h = _display_geometry()
    x = max(0, min(w - 1, int(w * float(x_pct) / 100.0)))
    y = max(0, min(h - 1, int(h * float(y_pct) / 100.0)))
    _xdotool(["mousemove", str(x), str(y), "click", "1"])


def _playerctl_players() -> list[str]:
    if not _which("playerctl"):
        return []
    r = _run(["playerctl", "-l"], timeout=5)
    if r.returncode != 0:
        return []
    return [p.strip() for p in (r.stdout or "").splitlines() if p.strip()]


def _mpris_open_uri_spotify(uri: str) -> bool:
    """Open a spotify: URI via MPRIS (Spotify must be running or autostart)."""
    dest = "org.mpris.MediaPlayer2.spotify"
    r = _run(
        [
            "dbus-send",
            "--print-reply",
            "--dest",
            dest,
            "/org/mpris/MediaPlayer2",
            "org.mpris.MediaPlayer2.Player.OpenUri",
            f"string:{uri}",
        ],
        timeout=10,
    )
    if r.returncode == 0:
        return True
    logger.debug("[SHORTCUTS] dbus OpenUri failed: %s", r.stderr)
    if _which("playerctl"):
        r2 = _run(["playerctl", "-p", "spotify", "open", uri], timeout=10)
        return r2.returncode == 0
    return False


def _ensure_spotify_running() -> bool:
    players = _playerctl_players()
    if any("spotify" in p.lower() for p in players):
        return True
    entry = resolve_installed_app("spotify")
    launched = _gtk_launch(entry.desktop_id) if entry else _gtk_launch("spotify")
    if not launched:
        return False
    time.sleep(0.6)
    for _ in range(60):
        time.sleep(0.25)
        players = _playerctl_players()
        if any("spotify" in p.lower() for p in players):
            return True
    logger.warning("[SHORTCUTS] Spotify launched but MPRIS not ready")
    return True


def _spotify_search_play(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Say what to search for in Spotify."
    if not _is_linux():
        return "Spotify shortcuts need Linux."
    if not _ensure_spotify_running():
        return "I couldn't start Spotify."
    encoded = urllib.parse.quote(q, safe="")
    uri = f"spotify:search:{encoded}"
    if not _mpris_open_uri_spotify(uri):
        return "Spotify didn't accept that search link."
    time.sleep(float(os.environ.get("SENSUS_SPOTIFY_SEARCH_WAIT", "2.2")))
    _wmctrl_activate_substr("Spotify")
    time.sleep(0.15)
    # After search loads, nudge the top result (layout-dependent).
    extra_keys = os.environ.get("SENSUS_SPOTIFY_SEARCH_KEYS", "Return").strip() or "Return"
    for key_token in extra_keys.split():
        _xdotool(["key", "--clearmodifiers", key_token])
        time.sleep(0.12)
    time.sleep(0.4)
    if _which("playerctl"):
        _run(["playerctl", "-p", "spotify", "play"], timeout=5)
        time.sleep(0.2)
        st = _run(["playerctl", "-p", "spotify", "status"], timeout=5)
        if st.returncode == 0 and "playing" not in (st.stdout or "").lower():
            _run(["playerctl", "-p", "spotify", "play"], timeout=5)
    return f"Searching Spotify for {q}."


def _try_volume(text: str) -> Optional[str]:
    t = text.lower().strip()
    needs_pactl = bool(
        re.search(r"\b(volume|sound|mute|unmute|audio)\b", t)
    )
    if needs_pactl and not _which("pactl"):
        return "pactl isn't available for volume."

    if _which("pactl"):
        if re.search(r"\bunmute\b", t) or re.search(
            r"\bturn\s+(the\s+)?(sound|audio)\s+back\s+on\b", t
        ):
            r = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "0"], timeout=5)
            return "Unmuted." if r.returncode == 0 else "I couldn't unmute."
        if re.search(r"\bmute\b", t) and "minute" not in t:
            r = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1"], timeout=5)
            return "Muted." if r.returncode == 0 else "I couldn't mute."

        step = max(1, min(50, int(os.environ.get("SENSUS_VOLUME_STEP_PCT", "10"))))
        if re.search(r"\b(volume|sound)\s+(up|higher|louder)\b", t) or re.search(
            r"\b(turn|crank)\s+up\s+(the\s+)?(volume|sound)\b", t
        ):
            r = _run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"+{step}%"],
                timeout=5,
            )
            return (
                f"Volume up about {step} percent."
                if r.returncode == 0
                else "I couldn't change the volume."
            )
        if re.search(r"\b(volume|sound)\s+(down|lower|quieter)\b", t) or re.search(
            r"\bturn\s+down\s+(the\s+)?(volume|sound)\b", t
        ):
            r = _run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"-{step}%"],
                timeout=5,
            )
            return (
                f"Volume down about {step} percent."
                if r.returncode == 0
                else "I couldn't change the volume."
            )

    m = re.search(
        r"(?:volume|sound)\s*(?:to|at)?\s*(\d{1,3})\s*(?:%|percent)?",
        text,
        re.I,
    )
    if not m:
        m = re.search(
            r"(?:turn|set)\s*(?:the\s*)?(?:volume|sound)\s*(?:to|at)\s*(\d{1,3})",
            text,
            re.I,
        )
    if not m:
        return None
    pct = int(m.group(1))
    pct = max(0, min(150, pct))
    if not _which("pactl"):
        return "pactl isn't available for volume."
    r = _run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
        timeout=5,
    )
    if r.returncode != 0:
        return "I couldn't change the volume."
    return f"Volume set to {pct} percent."


def _try_brightness(text: str) -> Optional[str]:
    t = text.lower().strip()
    if _which("brightnessctl"):
        step = max(1, min(50, int(os.environ.get("SENSUS_BRIGHTNESS_STEP_PCT", "10"))))
        if re.search(r"\b(brightness|screen)\s+(up|brighter|higher)\b", t) or re.search(
            r"\bbrighten\b", t
        ):
            r = _run(["brightnessctl", "set", f"+{step}%"], timeout=5)
            return (
                f"Brightness up about {step} percent."
                if r.returncode == 0
                else "I couldn't change brightness."
            )
        if re.search(
            r"\b(brightness|screen)\s+(down|dimmer|lower|darker)\b", t
        ) or re.search(r"\bdim\b", t):
            r = _run(["brightnessctl", "set", f"{step}%-"], timeout=5)
            return (
                f"Brightness down about {step} percent."
                if r.returncode == 0
                else "I couldn't change brightness."
            )

    m = re.search(
        r"(?:brightness|screen)\s*(?:to|at)?\s*(\d{1,3})\s*(?:%|percent)?",
        text,
        re.I,
    )
    if not m:
        return None
    pct = max(1, min(100, int(m.group(1))))
    if not _which("brightnessctl"):
        return "brightnessctl isn't installed."
    r = _run(["brightnessctl", "set", f"{pct}%"], timeout=5)
    if r.returncode != 0:
        return "I couldn't change brightness."
    return f"Brightness set to {pct} percent."


def _nmcli_wifi_state() -> Optional[str]:
    r = _run(["nmcli", "-t", "radio"], timeout=5)
    if r.returncode != 0:
        return None
    for line in (r.stdout or "").splitlines():
        lu = line.upper()
        if lu.startswith("WIFI:") and not lu.startswith("WIFI-HW:"):
            return line.split(":", 1)[-1].strip().lower()
    return None


def _try_wifi(text: str) -> Optional[str]:
    t = text.lower().strip()
    if not re.search(r"\b(wifi|wi-?fi|wireless)\b", t):
        return None
    if re.search(r"\bopen\b", t) and re.search(r"\bsettings\b", t):
        return None
    if not _which("nmcli"):
        return "NetworkManager (nmcli) isn't available. Install network-manager or use the Settings app."

    if re.search(r"\b(status|state|on\?|enabled\?)\b", t) or t in (
        "wifi",
        "wi-fi",
        "wireless",
    ):
        st = _nmcli_wifi_state()
        if st:
            return f"Wi-Fi radio is {st}."
        return "I couldn't read Wi-Fi state from NetworkManager."

    want_on = bool(
        re.search(r"\b(enable|on|start)\b", t)
        and not re.search(r"\b(disable|off|stop)\b", t)
    )
    want_off = bool(re.search(r"\b(disable|off|stop)\b", t))
    if re.search(r"\bturn\b", t):
        if re.search(r"\bturn\s+on\b", t) or re.search(
            r"\bturn\s+(the\s+)?(wifi|wi-?fi|wireless)\s+on\b", t
        ):
            want_on = True
        if re.search(r"\bturn\s+off\b", t) or re.search(
            r"\bturn\s+(the\s+)?(wifi|wi-?fi|wireless)\s+off\b", t
        ):
            want_off = True

    if want_on and want_off:
        return None
    if want_on:
        r = _run(["nmcli", "radio", "wifi", "on"], timeout=15)
        if r.returncode != 0:
            return "I couldn't enable Wi-Fi."
        return "Wi-Fi is on."
    if want_off:
        r = _run(["nmcli", "radio", "wifi", "off"], timeout=15)
        if r.returncode != 0:
            return "I couldn't disable Wi-Fi."
        return "Wi-Fi is off."
    return None


def _bluetooth_power(on: bool) -> bool:
    want = "on" if on else "off"
    if _which("bluetoothctl"):
        r = _run(["bluetoothctl", "power", want], timeout=15)
        if r.returncode == 0:
            return True
    rf = _which("rfkill")
    if rf:
        arg = "unblock" if on else "block"
        r2 = _run([rf, arg, "bluetooth"], timeout=10)
        return r2.returncode == 0
    return False


def _try_bluetooth(text: str) -> Optional[str]:
    t = text.lower().strip()
    if not (
        re.search(r"\bbluetooth\b", t)
        or re.search(r"blue\s*tooth", t)
        or re.search(r"\bbt\b", t)
    ):
        return None
    if re.search(r"\bopen\b", t) and re.search(r"\bsettings\b", t):
        return None

    if re.search(r"\b(status|state|on\?|enabled\?)\b", t):
        if _which("bluetoothctl"):
            r = _run(["bluetoothctl", "show"], timeout=10)
            if r.returncode == 0 and r.stdout:
                if "Powered: yes" in r.stdout:
                    return "Bluetooth is on."
                if "Powered: no" in r.stdout:
                    return "Bluetooth is off."
        return "I couldn't read Bluetooth state."

    want_on = bool(
        re.search(r"\b(enable|on|start)\b", t)
        and not re.search(r"\b(disable|off|stop)\b", t)
    )
    want_off = bool(re.search(r"\b(disable|off|stop)\b", t))
    if re.search(r"\bturn\b", t):
        if re.search(r"\bturn\s+on\b", t) or re.search(
            r"\bturn\s+(bluetooth|bt)\s+on\b", t
        ):
            want_on = True
        if re.search(r"\bturn\s+off\b", t) or re.search(
            r"\bturn\s+(bluetooth|bt)\s+off\b", t
        ):
            want_off = True

    if want_on and want_off:
        return None
    if want_on:
        if _bluetooth_power(True):
            return "Bluetooth is on."
        return "I couldn't enable Bluetooth."
    if want_off:
        if _bluetooth_power(False):
            return "Bluetooth is off."
        return "I couldn't disable Bluetooth."
    return None


def _try_airplane_mode(text: str) -> Optional[str]:
    t = text.lower().strip()
    if not re.search(r"\b(airplane|flight\s+mode)\b", t):
        return None

    # "airplane mode on" / "turn on airplane mode" => cut radios
    on_airplane = bool(
        re.search(r"\b(airplane|flight)\s+mode\s+(on|enabled)\b", t)
        or re.search(r"\b(enable|turn on)\s+(airplane|flight)\b", t)
        or re.search(r"\bturn\s+on\s+(airplane|flight)(\s+mode)?\b", t)
    )
    off_airplane = bool(
        re.search(r"\b(airplane|flight)\s+mode\s+(off|disabled)\b", t)
        or re.search(r"\b(disable|turn off)\s+(airplane|flight)\b", t)
        or re.search(r"\bturn\s+off\s+(airplane|flight)(\s+mode)?\b", t)
    )
    if on_airplane and off_airplane:
        return None

    if on_airplane:
        if _which("nmcli"):
            _run(["nmcli", "networking", "off"], timeout=15)
        _bluetooth_power(False)
        return "Airplane mode on — networking and Bluetooth are off."

    if off_airplane:
        if _which("nmcli"):
            r = _run(["nmcli", "networking", "on"], timeout=15)
            if r.returncode == 0:
                _run(["nmcli", "radio", "wifi", "on"], timeout=15)
        _bluetooth_power(True)
        return "Airplane mode off — networking and Bluetooth restored."

    return None


def _try_simple_media(text: str) -> Optional[str]:
    t = text.lower().strip()
    # Avoid stealing "play <song>" — require media phrasing
    if re.search(r"\b(in|on)\s+\w+\s+play\b", t):
        return None
    if re.search(r"\bplay\s+[\w\s]{8,}", t) and "spotify" not in t:
        return None

    if not _which("playerctl"):
        return None

    if re.search(r"\b(pause|hold)\b", t):
        _run(["playerctl", "pause"], timeout=5)
        return "Paused playback."
    if re.search(r"\b(unpause|resume)\b", t) or re.fullmatch(r"play", t.strip()):
        _run(["playerctl", "play"], timeout=5)
        return "Resuming playback."
    if re.search(r"\b(next|skip)\b", t):
        _run(["playerctl", "next"], timeout=5)
        return "Skipped to the next track."
    if re.search(r"\b(previous|back)\b", t):
        _run(["playerctl", "previous"], timeout=5)
        return "Previous track."
    return None


_RE_IN_APP_PLAY = re.compile(
    r"(?:^|\b)(?:in|on)\s+(.+?)\s+play\s+(.+)$",
    re.I,
)


def _try_in_app_play(text: str) -> Optional[str]:
    m = _RE_IN_APP_PLAY.search(text.strip())
    if not m:
        return None
    app_phrase = m.group(1).strip()
    query = m.group(2).strip()
    if not query:
        return None
    if "spotify" in app_phrase.lower():
        return _spotify_search_play(query)
    entry = resolve_installed_app(app_phrase)
    if not entry:
        return None
    if not _gtk_launch(entry.desktop_id):
        return f"I couldn't open {app_phrase}."
    time.sleep(float(os.environ.get("SENSUS_SHORTCUTS_APP_LAUNCH_WAIT", "0.8")))
    _wmctrl_activate_substr(entry.name[:12])
    # Generic: vision loop to focus search and type (works for many apps)
    return _vision_automation_loop(
        goal=f"In {entry.name}, search or navigate so this plays or opens: {query}. "
        f"Use search shortcuts if visible.",
        transcript=text,
    )


_RE_SPOTIFY_PLAY = re.compile(
    r"(?:\bspotify\b.*\bplay\s+(.+)$|\bplay\s+(.+)\s+(?:in|on)\s+spotify\b)",
    re.I,
)


def _try_spotify_phrase(text: str) -> Optional[str]:
    m = _RE_SPOTIFY_PLAY.search(text.strip())
    if not m:
        return None
    q = (m.group(1) or m.group(2) or "").strip()
    if not q:
        return None
    return _spotify_search_play(q)


_RE_OPEN = re.compile(
    r"^(?:open|launch|start)\s+(?:the\s+)?(.+)$",
    re.I,
)

_RE_USER_FOLDER = re.compile(
    r"(?:^|\b)(?:open|show|go to)\s+(?:the\s+|my\s+)?"
    r"(downloads|documents|pictures|videos|music|desktop|home)"
    r"(?!\s+player\b)(?:\s+folder)?\b",
    re.I,
)


def _xdg_open_path(path: Path) -> bool:
    p = path.expanduser().resolve()
    if not p.exists():
        return False
    xo = _which("xdg-open")
    if not xo:
        logger.warning("[SHORTCUTS] xdg-open not found")
        return False
    r = _run([xo, str(p)], timeout=15)
    return r.returncode == 0


def _try_open_user_folder(text: str) -> Optional[str]:
    """Open a well-known home subfolder in the default file manager (xdg-open)."""
    normalized = re.sub(
        r"(?i)^(let'?s|can you|could you|please|i want to|i need to)\s+",
        "",
        text.strip(),
    ).strip()
    m = _RE_USER_FOLDER.search(normalized)
    if not m:
        return None
    key = m.group(1).lower()
    home = Path.home()
    mapping = {
        "downloads": home / "Downloads",
        "documents": home / "Documents",
        "pictures": home / "Pictures",
        "videos": home / "Videos",
        "music": home / "Music",
        "desktop": home / "Desktop",
        "home": home,
    }
    target = mapping.get(key)
    if target is None:
        return None
    if not target.is_dir():
        return f"I couldn't find the folder {target.name}."
    if not _xdg_open_path(target):
        return f"I couldn't open {target.name}."
    return f"Opening {target.name}."


def _try_open_app(text: str) -> Optional[str]:
    normalized = re.sub(
        r"(?i)^(let'?s|can you|could you|please|i want to|i need to)\s+",
        "",
        text.strip(),
    ).strip()
    m = _RE_OPEN.match(normalized)
    if not m:
        return None
    target = m.group(1).strip().rstrip(".!?")
    # Don't steal volume/play sentences
    if re.search(r"\b(volume|brightness|play\s+)", target, re.I):
        return None
    entry = resolve_installed_app(target)
    if not entry:
        return None
    if _gtk_launch(entry.desktop_id):
        return f"Opening {entry.name}."
    return f"I couldn't launch {entry.name}."


_VISION_STEP_PROMPT = """You are a Ubuntu desktop UI automation planner. You see one screenshot.

User goal: {goal}

Reply with ONE JSON object only (no markdown). Use exactly one of these shapes:
{{"done": true, "summary": "short result for the user"}}
{{"click_percent": [x, y]}}  — x and y are 0-100 across the full screen
{{"keys": "ctrl+k"}} — a single xdotool key chord (lowercase modifiers)
{{"keys": "Return"}} or {{"keys": "Escape"}}
{{"type_text": "verbatim text to type"}}
{{"wait_ms": 400}}

Pick the single best next step. If the goal is already satisfied, use done."""


def _parse_vision_step(raw: str) -> Optional[dict]:
    s = _strip_json_fence(raw)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        logger.warning("[SHORTCUTS] vision JSON parse failed: %s", raw[:300])
        return None
    return obj if isinstance(obj, dict) else None


def _execute_vision_step(step: dict) -> bool:
    if step.get("done"):
        return True
    if "wait_ms" in step:
        time.sleep(float(step["wait_ms"]) / 1000.0)
        return False
    if "click_percent" in step:
        pair = step["click_percent"]
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            _click_percent(float(pair[0]), float(pair[1]))
        return False
    if "keys" in step:
        k = str(step["keys"]).strip()
        if k:
            chord = k if "+" in k else k.replace(" ", "+")
            _xdotool(["key", "--clearmodifiers", chord])
        return False
    if "type_text" in step:
        txt = str(step["type_text"])
        x = _which("xdotool")
        if x:
            _run([x, "type", "--delay", "15", "--", txt], timeout=60)
        return False
    return False


def _vision_automation_loop(goal: str, transcript: str) -> str:
    try:
        from sensus.vision.screen import describe_screen, FALLBACK_MESSAGE
    except Exception as e:
        logger.exception("[SHORTCUTS] vision import failed: %s", e)
        return "Vision isn't available for that app action."

    max_steps = int(os.environ.get("SENSUS_SHORTCUTS_VISION_STEPS", "4"))
    ctx = f"Transcript: {transcript}\n"
    for i in range(max_steps):
        prompt = _VISION_STEP_PROMPT.format(goal=goal) + f"\nStep {i + 1}. {ctx}"
        raw = describe_screen(prompt_override=prompt)
        if not raw or raw == FALLBACK_MESSAGE:
            return "I couldn't read the screen to automate that."
        step = _parse_vision_step(raw)
        if not step:
            return "I couldn't plan the next desktop step."
        if step.get("done"):
            return str(step.get("summary") or "Done.")
        _execute_vision_step(step)
        time.sleep(0.25)
    return "I reached the step limit for desktop automation."


def run_shortcuts_action(
    transcript: str,
    payload: dict,
) -> Optional[str]:
    """
    Handle local shortcuts from voice transcript (+ optional orchestrator payload).

    Returns a user-facing string when this module fully handles the request,
    or ``None`` to let the orchestrator answer normally.
    """
    if not _is_linux():
        return None

    text = (transcript or "").strip()
    if not text:
        return None

    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    detail_query = str(details.get("query") or details.get("search") or "").strip()

    for fn in (
        _try_volume,
        _try_brightness,
        _try_wifi,
        _try_bluetooth,
        _try_airplane_mode,
        _try_simple_media,
        _try_open_user_folder,
        _try_in_app_play,
        _try_spotify_phrase,
        _try_open_app,
    ):
        got = fn(text)
        if got:
            return got

    if detail_query and "spotify" in text.lower():
        return _spotify_search_play(detail_query)

    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    import sys

    line = " ".join(sys.argv[1:]).strip() or 'in spotify play dean martin dream a little dream'
    print(run_shortcuts_action(line, {"agent": "shortcuts", "action": "", "details": {}}))


if __name__ == "__main__":
    main()
