"""
GNOME Shell (Wayland-friendly) window list and focus via ``org.gnome.Shell.Eval``.

Uses the session D-Bus — no X11 EWMH. Disable with ``SENSUS_GNOME_SHELL_EVAL=0``.

Requires ``busctl`` (systemd) or ``gdbus`` (glib); Eval must be allowed for your
GNOME Shell version (standard on Ubuntu Desktop).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil

from sensus.agents.linux_exec import run

logger = logging.getLogger(__name__)

_SEP = "\x1f"

_JS_META = """
function _mw(a) {
  let m = null;
  try { if (a.get_meta_window) m = a.get_meta_window(); } catch (e) {}
  if (!m) try { m = a.meta_window; } catch (e) {}
  return m;
}
""".strip()

_LIST_JS = (
    _JS_META
    + """
(() => {
  const SEP = "\\u001f";
  const out = [];
  for (const a of global.get_window_actors()) {
    const m = _mw(a);
    if (!m) continue;
    let t = "";
    try { t = (m.get_title() || "").trim(); } catch (e) {}
    if (t) out.push(t);
  }
  return out.join(SEP);
})()
"""
).strip()


def _enabled() -> bool:
    return os.environ.get("SENSUS_GNOME_SHELL_EVAL", "1").strip() != "0"


def _parse_busctl_eval(stdout: str) -> tuple[bool, str]:
    """Parse ``busctl`` reply for ``Eval`` → ``(boolean, string)``."""
    text = (stdout or "").strip()
    if not text:
        return False, ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, ""
    ok_line = lines[0].strip()
    # systemd busctl prints typed values: ``b true``, ``s "hi"``
    ok = (
        ok_line == "true"
        or ok_line == "b true"
        or re.fullmatch(r"b\s+true", ok_line, re.I) is not None
    )
    if not ok:
        return False, (lines[1] if len(lines) > 1 else "")
    if len(lines) < 2:
        return True, ""
    rest = "\n".join(lines[1:]).strip()
    if rest.startswith("s "):
        rest = rest[2:].lstrip()
    if rest.startswith('"'):
        try:
            return True, json.loads(rest)
        except json.JSONDecodeError:
            pass
        if len(rest) >= 2 and rest.endswith('"'):
            return True, rest[1:-1]
    if rest.startswith("'") and len(rest) >= 2 and rest.endswith("'"):
        return True, rest[1:-1]
    return True, rest


def _parse_gdbus_eval(stdout: str) -> tuple[bool, str]:
    t = (stdout or "").strip()
    if not t or t.startswith("(false"):
        return False, ""
    m = re.match(r"\(\s*true\s*,\s*'((?:\\.|[^'\\])*)'\s*\)", t, re.DOTALL)
    if m:
        body = m.group(1).replace("\\'", "'").replace("\\\\", "\\")
        return True, body
    m = re.match(r'\(\s*true\s*,\s*"((?:\\.|[^"\\])*)"\s*\)', t, re.DOTALL)
    if m:
        try:
            return True, json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            return True, m.group(1)
    return False, ""


def eval_js(js: str) -> tuple[bool, str]:
    """Run JavaScript inside gnome-shell; return (success, result_string)."""
    if not _enabled():
        return False, ""
    js_one = " ".join(js.split())
    gdbus = shutil.which("gdbus")
    if gdbus:
        r = run(
            [
                gdbus,
                "call",
                "--session",
                "--dest",
                "org.gnome.Shell",
                "--object-path",
                "/org/gnome/Shell",
                "--method",
                "org.gnome.Shell.Eval",
                js_one,
            ],
            timeout=6,
        )
        if r.returncode == 0:
            out = _parse_gdbus_eval(r.stdout or "")
            if out[0]:
                return out
        logger.debug(
            "[GNOME_SHELL] gdbus Eval rc=%s out=%r err=%s",
            r.returncode,
            (r.stdout or "")[:200],
            r.stderr,
        )

    busctl = shutil.which("busctl")
    if busctl:
        r = run(
            [
                busctl,
                "--user",
                "call",
                "org.gnome.Shell",
                "/org/gnome/Shell",
                "org.gnome.Shell",
                "Eval",
                "s",
                js_one,
            ],
            timeout=6,
        )
        if r.returncode == 0:
            return _parse_busctl_eval(r.stdout or "")
        logger.debug(
            "[GNOME_SHELL] busctl Eval rc=%s out=%r err=%s",
            r.returncode,
            (r.stdout or "")[:200],
            r.stderr,
        )

    return False, ""


def list_window_titles() -> list[str]:
    """Return non-empty window titles (GNOME Shell only)."""
    ok, raw = eval_js(_LIST_JS)
    if not ok or not raw:
        return []
    return [p for p in raw.split(_SEP) if p.strip()]


def activate_title_substring(needle: str) -> bool:
    """Focus the first window whose title contains ``needle`` (case-insensitive)."""
    n = (needle or "").strip()
    if len(n) < 1:
        return False
    lit = json.dumps(n)
    js = (
        _JS_META
        + f"""
(() => {{
  const needle = ({lit}).toLowerCase();
  const tm = global.get_current_time();
  for (const a of global.get_window_actors()) {{
    const m = _mw(a);
    if (!m) continue;
    let t = "";
    try {{ t = (m.get_title() || "").toLowerCase(); }} catch (e) {{}}
    if (t && t.includes(needle)) {{
      m.activate(tm);
      return "1";
    }}
  }}
  return "";
}})()
"""
    ).strip()
    ok, raw = eval_js(js)
    return bool(ok and raw.strip())


def probe_works() -> bool:
    """Return True if Eval runs (quick sanity check)."""
    ok, _ = eval_js("(() => 'ok')()")
    return ok
