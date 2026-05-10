"""
Optional GTK 4 + WebKitGTK 6 overlay (Linux).

Requires system packages, e.g. Ubuntu:
  sudo apt install libgtk-4-dev libwebkitgtk-6.0-dev python3-gi python3-gir1.2-webkitgtk-6.0

If Gtk/WebKit are unavailable, ``run_gui_voice_loop`` raises ImportError and the voice loop
falls back to console-only mode.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_JS_GUARD = threading.Lock()


def _configure_webkit_env() -> None:
    """
    WebKitGTK enables a bubblewrap-based sandbox by default. That fails on many VMware guests
    (``bwrap: setting up uid map: Permission denied``), Docker, or kernels without usable user
    namespaces. We ship trusted ``ui.html`` only, so disabling the sandbox is acceptable here.

    Set ``SENSUS_WEBKIT_SANDBOX=1`` to keep the sandbox on (may still crash in constrained VMs).
    """
    if os.environ.get("SENSUS_WEBKIT_SANDBOX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if os.environ.get("WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS"):
        return
    os.environ["WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS"] = "1"
    logger.debug(
        "WebKit sandbox disabled (unset WEBKIT_DISABLE_* or set SENSUS_WEBKIT_SANDBOX=1 to force)."
    )


class OverlayGui:
    """Thread-safe API for updating the embedded HTML UI from any thread."""

    def __init__(
        self,
        *,
        web_view,
        submit_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        import gi

        gi.require_version("Gtk", "4.0")
        from gi.repository import GLib

        self._GLib = GLib
        self._web_view = web_view
        self._submit_callback: Callable[[str], None] = submit_callback or (
            lambda _t: None
        )
        self._close_callback: Optional[Callable[[], None]] = None
        self._open_session_callback: Optional[Callable[[str], None]] = None
        self._return_to_live_callback: Optional[Callable[[], None]] = None
        self._active_session_id: Optional[str] = None
        self._viewing_session_id: Optional[str] = None
        self._wave_tick_id: Optional[int] = None
        self._wave_phase = 0.0

    def set_submit_callback(self, cb: Callable[[str], None]) -> None:
        self._submit_callback = cb

    def set_close_callback(self, cb: Optional[Callable[[], None]]) -> None:
        """Invoked once when the overlay window is closed (before GTK quits)."""
        self._close_callback = cb

    def push_session_history(self, items: list[dict[str, Any]]) -> None:
        """Show recent sessions (from Db2) in the sidebar."""
        self.push_payload({"type": "sessions", "items": items})

    def set_open_session_callback(self, cb: Optional[Callable[[str], None]]) -> None:
        """Handle sidebar session selection (session id uuid string)."""
        self._open_session_callback = cb

    def set_return_to_live_callback(self, cb: Optional[Callable[[], None]]) -> None:
        """Reload live chat from Db2 when user speaks while a past session is open."""
        self._return_to_live_callback = cb

    def set_active_session_id(self, session_id: str) -> None:
        """Live voice session id — used to know when the UI is showing a past session."""
        self._active_session_id = session_id

    def push_chat_history(
        self,
        rows: list[tuple[str, str]],
        *,
        session_id: str,
        banner: str = "",
    ) -> None:
        """Replace chat bubbles from Db2 replay (roles ``user`` / ``assistant``)."""
        messages: list[dict[str, str]] = []
        for role, content in rows:
            r = (role or "").strip().lower()
            ui_role = "user" if r == "user" else "asst"
            messages.append({"role": ui_role, "text": content or ""})
        aid = (self._active_session_id or "").strip()
        sid = (session_id or "").strip()
        self._viewing_session_id = sid if sid and sid != aid else None
        self.push_payload(
            {
                "type": "chat_history",
                "sessionId": sid,
                "messages": messages,
                "banner": banner or "",
            }
        )

    def _dispatch(self, fn: Callable[[], None]) -> None:
        def _wrap() -> bool:
            fn()
            return False

        self._GLib.idle_add(_wrap)

    def _emit_js(self, payload: dict) -> None:
        """Run on main thread only."""
        # Pass JSON as a string literal and JSON.parse in-page so chat text cannot break
        # the JS parser (e.g. U+2028/U+2029, unusual escapes) across WebKit versions.
        payload_json = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))
        js_string_literal = json.dumps(payload_json)
        script = f"window.sensusReceive(JSON.parse({js_string_literal}));"
        wv = self._web_view

        def done(web_view: object, result: object, *_args: object) -> None:
            try:
                fin = getattr(web_view, "evaluate_javascript_finish", None)
                if callable(fin):
                    fin(result)
            except Exception as e:
                logger.warning("WebKit evaluate_javascript failed: %s", e)

        try:
            wv.evaluate_javascript(script, -1, None, None, None, done, None)
        except (TypeError, AttributeError):
            try:
                wv.evaluate_javascript(
                    script, len(script), None, None, None, done, None
                )
            except Exception:
                try:
                    wv.run_javascript(script, None, done, None)
                except Exception:
                    logger.debug("WebKit JS bridge failed", exc_info=True)

    def push_payload(self, payload: dict) -> None:
        def run() -> None:
            with _JS_GUARD:
                self._emit_js(payload)

        self._dispatch(run)

    def add_user_bubble(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        if self._viewing_session_id is not None:
            cb = self._return_to_live_callback
            if callable(cb):
                try:
                    cb()
                except Exception:
                    logger.exception("return_to_live_callback failed")
            return
        self.push_payload({"type": "bubble", "role": "user", "text": t})

    def add_assistant_bubble(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        self.push_payload({"type": "bubble", "role": "asst", "text": t})

    def set_interim(self, text: str) -> None:
        self.push_payload({"type": "interim", "text": text or ""})

    def clear_interim(self) -> None:
        self.set_interim("")

    def _wave_tick(self) -> bool:
        """GLib source; returns True to repeat."""
        from sensus.voice import tts

        GLib = self._GLib
        if not tts.is_speaking():
            self._wave_tick_id = None
            self.push_payload(
                {"type": "wave", "mode": "idle", "levels": [0.05] * 16}
            )
            return GLib.SOURCE_REMOVE

        self._wave_phase += 0.42
        levels = []
        for i in range(16):
            base = 0.38 + 0.42 * math.sin(self._wave_phase + i * 0.45)
            noise = random.uniform(-0.09, 0.09)
            levels.append(max(0.12, min(1.0, base + noise)))
        self.push_payload(
            {"type": "wave", "mode": "speaking", "levels": levels})
        return GLib.SOURCE_CONTINUE

    def notify_tts_started(self) -> None:
        """Call from TTS thread immediately before ``tts.speak`` (wave anim while speaking)."""

        def start() -> None:
            GLib = self._GLib
            if self._wave_tick_id is not None:
                GLib.source_remove(self._wave_tick_id)
            self._wave_tick_id = GLib.timeout_add(50, self._wave_tick)

        self._dispatch(start)

    def notify_tts_finished(self) -> None:
        def stop() -> None:
            GLib = self._GLib
            if self._wave_tick_id is not None:
                GLib.source_remove(self._wave_tick_id)
                self._wave_tick_id = None
            self.push_payload(
                {"type": "wave", "mode": "idle", "levels": [0.05] * 16}
            )

        self._dispatch(stop)

    def submit_from_ui(self, text: str) -> None:
        """Handle message from WebKit (already on GTK thread)."""
        t = (text or "").strip()
        if not t:
            return

        def run() -> None:
            try:
                self._submit_callback(t)
            except Exception:
                logger.exception("Overlay submit handler failed")

        threading.Thread(
            target=run, name="sensus-overlay-submit", daemon=True).start()


def _overlay_corner_xy(win: object, surf: object) -> tuple[int, int]:
    """Compute root (x, y) for the overlay using the current monitor workarea."""
    from gi.repository import Gdk

    display = Gdk.Display.get_default()
    monitor = display.get_monitor_at_surface(surf)
    workarea = monitor.get_workarea()
    ww = int(win.get_width())
    wh = int(win.get_height())
    if ww <= 1:
        ww = 460
    if wh <= 1:
        wh = 640

    corner = (
        os.environ.get("SENSUS_OVERLAY_CORNER", "top-right") or "top-right"
    ).strip().lower()
    top_left = corner in ("tl", "top-left", "topleft", "left", "nw")

    if top_left:
        x = workarea.x
        y = workarea.y
    else:
        x = workarea.x + workarea.width - ww
        y = workarea.y

    x = max(workarea.x, min(x, workarea.x + max(0, workarea.width - ww)))
    y = max(workarea.y, min(y, workarea.y + max(0, workarea.height - wh)))
    return (x, y)


def _pin_overlay_window(win: object) -> None:
    """
    Keep the overlay above other windows and move it to a screen corner when the compositor allows.

    Default corner is top-right. Set ``SENSUS_OVERLAY_CORNER=top-left`` for the previous behavior.
    Set ``SENSUS_OVERLAY_PIN=0`` to disable keep-above and corner placement (e.g. debugging).
    Wayland often ignores arbitrary (x,y); X11 + GdkX11 usually works.
    """
    from gi.repository import GLib

    if os.environ.get("SENSUS_OVERLAY_PIN", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return

    try:
        win.set_keep_above(True)
    except Exception:
        logger.debug("set_keep_above failed", exc_info=True)

    tries = [0]

    def try_corner() -> bool:
        tries[0] += 1
        if tries[0] > 80:
            return False
        surf = win.get_surface()
        if surf is None:
            return True
        try:
            x, y = _overlay_corner_xy(win, surf)
            surf.move(x, y)
        except Exception:
            try:
                from gi.repository import GdkX11

                x, y = _overlay_corner_xy(win, surf)
                xs = GdkX11.X11Surface.get_x11_surface(surf)
                if xs is not None:
                    xs.move(x, y)
            except Exception:
                logger.debug(
                    "Could not move overlay to configured corner (expected on some Wayland setups)",
                    exc_info=True,
                )
        return False

    GLib.idle_add(try_corner)


def _try_setup_gtk() -> tuple[object, object]:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("WebKit", "6.0")
    from gi.repository import Gtk, WebKit

    return Gtk, WebKit


def _script_message_to_json_text(message: object) -> str:
    """
    WebKitGTK 6+ passes ``JavaScriptCore.Value`` to ``script-message-received``.
    Older stacks used a wrapper with ``get_js_value()`` (JavaScriptResult).
    """
    if message is None:
        return ""
    jv: Any = message
    get_js = getattr(message, "get_js_value", None)
    if callable(get_js):
        try:
            jv = get_js()
        except Exception:
            logger.debug(
                "get_js_value failed on script message", exc_info=True)
            jv = message
    is_string = getattr(jv, "is_string", None)
    if callable(is_string) and is_string():
        to_s = getattr(jv, "to_string", None)
        if callable(to_s):
            return (to_s() or "").strip()
    to_json = getattr(jv, "to_json", None)
    if callable(to_json):
        try:
            return (to_json(0) or "").strip()
        except Exception:
            logger.debug("JavaScriptCore.Value.to_json failed", exc_info=True)
    if isinstance(message, str):
        return message.strip()
    return ""


def run_gui_voice_loop(session_fn: Callable[[OverlayGui], None]) -> None:
    """
    Run GTK main loop on this thread; invoke ``session_fn(gui)`` from a worker thread
    after the WebView is ready.
    """
    _configure_webkit_env()
    Gtk, WebKit = _try_setup_gtk()
    from gi.repository import GLib

    gui_holder: dict[str, OverlayGui] = {}

    def on_script_message(_mgr: object, message: object) -> None:
        raw = _script_message_to_json_text(message)
        if not raw:
            logger.debug("Empty script message payload (WebKit/JSC bridge)")
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        msg_type = data.get("type")
        g = gui_holder.get("gui")
        if msg_type == "send":
            text = str(data.get("text", "")).strip()
            if g:
                g.submit_from_ui(text)
            return
        if msg_type == "open_session":
            sid = str(data.get("session_id", "")).strip()
            if not sid or not g:
                return
            cb = getattr(g, "_open_session_callback", None)
            if not callable(cb):
                return

            def _run_open() -> None:
                try:
                    cb(sid)
                except Exception:
                    logger.exception("open_session callback failed")

            threading.Thread(
                target=_run_open,
                name="sensus-open-session",
                daemon=True,
            ).start()
            return

    def activate(app: object) -> None:
        win = Gtk.ApplicationWindow(application=app)
        win.set_title("Sensus")
        win.set_default_size(460, 640)
        win.set_decorated(False)

        try:
            ucm = WebKit.UserContentManager.new()
        except AttributeError:
            ucm = WebKit.UserContentManager()
        try:
            web_view = WebKit.WebView.new_with_user_content_manager(ucm)
        except AttributeError:
            web_view = WebKit.WebView(user_content_manager=ucm)

        try:
            ucm.register_script_message_handler("sensus", None)
        except TypeError:
            ucm.register_script_message_handler("sensus")

        ucm.connect("script-message-received::sensus", on_script_message)

        gui = OverlayGui(web_view=web_view)
        gui_holder["gui"] = gui

        win.set_child(web_view)

        html_path = Path(__file__).resolve().parent / "ui.html"
        web_view.load_uri(html_path.as_uri())

        def start_session() -> None:
            threading.Thread(
                target=lambda: session_fn(gui),
                name="sensus-voice-session",
                daemon=True,
            ).start()

        def _boot_session() -> bool:
            start_session()
            return False

        GLib.idle_add(_boot_session)
        win.present()
        _pin_overlay_window(win)

        def on_close(_w: object) -> None:
            g = gui_holder.get("gui")
            if g and g._close_callback:
                try:
                    g._close_callback()
                except Exception:
                    logger.exception("Overlay close callback failed")
            app.quit()

        win.connect("close-request", on_close)

    app = Gtk.Application(application_id="ai.sensus.overlay")
    app.connect("activate", activate)

    try:
        app.run(None)
    except KeyboardInterrupt:
        pass
