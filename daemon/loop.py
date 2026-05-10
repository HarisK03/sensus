"""
Integrated Sensus voice loop daemon.

Flow:
1) Stream microphone audio to Watson STT.
2) Handle partial transcripts (optional overlay interim line).
3) Route final transcripts to intent router.
4) Send transcript + route context to orchestrator.
5) Speak orchestrator reply in a non-blocking daemon thread.

GUI (Linux, GTK 4 + WebKit 6): ``python -m sensus.daemon.loop --gui``
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

from sensus.daemon.main import Orchestrator, dispatch_router_fast_path
from sensus.storage import db2
from sensus.daemon.router import classify_intent
from sensus.voice import tts
from sensus.voice.watson_stt import StreamingWatsonSTT, load_stt_env

logger = logging.getLogger(__name__)
STOP_WORDS = ("stop", "cancel", "quiet", "shut up")

_turn_lock = threading.Lock()


def _derive_session_title(first_user_text: str) -> str:
    """Short sidebar/Db2 title from the first user utterance (until close-summary runs)."""
    t = " ".join((first_user_text or "").split()).strip()
    if not t:
        return "Session"
    if len(t) > 72:
        return t[:69] + "…"
    return t


def _sidebar_sessions_payload(
    user_id: Optional[str],
    active_session_id: str,
) -> list[dict[str, Any]]:
    """Recent sessions for the overlay plus ``current`` flag."""
    rows = db2.list_recent_sessions(user_id, limit=10)
    out: list[dict[str, Any]] = []
    for it in rows:
        d = dict(it)
        d["current"] = d.get("id") == active_session_id
        out.append(d)
    return out


def _log_fast_path_result(routed_agent: str, message: str) -> None:
    """Console lines for router fast path (matches previous per-agent formatting)."""
    if routed_agent == "coding":
        snip = message[:800] + ("…" if len(message) > 800 else "")
        print(f"[CODING] {snip}")
        return
    if routed_agent == "shell":
        snip = message if len(message) <= 600 else message[:600] + "…"
        print(f"[SHELL] handled locally: {snip}")
        return
    if routed_agent == "browser":
        snip = message[:800] + ("…" if len(message) > 800 else "")
        print(f"[BROWSER] {snip}")
        return
    if routed_agent in ("desktop", "shortcuts"):
        print(f"[SHORTCUTS] handled locally: {message}")
        return
    print(f"[FAST] agent={routed_agent} {message}")


def _speak_async(text: str, gui: Optional[Any] = None) -> None:
    if not text.strip():
        return

    def run() -> None:
        try:
            if gui:
                gui.notify_tts_started()
            print(f"[TTS] speaking: {text}")
            tts.speak(text)
        except Exception:
            logger.exception("TTS failed")
        finally:
            if gui:
                gui.notify_tts_finished()

    threading.Thread(target=run, name="sensus-loop-tts", daemon=True).start()


def _process_user_text(
    transcript: str,
    orchestrator: Orchestrator,
    gui: Optional[Any],
    session_id: Optional[str],
    *,
    title_state: Optional[list[bool]] = None,
    user_id: Optional[str] = None,
) -> None:
    """Shared path for voice finals and typed overlay messages."""
    text = (transcript or "").strip()
    if not text:
        return

    print(f"[STT] final: {text}")

    if gui:
        gui.clear_interim()

    lower_text = text.lower()
    if tts.is_speaking():
        has_stop_word = any(
            re.search(rf"\b{re.escape(word)}\b", lower_text) for word in STOP_WORDS
        )
        if has_stop_word:
            print("[TTS] stop requested")
            tts.stop()
            return
        print("[TTS] waiting for current playback to finish...")
        tts.wait_until_idle()

    route = classify_intent(text)
    print(f"[ROUTER] agent={route.agent} confidence={route.confidence:.2f}")

    sid = session_id if db2.is_configured() else None
    if sid:
        db2.insert_message(sid, "user", text, route.agent)
        if title_state is not None and not title_state[0]:
            db2.update_session_name(sid, _derive_session_title(text))
            title_state[0] = True
            if gui and hasattr(gui, "push_session_history"):
                try:
                    gui.push_session_history(
                        _sidebar_sessions_payload(user_id, session_id)
                    )
                except Exception:
                    logger.exception(
                        "Failed to refresh session sidebar after title")

    if gui:
        gui.add_user_bubble(text)

    try:
        fast = dispatch_router_fast_path(text, route.agent, route.confidence)
        if fast:
            _log_fast_path_result(route.agent, fast)
            if sid:
                db2.insert_message(sid, "assistant", fast, route.agent)
            if gui:
                gui.add_assistant_bubble(fast)
            _speak_async(fast, gui)
            return
    except Exception:
        logger.exception("Router fast path failed")

    route_context = (
        "Routed intent context:\n"
        f"- transcript: {text}\n"
        f"- suggested_agent: {route.agent}\n"
        f"- confidence: {route.confidence:.2f}\n"
        "- Use this as routing guidance while generating your reply."
    )
    turn = orchestrator.turn(text, extra_system_suffix=route_context)
    response = (turn.reply_text or "").strip()
    print(f"[ORCHESTRATOR] response: {response}")

    if sid:
        db2.insert_message(sid, "assistant", response, route.agent)
    if gui:
        gui.add_assistant_bubble(response)
    _speak_async(response, gui)


def _run_voice_session(gui: Optional[Any] = None) -> None:
    key, url, model = load_stt_env()
    orchestrator = Orchestrator()
    session_id = str(uuid.uuid4())
    user_id = os.environ.get("SENSUS_USER_ID", "").strip() or None

    if gui:
        gui.set_active_session_id(session_id)

    db2.ensure_schema()
    if db2.is_configured():
        created = db2.create_session(session_id, user_id, name=None)
        if created and db2.is_configured():
            logger.info(
                "Db2 persistence on — writing to tables sessions and messages "
                "(schema is usually your Db2 user id)."
            )
        elif not db2.is_configured():
            logger.warning(
                "Db2 unreachable or timed out — continuing without persistence "
                "(see IBM_DB_CONN_STRING / IBM_DB_CONNECT_TIMEOUT)."
            )
        if gui and db2.is_configured():
            try:
                gui.push_session_history(
                    _sidebar_sessions_payload(user_id, session_id)
                )
            except Exception:
                logger.exception("Failed to load session history for sidebar")
    else:
        cs_set = bool((db2.connection_string() or "").strip())
        if cs_set:
            logger.warning(
                "IBM_DB_CONN_STRING is set but Db2 persistence is off "
                "(ibm_db missing or failed to import — pip install ibm-db + CLI driver)."
            )
        else:
            logger.warning(
                "Db2 persistence off — set IBM_DB_CONN_STRING in sensus/.env next to "
                "daemon/ and storage/ to save sessions and messages."
            )

    _finalize_lock = threading.Lock()
    _finalized = False

    def finalize_session() -> None:
        nonlocal _finalized
        with _finalize_lock:
            if _finalized:
                return
            _finalized = True
        if db2.is_configured():
            try:
                db2.finalize_session(session_id)
            except Exception:
                logger.exception("Session finalize (Db2 summary) failed")

    if gui and hasattr(gui, "set_close_callback"):
        gui.set_close_callback(finalize_session)

    title_state = [False]

    def locked_process(raw: str) -> None:
        with _turn_lock:
            _process_user_text(
                raw,
                orchestrator,
                gui,
                session_id,
                title_state=title_state,
                user_id=user_id,
            )

    if gui:
        gui.set_submit_callback(locked_process)

    def return_to_live_chat() -> None:
        if not gui:
            return
        if not db2.is_configured():
            gui.push_chat_history(
                [],
                session_id=session_id,
                banner="",
            )
            return
        rows = db2.fetch_messages_chronological(session_id)
        gui.push_chat_history(
            rows,
            session_id=session_id,
            banner="",
        )

    def on_open_session(selected_id: str) -> None:
        if not gui:
            return
        sid_key = (selected_id or "").strip()
        try:
            if not db2.is_configured():
                gui.push_chat_history(
                    [],
                    session_id=sid_key,
                    banner="Db2 not configured — set IBM_DB_CONN_STRING (and install ibm-db + CLI driver) to load session history.",
                )
                return
            rows = db2.fetch_messages_chronological(sid_key)
            banner = ""
            if not rows:
                banner = (
                    "No stored messages for this session. "
                    "Confirm IBM_DB_CONN_STRING, ibm-db + IBM CLI driver, and that the messages table exists "
                    "(scripts/db2_schema.sql or IBM_DB_AUTO_SCHEMA=1)."
                )
            gui.push_chat_history(rows, session_id=sid_key, banner=banner)
        except Exception:
            logger.exception("open_session / Db2 history failed")

    if gui:
        gui.set_return_to_live_callback(return_to_live_chat)
        gui.set_open_session_callback(on_open_session)

    def on_partial(text: str) -> None:
        if gui:
            gui.set_interim(text)

    def on_final(transcript: str) -> None:
        locked_process(transcript)

    stt = StreamingWatsonSTT(
        api_key=key,
        service_url=url,
        model=model,
        on_interim=on_partial,
        on_final=on_final,
    )

    print("[STT] starting Watson STT stream...")
    stt.start()
    print("[STT] listening... press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STT] stopping...")
    finally:
        stt.stop()
        finalize_session()


def run_voice_loop(use_gui: bool = False) -> None:
    if use_gui:
        try:
            from sensus.overlay.app import run_gui_voice_loop
        except Exception as e:
            logger.warning("GUI import failed (%s); running console-only.", e)
        else:

            def _session(gui: Any) -> None:
                _run_voice_session(gui)

            try:
                run_gui_voice_loop(_session)
            except Exception as e:
                _gui_fail_help(e)
                logger.warning("GUI failed (%s); running console-only.", e)
            else:
                return

    _run_voice_session(None)


def _gui_fail_help(err: Exception) -> None:
    """Log one-line install hint for common venv / missing typelib issues."""
    if isinstance(err, ModuleNotFoundError) and getattr(err, "name", "") == "gi":
        logger.warning(
            "Missing gi (PyGObject). In this venv run: pip install PyGObject "
            "(needs apt: libgirepository1.0-dev libcairo2-dev libgtk-4-dev pkg-config "
            "gir1.2-gtk-4.0 gir1.2-webkit-6.0)"
        )
        return
    msg = str(err).lower()
    if "gi" in msg and "no module" in msg:
        logger.warning(
            "Try: pip install PyGObject (inside the venv), plus GTK/WebKit gir packages from apt."
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Sensus voice loop daemon.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="GTK/WebKit overlay (Linux; requires Gtk 4 + WebKit 6 typelibs).",
    )
    args = parser.parse_args()
    run_voice_loop(use_gui=args.gui)


if __name__ == "__main__":
    main()
