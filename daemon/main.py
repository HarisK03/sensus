"""
Phase 1.5 — Orchestrator (Featherless OpenAI-compatible).

Maintains ``conversation_history`` as OpenAI-style role/content dicts (user + assistant turns).
Each turn appends the user transcript, calls the orchestrator model, appends the assistant reply.

**Session vs memory:** In-process ``conversation_history`` resets each process. Optional IBM Db2
persistence (``IBM_DB_CONN_STRING``) stores sessions/messages and sidebar history; see
``sensus/storage/db2.py``.

The model is instructed to answer either with **plain natural language** or with a single
**JSON tool payload** when it wants the shell to run an agent action.

Environment (``sensus/.env``):
  FEATHERLESS_AI_API_KEY — required (shared with router)
  FEATHERLESS_BASE_URL — optional
  FEATHERLESS_ORCHESTRATOR_MODEL — optional (default: google/gemma-4-31B-it); also used by router & browser planner
  FEATHERLESS_VISION_MODEL — optional (default: google/gemma-3-12b-it); used by ``sensus.vision.screen``
  FEATHERLESS_ORCHESTRATOR_TIMEOUT_SEC — optional (default 120)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_ORCHESTRATOR_MODEL = "google/gemma-4-31B-it"

TOOLS_SYSTEM_PROMPT = """You are Sensus OS — a voice-first Ubuntu assistant for visually impaired users.

You may receive routed context (e.g. suggested agent). Stay helpful, concise, and safe.

Available capabilities (tools are not executed here — you describe intent so downstream agents can act):

1. browser — Firefox via automation: open URLs, search, click elements using page semantics.
2. desktop — Native apps: focus windows, mouse/keyboard, AT-SPI accessible names.
3. shell — Bash commands: propose commands; in production the user confirms before run.
4. shortcuts — Volume (mute, steps, level), brightness, Wi‑Fi / Bluetooth / airplane mode, media keys, MPRIS/Spotify, launching apps.
5. coding — Generate a Python script from a spoken spec, save it, open in VS Code / Cursor if available, run in a new terminal (e.g. games, small tools).

When the user wants you to **perform** a concrete computer action (not just explain), reply with
**one JSON object only** (no markdown fences, no surrounding prose):
{"kind":"tool_call","agent":"browser|desktop|shell|shortcuts|coding","action":"short verb phrase","details":{}}

Put structured parameters inside "details" (strings, booleans, URLs, selectors if known).

**shell:** Always set ``details.command`` to a single shell-free command line suitable for ``shlex.split``
(no pipes unless safe). Paths may use ``~/``, ``$HOME``, ``$USER`` — they are expanded server-side. Example:
``{"kind":"tool_call","agent":"shell","action":"list downloads","details":{"command":"ls -la /home/me/Downloads"}}``

**coding:** Use when the user asks to **implement**, **write**, **create**, or **build** a program/script (especially Python), e.g. a pygame GUI or a game with a fixed secret word. Put extra literals in ``details.hints`` if the transcript is ambiguous. Example:
``{"kind":"tool_call","agent":"coding","action":"implement wordle","details":{"hints":"SECRET_WORD must be exactly: judge; use pygame window"}}``

When the user wants normal conversation, explanations, or non-action Q&A, reply with **plain text only**
(no JSON).

If the user says something conversational (their name, a question, small talk),
respond conversationally. Do NOT return a tool_call JSON for conversational input.

If routed intent context below says suggested_agent: conversation, reply with plain text only
unless the user clearly asks you to control the computer, browser, or apps.

**Live page data:** Never promise to check the browser, read reviews/ratings/prices on the current
page, compare search results, or say "one moment" / "I'll look" for page-specific facts unless you
**in the same reply** output a browser tool_call JSON. If the user wants information that requires
seeing the open page, use tool_call agent browser — do not stall on conversational filler.

**Checkout / cart:** If the user wants to open the cart, proceed to checkout, or pay on a site that
is already open in the browser, use a **browser** tool_call with a short action like
"proceed to checkout" or "open shopping cart" — do not only chat about it.

Either plain text OR one JSON object — never both."""


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def load_orchestrator_env() -> tuple[str, str, str, float]:
    """Return (api_key, base_url, model, timeout_sec)."""
    _try_load_dotenv()
    key = os.environ.get("FEATHERLESS_AI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FEATHERLESS_AI_API_KEY is not set.")
    base = os.environ.get("FEATHERLESS_BASE_URL",
                          DEFAULT_BASE_URL).strip().rstrip("/")
    model = (
        os.environ.get("FEATHERLESS_ORCHESTRATOR_MODEL",
                       DEFAULT_ORCHESTRATOR_MODEL).strip()
        or DEFAULT_ORCHESTRATOR_MODEL
    )
    timeout_sec = float(os.environ.get(
        "FEATHERLESS_ORCHESTRATOR_TIMEOUT_SEC", "120"))
    return key, base, model, timeout_sec


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _looks_like_tool_payload(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("kind") == "tool_call":
        return True
    agent = str(obj.get("agent", "")).lower()
    if agent in {"browser", "desktop", "shell", "shortcuts", "coding"}:
        return True
    if "tool_calls" in obj or obj.get("type") == "tool_call":
        return True
    return False


def parse_assistant_output(
    raw: str,
    *,
    native_tool_calls: Optional[list] = None,
) -> tuple[bool, Optional[dict], str]:
    """
    Decide whether the turn is plain text or tool JSON.

    Returns (is_tool_json, tool_payload_or_none, display_text_for_logs).
    """
    if native_tool_calls:
        payload: dict[str, Any] = {"kind": "tool_call", "calls": []}
        for tc in native_tool_calls:
            fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn else ""
            args_raw = getattr(fn, "arguments", "{}") if fn else "{}"
            try:
                args = json.loads(args_raw) if isinstance(
                    args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            payload["calls"].append({"name": name, "arguments": args})
        text = json.dumps(payload)
        return True, payload, text

    text = (raw or "").strip()
    if not text:
        return False, None, ""

    candidate = _strip_json_fence(text)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return False, None, text

    if isinstance(obj, dict) and _looks_like_tool_payload(obj):
        return True, obj, candidate

    return False, None, text


def _routed_agent_from_suffix(extra_system_suffix: str) -> Optional[str]:
    if not (extra_system_suffix or "").strip():
        return None
    m = re.search(r"suggested_agent:\s*(\w+)",
                  extra_system_suffix, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def tool_call_to_speech(parsed: dict) -> str:
    """Convert tool call payload to a short natural-language confirmation."""
    agent = str(parsed.get("agent", "")).strip().lower()
    action = str(parsed.get("action", "")).strip()
    details = parsed.get("details")
    details = details if isinstance(details, dict) else {}

    if agent == "browser":
        return "On it, opening the browser now."

    if agent == "shell":
        cmd = str(details.get("command", "")).strip()
        if cmd:
            return f"Running {cmd} to {action or 'handle that request'}."
        if action:
            return f"Running a shell command to {action}."
        return "Running that shell command now."

    if agent == "desktop":
        app = (
            str(details.get("application", "")).strip()
            or str(details.get("app", "")).strip()
            or str(details.get("target", "")).strip()
        )
        if app:
            return f"Opening {app} for you."
        if action:
            return f"Handling that desktop action: {action}."
        return "Handling that desktop action now."

    if agent == "shortcuts":
        if action:
            if action.lower().startswith("lower"):
                return "Lowering your volume."
            return f"Applying shortcut: {action}."
        return "Applying that shortcut now."

    if agent == "coding":
        return "Generating code, opening your editor, and starting a terminal run."

    if action:
        return f"On it, I will {action}."
    return "On it."


def dispatch_tool_call(transcript: str, payload: dict) -> Optional[str]:
    """Run the matching agent for a tool-call payload.

    Returns a TTS-ready natural-language string when an agent handled the call,
    or ``None`` when no agent owns this payload (caller should fall back to
    ``tool_call_to_speech``). Heavy agent modules are imported lazily so the
    daemon stays cheap to load when only conversation turns happen.
    """
    agent_name = str(payload.get("agent", "")).strip().lower()
    if agent_name == "shortcuts":
        try:
            from sensus.agents import shortcuts as shortcuts_agent
        except Exception:
            logger.exception("Failed to import shortcuts agent")
            return None
        try:
            return shortcuts_agent.run_shortcuts_action(transcript, payload)
        except Exception:
            logger.exception("Shortcuts agent execution failed")
            return None
    if agent_name == "desktop":
        try:
            from sensus.agents import desktop as desktop_agent
        except Exception:
            logger.exception("Failed to import desktop agent")
            return None
        try:
            return desktop_agent.run_desktop_action(transcript, payload)
        except Exception:
            logger.exception("Desktop agent execution failed")
            return None
    if agent_name == "shell":
        try:
            from sensus.agents import shell as shell_agent
        except Exception:
            logger.exception("Failed to import shell agent")
            return None
        try:
            return shell_agent.run_shell_action(transcript, payload)
        except Exception:
            logger.exception("Shell agent execution failed")
            return None
    if agent_name == "coding":
        try:
            from sensus.agents import coding as coding_agent
        except Exception:
            logger.exception("Failed to import coding agent")
            return None
        try:
            return coding_agent.run_coding_task(transcript, payload)
        except Exception:
            logger.exception("Coding agent execution failed")
            return None
    if agent_name != "browser":
        return None
    try:
        from sensus.agents import browser as browser_agent
    except Exception:
        logger.exception("Failed to import browser agent")
        return None
    try:
        return asyncio.run(browser_agent.run_browser_action(transcript, payload))
    except RuntimeError as e:
        # asyncio.run() refuses to nest inside an already-running loop.
        logger.warning("Browser agent could not run synchronously: %s", e)
        return None
    except Exception:
        logger.exception("Browser agent execution failed")
        return None


def dispatch_router_fast_path(
    transcript: str,
    suggested_agent: str,
    confidence: float,
    *,
    min_confidence: float = 0.68,
) -> Optional[str]:
    """Mirror high-confidence voice routing into :func:`dispatch_tool_call`.

    When the intent router is confident, run the same agent stack as structured tool
    payloads—``desktop`` tries the desktop agent first, then shortcuts—without paying
    for an orchestrator round trip. Returns ``None`` when the turn should fall through
    to :class:`Orchestrator` (low confidence, conversation, or unknown agent).
    """
    if confidence < min_confidence:
        return None
    agent = (suggested_agent or "").strip().lower()
    if agent == "conversation":
        return None

    empty = {"action": "", "details": {}}

    if agent == "desktop":
        fast = dispatch_tool_call(
            transcript, {"agent": "desktop", **empty})
        if fast:
            return fast
        return dispatch_tool_call(
            transcript, {"agent": "shortcuts", **empty})

    if agent == "shortcuts":
        return dispatch_tool_call(
            transcript, {"agent": "shortcuts", **empty})

    if agent in ("coding", "shell", "browser"):
        return dispatch_tool_call(
            transcript, {"agent": agent, **empty})

    return None


@dataclass
class OrchestratorTurnResult:
    """One user turn + model reply."""

    user_text: str
    assistant_raw: str
    is_tool_json: bool
    tool_payload: Optional[dict]
    reply_text: str


@dataclass
class Orchestrator:
    """
    Featherless orchestrator with rolling ``conversation_history``.

    ``conversation_history`` holds only ``user`` / ``assistant`` messages; system is injected each
    request. It does **not** persist across separate processes or CLI invocations unless you
    serialize it or add storage (Phase 4).
    """

    system_prompt: str = TOOLS_SYSTEM_PROMPT
    model: Optional[str] = None
    temperature: float = 0.4
    max_tokens: int = 2048
    conversation_history: List[dict[str, str]] = field(default_factory=list)

    def messages_for_request(self) -> list[dict[str, Any]]:
        return [{"role": "system", "content": self.system_prompt}, *self.conversation_history]

    def reset(self) -> None:
        self.conversation_history.clear()

    def turn(
        self,
        user_transcript: str,
        *,
        extra_system_suffix: str = "",
    ) -> OrchestratorTurnResult:
        """
        Append user message, run orchestrator model, append assistant message, return structured result.
        """
        text = (user_transcript or "").strip()
        if not text:
            return OrchestratorTurnResult(
                user_text="",
                assistant_raw="",
                is_tool_json=False,
                tool_payload=None,
                reply_text="",
            )

        routed = _routed_agent_from_suffix(extra_system_suffix)
        if routed == "conversation":
            lower = text.lower()
            _screen_phrases = (
                "what's on my screen",
                "what do you see",
                "describe my screen",
                "what's open",
                "what am i looking at",
            )
            if any(p in lower for p in _screen_phrases):
                try:
                    from sensus.vision.screen import describe_screen

                    description = describe_screen()
                except Exception:
                    description = "I wasn't able to capture the screen right now."
                self.conversation_history.append(
                    {"role": "user", "content": text})
                self.conversation_history.append(
                    {"role": "assistant", "content": description})
                return OrchestratorTurnResult(
                    user_text=text,
                    assistant_raw=description,
                    is_tool_json=False,
                    tool_payload=None,
                    reply_text=description,
                )

        sys_prompt = self.system_prompt + \
            (("\n\n" + extra_system_suffix.strip()) if extra_system_suffix else "")

        k, b, m_default, t = load_orchestrator_env()
        model = self.model or m_default

        from openai import OpenAI

        client = OpenAI(base_url=b, api_key=k, timeout=t)

        self.conversation_history.append({"role": "user", "content": text})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_prompt}, *self.conversation_history]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        content = (msg.content or "").strip()

        is_tool, payload, normalized = parse_assistant_output(
            content,
            native_tool_calls=tool_calls,
        )

        self.conversation_history.append(
            {"role": "assistant", "content": normalized})

        if is_tool and payload is not None:
            routed_agent = _routed_agent_from_suffix(extra_system_suffix)
            if routed_agent == "conversation":
                agent_name = str(payload.get("agent", "")).strip().lower()
                if agent_name in ("browser", "shortcuts", "desktop", "shell", "coding"):
                    logger.info(
                        "[ORCHESTRATOR] intercepted tool_call from conversation route, "
                        "re-dispatching to %s",
                        agent_name,
                    )
                    agent_reply = dispatch_tool_call(text, payload)
                    reply_text = (
                        agent_reply
                        if agent_reply
                        else tool_call_to_speech(payload)
                    )
                else:
                    c = (content or "").strip()
                    reply_text = c if c else "Sure — I'm here if you need anything."
            else:
                agent_reply = dispatch_tool_call(text, payload)
                reply_text = agent_reply if agent_reply else tool_call_to_speech(
                    payload)
        else:
            reply_text = normalized

        return OrchestratorTurnResult(
            user_text=text,
            assistant_raw=normalized,
            is_tool_json=is_tool,
            tool_payload=payload,
            reply_text=reply_text,
        )

    def iter_turns_text(self) -> Iterator[tuple[str, str]]:
        """Yield (role, content) for debugging."""
        for m in self.conversation_history:
            yield m["role"], m["content"]


def main() -> None:
    """CLI: ``--help``, optional one-shot prompt, or REPL when no prompt."""
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=(
            "Sensus orchestrator (Phase 1.5). In-memory conversation history only lasts for this "
            "process. Each separate command starts fresh; use REPL (no args) for multi-turn memory."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help='Single user utterance. Example: %(prog)s "what is my name"',
    )
    args = parser.parse_args()
    orch = Orchestrator()
    arg = " ".join(args.prompt).strip()
    if arg:
        r = orch.turn(arg)
        print("--- assistant (raw) ---")
        print(r.assistant_raw)
        print("--- parsed ---")
        print(f"tool_json={r.is_tool_json}")
        if r.tool_payload:
            print(json.dumps(r.tool_payload, indent=2))
        return

    print(
        "REPL: conversation_memory lasts until you quit (empty line). Ctrl+D EOF.",
        file=sys.stderr,
    )
    try:
        while True:
            line = input("you> ").strip()
            if not line:
                break
            out = orch.turn(line)
            print("sensus>", out.reply_text[:4000])
            print()
    except EOFError:
        pass


if __name__ == "__main__":
    main()
