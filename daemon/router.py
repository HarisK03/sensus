"""
Phase 1.4 — Intent routing via Featherless (OpenAI-compatible API).

Classifies a user transcript into: browser, desktop, shell, shortcuts, coding, conversation.
Uses a single fast instruct model (default: google/gemma-4-31B-it). No DeepSeek escalation.

Environment (e.g. ``sensus/.env``):
  FEATHERLESS_AI_API_KEY — required
  FEATHERLESS_BASE_URL — optional, default https://api.featherless.ai/v1
  FEATHERLESS_ORCHESTRATOR_MODEL — optional, default google/gemma-4-31B-it (same as main / browser planner)
  FEATHERLESS_ROUTER_TIMEOUT_SEC — optional, default 30 (HTTP timeout)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AGENTS = frozenset({"browser", "desktop", "shell",
                   "shortcuts", "coding", "conversation"})

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_MODEL = "google/gemma-4-31B-it"

SYSTEM_PROMPT = """Classify the user intent as one of: browser, desktop, shell, shortcuts, coding, conversation.

browser — web/search/page control in a browser.
desktop — native apps, windows, pointing/typing outside the browser.
shell — terminal commands or running programs as shell commands.
shortcuts — volume (mute, up/down, percent), brightness, Wi‑Fi / Bluetooth / airplane mode (nmcli, bluetoothctl), media keys, Spotify via MPRIS, launching apps (Files, Settings), "in <app> play …".
coding — implement / write / build / create a program or script from a description (e.g. Python game, hard-coded word, small CLI tool).
conversation — everything else (chat, facts, clarifications).

Reply in JSON only:
{"agent":"<one of the six>","confidence":0.0}

confidence is your certainty from 0.0 to 1.0. No markdown, no other keys."""


@dataclass(frozen=True)
class RouteResult:
    """Parsed routing decision from the classifier."""

    agent: str
    confidence: float
    raw_text: str

    @property
    def needs_escalation(self) -> bool:
        """Phase 1.4 originally escalated when confidence < 0.8; DeepSeek is optional."""
        return self.confidence < 0.8


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # sensus/sensus/.env when this file is sensus/sensus/daemon/router.py
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def load_featherless_env() -> tuple[str, str, str, float]:
    """
    Return (api_key, base_url, model, timeout_sec).
    Cached so repeated calls do not re-read disk.
    """
    _try_load_dotenv()
    key = os.environ.get("FEATHERLESS_AI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FEATHERLESS_AI_API_KEY is not set. Add it to sensus/.env or the environment."
        )
    base = os.environ.get("FEATHERLESS_BASE_URL",
                          DEFAULT_BASE_URL).strip().rstrip("/")
    model = (
        os.environ.get("FEATHERLESS_ORCHESTRATOR_MODEL", DEFAULT_MODEL).strip()
        or DEFAULT_MODEL
    )
    timeout_sec = float(os.environ.get("FEATHERLESS_ROUTER_TIMEOUT_SEC", "30"))
    return key, base, model, timeout_sec


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _parse_route_json(content: str) -> tuple[str, float]:
    payload = _strip_json_fence(content)
    data = json.loads(payload)
    agent = str(data.get("agent", "")).strip().lower()
    conf = data.get("confidence", 0.0)
    try:
        confidence = float(conf)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    if agent not in AGENTS:
        logger.warning(
            "Router returned unknown agent %r; mapping to conversation", agent)
        agent = "conversation"
        confidence = min(confidence, 0.7)
    return agent, confidence


def _maybe_boost_shell_route(
    transcript: str, agent: str, confidence: float
) -> tuple[str, float]:
    """
    Override flaky LLM routing when the utterance clearly implies terminal/shell work.

    Fixes follow-ups like "install the latest one" (conversation) or "file in downloads"
    (desktop) when the user means apt/deb or listing ~/Downloads.
    """
    low = (transcript or "").strip().lower()
    if not low:
        return agent, confidence

    # Install .deb / apt / vague follow-up after listing packages ("latest one", "that deb").
    if re.search(r"\binstall\b", low) and re.search(
        r"\b(latest|newest|last|deb|\.deb|package|downloads|"
        r"that\s+one|the\s+one|it|there|from\s+downloads)\b",
        low,
    ):
        return "shell", max(confidence, 0.9)

    if re.search(r"\b(want|need|would like|gonna|going to)\s+to\s+install\b", low) and re.search(
        r"\b(latest|newest|deb|\.deb|that|the\s+one|downloads)\b",
        low,
    ):
        return "shell", max(confidence, 0.9)

    # List or ask what's in Downloads (not "download a YouTube video").
    if re.search(r"\bdownloads\b", low) or re.search(
        r"\bdownload\s+folder\b", low
    ):
        if re.search(
            r"\b(files?|folders?|list|what(?:'s|\s+is|s)?|show|tell\s+me|"
            r"anything|stuff|file\s+in|in\s+my|in\s+the)\b",
            low,
        ):
            return "shell", max(confidence, 0.88)

    return agent, confidence


def classify_intent(
    transcript: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: Optional[float] = None,
) -> RouteResult:
    """
    Send ``transcript`` to Featherless and return ``RouteResult``.

    Uses JSON mode when the server supports it; falls back to parsing message content.
    Target latency for hackathon: keep transcripts short and model small; tune timeout via env.
    """
    text = (transcript or "").strip()
    if not text:
        return RouteResult(agent="conversation", confidence=0.0, raw_text="{}")

    k, b, m, t_default = load_featherless_env()
    k = api_key or k
    b = (base_url or b).rstrip("/")
    m = model or m
    t = timeout_sec if timeout_sec is not None else t_default

    from openai import OpenAI

    client = OpenAI(base_url=b, api_key=k, timeout=t)

    user_content = f"User said:\n{text}"

    raw_text = ""
    kwargs = dict(
        model=m,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=128,
    )
    try:
        completion = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as first:
        logger.debug(
            "Router JSON mode failed (%s); retrying without response_format", first)
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception:
            logger.exception("Featherless router call failed")
            raise

    try:
        choice = completion.choices[0]
        raw_text = (choice.message.content or "").strip()
    except (AttributeError, IndexError) as e:
        logger.exception("Unexpected completion shape: %s", e)
        raise RuntimeError("Invalid response from Featherless router") from e

    try:
        agent, confidence = _parse_route_json(raw_text)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            "Could not parse router JSON: %s — raw: %s", e, raw_text[:200])
        return RouteResult(agent="conversation", confidence=0.3, raw_text=raw_text)

    agent, confidence = _maybe_boost_shell_route(text, agent, confidence)
    return RouteResult(agent=agent, confidence=confidence, raw_text=raw_text)


def main() -> None:
    """Smoke test: ``python -m sensus.daemon.router`` with optional argv transcript."""
    import sys

    logging.basicConfig(level=logging.INFO)
    phrase = " ".join(sys.argv[1:]).strip(
    ) or "open Firefox and search for headphones"
    print(f"Transcript: {phrase!r}")
    r = classify_intent(phrase)
    print(
        f"agent={r.agent!r} confidence={r.confidence} escalation={r.needs_escalation}")
    print(f"raw: {r.raw_text}")


if __name__ == "__main__":
    main()
