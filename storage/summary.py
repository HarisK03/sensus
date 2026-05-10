"""One-line session summaries via DeepSeek (OpenAI-compatible API)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def load_deepseek_summary_env() -> tuple[Optional[str], str, str, float]:
    """Return (api_key_or_none, base_url, model, timeout_sec)."""
    _try_load_dotenv()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or None
    base = (
        os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        .strip()
        .rstrip("/")
    )
    model = (
        os.environ.get("DEEPSEEK_SUMMARY_MODEL", "deepseek-chat").strip()
        or "deepseek-chat"
    )
    timeout_sec = float(os.environ.get("DEEPSEEK_SUMMARY_TIMEOUT_SEC", "45"))
    return key, base, model, timeout_sec


def summarize_transcript(transcript_lines: list[tuple[str, str]]) -> str:
    """
    Produce a single-sentence summary from (role, content) pairs.

    Uses DeepSeek when ``DEEPSEEK_API_KEY`` is set; otherwise a short heuristic title.
    """
    chunks: list[str] = []
    for role, content in transcript_lines:
        r = (role or "").strip().lower()
        c = (content or "").strip()
        if not c:
            continue
        label = "User" if r == "user" else "Assistant"
        chunks.append(f"{label}: {c}")
    joined = "\n".join(chunks).strip()
    if not joined:
        return "Empty session"

    key, base, model, timeout_sec = load_deepseek_summary_env()
    if key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base, timeout=timeout_sec)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the conversation in exactly one clear English sentence "
                            "(max 200 characters). No quotes, no preamble."
                        ),
                    },
                    {"role": "user", "content": joined[:12000]},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            raw = (completion.choices[0].message.content or "").strip()
            one = raw.split("\n")[0].strip()
            one = one.strip("\"'")
            if len(one) > 220:
                one = one[:217] + "…"
            return one or fallback_title(transcript_lines)
        except Exception:
            logger.exception("DeepSeek session summary failed; using fallback title")

    return fallback_title(transcript_lines)


def fallback_title(transcript_lines: list[tuple[str, str]]) -> str:
    """First meaningful user line, truncated."""
    for role, content in transcript_lines:
        if (role or "").strip().lower() != "user":
            continue
        c = (content or "").strip().replace("\n", " ")
        if not c:
            continue
        if len(c) > 120:
            return c[:117] + "…"
        return c
    return "Session"
