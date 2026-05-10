"""
Vision module — describes screen content for blind users.
Used by:
  1. main.py when user says "what's on my screen"
  2. browser.py after every browser action
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    from openai import APIConnectionError, APIError, APITimeoutError

    _OPENAI_HTTP_LIKE_ERRORS = (APIError, APIConnectionError, APITimeoutError)
except ImportError:
    _OPENAI_HTTP_LIKE_ERRORS = ()

logger = logging.getLogger(__name__)

DEFAULT_SCREENSHOT_PATH = "/tmp/sensus_screen.png"
DEFAULT_VISION_MODEL = "google/gemma-3-12b-it"
FALLBACK_MESSAGE = "I wasn't able to capture the screen right now."

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"

_DEFAULT_VISION_PROMPT = (
    "Describe what is on this screen for a blind user. "
    "Be specific about: what application or website is open, "
    "key text content visible, interactive elements like buttons and inputs, "
    "any popups or dialogs, and the overall current state. "
    "Be concise — 2-4 sentences maximum. Speak directly, no preamble."
)


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _check_screenshot_tools() -> Optional[str]:
    for tool in ("scrot", "gnome-screenshot", "import"):
        path = shutil.which(tool)
        if path:
            return tool
    logger.warning(
        "[VISION] No screenshot tool found. Install scrot: sudo apt install scrot"
    )
    return None


_SCREENSHOT_TOOL = _check_screenshot_tools()


@lru_cache(maxsize=1)
def _load_vision_client_env() -> tuple[str, str, float]:
    """Featherless API + base URL + HTTP timeout (dotenv loaded here)."""
    _try_load_dotenv()
    key = os.environ.get("FEATHERLESS_AI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("FEATHERLESS_AI_API_KEY is not set.")
    base = os.environ.get("FEATHERLESS_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    timeout_sec = float(os.environ.get("FEATHERLESS_ORCHESTRATOR_TIMEOUT_SEC", "120"))
    return key, base, timeout_sec


def _vision_model() -> str:
    _try_load_dotenv()
    m = os.environ.get("FEATHERLESS_VISION_MODEL", DEFAULT_VISION_MODEL).strip()
    return m or DEFAULT_VISION_MODEL


def _vision_models_to_try() -> list[str]:
    """Primary vision model plus optional fallbacks (comma-separated or FALLBACK key)."""
    _try_load_dotenv()
    primary = (
        os.environ.get("FEATHERLESS_VISION_MODEL", DEFAULT_VISION_MODEL).strip()
        or DEFAULT_VISION_MODEL
    )
    fb1 = os.environ.get("FEATHERLESS_VISION_MODEL_FALLBACK", "").strip()
    fb_list = [
        p.strip()
        for p in os.environ.get("FEATHERLESS_VISION_MODEL_FALLBACKS", "").split(",")
        if p.strip()
    ]
    seen: set[str] = set()
    out: list[str] = []
    for m in (primary, fb1, *fb_list):
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _vision_api_error_is_model_access(e: BaseException) -> bool:
    """True when retrying with a different model may help (gated / 403 / etc.)."""
    s = str(e).lower()
    if "403" in s or "gated" in s or "oauth" in s or "permission" in s:
        return True
    err = getattr(e, "body", None)
    if isinstance(err, dict):
        msg = str(err.get("error", {}).get("message", "")).lower()
        if "gated" in msg or "403" in msg or "oauth" in msg:
            return True
    return False


def _read_image_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        logger.error("[VISION] could not read image at %s: %s", path, e)
        return None


def _bytes_to_vision_base64(raw: bytes) -> Optional[str]:
    """Downscale wide screenshots for lower latency; then base64-encode as PNG."""
    _try_load_dotenv()
    try:
        from PIL import Image

        img = Image.open(BytesIO(raw))
        img.load()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGBA")
        w, h = img.size
        max_w = max(256, int(os.environ.get("FEATHERLESS_VISION_MAX_WIDTH", "1280")))
        if w > max_w:
            new_h = max(1, int(h * (max_w / w)))
            img = img.resize((max_w, new_h), Image.Resampling.LANCZOS)
            logger.debug(
                "[VISION] resized image for API: %sx%s → %sx%s",
                w,
                h,
                max_w,
                new_h,
            )
        out = BytesIO()
        # Vision APIs expect PNG-friendly RGB
        if img.mode == "RGBA":
            rgb = Image.new("RGB", img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[3])
            img = rgb
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out, format="PNG", optimize=True)
        return base64.standard_b64encode(out.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning("[VISION] Pillow resize failed, using raw bytes: %s", e)
        try:
            return base64.standard_b64encode(raw).decode("ascii")
        except Exception:
            return None


def _read_image_base64(path: str) -> Optional[str]:
    raw = _read_image_bytes(path)
    if raw is None:
        return None
    return _bytes_to_vision_base64(raw)


def _stderr_text(result: subprocess.CompletedProcess) -> str:
    err = result.stderr
    if err is None:
        return ""
    if isinstance(err, bytes):
        return err.decode(errors="replace")
    return str(err)


def _subprocess_display_env() -> dict[str, str]:
    return {**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")}


def _screenshot_output_valid(path: str) -> bool:
    try:
        if not os.path.isfile(path):
            logger.error("[VISION] screenshot file missing: %s", path)
            return False
        if os.path.getsize(path) <= 0:
            logger.error("[VISION] screenshot file empty: %s", path)
            return False
    except OSError as e:
        logger.error("[VISION] could not stat screenshot %s: %s", path, e)
        return False
    return True


def _capture_desktop_screenshot(path: str) -> bool:
    """Return True if a non-empty image was written to ``path``."""
    attempts: list[tuple[str, list[str]]] = []
    if shutil.which("scrot"):
        attempts.append(("scrot", ["scrot", path]))
    if shutil.which("gnome-screenshot"):
        attempts.append(("gnome-screenshot", ["gnome-screenshot", "-f", path]))
    if shutil.which("import"):
        attempts.append(("import", ["import", "-window", "root", path]))

    if not attempts:
        logger.error("[VISION] no screenshot tools on PATH (install scrot)")
        return False

    if _SCREENSHOT_TOOL:
        attempts.sort(key=lambda x: 0 if x[0] == _SCREENSHOT_TOOL else 1)

    env = _subprocess_display_env()
    for name, cmd in attempts:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                timeout=10,
                env=env,
            )
            if result.returncode == 0:
                if _screenshot_output_valid(path):
                    logger.info("[VISION] screenshot via %s succeeded", name)
                    return True
                logger.error(
                    "[VISION] %s reported success but output invalid (VMware/headless X?)",
                    name,
                )
                continue
            logger.error("[VISION] %s failed: %s", name, _stderr_text(result))
        except FileNotFoundError:
            logger.error("[VISION] %s not installed", name)
        except subprocess.TimeoutExpired:
            logger.error("[VISION] %s timed out", name)

    logger.error("[VISION] all screenshot tools failed")
    return False


def _completion_to_debug_payload(completion: Any) -> Any:
    try:
        if hasattr(completion, "model_dump"):
            return completion.model_dump()
        if hasattr(completion, "dict"):
            return completion.dict()  # type: ignore[no-any-return]
    except Exception:
        pass
    return repr(completion)


def _call_vision_api(base64_png: str, prompt: str) -> str:
    import json

    k, b, t = _load_vision_client_env()
    models = _vision_models_to_try()

    from openai import OpenAI

    client = OpenAI(base_url=b, api_key=k, timeout=t)

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_png}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    last_err: Optional[BaseException] = None
    for mi, model in enumerate(models):
        logger.info(
            "[VISION] API call: model=%r base_url=%r image_b64_len=%s",
            model,
            b,
            len(base64_png),
        )
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=2048,
            )
            logger.debug(
                "[VISION] raw completion: %s",
                json.dumps(
                    _completion_to_debug_payload(completion), default=str
                )[:8000],
            )

            choice = completion.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            raw_msg = choice.message
            raw_content = raw_msg.content if raw_msg is not None else None
            logger.debug(
                "[VISION] finish_reason=%r raw_message_content=%r",
                finish_reason,
                raw_content,
            )

            result = (raw_content or "").strip()
            logger.info("[VISION] response: %s chars", len(result))
            if not result:
                if finish_reason == "length":
                    logger.warning(
                        "[VISION] model exhausted tokens on reasoning before producing output — "
                        "increase max_tokens or use a non-reasoning model"
                    )
                msg_dump: dict[str, Any] = {}
                try:
                    if raw_msg is not None and hasattr(raw_msg, "model_dump"):
                        msg_dump = raw_msg.model_dump()
                    elif raw_msg is not None and hasattr(raw_msg, "dict"):
                        msg_dump = raw_msg.dict()  # type: ignore[assignment]
                    else:
                        msg_dump = {"repr": repr(raw_msg)}
                except Exception as dump_exc:
                    msg_dump = {"dump_error": str(dump_exc)}
                logger.warning(
                    "[VISION] empty content after strip; finish_reason=%r message=%s",
                    finish_reason,
                    json.dumps(msg_dump, default=str),
                )
                return FALLBACK_MESSAGE
            return result
        except Exception as e:
            last_err = e
            if (
                _vision_api_error_is_model_access(e)
                and mi < len(models) - 1
            ):
                logger.warning(
                    "[VISION] model %r failed (%s); retrying with fallback",
                    model,
                    e,
                )
                continue
            if _OPENAI_HTTP_LIKE_ERRORS and isinstance(
                e, _OPENAI_HTTP_LIKE_ERRORS
            ):
                logger.error(
                    "[VISION] OpenAI SDK error (model %r): %s",
                    model,
                    e,
                    exc_info=True,
                )
                return FALLBACK_MESSAGE
            if httpx is not None and isinstance(
                e, (httpx.HTTPError, httpx.TimeoutException, httpx.RequestError)
            ):
                logger.error(
                    "[VISION] httpx error (model %r): %s",
                    model,
                    e,
                    exc_info=True,
                )
                return FALLBACK_MESSAGE
            logger.error(
                "[VISION] unexpected API error (model %r): %s",
                model,
                e,
                exc_info=True,
            )
            return FALLBACK_MESSAGE

    if last_err:
        logger.error("[VISION] all vision models failed: %s", last_err)
    return FALLBACK_MESSAGE


def describe_screen(image_path: str | None = None, prompt_override: str | None = None) -> str:
    path = (image_path or "").strip() or DEFAULT_SCREENSHOT_PATH
    prompt = (
        prompt_override
        if (prompt_override or "").strip()
        else _DEFAULT_VISION_PROMPT
    )

    if image_path is None:
        if not _capture_desktop_screenshot(path):
            return FALLBACK_MESSAGE

    b64 = _read_image_base64(path)
    if not b64:
        logger.error("[VISION] could not read image at %s", path)
        return FALLBACK_MESSAGE

    return _call_vision_api(b64, prompt)


def describe_image(image_path: str, prompt: str) -> str:
    path = (image_path or "").strip()
    if not path:
        logger.error("[VISION] describe_image: empty path")
        return FALLBACK_MESSAGE

    b64 = _read_image_base64(path)
    if not b64:
        logger.error("[VISION] could not read image at %s", path)
        return FALLBACK_MESSAGE

    p = (prompt or "").strip() or _DEFAULT_VISION_PROMPT
    return _call_vision_api(b64, p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(describe_screen())
