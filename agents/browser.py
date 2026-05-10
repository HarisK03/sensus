"""Browser agent — Playwright Firefox (IBM Equal Access checker present but disabled).

Maintains a singleton Firefox browser/page across calls. Each turn:

1. Gathers fresh page context (URL, title, Playwright DOM elements, merged map, popup text).
   IBM Equal Access is currently disabled; ``ibm_elements`` is always empty.
2. Asks the orchestrator model on Featherless for a **sequence** of actions (JSON array).
3. Executes actions in order until one fails or the list ends.
4. Describes results via a text-only DOM summary when possible, or a vision model on fallback.

Environment is shared with ``sensus.daemon.main``:
  FEATHERLESS_AI_API_KEY — required
  FEATHERLESS_BASE_URL — optional
  FEATHERLESS_ORCHESTRATOR_MODEL — optional (planner; default google/gemma-4-31B-it)
  FEATHERLESS_VISION_MODEL — optional (vision fallback screenshots; default google/gemma-3-12b-it)
  FEATHERLESS_ORCHESTRATOR_TIMEOUT_SEC — optional (default 120)
  SENSUS_BROWSER_FULLSCREEN — optional (default on): set ``0`` for large window instead of ``-fullscreen``.
  SENSUS_BROWSER_WINDOW_WIDTH / HEIGHT — optional when fullscreen is off (default 1920x1080).
  SENSUS_BROWSER_DOWNLOAD_DIR — optional; folder for Playwright saves (default ``~/Downloads``).
  SENSUS_BROWSER_DOWNLOAD_TIMEOUT_MS — optional; max wait for a download after click/navigate (default 120000).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from sensus.agents.browser_downloads import (
    default_download_timeout_ms,
    persist_playwright_download,
)

logger = logging.getLogger(__name__)

BROWSER_VISION_SCREENSHOT_PATH = "/tmp/sensus_browser.png"
# Default large window; set SENSUS_BROWSER_FULLSCREEN=1 for Firefox -fullscreen.
_DEFAULT_FF_WIDTH = os.environ.get("SENSUS_BROWSER_WINDOW_WIDTH", "1920")
_DEFAULT_FF_HEIGHT = os.environ.get("SENSUS_BROWSER_WINDOW_HEIGHT", "1080")
MAX_IBM_ELEMENTS = 60
MAX_DOM_ELEMENTS = 60
MAX_MERGED_ELEMENTS = 80
MAX_ACTIONS_PER_TURN = 8
CHECKER_TIMEOUT_SEC = 20

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
DEFAULT_ORCHESTRATOR_MODEL = "google/gemma-4-31B-it"

FALLBACK_NAVIGATE_GOOGLE = [
    {"action": "navigate", "url": "https://google.com"}]

# Persistent background event loop. All Playwright work runs on THIS loop so the
# browser survives across `dispatch_tool_call` invocations (each of which spins up
# its own short-lived `asyncio.run` loop in the calling thread).
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_started = threading.Event()
_loop_init_lock = threading.Lock()

_playwright: Any = None
_browser: Any = None
_page: Any = None
# bound to the persistent loop on first use
_lock: Optional[asyncio.Lock] = None


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def _load_llm_env() -> tuple[str, str, str, float]:
    """Match the env contract used by ``sensus.daemon.main.load_orchestrator_env``."""
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


def _normalize_element_item(item: dict) -> dict:
    return {
        "label": str(item.get("label") or ""),
        "role": str(item.get("role") or ""),
        "selector": str(item.get("selector") or "").strip(),
        "tag": str(item.get("tag") or ""),
    }


def _merge_ibm_and_dom(ibm: list[dict], dom: list[dict]) -> list[dict]:
    """Dedupe by selector; IBM order first, then DOM-only selectors; cap at MAX_MERGED_ELEMENTS."""
    by_sel: dict[str, dict] = {}
    for item in ibm:
        norm = _normalize_element_item(item)
        sel = norm["selector"]
        if not sel or sel in by_sel:
            continue
        by_sel[sel] = norm
        if len(by_sel) >= MAX_MERGED_ELEMENTS:
            return list(by_sel.values())
    for item in dom:
        norm = _normalize_element_item(item)
        sel = norm["selector"]
        if not sel or sel in by_sel:
            continue
        by_sel[sel] = norm
        if len(by_sel) >= MAX_MERGED_ELEMENTS:
            break
    return list(by_sel.values())[:MAX_MERGED_ELEMENTS]


def _browser_is_alive(browser: Any) -> bool:
    if browser is None:
        return False
    is_connected = getattr(browser, "is_connected", None)
    if is_connected is None:
        return True
    try:
        return bool(is_connected())
    except Exception:
        return False


def _ensure_persistent_loop() -> asyncio.AbstractEventLoop:
    """Start (or return) the dedicated background event loop for browser work."""
    global _loop, _loop_thread

    with _loop_init_lock:
        if _loop is not None and _loop.is_running():
            return _loop

        loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            _loop_started.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    logger.exception("Failed to close persistent browser loop")

        _loop = loop
        _loop_thread = threading.Thread(
            target=_runner, name="sensus-browser-loop", daemon=True
        )
        _loop_started.clear()
        _loop_thread.start()
        if not _loop_started.wait(timeout=5):
            raise RuntimeError("Persistent browser event loop failed to start")
        return loop


def _firefox_launch_args() -> list[str]:
    """Fullscreen by default (voice UX); set SENSUS_BROWSER_FULLSCREEN=0 for large window instead."""
    off = os.environ.get("SENSUS_BROWSER_FULLSCREEN", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    )
    if not off:
        return ["-fullscreen"]
    w = _DEFAULT_FF_WIDTH.strip() or "1920"
    h = _DEFAULT_FF_HEIGHT.strip() or "1080"
    return ["-width", w, "-height", h]


def _amazon_view_cart_url(current_url: str) -> str:
    """Same Amazon host as current page → cart view (avoids stuck #nav-cart clicks)."""
    p = urlparse(current_url or "")
    scheme = p.scheme if p.scheme in ("http", "https") else "https"
    host = (p.netloc or "").strip() or "www.amazon.com"
    return f"{scheme}://{host}/gp/cart/view.html?ref_=nav_cart"


# Try foreground overlays first so unscoped `role=link[...]` does not match the dimmed page.
_OVERLAY_ROOT_SELECTORS = (
    '[role="dialog"]',
    '[aria-modal="true"]',
    ".modal-dialog",
    ".modal",
)


async def _click_single_locator(loc: Any, *, timeout_ms: int = 10000) -> None:
    """Scroll into view, then normal → force → JS click on an existing Locator."""
    try:
        await loc.scroll_into_view_if_needed(timeout=min(5000, timeout_ms))
    except Exception:
        logger.debug("scroll_into_view_if_needed failed", exc_info=True)
    try:
        await loc.click(timeout=timeout_ms)
        return
    except Exception as first:
        logger.debug("[BROWSER] normal click failed: %s", first)
    try:
        await loc.click(timeout=5000, force=True)
        logger.info("[BROWSER] used force click")
        return
    except Exception as second:
        logger.debug("[BROWSER] force click failed: %s", second)
    await loc.evaluate(
        """el => {
        el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
        el.click();
    }"""
    )
    logger.info("[BROWSER] used JS click")


async def _reliable_click(page: Any, selector: str, *, timeout_ms: int = 10000) -> None:
    """Scroll into view, then normal → force → JS click.

    If ``selector`` has no `` >> `` chain, we try the same selector **inside** each
    visible dialog / modal root first (topmost last in DOM), then fall back to the
    global locator. This avoids clicking homepage controls hidden behind a modal.
    """
    sel = (selector or "").strip()
    if not sel:
        raise ValueError("empty selector")
    if " >> " in sel:
        await _click_single_locator(
            page.locator(sel).first, timeout_ms=timeout_ms
        )
        return

    overlay_budget = min(timeout_ms, 8500)
    for scope_sel in _OVERLAY_ROOT_SELECTORS:
        try:
            roots = page.locator(scope_sel)
            n = await roots.count()
            for idx in range(n - 1, -1, -1):
                root = roots.nth(idx)
                try:
                    if not await root.is_visible():
                        continue
                except Exception:
                    continue
                inner = root.locator(sel)
                try:
                    ic = await inner.count()
                except Exception:
                    continue
                if ic == 0:
                    continue
                target = inner.first
                try:
                    if not await target.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    await _click_single_locator(
                        target, timeout_ms=overlay_budget
                    )
                    logger.info(
                        "[BROWSER] overlay-scoped click: %s #%s → %s",
                        scope_sel,
                        idx,
                        sel[:80],
                    )
                    return
                except Exception:
                    continue
        except Exception:
            continue

    await _click_single_locator(
        page.locator(sel).first, timeout_ms=timeout_ms
    )


async def _ensure_browser() -> Any:
    """Launch Firefox once, then reuse the same window/page across calls."""
    global _playwright, _browser, _page, _lock

    from playwright.async_api import async_playwright

    if _lock is None:
        _lock = asyncio.Lock()

    if _playwright is None:
        _playwright = await async_playwright().start()

    if not _browser_is_alive(_browser):
        _page = None
        ff_args = _firefox_launch_args()
        _browser = await _playwright.firefox.launch(
            headless=False,
            args=ff_args,
        )
        if "-fullscreen" in ff_args:
            logger.info(
                "[BROWSER] Firefox launched fullscreen (SENSUS_BROWSER_FULLSCREEN=0 for windowed)")
        else:
            logger.info(
                "[BROWSER] Firefox launched window %sx%s",
                _DEFAULT_FF_WIDTH,
                _DEFAULT_FF_HEIGHT,
            )

    if _page is None or _page.is_closed():
        _page = await _browser.new_page(no_viewport=True)
        try:
            await _page.goto("about:blank")
        except Exception:
            logger.debug(
                "Initial about:blank navigation failed", exc_info=True)

    return _page


def _run_a11y_checker(url: str) -> list[dict]:
    """Run the Node sidecar synchronously; return [] on any failure."""
    if not url or url == "about:blank":
        return []
    project_root = Path(__file__).resolve().parents[2]
    checker_js = project_root / "accessibility" / "checker.js"
    if not checker_js.is_file():
        logger.warning(
            "Accessibility sidecar not found at %s — run `cd accessibility && npm install`; "
            "continuing with empty element list.",
            checker_js,
        )
        return []
    try:
        result = subprocess.run(
            ["node", "accessibility/checker.js", url],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=CHECKER_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Accessibility checker timed out for %s", url)
        return []
    except FileNotFoundError:
        logger.warning(
            "`node` is not installed; accessibility checker skipped.")
        return []
    except Exception:
        logger.exception("Accessibility checker failed unexpectedly")
        return []

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0 and stderr:
        logger.warning(
            "Accessibility checker exited %s: %s",
            result.returncode,
            stderr[:200],
        )
    if not stdout:
        logger.warning(
            "Accessibility checker returned empty stdout for %s", url)
        return []
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        logger.warning(
            "Accessibility checker returned non-JSON stdout (truncated): %s",
            stdout[:200],
        )
        return []
    if not isinstance(data, list):
        logger.warning("Accessibility checker JSON was not a list for %s", url)
        return []
    logger.info(
        "[BROWSER] IBM Equal Access found %s elements on %s", len(data), url)
    return data


async def _extract_elements_playwright(page: Any) -> list[dict]:
    """Playwright-native element extraction.

    Prefer stable Playwright ``role=…[name=/…/i]`` selectors for interactive
    controls without ids so ``locator().first`` does not grab the wrong ``a[href]``
    (e.g. footer links before homepage tiles on sites like lichess.org).
    """
    elements = await page.evaluate(
        """() => {
        const LABEL_MAX = 120;

        function normalizeLabel(s) {
            if (!s) return '';
            return s.replace(/\\s+/g, ' ').trim().slice(0, LABEL_MAX);
        }

        function escapeRx(s) {
            let o = '';
            const special = '.*+?^${}()|[]\\\\';
            for (let i = 0; i < s.length; i++) {
                const c = s[i];
                o += special.indexOf(c) >= 0 ? '\\\\' + c : c;
            }
            return o;
        }

        function ariaRole(el) {
            const explicit = el.getAttribute('role');
            if (explicit) return explicit;
            const tag = el.tagName.toLowerCase();
            if (tag === 'a' && el.hasAttribute('href')) return 'link';
            if (tag === 'button') return 'button';
            if (tag === 'input') {
                const t = (el.getAttribute('type') || 'text').toLowerCase();
                if (t === 'submit' || t === 'button' || t === 'reset') return 'button';
                if (t === 'search') return 'searchbox';
                if (t === 'checkbox') return 'checkbox';
            }
            return null;
        }

        function accessibleName(el) {
            const al = el.getAttribute('aria-label');
            if (al && al.trim()) return normalizeLabel(al);
            const ph = el.placeholder;
            if (ph && ph.trim()) return normalizeLabel(ph);
            const tn = el.textContent || '';
            const line = tn.split(/\\n/)[0];
            return normalizeLabel(line);
        }

        function roleNameSelector(ariaR, name) {
            if (!ariaR || !name || name.length < 2) return null;
            const rx = escapeRx(name);
            return 'role=' + ariaR + '[name=/' + rx + '/i]';
        }

        function makeEntry(el, sel, labelPrefix, scopePrefix) {
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return null;
            const st = window.getComputedStyle(el);
            if (st.display === 'none' || st.visibility === 'hidden') return null;
            const op = parseFloat(st.opacity || '1');
            if (isNaN(op) || op < 0.05) return null;
            const tag = el.tagName.toLowerCase();
            let selector;
            if (el.id) {
                try { selector = '#' + CSS.escape(el.id); } catch (e) { selector = '#' + el.id; }
            } else if (el.name && (tag === 'input' || tag === 'select' || tag === 'textarea')) {
                selector = '[name=' + JSON.stringify(el.name) + ']';
            } else {
                const ar = ariaRole(el);
                const nm = accessibleName(el);
                const rns = roleNameSelector(ar, nm);
                selector = rns || sel;
            }
            if (scopePrefix) {
                selector = scopePrefix + selector;
            }
            let label = (
                el.getAttribute('aria-label')
                || el.placeholder
                || normalizeLabel((el.textContent || '').split(/\\n/)[0])
                || el.name
                || ''
            );
            if (labelPrefix && label) {
                label = labelPrefix + label;
            } else if (labelPrefix) {
                label = labelPrefix.trim();
            }
            return {
                selector: selector,
                label: String(label).slice(0, LABEL_MAX),
                role: el.getAttribute('role') || tag,
                tag: tag
            };
        }

        const selectors = [
            'input[type="search"]', 'input[type="text"]',
            'button[type="submit"]', 'button[type="button"]',
            'a[href]', '[role="button"]', '[role="searchbox"]',
            'select', 'textarea', 'form'
        ];

        const modalSel = [
            '[role="dialog"]', '[aria-modal="true"]', '.modal',
            '.modal-dialog'
        ];
        const modalRootsAll = [];
        for (const ms of modalSel) {
            for (const el of document.querySelectorAll(ms)) {
                const rect = el.getBoundingClientRect();
                const st = window.getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden') continue;
                if (rect.width < 64 || rect.height < 64) continue;
                modalRootsAll.push(el);
            }
        }
        const topModalRoots = [];
        for (const r of modalRootsAll) {
            let inner = false;
            for (const o of modalRootsAll) {
                if (r !== o && o.contains(r)) { inner = true; break; }
            }
            if (!inner) topModalRoots.push(r);
        }

        function scopeForRoot(root) {
            if (!root || !root.getAttribute) return 'role=dialog >> ';
            if (root.id) {
                try { return '#' + CSS.escape(root.id) + ' >> '; } catch (e) { return '#' + root.id + ' >> '; }
            }
            if (root.getAttribute('role') === 'dialog' ||
                root.getAttribute('aria-modal') === 'true') {
                return 'role=dialog >> ';
            }
            try {
                if (root.matches && root.matches('.modal-dialog')) {
                    return '.modal-dialog >> ';
                }
                if (root.matches && root.matches('.modal')) {
                    return '.modal >> ';
                }
            } catch (e) {}
            return 'role=dialog >> ';
        }

        const found = [];
        const seen = new Set();

        function pushEntry(el, sel, labelPrefix, scopePrefix) {
            const e = makeEntry(el, sel, labelPrefix, scopePrefix);
            if (!e) return;
            const k = e.selector + '|' + e.label.slice(0, 40);
            if (seen.has(k)) return;
            seen.add(k);
            found.push(e);
        }

        const MODAL_CAP = 45;
        for (const root of topModalRoots) {
            const rootScope = scopeForRoot(root);
            for (const sel of selectors) {
                for (const el of root.querySelectorAll(sel)) {
                    pushEntry(el, sel, '[dialog] ', rootScope);
                    if (found.length >= MODAL_CAP) break;
                }
                if (found.length >= MODAL_CAP) break;
            }
            if (found.length >= MODAL_CAP) break;
        }

        function inAnyModal(el) {
            return topModalRoots.some(r => r.contains(el));
        }

        for (const sel of selectors) {
            for (const el of document.querySelectorAll(sel)) {
                if (inAnyModal(el)) continue;
                pushEntry(el, sel, '', '');
                if (found.length >= 60) break;
            }
            if (found.length >= 60) break;
        }
        return found;
    }"""
    )
    return elements if isinstance(elements, list) else []


async def _detect_popup_text(page: Any) -> str:
    """Best-effort visible dialog / modal text."""
    try:
        raw = await page.evaluate(
            """() => {
            const sels = [
                '[role="dialog"]',
                '.modal',
                '[aria-modal="true"]',
                '.modal-dialog',
            ];
            for (const sel of sels) {
                for (const el of document.querySelectorAll(sel)) {
                    const st = window.getComputedStyle(el);
                    if (st.display === 'none' || st.visibility === 'hidden') continue;
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 40 || rect.height < 40) continue;
                    const t = (el.innerText || '').trim();
                    if (t.length > 3) return t.slice(0, 500);
                }
            }
            return '';
        }"""
        )
        return str(raw or "").strip()
    except Exception:
        logger.debug("popup detection failed", exc_info=True)
        return ""


async def _try_dismiss_blocking_overlays(page: Any) -> None:
    """Best-effort dismiss of modals that intercept clicks (Amazon region, cookies, etc.)."""
    try:
        host = (urlparse(page.url or "").hostname or "").lower()
    except Exception:
        host = ""

    if "amazon." not in host:
        return

    dismiss_selectors = [
        "#sp-cc-accept",
        "input#sp-cc-accept",
        "#sp-cc-customize + #sp-cc-accept",
        "button#sp-cc-accept",
        "#redir-modal .a-button-primary button",
        "#redir-modal button.a-button-text",
        "#redir-tap-via-cnp-root-continue button",
        "button[aria-label='Continue shopping']",
        "[data-action-type='DISMISS']",
    ]
    for sel in dismiss_selectors:
        try:
            loc = page.locator(sel)
            if await loc.count() == 0:
                continue
            first = loc.first
            if not await first.is_visible():
                continue
            await first.click(timeout=2500)
            await asyncio.sleep(0.35)
            logger.info("[BROWSER] dismissed overlay via %s", sel)
        except Exception:
            continue

    try:
        removed = await page.evaluate(
            """() => {
            const ids = ['redir-modal'];
            let n = 0;
            for (const id of ids) {
                const el = document.getElementById(id);
                if (el) { el.remove(); n++; }
            }
            document.querySelectorAll('.redir-modal-bg').forEach(el => {
                el.remove(); n++;
            });
            return n;
        }"""
        )
        if removed:
            logger.info(
                "[BROWSER] removed %s blocking overlay node(s) from DOM", removed)
            await asyncio.sleep(0.2)
    except Exception:
        logger.debug("overlay removal script failed", exc_info=True)


async def _gather_page_context(page: Any) -> dict:
    """Fresh URL, title, IBM list, DOM list, merged map, and popup snippet."""
    current_url = page.url or "about:blank"
    current_title = ""
    try:
        current_title = await page.title()
    except Exception:
        logger.debug("page.title() failed", exc_info=True)

    # ibm_raw = await loop.run_in_executor(None, _run_a11y_checker, current_url)
    ibm_elements = []  # IBM Equal Access disabled — using Playwright DOM only

    try:
        dom_raw = await _extract_elements_playwright(page)
    except Exception:
        logger.exception("Playwright element extraction failed")
        dom_raw = []
    dom_elements = dom_raw[:MAX_DOM_ELEMENTS] if isinstance(
        dom_raw, list) else []

    merged_elements = _merge_ibm_and_dom(ibm_elements, dom_elements)

    popup_raw = await _detect_popup_text(page)
    popup = popup_raw if popup_raw else None

    return {
        "url": current_url,
        "title": current_title,
        "ibm_elements": ibm_elements,
        "dom_elements": dom_elements,
        "merged_elements": merged_elements,
        "popup": popup,
    }


def _overlay_blocks_page(ctx: dict) -> bool:
    """True when a modal/dialog likely covers the page (URL may be unchanged)."""
    if (ctx.get("popup") or "").strip():
        return True
    scoped_prefixes = (
        "role=dialog >>",
        ".modal >>",
        ".modal-dialog >>",
    )
    for item in ctx.get("merged_elements") or []:
        if not isinstance(item, dict):
            continue
        sel = str(item.get("selector") or "")
        lab = str(item.get("label") or "")
        if lab.startswith("[dialog] "):
            return True
        if any(sel.startswith(p) for p in scoped_prefixes):
            return True
        if " >> " in sel and sel.split(" >> ", 1)[0].startswith("#"):
            return True
    return False


def _plan_action_sequence(transcript: str, tool_call: dict, ctx: dict) -> list[dict]:
    """Synchronous Featherless call; returns up to MAX_ACTIONS_PER_TURN action dicts."""
    key, base, _, timeout_sec = _load_llm_env()
    PLANNER_MODEL = (
        os.environ.get("FEATHERLESS_ORCHESTRATOR_MODEL",
                       DEFAULT_ORCHESTRATOR_MODEL).strip()
        or DEFAULT_ORCHESTRATOR_MODEL
    )
    from openai import OpenAI

    client = OpenAI(base_url=base, api_key=key, timeout=timeout_sec)

    popup_line = f"- POPUP VISIBLE: {ctx['popup']}" if ctx.get("popup") else ""
    system_prompt = f"""You are a browser automation agent controlling Firefox for a blind user.

CURRENT PAGE:
- URL: {ctx['url']}
- Title: {ctx['title']}
{popup_line}

AVAILABLE ELEMENTS:
{json.dumps(ctx['merged_elements'], indent=2)}

MODALS AND OVERLAYS:
- If POPUP VISIBLE is non-empty OR elements include `[dialog]` labels or selectors
  with a **scope prefix** (`role=dialog >>`, `.modal >>`, `#element-id >>`, etc.),
  a **modal is open**. The **URL and title often stay the same** as the page underneath.
- Prefer actions on **dialog** rows first. Use scoped selectors when listed — they
  target the foreground, not duplicate controls on the dimmed page.
- To confirm a flow (e.g. start game), click the **primary button inside the dialog**
  (often at the bottom), not a similar label on the background.
- **Never** plan only a bare `role=link[name=/.../i]` or `role=button[name=/.../i]`
  when `[dialog]` rows or scoped selectors exist — unscoped matches hit the **hidden**
  page under the modal. Copy the full selector (`role=dialog >> …`, `#id >> …`).

INTERSTITIAL DETECTION:
- Before planning the user's requested actions, check if the current page is an
  interstitial, captcha, age gate, cookie consent, or "continue" page by looking
  at the page title and available elements.
- Signs of an interstitial: title contains words like "verification", "continue",
  "robot", "captcha", "before you continue", "cookies"; OR elements list contains
  buttons with labels like "Continue shopping", "Accept", "I'm not a robot",
  "Accept all", "Agree", "Continue".
- If an interstitial is detected, prepend actions to dismiss it FIRST (click the
  continue/accept button using a selector from the element list), then add a wait
  of 2000ms, then proceed with the user's original request in the same array.
- For Amazon specifically: if URL contains amazon.com and elements include a
  "Continue shopping" button, click it first and wait 2000ms before searching.
- Amazon often shows a full-screen regional redirect overlay (`#redir-modal`,
  `.redir-modal-bg`) that **blocks all clicks** until dismissed. If the URL or
  title suggests amazon.* OR elements look like search results but actions might
  fail, prepend: click the modal's primary / Continue / cookie accept button
  (e.g. `#sp-cc-accept`, `#redir-modal .a-button-primary`), then `wait` 2000ms,
  before clicking "Add to cart" or typing in search.

DOWNLOADS (installers, .deb, AppImage, disk files):
- When the user wants an actual **file download**, set `"expect_download": true`
  on the `navigate` or `click` step that starts the transfer (Playwright waits
  for the browser download event and saves the file).
- If you know a **direct download URL** from the vendor (HTTP response is the
  file, not only a "thank you" HTML page), prefer
  `{{"action": "navigate", "url": "...", "expect_download": true}}`.
  Example — VS Code stable Linux 64-bit .deb:
  `https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-x64`
- On "thanks for downloading" pages, click the **direct download** / **retry**
  / **click here** link using a selector from the element list with
  `"expect_download": true`.
- Optional per-step `"download_timeout_ms"` (e.g. 300000 for large installers).
- A thank-you page without `expect_download` does **not** prove the file saved;
  always attach `expect_download` to the step that triggers the download.

AMAZON CART AND CHECKOUT (critical):
- If the user wants the **shopping cart**, **checkout**, **proceed to checkout**, or **pay**:
  do NOT click random product tiles, sponsored links, or `/dp/` product pages from search results.
- If CURRENT URL path looks like a **cart** (`/gp/cart`, `/cart`): use only checkout controls:
  prefer `input[name="proceedToRetailCheckout"]`, then `#sc-buy-box-ptc-button`,
  then `input[name="proceedToCheckout"]`, or an `a`/`button` whose visible text is
  exactly "Proceed to checkout" / "Checkout" on the cart chrome — not a product name.
- If you need the cart from elsewhere on Amazon: prefer
  `{{"action": "navigate", "url": "<same host as CURRENT PAGE>/gp/cart/view.html"}}`
  using the SAME amazon.* host as CURRENT PAGE (e.g. www.amazon.ca → that host's cart URL),
  instead of clicking `#nav-cart` when possible.
- Search results are NOT checkout: never interpret "checkout" as "open a random search hit".

LICHESS (lichess.org / lichess.com on CURRENT URL):
- If the user wants **play against the computer** / **bot** / **AI** / **Stockfish** / **engine**
  / **machine** / **offline solo**: do **not** click "Create lobby game", "Lobby", pairing,
  "Challenge a friend", or other **human** matchmaking — those start wrong flows.
  Choose the control whose **label** clearly mentions **computer** (e.g. "Play against computer").
- If the user wants a **human opponent** / **online** / **lobby** / **multiplayer**, then
  lobby / create-game actions are appropriate.
- If the page is a **404**, **wrong team/network admin**, or clearly not the main site,
  `{{"action": "navigate", "url": "https://lichess.org/"}}` then `{{"action": "wait", "ms": 1500}}`
  before continuing.
- Prefer **role=link** / **role=button** selectors with a **name=** pattern from the list;
  never choose a bare `a[href]` entry when a **role=** row exists for the same control.
- **Game setup** dialog: after "Play against computer" opens setup, the user must
  confirm with the **Play against computer** (or **Start**) control **inside the
  dialog**; time controls and level are in that same overlay.

RULES:
- Return a JSON array of actions to complete the user's request in sequence.
- Act on the CURRENT page unless the user says "go to" or "open" a different site.
- Prefer selectors that include **role=** and **name=** from the element list over bare
  tag selectors (`a[href]`, `button[type="button"]` without a name) — bare tags match
  the first node on the page and often activate the wrong link (footer, nav, etc.).
- Use selectors from the element list when possible. For Amazon cart/checkout steps above,
  you MAY use the exact selectors named in AMAZON CART AND CHECKOUT even if missing from the list.
- On lichess.org, if the user needs **computer** play and no matching row appears, you MAY use
  `role=link[name=/computer/i]` or `role=button[name=/computer/i]` even if absent from the list.
- For search: always include a keypress Enter after typing.
- For popups: dismiss or interact with them first before other actions.
- Maximum {MAX_ACTIONS_PER_TURN} actions per turn.

ACTION TYPES:
{{"action": "navigate", "url": "https://..."}}
{{"action": "navigate", "url": "https://...", "expect_download": true}}
{{"action": "click", "selector": "#id"}}
{{"action": "click", "selector": "#id", "expect_download": true}}
{{"action": "type", "selector": "#id", "text": "..."}}
{{"action": "keypress", "key": "Enter"}}
{{"action": "scroll", "direction": "down"}}
{{"action": "back"}}
{{"action": "wait", "ms": 1500}}

Respond with a JSON array only. Example:
[
  {{"action": "type", "selector": "#twotabsearchtextbox", "text": "wireless headphones"}},
  {{"action": "keypress", "key": "Enter"}},
  {{"action": "wait", "ms": 2000}}
]
"""

    planner_tool = dict(tool_call) if isinstance(tool_call, dict) else {}
    user_content = (
        f"User transcript: {transcript}\n"
        f"Orchestrator tool call: {json.dumps(planner_tool)}\n"
    )

    try:
        completion = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
    except Exception:
        logger.exception("Featherless action-sequence planner failed")
        return list(FALLBACK_NAVIGATE_GOOGLE)

    raw = (completion.choices[0].message.content or "").strip()
    candidate = _strip_json_fence(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning(
            "Browser planner returned non-JSON (truncated): %s", raw[:200])
        return list(FALLBACK_NAVIGATE_GOOGLE)

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        logger.warning("Browser planner returned non-array JSON")
        return list(FALLBACK_NAVIGATE_GOOGLE)

    actions: list[dict] = []
    for item in parsed:
        if isinstance(item, dict) and str(item.get("action", "")).strip():
            actions.append(item)
        if len(actions) >= MAX_ACTIONS_PER_TURN:
            break
    if not actions:
        logger.warning("Browser planner returned empty action list")
        return list(FALLBACK_NAVIGATE_GOOGLE)
    return actions


def _step_expect_download(step: dict) -> bool:
    v = step.get("expect_download")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(v, (int, float)):
        return bool(v)
    return False


def _step_download_timeout_ms(step: dict) -> int:
    default = default_download_timeout_ms()
    try:
        v = int(step.get("download_timeout_ms", default))
    except (TypeError, ValueError):
        return default
    return max(3000, min(v, 600_000))


async def _amazon_aware_click(page: Any, step: dict, *, timeout_ms: int = 12000) -> None:
    sel = str(step.get("selector", "")).strip()
    host = (urlparse(page.url or "").hostname or "").lower()
    try:
        await _reliable_click(page, sel, timeout_ms=timeout_ms)
    except Exception as click_err:
        if "amazon." in host and (
            "nav-cart" in sel
            or sel.endswith("nav-cart")
            or "cart/view" in sel
        ):
            dest = _amazon_view_cart_url(page.url or "")
            logger.warning(
                "[BROWSER] cart control click failed (%s); opening %s",
                click_err,
                dest,
            )
            await page.goto(
                dest, wait_until="domcontentloaded", timeout=15000
            )
        else:
            raise


async def _execute_sequence(page: Any, actions: list[dict]) -> tuple[str, list[str]]:
    """Execute actions in order. Skip failed steps; stop only after 2 consecutive failures.

    Returns a summary string for logging/LLM context and absolute paths of files
    saved via ``expect_download`` steps (may be empty).
    """
    done: list[str] = []
    saved_downloads: list[str] = []
    consecutive_fail = 0
    for step in actions:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action", "")).strip().lower()
        try:
            if action == "navigate":
                url = str(step.get("url", "")).strip()
                if not url:
                    raise ValueError("navigate action missing url")
                if _step_expect_download(step):
                    dl_ms = _step_download_timeout_ms(step)
                    try:
                        async with page.expect_download(timeout=dl_ms) as dl_info:
                            await page.goto(
                                url,
                                wait_until="domcontentloaded",
                                timeout=15000,
                            )
                        dl = await dl_info.value
                        path = await persist_playwright_download(dl)
                        saved_downloads.append(str(path))
                        logger.info("[BROWSER] download saved to %s", path)
                    except Exception as dl_err:
                        logger.warning(
                            "[BROWSER] navigate+expect_download failed: %s",
                            dl_err,
                        )
                        raise
                else:
                    await page.goto(
                        url, wait_until="domcontentloaded", timeout=15000
                    )
            elif action == "click":
                await _try_dismiss_blocking_overlays(page)
                if _step_expect_download(step):
                    dl_ms = _step_download_timeout_ms(step)
                    try:
                        async with page.expect_download(timeout=dl_ms) as dl_info:
                            await _amazon_aware_click(
                                page, step, timeout_ms=12000
                            )
                        dl = await dl_info.value
                        path = await persist_playwright_download(dl)
                        saved_downloads.append(str(path))
                        logger.info("[BROWSER] download saved to %s", path)
                    except Exception as dl_err:
                        logger.warning(
                            "[BROWSER] click+expect_download failed: %s",
                            dl_err,
                        )
                        raise
                else:
                    await _amazon_aware_click(page, step, timeout_ms=12000)
            elif action == "type":
                await _try_dismiss_blocking_overlays(page)
                await page.fill(step["selector"], step["text"], timeout=8000)
            elif action == "keypress":
                await page.keyboard.press(step.get("key", "Enter"))
            elif action == "scroll":
                direction = step.get("direction", "down")
                delta = 500 if direction == "down" else -500
                await page.evaluate(f"window.scrollBy(0, {delta})")
            elif action == "back":
                await page.go_back(wait_until="domcontentloaded", timeout=10000)
            elif action == "wait":
                await asyncio.sleep(step.get("ms", 1000) / 1000)
            else:
                logger.warning("[BROWSER] unknown action %r skipped", action)
                continue
            done.append(action)
            consecutive_fail = 0
        except Exception as e:
            logger.warning(
                "[BROWSER] action %r failed (skipping): %s", action, e)
            consecutive_fail += 1
            if consecutive_fail >= 2:
                logger.warning(
                    "[BROWSER] stopping sequence after %s consecutive failures",
                    consecutive_fail,
                )
                break
    summary = ", ".join(done) if done else "no actions completed"
    return summary, saved_downloads


_DOM_DESCRIPTION_THIN_PHRASES = (
    "unable to determine",
    "cannot describe",
    "no information",
    "can't describe",
    "cannot tell",
    "can't tell",
    "not enough information",
    "unclear what",
    "i cannot summarize",
)


def _dom_description_needs_vision(tier1: str) -> bool:
    """True when tier-1 text is too short or signals uncertainty → use vision."""
    t = (tier1 or "").strip()
    if not t:
        return True
    if len(t) < 80:
        return True
    low = t.lower()
    return any(p in low for p in _DOM_DESCRIPTION_THIN_PHRASES)


def _describe_result_from_context(
    transcript: str,
    actions_summary: str,
    page_url: str,
    page_title: str,
    merged_elements: list[dict],
    saved_downloads: Optional[list[str]] = None,
    popup_text: Optional[str] = None,
) -> str:
    """Text-only LLM: what the user likely sees after actions (cheap tier)."""
    key, base, _, timeout_sec = _load_llm_env()
    PLANNER_MODEL = (
        os.environ.get("FEATHERLESS_ORCHESTRATOR_MODEL",
                       DEFAULT_ORCHESTRATOR_MODEL).strip()
        or DEFAULT_ORCHESTRATOR_MODEL
    )
    from openai import OpenAI

    client = OpenAI(base_url=base, api_key=key, timeout=timeout_sec)
    elements_json = json.dumps(merged_elements[:MAX_MERGED_ELEMENTS], indent=2)
    system_prompt = (
        "You describe the browser state for a blind user after automation ran. "
        "Use the element list as hints about what is on the page; it may be incomplete. "
        "If a dialog / modal snippet is provided, **lead with that** — the URL may still "
        "show the underlying site (e.g. homepage) even though a setup window is in front. "
        "Rows labeled `[dialog]` or selectors with `role=dialog >>` refer to the foreground. "
        "If the context lists files saved to disk, confirm the download finished and mention "
        "them in plain language. "
        "Output may be read aloud by text-to-speech: do **not** spell long paths or cryptic "
        "filenames. Prefer short labels (e.g. 'the VS Code installer in your Downloads folder') "
        "instead of reading every character of a .deb or path. "
        "Reply in 2-3 short plain sentences: what page or state they are likely seeing, "
        "and one useful detail (e.g. search box, main heading). "
        "If you truly cannot infer the result from this data, say exactly: "
        "I cannot describe the result with confidence."
    )
    dl_lines = ""
    if saved_downloads:
        dl_lines = (
            "\nFiles saved to disk this session (paths are for reference only; when speaking, "
            "use a short friendly description — not the full path):\n"
            + "\n".join(saved_downloads)
            + "\nConfirm the download finished; describe what it is in human terms.\n"
        )
    popup_lines = ""
    if (popup_text or "").strip():
        popup_lines = (
            "\nVisible dialog / modal text (abridged):\n"
            + (popup_text or "").strip()[:900]
            + "\n"
        )
    user_content = (
        f"User asked: {transcript}\n"
        f"Actions performed: {actions_summary}\n"
        f"Current URL: {page_url}\n"
        f"Page title: {page_title}\n"
        f"{popup_lines}"
        f"{dl_lines}"
        f"Elements (partial):\n{elements_json}\n"
    )
    try:
        completion = client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("[BROWSER] DOM description LLM failed")
        return ""


async def _do_browser_action(transcript: str, orchestrator_response: dict) -> str:
    """Gather context, plan, execute, then DOM description or vision fallback."""
    from sensus.vision.screen import FALLBACK_MESSAGE, describe_image

    await _ensure_browser()
    assert _lock is not None
    async with _lock:
        page = await _ensure_browser()
        loop = asyncio.get_running_loop()

        ctx = await _gather_page_context(page)
        url_before = ctx["url"]
        actions = await loop.run_in_executor(
            None,
            _plan_action_sequence,
            transcript,
            orchestrator_response,
            ctx,
        )

        actions_summary, saved_downloads = await _execute_sequence(page, actions)
        url_after_execute = (page.url or "").rstrip("/")

        await asyncio.sleep(0.5)

        ctx_after = await _gather_page_context(page)
        if url_after_execute != (url_before or "").rstrip("/"):
            logger.info(
                "[BROWSER] page changed during execution, refreshing context")
            ctx_after = await _gather_page_context(page)

        url_after = ctx_after["url"]
        title_after = ctx_after["title"]
        merged_after = ctx_after["merged_elements"]

        popup_after = ctx_after.get("popup") or ""

        tier1 = await loop.run_in_executor(
            None,
            partial(
                _describe_result_from_context,
                transcript,
                actions_summary,
                url_after,
                title_after,
                merged_after,
                saved_downloads,
                popup_after,
            ),
        )

        tier1_stripped = (tier1 or "").strip()
        overlay_open = _overlay_blocks_page(ctx_after)
        if overlay_open and not _dom_description_needs_vision(tier1):
            logger.info(
                "[BROWSER] modal/overlay detected — using vision for foreground state"
            )
        if (
            not _dom_description_needs_vision(tier1)
            and not overlay_open
        ):
            logger.info(
                "[BROWSER] DOM description sufficient, skipping vision")
            return tier1_stripped or "I've updated the page for you."

        if overlay_open:
            logger.info(
                "[BROWSER] vision fallback (overlay open, %s chars tier-1)",
                len(tier1_stripped),
            )
        else:
            logger.info(
                "[BROWSER] DOM description thin (%s chars), using vision fallback",
                len(tier1_stripped),
            )
        dl_fragment = ""
        if saved_downloads:
            dl_fragment = (
                f" File(s) saved: {'; '.join(saved_downloads)}."
            )
        overlay_note = ""
        if overlay_open:
            overlay_note = (
                " A modal or overlay is likely covering the page — describe that "
                "foreground first (controls, title). The URL may still be the parent page. "
            )
        vision_prompt = (
            f"A blind user asked: '{transcript}'. "
            f"The browser just performed: {actions_summary} on {page.url}.{dl_fragment} "
            f"{overlay_note}"
            f"Describe the result in 2-3 sentences. "
            f"If search results are visible, name the top 3 with prices if shown. "
            f"If a popup or dialog is visible, describe it in detail. "
            f"End with a natural follow-up question if appropriate. "
            f"Be direct, no preamble."
        )
        screenshot_ok = False
        try:
            vp = page.viewport_size
            if not vp:
                vp = await page.evaluate(
                    "() => ({ width: window.innerWidth, height: window.innerHeight })"
                )
            vw = int(vp.get("width", 1280))
            vh = int(vp.get("height", 720))
            clip = {"x": 0, "y": 0, "width": vw, "height": vh}
            await page.screenshot(
                path=BROWSER_VISION_SCREENSHOT_PATH,
                full_page=False,
                clip=clip,
            )
            screenshot_ok = True
        except Exception as e:
            logger.error("[BROWSER] screenshot failed: %s", e, exc_info=True)

        if not screenshot_ok:
            return (
                tier1
                or "I performed the action but couldn't capture the screen."
            )

        try:
            vision = await loop.run_in_executor(
                None,
                describe_image,
                BROWSER_VISION_SCREENSHOT_PATH,
                vision_prompt,
            )
        except Exception as e:
            logger.error("[BROWSER] vision describe failed: %s",
                         e, exc_info=True)
            return tier1 or FALLBACK_MESSAGE

        if vision == FALLBACK_MESSAGE or not (vision or "").strip():
            return tier1 or vision or FALLBACK_MESSAGE
        return vision


async def run_browser_action(transcript: str, orchestrator_response: dict) -> str:
    """Main entry point — dispatches the work to the persistent browser loop."""
    target_loop = _ensure_persistent_loop()
    coro = _do_browser_action(transcript, orchestrator_response)
    cf_future = asyncio.run_coroutine_threadsafe(coro, target_loop)
    return await asyncio.wrap_future(cf_future)


async def close_browser() -> None:
    """Tear down the singleton browser and stop the persistent loop."""
    global _loop, _loop_thread, _playwright, _browser, _page, _lock

    target_loop = _loop
    if target_loop is None or not target_loop.is_running():
        _playwright = None
        _browser = None
        _page = None
        _lock = None
        _loop = None
        _loop_thread = None
        return

    async def _shutdown() -> None:
        global _playwright, _browser, _page
        try:
            if _page is not None and not _page.is_closed():
                await _page.close()
        except Exception:
            logger.exception("Failed to close page")
        _page = None

        try:
            if _browser is not None:
                await _browser.close()
        except Exception:
            logger.exception("Failed to close browser")
        _browser = None

        try:
            if _playwright is not None:
                await _playwright.stop()
        except Exception:
            logger.exception("Failed to stop playwright")
        _playwright = None

    cf_future = asyncio.run_coroutine_threadsafe(_shutdown(), target_loop)
    try:
        await asyncio.wrap_future(cf_future)
    finally:
        target_loop.call_soon_threadsafe(target_loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=5)
        _lock = None
        _loop = None
        _loop_thread = None
        _loop_started.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _root = Path(__file__).resolve().parents[2]
    if "--skip-sidecar-test" not in sys.argv:
        _sidecar = subprocess.run(
            ["node", "accessibility/checker.js", "https://google.com"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(_root),
        )
        _elements = json.loads(_sidecar.stdout or "[]")
        print(f"[TEST] IBM Equal Access returned {len(_elements)} elements")
        if _elements:
            print(f"[TEST] Sample: {_elements[0]}")

    _default_tool_call: dict = {
        "agent": "browser",
        "action": "",
        "details": {},
    }

    async def _interactive() -> None:
        print(
            "\nBrowser REPL — one headed Firefox window; each line is a new request.\n"
            "  Type what you want (e.g. open google, search for pandas, scroll down).\n"
            "  quit / q / empty line — stop and close the browser.\n"
            "  Advanced: paste a JSON object on one line to use as the orchestrator tool_call\n"
            '  instead of the default (must start with "{").\n'
        )
        while True:
            try:
                line = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in ("quit", "q", "exit"):
                break
            tool_call: dict
            if line.startswith("{"):
                try:
                    loaded = json.loads(line)
                except json.JSONDecodeError:
                    print("Invalid JSON; try again.")
                    continue
                if not isinstance(loaded, dict):
                    print("Tool call JSON must be an object; try again.")
                    continue
                tool_call = {**_default_tool_call, **loaded}
                try:
                    transcript_line = input(
                        "Transcript for this call> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not transcript_line:
                    transcript_line = str(
                        tool_call.get("action", "")) or "browser"
            else:
                transcript_line = line
                tool_call = dict(_default_tool_call)

            reply = await run_browser_action(transcript_line, tool_call)
            print(f"Sensus: {reply}\n")

    async def _main() -> None:
        await _interactive()
        await close_browser()

    asyncio.run(_main())
