"""
IBM Db2 persistence for sessions, messages, and preferences.

Connection uses ``ibm_db.connect(conn_string, '', '')`` — set ``IBM_DB_CONN_STRING`` to a full
Db2 CLI connection string, for example::

    DATABASE=BLUDB;HOSTNAME=your-host.databases.appdomain.cloud;PORT=50001;
    PROTOCOL=TCPIP;UID=myuser;PWD=secret;SECURITY=SSL

DDL (run once on your instance) — or use ``scripts/db2_schema.sql`` in the IBM SQL UI::

    CREATE TABLE sessions (
      id VARCHAR(36) PRIMARY KEY,
      user_id VARCHAR(36),
      name VARCHAR(255),
      created_at TIMESTAMP,
      summary CLOB
    );
    CREATE TABLE messages (
      id VARCHAR(36) PRIMARY KEY,
      session_id VARCHAR(36),
      role VARCHAR(16),
      content CLOB,
      agent VARCHAR(32),
      created_at TIMESTAMP
    );
    CREATE TABLE preferences (
      user_id VARCHAR(36) PRIMARY KEY,
      prefs_json CLOB
    );

Requires the ``ibm-db`` wheel plus IBM ODBC/CLI driver (see IBM docs). If ``ibm_db`` is not
installed or ``IBM_DB_CONN_STRING`` is unset, all functions no-op safely.

Connection waits default to **30 seconds** (Python-side and CLI ``ConnectTimeout``). Override with
``IBM_DB_CONNECT_TIMEOUT`` (seconds). Set to ``0`` for legacy behavior (no timeout — can hang
forever if the server is unreachable).
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_ibm_db: Any = None
# After connect timeout or explicit disable, skip Db2 for the rest of the process.
_db2_disabled: bool = False


def _disable_db2(reason: str) -> None:
    global _db2_disabled
    _db2_disabled = True
    logger.error(
        "%s Sensus continues without Db2 persistence until you restart. "
        "Check IBM_DB_CONN_STRING / network / firewall, or set IBM_DB_CONNECT_TIMEOUT=0 "
        "only if you intentionally wait indefinitely.",
        reason,
    )


def _connect_timeout_and_merge_cli() -> tuple[Optional[float], bool]:
    """
    Returns (seconds for Python-side wait, whether to merge ConnectTimeout into conn string).

    Env IBM_DB_CONNECT_TIMEOUT (default ``30``): max seconds to wait for ``ibm_db.connect``.
    Set to ``0`` to restore legacy behavior (no thread timeout; connection string unchanged).
    """
    raw = os.environ.get("IBM_DB_CONNECT_TIMEOUT", "30").strip().lower()
    if raw in ("0", "none", "off", "false"):
        return None, False
    try:
        sec = max(1, int(float(raw)))
    except ValueError:
        sec = 30
    return float(sec), True


def _merge_connect_timeout(conn_string: str, seconds: int) -> str:
    """Append IBM CLI ``ConnectTimeout`` if not already present (TCPIP)."""
    if not conn_string:
        return conn_string
    if re.search(r"(?:^|;)\s*ConnectTimeout\s*=", conn_string, re.I):
        return conn_string
    sep = "" if conn_string.rstrip().endswith(";") else ";"
    return f"{conn_string}{sep}ConnectTimeout={seconds};"


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _import_ibm_db() -> Any:
    global _ibm_db
    if _ibm_db is False:
        return None
    if _ibm_db is not None:
        return _ibm_db
    try:
        import ibm_db as ibm_db_mod

        _ibm_db = ibm_db_mod
        return _ibm_db
    except ImportError:
        logger.warning(
            "ibm_db not installed — Db2 persistence disabled. "
            "Install ibm-db and IBM CLI driver for Db2."
        )
        _ibm_db = False
        return None


def connection_string() -> str:
    """Read current env each time (do not cache — ``load_dotenv`` / edits must be visible)."""
    _try_load_dotenv()
    return os.environ.get("IBM_DB_CONN_STRING", "").strip()


def is_configured() -> bool:
    if _db2_disabled:
        return False
    return bool(connection_string()) and _import_ibm_db() is not None


def _resolve_connection_string() -> str:
    """Full CLI string with optional ``ConnectTimeout`` (see ``_connect_timeout_and_merge_cli``)."""
    cs = connection_string()
    timeout_sec, merge_cli = _connect_timeout_and_merge_cli()
    if merge_cli and timeout_sec is not None:
        cs = _merge_connect_timeout(cs, int(timeout_sec))
    return cs


def _connect():
    ibm_db = _import_ibm_db()
    if ibm_db is None:
        raise RuntimeError("ibm_db unavailable")
    if _db2_disabled:
        raise RuntimeError("Db2 disabled after connection failure")
    cs = _resolve_connection_string()
    if not cs:
        raise RuntimeError("IBM_DB_CONN_STRING not set")

    def _open() -> Any:
        return ibm_db.connect(cs, "", "")

    py_timeout, _ = _connect_timeout_and_merge_cli()
    if py_timeout is None:
        return _open()

    ex = ThreadPoolExecutor(max_workers=1)
    fut: Any = ex.submit(_open)
    try:
        return fut.result(timeout=py_timeout)
    except FuturesTimeoutError:
        _disable_db2(
            f"Db2 connection timed out after {int(py_timeout)}s "
            "(increase IBM_DB_CONNECT_TIMEOUT or fix network)."
        )
        raise RuntimeError("Db2 connection timed out") from None
    finally:
        ex.shutdown(wait=False)


def _exec_ddl(conn: Any, sql: str, label: str) -> None:
    """Run one DDL statement; log failures loudly (except table-already-exists)."""
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    try:
        stmt = ibm_db.exec_immediate(conn, sql)
        if stmt:
            ibm_db.free_stmt(stmt)
        logger.info("Db2 DDL applied: %s", label)
    except Exception as e:
        msg = str(e)
        if "SQL0605" in msg or "-601" in msg or "already exists" in msg.lower():
            logger.debug("Db2 DDL skipped (already exists): %s", label)
            return
        logger.warning("Db2 DDL failed (%s): %s", label, msg)


def _maybe_log_missing_tables_hint(exc: BaseException) -> None:
    msg = str(exc)
    if (
        "SQL0204N" in msg
        or "SQLSTATE=42704" in msg
        or "SQLCODE=-204" in msg
        or "42704" in msg
    ):
        logger.error(
            "Db2: table or view not found — create schema first. "
            "Run scripts/db2_schema.sql in the IBM Db2 SQL console, "
            "or set IBM_DB_AUTO_SCHEMA=1 in .env (development only), then restart."
        )


def ensure_schema() -> None:
    """Create tables when ``IBM_DB_AUTO_SCHEMA=1`` (development convenience)."""
    _try_load_dotenv()
    if os.environ.get("IBM_DB_AUTO_SCHEMA", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    if not is_configured():
        return
    with _lock:
        try:
            conn = _connect()
        except Exception as e:
            logger.warning("Db2 ensure_schema: connect failed (%s)", e)
            return
        try:
            _exec_ddl(
                conn,
                """
                CREATE TABLE sessions (
                  id VARCHAR(36) NOT NULL PRIMARY KEY,
                  user_id VARCHAR(36),
                  name VARCHAR(255),
                  created_at TIMESTAMP,
                  summary CLOB)
                """,
                "sessions",
            )
            _exec_ddl(
                conn,
                """
                CREATE TABLE messages (
                  id VARCHAR(36) NOT NULL PRIMARY KEY,
                  session_id VARCHAR(36),
                  role VARCHAR(16),
                  content CLOB,
                  agent VARCHAR(32),
                  created_at TIMESTAMP)
                """,
                "messages",
            )
            _exec_ddl(
                conn,
                """
                CREATE TABLE preferences (
                  user_id VARCHAR(36) NOT NULL PRIMARY KEY,
                  prefs_json CLOB)
                """,
                "preferences",
            )
            ibm_db = _import_ibm_db()
            assert ibm_db is not None
            ibm_db.commit(conn)
        finally:
            ibm_db = _import_ibm_db()
            assert ibm_db is not None
            ibm_db.close(conn)


def _ts_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def create_session(
    session_id: str,
    user_id: Optional[str],
    *,
    name: Optional[str] = None,
) -> bool:
    """Insert session row. Returns False if Db2 is off or the insert failed."""
    if not is_configured():
        return False
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    uid = user_id if (user_id or "").strip() else None
    sql = (
        "INSERT INTO sessions (id, user_id, name, created_at, summary) "
        "VALUES (?, ?, ?, ?, NULL)"
    )
    with _lock:
        try:
            conn = _connect()
        except Exception as e:
            logger.warning("Db2 create_session: connect failed (%s)", e)
            return False
        try:
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, session_id)
            ibm_db.bind_param(stmt, 2, uid)
            ibm_db.bind_param(stmt, 3, (name[:255] if name else None))
            ibm_db.bind_param(stmt, 4, _ts_now())
            ibm_db.execute(stmt)
            ibm_db.commit(conn)
            return True
        except Exception as e:
            logger.exception("Db2 create_session failed")
            _maybe_log_missing_tables_hint(e)
            try:
                ibm_db.rollback(conn)
            except Exception:
                pass
            return False
        finally:
            ibm_db.close(conn)


def insert_message(
    session_id: str,
    role: str,
    content: str,
    agent: Optional[str] = None,
) -> None:
    if not is_configured():
        return
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    mid = str(uuid.uuid4())
    ag = (agent or "").strip()[:32] or None
    sql = (
        "INSERT INTO messages (id, session_id, role, content, agent, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, mid)
            ibm_db.bind_param(stmt, 2, session_id)
            ibm_db.bind_param(stmt, 3, (role or "")[:16])
            ibm_db.bind_param(stmt, 4, content or "")
            ibm_db.bind_param(stmt, 5, ag)
            ibm_db.bind_param(stmt, 6, _ts_now())
            ibm_db.execute(stmt)
            ibm_db.commit(conn)
        except Exception as e:
            logger.exception("Db2 insert_message failed")
            _maybe_log_missing_tables_hint(e)
            try:
                ibm_db.rollback(conn)
            except Exception:
                pass
        finally:
            ibm_db.close(conn)


def _assoc_get(row: dict[str, Any], *names: str) -> Any:
    """ibm_db column keys vary by driver (ROLE vs role); tolerate case drift."""
    for n in names:
        if n in row:
            return row[n]
    upper = {str(k).upper(): v for k, v in row.items()}
    for n in names:
        if n.upper() in upper:
            return upper[n.upper()]
    return None


def _ibm_text(val: Any, *, strip: bool = False) -> str:
    """Normalize Db2 CHAR/VARCHAR/CLOB/buffer objects from ibm_db.fetch_assoc."""
    if val is None:
        return ""
    if hasattr(val, "read"):
        try:
            chunk = val.read()
        except Exception:
            return ""
        if isinstance(chunk, bytes):
            body = chunk.decode("utf-8", errors="replace")
        else:
            body = str(chunk)
        return body.strip() if strip else body
    if isinstance(val, memoryview):
        body = bytes(val).decode("utf-8", errors="replace")
        return body.strip() if strip else body
    if isinstance(val, bytes):
        body = val.decode("utf-8", errors="replace")
        return body.strip() if strip else body
    s = str(val)
    return s.strip() if strip else s


def fetch_messages_for_summary(session_id: str) -> list[tuple[str, str]]:
    """Return (role, content) in chronological order."""
    if not is_configured():
        return []
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    sid = (session_id or "").strip()
    if not sid:
        return []
    sql = (
        "SELECT role, content FROM messages WHERE session_id = ? "
        "ORDER BY created_at ASC"
    )
    rows: list[tuple[str, str]] = []
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, sid)
            ibm_db.execute(stmt)
            row = ibm_db.fetch_assoc(stmt)
            while row:
                r = _ibm_text(_assoc_get(row, "ROLE", "role"), strip=True)
                c = _ibm_text(_assoc_get(row, "CONTENT", "content"), strip=False)
                rows.append((r, c))
                row = ibm_db.fetch_assoc(stmt)
        except Exception:
            logger.exception("Db2 fetch_messages_for_summary failed")
        finally:
            ibm_db.close(conn)
    return rows


def fetch_messages_chronological(session_id: str) -> list[tuple[str, str]]:
    """Alias for :func:`fetch_messages_for_summary` — messages in time order for UI replay."""
    return fetch_messages_for_summary(session_id)


def session_has_concluded_summary(session_id: str) -> bool:
    """
    True when ``sessions.summary`` exists and is non-empty (session was finalized on overlay close).

    Used to decide whether sidebar history can be replayed from Db2 for concluded sessions only.
    """
    if not is_configured():
        return False
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    sql = "SELECT summary FROM sessions WHERE id = ?"
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, session_id)
            ibm_db.execute(stmt)
            row = ibm_db.fetch_assoc(stmt)
            if not row:
                return False
            s = row.get("SUMMARY") or row.get("summary") or ""
            if hasattr(s, "read"):
                try:
                    s = s.read()
                except Exception:
                    s = str(s)
            return bool(str(s).strip())
        except Exception:
            logger.exception("Db2 session_has_concluded_summary failed")
            return False
        finally:
            ibm_db.close(conn)


def update_session_name(session_id: str, name: str) -> None:
    """Update ``sessions.name`` only (provisional title from live conversation)."""
    if not is_configured():
        return
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    sql = "UPDATE sessions SET name = ? WHERE id = ?"
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, (name or "")[:255])
            ibm_db.bind_param(stmt, 2, session_id)
            ibm_db.execute(stmt)
            ibm_db.commit(conn)
        except Exception:
            logger.exception("Db2 update_session_name failed")
            try:
                ibm_db.rollback(conn)
            except Exception:
                pass
        finally:
            ibm_db.close(conn)


def update_session_title_and_summary(
    session_id: str,
    *,
    name: str,
    summary_text: str,
) -> None:
    if not is_configured():
        return
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    sql = "UPDATE sessions SET name = ?, summary = ? WHERE id = ?"
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql)
            ibm_db.bind_param(stmt, 1, (name or "")[:255])
            ibm_db.bind_param(stmt, 2, summary_text or "")
            ibm_db.bind_param(stmt, 3, session_id)
            ibm_db.execute(stmt)
            ibm_db.commit(conn)
        except Exception:
            logger.exception("Db2 update_session_title_and_summary failed")
            try:
                ibm_db.rollback(conn)
            except Exception:
                pass
        finally:
            ibm_db.close(conn)


def list_recent_sessions(
    user_id: Optional[str],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Latest sessions, newest first.

    When ``user_id`` is set, filter ``sessions.user_id``; otherwise return recent rows for any user
    (single-user deployments).
    """
    if not is_configured():
        return []
    lim = max(1, min(int(limit), 100))
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    uid = (user_id or "").strip()
    if uid:
        sql = (
            "SELECT id, name, created_at, summary FROM sessions WHERE user_id = ? "
            "ORDER BY created_at DESC FETCH FIRST ? ROWS ONLY"
        )
    else:
        sql = (
            "SELECT id, name, created_at, summary FROM sessions "
            "ORDER BY created_at DESC FETCH FIRST ? ROWS ONLY"
        )
    out: list[dict[str, Any]] = []
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql)
            if uid:
                ibm_db.bind_param(stmt, 1, uid)
                ibm_db.bind_param(stmt, 2, lim)
            else:
                ibm_db.bind_param(stmt, 1, lim)
            ibm_db.execute(stmt)
            row = ibm_db.fetch_assoc(stmt)
            while row:
                rid = row.get("ID") or row.get("id")
                raw_nm = row.get("NAME") or row.get("name")
                name = "" if raw_nm is None else str(raw_nm).strip()
                created = row.get("CREATED_AT") or row.get("created_at")
                summ = row.get("SUMMARY") or row.get("summary") or ""
                if hasattr(summ, "read"):
                    try:
                        summ = summ.read()
                    except Exception:
                        summ = str(summ)
                created_iso = ""
                if created is not None:
                    created_iso = str(created).strip()
                out.append(
                    {
                        "id": str(rid).strip() if rid is not None else "",
                        "name": str(name),
                        "created_at": created_iso,
                        "summary": str(summ or ""),
                    }
                )
                row = ibm_db.fetch_assoc(stmt)
        except Exception:
            logger.exception("Db2 list_recent_sessions failed")
        finally:
            ibm_db.close(conn)
    return out


def save_preferences(user_id: str, prefs_json: str) -> None:
    if not is_configured():
        return
    uid = (user_id or "").strip()[:36]
    if not uid:
        return
    ibm_db = _import_ibm_db()
    assert ibm_db is not None
    sql_sel = "SELECT user_id FROM preferences WHERE user_id = ?"
    sql_upd = "UPDATE preferences SET prefs_json = ? WHERE user_id = ?"
    sql_ins = "INSERT INTO preferences (user_id, prefs_json) VALUES (?, ?)"
    with _lock:
        conn = _connect()
        try:
            stmt = ibm_db.prepare(conn, sql_sel)
            ibm_db.bind_param(stmt, 1, uid)
            ibm_db.execute(stmt)
            row = ibm_db.fetch_assoc(stmt)
            if row:
                stmt2 = ibm_db.prepare(conn, sql_upd)
                ibm_db.bind_param(stmt2, 1, prefs_json)
                ibm_db.bind_param(stmt2, 2, uid)
                ibm_db.execute(stmt2)
            else:
                stmt2 = ibm_db.prepare(conn, sql_ins)
                ibm_db.bind_param(stmt2, 1, uid)
                ibm_db.bind_param(stmt2, 2, prefs_json)
                ibm_db.execute(stmt2)
            ibm_db.commit(conn)
        except Exception:
            logger.exception("Db2 save_preferences failed")
            try:
                ibm_db.rollback(conn)
            except Exception:
                pass
        finally:
            ibm_db.close(conn)


def finalize_session(session_id: str) -> None:
    """DeepSeek one-line summary + update ``sessions`` row."""
    if not is_configured():
        return
    from sensus.storage.summary import summarize_transcript

    rows = fetch_messages_for_summary(session_id)
    if not rows:
        return
    line = summarize_transcript(rows)
    update_session_title_and_summary(session_id, name=line, summary_text=line)
