"""Persistent storage for completed analytics requests.

History lives in its own DuckDB file (``history.duckdb``), separate from the
analytics database, so that:

* rebuilding or deleting ``analytics.duckdb`` never destroys history,
* the two databases hold independent single-writer locks,
* tests can point at a temporary file without touching either production store.

The repository stores safe metadata only. It never records generated SQL, the
table schema, prompts, exception text, absolute paths, or result rows. Chart
references are stored as a bare filename; the URL is rebuilt and re-validated at
read time.

Connections are opened per operation and closed immediately, guarded by a
module-level lock, which keeps the store safe for the Flask worker and the CLI
without introducing background jobs or an async layer.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Optional

import duckdb

from config import (
    HISTORY_DATABASE,
    HISTORY_DEFAULT_LIMIT,
    HISTORY_MAX_LIMIT,
    HISTORY_TABLE,
)

logger = logging.getLogger(__name__)

# Rebound by tests to a temporary file, mirroring how chart.CHARTS_DIR is
# redirected. Production code always uses the configured location.
HISTORY_DATABASE = HISTORY_DATABASE
HISTORY_TABLE = HISTORY_TABLE

# A stored chart reference must be a plain PNG filename, never a path.
_CHART_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.png$", re.IGNORECASE)

_lock = RLock()

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {HISTORY_TABLE} (
    history_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    original_question TEXT NOT NULL,
    answer TEXT,
    row_count BIGINT,
    truncated BOOLEAN,
    max_rows BIGINT,
    chart_requested BOOLEAN,
    chart_type TEXT,
    chart_filename TEXT,
    chart_note TEXT,
    answer_fallback_used BOOLEAN,
    refused BOOLEAN,
    success BOOLEAN,
    elapsed_seconds DOUBLE,
    error_code TEXT
)
"""

_CREATE_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_{HISTORY_TABLE}_created_at "
    f"ON {HISTORY_TABLE} (created_at)"
)


class HistoryError(RuntimeError):
    """Raised when history storage cannot complete an operation."""


@dataclass(frozen=True)
class HistoryRecord:
    """One saved analytics request.

    Deliberately has no ``sql`` field: the public model cannot leak a statement
    it never carries.
    """

    history_id: str
    created_at: datetime
    original_question: str
    answer: Optional[str]
    row_count: Optional[int]
    truncated: bool
    max_rows: Optional[int]
    chart_requested: bool
    chart_type: Optional[str]
    chart_filename: Optional[str]
    chart_note: Optional[str]
    answer_fallback_used: bool
    refused: bool
    success: bool
    elapsed_seconds: Optional[float]
    error_code: Optional[str] = None


def _database_path() -> Path:
    """Resolve the history database path at call time so tests can redirect it."""
    return Path(HISTORY_DATABASE)


def _connect() -> duckdb.DuckDBPyConnection:
    """Open a short-lived connection, creating the parent directory if needed."""
    path = _database_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(path))
    except Exception as exc:
        logger.error("Unable to open the history database: %s", exc)
        raise HistoryError("Unable to open the history database.") from exc


def initialize_history_database() -> None:
    """Create the history table and index if they do not already exist."""
    with _lock:
        connection = _connect()
        try:
            connection.execute(_CREATE_TABLE_SQL)
            connection.execute(_CREATE_INDEX_SQL)
        except Exception as exc:
            logger.error("Unable to initialize history storage: %s", exc)
            raise HistoryError("Unable to initialize history storage.") from exc
        finally:
            connection.close()
    logger.info("History storage ready at '%s'.", _database_path().name)


def safe_chart_filename(value: Any) -> Optional[str]:
    """Return a bare PNG filename, or None when the value is not one.

    Applied on write and again on read, so a record written by an older build
    cannot introduce a path into a URL.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        value = value.name
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    # Reject anything carrying directory structure before pattern matching.
    if not candidate or "/" in candidate or "\\" in candidate:
        return None
    return candidate if _CHART_FILENAME_RE.fullmatch(candidate) else None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _as_utc(value: Any) -> datetime:
    """Normalize a stored timestamp to timezone-aware UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def save_history(
    *,
    original_question: str,
    answer: Optional[str],
    row_count: Optional[int] = None,
    truncated: bool = False,
    max_rows: Optional[int] = None,
    chart_requested: bool = False,
    chart_type: Optional[str] = None,
    chart_filename: Any = None,
    chart_note: Optional[str] = None,
    answer_fallback_used: bool = False,
    refused: bool = False,
    success: bool = True,
    elapsed_seconds: Optional[float] = None,
    error_code: Optional[str] = None,
) -> str:
    """Persist one completed analytics request and return its generated id.

    The id is a server-generated UUID and never derived from the question.
    """
    if not isinstance(original_question, str) or not original_question.strip():
        raise HistoryError("A question is required to save history.")

    history_id = str(uuid.uuid4())
    # DuckDB TIMESTAMP is timezone-naive: handing it an aware value converts to
    # local time, so every timestamp would come back shifted by the machine's
    # UTC offset. Store the naive UTC wall time and re-attach UTC on read.
    created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    stored_filename = safe_chart_filename(chart_filename)

    with _lock:
        connection = _connect()
        try:
            connection.execute(_CREATE_TABLE_SQL)
            connection.execute(
                f"""
                INSERT INTO {HISTORY_TABLE} (
                    history_id, created_at, original_question, answer, row_count,
                    truncated, max_rows, chart_requested, chart_type,
                    chart_filename, chart_note, answer_fallback_used, refused,
                    success, elapsed_seconds, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    history_id,
                    created_at,
                    original_question,
                    answer,
                    _coerce_int(row_count),
                    bool(truncated),
                    _coerce_int(max_rows),
                    bool(chart_requested),
                    chart_type if isinstance(chart_type, str) else None,
                    stored_filename,
                    chart_note if isinstance(chart_note, str) else None,
                    bool(answer_fallback_used),
                    bool(refused),
                    bool(success),
                    _coerce_float(elapsed_seconds),
                    error_code if isinstance(error_code, str) else None,
                ],
            )
        except Exception as exc:
            logger.error("Unable to save history: %s", exc)
            raise HistoryError("Unable to save history.") from exc
        finally:
            connection.close()

    logger.info("History saved (id=%s, chart=%s).", history_id, stored_filename is not None)
    return history_id


def list_history(limit: int = HISTORY_DEFAULT_LIMIT) -> list[HistoryRecord]:
    """Return saved history newest first, bounded by the server-side maximum."""
    resolved = _coerce_int(limit)
    if resolved is None or resolved < 1:
        resolved = HISTORY_DEFAULT_LIMIT
    resolved = min(resolved, HISTORY_MAX_LIMIT)

    with _lock:
        connection = _connect()
        try:
            connection.execute(_CREATE_TABLE_SQL)
            rows = connection.execute(
                f"""
                SELECT history_id, created_at, original_question, answer, row_count,
                       truncated, max_rows, chart_requested, chart_type,
                       chart_filename, chart_note, answer_fallback_used, refused,
                       success, elapsed_seconds, error_code
                FROM {HISTORY_TABLE}
                ORDER BY created_at DESC, history_id DESC
                LIMIT ?
                """,
                [resolved],
            ).fetchall()
        except Exception as exc:
            logger.error("Unable to read history: %s", exc)
            raise HistoryError("Unable to read history.") from exc
        finally:
            connection.close()

    return [
        HistoryRecord(
            history_id=str(row[0]),
            created_at=_as_utc(row[1]),
            original_question=row[2] if isinstance(row[2], str) else "",
            answer=row[3] if isinstance(row[3], str) else None,
            row_count=_coerce_int(row[4]),
            truncated=bool(row[5]),
            max_rows=_coerce_int(row[6]),
            chart_requested=bool(row[7]),
            chart_type=row[8] if isinstance(row[8], str) else None,
            # Re-validated on read so a legacy or hand-edited row cannot supply a path.
            chart_filename=safe_chart_filename(row[9]),
            chart_note=row[10] if isinstance(row[10], str) else None,
            answer_fallback_used=bool(row[11]),
            refused=bool(row[12]),
            success=bool(row[13]),
            elapsed_seconds=_coerce_float(row[14]),
            error_code=row[15] if isinstance(row[15], str) else None,
        )
        for row in rows
    ]


def clear_history() -> int:
    """Delete every saved history row and return how many were removed.

    Only rows in the history table are affected. The analytics database, chart
    files, and logs are untouched.
    """
    with _lock:
        connection = _connect()
        try:
            connection.execute(_CREATE_TABLE_SQL)
            remaining = connection.execute(
                f"SELECT COUNT(*) FROM {HISTORY_TABLE}"
            ).fetchone()[0]
            connection.execute(f"DELETE FROM {HISTORY_TABLE}")
        except Exception as exc:
            logger.error("Unable to clear history: %s", exc)
            raise HistoryError("Unable to clear history.") from exc
        finally:
            connection.close()

    deleted = int(remaining or 0)
    logger.info("History cleared (%s records removed).", deleted)
    return deleted
