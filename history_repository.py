"""Persistent legacy history and conversation storage.

The history database is deliberately independent from analytics.duckdb. It
contains only presentation-safe records. Generated SQL, database schema,
prompts, provider payloads, result frames, and chart paths never enter this
module's conversation tables. Chart references remain bare PNG filenames and
are validated on both write and read.

The original query_history table remains a compatibility archive for the CLI
and older callers.  Browser conversations use the tables below.  Existing
query-history records are migrated into one-message-pair conversations by an
idempotent mapping table; the original records are never changed or removed.
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
    CONVERSATION_CONTEXT_MAX_CHARS,
    CONVERSATION_CONTEXT_MAX_MESSAGE_CHARS,
    CONVERSATION_CONTEXT_MAX_MESSAGES,
    CONVERSATION_DEFAULT_LIMIT,
    CONVERSATION_MESSAGE_LIMIT,
    CONVERSATION_TITLE_MAX_CHARS,
    GROQ_API_KEY,
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

# Conversation data shares history.duckdb but not the legacy table.  Keeping
# the names explicit also prevents accidental overlap with application data.
CONVERSATIONS_TABLE = "conversations"
CONVERSATION_MESSAGES_TABLE = "conversation_messages"
LEGACY_HISTORY_MIGRATIONS_TABLE = "legacy_history_migrations"

# The same configuration is imported above and used here as a defensive
# repository boundary; callers cannot request a larger persisted transcript.
CONVERSATION_MESSAGE_DEFAULT_LIMIT = CONVERSATION_MESSAGE_LIMIT
CONVERSATION_MESSAGE_MAX_LIMIT = CONVERSATION_MESSAGE_LIMIT
CONVERSATION_TEXT_MAX_CHARS = 4000
CONVERSATION_METADATA_TEXT_MAX_CHARS = 1000
_NEW_CONVERSATION_TITLE = "New chat"

# A stored chart reference must be a plain PNG filename, never a path.
_CHART_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.png$", re.IGNORECASE)

# Conversation text is user-visible.  These redactions protect against an
# accidental secret or local path in a question, answer, or legacy row without
# turning the conversation store into a log of internal details.
_GROQ_KEY_RE = re.compile(r"\bgsk_[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
_API_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
_BEARER_TOKEN_RE = re.compile(
    r"\bbearer\s+[A-Za-z0-9._~+/=-]{6,}\b", re.IGNORECASE
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:(?:[A-Za-z0-9]+_)?api[_ -]?key|"
    r"(?:[A-Za-z0-9]+_)?access[_ -]?token|auth(?:orization)?|bearer|"
    r"password|secret)\b\s*(?:=|:)\s*(?:['\"])?[A-Za-z0-9_./+=-]{6,}",
    re.IGNORECASE,
)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s'\"<>]+|/(?:home|users|etc|app|tmp|root|opt|usr|var|"
    r"mnt|srv|bin|lib|private)(?:/[^\s'\"<>]*)?)",
    re.IGNORECASE,
)

# A direct statement typed into the browser is not a conversational business
# question.  It remains visible in the legacy archive where applicable, but
# is not copied into the new message store.  This is intentionally narrower
# than the metadata safety gate: natural-language refusals such as "Show the
# database schema" remain normal visible chat messages.
_RAW_SQL_START_RE = re.compile(
    r"^\s*(?:select|with|insert|update|delete|drop|alter|create|truncate|"
    r"merge|attach|detach|copy|pragma|vacuum|install|load|grant|revoke|"
    r"exec(?:ute)?)\b",
    re.IGNORECASE,
)
_INTERNAL_RESPONSE_RE = re.compile(
    r"\b(?:select\s+[\s\S]{0,240}\s+from|with\s+[\s\S]{0,240}\s+select|"
    r"(?:create|drop|alter)\s+table|information_schema|duckdb_(?:tables|"
    r"columns|settings)|(?:system|developer)\s+prompt|raw\s+provider\s+payload)\b",
    re.IGNORECASE,
)
_TITLE_UNSAFE_RE = re.compile(
    r"\b(?:select|insert|update|delete|drop|alter|create|truncate|"
    r"merge|attach|detach|copy|pragma|vacuum|grant|revoke|exec(?:ute)?|"
    r"sql|schema|column|field|table|database|prompt|provider|model|"
    r"(?:[A-Za-z0-9]+_)?api[_ -]?key|token|password|secret|implementation|architecture|"
    r"framework|flask|docker|python|repository|codebase|configuration|"
    r"config|filesystem|file\s+path|logs?)\b|"
    r"\bwith\s+[A-Za-z_][A-Za-z0-9_]*\s+as\b|"
    r"(?:gsk_|sk-)|[A-Za-z]:[\\/]|/(?:home|users|etc|app|tmp|root|opt|usr|var|"
    r"mnt|srv|bin|lib|private)(?:/|\b)",
    re.IGNORECASE,
)
_CONTEXT_UNSAFE_RE = re.compile(
    r"\b(?:select\s+[\s\S]{0,240}\s+from|with\s+[\s\S]{0,240}\s+select|"
    r"insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|"
    r"create\s+table|information_schema|duckdb_(?:tables|columns|settings)|"
    r"(?:system|developer)\s+prompt|raw\s+provider\s+payload|"
    r"(?:[A-Za-z0-9]+_)?api[_ -]?key|"
    r"(?:[A-Za-z0-9]+_)?access[_ -]?token|password|secret)\b|"
    r"(?:gsk_|sk-)|[A-Za-z]:[\\/]|/(?:home|users|etc|app|tmp|root|opt|usr|var|"
    r"mnt|srv|bin|lib|private)(?:/|\b)",
    re.IGNORECASE,
)

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

_CREATE_CONVERSATIONS_SQL = f"""
CREATE TABLE IF NOT EXISTS {CONVERSATIONS_TABLE} (
    conversation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
)
"""

_CREATE_CONVERSATION_MESSAGES_SQL = f"""
CREATE TABLE IF NOT EXISTS {CONVERSATION_MESSAGES_TABLE} (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    ordinal BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    row_count BIGINT,
    truncated BOOLEAN NOT NULL DEFAULT FALSE,
    max_rows BIGINT,
    chart_requested BOOLEAN NOT NULL DEFAULT FALSE,
    chart_type TEXT,
    chart_filename TEXT,
    chart_note TEXT,
    answer_fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
    refused BOOLEAN NOT NULL DEFAULT FALSE,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    elapsed_seconds DOUBLE
)
"""

_CREATE_LEGACY_MIGRATIONS_SQL = f"""
CREATE TABLE IF NOT EXISTS {LEGACY_HISTORY_MIGRATIONS_TABLE} (
    history_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    migrated_at TIMESTAMP NOT NULL
)
"""

_CREATE_CONVERSATIONS_UPDATED_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_{CONVERSATIONS_TABLE}_updated_at "
    f"ON {CONVERSATIONS_TABLE} (updated_at)"
)
_CREATE_CONVERSATION_MESSAGES_ORDER_INDEX_SQL = (
    f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{CONVERSATION_MESSAGES_TABLE}_order "
    f"ON {CONVERSATION_MESSAGES_TABLE} (conversation_id, ordinal)"
)
_CREATE_LEGACY_MIGRATIONS_CONVERSATION_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS idx_{LEGACY_HISTORY_MIGRATIONS_TABLE}_conversation "
    f"ON {LEGACY_HISTORY_MIGRATIONS_TABLE} (conversation_id)"
)


class HistoryError(RuntimeError):
    """Raised when history storage cannot complete an operation."""


@dataclass(frozen=True)
class HistoryRecord:
    """One saved legacy analytics request.

    Deliberately has no sql field: the public model cannot leak a statement it
    never carries.
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


@dataclass(frozen=True)
class ConversationSummary:
    """Safe list representation of a persisted conversation."""

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int


@dataclass(frozen=True)
class ConversationMessage:
    """One persisted display message and its safe analytics metadata."""

    message_id: str
    conversation_id: str
    ordinal: int
    role: str
    content: str
    created_at: datetime
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


@dataclass(frozen=True)
class Conversation:
    """A loaded transcript.  Result rows are intentionally absent."""

    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: tuple[ConversationMessage, ...]
    messages_truncated: bool


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


def _utc_now_naive() -> datetime:
    """Return a DuckDB TIMESTAMP-compatible UTC wall time."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_utc(value: Any) -> datetime:
    """Normalize a stored timestamp to timezone-aware UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _as_naive_utc(value: Any) -> datetime:
    """Convert a stored or supplied time to DuckDB's UTC-naive TIMESTAMP form."""
    return _as_utc(value).replace(tzinfo=None)


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


def _clip_text(value: str, limit: int) -> str:
    """Keep untrusted display text bounded without splitting a trailing space."""
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip() + "…"


def _redact_text(value: str) -> str:
    """Remove obvious credentials and absolute paths from persisted display text."""
    text = value.replace("\x00", "")
    if GROQ_API_KEY:
        text = text.replace(GROQ_API_KEY, "[REDACTED]")
    text = _GROQ_KEY_RE.sub("[REDACTED]", text)
    text = _API_KEY_RE.sub("[REDACTED]", text)
    text = _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text)
    text = _SECRET_ASSIGNMENT_RE.sub("[REDACTED]", text)
    return _ABSOLUTE_PATH_RE.sub("[REDACTED PATH]", text)


def _safe_message_content(value: Any, *, role: str) -> str:
    """Return bounded text that is safe to retain as a chat message.

    This accepts only display text.  It never serializes objects such as a
    result frame or provider response.  Direct SQL is replaced rather than
    copied into the conversation store; natural-language safety refusals remain
    visible because they do not begin with a statement.
    """
    if role not in {"user", "assistant"}:
        raise HistoryError("Conversation messages must have a supported role.")

    if not isinstance(value, str):
        value = ""
    text = _redact_text(value).strip()
    if not text:
        return (
            "No saved assistant answer is available for this turn."
            if role == "assistant"
            else "A submitted question could not be retained safely."
        )

    if _RAW_SQL_START_RE.search(text):
        return (
            "The assistant response was not retained because it contained unsafe internal text."
            if role == "assistant"
            else "A direct database command was not retained in this conversation."
        )

    # The assistant should never return internal implementation material.  A
    # normal refusal that merely explains a boundary is still retained.
    if role == "assistant" and _INTERNAL_RESPONSE_RE.search(text):
        return "The assistant response was not retained because it contained unsafe internal text."

    return _clip_text(text, CONVERSATION_TEXT_MAX_CHARS)


def _safe_metadata_text(value: Any) -> Optional[str]:
    """Return bounded chart metadata without credentials or filesystem paths."""
    if not isinstance(value, str):
        return None
    text = _redact_text(value).strip()
    if not text or _INTERNAL_RESPONSE_RE.search(text):
        return None
    return _clip_text(text, CONVERSATION_METADATA_TEXT_MAX_CHARS)


def safe_conversation_title(question: Any) -> str:
    """Make a local, non-model title from a first question.

    Conversation titles are deliberately conservative.  A question that looks
    like SQL, metadata, a secret, or an implementation request gets the neutral
    title instead of exposing that material in the sidebar.
    """
    if not isinstance(question, str):
        return _NEW_CONVERSATION_TITLE
    if not question.strip() or _TITLE_UNSAFE_RE.search(question):
        return _NEW_CONVERSATION_TITLE
    text = _redact_text(question)
    if not text:
        return _NEW_CONVERSATION_TITLE
    collapsed = re.sub(r"\s+", " ", text).strip()
    return _clip_text(collapsed, CONVERSATION_TITLE_MAX_CHARS) if collapsed else _NEW_CONVERSATION_TITLE


def normalize_conversation_id(value: Any) -> str:
    """Validate and canonicalize a server-generated UUID conversation id."""
    if not isinstance(value, str):
        raise HistoryError("Conversation id is invalid.")
    candidate = value.strip()
    try:
        normalized = str(uuid.UUID(candidate))
    except (AttributeError, ValueError, TypeError) as exc:
        raise HistoryError("Conversation id is invalid.") from exc
    if candidate.lower() != normalized:
        raise HistoryError("Conversation id is invalid.")
    return normalized


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


def _prepare_legacy_storage(connection: duckdb.DuckDBPyConnection) -> None:
    """Ensure the original compatibility table exists."""
    connection.execute(_CREATE_TABLE_SQL)
    connection.execute(_CREATE_INDEX_SQL)


def _prepare_conversation_storage(connection: duckdb.DuckDBPyConnection) -> int:
    """Ensure conversation tables exist and migrate any unseen legacy rows."""
    _prepare_legacy_storage(connection)
    connection.execute(_CREATE_CONVERSATIONS_SQL)
    connection.execute(_CREATE_CONVERSATION_MESSAGES_SQL)
    connection.execute(_CREATE_LEGACY_MIGRATIONS_SQL)
    connection.execute(_CREATE_CONVERSATIONS_UPDATED_INDEX_SQL)
    connection.execute(_CREATE_CONVERSATION_MESSAGES_ORDER_INDEX_SQL)
    connection.execute(_CREATE_LEGACY_MIGRATIONS_CONVERSATION_INDEX_SQL)
    return _migrate_legacy_history(connection)


def _insert_message(
    connection: duckdb.DuckDBPyConnection,
    *,
    message_id: str,
    conversation_id: str,
    ordinal: int,
    role: str,
    content: str,
    created_at: datetime,
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
) -> None:
    """Insert one already-sanitized message using bound values only."""
    connection.execute(
        f"""
        INSERT INTO {CONVERSATION_MESSAGES_TABLE} (
            message_id, conversation_id, ordinal, role, content, created_at,
            row_count, truncated, max_rows, chart_requested, chart_type,
            chart_filename, chart_note, answer_fallback_used, refused, success,
            elapsed_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            message_id,
            conversation_id,
            ordinal,
            role,
            content,
            created_at,
            _coerce_int(row_count),
            bool(truncated),
            _coerce_int(max_rows),
            bool(chart_requested),
            _safe_metadata_text(chart_type),
            safe_chart_filename(chart_filename),
            _safe_metadata_text(chart_note),
            bool(answer_fallback_used),
            bool(refused),
            bool(success),
            _coerce_float(elapsed_seconds),
        ],
    )


def _migrate_legacy_history(connection: duckdb.DuckDBPyConnection) -> int:
    """Copy each unmapped legacy row into one safe two-message conversation.

    The mapping row is inserted with the conversation and both messages in one
    transaction.  A crash or constraint error therefore leaves no partial
    migration and a later initialization can retry safely.
    """
    legacy_rows = connection.execute(
        f"""
        SELECT h.history_id, h.created_at, h.original_question, h.answer,
               h.row_count, h.truncated, h.max_rows, h.chart_requested,
               h.chart_type, h.chart_filename, h.chart_note,
               h.answer_fallback_used, h.refused, h.success,
               h.elapsed_seconds, h.error_code
        FROM {HISTORY_TABLE} AS h
        LEFT JOIN {LEGACY_HISTORY_MIGRATIONS_TABLE} AS m
          ON m.history_id = h.history_id
        WHERE m.history_id IS NULL
        ORDER BY h.created_at ASC, h.history_id ASC
        """
    ).fetchall()
    if not legacy_rows:
        return 0

    migrated_at = _utc_now_naive()
    connection.execute("BEGIN TRANSACTION")
    try:
        for row in legacy_rows:
            history_id = str(row[0])
            created_at = _as_naive_utc(row[1])
            conversation_id = str(uuid.uuid4())
            user_content = _safe_message_content(row[2], role="user")
            legacy_answer = (
                row[3]
                if isinstance(row[3], str) and row[3].strip()
                else "No saved assistant answer is available for this legacy request."
            )
            assistant_content = _safe_message_content(legacy_answer, role="assistant")
            title = safe_conversation_title(row[2])

            connection.execute(
                f"""
                INSERT INTO {CONVERSATIONS_TABLE} (
                    conversation_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                [conversation_id, title, created_at, created_at],
            )
            _insert_message(
                connection,
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                ordinal=1,
                role="user",
                content=user_content,
                created_at=created_at,
            )
            _insert_message(
                connection,
                message_id=str(uuid.uuid4()),
                conversation_id=conversation_id,
                ordinal=2,
                role="assistant",
                content=assistant_content,
                created_at=created_at,
                row_count=_coerce_int(row[4]),
                truncated=bool(row[5]),
                max_rows=_coerce_int(row[6]),
                chart_requested=bool(row[7]),
                chart_type=row[8] if isinstance(row[8], str) else None,
                chart_filename=row[9],
                chart_note=row[10] if isinstance(row[10], str) else None,
                answer_fallback_used=bool(row[11]),
                refused=bool(row[12]),
                success=bool(row[13]),
                elapsed_seconds=_coerce_float(row[14]),
            )
            connection.execute(
                f"""
                INSERT INTO {LEGACY_HISTORY_MIGRATIONS_TABLE} (
                    history_id, conversation_id, migrated_at
                ) VALUES (?, ?, ?)
                """,
                [history_id, conversation_id, migrated_at],
            )
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")

    logger.info("Migrated %s legacy history record(s) into conversations.", len(legacy_rows))
    return len(legacy_rows)


def initialize_history_database() -> None:
    """Create legacy and conversation tables and migrate unseen legacy records."""
    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
        except Exception as exc:
            logger.error("Unable to initialize history storage: %s", exc)
            raise HistoryError("Unable to initialize history storage.") from exc
        finally:
            connection.close()
    logger.info("History storage ready at '%s'.", _database_path().name)


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
    """Persist one legacy completed analytics request and return its UUID.

    This compatibility API intentionally remains flat for the CLI.  The next
    conversation initialization migrates any unseen row exactly once.
    """
    if not isinstance(original_question, str) or not original_question.strip():
        raise HistoryError("A question is required to save history.")

    history_id = str(uuid.uuid4())
    created_at = _utc_now_naive()
    stored_filename = safe_chart_filename(chart_filename)

    with _lock:
        connection = _connect()
        try:
            _prepare_legacy_storage(connection)
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


def _history_record_from_row(row: tuple[Any, ...]) -> HistoryRecord:
    """Build a safe legacy model while revalidating chart filenames."""
    return HistoryRecord(
        history_id=str(row[0]),
        created_at=_as_utc(row[1]),
        original_question=row[2] if isinstance(row[2], str) else "",
        answer=row[3] if isinstance(row[3], str) else None,
        row_count=_coerce_int(row[4]),
        truncated=bool(row[5]),
        max_rows=_coerce_int(row[6]),
        chart_requested=bool(row[7]),
        chart_type=row[8] if isinstance(row[8], str) else None,
        chart_filename=safe_chart_filename(row[9]),
        chart_note=row[10] if isinstance(row[10], str) else None,
        answer_fallback_used=bool(row[11]),
        refused=bool(row[12]),
        success=bool(row[13]),
        elapsed_seconds=_coerce_float(row[14]),
        error_code=row[15] if isinstance(row[15], str) else None,
    )


def list_history(limit: int = HISTORY_DEFAULT_LIMIT) -> list[HistoryRecord]:
    """Return legacy records newest first, bounded by the server-side maximum."""
    resolved = _coerce_int(limit)
    if resolved is None or resolved < 1:
        resolved = HISTORY_DEFAULT_LIMIT
    resolved = min(resolved, HISTORY_MAX_LIMIT)

    with _lock:
        connection = _connect()
        try:
            _prepare_legacy_storage(connection)
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

    return [_history_record_from_row(row) for row in rows]


def clear_history() -> int:
    """Delete legacy rows only, leaving conversations and all files untouched."""
    with _lock:
        connection = _connect()
        try:
            _prepare_legacy_storage(connection)
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
    logger.info("Legacy history cleared (%s records removed).", deleted)
    return deleted


def _summary_from_row(row: tuple[Any, ...]) -> ConversationSummary:
    """Convert a conversation summary query row into a safe dataclass."""
    return ConversationSummary(
        conversation_id=str(row[0]),
        title=row[1] if isinstance(row[1], str) and row[1] else _NEW_CONVERSATION_TITLE,
        created_at=_as_utc(row[2]),
        updated_at=_as_utc(row[3]),
        message_count=max(0, _coerce_int(row[4]) or 0),
    )


def _message_from_row(row: tuple[Any, ...]) -> ConversationMessage:
    """Convert a stored row and revalidate the bare chart filename."""
    role = row[3] if row[3] in {"user", "assistant"} else "assistant"
    return ConversationMessage(
        message_id=str(row[0]),
        conversation_id=str(row[1]),
        ordinal=max(0, _coerce_int(row[2]) or 0),
        role=role,
        content=_safe_message_content(row[4], role=role),
        created_at=_as_utc(row[5]),
        row_count=_coerce_int(row[6]),
        truncated=bool(row[7]),
        max_rows=_coerce_int(row[8]),
        chart_requested=bool(row[9]),
        chart_type=row[10] if isinstance(row[10], str) else None,
        chart_filename=safe_chart_filename(row[11]),
        chart_note=_safe_metadata_text(row[12]),
        answer_fallback_used=bool(row[13]),
        refused=bool(row[14]),
        success=bool(row[15]),
        elapsed_seconds=_coerce_float(row[16]),
    )


def _conversation_limit(value: Any) -> int:
    """Resolve a public list limit without permitting an unbounded response."""
    resolved = _coerce_int(value)
    if resolved is None or resolved < 1:
        return CONVERSATION_DEFAULT_LIMIT
    return min(resolved, HISTORY_MAX_LIMIT)


def _message_limit(value: Any) -> int:
    """Resolve a transcript limit with a distinct, bounded maximum."""
    resolved = _coerce_int(value)
    if resolved is None or resolved < 1:
        return CONVERSATION_MESSAGE_DEFAULT_LIMIT
    return min(resolved, CONVERSATION_MESSAGE_MAX_LIMIT)


def create_conversation(title: Optional[str] = None) -> ConversationSummary:
    """Create and return an empty persistent conversation.

    Callers do not need to supply a title.  A user-supplied title is sanitized
    with the same local title policy as a first question.
    """
    resolved_title = (
        safe_conversation_title(title) if isinstance(title, str) and title.strip() else _NEW_CONVERSATION_TITLE
    )
    conversation_id = str(uuid.uuid4())
    created_at = _utc_now_naive()

    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            connection.execute(
                f"""
                INSERT INTO {CONVERSATIONS_TABLE} (
                    conversation_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                [conversation_id, resolved_title, created_at, created_at],
            )
        except Exception as exc:
            logger.error("Unable to create conversation: %s", exc)
            raise HistoryError("Unable to create conversation.") from exc
        finally:
            connection.close()

    logger.info("Conversation created (id=%s).", conversation_id)
    return ConversationSummary(
        conversation_id=conversation_id,
        title=resolved_title,
        created_at=_as_utc(created_at),
        updated_at=_as_utc(created_at),
        message_count=0,
    )


def list_conversations(limit: int = CONVERSATION_DEFAULT_LIMIT) -> list[ConversationSummary]:
    """List conversation summaries newest first without loading transcripts."""
    resolved = _conversation_limit(limit)

    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            rows = connection.execute(
                f"""
                SELECT c.conversation_id, c.title, c.created_at, c.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM {CONVERSATIONS_TABLE} AS c
                LEFT JOIN {CONVERSATION_MESSAGES_TABLE} AS m
                  ON m.conversation_id = c.conversation_id
                GROUP BY c.conversation_id, c.title, c.created_at, c.updated_at
                ORDER BY c.updated_at DESC, c.conversation_id DESC
                LIMIT ?
                """,
                [resolved],
            ).fetchall()
        except Exception as exc:
            logger.error("Unable to list conversations: %s", exc)
            raise HistoryError("Unable to list conversations.") from exc
        finally:
            connection.close()

    return [_summary_from_row(row) for row in rows]


def load_conversation(
    conversation_id: str,
    limit: int = CONVERSATION_MESSAGE_DEFAULT_LIMIT,
) -> Optional[Conversation]:
    """Load saved messages only; this function never invokes analytics or a model."""
    normalized_id = normalize_conversation_id(conversation_id)
    resolved_limit = _message_limit(limit)

    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            header = connection.execute(
                f"""
                SELECT c.conversation_id, c.title, c.created_at, c.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM {CONVERSATIONS_TABLE} AS c
                LEFT JOIN {CONVERSATION_MESSAGES_TABLE} AS m
                  ON m.conversation_id = c.conversation_id
                WHERE c.conversation_id = ?
                GROUP BY c.conversation_id, c.title, c.created_at, c.updated_at
                """,
                [normalized_id],
            ).fetchone()
            if header is None:
                return None

            rows = connection.execute(
                f"""
                SELECT message_id, conversation_id, ordinal, role, content,
                       created_at, row_count, truncated, max_rows,
                       chart_requested, chart_type, chart_filename, chart_note,
                       answer_fallback_used, refused, success, elapsed_seconds
                FROM (
                    SELECT message_id, conversation_id, ordinal, role, content,
                           created_at, row_count, truncated, max_rows,
                           chart_requested, chart_type, chart_filename, chart_note,
                           answer_fallback_used, refused, success, elapsed_seconds
                    FROM {CONVERSATION_MESSAGES_TABLE}
                    WHERE conversation_id = ?
                    ORDER BY ordinal DESC
                    LIMIT ?
                ) AS recent_messages
                ORDER BY ordinal ASC
                """,
                [normalized_id, resolved_limit],
            ).fetchall()
        except HistoryError:
            raise
        except Exception as exc:
            logger.error("Unable to load conversation: %s", exc)
            raise HistoryError("Unable to load conversation.") from exc
        finally:
            connection.close()

    summary = _summary_from_row(header)
    return Conversation(
        conversation_id=summary.conversation_id,
        title=summary.title,
        created_at=summary.created_at,
        updated_at=summary.updated_at,
        message_count=summary.message_count,
        messages=tuple(_message_from_row(row) for row in rows),
        messages_truncated=summary.message_count > len(rows),
    )


def save_conversation_turn(
    conversation_id: Optional[str],
    user_content: str,
    assistant_content: Optional[str],
    *,
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
    update_title: bool = True,
) -> ConversationSummary:
    """Atomically append a user and assistant message pair.

    Passing None creates a conversation lazily within the same transaction, so
    a failed persistence operation cannot leave either an empty new chat or an
    orphan user message.  A supplied id must already exist.
    """
    if not isinstance(user_content, str) or not user_content.strip():
        raise HistoryError("A question is required to save a conversation turn.")
    if not isinstance(assistant_content, str) or not assistant_content.strip():
        raise HistoryError("An assistant answer is required to save a conversation turn.")
    if not isinstance(update_title, bool):
        raise HistoryError("Conversation title policy is invalid.")

    # Only an omitted value requests lazy creation.  An empty or malformed
    # supplied value must not silently create a different conversation.
    normalized_id = (
        normalize_conversation_id(conversation_id)
        if conversation_id is not None
        else None
    )
    safe_user = _safe_message_content(user_content, role="user")
    safe_assistant = _safe_message_content(assistant_content, role="assistant")
    completed_at = _utc_now_naive()

    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                if normalized_id is None:
                    resolved_id = str(uuid.uuid4())
                    title = (
                        safe_conversation_title(user_content)
                        if update_title
                        else _NEW_CONVERSATION_TITLE
                    )
                    connection.execute(
                        f"""
                        INSERT INTO {CONVERSATIONS_TABLE} (
                            conversation_id, title, created_at, updated_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        [resolved_id, title, completed_at, completed_at],
                    )
                    prior_messages = 0
                else:
                    existing = connection.execute(
                        f"""
                        SELECT title, created_at
                        FROM {CONVERSATIONS_TABLE}
                        WHERE conversation_id = ?
                        """,
                        [normalized_id],
                    ).fetchone()
                    if existing is None:
                        raise HistoryError("Conversation was not found.")
                    resolved_id = normalized_id
                    title = existing[0] if isinstance(existing[0], str) else _NEW_CONVERSATION_TITLE
                    prior_messages = int(
                        connection.execute(
                            f"""
                            SELECT COUNT(*)
                            FROM {CONVERSATION_MESSAGES_TABLE}
                            WHERE conversation_id = ?
                            """,
                            [resolved_id],
                        ).fetchone()[0]
                        or 0
                    )

                next_ordinal = prior_messages + 1
                _insert_message(
                    connection,
                    message_id=str(uuid.uuid4()),
                    conversation_id=resolved_id,
                    ordinal=next_ordinal,
                    role="user",
                    content=safe_user,
                    created_at=completed_at,
                )
                _insert_message(
                    connection,
                    message_id=str(uuid.uuid4()),
                    conversation_id=resolved_id,
                    ordinal=next_ordinal + 1,
                    role="assistant",
                    content=safe_assistant,
                    created_at=completed_at,
                    row_count=row_count,
                    truncated=truncated,
                    max_rows=max_rows,
                    chart_requested=chart_requested,
                    chart_type=chart_type,
                    chart_filename=chart_filename,
                    chart_note=chart_note,
                    answer_fallback_used=answer_fallback_used,
                    refused=refused,
                    success=success,
                    elapsed_seconds=elapsed_seconds,
                )

                # A social opening may deliberately leave the neutral title in
                # place. The first later analytics turn can then supply the title;
                # once a meaningful title exists, follow-ups never rename it.
                updated_title = (
                    safe_conversation_title(user_content)
                    if update_title and title == _NEW_CONVERSATION_TITLE
                    else title
                )
                connection.execute(
                    f"""
                    UPDATE {CONVERSATIONS_TABLE}
                    SET title = ?, updated_at = ?
                    WHERE conversation_id = ?
                    """,
                    [updated_title, completed_at, resolved_id],
                )
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
        except HistoryError:
            raise
        except Exception as exc:
            logger.error("Unable to save conversation turn: %s", exc)
            raise HistoryError("Unable to save conversation turn.") from exc
        finally:
            connection.close()

    logger.info("Conversation turn saved (id=%s).", resolved_id)
    return ConversationSummary(
        conversation_id=resolved_id,
        title=updated_title,
        created_at=_as_utc(completed_at if normalized_id is None else existing[1]),
        updated_at=_as_utc(completed_at),
        message_count=prior_messages + 2,
    )


def _context_text(value: Any) -> str:
    """Return a safe semantic-context fragment, never internal material."""
    if not isinstance(value, str):
        return ""
    text = _redact_text(value).strip()
    if not text or _CONTEXT_UNSAFE_RE.search(text):
        return ""
    return _clip_text(text, CONVERSATION_CONTEXT_MAX_MESSAGE_CHARS)


def get_conversation_context(conversation_id: str) -> str:
    """Return the bounded recent semantic context for one known conversation.

    Only user and assistant display text is included.  The newest useful
    messages win when the character budget is exhausted, while the final result
    is ordered chronologically for the routing model.
    """
    normalized_id = normalize_conversation_id(conversation_id)

    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            exists = connection.execute(
                f"SELECT 1 FROM {CONVERSATIONS_TABLE} WHERE conversation_id = ?",
                [normalized_id],
            ).fetchone()
            if exists is None:
                raise HistoryError("Conversation was not found.")
            rows = connection.execute(
                f"""
                SELECT role, content
                FROM {CONVERSATION_MESSAGES_TABLE}
                WHERE conversation_id = ?
                ORDER BY ordinal DESC
                LIMIT ?
                """,
                [normalized_id, CONVERSATION_CONTEXT_MAX_MESSAGES],
            ).fetchall()
        except HistoryError:
            raise
        except Exception as exc:
            logger.error("Unable to read conversation context: %s", exc)
            raise HistoryError("Unable to read conversation context.") from exc
        finally:
            connection.close()

    remaining = CONVERSATION_CONTEXT_MAX_CHARS
    selected: list[str] = []
    for role, content in rows:
        if role not in {"user", "assistant"}:
            continue
        text = _context_text(content)
        if not text:
            continue
        segment = f"{role.upper()}: {text}"
        if len(segment) > remaining:
            if not selected and remaining > len(role) + 3:
                selected.append(_clip_text(segment, remaining))
            break
        selected.append(segment)
        remaining -= len(segment) + 1
        if remaining <= 0:
            break

    return "\n".join(reversed(selected))


def delete_conversation(conversation_id: str) -> bool:
    """Delete one chat and its messages, leaving charts and legacy data alone."""
    normalized_id = normalize_conversation_id(conversation_id)

    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                exists = connection.execute(
                    f"SELECT 1 FROM {CONVERSATIONS_TABLE} WHERE conversation_id = ?",
                    [normalized_id],
                ).fetchone()
                if exists is None:
                    connection.execute("ROLLBACK")
                    return False
                connection.execute(
                    f"DELETE FROM {CONVERSATION_MESSAGES_TABLE} WHERE conversation_id = ?",
                    [normalized_id],
                )
                connection.execute(
                    f"DELETE FROM {CONVERSATIONS_TABLE} WHERE conversation_id = ?",
                    [normalized_id],
                )
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
        except HistoryError:
            raise
        except Exception as exc:
            logger.error("Unable to delete conversation: %s", exc)
            raise HistoryError("Unable to delete conversation.") from exc
        finally:
            connection.close()

    # Deliberately do not delete migration mappings: old query_history rows must
    # not recreate a chat a user explicitly deleted.  PNGs are likewise left in
    # place because no chart path is ever trusted for deletion.
    logger.info("Conversation deleted (id=%s).", normalized_id)
    return True


def delete_all_conversations() -> int:
    """Delete all conversation rows and messages, never charts or legacy history."""
    with _lock:
        connection = _connect()
        try:
            _prepare_conversation_storage(connection)
            connection.execute("BEGIN TRANSACTION")
            try:
                remaining = connection.execute(
                    f"SELECT COUNT(*) FROM {CONVERSATIONS_TABLE}"
                ).fetchone()[0]
                connection.execute(f"DELETE FROM {CONVERSATION_MESSAGES_TABLE}")
                connection.execute(f"DELETE FROM {CONVERSATIONS_TABLE}")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
        except Exception as exc:
            logger.error("Unable to delete all conversations: %s", exc)
            raise HistoryError("Unable to delete all conversations.") from exc
        finally:
            connection.close()

    deleted = int(remaining or 0)
    logger.info("All conversations deleted (%s records removed).", deleted)
    return deleted
