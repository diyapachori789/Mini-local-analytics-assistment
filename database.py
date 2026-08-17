from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from config import (
    CSV_PATH,
    DATABASE_NAME,
    MAX_RESULT_ROWS,
    RESTRICT_FILE_ACCESS,
    TABLE_NAME,
    TABLE_SCHEMA,
)
from sql_guard import validate_sql

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database module errors."""


class CsvNotFoundError(DatabaseError):
    """Raised when the CSV data file does not exist."""


class SqlExecutionError(DatabaseError):
    """Raised when SQL execution fails."""


class SqlValidationError(SqlExecutionError):
    """Raised when SQL is refused before execution because it is not read-only.

    Subclasses SqlExecutionError so existing callers that handle execution
    failures keep working, while callers that care can distinguish "refused"
    from "ran and failed".
    """


class DatabaseConnectionError(DatabaseError):
    """Raised when the database connection cannot be created."""


def _validate_csv_path(csv_path: Path) -> None:
    """Validate that the CSV path exists and is a file."""
    if not csv_path.exists():
        raise CsvNotFoundError(f"CSV file not found at {csv_path}")
    if not csv_path.is_file():
        raise CsvNotFoundError(f"CSV path is not a file: {csv_path}")


# Active DuckDB connection for the module lifecycle.
_connection: Optional[duckdb.DuckDBPyConnection] = None

# One DuckDB connection is shared by the whole process, and a DuckDB connection
# is not safe to use from two threads at once. The web server used to be single
# threaded, which made that true by accident; it no longer is, so the invariant
# is stated here rather than depending on how the server happens to be
# configured. Held only around the database call itself, never around a model
# request, so a slow answer cannot block another user's query.
_connection_lock = threading.Lock()


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Return the active DuckDB connection or raise if uninitialized."""
    if _connection is None:
        raise DatabaseConnectionError(
            "Database is not initialized. Call initialize_database() first."
        )
    return _connection


def close_connection() -> None:
    """Close the active DuckDB connection and reset module state."""
    global _connection
    if _connection is None:
        return

    try:
        _connection.close()
        logger.info("Database connection closed.")
    finally:
        _connection = None


def _restrict_file_access(conn: duckdb.DuckDBPyConnection) -> None:
    """Disable DuckDB filesystem access once the dataset has been loaded.

    When DuckDB accepts this setting, it is a one-way latch: external access cannot
    be re-enabled while the database is running. Applying it is intentionally
    best-effort, however; startup logs a warning and continues if DuckDB rejects
    the setting, while the SQL validator and database revalidation still apply.
    """
    if not RESTRICT_FILE_ACCESS:
        return

    try:
        conn.execute("SET enable_external_access = false")
        logger.debug("Filesystem access disabled for the DuckDB connection.")
    except Exception as exc:
        # Never fail startup over a hardening step; the SQL validator still applies.
        logger.warning("Could not disable DuckDB filesystem access: %s", exc)


def initialize_database() -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB, validate CSV data, and create the opportunities table."""
    global _connection

    # Close any existing connection before creating a new one.
    close_connection()

    # Validate that the CSV dataset exists.
    _validate_csv_path(CSV_PATH)

    logger.info("Initializing database '%s'.", DATABASE_NAME)

    try:
        _connection = duckdb.connect(DATABASE_NAME)
    except Exception as exc:
        logger.error("Unable to connect to DuckDB database '%s': %s", DATABASE_NAME, exc)
        raise DatabaseConnectionError(
            f"Unable to connect to DuckDB database '{DATABASE_NAME}'."
        ) from exc

    create_table_sql = (
        f"CREATE OR REPLACE TABLE {TABLE_NAME} AS "
        "SELECT * FROM read_csv_auto(?)"
    )

    try:
        _connection.execute(create_table_sql, [str(CSV_PATH)])
    except Exception as exc:
        _connection.close()
        _connection = None
        logger.error("Failed to create table '%s' from '%s': %s", TABLE_NAME, CSV_PATH, exc)
        raise DatabaseError(
            f"Failed to create table '{TABLE_NAME}' from CSV data."
        ) from exc

    # Harden the connection only after the CSV has been read.
    _restrict_file_access(_connection)

    row_count = _connection.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    logger.info(
        "Database initialized: table '%s' loaded with %s rows from '%s'.",
        TABLE_NAME,
        row_count,
        CSV_PATH.name,
    )

    return _connection


@dataclass(frozen=True, eq=False)
class QueryResult:
    """Structured result of a read-only query.

    ``eq=False`` because DataFrame comparison is element-wise and would make the
    generated ``__eq__`` raise on truth-value ambiguity.
    """

    frame: pd.DataFrame
    sql: str
    row_count: int
    truncated: bool
    max_rows: Optional[int]

    @property
    def columns(self) -> list[str]:
        """Column names returned by the query, in order."""
        return list(self.frame.columns)

    @property
    def is_empty(self) -> bool:
        """True when the query ran successfully but matched no rows."""
        return self.row_count == 0


def run_query(sql: str, max_rows: Optional[int] = MAX_RESULT_ROWS) -> QueryResult:
    """Execute a validated, read-only query and return a structured result.

    The statement is validated here even when the caller has already validated
    it. This function is the last checkpoint before DuckDB and must not depend on
    callers having done the right thing: anything that is not a single
    SELECT/WITH statement is refused, so CREATE, DROP, INSERT, UPDATE, DELETE and
    ALTER can never reach the database through this path.

    ``max_rows`` bounds how many rows are read out of DuckDB. The SQL is never
    rewritten to add a LIMIT, because that would change the meaning of a query
    that already has its own LIMIT or ORDER BY. Instead the fetch simply stops,
    and one extra row is read to report whether more were available. Pass
    ``max_rows=None`` to fetch the complete result.
    """
    if not sql or not isinstance(sql, str):
        raise SqlValidationError("SQL query must be a non-empty string.")

    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least 1 when set.")

    try:
        safe_sql = validate_sql(sql)
    except ValueError as exc:
        logger.error("Refused to execute unsafe SQL: %s", exc)
        raise SqlValidationError(f"Refused to execute unsafe SQL: {exc}") from exc

    conn = _get_connection()
    logger.info("SQL execution started (max_rows=%s).", max_rows)

    try:
        # The cursor and its pending rows belong to the connection, so fetching
        # has to stay inside the lock: releasing it after execute() would let a
        # second thread run a query and invalidate the result being read here.
        with _connection_lock:
            result = conn.execute(safe_sql)
            if max_rows is None:
                frame = result.df()
                truncated = False
            else:
                # One row beyond the cap is fetched purely to detect truncation.
                rows = result.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                columns = [description[0] for description in result.description]
                frame = pd.DataFrame(rows[:max_rows], columns=columns)
    except Exception as exc:
        logger.error("SQL execution failed: %s", exc)
        raise SqlExecutionError(f"Failed to execute SQL: {exc}") from exc

    if truncated:
        logger.warning(
            "Result truncated at %s rows; the query matched more.", max_rows
        )
    logger.info("SQL execution succeeded, %s rows returned.", len(frame))

    return QueryResult(
        frame=frame,
        sql=safe_sql,
        row_count=len(frame),
        truncated=truncated,
        max_rows=max_rows,
    )


def execute_query(sql: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Execute a validated, read-only SQL query and return rows as a DataFrame.

    Thin wrapper over :func:`run_query` for callers that only need the rows.
    ``max_rows`` defaults to ``None`` (complete result) so existing behaviour is
    unchanged; use :func:`run_query` when row count or truncation matters.
    """
    return run_query(sql, max_rows=max_rows).frame


def column_identifiers() -> frozenset[str]:
    """Return the table's column names, lower-cased.

    Exposed so the output filters can recognise a real identifier instead of
    keeping their own copy of the schema, which would silently stop protecting
    any column added later. Returns names only - never types, DDL or values -
    and is used to *reject* text, never to build it.

    An unavailable database yields an empty set rather than raising: this
    supports a safety check, and a check that crashes is worse than one that
    falls back to the pattern rules that do not need the schema.
    """
    try:
        conn = _get_connection()
        with _connection_lock:
            rows = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ? AND table_schema = ?",
                [TABLE_NAME, TABLE_SCHEMA],
            ).fetchall()
    except Exception as exc:  # noqa: BLE001 - a filter must not fail closed here
        logger.warning("Column identifiers unavailable for output filtering: %s", exc)
        return frozenset()
    return frozenset(str(row[0]).strip().lower() for row in rows if row and row[0])


def get_schema() -> str:
    """Return the opportunities table schema as Markdown text."""
    conn = _get_connection()

    # Filter by schema as well as table name so an identically named table in
    # another attached schema cannot produce duplicated or wrong columns.
    schema_query = (
        "SELECT column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_name = ? AND table_schema = ? "
        "ORDER BY ordinal_position"
    )

    try:
        with _connection_lock:
            result = conn.execute(schema_query, [TABLE_NAME, TABLE_SCHEMA]).fetchall()
    except Exception as exc:
        logger.error("Unable to retrieve schema for table '%s': %s", TABLE_NAME, exc)
        raise DatabaseError(
            f"Unable to retrieve schema for table '{TABLE_NAME}'."
        ) from exc

    if not result:
        logger.warning(
            "Schema lookup returned no columns for table '%s.%s'.",
            TABLE_SCHEMA,
            TABLE_NAME,
        )
        return ""

    schema_lines = ["| column | type |", "| --- | --- |"]
    for column, data_type in result:
        schema_lines.append(f"| {column} | {data_type} |")

    return "\n".join(schema_lines)
