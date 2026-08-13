"""Database layer: lifecycle, guards, schema extraction and execution hardening."""

from __future__ import annotations

import pytest

import database
from database import (
    DatabaseConnectionError,
    SqlExecutionError,
    SqlValidationError,
)

EXPECTED_ROWS = 300
EXPECTED_COLUMNS = 13


class TestConnectionGuards:
    """Every public entry point must refuse to work without initialization."""

    def test_get_schema_requires_initialization(self, clean_database_state):
        with pytest.raises(DatabaseConnectionError, match="not initialized"):
            database.get_schema()

    def test_execute_query_requires_initialization(self, clean_database_state):
        with pytest.raises(DatabaseConnectionError, match="not initialized"):
            database.execute_query("SELECT 1;")

    def test_close_is_idempotent(self, clean_database_state):
        database.close_connection()
        database.close_connection()  # must not raise


class TestInitialization:
    def test_loads_expected_dataset(self, initialized_database):
        frame = database.execute_query("SELECT * FROM opportunities;")
        assert len(frame) == EXPECTED_ROWS
        assert len(frame.columns) == EXPECTED_COLUMNS

    def test_repeated_initialization_is_safe(self, initialized_database):
        database.initialize_database()
        frame = database.execute_query("SELECT COUNT(*) AS c FROM opportunities;")
        assert frame["c"].iloc[0] == EXPECTED_ROWS


class TestSchema:
    def test_schema_is_markdown_with_all_columns(self, initialized_database):
        schema = database.get_schema()
        assert schema.startswith("| column | type |")
        # Header row + separator + one row per column.
        assert len(schema.splitlines()) == EXPECTED_COLUMNS + 2

    def test_schema_contains_known_columns(self, initialized_database):
        schema = database.get_schema()
        for column in ("opportunity_id", "account_name", "amount", "close_date"):
            assert f"| {column} |" in schema


class TestExecuteQueryIsReadOnly:
    """execute_query must refuse anything that is not a read-only statement.

    This is the last checkpoint before DuckDB. It validates independently of the
    caller, so a bug elsewhere cannot turn into a write against the database.
    """

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE sneaky AS SELECT 1;",
            "CREATE OR REPLACE TABLE opportunities AS SELECT 1;",
            "DROP TABLE opportunities;",
            "INSERT INTO opportunities VALUES (1);",
            "UPDATE opportunities SET amount = 0;",
            "DELETE FROM opportunities;",
            "ALTER TABLE opportunities ADD COLUMN x INT;",
            "TRUNCATE opportunities;",
            "ATTACH 'evil.db' AS evil;",
            "COPY (SELECT * FROM opportunities) TO 'exfil.csv';",
            "SELECT 1; DROP TABLE opportunities;",
        ],
    )
    def test_write_statements_are_refused(self, initialized_database, sql):
        with pytest.raises(SqlValidationError, match="Refused to execute unsafe SQL"):
            database.execute_query(sql)

    def test_refusal_leaves_the_database_untouched(self, initialized_database):
        """A refused CREATE must not appear in the catalog."""
        with pytest.raises(SqlValidationError):
            database.execute_query("CREATE TABLE sneaky AS SELECT 1;")

        # The guard now blocks metadata sources, so the catalog is inspected on a
        # direct connection instead of through the guarded query path.
        connection = database._get_connection()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert tables == {"opportunities"}

    def test_validation_error_is_an_execution_error(self, initialized_database):
        """SqlValidationError subclasses SqlExecutionError for existing handlers."""
        assert issubclass(SqlValidationError, SqlExecutionError)
        with pytest.raises(SqlExecutionError):
            database.execute_query("DROP TABLE opportunities;")

    @pytest.mark.parametrize("value", [None, "", 0, []])
    def test_rejects_non_sql_input(self, initialized_database, value):
        with pytest.raises(SqlValidationError):
            database.execute_query(value)

    def test_valid_select_still_runs(self, initialized_database):
        frame = database.execute_query(
            "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"
        )
        assert len(frame) == 4
        assert set(frame.columns) == {"region", "deals"}

    def test_cte_still_runs(self, initialized_database):
        frame = database.execute_query(
            "WITH won AS (SELECT amount FROM opportunities WHERE is_won = TRUE) "
            "SELECT SUM(amount) AS total FROM won;"
        )
        assert frame["total"].iloc[0] > 0


class TestExecutionErrors:
    def test_unknown_column_raises_execution_error(self, initialized_database):
        with pytest.raises(SqlExecutionError, match="Failed to execute SQL"):
            database.execute_query("SELECT nonexistent_column FROM opportunities;")


class TestFilesystemLatch:
    """External file access is disabled once the dataset is loaded."""

    def test_cannot_read_arbitrary_files(self, initialized_database):
        # read_csv_auto is not a forbidden keyword, so this reaches DuckDB and
        # must be stopped by the enable_external_access latch.
        with pytest.raises(SqlExecutionError):
            database.execute_query("SELECT * FROM read_csv_auto('C:/Windows/win.ini');")
