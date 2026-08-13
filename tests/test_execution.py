"""Phase 4: SQL execution against DuckDB.

Covers successful execution across query shapes, the structured result contract,
the row-cap strategy, and the guarantee that no write statement can reach the
database. No API key and no network are required.
"""

from __future__ import annotations

import pandas as pd
import pytest

import database
from config import MAX_RESULT_ROWS
from database import (
    DatabaseConnectionError,
    QueryResult,
    SqlExecutionError,
    SqlValidationError,
    run_query,
)

TOTAL_ROWS = 300
TOTAL_COLUMNS = 13


class TestSuccessfulExecution:
    """Each supported query shape must execute and return the right rows."""

    def test_plain_select(self, initialized_database):
        result = run_query("SELECT opportunity_id, amount FROM opportunities;")
        assert result.row_count == TOTAL_ROWS
        assert result.columns == ["opportunity_id", "amount"]

    def test_select_star(self, initialized_database):
        result = run_query("SELECT * FROM opportunities;")
        assert result.row_count == TOTAL_ROWS
        assert len(result.columns) == TOTAL_COLUMNS

    def test_select_with_where(self, initialized_database):
        result = run_query(
            "SELECT opportunity_id, region FROM opportunities WHERE region = 'EMEA';"
        )
        assert 0 < result.row_count < TOTAL_ROWS
        assert set(result.frame["region"]) == {"EMEA"}

    def test_aggregation(self, initialized_database):
        result = run_query(
            "SELECT COUNT(*) AS deals, SUM(amount) AS total FROM opportunities;"
        )
        assert result.row_count == 1
        assert result.frame["deals"].iloc[0] == TOTAL_ROWS
        assert result.frame["total"].iloc[0] > 0

    def test_group_by(self, initialized_database):
        result = run_query(
            "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"
        )
        assert result.row_count == 4
        assert result.frame["deals"].sum() == TOTAL_ROWS

    def test_group_by_having(self, initialized_database):
        result = run_query(
            "SELECT stage, COUNT(*) AS deals FROM opportunities "
            "GROUP BY stage HAVING COUNT(*) > 40;"
        )
        assert all(count > 40 for count in result.frame["deals"])

    def test_order_by(self, initialized_database):
        result = run_query(
            "SELECT amount FROM opportunities ORDER BY amount DESC;"
        )
        amounts = list(result.frame["amount"])
        assert amounts == sorted(amounts, reverse=True)

    def test_limit(self, initialized_database):
        result = run_query(
            "SELECT opportunity_id, amount FROM opportunities ORDER BY amount DESC LIMIT 5;"
        )
        assert result.row_count == 5
        assert result.truncated is False

    def test_cte(self, initialized_database):
        result = run_query(
            "WITH won AS (SELECT amount FROM opportunities WHERE is_won = TRUE) "
            "SELECT COUNT(*) AS c, SUM(amount) AS total FROM won;"
        )
        assert result.row_count == 1
        assert result.frame["c"].iloc[0] > 0


class TestEmptyResult:
    """An empty result is a success, not an error."""

    def test_empty_result_is_not_an_error(self, initialized_database):
        result = run_query(
            "SELECT opportunity_id FROM opportunities WHERE amount < 0;"
        )
        assert result.row_count == 0
        assert result.is_empty is True
        assert result.truncated is False

    def test_empty_result_preserves_columns(self, initialized_database):
        result = run_query(
            "SELECT opportunity_id, region FROM opportunities WHERE region = 'NOWHERE';"
        )
        assert result.columns == ["opportunity_id", "region"]
        assert isinstance(result.frame, pd.DataFrame)


class TestResultStructure:
    """The QueryResult contract: columns, rows, row count."""

    def test_returns_query_result(self, initialized_database):
        assert isinstance(run_query("SELECT 1 AS x;"), QueryResult)

    def test_frame_is_a_dataframe(self, initialized_database):
        assert isinstance(run_query("SELECT 1 AS x;").frame, pd.DataFrame)

    def test_row_count_matches_frame_length(self, initialized_database):
        result = run_query("SELECT opportunity_id FROM opportunities;")
        assert result.row_count == len(result.frame)

    def test_columns_match_frame_columns(self, initialized_database):
        result = run_query("SELECT region, owner FROM opportunities;")
        assert result.columns == list(result.frame.columns) == ["region", "owner"]

    def test_executed_sql_is_recorded_verbatim(self, initialized_database):
        sql = "SELECT region FROM opportunities;"
        assert run_query(sql).sql == sql

    def test_is_empty_is_false_for_rows(self, initialized_database):
        assert run_query("SELECT 1 AS x;").is_empty is False

    def test_result_is_not_compared_elementwise(self, initialized_database):
        """A frozen dataclass holding a DataFrame must not break on ==."""
        first = run_query("SELECT 1 AS x;")
        second = run_query("SELECT 1 AS x;")
        assert first != second  # identity comparison, must not raise


class TestRowCap:
    """The cap bounds the fetch; it never rewrites the query."""

    def test_cap_limits_rows_returned(self, initialized_database):
        result = run_query("SELECT opportunity_id FROM opportunities;", max_rows=10)
        assert result.row_count == 10
        assert result.truncated is True
        assert result.max_rows == 10

    def test_not_truncated_when_result_fits(self, initialized_database):
        result = run_query("SELECT opportunity_id FROM opportunities;", max_rows=500)
        assert result.row_count == TOTAL_ROWS
        assert result.truncated is False

    def test_cap_exactly_equal_to_row_count_is_not_truncated(self, initialized_database):
        result = run_query(
            "SELECT opportunity_id FROM opportunities;", max_rows=TOTAL_ROWS
        )
        assert result.row_count == TOTAL_ROWS
        assert result.truncated is False

    def test_user_limit_is_respected_not_overridden(self, initialized_database):
        """A smaller user LIMIT must win over the larger cap."""
        result = run_query(
            "SELECT opportunity_id FROM opportunities LIMIT 3;", max_rows=1000
        )
        assert result.row_count == 3
        assert result.truncated is False

    def test_order_by_semantics_survive_the_cap(self, initialized_database):
        """Capping must return the top rows of the ordered result, not any rows."""
        full = run_query(
            "SELECT amount FROM opportunities ORDER BY amount DESC;", max_rows=None
        )
        capped = run_query(
            "SELECT amount FROM opportunities ORDER BY amount DESC;", max_rows=5
        )
        assert list(capped.frame["amount"]) == list(full.frame["amount"][:5])

    def test_none_disables_the_cap(self, initialized_database):
        result = run_query("SELECT opportunity_id FROM opportunities;", max_rows=None)
        assert result.row_count == TOTAL_ROWS
        assert result.truncated is False
        assert result.max_rows is None

    def test_default_cap_comes_from_config(self, initialized_database):
        assert run_query("SELECT 1 AS x;").max_rows == MAX_RESULT_ROWS

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_rejects_nonsensical_cap(self, initialized_database, bad):
        with pytest.raises(ValueError, match="max_rows must be at least 1"):
            run_query("SELECT 1 AS x;", max_rows=bad)


class TestExecuteQueryWrapper:
    """execute_query keeps its original DataFrame contract."""

    def test_returns_dataframe(self, initialized_database):
        frame = database.execute_query("SELECT region FROM opportunities;")
        assert isinstance(frame, pd.DataFrame)

    def test_defaults_to_the_complete_result(self, initialized_database):
        frame = database.execute_query("SELECT opportunity_id FROM opportunities;")
        assert len(frame) == TOTAL_ROWS

    def test_accepts_an_explicit_cap(self, initialized_database):
        frame = database.execute_query(
            "SELECT opportunity_id FROM opportunities;", max_rows=7
        )
        assert len(frame) == 7


class TestWriteStatementsAreRejected:
    """No write or file-access statement may reach DuckDB."""

    @pytest.mark.parametrize(
        "sql",
        [
            "CREATE TABLE evil AS SELECT 1;",
            "CREATE OR REPLACE TABLE opportunities AS SELECT 1;",
            "INSERT INTO opportunities VALUES (1);",
            "UPDATE opportunities SET amount = 0;",
            "DELETE FROM opportunities;",
            "DROP TABLE opportunities;",
            "ALTER TABLE opportunities ADD COLUMN x INT;",
            "TRUNCATE opportunities;",
            "COPY (SELECT * FROM opportunities) TO 'exfil.csv';",
            "ATTACH 'evil.db' AS evil;",
            "DETACH evil;",
            "INSTALL httpfs;",
            "PRAGMA database_list;",
        ],
    )
    def test_write_statement_is_refused(self, initialized_database, sql):
        with pytest.raises(SqlValidationError, match="Refused to execute unsafe SQL"):
            run_query(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE opportunities;",
            "SELECT 1; DELETE FROM opportunities;",
            "SELECT 1 --x\n; DROP TABLE opportunities;",
        ],
    )
    def test_stacked_statement_is_refused(self, initialized_database, sql):
        with pytest.raises(SqlValidationError):
            run_query(sql)

    def test_database_is_unchanged_after_refusals(self, initialized_database):
        """The strongest check: refused writes leave no trace."""
        for sql in (
            "CREATE TABLE evil AS SELECT 1;",
            "DROP TABLE opportunities;",
            "DELETE FROM opportunities;",
            "SELECT 1; CREATE TABLE evil2 AS SELECT 1;",
        ):
            with pytest.raises(SqlValidationError):
                run_query(sql)

        # The guard now blocks metadata sources, so the catalog is read on a
        # direct connection rather than through the guarded query path.
        connection = database._get_connection()
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert tables == {"opportunities"}

        rows = run_query("SELECT COUNT(*) AS c FROM opportunities;")
        assert rows.frame["c"].iloc[0] == TOTAL_ROWS

    def test_filesystem_access_is_blocked(self, initialized_database):
        """read_csv_auto is not a banned keyword; the latch must stop it."""
        with pytest.raises(SqlExecutionError):
            run_query("SELECT * FROM read_csv_auto('C:/Windows/win.ini');")


class TestExecutionFailures:
    """Errors must surface as typed exceptions, never as raw DuckDB errors."""

    def test_unknown_column(self, initialized_database):
        with pytest.raises(SqlExecutionError, match="Failed to execute SQL"):
            run_query("SELECT nonexistent_column FROM opportunities;")

    def test_unknown_table(self, initialized_database):
        with pytest.raises(SqlExecutionError, match="Failed to execute SQL"):
            run_query("SELECT 1 FROM no_such_table;")

    def test_syntax_error(self, initialized_database):
        with pytest.raises(SqlExecutionError, match="Failed to execute SQL"):
            run_query("SELECT FROM WHERE;")

    def test_type_error_surfaces_as_execution_error(self, initialized_database):
        with pytest.raises(SqlExecutionError):
            run_query("SELECT SUM(account_name) FROM opportunities;")

    @pytest.mark.parametrize("value", [None, "", 0, [], {}])
    def test_non_sql_input_is_refused(self, initialized_database, value):
        with pytest.raises(SqlValidationError):
            run_query(value)

    def test_requires_an_initialized_database(self, clean_database_state):
        with pytest.raises(DatabaseConnectionError, match="not initialized"):
            run_query("SELECT 1;")

    def test_validation_error_is_an_execution_error(self):
        """Existing handlers that catch SqlExecutionError keep working."""
        assert issubclass(SqlValidationError, SqlExecutionError)


class TestLogging:
    """Execution events are logged at the documented levels."""

    def test_success_is_logged_with_row_count(self, initialized_database, caplog):
        with caplog.at_level("INFO", logger="database"):
            run_query("SELECT region FROM opportunities GROUP BY region;")
        messages = [record.getMessage() for record in caplog.records]
        assert any("SQL execution started" in message for message in messages)
        assert any("SQL execution succeeded, 4 rows returned" in message for message in messages)

    def test_truncation_is_logged_as_a_warning(self, initialized_database, caplog):
        with caplog.at_level("WARNING", logger="database"):
            run_query("SELECT opportunity_id FROM opportunities;", max_rows=5)
        assert any("truncated" in record.getMessage() for record in caplog.records)

    def test_failure_is_logged_as_an_error(self, initialized_database, caplog):
        with caplog.at_level("ERROR", logger="database"):
            with pytest.raises(SqlExecutionError):
                run_query("SELECT nonexistent_column FROM opportunities;")
        assert any(
            "SQL execution failed" in record.getMessage() for record in caplog.records
        )

    def test_refusal_is_logged_as_an_error(self, initialized_database, caplog):
        with caplog.at_level("ERROR", logger="database"):
            with pytest.raises(SqlValidationError):
                run_query("DROP TABLE opportunities;")
        assert any(
            "Refused to execute unsafe SQL" in record.getMessage()
            for record in caplog.records
        )
