"""Phase 4: CLI result rendering and per-question error handling."""

from __future__ import annotations

import pandas as pd
import pytest

import analytics_service
import app
import chart
import history_repository
import logging_config
from config import DISPLAY_ROWS
from database import DatabaseError, QueryResult, SqlExecutionError, SqlValidationError
from intent import Intent, QueryPlan, parse_plan
from llm import INVALID_QUESTION


def make_result(frame: pd.DataFrame, *, truncated=False, max_rows=1000) -> QueryResult:
    return QueryResult(
        frame=frame,
        sql="SELECT 1;",
        row_count=len(frame),
        truncated=truncated,
        max_rows=max_rows,
    )


class TestRenderResult:
    def test_empty_result_reports_columns(self):
        result = make_result(pd.DataFrame(columns=["region", "total"]))
        rendered = app.render_result(result)
        assert "No rows matched" in rendered
        assert "region, total" in rendered

    def test_rows_and_row_count_are_shown(self):
        frame = pd.DataFrame({"region": ["NA", "EMEA"], "deals": [92, 69]})
        rendered = app.render_result(make_result(frame))
        assert "EMEA" in rendered
        assert "Rows: 2" in rendered
        assert "capped" not in rendered

    def test_truncation_is_reported(self):
        frame = pd.DataFrame({"x": range(10)})
        rendered = app.render_result(make_result(frame, truncated=True, max_rows=10))
        assert "capped at 10" in rendered
        assert "matched more" in rendered

    def test_display_is_capped_but_row_count_is_honest(self):
        frame = pd.DataFrame({"x": range(DISPLAY_ROWS + 15)})
        rendered = app.render_result(make_result(frame))
        assert f"Rows: {DISPLAY_ROWS + 15}" in rendered
        assert f"Showing the first {DISPLAY_ROWS}" in rendered
        # The row after the display cap must not be printed.
        assert str(DISPLAY_ROWS + 14) not in rendered.split("Rows:")[0]

    def test_short_result_has_no_showing_notice(self):
        rendered = app.render_result(make_result(pd.DataFrame({"x": [1, 2]})))
        assert "Showing the first" not in rendered


class TestAnswerQuestion:
    """One question must never crash the session."""

    def test_successful_question_prints_only_the_answer(
        self, initialized_database, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: parse_plan(
                "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"
            ),
        )
        monkeypatch.setattr(
            app, "generate_answer", lambda q, r: "There are four regions."
        )
        assert app.answer_question("deals by region") is True

        output = capsys.readouterr().out
        assert "Answer:" in output
        assert "There are four regions." in output
        # SQL and row count belong in the log, not on the user's screen.
        assert "SELECT" not in output
        assert "GROUP BY" not in output
        assert "Rows:" not in output

    def test_sql_and_row_count_are_logged_not_printed(
        self, initialized_database, monkeypatch, capsys, caplog
    ):
        """Requirement: SQL stays available for debugging via the log."""
        sql = "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"
        monkeypatch.setattr(app, "generate_plan", lambda q: parse_plan(sql))
        monkeypatch.setattr(app, "generate_answer", lambda q, r: "Four regions.")

        with caplog.at_level("INFO", logger="app"):
            app.answer_question("deals by region")

        logged = " ".join(record.getMessage() for record in caplog.records)
        assert sql in logged
        assert "rows=4" in logged
        assert sql not in capsys.readouterr().out

    def test_answer_receives_the_question_and_result(
        self, initialized_database, monkeypatch, capsys
    ):
        seen = {}

        def capture(question, result):
            seen["question"] = question
            seen["row_count"] = result.row_count
            return "ok"

        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: parse_plan("SELECT region FROM opportunities GROUP BY region;"),
        )
        monkeypatch.setattr(app, "generate_answer", capture)
        app.answer_question("which regions?")

        assert seen["question"] == "which regions?"
        assert seen["row_count"] == 4

    def test_refusal_is_never_executed(self, initialized_database, monkeypatch, capsys):
        monkeypatch.setattr(app, "generate_plan", lambda q: parse_plan(INVALID_QUESTION))

        def fail(*args, **kwargs):
            raise AssertionError("run_query must not be called for a refusal")

        monkeypatch.setattr(app, "run_query", fail)

        assert app.answer_question("capital of France?") is True
        # A friendly, category-specific reply instead of one fixed sentence.
        output = capsys.readouterr().out
        assert "analytics" in output.lower() or "sales" in output.lower()
        assert "Answer:" in output

    def test_generation_failure_is_handled(self, initialized_database, monkeypatch, capsys):
        def boom(_):
            raise ValueError("Question cannot be empty.")

        monkeypatch.setattr(app, "generate_plan", boom)
        assert app.answer_question("   ") is False
        assert "Unable to generate SQL" in capsys.readouterr().out

    def test_api_failure_is_handled(self, initialized_database, monkeypatch, capsys):
        def boom(_):
            raise RuntimeError("Groq API request failed: 401")

        monkeypatch.setattr(app, "generate_plan", boom)
        assert app.answer_question("anything") is False
        assert "Unable to reach the language model" in capsys.readouterr().out

    def test_refused_execution_becomes_a_friendly_refusal(
        self, initialized_database, monkeypatch, capsys
    ):
        """Unsafe generated SQL is now a refusal, not a raw error message."""
        import refusal

        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: QueryPlan(intent=Intent.DATA_QUERY, sql="DROP TABLE opportunities;"),
        )
        assert app.answer_question("DROP TABLE opportunities") is True

        output = capsys.readouterr().out
        assert any(
            template in output
            for template in refusal._TEMPLATES[refusal.RefusalCategory.UNSAFE_SQL]
        )
        # The internal guard wording never reaches the terminal.
        assert "disallowed statement" not in output

    def test_execution_failure_is_handled(self, initialized_database, monkeypatch, capsys):
        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: parse_plan("SELECT nonexistent_column FROM opportunities;"),
        )
        assert app.answer_question("bad column") is False
        assert "Query failed" in capsys.readouterr().out

    def test_unexpected_error_is_contained(self, initialized_database, monkeypatch, capsys):
        monkeypatch.setattr(app, "generate_plan", lambda q: parse_plan("SELECT 1 AS x;"))

        def explode(*args, **kwargs):
            raise MemoryError("simulated")

        monkeypatch.setattr(app, "run_query", explode)
        assert app.answer_question("anything") is False
        assert "unexpected error" in capsys.readouterr().out.lower()


class TestAnswerFallback:
    """A failed second LLM call must never discard the database result."""

    @pytest.fixture(autouse=True)
    def _stub_sql(self, monkeypatch):
        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: parse_plan(
                "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"
            ),
        )

    def _fail_answer(self, monkeypatch, error):
        def boom(question, result):
            raise error

        monkeypatch.setattr(app, "generate_answer", boom)

    def test_api_failure_still_shows_the_result(
        self, initialized_database, monkeypatch, capsys
    ):
        failure_message = "Answer generation failed: boom"
        self._fail_answer(monkeypatch, RuntimeError(failure_message))
        app.answer_question("deals by region")

        output = capsys.readouterr().out
        assert "The natural-language answer could not be generated." in output
        assert "Showing the query result instead:" in output
        assert failure_message not in output
        assert "Rows: 4" in output
        # The actual data must survive the failure.
        assert "EMEA" in output

    def test_fallback_does_not_print_the_sql(
        self, initialized_database, monkeypatch, capsys
    ):
        """Even on the fallback path, the statement stays in the log only."""
        self._fail_answer(monkeypatch, RuntimeError("boom"))
        app.answer_question("deals by region")

        output = capsys.readouterr().out
        assert "GROUP BY" not in output
        assert "EMEA" in output

    def test_provider_diagnostic_does_not_reach_stdout_but_result_survives(
        self, initialized_database, monkeypatch, capsys
    ):
        failure_message = "Groq rate limit reached"
        self._fail_answer(monkeypatch, RuntimeError(failure_message))
        app.answer_question("deals by region")

        output = capsys.readouterr().out
        assert failure_message not in output
        assert "The natural-language answer could not be generated." in output
        assert "EMEA" in output
        assert "Rows: 4" in output

    def test_unexpected_answer_error_still_shows_the_result(
        self, initialized_database, monkeypatch, capsys
    ):
        self._fail_answer(monkeypatch, MemoryError("simulated"))
        app.answer_question("deals by region")

        output = capsys.readouterr().out
        assert "The natural-language answer could not be generated." in output
        assert "simulated" not in output
        assert "EMEA" in output

    def test_fallback_keeps_fake_secret_off_stdout_and_redacts_its_log(
        self, initialized_database, monkeypatch, capsys, caplog
    ):
        fake_key = "gsk_phase7_cli_fallback_secret"
        self._fail_answer(monkeypatch, RuntimeError(f"Provider diagnostic: {fake_key}"))
        monkeypatch.setattr(logging_config, "GROQ_API_KEY", fake_key)
        redactor = logging_config.SecretRedactingFilter()
        caplog.handler.addFilter(redactor)

        try:
            with caplog.at_level("ERROR"):
                assert app.answer_question("deals by region") is False
        finally:
            caplog.handler.removeFilter(redactor)

        output = capsys.readouterr().out
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert fake_key not in output
        assert fake_key not in logged
        assert "***REDACTED***" in logged
        assert "Answer generation failed, preserving the query result" in logged
        assert "EMEA" in output
        assert "Rows: 4" in output

    def test_sql_answer_rejection_falls_back(
        self, initialized_database, monkeypatch, capsys
    ):
        self._fail_answer(
            monkeypatch, RuntimeError("Answer generation produced SQL instead of an answer.")
        )
        app.answer_question("deals by region")

        output = capsys.readouterr().out
        assert "Rows: 4" in output
        assert "EMEA" in output

    def test_fallback_is_reported_as_a_failure(
        self, initialized_database, monkeypatch, capsys
    ):
        """The user sees their data, but the call is still not a success."""
        self._fail_answer(monkeypatch, RuntimeError("boom"))
        assert app.answer_question("deals by region") is False


class TestChartIntegration:
    """Useful/requested charts share the answer's authoritative QueryResult."""

    @pytest.fixture(autouse=True)
    def _stub_pipeline(self, monkeypatch):
        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: parse_plan(
                "SELECT owner, COUNT(*) AS closed_won FROM opportunities "
                "WHERE is_won = TRUE GROUP BY owner;"
            ),
        )
        monkeypatch.setattr(app, "generate_answer", lambda q, r: "C. Mehta closed the most.")

    def test_scalar_question_generates_no_chart(
        self, initialized_database, monkeypatch, capsys
    ):
        def forbidden(*args, **kwargs):
            raise AssertionError("no chart should be created for a scalar result")

        monkeypatch.setattr(
            app,
            "generate_plan",
            lambda q: parse_plan("SELECT COUNT(*) AS opportunity_count FROM opportunities;"),
        )
        monkeypatch.setattr(app, "create_chart", forbidden)
        assert app.answer_question("How many opportunities are there?") is True

        output = capsys.readouterr().out
        assert "Answer:" in output
        assert "Chart" not in output

    def test_chart_request_generates_a_chart(
        self, initialized_database, monkeypatch, capsys, tmp_path
    ):
        target = tmp_path / "chart.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\n")
        monkeypatch.setattr(
            app, "create_chart", lambda q, r: (target, chart.ChartType.BAR, None)
        )

        assert app.answer_question(
            "How many opportunities did each owner close won? — and chart it."
        ) is True

        output = capsys.readouterr().out
        assert "C. Mehta closed the most." in output
        assert "Chart generated successfully:" in output

    def test_chart_uses_the_same_result_as_the_answer(
        self, initialized_database, monkeypatch, tmp_path
    ):
        """One query, one result, two consumers."""
        seen = {}

        def capture_answer(question, result):
            seen["answer_result"] = id(result)
            return "ok"

        def capture_chart(question, result):
            seen["chart_result"] = id(result)
            target = tmp_path / "c.png"
            target.write_bytes(b"x")
            return target, chart.ChartType.BAR, None

        monkeypatch.setattr(app, "generate_answer", capture_answer)
        monkeypatch.setattr(app, "create_chart", capture_chart)
        app.answer_question("Closed won by owner and chart it.")

        assert seen["answer_result"] == seen["chart_result"]

    def test_only_one_query_is_executed(self, initialized_database, monkeypatch, tmp_path):
        calls = []
        real_run_query = app.run_query

        def counting(sql, *args, **kwargs):
            calls.append(sql)
            return real_run_query(sql, *args, **kwargs)

        target = tmp_path / "c.png"
        target.write_bytes(b"x")
        monkeypatch.setattr(app, "run_query", counting)
        monkeypatch.setattr(
            app, "create_chart", lambda q, r: (target, chart.ChartType.BAR, None)
        )
        app.answer_question("Closed won by owner and chart it.")
        assert len(calls) == 1

    def test_chart_failure_preserves_the_answer(
        self, initialized_database, monkeypatch, capsys
    ):
        def boom(question, result):
            raise chart.ChartError("nothing to chart")

        monkeypatch.setattr(app, "create_chart", boom)
        assert app.answer_question("Closed won by owner and chart it.") is True

        output = capsys.readouterr().out
        assert "C. Mehta closed the most." in output
        assert "Chart not generated" in output

    def test_unexpected_chart_error_is_contained(
        self, initialized_database, monkeypatch, capsys
    ):
        def boom(question, result):
            raise MemoryError("simulated")

        monkeypatch.setattr(app, "create_chart", boom)
        assert app.answer_question("Closed won by owner and chart it.") is True
        assert "Chart not generated" in capsys.readouterr().out

    def test_fallback_note_is_shown(self, initialized_database, monkeypatch, capsys, tmp_path):
        target = tmp_path / "c.png"
        target.write_bytes(b"x")
        monkeypatch.setattr(
            app,
            "create_chart",
            lambda q, r: (target, chart.ChartType.LINE, "A line chart was used because ..."),
        )
        app.answer_question("Show monthly pipeline as a pie chart.")
        assert "A line chart was used because" in capsys.readouterr().out

    def test_sql_still_hidden_when_charting(
        self, initialized_database, monkeypatch, capsys, tmp_path
    ):
        target = tmp_path / "c.png"
        target.write_bytes(b"x")
        monkeypatch.setattr(
            app, "create_chart", lambda q, r: (target, chart.ChartType.BAR, None)
        )
        app.answer_question("Closed won by owner and chart it.")

        output = capsys.readouterr().out
        assert "SELECT" not in output
        assert "GROUP BY" not in output

    def test_chart_directive_is_stripped_before_the_model(
        self, initialized_database, monkeypatch, tmp_path
    ):
        """The SQL model must not be asked to draw anything."""
        seen = {}
        target = tmp_path / "c.png"
        target.write_bytes(b"x")

        def capture_sql(question):
            seen["sql_question"] = question
            return parse_plan("SELECT owner FROM opportunities GROUP BY owner;")

        monkeypatch.setattr(app, "generate_plan", capture_sql)
        monkeypatch.setattr(
            app, "create_chart", lambda q, r: (target, chart.ChartType.BAR, None)
        )
        app.answer_question("How many did each owner close won? — and chart it.")

        assert "chart" not in seen["sql_question"].lower()
        assert "owner" in seen["sql_question"].lower()


class TestLogDetails:
    """SQL and row count go to the log, never to stdout."""

    def test_logs_sql_and_row_count(self, capsys, caplog):
        with caplog.at_level("INFO", logger="app"):
            app.log_details("SELECT 1;", make_result(pd.DataFrame({"x": [1, 2]})))

        logged = " ".join(record.getMessage() for record in caplog.records)
        assert "SELECT 1;" in logged
        assert "rows=2" in logged
        assert capsys.readouterr().out == ""

    def test_logs_truncation(self, caplog):
        with caplog.at_level("INFO", logger="app"):
            app.log_details(
                "SELECT 1;",
                make_result(pd.DataFrame({"x": range(5)}), truncated=True, max_rows=5),
            )
        logged = " ".join(record.getMessage() for record in caplog.records)
        assert "capped at 5" in logged

    def test_refusal_error_precedes_generic_database_error(self):
        """SqlValidationError must be caught before DatabaseError in app flow."""
        assert issubclass(SqlValidationError, SqlExecutionError)
        assert issubclass(SqlExecutionError, DatabaseError)


class TestCliHistoryPersistence:
    """The CLI persists completed analyses through the same repository."""

    @pytest.fixture(autouse=True)
    def _stub_pipeline(self, monkeypatch):
        monkeypatch.setattr(
            app,
            "process_question",
            lambda question, **_kwargs: analytics_service.AnalysisResponse(
                original_question=question,
                effective_question=question,
                analytical_question=question,
                answer="NA leads with 92 deals.",
                result=make_result(pd.DataFrame({"region": ["NA"], "deals": [92]})),
                chart_requested=False,
                chart_path=None,
                chart_type=None,
                chart_note=None,
                answer_fallback_used=False,
                answer_error=None,
                chart_error=None,
                refused=False,
                elapsed_seconds=0.5,
            ),
        )

    def test_successful_cli_question_saves_one_record(self, capsys):
        assert app.answer_question("Deals by region") is True
        capsys.readouterr()

        records = history_repository.list_history()
        assert len(records) == 1
        assert records[0].original_question == "Deals by region"
        assert records[0].success is True

    def test_cli_history_failure_does_not_break_the_answer(self, monkeypatch, capsys):
        def boom(**_kwargs):
            raise RuntimeError("history unavailable")

        monkeypatch.setattr(history_repository, "save_history", boom)
        assert app.answer_question("Deals by region") is True

        output = capsys.readouterr().out
        assert "NA leads with 92 deals." in output
        assert "history unavailable" not in output

    def test_cli_history_stores_no_sql(self, capsys):
        app.answer_question("Deals by region")
        capsys.readouterr()
        record = history_repository.list_history()[0]
        assert not hasattr(record, "sql")
        assert "SELECT" not in repr(record).upper()


class TestExitBehaviour:
    def test_exit_commands(self):
        assert "exit" in app.EXIT_COMMANDS
        assert "quit" in app.EXIT_COMMANDS

    def test_exit_codes_are_distinct(self):
        codes = {app.EXIT_OK, app.EXIT_FAILURE, app.EXIT_CONFIG_ERROR, app.EXIT_INTERRUPTED}
        assert len(codes) == 4
