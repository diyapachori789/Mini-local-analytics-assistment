"""Multi-entity comparison handling.

Covers the reported defect: "compare OPP-1003 to OPP-1014" returned only one
row, the answer still described both, and a pie chart showed the single row as
100%. Everything here is offline - SQL generation is mocked and no Groq call is
made.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

import analytics_service
import llm
from analytics_service import incomplete_comparison_reason, process_question
from chart import ChartType
from config import CSV_PATH
from database import QueryResult

BOTH_IDS = ("OPP-1003", "OPP-1014")


def make_result(frame: pd.DataFrame) -> QueryResult:
    return QueryResult(
        frame=frame,
        sql="SELECT 1;",
        row_count=len(frame),
        truncated=False,
        max_rows=1000,
    )


def two_entity_result() -> QueryResult:
    return make_result(
        pd.DataFrame(
            {
                "opportunity_id": ["OPP-1003", "OPP-1014"],
                "account_name": ["Summit Industries", "Acme Corp"],
                "amount": [248177, 75268],
            }
        )
    )


def one_entity_result() -> QueryResult:
    return make_result(
        pd.DataFrame(
            {
                "opportunity_id": ["OPP-1003"],
                "account_name": ["Summit Industries"],
                "amount": [248177],
            }
        )
    )


class TestSourceDataContainsBothIds:
    """Separates data existence from generation correctness."""

    @pytest.fixture
    def dataset(self):
        connection = duckdb.connect()
        connection.execute(
            "CREATE TABLE opportunities AS SELECT * FROM read_csv_auto(?)", [str(CSV_PATH)]
        )
        yield connection
        connection.close()

    def test_both_ids_exist_in_the_fixture_data(self, dataset):
        rows = dataset.execute(
            "SELECT opportunity_id FROM opportunities WHERE opportunity_id IN (?, ?)",
            list(BOTH_IDS),
        ).fetchall()
        assert {row[0] for row in rows} == set(BOTH_IDS)

    def test_hand_written_comparison_sql_returns_two_rows(self, dataset):
        frame = dataset.execute(
            "SELECT opportunity_id, account_name, amount FROM opportunities "
            "WHERE opportunity_id IN ('OPP-1003', 'OPP-1014') ORDER BY opportunity_id"
        ).df()
        assert len(frame) == 2
        assert list(frame["opportunity_id"]) == ["OPP-1003", "OPP-1014"]


class TestIdentifierExtraction:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("compare OPP-1003 to OPP-1014", ["OPP-1003", "OPP-1014"]),
            ("compare OPP-1001 vs OPP-1002 vs OPP-1003", ["OPP-1001", "OPP-1002", "OPP-1003"]),
            ("show OPP-1003", ["OPP-1003"]),
            ("opp-1003 and OPP-1003", ["OPP-1003"]),
            ("pipeline by region", []),
            ("compare Acme Labs and Vertex Labs", []),
        ],
    )
    def test_extracts_named_identifiers(self, question, expected):
        assert llm.extract_opportunity_ids(question) == expected

    @pytest.mark.parametrize("value", [None, 42, [], {}])
    def test_handles_bad_input(self, value):
        assert llm.extract_opportunity_ids(value) == []


class TestComparisonDetection:
    @pytest.mark.parametrize(
        "question",
        [
            "compare OPP-1003 to OPP-1014",
            "Compare OPP-1003 with OPP-1014",
            "OPP-1003 versus OPP-1014",
            "OPP-1003 vs OPP-1014",
            "difference between OPP-1003 and OPP-1014",
            "compare NA and EMEA",
        ],
    )
    def test_detects_comparison_wording(self, question):
        assert llm.is_comparison_question(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "show OPP-1003",
            "pipeline by region",
            "What is the overall win rate?",
            "list opportunities for OPP-1003 and OPP-1014",
        ],
    )
    def test_naming_values_alone_is_not_a_comparison(self, question):
        """Several values without comparison wording must not trigger the guard."""
        assert llm.is_comparison_question(question) is False


class TestMissingIdentifierDetection:
    def test_detects_a_dropped_identifier_in_sql(self):
        sql = "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id = 'OPP-1003';"
        assert llm.missing_identifiers("compare OPP-1003 to OPP-1014", sql) == ["OPP-1014"]

    def test_accepts_sql_containing_every_identifier(self):
        sql = (
            "SELECT opportunity_id, amount FROM opportunities "
            "WHERE opportunity_id IN ('OPP-1003', 'OPP-1014');"
        )
        assert llm.missing_identifiers("compare OPP-1003 to OPP-1014", sql) == []

    def test_case_differences_still_count_as_present(self):
        sql = "SELECT * FROM opportunities WHERE opportunity_id IN ('opp-1003', 'opp-1014');"
        assert llm.missing_identifiers("compare OPP-1003 to OPP-1014", sql) == []

    def test_three_identifiers_all_required(self):
        sql = "SELECT * FROM opportunities WHERE opportunity_id IN ('OPP-1001', 'OPP-1002');"
        question = "compare OPP-1001 vs OPP-1002 vs OPP-1003"
        assert llm.missing_identifiers(question, sql) == ["OPP-1003"]


class TestGenerationSafeguard:
    """A dropped identifier triggers one bounded retry, then a refusal."""

    def _client(self, monkeypatch, responses):
        calls = []

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)
                self.finish_reason = "stop"

        class FakeResponse:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse(responses[min(len(calls) - 1, len(responses) - 1)])

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())
        monkeypatch.setattr(llm, "get_schema", lambda: "| column | type |\n| --- | --- |")
        return calls

    def test_complete_sql_is_accepted_without_a_retry(self, monkeypatch):
        good = "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id IN ('OPP-1003', 'OPP-1014');"
        calls = self._client(monkeypatch, [good])

        sql = llm.generate_sql("compare OPP-1003 to OPP-1014")
        assert "OPP-1003" in sql and "OPP-1014" in sql
        assert len(calls) == 1, "a complete statement must not be regenerated"

    def test_dropped_identifier_triggers_one_retry_that_can_succeed(self, monkeypatch):
        bad = "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id = 'OPP-1003';"
        good = "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id IN ('OPP-1003', 'OPP-1014');"
        calls = self._client(monkeypatch, [bad, good])

        sql = llm.generate_sql("compare OPP-1003 to OPP-1014")
        assert "OPP-1014" in sql
        assert len(calls) == 2, "exactly one corrective retry"
        assert "OPP-1014" in calls[1]["messages"][1]["content"]

    def test_persistent_omission_refuses_rather_than_running(self, monkeypatch):
        bad = "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id = 'OPP-1003';"
        calls = self._client(monkeypatch, [bad, bad])

        with pytest.raises(ValueError, match="OPP-1014"):
            llm.generate_sql("compare OPP-1003 to OPP-1014")
        assert len(calls) == 2, "the retry is bounded at one"

    def test_single_id_lookup_is_unaffected(self, monkeypatch):
        good = "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id = 'OPP-1003';"
        calls = self._client(monkeypatch, [good])

        assert "OPP-1003" in llm.generate_sql("show OPP-1003")
        assert len(calls) == 1

    def test_questions_without_identifiers_are_unaffected(self, monkeypatch):
        good = "SELECT region, SUM(amount) AS total FROM opportunities GROUP BY region;"
        calls = self._client(monkeypatch, [good])

        assert llm.generate_sql("pipeline by region") == good
        assert len(calls) == 1

    def test_refusal_short_circuits_before_the_safeguard(self, monkeypatch):
        calls = self._client(monkeypatch, ["INVALID_QUESTION"])
        assert llm.generate_sql("compare OPP-1003 to OPP-1014") == llm.INVALID_QUESTION
        assert len(calls) == 1


class TestSqlPromptRules:
    def test_prompt_requires_every_named_value(self, monkeypatch):
        captured = {}

        class FakeMessage:
            content = "SELECT opportunity_id FROM opportunities WHERE opportunity_id IN ('OPP-1003', 'OPP-1014');"

        class FakeChoice:
            message = FakeMessage()
            finish_reason = "stop"

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())
        monkeypatch.setattr(llm, "get_schema", lambda: "| column | type |")
        llm.generate_sql("compare OPP-1003 to OPP-1014")

        system = captured["messages"][0]["content"]
        assert "FILTERS - keep every value the user names" in system
        assert "Never silently drop one" in system
        assert "IN (...)" in system
        assert "Compare OPP-1003 to OPP-1014." in system
        assert "Compare Acme Labs and Vertex Labs." in system
        assert "account_name IN ('Acme Labs', 'Vertex Labs')" in system


class TestAnswerGrounding:
    """The answer must never assert facts about an entity that never came back."""

    def test_absent_identifier_is_detected(self):
        assert llm.absent_identifiers("compare OPP-1003 to OPP-1014", one_entity_result()) == [
            "OPP-1014"
        ]

    def test_no_absent_identifiers_when_both_returned(self):
        assert llm.absent_identifiers("compare OPP-1003 to OPP-1014", two_entity_result()) == []

    def test_empty_result_reports_every_named_identifier(self):
        empty = make_result(pd.DataFrame(columns=["opportunity_id", "amount"]))
        assert llm.absent_identifiers("compare OPP-1003 to OPP-1014", empty) == list(BOTH_IDS)

    def test_prompt_marks_the_missing_entity(self, monkeypatch):
        captured = {}

        class FakeMessage:
            content = "Only OPP-1003 is in the returned data."

        class FakeChoice:
            message = FakeMessage()
            finish_reason = "stop"

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())
        llm.generate_answer("compare OPP-1003 to OPP-1014", one_entity_result())

        user_content = captured["messages"][1]["content"]
        assert "NOT IN RESULT: OPP-1014" in user_content
        assert "Do not state any fact or comparison involving them" in user_content

        system = captured["messages"][0]["content"]
        assert "Only discuss things that actually appear in the result rows" in system
        assert "Only compare things when every one of them is present" in system

    def test_no_marker_when_the_result_is_complete(self, monkeypatch):
        captured = {}

        class FakeMessage:
            content = "OPP-1003 is 248,177 and OPP-1014 is 75,268."

        class FakeChoice:
            message = FakeMessage()
            finish_reason = "stop"

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())
        llm.generate_answer("compare OPP-1003 to OPP-1014", two_entity_result())
        assert "NOT IN RESULT" not in captured["messages"][1]["content"]


class TestComparisonChartSafety:
    def test_incomplete_comparison_is_blocked(self):
        reason = incomplete_comparison_reason("compare OPP-1003 to OPP-1014", one_entity_result())
        assert reason is not None
        assert "1 of the 2" in reason

    def test_complete_comparison_is_allowed(self):
        assert incomplete_comparison_reason(
            "compare OPP-1003 to OPP-1014", two_entity_result()
        ) is None

    def test_single_entity_question_is_allowed(self):
        assert incomplete_comparison_reason("show OPP-1003", one_entity_result()) is None

    def test_grouped_question_is_never_blocked(self):
        grouped = make_result(pd.DataFrame({"region": ["NA"], "total": [100]}))
        assert incomplete_comparison_reason("pipeline by region", grouped) is None

    def test_comparison_without_identifiers_is_not_blocked(self):
        grouped = make_result(pd.DataFrame({"region": ["NA"], "total": [100]}))
        assert incomplete_comparison_reason("compare NA and EMEA", grouped) is None


class TestServiceIntegration:
    """One execution, one QueryResult, correct chart decision."""

    def _run(self, question, result, chart_calls):
        executions = []

        def generate(_question):
            return "SELECT opportunity_id, amount FROM opportunities WHERE opportunity_id IN ('OPP-1003', 'OPP-1014');"

        def run_query(sql):
            executions.append(sql)
            return result

        def answer(_question, supplied):
            assert supplied is result
            return "grounded answer"

        def chart(_question, supplied):
            assert supplied is result
            chart_calls.append(supplied)
            return (__import__("pathlib").Path("c.png"), ChartType.BAR, None)

        response = process_question(
            question,
            sql_generator=generate,
            query_runner=run_query,
            answer_generator=answer,
            chart_creator=chart,
            query_logger=lambda *_a: None,
        )
        return response, executions

    def test_incomplete_comparison_keeps_answer_and_drops_chart(self):
        chart_calls = []
        response, executions = self._run(
            "compare OPP-1003 to OPP-1014 and chart it", one_entity_result(), chart_calls
        )

        assert len(executions) == 1, "still exactly one analytical execution"
        assert chart_calls == [], "no chart generated for an incomplete comparison"
        assert response.chart_path is None
        assert response.chart_requested is True
        assert "only 1 of the 2" in response.chart_note
        # The answer and the result table survive intact.
        assert response.answer == "grounded answer"
        assert response.result.row_count == 1

    def test_complete_comparison_still_charts(self):
        chart_calls = []
        response, executions = self._run(
            "compare OPP-1003 to OPP-1014 and chart it", two_entity_result(), chart_calls
        )

        assert len(executions) == 1
        assert len(chart_calls) == 1
        assert response.chart_path is not None
        assert response.chart_note is None

    def test_unrelated_chart_question_is_unaffected(self):
        chart_calls = []
        grouped = make_result(pd.DataFrame({"region": ["NA", "EMEA"], "total": [100, 90]}))
        response, executions = self._run("pipeline by region and chart it", grouped, chart_calls)

        assert len(executions) == 1
        assert len(chart_calls) == 1
        assert response.chart_path is not None
