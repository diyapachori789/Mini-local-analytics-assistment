"""Offline Phase 7 coverage for the shared analytics service.

Every backend boundary is injected or faked in this module.  These tests must
never construct a Groq client, execute DuckDB SQL, or create a real chart.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import analytics_service
import app
from chart import ChartDecision, ChartError, ChartType, is_chart_request, requested_chart_type
from database import QueryResult
from intent import Intent, QueryPlan
from llm import INVALID_QUESTION, NO_DATA_ANSWER


def make_result(
    frame: pd.DataFrame | None = None,
    *,
    truncated: bool = False,
    max_rows: int | None = 1000,
) -> QueryResult:
    """Build an in-memory result whose SQL must never leave the service layer."""
    actual_frame = (
        frame
        if frame is not None
        else pd.DataFrame({"region": ["NA", "EMEA"], "deals": [92, 69]})
    )
    return QueryResult(
        frame=actual_frame,
        sql="SELECT secret_internal_sql FROM opportunities;",
        row_count=len(actual_frame),
        truncated=truncated,
        max_rows=max_rows,
    )


class TestChartQuestionAdapter:
    """Legacy API clients can still add presentation intent safely."""

    def test_plain_question_is_normalized_but_not_given_a_chart(self):
        assert (
            analytics_service.adapt_question_for_chart(
                "  Total open pipeline amount by stage.  "
            )
            == "Total open pipeline amount by stage."
        )

    @pytest.mark.parametrize(
        ("chart_type", "expected_type"),
        [
            ("auto", None),
            ("bar", ChartType.BAR),
            ("line", ChartType.LINE),
            ("pie", ChartType.PIE),
            ("scatter", ChartType.SCATTER),
        ],
    )
    def test_toggle_adds_deterministic_chart_intent(self, chart_type, expected_type):
        effective = analytics_service.adapt_question_for_chart(
            "Total open pipeline amount by stage.",
            chart_requested=True,
            chart_type=chart_type,
        )

        assert is_chart_request(effective)
        assert requested_chart_type(effective) is expected_type

    def test_explicit_typed_chart_type_is_authoritative_over_ui_preference(self):
        typed_question = "Show pipeline by region as a line chart."

        effective = analytics_service.adapt_question_for_chart(
            typed_question,
            chart_requested=True,
            chart_type="pie",
        )

        assert effective == typed_question
        assert requested_chart_type(effective) is ChartType.LINE

    def test_generic_typed_chart_intent_is_refined_by_specific_ui_preference(self):
        typed_question = "Show pipeline by stage and chart it."

        effective = analytics_service.adapt_question_for_chart(
            typed_question,
            chart_requested=True,
            chart_type="pie",
        )

        assert is_chart_request(effective)
        assert requested_chart_type(effective) is ChartType.PIE

    def test_generic_typed_chart_intent_stays_generic_when_ui_type_is_auto(self):
        typed_question = "Show pipeline by stage and chart it."

        effective = analytics_service.adapt_question_for_chart(
            typed_question,
            chart_requested=True,
            chart_type="auto",
        )

        assert effective == typed_question
        assert requested_chart_type(effective) is None

    def test_typed_chart_intent_remains_honored_when_ui_toggle_is_off(self):
        typed_question = "Show pipeline by stage and chart it."

        effective = analytics_service.adapt_question_for_chart(
            typed_question,
            chart_requested=False,
        )

        assert effective == typed_question
        assert is_chart_request(effective)
        assert requested_chart_type(effective) is None

    @pytest.mark.parametrize("bad_type", [None, "area", "BAR", 1])
    def test_adapter_rejects_invalid_chart_type(self, bad_type):
        with pytest.raises(ValueError, match="chart_type"):
            analytics_service.adapt_question_for_chart(
                "Total pipeline by region",
                chart_requested=True,
                chart_type=bad_type,
            )


class TestProcessQuestion:
    """The service owns exactly one SQL execution and one authoritative result."""

    def test_executes_once_and_passes_one_result_to_answer_and_chart(self):
        result = make_result()
        calls: list[tuple[str, object]] = []

        def generate_sql(question: str) -> str:
            calls.append(("sql", question))
            assert "chart" not in question.lower()
            return "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"

        def run_query(sql: str) -> QueryResult:
            calls.append(("query", sql))
            return result

        def generate_answer(question: str, supplied_result: QueryResult) -> str:
            calls.append(("answer", supplied_result))
            assert supplied_result is result
            return "NA has the most deals."

        def create_chart(question: str, supplied_result: QueryResult):
            calls.append(("chart", supplied_result))
            assert supplied_result is result
            assert requested_chart_type(question) is ChartType.BAR
            return Path("service-test-chart.png"), ChartType.BAR, None

        def log_query(sql: str, supplied_result: QueryResult) -> None:
            calls.append(("log", supplied_result))
            assert supplied_result is result

        response = analytics_service.process_question(
            "Show deals by region as a bar chart.",
            original_question="Show deals by region as a bar chart.",
            sql_generator=generate_sql,
            query_runner=run_query,
            answer_generator=generate_answer,
            chart_creator=create_chart,
            query_logger=log_query,
        )

        assert [name for name, _ in calls].count("sql") == 1
        assert [name for name, _ in calls].count("query") == 1
        assert [name for name, _ in calls].count("answer") == 1
        assert [name for name, _ in calls].count("chart") == 1
        assert response.result is result
        assert response.answer == "NA has the most deals."
        assert response.chart_requested is True
        assert response.chart_path == Path("service-test-chart.png")
        assert response.chart_type is ChartType.BAR
        assert response.answer_fallback_used is False

    def test_plain_question_never_creates_a_chart(self):
        result = make_result(pd.DataFrame({"win_rate": [0.42]}))
        chart_calls = 0

        def forbidden_chart(*_args, **_kwargs):
            nonlocal chart_calls
            chart_calls += 1
            raise AssertionError("A plain question must not create a chart")

        response = analytics_service.process_question(
            "What is the overall win rate?",
            sql_generator=lambda _question: "SELECT 1;",
            query_runner=lambda _sql: result,
            answer_generator=lambda _question, _result: "There are opportunities in two regions.",
            chart_creator=forbidden_chart,
            query_logger=lambda _sql, _result: None,
        )

        assert chart_calls == 0
        assert response.result is result
        assert response.chart_requested is False
        assert response.chart_path is None
        assert response.chart_decision is ChartDecision.NO_CHART

    def test_categorical_comparison_automatically_uses_same_result_for_chart(self):
        result = make_result()
        query_calls = 0
        chart_results: list[QueryResult] = []

        def run_query(_sql: str) -> QueryResult:
            nonlocal query_calls
            query_calls += 1
            return result

        def create_chart(_question: str, supplied_result: QueryResult):
            chart_results.append(supplied_result)
            return Path("automatic-bar.png"), ChartType.BAR, None

        response = analytics_service.process_question(
            "Compare opportunities across regions.",
            sql_generator=lambda _question: "SELECT 1;",
            query_runner=run_query,
            answer_generator=lambda _question, supplied_result: (
                "NA leads." if supplied_result is result else pytest.fail("wrong result")
            ),
            chart_creator=create_chart,
            query_logger=lambda _sql, _result: None,
        )

        assert query_calls == 1
        assert chart_results == [result]
        assert response.chart_decision is ChartDecision.AUTO_USEFUL
        assert response.chart_requested is True
        assert response.chart_type is ChartType.BAR

    def test_explicit_scalar_request_returns_explanation_without_chart_call(self):
        result = make_result(pd.DataFrame({"win_rate": [0.42]}))

        response = analytics_service.process_question(
            "What is the win rate? Show me a chart.",
            sql_generator=lambda _question: "SELECT 1;",
            query_runner=lambda _sql: result,
            answer_generator=lambda _question, _result: "The win rate is 42%.",
            chart_creator=lambda *_args: pytest.fail("a scalar must not be charted"),
            query_logger=lambda _sql, _result: None,
        )

        assert response.chart_decision is ChartDecision.USER_REQUESTED
        assert response.chart_requested is True
        assert response.chart_path is None
        assert response.chart_note and "single value" in response.chart_note

    def test_follow_up_chart_request_uses_context_and_one_fresh_query(self):
        result = make_result()
        planner_calls: list[tuple[str, str | None]] = []
        query_calls = 0

        def plan(question: str, *, conversation_context: str | None = None) -> QueryPlan:
            planner_calls.append((question, conversation_context))
            return QueryPlan(intent=Intent.DATA_QUERY, sql="SELECT region, COUNT(*) FROM opportunities GROUP BY region;")

        def query(_sql: str) -> QueryResult:
            nonlocal query_calls
            query_calls += 1
            return result

        response = analytics_service.process_question(
            "Show me a chart.",
            conversation_context="USER: Compare regional opportunity counts.",
            plan_generator=plan,
            query_runner=query,
            answer_generator=lambda _question, supplied_result: (
                "NA leads." if supplied_result is result else pytest.fail("wrong result")
            ),
            chart_creator=lambda _question, supplied_result: (
                Path("follow-up.png"), ChartType.BAR, None
            ) if supplied_result is result else pytest.fail("wrong result"),
            query_logger=lambda *_args: None,
        )

        assert planner_calls == [
            ("Show me a chart.", "USER: Compare regional opportunity counts.")
        ]
        assert query_calls == 1
        assert response.result is result
        assert response.chart_decision is ChartDecision.USER_REQUESTED
        assert response.chart_path == Path("follow-up.png")

    @pytest.mark.parametrize("blank", ["", "   ", "\ufeff", "\u200b\u00a0"])
    def test_blank_or_invisible_question_makes_no_generation_call(self, blank):
        def forbidden_sql_generator(_question: str) -> str:
            raise AssertionError("A blank question must not reach SQL generation")

        with pytest.raises(ValueError, match="Question cannot be empty"):
            analytics_service.process_question(blank, sql_generator=forbidden_sql_generator)

    def test_refusal_does_not_execute_or_answer(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("A refused question must not continue through the pipeline")

        response = analytics_service.process_question(
            "What is the capital of France?",
            sql_generator=lambda _question: INVALID_QUESTION,
            query_runner=forbidden,
            answer_generator=forbidden,
            chart_creator=forbidden,
            query_logger=forbidden,
        )

        assert response.refused is True
        assert response.result is None
        assert response.chart_requested is False
        assert response.chart_decision is ChartDecision.NO_CHART
        # The reply is now a category-specific friendly template rather than one
        # fixed sentence. "capital of France" is an out-of-scope question.
        import refusal

        assert response.answer in refusal._TEMPLATES[refusal.RefusalCategory.OUT_OF_SCOPE]

    def test_conceptual_question_never_creates_a_chart(self):
        response = analytics_service.process_question(
            "Visualize what pipeline means.",
            plan_generator=lambda _question: QueryPlan(
                intent=Intent.DATA_EXPLANATION, sql=None
            ),
            conceptual_answer_generator=lambda _question: "Pipeline is the value of active opportunities.",
            query_runner=lambda _sql: pytest.fail("conceptual questions execute no query"),
            chart_creator=lambda *_args: pytest.fail("conceptual questions create no chart"),
            query_logger=lambda *_args: None,
        )

        assert response.refused is False
        assert response.result is None
        assert response.chart_requested is False
        assert response.chart_decision is ChartDecision.NO_CHART

    def test_refusal_with_chart_wording_never_creates_a_chart(self):
        response = analytics_service.process_question(
            "Show the database schema as a chart.",
            plan_generator=lambda _question: pytest.fail("structure refusal is local"),
            query_runner=lambda _sql: pytest.fail("refusals execute no query"),
            chart_creator=lambda *_args: pytest.fail("refusals create no chart"),
            query_logger=lambda *_args: None,
        )

        assert response.refused is True
        assert response.result is None
        assert response.chart_requested is False
        assert response.chart_decision is ChartDecision.NO_CHART

    def test_empty_result_uses_local_no_data_answer_without_answer_generation(self):
        result = make_result(pd.DataFrame(columns=["region", "deals"]))

        def forbidden_answer(*_args, **_kwargs):
            raise AssertionError("An empty result must not trigger answer generation")

        response = analytics_service.process_question(
            "Show deals for a nonexistent region",
            sql_generator=lambda _question: "SELECT region FROM opportunities WHERE 1 = 0;",
            query_runner=lambda _sql: result,
            answer_generator=forbidden_answer,
            query_logger=lambda _sql, _result: None,
        )

        assert response.result is result
        assert response.answer == NO_DATA_ANSWER
        assert response.answer_fallback_used is False
        assert response.answer_error is None

    def test_answer_failure_retains_the_same_authoritative_result(self):
        result = make_result()
        answer_results: list[QueryResult] = []

        def failing_answer(_question: str, supplied_result: QueryResult) -> str:
            answer_results.append(supplied_result)
            raise RuntimeError("simulated answer outage")

        response = analytics_service.process_question(
            "Deals by region",
            sql_generator=lambda _question: "SELECT 1;",
            query_runner=lambda _sql: result,
            answer_generator=failing_answer,
            query_logger=lambda _sql, _result: None,
        )

        assert answer_results == [result]
        assert response.result is result
        assert response.answer is None
        assert response.answer_fallback_used is True
        assert isinstance(response.answer_error, RuntimeError)

    def test_chart_fallback_stays_with_the_existing_chart_layer(self):
        result = make_result()

        response = analytics_service.process_question(
            "Show monthly pipeline as a pie chart.",
            sql_generator=lambda _question: "SELECT 1;",
            query_runner=lambda _sql: result,
            answer_generator=lambda _question, _result: "The trend is shown below.",
            chart_creator=lambda _question, supplied_result: (
                Path("fallback-chart.png"),
                ChartType.LINE,
                "A line chart was used because the data is temporal.",
            ),
            query_logger=lambda _sql, _result: None,
        )

        assert response.result is result
        assert response.chart_type is ChartType.LINE
        assert response.chart_note == "A line chart was used because the data is temporal."

    def test_chart_failure_preserves_the_answer_and_result(self):
        result = make_result()

        def failing_chart(_question: str, supplied_result: QueryResult):
            assert supplied_result is result
            raise ChartError("The result cannot be charted")

        response = analytics_service.process_question(
            "Deals by region and chart it.",
            sql_generator=lambda _question: "SELECT 1;",
            query_runner=lambda _sql: result,
            answer_generator=lambda _question, _result: "The table remains available.",
            chart_creator=failing_chart,
            query_logger=lambda _sql, _result: None,
        )

        assert response.result is result
        assert response.answer == "The table remains available."
        assert response.chart_path is None
        assert response.chart_error == "The result cannot be charted"


class TestCliCompatibility:
    """The terminal adapter consumes the service response without re-querying."""

    def test_answer_question_delegates_to_the_shared_service(self, monkeypatch, capsys):
        result = make_result()
        response = analytics_service.AnalysisResponse(
            original_question="Deals by region",
            effective_question="Deals by region",
            analytical_question="Deals by region",
            answer="NA has the most deals.",
            result=result,
            chart_requested=False,
            chart_path=None,
            chart_type=None,
            chart_note=None,
            answer_fallback_used=False,
            answer_error=None,
            chart_error=None,
            refused=False,
            elapsed_seconds=0.01,
        )
        seen: list[str] = []

        def fake_process(question: str, **_callbacks):
            seen.append(question)
            return response

        monkeypatch.setattr(app, "process_question", fake_process)

        assert app.answer_question("Deals by region") is True
        output = capsys.readouterr().out
        assert seen == ["Deals by region"]
        assert "NA has the most deals." in output
        assert "secret_internal_sql" not in output
