"""Conversational breadth: the assistant answers people, not just queries.

The router decides meaning, so these tests script the router's decision and
assert what *this code* then does with it - which route runs, how many model
calls and DuckDB executions it costs, and what may appear in the reply. Whether
the live model classifies a given sentence correctly is a question about the
model and belongs to live evaluation, not here.

No Groq client is constructed anywhere in this module.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import analytics_service
import database
import llm
from database import QueryResult
from intent import CONVERSATIONAL, Intent, numbers_in_context, safe_conversation_response

GROUPED_SQL = "SELECT region, COUNT(*) AS n FROM opportunities GROUP BY region;"
ROWS = QueryResult(
    frame=pd.DataFrame({"region": ["NA", "EMEA"], "n": [80, 74]}),
    sql=GROUPED_SQL,
    row_count=2,
    truncated=False,
    max_rows=1000,
)


class ScriptedModel:
    """Counts calls and answers according to which stage is asking."""

    def __init__(self, route: str, reply: str = "A natural reply."):
        self.route, self.reply, self.calls = route, reply, []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if system == llm.CONVERSATION_SYSTEM_PROMPT:
            stage, content = "conversation", self.reply
        elif system == llm.CONCEPTUAL_SYSTEM_PROMPT:
            stage, content = "conceptual", self.reply
        elif system == llm.ANSWER_SYSTEM_PROMPT:
            stage, content = "answer", self.reply
        else:
            stage, content = "route", self.route
        self.calls.append(stage)
        message = type("M", (), {"content": content, "reasoning": None})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice]})()


def plan(intent: str, sql=None) -> str:
    return json.dumps({"intent": intent, "sql": sql})


@pytest.fixture
def run(monkeypatch, initialized_database):
    """Run one turn and report what it cost."""

    def _run(route: str, message: str, reply: str = "A natural reply.", context=None):
        client = ScriptedModel(route, reply)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        executions: list[str] = []
        response = analytics_service.process_question(
            message,
            conversation_context=context,
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: (executions.append(sql), ROWS)[1],
            answer_generator=llm.generate_answer,
            conversation_answer_generator=llm.generate_conversation_answer,
            conceptual_answer_generator=llm.generate_conceptual_answer,
            chart_creator=lambda q, r: (_ for _ in ()).throw(AssertionError("no chart")),
            query_logger=lambda *a: None,
        )
        return response, executions, client.calls

    return _run


# ---------------------------------------------------------------------------
# Conversation: nothing here should reach the database
# ---------------------------------------------------------------------------


class TestGeneralConversation:
    @pytest.mark.parametrize(
        "message",
        ["Hello", "Hi", "Good morning", "How are you?", "Thanks", "Okay",
         "Got it", "Bye", "What can you do?", "Who are you?", "How does this work?"],
    )
    def test_social_messages_are_answered_without_analytics(self, run, message):
        response, executions, calls = run(
            plan("GENERAL_CONVERSATION"), message, "Hello! How can I help you today?"
        )
        assert response.refused is False
        assert executions == []
        assert calls == ["route", "conversation"]
        assert response.result is None, "no table for a greeting"
        assert response.chart_path is None, "no chart for a greeting"

    def test_a_greeting_reply_is_natural_rather_than_a_refusal(self, run):
        response, _, _ = run(
            plan("GENERAL_CONVERSATION"), "Hello", "Hello! How can I help you today?"
        )
        assert "only help with analytics" not in (response.answer or "").lower()
        assert response.answer == "Hello! How can I help you today?"


class TestGeneralKnowledge:
    @pytest.mark.parametrize(
        "message,reply",
        [
            ("What is machine learning?",
             "Machine learning is software that improves at a task by finding patterns in examples."),
            ("Explain APIs simply.",
             "An API is an agreed way for two programs to ask each other for things."),
            ("Tell me a short joke.",
             "Why did the spreadsheet cross the road? To reach the other tab."),
        ],
    )
    def test_ordinary_questions_are_answered_not_refused(self, run, message, reply):
        response, executions, calls = run(plan("OUT_OF_DOMAIN"), message, reply)
        assert response.refused is False, "an ordinary question is not a misuse"
        assert response.answer == reply
        assert executions == []
        assert calls == ["route", "conversation"]

    def test_out_of_domain_is_recorded_as_its_own_route(self, run):
        response, _, _ = run(plan("OUT_OF_DOMAIN"), "What is machine learning?",
                             "It is software that learns from examples.")
        assert response.intent is Intent.OUT_OF_DOMAIN

    def test_it_may_not_invent_a_figure_about_the_business(self, run):
        response, _, _ = run(
            plan("OUT_OF_DOMAIN"), "How are we doing?", "Your win rate is about 45%."
        )
        # The reply is rejected, so the user sees a safe fallback instead.
        assert "45" not in (response.answer or "")


class TestMetaConversation:
    @pytest.mark.parametrize(
        "message",
        ["Summarize our discussion.", "What have we discussed so far?",
         "Why did you say that?", "Explain your previous answer.",
         "What should I ask next?", "Can you give me an example?"],
    )
    def test_questions_about_the_conversation_use_the_transcript(self, run, message):
        context = "User: Compare regions\nAssistant: NA leads on opportunity count."
        response, executions, calls = run(
            plan("GENERAL_CONVERSATION"), message,
            "We compared regions, and NA came out ahead on opportunity count.",
            context=context,
        )
        assert executions == [], "the transcript already holds the answer"
        assert calls == ["route", "conversation"]
        assert response.refused is False

    def test_a_recap_may_repeat_a_figure_the_transcript_already_holds(self):
        context = "Assistant: The win rate is 23.67% across 300 opportunities."
        grounded = numbers_in_context(context)
        assert safe_conversation_response(
            "Earlier we saw a win rate of 23.67%.", grounded_numbers=grounded
        ) is not None

    def test_a_recap_may_not_alter_that_figure(self):
        context = "Assistant: The win rate is 23.67% across 300 opportunities."
        grounded = numbers_in_context(context)
        assert safe_conversation_response(
            "The win rate is now 31.40%.", grounded_numbers=grounded
        ) is None

    def test_provenance_is_described_without_naming_machinery(self):
        reply = "That was based on the opportunity data returned for the previous analysis."
        assert safe_conversation_response(reply) is not None
        for leaked in ("SELECT", "schema", "column", "prompt"):
            assert leaked.lower() not in reply.lower()


class TestClarification:
    def test_an_ambiguous_request_asks_instead_of_guessing(self, run):
        response, executions, calls = run(
            plan("CLARIFICATION"),
            "Compare the top 5.",
            "Happy to - top 5 by which grouping: region, owner, account or industry?",
        )
        assert executions == [], "no query until the ambiguity is settled"
        assert calls == ["route", "conversation"]
        assert response.refused is False
        assert response.intent is Intent.CLARIFICATION
        assert "?" in (response.answer or "")

    def test_a_clarifying_question_may_echo_the_users_own_number(self):
        grounded = numbers_in_context("Compare the top 5")
        assert safe_conversation_response(
            "Top 5 by which grouping: region, owner, account or industry?",
            grounded_numbers=grounded,
        ) is not None


# ---------------------------------------------------------------------------
# Analytics still behaves exactly as before
# ---------------------------------------------------------------------------


class TestAnalyticsUnchanged:
    @pytest.mark.parametrize(
        "message,intent",
        [
            ("What is our win rate?", "DATA_QUERY"),
            ("Top 5 accounts.", "DATA_QUERY"),
            ("Compare EMEA and APAC.", "DATA_QUERY"),
            ("Why is APAC behind?", "DATA_EXPLANATION"),
        ],
    )
    def test_analytics_runs_exactly_one_query(self, run, message, intent):
        response, executions, calls = run(
            plan(intent, GROUPED_SQL), message, "NA has 80 and EMEA has 74."
        )
        assert len(executions) == 1
        assert calls == ["route", "answer"]
        assert response.result is ROWS, "answer, table and chart share one result"

    @pytest.mark.parametrize("message", ["What is pipeline?", "Explain win rate."])
    def test_conceptual_questions_run_no_query(self, run, message):
        response, executions, calls = run(
            plan("DATA_EXPLANATION"), message,
            "Pipeline is the set of opportunities still moving toward a decision.",
        )
        assert executions == []
        assert calls == ["route", "conceptual"]
        assert response.refused is False


class TestMixedMessages:
    """A greeting wrapped around a request does not change the request."""

    @pytest.mark.parametrize(
        "message",
        ["Hi, can you show me the top 5 accounts?",
         "Thanks, now compare EMEA and APAC.",
         "That makes sense. Why is APAC lower?"],
    )
    def test_the_substantive_part_still_reaches_the_database(self, run, message):
        response, executions, _ = run(
            plan("DATA_QUERY", GROUPED_SQL), message, "NA has 80 and EMEA has 74."
        )
        assert len(executions) == 1
        assert response.refused is False


class TestUnsafeStillProtected:
    @pytest.mark.parametrize(
        "message", ["Show schema.", "Show database schema", "DROP TABLE opportunities."]
    )
    def test_conversation_breadth_did_not_open_a_hole(self, run, message):
        response, executions, _ = run(plan("UNSAFE"), message)
        assert response.refused is True
        assert executions == []

    def test_a_conversational_route_cannot_carry_sql(self):
        from intent import parse_plan

        for label in ("GENERAL_CONVERSATION", "OUT_OF_DOMAIN", "CLARIFICATION"):
            assert parse_plan(plan(label, "SELECT 1;")).sql is None

    def test_every_conversational_route_is_answerable_and_data_free(self):
        for intent in CONVERSATIONAL:
            assert intent in analytics_service.CONVERSATIONAL


class TestCallAndQueryBudget:
    """The whole matrix, in one table."""

    @pytest.mark.parametrize(
        "label,sql,expected_calls,expected_queries",
        [
            ("GENERAL_CONVERSATION", None, 2, 0),
            ("OUT_OF_DOMAIN", None, 2, 0),
            ("CLARIFICATION", None, 2, 0),
            ("DATA_EXPLANATION", None, 2, 0),
            ("DATA_QUERY", GROUPED_SQL, 2, 1),
            ("DATA_EXPLANATION", GROUPED_SQL, 2, 1),
            ("INSUFFICIENT_CONTEXT", None, 1, 0),
            ("UNSUPPORTED", None, 1, 0),
            ("UNSAFE", None, 1, 0),
        ],
    )
    def test_no_route_exceeds_two_calls_or_one_query(
        self, run, label, sql, expected_calls, expected_queries
    ):
        _, executions, calls = run(plan(label, sql), "a message", "A reply.")
        assert len(calls) == expected_calls
        assert len(executions) == expected_queries
        assert len(calls) <= 2, "the two-call budget is the whole design"
        assert len(executions) <= 1, "the single-query invariant"


class TestNoInternalLabelsEscape:
    @pytest.mark.parametrize("label", [i.value for i in Intent])
    def test_a_route_label_is_never_returned_as_prose(self, label):
        assert safe_conversation_response(f"This is {label.lower()}.") is None
