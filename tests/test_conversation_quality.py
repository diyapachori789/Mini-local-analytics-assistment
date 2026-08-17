"""Conversational replies should sound like a person, not a form.

Routing correctness is covered elsewhere. What is pinned here is the *style*
contract: the instructions the answer stage is given, what the output filter
lets through, and what an ordinary exchange costs. Model prose is not asserted
- that would be testing the model - so the wording checks target the prompt,
which is the thing this code actually controls.

No Groq client is constructed in this module.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import analytics_service
import llm
from database import QueryResult
from intent import Intent, numbers_in_context, safe_conversation_response

PROMPT = llm.CONVERSATION_SYSTEM_PROMPT

GROUPED_SQL = "SELECT account_name, SUM(amount) AS total FROM opportunities GROUP BY account_name;"
ROWS = QueryResult(
    frame=pd.DataFrame({"account_name": ["Acme", "Summit"], "total": [900, 800]}),
    sql=GROUPED_SQL, row_count=2, truncated=False, max_rows=1000,
)


class Scripted:
    def __init__(self, route: str, reply: str):
        self.route, self.reply, self.calls = route, reply, []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        stage = "route" if "routing and SQL layer" in system else "reply"
        self.calls.append(stage)
        content = self.route if stage == "route" else self.reply
        message = type("M", (), {"content": content, "reasoning": None})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice]})()


@pytest.fixture
def turn(monkeypatch, initialized_database):
    def _turn(intent: str, message: str, reply: str, sql=None, context=None):
        client = Scripted(json.dumps({"intent": intent, "sql": sql}), reply)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        executions: list[str] = []
        response = analytics_service.process_question(
            message,
            conversation_context=context,
            plan_generator=llm.generate_plan,
            query_runner=lambda s: (executions.append(s), ROWS)[1],
            answer_generator=llm.generate_answer,
            conversation_answer_generator=llm.generate_conversation_answer,
            conceptual_answer_generator=llm.generate_conceptual_answer,
            chart_creator=lambda q, r: (_ for _ in ()).throw(AssertionError("no chart")),
            query_logger=lambda *a: None,
        )
        return response, executions, client.calls

    return _turn


class TestThePromptNoLongerForcesAnalytics:
    """The lines that produced 'How can I help you with analytics today?'."""

    def test_identity_is_conversational_first(self):
        assert "natural conversational assistant" in PROMPT
        assert "focused on\nhelping people understand sales-opportunity analytics" not in PROMPT

    def test_the_stay_analytics_focused_instruction_is_gone(self):
        assert "Stay analytics-focused" not in PROMPT

    def test_redirecting_ordinary_conversation_is_forbidden(self):
        assert "Do not steer ordinary conversation back to analytics" in PROMPT
        assert "A greeting does not" in PROMPT

    def test_the_bad_greeting_is_named_as_the_counter_example(self):
        assert "How can I help you with sales-opportunity analytics today?" in PROMPT
        assert "answers a question nobody asked" in PROMPT

    def test_off_topic_answers_may_not_be_wrapped_in_a_deflection(self):
        assert "do not append an\n  offer to help with analytics" in PROMPT
        assert "Asked for a joke, tell one." in PROMPT

    def test_capabilities_are_still_described_when_asked(self):
        assert "This is the moment to\n  describe the analytics" in PROMPT


class TestNameHandling:
    def test_an_explicitly_given_name_may_be_used(self):
        assert "If the person states their own name in this conversation" in PROMPT
        assert "Greeting them back by name is the ordinary human response" in PROMPT

    def test_an_inferred_name_is_still_forbidden(self):
        assert "Never take\n  one from an email address" in PROMPT
        assert "never guess one from how they\n  write" in PROMPT

    def test_a_name_is_not_treated_as_identity(self):
        assert "never treat\n  it as authorisation" in PROMPT

    def test_a_name_survives_the_context_sanitiser(self):
        context = "User: Hello my name is Diya\nAssistant: Nice to meet you, Diya!"
        assert "Diya" in llm._safe_conversation_context(context)

    @pytest.mark.parametrize(
        "reply",
        [
            "Hi Diya! Nice to meet you. How can I help you today?",
            "Nice to meet you, Diya!",
            "Hello Diya! What would you like to talk about?",
        ],
    )
    def test_a_natural_name_reply_is_not_blocked(self, reply):
        assert safe_conversation_response(reply) is not None

    def test_an_email_address_is_blocked_even_though_a_name_is_not(self):
        """'Greet them by the name in their email' is the shortcut to prevent."""
        assert safe_conversation_response("Hi diya.pachori@example.com!") is None
        assert safe_conversation_response("Hi Diya!") is not None

    def test_a_filesystem_derived_name_is_blocked(self):
        assert safe_conversation_response("Your file is at C:/Users/diyad/notes.txt") is None

    def test_a_name_from_earlier_context_reaches_the_answer_stage(self, monkeypatch):
        captured = {}

        class Fake:
            def __init__(self):
                self.chat = self

            @property
            def completions(self):
                return self

            def create(self, **kwargs):
                captured["user"] = kwargs["messages"][1]["content"]
                message = type("M", (), {"content": "Hi Diya!", "reasoning": None})()
                choice = type("C", (), {"message": message, "finish_reason": "stop"})()
                return type("R", (), {"choices": [choice]})()

        monkeypatch.setattr(llm, "_get_client", Fake)
        answer = llm.generate_conversation_answer(
            "Hello again", conversation_context="User: My name is Diya"
        )
        assert "Diya" in captured["user"], "the name must be visible to the answer stage"
        assert answer == "Hi Diya!"


class TestOrdinaryExchangesCostNothing:
    @pytest.mark.parametrize(
        "message,reply",
        [
            ("Hello", "Hi! How can I help you today?"),
            ("Hello my name is Diya", "Hi Diya! Nice to meet you."),
            ("How are you?", "I'm doing well, thanks for asking! How can I help?"),
            ("Thanks", "You're welcome!"),
            ("What can you do?",
             "I can chat normally, and dig into your opportunity data - by region, "
             "owner or account, trends over time, with charts where they help."),
            ("Who are you?",
             "I'm Analytics Assistant. I can chat with you normally and analyse "
             "your opportunity data when you need it."),
        ],
    )
    def test_conversation_runs_no_query_and_two_calls(self, turn, message, reply):
        response, executions, calls = turn("GENERAL_CONVERSATION", message, reply)
        assert executions == []
        assert calls == ["route", "reply"]
        assert response.answer == reply
        assert response.result is None and response.chart_path is None

    @pytest.mark.parametrize(
        "message,reply",
        [
            ("Tell me a joke",
             "Why did the spreadsheet cross the road? To reach the other tab."),
            ("What is machine learning?",
             "Machine learning is software that improves at a task by finding "
             "patterns in examples rather than being told every rule."),
        ],
    )
    def test_off_topic_answers_are_given_not_deflected(self, turn, message, reply):
        response, executions, _ = turn("OUT_OF_DOMAIN", message, reply)
        assert executions == []
        assert response.refused is False
        assert response.answer == reply
        lowered = (response.answer or "").lower()
        for deflection in ("i'm mainly focused on", "i can only help with",
                           "with your analytics", "sales-opportunity analytics"):
            assert deflection not in lowered

    def test_a_greeting_reply_carries_no_analytics_redirect(self, turn):
        response, _, _ = turn("GENERAL_CONVERSATION", "Hello", "Hi! How can I help you today?")
        assert "analytics" not in (response.answer or "").lower()


class TestAnalyticsStillWins:
    @pytest.mark.parametrize(
        "message",
        ["Hi, show me the top 5 accounts",
         "Hi Diya here - can you show me our top 5 accounts?",
         "Thanks, compare EMEA and APAC",
         "Thanks. Now compare EMEA and APAC."],
    )
    def test_a_polite_wrapper_does_not_hide_the_request(self, turn, message):
        response, executions, calls = turn(
            "DATA_QUERY", message, "Acme leads with 900.", sql=GROUPED_SQL
        )
        assert len(executions) == 1
        assert calls == ["route", "reply"]
        assert response.result is ROWS


class TestConversationalRepliesStayGrounded:
    def test_no_business_figure_may_be_invented(self, turn):
        response, _, _ = turn(
            "GENERAL_CONVERSATION", "How are we doing?", "Your win rate is 45%."
        )
        assert "45" not in (response.answer or "")

    def test_a_figure_already_in_the_transcript_may_be_repeated(self):
        context = "Assistant: The win rate is 23.67% across 300 opportunities."
        assert safe_conversation_response(
            "Earlier we saw a win rate of 23.67%.",
            grounded_numbers=numbers_in_context(context),
        ) is not None

    @pytest.mark.parametrize("leak", ["SELECT * FROM opportunities;",
                                      "The opportunity_id column stores it.",
                                      "This is general_conversation."])
    def test_internal_detail_never_reaches_a_conversational_reply(self, leak):
        assert safe_conversation_response(leak) is None
