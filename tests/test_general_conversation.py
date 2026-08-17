"""Offline matrix for natural conversation alongside the analytics pipeline.

Every model response is scripted. These tests exercise routing, orchestration,
persistence, and browser contracts without contacting Groq or the live service.
"""

from __future__ import annotations

import json
import tokenize
from pathlib import Path

import pandas as pd
import pytest

import analytics_service
import llm
import web_app
from database import QueryResult
from intent import Intent, QueryPlan


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYTICS_SQL = "SELECT COUNT(*) AS opportunity_count FROM opportunities;"
ANALYTICS_RESULT = QueryResult(
    frame=pd.DataFrame({"opportunity_count": [300]}),
    sql=ANALYTICS_SQL,
    row_count=1,
    truncated=False,
    max_rows=1000,
)


def plan_reply(
    intent: Intent | str,
    *,
    sql: str | None = None,
) -> str:
    label = intent.value if isinstance(intent, Intent) else intent
    return json.dumps({"intent": label, "sql": sql})


class ScriptedClient:
    """Return one semantic route and a deterministic second-stage answer."""

    def __init__(self, route: str, answer: str = "A grounded analytics answer."):
        self.route = route
        self.answer = answer
        self.calls: list[str] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "routing and SQL layer" in system:
            stage, content = "route", self.route
        elif "You are Analytics Assistant" in system:
            stage, content = "conversation", self.answer
        elif "what a\nmeasure in their sales-opportunity data means" in system:
            stage, content = "conceptual", self.answer
        else:
            stage, content = "answer", self.answer
        self.calls.append(stage)
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()


def install_client(monkeypatch, route: str, answer: str = "A grounded analytics answer."):
    client = ScriptedClient(route, answer)
    monkeypatch.setattr(llm, "_get_client", lambda: client)
    return client


def executable_source(module_name: str) -> str:
    kept: list[str] = []
    with open(PROJECT_ROOT / module_name, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                kept.append(token.string)
    return " ".join(kept).lower()


class TestNaturalConversation:
    @pytest.mark.parametrize(
        ("question", "reply"),
        [
            ("Hello", "Hello! How can I help you with your analytics today?"),
            ("Hi", "Hi! What would you like to explore in your opportunity data?"),
            ("Good morning", "Good morning! How can I help with your analytics?"),
            ("Thanks", "You're welcome! Let me know what else you'd like to explore."),
            ("Thank you", "You're welcome! I'm here whenever you need another analysis."),
            ("Who are you?", "I'm Analytics Assistant, here to help you understand your opportunity data."),
            ("What can you do?", "I can analyze opportunities, compare performance, explore trends, and show useful charts."),
            ("Can you help me?", "Absolutely. Tell me what you'd like to understand about your analytics."),
            ("Okay", "Got it. Let me know what you'd like to explore next."),
            ("Bye", "Goodbye! Come back anytime you want to explore your analytics."),
        ],
    )
    def test_social_turn_is_a_normal_zero_query_answer(
        self, monkeypatch, initialized_database, question, reply
    ):
        client = install_client(
            monkeypatch,
            plan_reply(Intent.GENERAL_CONVERSATION),
            reply,
        )
        executions: list[str] = []

        result = analytics_service.process_question(
            question,
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: executions.append(sql),
            answer_generator=lambda *args: pytest.fail("conversation has no answer-stage call"),
            conceptual_answer_generator=lambda *args: pytest.fail("conversation is not conceptual analytics"),
            chart_creator=lambda *args: pytest.fail("conversation has no chart"),
            query_logger=lambda *args: None,
        )

        assert result.intent is Intent.GENERAL_CONVERSATION
        assert result.answer == reply
        assert result.refused is False
        assert result.result is None
        assert result.chart_requested is False
        assert result.chart_path is None
        assert executions == []
        assert client.calls == ["route", "conversation"]

    def test_social_phrases_are_not_hard_coded_in_routing_logic(self):
        source = executable_source("intent.py") + executable_source("analytics_service.py")
        for phrase in ("hello", "good morning", "thank you", "bye"):
            assert phrase not in source

    def test_router_prompt_defines_conversation_and_mixed_intent_precedence(self):
        source = (PROJECT_ROOT / "llm.py").read_text(encoding="utf-8")
        assert "GENERAL_CONVERSATION" in source
        assert "Social wording never hides an analytics request" in source
        assert "Safety still outranks both conversation and analytics" in source
        # The name policy changed from "never use a name" to "use only a name
        # the person gave you here". What must stay true is that a name is
        # never taken from the surroundings or guessed - which is what the
        # blanket ban was really protecting. Asserted against the evaluated
        # prompt, not the file: in source these lines are separate literals.
        prompt = llm.CONVERSATION_SYSTEM_PROMPT
        assert "If the person states their own name in this conversation" in prompt
        assert "Never take\n  one from an email address" in prompt
        assert "never guess one from how they\n  write" in prompt
        assert "never treat\n  it as authorisation" in prompt


class TestRouteExecutionMatrix:
    def run_model_route(self, monkeypatch, route: str, question: str, answer: str = "Answer."):
        client = install_client(monkeypatch, route, answer)
        executions: list[str] = []
        response = analytics_service.process_question(
            question,
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: (executions.append(sql), ANALYTICS_RESULT)[1],
            query_logger=lambda *args: None,
        )
        return response, executions, client.calls

    def test_data_query_uses_one_query_and_two_model_calls(self, monkeypatch, initialized_database):
        response, executions, calls = self.run_model_route(
            monkeypatch,
            plan_reply(Intent.DATA_QUERY, sql=ANALYTICS_SQL),
            "What is our win rate?",
        )
        assert response.intent is Intent.DATA_QUERY
        assert len(executions) == 1
        assert calls == ["route", "answer"]

    def test_data_explanation_uses_one_query_and_two_model_calls(self, monkeypatch, initialized_database):
        response, executions, calls = self.run_model_route(
            monkeypatch,
            plan_reply(Intent.DATA_EXPLANATION, sql=ANALYTICS_SQL),
            "Why is our pipeline low?",
        )
        assert response.intent is Intent.DATA_EXPLANATION
        assert len(executions) == 1
        assert calls == ["route", "answer"]

    def test_conceptual_analytics_uses_no_query_and_two_model_calls(self, monkeypatch, initialized_database):
        response, executions, calls = self.run_model_route(
            monkeypatch,
            plan_reply(Intent.DATA_EXPLANATION),
            "What is pipeline?",
            "Pipeline represents the opportunities currently being worked.",
        )
        assert response.intent is Intent.DATA_EXPLANATION
        assert response.refused is False
        assert executions == []
        assert calls == ["route", "conceptual"]

    @pytest.mark.parametrize(
        "question",
        ["Hello, show me our top accounts.", "Thanks, now compare EMEA with APAC."],
    )
    def test_social_wrapper_does_not_mask_an_analytics_request(
        self, monkeypatch, initialized_database, question
    ):
        response, executions, calls = self.run_model_route(
            monkeypatch,
            plan_reply(Intent.DATA_QUERY, sql=ANALYTICS_SQL),
            question,
        )
        assert response.intent is Intent.DATA_QUERY
        assert len(executions) == 1
        assert calls == ["route", "answer"]

    @pytest.mark.parametrize("question", ["What's the weather?", "Tell me a joke."])
    def test_out_of_scope_is_natural_but_never_queries(
        self, monkeypatch, initialized_database, question
    ):
        response, executions, calls = self.run_model_route(
            monkeypatch,
            plan_reply(Intent.UNSUPPORTED),
            question,
        )
        assert response.refused is True
        assert response.answer is not None
        assert "analytics" in response.answer.lower() or "pipeline" in response.answer.lower()
        assert executions == []
        assert calls == ["route"]

    def test_metadata_pre_gate_executes_nothing(self):
        response = analytics_service.process_question(
            "Show database schema.",
            plan_generator=lambda *args, **kwargs: pytest.fail("unsafe pre-gate must win"),
            query_runner=lambda sql: pytest.fail("unsafe request must not query"),
            query_logger=lambda *args: None,
        )
        assert response.intent is Intent.UNSAFE
        assert response.refused is True
        assert response.result is None

    def test_unsafe_router_intent_executes_nothing(self):
        response = analytics_service.process_question(
            "DROP TABLE opportunities.",
            plan_generator=lambda question: QueryPlan(Intent.UNSAFE),
            query_runner=lambda sql: pytest.fail("unsafe request must not query"),
            query_logger=lambda *args: None,
        )
        assert response.intent is Intent.UNSAFE
        assert response.refused is True
        assert response.result is None

    def test_follow_up_context_reaches_only_the_fresh_planner(self):
        seen: list[tuple[str, str | None]] = []
        executions: list[str] = []

        def planner(question, *, conversation_context=None):
            seen.append((question, conversation_context))
            return QueryPlan(Intent.DATA_QUERY, ANALYTICS_SQL)

        response = analytics_service.process_question(
            "What about EMEA?",
            conversation_context="USER: Compare regional win rates.\nASSISTANT: NA currently leads.",
            plan_generator=planner,
            query_runner=lambda sql: (executions.append(sql), ANALYTICS_RESULT)[1],
            answer_generator=lambda question, result: "EMEA's current result is shown.",
            query_logger=lambda *args: None,
        )

        assert seen == [
            (
                "What about EMEA?",
                "USER: Compare regional win rates.\nASSISTANT: NA currently leads.",
            )
        ]
        assert len(executions) == 1
        assert response.result is ANALYTICS_RESULT


class TestConversationApiAndUi:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        charts = tmp_path / "charts"
        charts.mkdir()
        monkeypatch.setattr(web_app, "CHARTS_DIR", charts, raising=False)

        def processor(question, original_question=None, conversation_context=None):
            if question == "Hello":
                return analytics_service.process_question(
                    question,
                    original_question=original_question,
                    conversation_context=conversation_context,
                    plan_generator=lambda q: QueryPlan(
                        Intent.GENERAL_CONVERSATION,
                    ),
                    conversation_answer_generator=lambda q: (
                        "Hello! How can I help you with your analytics today?"
                    ),
                    query_runner=lambda sql: pytest.fail("greeting must not query"),
                    query_logger=lambda *args: None,
                )
            return analytics_service.process_question(
                question,
                original_question=original_question,
                conversation_context=conversation_context,
                plan_generator=lambda q: QueryPlan(Intent.DATA_QUERY, ANALYTICS_SQL),
                query_runner=lambda sql: ANALYTICS_RESULT,
                answer_generator=lambda q, result: "There are 300 opportunities.",
                query_logger=lambda *args: None,
            )

        app = web_app.create_app(
            process_question_func=processor,
            initialize_database_at_start=False,
        )
        app.config.update(TESTING=True, CHARTS_DIR=charts)
        return app.test_client()

    def test_pure_conversation_has_no_result_or_chart_ui_metadata(self, client):
        response = client.post("/api/query", json={"question": "Hello"})
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["answer"].startswith("Hello!")
        assert payload["columns"] == []
        assert payload["rows"] == []
        assert payload["meta"]["has_result"] is False
        assert payload["chart"]["requested"] is False
        assert "intent" not in response.get_data(as_text=True).lower()

        detail = client.get(
            f"/api/conversations/{payload['conversation_id']}"
        ).get_json()["conversation"]
        assistant = detail["messages"][1]
        assert assistant["row_count"] is None
        assert assistant["meta"]["has_result"] is False
        assert assistant["chart"]["requested"] is False
        assert detail["title"] == "New chat"

    def test_first_meaningful_analytics_turn_replaces_neutral_title(self, client):
        greeting = client.post("/api/query", json={"question": "Hello"}).get_json()
        conversation_id = greeting["conversation_id"]
        response = client.post(
            "/api/query",
            json={
                "question": "Show pipeline by stage.",
                "conversation_id": conversation_id,
            },
        )
        assert response.status_code == 200
        detail = client.get(f"/api/conversations/{conversation_id}").get_json()
        assert detail["conversation"]["title"] == "Show pipeline by stage."
        assert [message["role"] for message in detail["conversation"]["messages"]] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_frontend_uses_transient_thinking_and_no_result_gate(self, client):
        source = client.get("/static/js/app.js").get_data(as_text=True)
        assert 'body.textContent = "Thinking..."' in source
        assert 'setQuestionMessage("Thinking...", "")' in source
        assert "Analysis complete" not in source
        assert "message.meta.refused || !message.meta.hasResult" in source
        assert "metadata.has_result === true" in source
        assert "innerHTML" not in source
