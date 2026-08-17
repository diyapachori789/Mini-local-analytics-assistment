"""Refused questions end to end: HTTP 200, friendly text, no red error.

The processor is the real analytics_service with a stubbed SQL generator, so the
refusal path is exercised for real without contacting Groq.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import analytics_service
import history_repository
import refusal
import web_app
from database import DatabaseConnectionError, QueryResult, SqlExecutionError, SqlValidationError
from llm import INVALID_QUESTION
from refusal import RefusalCategory, classify_refusal


@pytest.fixture
def client_factory(monkeypatch, tmp_path):
    """Build a client whose SQL generator is scripted, not a real model."""
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    monkeypatch.setattr(web_app, "CHARTS_DIR", charts_dir, raising=False)

    def build(sql_generator, query_runner=None, answer_generator=None):
        executions = []

        def runner(sql):
            executions.append(sql)
            if query_runner is not None:
                return query_runner(sql)
            frame = pd.DataFrame({"region": ["NA"], "deals": [92]})
            return QueryResult(frame=frame, sql=sql, row_count=1, truncated=False, max_rows=1000)

        def processor(question, original_question=None):
            return analytics_service.process_question(
                question,
                original_question=original_question,
                sql_generator=sql_generator,
                query_runner=runner,
                answer_generator=answer_generator or (lambda q, r: "answered"),
                chart_creator=lambda q, r: (_ for _ in ()).throw(
                    AssertionError("a refused question must never create a chart")
                ),
                query_logger=lambda *a: None,
            )

        application = web_app.create_app(process_question_func=processor)
        application.config.update(TESTING=True, CHARTS_DIR=charts_dir)
        application.executions = executions
        return application

    return build


def refuse_generator(_question):
    """Stands in for the model deciding the question is unanswerable."""
    return INVALID_QUESTION


def ask(client, question, **extra):
    payload = {"question": question}
    payload.update(extra)
    return client.post("/api/query", json=payload)


def load_saved_conversation(client, conversation_id: str) -> dict:
    """Fetch the safe persisted transcript for a browser query."""
    response = client.get(f"/api/conversations/{conversation_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    return payload["conversation"]


class TestRefusedResponseContract:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Show all tables", RefusalCategory.METADATA),
            ("Describe opportunities table", RefusalCategory.METADATA),
            ("What columns are in the database?", RefusalCategory.METADATA),
            ("DROP TABLE opportunities", RefusalCategory.UNSAFE_SQL),
            ("DELETE FROM opportunities", RefusalCategory.UNSAFE_SQL),
            ("What is the weather today?", RefusalCategory.OUT_OF_SCOPE),
            ("Tell me a joke", RefusalCategory.OUT_OF_SCOPE),
            ("deals asdkjh ???", RefusalCategory.UNSUPPORTED),
        ],
    )
    def test_refusal_is_a_successful_response_with_the_right_reply(
        self, client_factory, question, expected
    ):
        application = client_factory(refuse_generator)
        response = ask(application.test_client(), question)
        payload = response.get_json()

        # A refusal succeeds: no red error path in the browser.
        assert response.status_code == 200
        assert payload["ok"] is True
        assert payload["refused"] is True
        assert payload["meta"]["refused"] is True

        # The reply is the deterministic template for this category.
        assert classify_refusal(question) is expected
        assert payload["answer"] in refusal._TEMPLATES[expected]

    def test_refused_payload_has_an_empty_result(self, client_factory):
        application = client_factory(refuse_generator)
        payload = ask(application.test_client(), "Show all tables").get_json()

        assert payload["columns"] == []
        assert payload["rows"] == []
        assert payload["row_count"] == 0
        assert payload["truncated"] is False

    def test_refused_payload_has_no_chart(self, client_factory):
        application = client_factory(refuse_generator)
        payload = ask(application.test_client(), "Show all tables", chart_requested=True).get_json()

        assert payload["chart"]["requested"] is False
        assert payload["chart"]["url"] is None
        assert payload["chart"]["type"] is None

    def test_no_sql_is_executed_for_a_refusal(self, client_factory):
        application = client_factory(refuse_generator)
        ask(application.test_client(), "Show all tables")
        assert application.executions == [], "a refused question must not query the database"

    def test_same_question_gives_the_same_reply(self, client_factory):
        application = client_factory(refuse_generator)
        client = application.test_client()
        first = ask(client, "Show all tables").get_json()["answer"]
        for _ in range(4):
            assert ask(client, "Show all tables").get_json()["answer"] == first


class TestUnsafeSqlIsRefusedNotErrored:
    """A generated statement the guard rejects becomes a friendly refusal."""

    def test_guard_rejection_returns_200_with_a_friendly_reply(self, client_factory):
        def generates_unsafe_sql(_question):
            return "DROP TABLE opportunities;"

        def runner(sql):
            raise SqlValidationError("Refused to execute unsafe SQL: disallowed statement.")

        application = client_factory(generates_unsafe_sql, query_runner=runner)
        response = ask(application.test_client(), "DROP TABLE opportunities")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["refused"] is True
        assert payload["answer"] in refusal._TEMPLATES[RefusalCategory.UNSAFE_SQL]
        # The underlying reason never reaches the browser.
        assert "DROP" not in payload["answer"]
        assert "disallowed" not in response.get_data(as_text=True)

    def test_metadata_sql_is_blocked_and_refused(self, client_factory):
        """The guard now stops information_schema, so this cannot leak a schema."""
        import sql_guard

        def generates_metadata_sql(_question):
            return "SELECT table_name FROM information_schema.tables;"

        def runner(sql):
            sql_guard.validate_sql(sql)  # raises for metadata sources
            raise AssertionError("metadata SQL must never reach execution")

        def processor_runner(sql):
            try:
                sql_guard.validate_sql(sql)
            except ValueError as exc:
                raise SqlValidationError(str(exc)) from exc
            raise AssertionError("unreachable")

        application = client_factory(generates_metadata_sql, query_runner=processor_runner)
        response = ask(application.test_client(), "Show all tables")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["refused"] is True
        body = response.get_data(as_text=True)
        for leaked in ("information_schema", "table_name", "opportunity_id", "VARCHAR"):
            assert leaked not in body

    def test_generation_valueerror_is_refused(self, client_factory):
        """The multi-entity safeguard raising ValueError is an unsupported question."""

        def cannot_generate(_question):
            raise ValueError("The generated query left out OPP-1014.")

        application = client_factory(cannot_generate)
        response = ask(application.test_client(), "compare OPP-1003 to OPP-1014")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["refused"] is True
        # The internal reason is not shown to the user.
        assert "OPP-1014" not in payload["answer"]


class TestRealFailuresStayErrors:
    """Genuine outages must not be dressed up as friendly refusals."""

    def test_model_outage_is_still_an_error(self, client_factory):
        def provider_down(_question):
            raise RuntimeError("Groq API request failed: 503")

        response = ask(client_factory(provider_down).test_client(), "win rate by region")
        assert response.status_code == 503
        assert response.get_json()["ok"] is False
        assert "Groq" not in response.get_data(as_text=True)

    def test_database_execution_failure_is_still_an_error(self, client_factory):
        def runner(sql):
            raise SqlExecutionError("Failed to execute SQL: Binder Error")

        application = client_factory(lambda q: "SELECT bad FROM opportunities;", query_runner=runner)
        response = ask(application.test_client(), "win rate by region")
        assert response.status_code == 422
        assert response.get_json()["ok"] is False
        assert "Binder" not in response.get_data(as_text=True)

    def test_database_unavailable_is_still_an_error(self, client_factory):
        def runner(sql):
            raise DatabaseConnectionError("Database is not initialized.")

        application = client_factory(lambda q: "SELECT 1;", query_runner=runner)
        response = ask(application.test_client(), "win rate by region")
        assert response.status_code == 503
        assert response.get_json()["ok"] is False


class TestNormalPathUnchanged:
    def test_supported_question_still_returns_rows(self, client_factory):
        application = client_factory(
            lambda q: "SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;"
        )
        payload = ask(application.test_client(), "Show the first 5 opportunities").get_json()

        assert payload["refused"] is False
        assert payload["row_count"] == 1
        assert payload["columns"] == ["region", "deals"]
        assert len(application.executions) == 1, "single-query invariant"


class TestRefusalConversation:
    def test_refused_question_is_persisted(self, client_factory):
        application = client_factory(refuse_generator)
        client = application.test_client()
        payload = ask(client, "Show all tables").get_json()

        conversation = load_saved_conversation(client, payload["conversation_id"])
        assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
        user, assistant = conversation["messages"]
        assert user["content"] == "Show all tables"
        assert assistant["meta"]["refused"] is True
        assert assistant["row_count"] is None or assistant["row_count"] == 0
        assert assistant["chart"]["url"] is None
        assert assistant["content"] in refusal._TEMPLATES[RefusalCategory.METADATA]
        # Browser turns are not duplicated into the retained legacy flat endpoint.
        assert history_repository.list_history() == []

    def test_conversation_detail_stores_no_sql_for_a_refusal(self, client_factory):
        application = client_factory(refuse_generator)
        client = application.test_client()
        payload = ask(client, "DROP TABLE opportunities").get_json()

        assistant = load_saved_conversation(client, payload["conversation_id"])["messages"][-1]
        rendered = str(assistant)
        assert "sql" not in assistant
        for leaked in ("private_generated_sql", "information_schema", "Traceback", "gsk_", "C:\\"):
            assert leaked not in rendered
        assert "DROP" not in assistant["content"].upper()


class TestFrontendRefusalRendering:
    def test_client_renders_refusals_without_error_styling(self, client_factory):
        application = client_factory(refuse_generator)
        source = application.test_client().get("/static/js/app.js").get_data(as_text=True)

        assert "payload.refused === true" in source
        assert 'message.meta.refused ? "Assistant response" : "Analytics Assistant"' in source
        # A refusal is an assistant message in the stream, with no data block to
        # leave behind from a prior turn.
        assert "ui.conversationStream" in source
        data_block = source.split("function appendDataBlock")[1].split("function createDataToggle")[0]
        assert "if (message.meta.refused || !message.meta.hasResult) {" in data_block
        assert "return;" in data_block
        assert "renderResultTable" not in source
        assert "resetChart" not in source

    def test_client_uses_no_innerhtml(self, client_factory):
        application = client_factory(refuse_generator)
        source = application.test_client().get("/static/js/app.js").get_data(as_text=True)
        assert "innerHTML" not in source
        assert "textContent" in source
