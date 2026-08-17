"""Offline regression coverage for persistent conversation storage and APIs.

The module deliberately injects every analytics response.  It uses the
per-test history database supplied by ``conftest.py`` and must never initialize
the analytics database or construct a Groq client.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import fields

import duckdb
import pandas as pd
import pytest

import history_repository
import web_app
from analytics_service import AnalysisResponse
from config import (
    CONVERSATION_CONTEXT_MAX_CHARS,
    CONVERSATION_CONTEXT_MAX_MESSAGES,
    CONVERSATION_CONTEXT_MAX_MESSAGE_CHARS,
    CONVERSATION_TITLE_MAX_CHARS,
)
from database import QueryResult


def make_result() -> QueryResult:
    """Build an in-memory result whose statement must never be persisted."""
    frame = pd.DataFrame({"region": ["NA", "EMEA"], "deals": [92, 69]})
    return QueryResult(
        frame=frame,
        sql="SELECT private_generated_sql FROM opportunities;",
        row_count=len(frame),
        truncated=False,
        max_rows=1000,
    )


def make_response(question: str, *, result: QueryResult | None = None) -> AnalysisResponse:
    """Return a completed response without executing analytics work."""
    return AnalysisResponse(
        original_question=question,
        effective_question=question,
        analytical_question=question,
        answer="NA leads with 92 deals.",
        result=make_result() if result is None else result,
        chart_requested=False,
        chart_path=None,
        chart_type=None,
        chart_note=None,
        answer_fallback_used=False,
        answer_error=None,
        chart_error=None,
        refused=False,
        elapsed_seconds=0.125,
    )


def save_turn(
    conversation_id: str,
    user_content: str,
    assistant_content: str | None = "Grounded answer.",
    **overrides: object,
):
    """Save a normal completed turn with explicit safe metadata."""
    values: dict[str, object] = {
        "row_count": 2,
        "truncated": False,
        "max_rows": 1000,
        "chart_requested": False,
        "chart_type": None,
        "chart_filename": None,
        "chart_note": None,
        "answer_fallback_used": False,
        "refused": False,
        "success": True,
        "elapsed_seconds": 0.125,
    }
    values.update(overrides)
    return history_repository.save_conversation_turn(
        conversation_id,
        user_content,
        assistant_content,
        **values,
    )


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    """Build Flask test clients with only faked analytics processing."""
    charts_dir = tmp_path / "charts"
    charts_dir.mkdir()
    monkeypatch.setattr(web_app, "CHARTS_DIR", charts_dir, raising=False)

    def build(processor):
        application = web_app.create_app(
            process_question_func=processor,
            initialize_database_at_start=False,
        )
        application.config.update(TESTING=True, CHARTS_DIR=charts_dir)
        return application

    return build


def create_api_conversation(client, *, title: str | None = None) -> dict[str, object]:
    payload = {} if title is None else {"title": title}
    response = client.post("/api/conversations", json=payload)
    assert response.status_code in {200, 201}
    body = response.get_json()
    assert body["ok"] is True
    return body["conversation"]


class TestConversationRepository:
    """The history database owns conversation records and safe message data."""

    def test_initialize_migrates_legacy_history_once_without_losing_it(
        self, isolated_history_database
    ):
        history_id = history_repository.save_history(
            original_question="Legacy pipeline question",
            answer="Legacy answer.",
        )
        null_answer_id = history_repository.save_history(
            original_question="Legacy question without an answer",
            answer=None,
            success=False,
        )

        history_repository.initialize_history_database()
        history_repository.initialize_history_database()

        legacy_ids = {record.history_id for record in history_repository.list_history()}
        assert legacy_ids == {history_id, null_answer_id}

        conversations = history_repository.list_conversations()
        assert len(conversations) == 2, "Repeated initialization must not duplicate migration."
        migrated = {
            conversation.messages[0].content: conversation
            for summary in conversations
            if (conversation := history_repository.load_conversation(summary.conversation_id))
            is not None
        }
        first = migrated["Legacy pipeline question"]
        assert [message.role for message in first.messages] == ["user", "assistant"]
        assert first.messages[1].content == "Legacy answer."
        null_answer = migrated["Legacy question without an answer"]
        assert null_answer.messages[1].content

        connection = duckdb.connect(str(isolated_history_database))
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ).fetchall()
            }
        finally:
            connection.close()
        assert {"conversations", "conversation_messages"}.issubset(tables)

    def test_create_list_load_and_delete_conversations(self):
        first = history_repository.create_conversation("First thread")
        second = history_repository.create_conversation("Second thread")

        summaries = history_repository.list_conversations()
        assert {summary.conversation_id for summary in summaries} == {
            first.conversation_id,
            second.conversation_id,
        }
        assert all(summary.message_count == 0 for summary in summaries)

        loaded = history_repository.load_conversation(first.conversation_id)
        assert loaded is not None
        assert loaded.conversation_id == first.conversation_id
        assert loaded.title == "First thread"
        assert loaded.messages == ()

        assert history_repository.delete_conversation(first.conversation_id) is True
        assert history_repository.load_conversation(first.conversation_id) is None
        assert history_repository.delete_conversation(first.conversation_id) is False
        assert history_repository.delete_all_conversations() == 1
        assert history_repository.list_conversations() == []

    def test_save_turn_is_atomic_and_persists_a_complete_user_assistant_pair(self):
        summary = history_repository.create_conversation()

        with pytest.raises(history_repository.HistoryError):
            save_turn(summary.conversation_id, "Keep this user message out on failure.", object())

        after_failed_turn = history_repository.load_conversation(summary.conversation_id)
        assert after_failed_turn is not None
        assert after_failed_turn.messages == ()

        updated = save_turn(
            summary.conversation_id,
            "Show pipeline by region.",
            "NA leads with 92 deals.",
        )
        loaded = history_repository.load_conversation(summary.conversation_id)
        assert loaded is not None
        assert updated.message_count == 2
        assert [message.role for message in loaded.messages] == ["user", "assistant"]
        assert [message.ordinal for message in loaded.messages] == [1, 2]
        assert [message.content for message in loaded.messages] == [
            "Show pipeline by region.",
            "NA leads with 92 deals.",
        ]

    def test_first_user_turn_creates_a_bounded_stable_title(self):
        summary = history_repository.create_conversation()
        first_question = "  Show pipeline by region for the current fiscal year, with detail.  "

        saved = save_turn(summary.conversation_id, first_question)
        save_turn(summary.conversation_id, "Now compare the owners.")
        loaded = history_repository.load_conversation(summary.conversation_id)

        assert loaded is not None
        assert 0 < len(saved.title) <= CONVERSATION_TITLE_MAX_CHARS
        assert saved.title.lower().startswith("show pipeline by region")
        assert loaded.title == saved.title
        assert loaded.message_count == 4

    def test_context_is_bounded_to_recent_messages_and_isolated_per_conversation(self):
        first = history_repository.create_conversation()
        second = history_repository.create_conversation()

        for number in range(5):
            save_turn(
                first.conversation_id,
                f"First conversation user message {number}: " + ("u" * 180),
                f"First conversation assistant message {number}: " + ("a" * 180),
            )
        save_turn(second.conversation_id, "Other conversation marker", "Other answer marker")

        context = history_repository.get_conversation_context(first.conversation_id)

        assert isinstance(context, str)
        assert len(context) <= CONVERSATION_CONTEXT_MAX_CHARS
        assert "First conversation user message 4" in context
        assert "First conversation assistant message 4" in context
        assert "First conversation user message 0" not in context
        assert "Other conversation marker" not in context

        # The context window is expressed as messages, not arbitrary history
        # rows, and each stored item has an independent bounded contribution.
        recent_lines = [line for line in context.splitlines() if line.strip()]
        assert len(recent_lines) <= CONVERSATION_CONTEXT_MAX_MESSAGES
        assert all(len(line) <= CONVERSATION_CONTEXT_MAX_MESSAGE_CHARS + 32 for line in recent_lines)

    def test_public_conversation_models_cannot_carry_sql_or_schema_fields(
        self, isolated_history_database
    ):
        summary = history_repository.create_conversation("Safe thread")
        save_turn(summary.conversation_id, "Question", "Answer")
        conversation = history_repository.load_conversation(summary.conversation_id)

        assert conversation is not None
        models = (
            type(summary),
            type(conversation),
            type(conversation.messages[0]),
        )
        names = {
            field.name.lower()
            for model in models
            for field in fields(model)
        }
        assert not any("sql" in name or "schema" in name for name in names)
        assert not any("path" in name for name in names)

        # Storage mirrors the public model boundary: saving a turn must not
        # create a place to retain generated query text or database metadata.
        connection = duckdb.connect(str(isolated_history_database))
        try:
            stored_names = {
                str(row[1]).lower()
                for row in connection.execute(
                    "SELECT table_name, column_name FROM information_schema.columns "
                    "WHERE table_schema = 'main' "
                    "AND table_name IN ('conversations', 'conversation_messages')"
                ).fetchall()
            }
        finally:
            connection.close()
        assert not any("sql" in name or "schema" in name for name in stored_names)

    def test_retained_assistant_text_and_context_never_keep_internal_sql_or_secrets(self):
        summary = history_repository.create_conversation()
        fake_key = "gsk_conversation_test_secret"
        save_turn(
            summary.conversation_id,
            "Show the current pipeline.",
            f"SELECT private_generated_sql; provider key {fake_key} at C:\\secret\\trace.log",
        )

        loaded = history_repository.load_conversation(summary.conversation_id)
        assert loaded is not None
        assistant_text = loaded.messages[1].content
        context = history_repository.get_conversation_context(summary.conversation_id)
        for forbidden in ("private_generated_sql", fake_key, "C:\\secret"):
            assert forbidden not in assistant_text
            assert forbidden not in context

    def test_persisted_turns_redact_bearer_values_provider_keys_and_unix_paths(self):
        summary = history_repository.create_conversation()
        bearer_value = "bearer_token_value_12345"
        provider_key = "provider_secret_value_67890"
        save_turn(
            summary.conversation_id,
            "Compare regions. Authorization: Bearer "
            f"{bearer_value}; OPENAI_API_KEY={provider_key}; see /root/.env and /opt/private.txt.",
            "The current regional comparison is ready.",
        )

        loaded = history_repository.load_conversation(summary.conversation_id)
        assert loaded is not None
        retained = loaded.messages[0].content
        context = history_repository.get_conversation_context(summary.conversation_id)
        for forbidden in (bearer_value, provider_key, "/root/.env", "/opt/private.txt"):
            assert forbidden not in retained
            assert forbidden not in context
        assert loaded.title == "New chat"

    def test_bounded_transcript_sets_an_explicit_truncation_flag(self):
        summary = history_repository.create_conversation()
        for number in range(3):
            save_turn(summary.conversation_id, f"Question {number}", f"Answer {number}")

        loaded = history_repository.load_conversation(summary.conversation_id, limit=2)
        assert loaded is not None
        assert loaded.message_count == 6
        assert loaded.messages_truncated is True
        assert [message.ordinal for message in loaded.messages] == [5, 6]


class TestConversationApi:
    """The HTTP adapter exposes summaries/details without replaying analytics."""

    def test_crud_routes_return_only_the_documented_safe_shapes(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: pytest.fail("no analytics"))
        client = application.test_client()

        assert client.get("/api/conversations").get_json() == {
            "ok": True,
            "conversations": [],
        }
        first = create_api_conversation(client)
        second = create_api_conversation(client)

        listed = client.get("/api/conversations")
        assert listed.status_code == 200
        entries = listed.get_json()["conversations"]
        assert {entry["conversation_id"] for entry in entries} == {
            first["conversation_id"],
            second["conversation_id"],
        }
        assert set(entries[0]) == {
            "conversation_id",
            "title",
            "created_at",
            "updated_at",
            "message_count",
        }

        deleted = client.delete(f"/api/conversations/{first['conversation_id']}")
        assert deleted.status_code == 200
        assert deleted.get_json() == {"ok": True, "deleted": 1}
        cleared = client.delete("/api/conversations")
        assert cleared.status_code == 200
        assert cleared.get_json() == {"ok": True, "deleted": 1}

    def test_query_keeps_current_rows_but_detail_has_messages_only_and_never_replays_processor(
        self, app_factory
    ):
        calls: list[tuple[str, dict[str, object]]] = []

        def processor(question: str, **kwargs: object) -> AnalysisResponse:
            calls.append((question, dict(kwargs)))
            original = kwargs.get("original_question")
            return make_response(original if isinstance(original, str) else question)

        client = app_factory(processor).test_client()
        conversation = create_api_conversation(client)
        response = client.post(
            "/api/query",
            json={
                "question": "Show deals by region.",
                "conversation_id": conversation["conversation_id"],
            },
        )
        payload = response.get_json()
        rendered = json.dumps(payload)

        assert response.status_code == 200
        assert payload["conversation_id"] == conversation["conversation_id"]
        assert payload["conversation_saved"] is True
        assert payload["columns"] == ["region", "deals"]
        assert payload["rows"] == [["NA", 92], ["EMEA", 69]]
        assert len(calls) == 1
        assert "private_generated_sql" not in rendered
        assert "schema" not in rendered.lower()

        detail = client.get(f"/api/conversations/{conversation['conversation_id']}")
        detail_payload = detail.get_json()
        detail_rendered = json.dumps(detail_payload)

        assert detail.status_code == 200
        assert len(calls) == 1, "Loading saved conversation must not run analytics."
        saved = detail_payload["conversation"]
        assert set(saved) == {
            "conversation_id",
            "title",
            "created_at",
            "updated_at",
            "messages_truncated",
            "messages",
        }
        assert "rows" not in saved and "columns" not in saved
        assert saved["messages_truncated"] is False
        assert [message["role"] for message in saved["messages"]] == ["user", "assistant"]
        assert saved["messages"][0]["content"] == "Show deals by region."
        assert saved["messages"][1]["content"] == "NA leads with 92 deals."
        for message in saved["messages"]:
            assert set(message) == {
                "id",
                "role",
                "content",
                "created_at",
                "row_count",
                "truncated",
                "chart",
                "meta",
            }
            assert set(message["chart"]) == {"requested", "type", "url", "note", "available"}
            assert set(message["meta"]) == {
                "refused",
                "success",
                "answer_fallback_used",
                "has_result",
                "elapsed_ms",
            }
            assert "rows" not in message
            assert "columns" not in message
        assert '"rows"' not in detail_rendered
        assert '"columns"' not in detail_rendered
        for forbidden in ("private_generated_sql", "schema", "GROQ", "gsk_", "Traceback", "C:\\"):
            assert forbidden not in detail_rendered

    def test_query_without_a_conversation_creates_one_only_after_the_fake_processor_succeeds(
        self, app_factory
    ):
        calls = 0

        def processor(question: str, **kwargs: object) -> AnalysisResponse:
            nonlocal calls
            calls += 1
            original = kwargs.get("original_question")
            return make_response(original if isinstance(original, str) else question)

        client = app_factory(processor).test_client()
        response = client.post("/api/query", json={"question": "Show deals by region."})
        payload = response.get_json()

        assert response.status_code == 200
        assert isinstance(payload["conversation_id"], str)
        assert payload["conversation_id"]
        assert payload["conversation_saved"] is True
        assert calls == 1
        assert len(client.get("/api/conversations").get_json()["conversations"]) == 1

    @pytest.mark.parametrize("conversation_id", ["not-a-conversation-id", str(uuid.uuid4())])
    def test_invalid_or_missing_provided_conversation_id_is_rejected_before_processor(
        self, app_factory, conversation_id
    ):
        calls = 0

        def forbidden_processor(*_args: object, **_kwargs: object) -> AnalysisResponse:
            nonlocal calls
            calls += 1
            raise AssertionError("Invalid conversation ids must not reach analytics processing")

        client = app_factory(forbidden_processor).test_client()
        response = client.post(
            "/api/query",
            json={"question": "Show deals by region.", "conversation_id": conversation_id},
        )
        body = response.get_json()
        rendered = json.dumps(body)

        assert response.status_code in {400, 404}
        assert body["ok"] is False
        assert body["error"]["code"]
        assert calls == 0
        assert "private_generated_sql" not in rendered
        assert "Traceback" not in rendered
