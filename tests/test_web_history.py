"""Persistent conversations through the web adapter, plus legacy history and charts.

Every test is offline: the analytics processor is a stub, history uses the
temporary database from the autouse conftest fixture, and no Groq call is made.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import history_repository
import web_app
from analytics_service import AnalysisResponse
from chart import ChartType, create_chart
from config import DATABASE_NAME
from database import QueryResult, SqlValidationError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_result(frame: pd.DataFrame | None = None) -> QueryResult:
    data = frame if frame is not None else pd.DataFrame(
        {"region": ["NA", "EMEA", "LATAM", "APAC"], "deals": [92, 69, 69, 70]}
    )
    return QueryResult(
        frame=data,
        sql="SELECT region, COUNT(*) AS deals FROM opportunities GROUP BY region;",
        row_count=len(data),
        truncated=False,
        max_rows=1000,
    )


def make_response(**overrides) -> AnalysisResponse:
    defaults = {
        "original_question": "What is the win rate by region?",
        "effective_question": "What is the win rate by region?",
        "analytical_question": "What is the win rate by region?",
        "answer": "NA has the highest win rate.",
        "result": make_result(),
        "chart_requested": False,
        "chart_path": None,
        "chart_type": None,
        "chart_note": None,
        "answer_fallback_used": False,
        "answer_error": None,
        "chart_error": None,
        "refused": False,
        "elapsed_seconds": 0.0123,
    }
    defaults.update(overrides)
    return AnalysisResponse(**defaults)


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    """Build isolated apps that serve charts from a pytest directory."""
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

    build.charts_dir = charts_dir
    return build


def ask(client, question: str = "What is the win rate by region?", **extra):
    payload = {"question": question}
    payload.update(extra)
    return client.post("/api/query", json=payload)


def load_saved_conversation(client, conversation_id: str) -> dict:
    """Fetch the safe persisted transcript for one completed browser turn."""
    response = client.get(f"/api/conversations/{conversation_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    return payload["conversation"]


class TestHistoryApiContract:
    def test_empty_history_initially(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        assert client.get("/api/history").get_json() == {"ok": True, "history": []}

    def test_entries_are_newest_first(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        for question in ("first", "second", "third"):
            history_repository.save_history(original_question=question, answer="ok")
        entries = client.get("/api/history").get_json()["history"]
        assert [entry["question"] for entry in entries] == ["third", "second", "first"]

    def test_entry_shape_is_the_documented_contract(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(
            original_question="Deals by region",
            answer="NA leads.",
            row_count=4,
            chart_requested=True,
            chart_type="bar",
            chart_filename="missing.png",
            elapsed_seconds=1.5,
        )
        entry = client.get("/api/history").get_json()["history"][0]
        assert set(entry) == {
            "id", "created_at", "question", "answer", "row_count", "truncated", "chart", "meta",
        }
        assert set(entry["chart"]) == {"requested", "type", "url", "note", "available"}
        assert set(entry["meta"]) == {
            "answer_fallback_used", "refused", "success", "elapsed_seconds",
        }

    def test_created_at_is_iso8601_utc(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(original_question="q", answer="a")
        created = client.get("/api/history").get_json()["history"][0]["created_at"]
        assert created.endswith("+00:00")
        assert datetime.fromisoformat(created).tzinfo is not None

    @pytest.mark.parametrize(
        "forbidden",
        ["SELECT", "sql", "schema", "information_schema", "GROQ", "gsk_", "C:\\", "Traceback"],
    )
    def test_history_exposes_no_sql_schema_paths_or_secrets(self, app_factory, forbidden):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(
            original_question="Deals by region",
            answer="NA leads.",
            chart_requested=True,
            chart_filename="chart.png",
        )
        assert forbidden not in client.get("/api/history").get_data(as_text=True)


class TestHistoryChartAvailability:
    def test_existing_chart_gets_a_relative_url(self, app_factory):
        (app_factory.charts_dir / "saved_chart.png").write_bytes(PNG_SIGNATURE)
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(
            original_question="q", answer="a", chart_requested=True,
            chart_type="bar", chart_filename="saved_chart.png",
        )
        chart = client.get("/api/history").get_json()["history"][0]["chart"]
        assert chart["available"] is True
        assert chart["url"] == "/charts/saved_chart.png"

    def test_missing_chart_is_reported_unavailable_without_losing_the_record(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(
            original_question="q", answer="a", chart_requested=True,
            chart_type="bar", chart_filename="deleted.png",
        )
        entries = client.get("/api/history").get_json()["history"]
        assert len(entries) == 1
        assert entries[0]["chart"]["available"] is False
        assert entries[0]["chart"]["url"] is None


class TestHistoryLimits:
    def test_default_limit_bounds_the_page(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        for index in range(60):
            history_repository.save_history(original_question=f"q{index}", answer="a")
        assert len(client.get("/api/history").get_json()["history"]) == 50

    def test_explicit_limit_is_honoured(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        for index in range(10):
            history_repository.save_history(original_question=f"q{index}", answer="a")
        assert len(client.get("/api/history?limit=3").get_json()["history"]) == 3

    @pytest.mark.parametrize("raw", ["0", "-1", "101", "9999", "abc", "1.5", "50;DROP", "  "])
    def test_invalid_limits_are_rejected(self, app_factory, raw):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        assert client.get(f"/api/history?limit={raw}").status_code == 400

    def test_unknown_query_parameters_are_rejected(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        assert client.get("/api/history?bogus=1").status_code == 400


class TestHistoryRoutesDoNoAnalytics:
    def test_history_route_never_calls_the_processor(self, app_factory):
        def forbidden(*_a, **_k):
            raise AssertionError("history must not run analytics")

        client = app_factory(forbidden).test_client()
        assert client.get("/api/history").status_code == 200
        assert client.delete("/api/history").status_code == 200

    def test_history_route_never_touches_groq_or_analytics_sql(self, app_factory, monkeypatch):
        import database
        import llm

        def forbidden(*_a, **_k):
            raise AssertionError("history must not query or call the model")

        monkeypatch.setattr(database, "run_query", forbidden)
        monkeypatch.setattr(database, "execute_query", forbidden)
        monkeypatch.setattr(llm, "generate_sql", forbidden)
        monkeypatch.setattr(llm, "generate_answer", forbidden)

        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(original_question="q", answer="a")
        assert client.get("/api/history").status_code == 200

    def test_storage_failure_returns_a_safe_error(self, app_factory, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("disk on fire at C:\\secret")

        monkeypatch.setattr(history_repository, "list_history", boom)
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        response = client.get("/api/history")
        body = response.get_data(as_text=True)
        assert response.status_code == 503
        assert "C:\\" not in body and "disk on fire" not in body


class TestClearHistoryApi:
    def test_delete_clears_records_only(self, app_factory):
        chart = app_factory.charts_dir / "keep.png"
        chart.write_bytes(PNG_SIGNATURE)
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        history_repository.save_history(original_question="q", answer="a")

        response = client.delete("/api/history")
        assert response.get_json() == {"ok": True, "deleted": 1}
        assert client.get("/api/history").get_json()["history"] == []
        assert chart.exists(), "charts must survive a history delete"

    def test_delete_does_not_remove_the_analytics_database(self, app_factory):
        existed = Path(DATABASE_NAME).exists()
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        client.delete("/api/history")
        assert Path(DATABASE_NAME).exists() == existed

    def test_delete_on_empty_history_is_safe(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        assert client.delete("/api/history").get_json() == {"ok": True, "deleted": 0}

    def test_delete_failure_returns_a_safe_error(self, app_factory, monkeypatch):
        def boom():
            raise RuntimeError("boom at C:\\secret")

        monkeypatch.setattr(history_repository, "clear_history", boom)
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        response = client.delete("/api/history")
        assert response.status_code == 503
        assert "C:\\" not in response.get_data(as_text=True)


class TestQueryPersistence:
    """Exactly one persisted conversation turn per completed analytics response."""

    def test_successful_query_saves_one_user_and_assistant_turn(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        payload = ask(client).get_json()

        assert payload["conversation_saved"] is True
        conversation = load_saved_conversation(client, payload["conversation_id"])
        assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
        assert conversation["messages"][0]["content"] == "What is the win rate by region?"
        assert conversation["messages"][1]["content"] == "NA has the highest win rate."
        assert conversation["messages"][1]["meta"]["success"] is True
        assert client.get("/api/conversations").get_json()["conversations"][0]["message_count"] == 2
        # Browser queries do not create a second legacy flat-history record.
        assert history_repository.list_history() == []

    def test_refusal_saves_one_conversation_turn(self, app_factory):
        response = make_response(answer="This question cannot be answered.", refused=True, result=None)
        client = app_factory(lambda *_a, **_k: response).test_client()
        payload = ask(client).get_json()

        conversation = load_saved_conversation(client, payload["conversation_id"])
        assistant = conversation["messages"][-1]
        assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
        assert assistant["meta"]["refused"] is True
        assert assistant["row_count"] is None
        assert assistant["chart"]["url"] is None

    def test_empty_result_saves_row_metadata_but_not_result_rows(self, app_factory):
        empty = make_result(pd.DataFrame(columns=["region", "deals"]))
        client = app_factory(lambda *_a, **_k: make_response(result=empty)).test_client()
        payload = ask(client).get_json()

        assistant = load_saved_conversation(client, payload["conversation_id"])["messages"][-1]
        assert assistant["row_count"] == 0
        assert "columns" not in assistant
        assert "rows" not in assistant

    def test_answer_fallback_saves_one_conversation_turn(self, app_factory):
        response = make_response(
            answer=None,
            answer_fallback_used=True,
            answer_error=RuntimeError("provider down"),
        )
        client = app_factory(lambda *_a, **_k: response).test_client()
        payload = ask(client).get_json()

        assistant = load_saved_conversation(client, payload["conversation_id"])["messages"][-1]
        assert assistant["content"] == "The answer could not be generated. The returned data is shown below."
        assert assistant["meta"]["answer_fallback_used"] is True
        assert assistant["meta"]["success"] is False

    def test_chart_failure_still_saves_a_safe_conversation_message(self, app_factory):
        response = make_response(chart_requested=True, chart_error="nothing to chart")
        client = app_factory(lambda *_a, **_k: response).test_client()
        payload = ask(client).get_json()

        assistant = load_saved_conversation(client, payload["conversation_id"])["messages"][-1]
        assert assistant["chart"] == {
            "requested": True,
            "type": None,
            "url": None,
            "note": None,
            "available": False,
        }
        assert "error_code" not in assistant

    def test_chart_success_exposes_only_a_safe_historic_chart_url(self, app_factory):
        chart = app_factory.charts_dir / "generated.png"
        chart.write_bytes(PNG_SIGNATURE)
        response = make_response(chart_requested=True, chart_path=chart, chart_type=ChartType.BAR)
        client = app_factory(lambda *_a, **_k: response).test_client()
        payload = ask(client).get_json()

        detail_response = client.get(f"/api/conversations/{payload['conversation_id']}")
        assistant = detail_response.get_json()["conversation"]["messages"][-1]
        assert assistant["chart"]["available"] is True
        assert assistant["chart"]["url"] == "/charts/generated.png"
        assert str(chart.parent) not in detail_response.get_data(as_text=True)

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": ""},
            {"question": "   "},
            {"question": 42},
            {"question": "valid", "chart_type": "area"},
            {"question": "valid", "unexpected": True},
            {"question": "x" * 2001},
        ],
    )
    def test_validation_failures_save_nothing(self, app_factory, payload):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        assert client.post("/api/query", json=payload).status_code == 400
        assert client.get("/api/conversations").get_json() == {"ok": True, "conversations": []}

    def test_processor_exception_saves_nothing(self, app_factory):
        def boom(*_a, **_k):
            raise RuntimeError("model unavailable")

        client = app_factory(boom).test_client()
        assert ask(client).status_code == 503
        assert client.get("/api/conversations").get_json() == {"ok": True, "conversations": []}

    def test_sql_refusal_saves_nothing(self, app_factory):
        def boom(*_a, **_k):
            raise SqlValidationError("refused")

        client = app_factory(boom).test_client()
        assert ask(client).status_code == 400
        assert client.get("/api/conversations").get_json() == {"ok": True, "conversations": []}

    def test_non_analytics_routes_save_nothing(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        client.get("/")
        client.get("/api/status")
        client.get("/api/history")
        client.get("/api/conversations")
        client.get("/static/js/app.js")
        assert client.get("/api/conversations").get_json() == {"ok": True, "conversations": []}

    def test_conversation_transcript_is_not_part_of_the_query_payload(self, app_factory):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        payload = ask(client).get_json()
        assert "history" not in payload
        assert "messages" not in payload


class TestPersistenceFailureIsNonFatal:
    def test_analysis_still_succeeds(self, app_factory, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("conversation disk failure")

        monkeypatch.setattr(history_repository, "save_conversation_turn", boom)
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        response = ask(client)
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["ok"] is True
        assert payload["conversation_saved"] is False
        assert payload["conversation_id"] is None
        assert payload["answer"] == "NA has the highest win rate."
        assert "conversation disk failure" not in response.get_data(as_text=True)
        assert client.get("/api/conversations").get_json() == {"ok": True, "conversations": []}

    def test_no_second_execution_or_extra_model_call(self, app_factory, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("conversation failure")

        monkeypatch.setattr(history_repository, "save_conversation_turn", boom)
        calls = []

        def processor(*args, **kwargs):
            calls.append(args)
            return make_response()

        client = app_factory(processor).test_client()
        assert ask(client).status_code == 200
        # One processor call is one SQL execution and one answer generation.
        assert len(calls) == 1


class TestChartHttpIntegration:
    """A real PNG travels from chart.py through the API to an HTTP response."""

    @pytest.fixture
    def real_chart_response(self, app_factory, monkeypatch):
        monkeypatch.setattr("chart.CHARTS_DIR", app_factory.charts_dir)
        result = make_result()
        path, chart_type, note = create_chart("Deals by region", result, chart_type=ChartType.BAR)
        return app_factory, make_response(
            result=result, chart_requested=True, chart_path=path,
            chart_type=chart_type, chart_note=note,
        ), path

    def test_generated_chart_is_served_over_http(self, real_chart_response):
        factory, response, path = real_chart_response
        client = factory(lambda *_a, **_k: response).test_client()

        payload = ask(client).get_json()
        url = payload["chart"]["url"]
        assert url == f"/charts/{path.name}"
        assert payload["chart"]["type"] == "bar"

        image = client.get(url)
        assert image.status_code == 200
        assert image.headers["Content-Type"] == "image/png"
        body = image.get_data()
        assert body[:8] == PNG_SIGNATURE
        assert len(body) == path.stat().st_size

    def test_chart_url_exposes_no_absolute_path(self, real_chart_response):
        factory, response, path = real_chart_response
        client = factory(lambda *_a, **_k: response).test_client()
        body = ask(client).get_data(as_text=True)
        assert str(path.parent) not in body
        assert "C:\\" not in body

    def test_saved_conversation_message_serves_the_same_chart_without_regenerating(
        self, real_chart_response
    ):
        factory, response, path = real_chart_response
        client = factory(lambda *_a, **_k: response).test_client()
        query_payload = ask(client).get_json()

        # Reading a historic conversation message must not create another PNG.
        before = sorted(p.name for p in factory.charts_dir.glob("*.png"))
        conversation = load_saved_conversation(client, query_payload["conversation_id"])
        after = sorted(p.name for p in factory.charts_dir.glob("*.png"))
        assistant = conversation["messages"][-1]

        assert before == after
        assert assistant["chart"]["available"] is True
        assert client.get(assistant["chart"]["url"]).status_code == 200

    @pytest.mark.parametrize(
        "bad",
        ["../config.py", "..%2Fconfig.py", "sub/dir.png", "notes.txt", "missing.png",
         "%2e%2e%2fapp.py", ".env"],
    )
    def test_traversal_and_non_png_requests_are_rejected(self, app_factory, bad):
        client = app_factory(lambda *_a, **_k: make_response()).test_client()
        assert client.get(f"/charts/{bad}").status_code == 404
