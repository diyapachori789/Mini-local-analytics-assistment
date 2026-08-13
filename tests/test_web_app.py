"""Offline Phase 7 coverage for the Flask adapter and its public API.

The service is injected into each application instance.  Consequently these
tests exercise no real Groq, DuckDB, or matplotlib work.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

import web_app
from analytics_service import AnalysisResponse
from chart import ChartType
from config import ConfigurationError
from database import QueryResult, SqlValidationError


def make_result(
    frame: pd.DataFrame | None = None,
    *,
    truncated: bool = False,
    max_rows: int | None = 1000,
) -> QueryResult:
    """Return a result containing SQL that must never reach browser JSON."""
    actual_frame = frame if frame is not None else pd.DataFrame(
        {"region": ["NA", "EMEA"], "win_rate": [25.0, 24.64]}
    )
    return QueryResult(
        frame=actual_frame,
        sql="SELECT private_generated_sql FROM opportunities;",
        row_count=len(actual_frame),
        truncated=truncated,
        max_rows=max_rows,
    )


def make_response(
    *,
    question: str = "What is the win rate by region?",
    result: QueryResult | None = None,
    answer: str | None = "NA has the highest win rate.",
    chart_requested: bool = False,
    chart_path: Path | None = None,
    chart_type: ChartType | None = None,
    chart_note: str | None = None,
    answer_fallback_used: bool = False,
    answer_error: Exception | None = None,
    chart_error: str | None = None,
    refused: bool = False,
) -> AnalysisResponse:
    """Build a realistic service outcome without any analytics side effects."""
    return AnalysisResponse(
        original_question=question,
        effective_question=question,
        analytical_question=question,
        answer=answer,
        result=result if result is not None else make_result(),
        chart_requested=chart_requested,
        chart_path=chart_path,
        chart_type=chart_type,
        chart_note=chart_note,
        answer_fallback_used=answer_fallback_used,
        answer_error=answer_error,
        chart_error=chart_error,
        refused=refused,
        elapsed_seconds=0.0123,
    )


@pytest.fixture
def app_factory(monkeypatch, tmp_path):
    """Create isolated Flask apps that serve charts from a pytest directory."""
    monkeypatch.setattr(web_app, "CHARTS_DIR", tmp_path, raising=False)

    def build(processor):
        application = web_app.create_app(
            process_question_func=processor,
            initialize_database_at_start=False,
        )
        application.config.update(TESTING=True, CHARTS_DIR=tmp_path)
        return application

    return build


class TestValueSerialization:
    """Browser JSON preserves database values without business recalculation."""

    def test_primitives_and_missing_values_are_json_safe(self):
        assert web_app.serialize_value("NA") == "NA"
        assert web_app.serialize_value(12) == 12
        assert web_app.serialize_value(12.5) == 12.5
        assert web_app.serialize_value(True) is True
        assert web_app.serialize_value(None) is None
        assert web_app.serialize_value(pd.NA) is None
        assert web_app.serialize_value(float("nan")) is None
        assert web_app.serialize_value(pd.NaT) is None

    def test_dates_timestamps_and_decimal_keep_their_meaning(self):
        moment = datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)

        assert web_app.serialize_value(date(2025, 2, 3)) == "2025-02-03"
        assert web_app.serialize_value(moment) == moment.isoformat()
        assert web_app.serialize_value(pd.Timestamp(moment)) == moment.isoformat()
        assert web_app.serialize_value(Decimal("100.250")) == "100.250"

    def test_pandas_backed_numeric_scalars_become_native_json_values(self):
        integer = pd.Series([7], dtype="int64").iloc[0]
        decimal = pd.Series([2.75], dtype="float64").iloc[0]
        boolean = pd.Series([True], dtype="bool").iloc[0]

        assert web_app.serialize_value(integer) == 7
        assert isinstance(web_app.serialize_value(integer), int)
        assert web_app.serialize_value(decimal) == 2.75
        assert isinstance(web_app.serialize_value(decimal), float)
        assert web_app.serialize_value(boolean) is True

    def test_serializer_has_no_direct_numpy_dependency(self):
        source = Path(web_app.__file__).read_text(encoding="utf-8")

        assert "import numpy" not in source
        assert "from numpy" not in source


class TestApplicationShell:
    def test_index_and_static_assets_are_served(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        client = application.test_client()

        index = client.get("/")
        css = client.get("/static/css/app.css")
        javascript = client.get("/static/js/app.js")

        assert index.status_code == 200
        assert b"Mini Local Analytics" in index.data
        assert css.status_code == 200
        assert b"--" in css.data
        assert javascript.status_code == 200
        assert b"fetch" in javascript.data
        assert b"private_generated_sql" not in index.data
        assert b"QueryResult.sql" not in index.data

    def test_question_input_limit_matches_the_server_limit(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        source = application.test_client().get("/").get_data(as_text=True)

        assert f'maxlength="{web_app.MAX_WEB_QUESTION_CHARS}"' in source

    def test_status_exposes_only_safe_runtime_metadata(self, app_factory, tmp_path):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        payload = application.test_client().get("/api/status").get_json()
        rendered = json.dumps(payload)

        assert payload["ok"] is True
        assert "opportunities" in rendered
        assert "DuckDB" in rendered
        assert str(tmp_path) not in rendered
        assert "GROQ_API_KEY" not in rendered
        assert "schema" not in rendered.lower()

    def test_security_headers_are_attached_to_html_and_json(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        client = application.test_client()

        for response in (client.get("/"), client.get("/api/status")):
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert (
                response.headers.get("X-Frame-Options") == "DENY"
                or "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            )


class TestQueryApiValidation:
    def test_missing_or_malformed_json_is_rejected_without_calling_service(self, app_factory):
        calls = 0

        def forbidden_processor(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("Invalid payloads must not reach the service")

        client = app_factory(forbidden_processor).test_client()
        responses = (
            client.post("/api/query"),
            client.post("/api/query", data="not json", content_type="text/plain"),
            client.post("/api/query", json=[]),
        )

        for response in responses:
            payload = response.get_json()
            assert response.status_code == 400
            assert payload["ok"] is False
            assert payload["error"]["code"]
            assert payload["error"]["message"]
        assert calls == 0

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"question": None},
            {"question": 42},
            {"question": "   "},
            {"question": "\ufeff\u200b\u00a0"},
            {"question": "Deals by region", "chart_requested": "yes"},
            {"question": "Deals by region", "chart_type": "area"},
            {"question": "Deals by region", "history": [{"question": "do not send"}]},
        ],
    )
    def test_invalid_payload_fields_are_rejected_before_service(self, app_factory, payload):
        def forbidden_processor(*_args, **_kwargs):
            raise AssertionError("Invalid payloads must not reach the service")

        response = app_factory(forbidden_processor).test_client().post("/api/query", json=payload)
        body = response.get_json()

        assert response.status_code == 400
        assert body["ok"] is False
        assert body["error"]["code"]

    def test_question_at_server_limit_is_accepted(self, app_factory):
        question = "x" * web_app.MAX_WEB_QUESTION_CHARS
        seen: list[tuple[str, str | None]] = []

        def processor(effective_question, **kwargs):
            seen.append((effective_question, kwargs.get("original_question")))
            return make_response(question=question)

        response = app_factory(processor).test_client().post(
            "/api/query", json={"question": question}
        )

        assert response.status_code == 200
        assert seen == [(question, question)]

    def test_oversized_question_is_rejected_before_adaptation_or_processing(
        self, app_factory, monkeypatch
    ):
        question = "x" * (web_app.MAX_WEB_QUESTION_CHARS + 1)
        adapter_calls = 0
        processor_calls = 0

        def forbidden_adapter(*_args, **_kwargs):
            nonlocal adapter_calls
            adapter_calls += 1
            raise AssertionError("Oversized input must not be adapted")

        def forbidden_processor(*_args, **_kwargs):
            nonlocal processor_calls
            processor_calls += 1
            raise AssertionError("Oversized input must not reach analytics processing")

        monkeypatch.setattr(web_app, "adapt_question_for_chart", forbidden_adapter)
        response = app_factory(forbidden_processor).test_client().post(
            "/api/query", json={"question": question}
        )
        payload = response.get_json()
        rendered = json.dumps(payload)

        assert response.status_code == 400
        assert payload == {
            "ok": False,
            "error": {
                "code": "question_too_long",
                "message": "Questions must be 2,000 characters or fewer.",
            },
        }
        assert question not in rendered
        assert adapter_calls == 0
        assert processor_calls == 0


class TestQueryApiSuccess:
    def test_toggle_adapts_question_and_returns_an_explicit_safe_result(self, app_factory):
        typed_question = "Total pipeline by region"
        response_object = make_response(question=typed_question)
        seen: dict[str, object] = {}

        def processor(*args, **kwargs):
            seen["effective_question"] = args[0]
            seen["original_question"] = kwargs.get("original_question")
            return response_object

        response = app_factory(processor).test_client().post(
            "/api/query",
            json={
                "question": typed_question,
                "chart_requested": True,
                "chart_type": "bar",
            },
        )
        payload = response.get_json()
        rendered = json.dumps(payload)

        assert response.status_code == 200
        assert seen["original_question"] == typed_question
        assert seen["effective_question"] == "Total pipeline by region and show it as a bar chart."
        assert payload["ok"] is True
        assert payload["question"] == typed_question
        assert payload["answer"] == "NA has the highest win rate."
        assert payload["columns"] == ["region", "win_rate"]
        assert payload["rows"] == [["NA", 25.0], ["EMEA", 24.64]]
        assert payload["row_count"] == 2
        assert payload["truncated"] is False
        assert payload["max_rows"] == 1000
        assert payload["chart"] == {
            "requested": False,
            "url": None,
            "type": None,
            "note": None,
        }
        assert payload["meta"]["answer_fallback_used"] is False
        assert payload["meta"]["elapsed_seconds"] == pytest.approx(0.012)
        assert "private_generated_sql" not in rendered
        assert "schema" not in rendered.lower()

    def test_typed_chart_instruction_wins_over_the_ui_preference(self, app_factory):
        typed_question = "Show pipeline by region as a pie chart."
        seen: list[str] = []

        def processor(*args, **_kwargs):
            seen.append(args[0])
            return make_response(question=typed_question)

        response = app_factory(processor).test_client().post(
            "/api/query",
            json={
                "question": typed_question,
                "chart_requested": True,
                "chart_type": "bar",
            },
        )

        assert response.status_code == 200
        assert seen == [typed_question]

    def test_chart_fallback_uses_creator_output_and_never_exposes_a_local_path(
        self, app_factory, tmp_path
    ):
        chart_file = tmp_path / "fallback.png"
        chart_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        response_object = make_response(
            chart_requested=True,
            chart_path=chart_file,
            chart_type=ChartType.LINE,
            chart_note="A line chart was used because the data is temporal.",
        )

        response = app_factory(lambda *_args, **_kwargs: response_object).test_client().post(
            "/api/query",
            json={"question": "Show monthly pipeline as a pie chart."},
        )
        payload = response.get_json()
        rendered = json.dumps(payload)

        assert response.status_code == 200
        assert payload["chart"] == {
            "requested": True,
            "url": "/charts/fallback.png",
            "type": "line",
            "note": "A line chart was used because the data is temporal.",
        }
        assert str(tmp_path) not in rendered

    def test_truncated_response_keeps_the_configured_cap_without_claiming_a_total(
        self, app_factory
    ):
        result = make_result(
            pd.DataFrame({"opportunity_id": range(3)}),
            truncated=True,
            max_rows=3,
        )
        response = app_factory(
            lambda *_args, **_kwargs: make_response(result=result)
        ).test_client().post("/api/query", json={"question": "List opportunities"})
        payload = response.get_json()

        assert payload["row_count"] == 3
        assert payload["truncated"] is True
        assert payload["max_rows"] == 3


class TestQueryApiErrors:
    def test_known_exception_returns_a_safe_message_without_fake_key_in_logs(
        self, app_factory, caplog
    ):
        fake_key = "gsk_phase7_not_for_browser_or_logs"

        def broken_processor(*_args, **_kwargs):
            raise ConfigurationError(f"Missing key: {fake_key}")

        with caplog.at_level("INFO"):
            response = app_factory(broken_processor).test_client().post(
                "/api/query", json={"question": "Deals by region"}
            )

        payload = response.get_json()
        rendered = json.dumps(payload)
        logged = "\n".join(record.getMessage() for record in caplog.records)
        assert response.status_code >= 400
        assert payload["ok"] is False
        assert payload["error"]["code"]
        assert fake_key not in rendered
        assert "ConfigurationError" not in rendered
        assert fake_key not in logged

    def test_sql_validation_error_has_no_sql_or_traceback(self, app_factory):
        def refused_processor(*_args, **_kwargs):
            raise SqlValidationError("Refused SELECT private_generated_sql;")

        response = app_factory(refused_processor).test_client().post(
            "/api/query", json={"question": "Unsafe request"}
        )
        payload = response.get_json()
        rendered = json.dumps(payload)

        assert response.status_code >= 400
        assert payload["ok"] is False
        assert "private_generated_sql" not in rendered
        assert "Traceback" not in rendered


class TestChartRoute:
    def test_serves_only_a_png_in_the_configured_chart_directory(self, app_factory, tmp_path):
        png = tmp_path / "allowed.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        client = app_factory(lambda *_args, **_kwargs: make_response()).test_client()

        response = client.get("/charts/allowed.png")

        assert response.status_code == 200
        assert response.mimetype == "image/png"
        assert response.data == png.read_bytes()

    @pytest.mark.parametrize(
        "path",
        ["../config.py", "%2e%2e%2fconfig.py", "allowed.txt", "nested/allowed.png"],
    )
    def test_rejects_path_traversal_and_non_png_chart_names(self, app_factory, path):
        client = app_factory(lambda *_args, **_kwargs: make_response()).test_client()

        response = client.get(f"/charts/{path}", follow_redirects=True)

        assert response.status_code in {400, 404}


class TestClientSourceSafety:
    def test_javascript_prevents_duplicate_submission_and_never_uses_html_injection(
        self, app_factory
    ):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        source = application.test_client().get("/static/js/app.js").get_data(as_text=True)

        assert "isSubmitting" in source
        assert "if (isSubmitting)" in source
        assert "innerHTML" not in source
        assert "eval(" not in source
        assert "new Function" not in source

    def test_client_submits_only_the_current_question_and_chart_preferences(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        source = application.test_client().get("/static/js/app.js").get_data(as_text=True)
        request_start = source.index("const requestBody")
        request_end = source.index("setSubmitting(true)", request_start)
        request_body_source = source[request_start:request_end]

        assert source.count('fetch("/api/query"') == 1
        assert "history" not in request_body_source
        assert "question" in request_body_source
        assert "chart_requested" in request_body_source
        assert "chart_type" in request_body_source

    def test_client_has_accurate_truncation_copy_and_no_generated_sql_interface(
        self, app_factory
    ):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        source = application.test_client().get("/static/js/app.js").get_data(as_text=True)

        assert "Additional rows were not fetched" in source
        assert "generated SQL" not in source
        assert "QueryResult.sql" not in source

    def test_client_status_logic_distinguishes_ready_warning_and_error(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        client = application.test_client()
        source = client.get("/static/js/app.js").get_data(as_text=True)
        css_source = client.get("/static/css/app.css").get_data(as_text=True)

        state_start = source.index("function determineStatusState")
        state_end = source.index("function setStatusDotState", state_start)
        state_helper = source[state_start:state_end]
        dot_start = state_end
        dot_end = source.index("function pickStatusValue", dot_start)
        dot_helper = source[dot_start:dot_end]
        load_start = source.index("async function loadStatus")
        load_end = source.index("async function readJson", load_start)
        load_status = source[load_start:load_end]
        unavailable_start = source.index("function renderUnavailableStatus")
        unavailable_end = state_start
        unavailable_status = source[unavailable_start:unavailable_end]

        # Only a ready database plus configured API is healthy. Recognized
        # dependency failures are warnings; invalid status values are errors.
        assert 'const validDatabase = database === "Ready" || database === "Not initialized";' in state_helper
        assert 'const validApi = api === "Configured" || api === "Missing";' in state_helper
        assert "if (!validDatabase || !validApi)" in state_helper
        assert 'database === "Ready" && api === "Configured"' in state_helper
        assert '"Not initialized"' in state_helper
        assert '"Missing"' in state_helper
        assert 'return "ready";' in state_helper
        assert 'return "warning";' in state_helper
        assert 'return "error";' in state_helper
        assert "renderUnavailableStatus();" in load_status
        assert 'setStatusDotState("error");' in unavailable_status

        # Every update removes stale status state before adding the new one.
        assert 'classList.remove("is-ready", "is-warning", "is-error")' in dot_helper
        assert "classList.add(`is-${state}`)" in dot_helper
        assert ".status-dot.is-ready" in css_source
        assert ".status-dot.is-warning" in css_source
        assert ".status-dot.is-error" in css_source

    def test_history_and_chart_labels_match_their_storage_lifetimes(self, app_factory):
        """Copy must describe persistent storage, not the old browser-local one."""
        application = app_factory(lambda *_args, **_kwargs: make_response())
        client = application.test_client()
        index_source = client.get("/").get_data(as_text=True)
        javascript_source = client.get("/static/js/app.js").get_data(as_text=True)

        assert "SAVED LOCALLY" in index_source
        assert "Saved on this machine and kept until you delete it." in index_source
        assert "Charts from your saved history, kept until you delete it." in index_source
        # The superseded browser-only wording must not survive anywhere.
        assert "BROWSER-LOCAL" not in index_source
        assert "CURRENT SESSION" not in index_source
        assert "stored in this browser" not in index_source
        assert "this browser session" not in index_source

        # History is fetched from the backend; localStorage keeps preferences only.
        assert "/api/history" in javascript_source
        assert "sessionCharts" not in javascript_source
        assert "STORAGE_KEYS.history" not in javascript_source

    def test_client_treats_backend_history_as_authoritative(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        javascript_source = application.test_client().get("/static/js/app.js").get_data(as_text=True)

        # Legacy browser history is discarded rather than merged.
        assert "LEGACY_HISTORY_KEY" in javascript_source
        assert "removeStoredValue(LEGACY_HISTORY_KEY)" in javascript_source
        # Preferences remain the only localStorage responsibility.
        assert "theme: \"mini-local-analytics.theme\"" in javascript_source
        assert "view: \"mini-local-analytics.view\"" in javascript_source
        assert "history: \"mini-local-analytics.history\"" not in javascript_source

    def test_clear_saved_history_is_confirmed_and_separate_from_clear_session(self, app_factory):
        application = app_factory(lambda *_args, **_kwargs: make_response())
        client = application.test_client()
        index_source = client.get("/").get_data(as_text=True)
        javascript_source = client.get("/static/js/app.js").get_data(as_text=True)

        assert "data-clear-history" in index_source
        assert "Clear saved history" in index_source
        assert "window.confirm(" in javascript_source
        assert 'method: "DELETE"' in javascript_source
        # Clear session must remain display-only.
        clear_session = javascript_source.split("function clearSession()")[1].split("async function")[0]
        assert "DELETE" not in clear_session
        assert "/api/history" not in clear_session
