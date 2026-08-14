"""Local Flask adapter for the Mini Local Analytics Assistant.

The browser UI never implements analytics logic itself.  It validates a small
JSON request, delegates one effective question to ``analytics_service``, and
serializes only the safe portions of that response.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import date, datetime
from decimal import Decimal
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Optional

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from analytics_service import AnalysisResponse, adapt_question_for_chart, process_question
from config import (
    CHARTS_DIR,
    GROQ_API_KEY,
    HISTORY_DEFAULT_LIMIT,
    HISTORY_MAX_LIMIT,
    MODEL_NAME,
    TABLE_NAME,
    WEB_HOST,
    WEB_PORT,
    WEB_URL,
    ConfigurationError,
)
from database import (
    DatabaseConnectionError,
    DatabaseError,
    SqlExecutionError,
    SqlValidationError,
    close_connection,
    initialize_database,
)
import history_repository
from llm import normalize_question
from logging_config import setup_logging, setup_web_logging

# Named explicitly rather than using __name__: running this file directly makes
# __name__ equal "__main__", and the dedicated web log handler is bound to the
# "web_app" logger. A fixed name keeps the log complete however it is started.
logger = logging.getLogger("web_app")

_ALLOWED_CHART_TYPES = frozenset({"auto", "bar", "line", "pie", "scatter"})
MAX_WEB_QUESTION_CHARS = 2000

# Bind address and port. Loopback and 8000 by default; overridable through
# WEB_HOST / WEB_PORT so a container can bind 0.0.0.0 without changing local use.
SERVER_HOST = WEB_HOST
SERVER_PORT = WEB_PORT
# Small on purpose: DuckDB serialises behind one lock, so extra threads buy
# nothing for queries. They exist so a slow or idle client cannot stop the
# server answering everyone else.
SERVER_THREADS = 4
_CHART_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.png$", re.IGNORECASE)
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


class ApiRequestError(ValueError):
    """A malformed browser request that can safely be explained to the user."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _is_missing_scalar(value: Any) -> bool:
    """Return whether a scalar is missing without depending directly on NumPy."""
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    try:
        return bool(missing)
    except (TypeError, ValueError):
        return False


def _native_scalar(value: Any) -> Any:
    """Unwrap pandas/NumPy-backed scalar values through their standard API."""
    item = getattr(value, "item", None)
    if not callable(item):
        return value
    try:
        return item()
    except (AttributeError, TypeError, ValueError):
        return value


def serialize_value(value: Any) -> Any:
    """Convert a DataFrame cell to a JSON-safe value without changing its meaning."""
    if _is_missing_scalar(value):
        return None
    value = _native_scalar(value)
    if _is_missing_scalar(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        # A string retains Decimal precision that a JSON floating point value may lose.
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        return numeric_value if math.isfinite(numeric_value) else None
    if isinstance(value, str):
        return value

    # DuckDB values beyond the expected scalar set are preserved as readable text.
    return str(value)


def _serialize_result(response: AnalysisResponse) -> tuple[list[str], list[list[Any]], int, bool, Optional[int]]:
    """Return the browser-safe representation of the existing QueryResult."""
    if response.result is None:
        return [], [], 0, False, None

    result = response.result
    rows = [
        [serialize_value(value) for value in row]
        for row in result.frame.itertuples(index=False, name=None)
    ]
    return result.columns, rows, result.row_count, result.truncated, result.max_rows


def _chart_url(path: Optional[Path]) -> Optional[str]:
    """Create a relative chart URL without exposing the local path."""
    if path is None:
        return None
    return f"/charts/{path.name}"


def public_response(response: AnalysisResponse) -> dict[str, Any]:
    """Build the explicit public API contract without serializing QueryResult.sql."""
    columns, rows, row_count, truncated, max_rows = _serialize_result(response)
    chart_note = response.chart_note
    if response.chart_error:
        chart_note = "The chart could not be generated for this result."

    return {
        "ok": True,
        # Top level as well as in meta: the browser needs it before deciding how
        # to render, and a refusal is a successful request with nothing to show.
        "refused": response.refused,
        "question": response.original_question,
        "answer": response.answer
        or "The answer could not be generated. The returned data is shown below.",
        "columns": columns,
        "rows": rows,
        "row_count": row_count,
        "truncated": truncated,
        "max_rows": max_rows,
        "chart": {
            "requested": response.chart_requested,
            "url": _chart_url(response.chart_path),
            "type": response.chart_type.value if response.chart_type else None,
            "note": chart_note,
        },
        "meta": {
            "answer_fallback_used": response.answer_fallback_used,
            "elapsed_seconds": round(response.elapsed_seconds, 3),
            "refused": response.refused,
        },
    }


def _error_response(code: str, message: str, status: int) -> tuple[Response, int]:
    """Return the uniform, non-diagnostic JSON error contract."""
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), status


def save_completed_history(response: AnalysisResponse) -> bool:
    """Persist one completed analytics response. Never raises.

    Called only after the analytics workflow has finished, using values already
    present on the response. It performs no query, no model call, and no chart
    work, so a storage failure cannot cost the user their answer.
    """
    try:
        history_repository.save_history(
            original_question=response.original_question,
            answer=response.answer,
            row_count=response.result.row_count if response.result is not None else None,
            truncated=bool(response.result.truncated) if response.result is not None else False,
            max_rows=response.result.max_rows if response.result is not None else None,
            chart_requested=response.chart_requested,
            chart_type=response.chart_type.value if response.chart_type else None,
            chart_filename=response.chart_path.name if response.chart_path else None,
            chart_note=response.chart_note,
            answer_fallback_used=response.answer_fallback_used,
            refused=response.refused,
            success=not response.answer_fallback_used,
            elapsed_seconds=response.elapsed_seconds,
            error_code="chart_failed" if response.chart_error else None,
        )
        return True
    except Exception as exc:
        # A history failure is reported as metadata, never as a request failure.
        logger.error("History persistence failed for a completed analysis: %s", str(exc))
        return False


def _history_chart_payload(record: history_repository.HistoryRecord, charts_dir: Path) -> dict[str, Any]:
    """Build the chart section of a history entry, verifying the file still exists."""
    filename = history_repository.safe_chart_filename(record.chart_filename)
    available = bool(filename) and _is_safe_chart_filename(filename, charts_dir)
    return {
        "requested": record.chart_requested,
        "type": record.chart_type,
        "url": f"/charts/{filename}" if available else None,
        "note": record.chart_note,
        "available": available,
    }


def public_history_entry(
    record: history_repository.HistoryRecord, charts_dir: Path
) -> dict[str, Any]:
    """Serialize one history record for the browser.

    Built field by field from the record, which carries no SQL, schema, path, or
    exception text to leak.
    """
    return {
        "id": record.history_id,
        "created_at": record.created_at.isoformat(),
        "question": record.original_question,
        "answer": record.answer,
        "row_count": record.row_count,
        "truncated": record.truncated,
        "chart": _history_chart_payload(record, charts_dir),
        "meta": {
            "answer_fallback_used": record.answer_fallback_used,
            "refused": record.refused,
            "success": record.success,
            "elapsed_seconds": record.elapsed_seconds,
        },
    }


def _parse_history_limit(raw_limit: Optional[str]) -> int:
    """Validate the optional ?limit= parameter against the server-side maximum."""
    if raw_limit is None or raw_limit == "":
        return HISTORY_DEFAULT_LIMIT
    if not re.fullmatch(r"[0-9]{1,4}", raw_limit):
        raise ApiRequestError("invalid_limit", "The limit must be a whole number.")
    limit = int(raw_limit)
    if limit < 1 or limit > HISTORY_MAX_LIMIT:
        raise ApiRequestError(
            "invalid_limit",
            f"The limit must be between 1 and {HISTORY_MAX_LIMIT}.",
        )
    return limit


def _parse_query_payload(payload: Any) -> tuple[str, bool, str]:
    """Validate browser JSON and return its raw question and chart preferences."""
    if not isinstance(payload, dict):
        raise ApiRequestError("invalid_payload", "Send a JSON object with a question.")
    allowed_fields = {"question", "chart_requested", "chart_type"}
    if set(payload) - allowed_fields:
        raise ApiRequestError("invalid_payload", "Send only question and chart preferences.")

    if "question" not in payload:
        raise ApiRequestError("missing_question", "Enter a business question to continue.")
    question = payload["question"]
    if not isinstance(question, str):
        raise ApiRequestError("invalid_question", "The question must be text.")
    normalized_question = normalize_question(question)
    if not normalized_question:
        raise ApiRequestError("blank_question", "Enter a business question to continue.")
    if (
        len(question) > MAX_WEB_QUESTION_CHARS
        or len(normalized_question) > MAX_WEB_QUESTION_CHARS
    ):
        raise ApiRequestError(
            "question_too_long",
            "Questions must be 2,000 characters or fewer.",
        )

    chart_requested = payload.get("chart_requested", False)
    if not isinstance(chart_requested, bool):
        raise ApiRequestError("invalid_chart_request", "Chart selection must be true or false.")

    chart_type = payload.get("chart_type", "auto")
    if not isinstance(chart_type, str) or chart_type not in _ALLOWED_CHART_TYPES:
        raise ApiRequestError(
            "invalid_chart_type",
            "Choose auto, bar, line, pie, or scatter for the chart type.",
        )
    return question, chart_requested, chart_type


def _is_safe_chart_filename(filename: str, charts_dir: Path) -> bool:
    """Reject traversal and any chart path outside the configured directory."""
    if not isinstance(filename, str) or not _CHART_FILENAME_RE.fullmatch(filename):
        return False
    root = charts_dir.resolve()
    candidate = (root / filename).resolve()
    return candidate.parent == root and candidate.is_file()


def create_app(
    *,
    process_question_func: Optional[Callable[..., AnalysisResponse]] = None,
    history_saver: Optional[Callable[[AnalysisResponse], bool]] = None,
    initialize_database_at_start: bool = False,
) -> Flask:
    """Create the local Flask application without initializing per request."""
    app = Flask(__name__)
    app.config["CHARTS_DIR"] = Path(CHARTS_DIR)
    processor = process_question_func or process_question
    saver = history_saver or save_completed_history
    processing_lock = RLock()
    status = {
        "database": "Not initialized",
        "table": TABLE_NAME,
        "analytics_engine": "DuckDB",
        "model": MODEL_NAME,
        "web_mode": "Local",
        "api": "Configured" if GROQ_API_KEY else "Missing",
    }

    if initialize_database_at_start:
        initialize_database()
        status["database"] = "Ready"
        # History storage is independent of the analytics database, so a
        # history problem must not stop the analytics service from starting.
        try:
            history_repository.initialize_history_database()
        except Exception:
            logger.exception("History storage could not be initialized.")

    app.extensions["analytics_processor"] = processor
    app.extensions["analytics_processing_lock"] = processing_lock
    app.extensions["analytics_status"] = status
    app.extensions["history_saver"] = saver

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers["Content-Security-Policy"] = _CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/status")
    def api_status() -> Response:
        return jsonify({"ok": True, "status": app.extensions["analytics_status"]})

    @app.post("/api/query")
    def api_query() -> tuple[Response, int] | Response:
        if not request.is_json:
            return _error_response("invalid_payload", "Send a JSON request body.", 400)

        payload = request.get_json(silent=True)
        try:
            question, chart_requested, chart_type = _parse_query_payload(payload)
            effective_question = adapt_question_for_chart(
                question,
                chart_requested=chart_requested,
                chart_type=chart_type,
            )
        except ApiRequestError as exc:
            return _error_response(exc.code, exc.message, 400)
        except ValueError:
            return _error_response("invalid_question", "Enter a business question to continue.", 400)

        logger.info("Web analytics request started (chart_requested=%s).", chart_requested)
        try:
            with app.extensions["analytics_processing_lock"]:
                response = app.extensions["analytics_processor"](
                    effective_question,
                    original_question=question,
                )
        except SqlValidationError:
            logger.warning("Web analytics request was refused by SQL validation.")
            return _error_response("sql_refused", "The requested analysis could not be run safely.", 400)
        except ConfigurationError:
            logger.error("Web analytics request could not run because configuration is missing.")
            return _error_response(
                "configuration_error",
                "The analytics service is not configured for requests.",
                503,
            )
        except DatabaseConnectionError:
            logger.error("Web analytics request failed because the database is unavailable.")
            return _error_response("database_unavailable", "The local analytics database is unavailable.", 503)
        except SqlExecutionError:
            logger.error("Web analytics request failed during SQL execution.")
            return _error_response("query_failed", "The requested analysis could not be completed.", 422)
        except DatabaseError:
            logger.error("Web analytics request failed in the database layer.")
            return _error_response("database_error", "The local analytics service is unavailable.", 503)
        except RuntimeError:
            logger.error("Web analytics request failed while contacting the language model.")
            return _error_response(
                "language_model_unavailable",
                "The language model is unavailable. Please try again later.",
                503,
            )
        except Exception:
            logger.exception("Unexpected web analytics request failure.")
            return _error_response("internal_error", "An unexpected error occurred.", 500)

        logger.info(
            "Web analytics request completed in %.3fs (rows=%s, chart=%s, refused=%s, fallback=%s).",
            response.elapsed_seconds,
            response.result.row_count if response.result is not None else 0,
            response.chart_path is not None,
            response.refused,
            response.answer_fallback_used,
        )

        # Exactly one history record per completed analytics response, written
        # from the finished response only.
        history_saved = app.extensions["history_saver"](response)
        if not history_saved:
            logger.warning("History was not persisted for a completed analytics response.")

        payload = public_response(response)
        payload["history_saved"] = history_saved
        return jsonify(payload)

    @app.get("/api/history")
    def api_history() -> tuple[Response, int] | Response:
        """Return saved history. Performs no analytics query and no model call."""
        try:
            limit = _parse_history_limit(request.args.get("limit"))
        except ApiRequestError as exc:
            return _error_response(exc.code, exc.message, 400)
        if set(request.args) - {"limit"}:
            return _error_response("invalid_query", "Only a limit parameter is supported.", 400)

        charts_dir = Path(app.config["CHARTS_DIR"])
        try:
            records = history_repository.list_history(limit)
        except Exception:
            logger.exception("Unable to read saved history.")
            return _error_response("history_unavailable", "Saved history is unavailable.", 503)

        return jsonify(
            {
                "ok": True,
                "history": [public_history_entry(record, charts_dir) for record in records],
            }
        )

    @app.delete("/api/history")
    def api_clear_history() -> tuple[Response, int] | Response:
        """Delete saved history rows only; analytics data and charts are untouched."""
        try:
            deleted = history_repository.clear_history()
        except Exception:
            logger.exception("Unable to clear saved history.")
            return _error_response("history_unavailable", "Saved history could not be cleared.", 503)

        logger.info("Saved history cleared through the web interface.")
        return jsonify({"ok": True, "deleted": deleted})

    @app.get("/charts/<filename>")
    def serve_chart(filename: str) -> tuple[Response, int] | Response:
        charts_dir = Path(app.config["CHARTS_DIR"])
        if not _is_safe_chart_filename(filename, charts_dir):
            # The rejected name is deliberately not echoed back to the browser.
            logger.warning("Rejected a chart request for an unsafe or missing filename.")
            return _error_response("chart_not_found", "The requested chart was not found.", 404)
        return send_from_directory(charts_dir.resolve(), filename, mimetype="image/png")

    return app


def serve_forever(app: Flask) -> None:
    """Run the application on a server that a stalled connection cannot wedge.

    Werkzeug's development server sets no socket timeout, so a client that
    opens a connection and never sends a request line blocks the handler
    indefinitely. With ``threaded=False`` that blocked one handler *and* every
    other request behind it: the port kept accepting connections while nothing
    was ever answered. On a public address that is not hypothetical - port
    scanners do exactly this, continuously - and it took the app down within
    minutes of being exposed.

    Waitress reads requests through a select loop rather than a thread per
    connection, so an idle client costs no worker at all, and ``channel_timeout``
    closes it rather than waiting forever. Falling back to the development
    server keeps this importable if waitress is not installed, which only
    affects local use.
    """
    try:
        from waitress import serve
    except ImportError:
        logger.warning(
            "waitress is not installed; using the development server, which a "
            "stalled connection can block. Do not expose this to a network."
        )
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
        return

    logger.info("Serving with waitress (threads=%s).", SERVER_THREADS)
    serve(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        threads=SERVER_THREADS,
        # Reap a client that connects and then says nothing. This is the
        # setting that makes the app survive being on the open internet.
        channel_timeout=30,
        # The app is the only thing on this port; no proxy sets these headers.
        ident=None,
    )


def main() -> int:
    """Initialize the local database once and start a loopback-only server."""
    setup_logging()
    web_logging_ready = setup_web_logging()
    logger.info(
        "Web application starting (host=%s, port=%s, debug=False, dedicated_log=%s).",
        SERVER_HOST,
        SERVER_PORT,
        web_logging_ready,
    )
    try:
        app = create_app(initialize_database_at_start=True)
    except DatabaseError as exc:
        logger.critical("Web database initialization failed: %s", exc)
        print("Database initialization failed. See logs/app.log for details.")
        return 1

    # Werkzeug announces its address through the werkzeug logger at INFO, but the
    # console handler runs at WARNING, so that banner only reaches logs/app.log.
    # The address is printed here instead: startup must state where to connect
    # regardless of how logging is configured. Bound to the same constants the
    # server uses, so the printed URL cannot drift from the real one.
    # WEB_URL, not the bind address: a container binds 0.0.0.0, which is not an
    # address a browser can open.
    print("Mini Local Analytics Assistant", flush=True)
    print(f"Open in browser: {WEB_URL}", flush=True)
    print("Press Ctrl+C to stop the server.", flush=True)

    try:
        serve_forever(app)
    except KeyboardInterrupt:
        logger.info("Web application stopped by the user.")
    finally:
        close_connection()
        logger.info("Web application finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
