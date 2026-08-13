"""Persistent query-history storage.

Every test runs against a temporary DuckDB file supplied by the autouse
``isolated_history_database`` fixture in conftest, so the production
``history.duckdb`` is never touched.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

import history_repository
from history_repository import (
    HistoryError,
    HistoryRecord,
    clear_history,
    initialize_history_database,
    list_history,
    safe_chart_filename,
    save_history,
)


def save_sample(question: str = "Total pipeline by region", **overrides) -> str:
    payload = {
        "original_question": question,
        "answer": "NA leads the pipeline.",
        "row_count": 4,
        "truncated": False,
        "max_rows": 1000,
        "chart_requested": False,
        "chart_type": None,
        "chart_filename": None,
        "chart_note": None,
        "answer_fallback_used": False,
        "refused": False,
        "success": True,
        "elapsed_seconds": 1.25,
    }
    payload.update(overrides)
    return save_history(**payload)


class TestInitialization:
    def test_creates_database_file(self, isolated_history_database):
        assert not isolated_history_database.exists()
        initialize_history_database()
        assert isolated_history_database.exists()

    def test_creates_table(self, isolated_history_database):
        initialize_history_database()
        connection = duckdb.connect(str(isolated_history_database))
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
            }
        finally:
            connection.close()
        assert history_repository.HISTORY_TABLE in tables

    def test_initialization_is_idempotent(self):
        initialize_history_database()
        initialize_history_database()
        assert list_history() == []

    def test_save_works_without_explicit_initialization(self):
        """A first save must not fail because init was never called."""
        save_sample()
        assert len(list_history()) == 1


class TestSaveAndList:
    def test_saves_and_returns_a_uuid(self):
        history_id = save_sample()
        assert isinstance(history_id, str)
        assert len(history_id) == 36
        assert history_id.count("-") == 4

    def test_ids_are_unique_for_identical_questions(self):
        """Ids must be server-generated, never derived from the question."""
        first = save_sample("Same question")
        second = save_sample("Same question")
        assert first != second

    def test_stored_fields_round_trip(self):
        save_sample(
            "Deals by owner",
            answer="C. Mehta leads.",
            row_count=5,
            truncated=True,
            max_rows=1000,
            chart_requested=True,
            chart_type="bar",
            chart_filename="deals_by_owner_20260811_120000.png",
            chart_note="A bar chart was used.",
            elapsed_seconds=2.5,
        )
        record = list_history()[0]
        assert record.original_question == "Deals by owner"
        assert record.answer == "C. Mehta leads."
        assert record.row_count == 5
        assert record.truncated is True
        assert record.max_rows == 1000
        assert record.chart_requested is True
        assert record.chart_type == "bar"
        assert record.chart_filename == "deals_by_owner_20260811_120000.png"
        assert record.chart_note == "A bar chart was used."
        assert record.elapsed_seconds == pytest.approx(2.5)

    def test_lists_newest_first(self):
        for question in ("first", "second", "third"):
            save_sample(question)
        assert [record.original_question for record in list_history()] == [
            "third",
            "second",
            "first",
        ]

    def test_multiple_records_persist(self):
        for index in range(7):
            save_sample(f"question {index}")
        assert len(list_history()) == 7

    def test_records_survive_reconnection(self, isolated_history_database):
        save_sample("persisted question")
        # Every operation opens its own connection, so this proves durability
        # on disk rather than in a cached handle.
        connection = duckdb.connect(str(isolated_history_database))
        try:
            count = connection.execute(
                f"SELECT COUNT(*) FROM {history_repository.HISTORY_TABLE}"
            ).fetchone()[0]
        finally:
            connection.close()
        assert count == 1
        assert list_history()[0].original_question == "persisted question"

    def test_refusal_and_fallback_flags_persist(self):
        save_sample("Capital of France?", refused=True, success=False, answer=None)
        record = list_history()[0]
        assert record.refused is True
        assert record.success is False
        assert record.answer is None

    def test_rejects_blank_question(self):
        with pytest.raises(HistoryError, match="question is required"):
            save_history(original_question="   ", answer="x")


class TestLimits:
    def test_default_limit_is_applied(self):
        for index in range(60):
            save_sample(f"question {index}")
        assert len(list_history()) == 50

    def test_explicit_limit_is_honoured(self):
        for index in range(10):
            save_sample(f"question {index}")
        assert len(list_history(3)) == 3

    def test_limit_is_capped_at_the_server_maximum(self):
        for index in range(5):
            save_sample(f"question {index}")
        # A caller cannot request an unbounded page.
        assert len(list_history(100_000)) == 5

    @pytest.mark.parametrize("bad", [0, -1, None, "abc"])
    def test_invalid_limits_fall_back_to_the_default(self, bad):
        save_sample()
        assert len(list_history(bad)) == 1


class TestTimestamps:
    def test_timestamp_is_utc_aware(self):
        save_sample()
        created = list_history()[0].created_at
        assert created.tzinfo is not None
        assert created.utcoffset().total_seconds() == 0

    def test_timestamp_is_close_to_now(self):
        save_sample()
        delta = abs((datetime.now(timezone.utc) - list_history()[0].created_at).total_seconds())
        assert delta < 60

    def test_timestamp_serializes_to_iso8601(self):
        save_sample()
        text = list_history()[0].created_at.isoformat()
        assert "T" in text
        assert text.endswith("+00:00")


class TestChartFilenameSafety:
    @pytest.mark.parametrize(
        "value",
        [
            "chart.png",
            "deals_by_region_20260811_120000.png",
            "a-b.c_1.PNG",
        ],
    )
    def test_accepts_plain_png_names(self, value):
        assert safe_chart_filename(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "../secret.png",
            "sub/dir/chart.png",
            "sub\\dir\\chart.png",
            "C:\\Projects\\charts\\chart.png",
            "/etc/passwd",
            "chart.txt",
            "chart",
            "",
            "   ",
            None,
            42,
            ".hidden.png",
        ],
    )
    def test_rejects_paths_and_non_png(self, value):
        assert safe_chart_filename(value) is None

    def test_a_path_is_reduced_to_its_name(self):
        assert safe_chart_filename(Path("C:/Projects/charts/chart.png")) == "chart.png"

    def test_absolute_path_is_never_stored(self):
        save_sample(chart_requested=True, chart_filename="C:\\Projects\\charts\\evil.png")
        assert list_history()[0].chart_filename is None

    def test_stored_filename_has_no_directory_component(self):
        save_sample(chart_requested=True, chart_filename="chart.png")
        stored = list_history()[0].chart_filename
        assert stored == "chart.png"
        assert "/" not in stored and "\\" not in stored

    def test_malformed_stored_value_is_neutralized_on_read(self, isolated_history_database):
        """A hand-edited row must not be able to inject a path at read time."""
        save_sample(chart_requested=True, chart_filename="chart.png")
        connection = duckdb.connect(str(isolated_history_database))
        try:
            connection.execute(
                f"UPDATE {history_repository.HISTORY_TABLE} SET chart_filename = ?",
                ["../../etc/passwd"],
            )
        finally:
            connection.close()
        assert list_history()[0].chart_filename is None


class TestPublicModelSafety:
    def test_record_has_no_sql_field(self):
        names = {field.name for field in fields(HistoryRecord)}
        assert "sql" not in names
        assert not any("sql" in name.lower() for name in names)

    def test_record_has_no_schema_or_path_field(self):
        names = {field.name for field in fields(HistoryRecord)}
        assert not any("schema" in name.lower() for name in names)
        assert not any("path" in name.lower() for name in names)

    def test_saved_record_never_exposes_sql(self):
        save_sample()
        record = list_history()[0]
        assert not hasattr(record, "sql")
        assert "SELECT" not in repr(record).upper()


class TestClearHistory:
    def test_clears_all_records(self):
        for index in range(4):
            save_sample(f"question {index}")
        assert clear_history() == 4
        assert list_history() == []

    def test_clearing_empty_history_is_safe(self):
        assert clear_history() == 0

    def test_table_survives_clearing(self):
        save_sample()
        clear_history()
        # The table must still exist so the next save does not fail.
        save_sample("after clear")
        assert len(list_history()) == 1

    def test_clearing_does_not_remove_the_database_file(self, isolated_history_database):
        save_sample()
        clear_history()
        assert isolated_history_database.exists()


class TestFailureHandling:
    def test_open_failure_raises_history_error(self, monkeypatch, tmp_path):
        # A directory cannot be opened as a database file.
        monkeypatch.setattr(history_repository, "HISTORY_DATABASE", tmp_path)
        with pytest.raises(HistoryError):
            save_sample()

    def test_isolation_fixture_points_away_from_production(self, isolated_history_database):
        production = Path(__file__).resolve().parent.parent / "history.duckdb"
        assert isolated_history_database != production
        assert history_repository.HISTORY_DATABASE != production
