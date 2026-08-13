"""Phase 6: chart intent detection, type selection and PNG rendering.

Charts are rendered for real (matplotlib, Agg backend) but from mocked query
results, so this file makes no API calls and consumes no tokens.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

import chart
from chart import (
    ChartError,
    ChartType,
    create_chart,
    is_chart_request,
    requested_chart_type,
    resolve_chart_type,
    select_chart_type,
    strip_chart_directive,
    terminal_link,
)
from database import QueryResult


def make_result(frame: pd.DataFrame, *, sql="SELECT 1;") -> QueryResult:
    return QueryResult(
        frame=frame,
        sql=sql,
        row_count=len(frame),
        truncated=False,
        max_rows=1000,
    )


# The assignment's own example data.
OWNER_WINS = make_result(
    pd.DataFrame(
        {
            "owner": ["C. Mehta", "A. Rao", "E. Shah", "D. Patel", "B. Singh"],
            "closed_won": [19, 15, 15, 13, 9],
        }
    )
)
REGION_AMOUNTS = make_result(
    pd.DataFrame(
        {"region": ["NA", "EMEA", "LATAM", "APAC"], "total": [6260208, 6401782, 6523968, 5454505]}
    )
)
STAGE_SHARE = make_result(
    pd.DataFrame(
        {
            "stage": ["Qualification", "Proposal", "Prospecting", "Negotiation"],
            "total_amount": [4224186, 3826472, 3574616, 3392955],
        }
    )
)
MONTHLY_TREND = make_result(
    pd.DataFrame(
        {
            "month": [dt.date(2025, 1, 1), dt.date(2025, 2, 1), dt.date(2025, 3, 1)],
            "pipeline": [120000, 145000, 133000],
        }
    )
)
NUMERIC_PAIR = make_result(
    pd.DataFrame(
        {"amount": [10, 25, 40, 55, 70, 85], "win_rate": [0.1, 0.2, 0.3, 0.25, 0.4, 0.5]}
    )
)
MANY_CATEGORIES = make_result(
    pd.DataFrame({"account": [f"Acct {n}" for n in range(12)], "value": list(range(12))})
)
EMPTY = make_result(pd.DataFrame(columns=["region", "total"]))


@pytest.fixture
def charts_dir(tmp_path, monkeypatch):
    """Redirect chart output to a temporary directory."""
    target = tmp_path / "charts"
    monkeypatch.setattr(chart, "CHARTS_DIR", target)
    return target


# --- 1. Intent detection ---------------------------------------------------


class TestIsChartRequest:
    @pytest.mark.parametrize(
        "question",
        [
            "chart it",
            "How many opportunities did each owner close won? — and chart it.",
            "show it as a chart",
            "make a chart",
            "create a graph",
            "plot it",
            "visualize this",
            "visualise this",
            "show me a visualization",
            "show as a pie chart",
            "show as a bar chart",
            "show as a line chart",
            "show as a scatter plot",
            "Compare opportunity amounts by region using a bar graph.",
            "Show the monthly pipeline trend as a line chart.",
        ],
    )
    def test_detects_chart_intent(self, question):
        assert is_chart_request(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "What is the overall win rate?",
            "Which region has the highest amount?",
            "What is the total pipeline?",
            "How many opportunities did each owner close won?",
            "Show all opportunities in EMEA.",
            "What is the total amount of closed won opportunities?",
        ],
    )
    def test_ignores_ordinary_questions(self, question):
        """A chart must be asked for, never inferred from the result shape."""
        assert is_chart_request(question) is False

    @pytest.mark.parametrize("value", [None, "", "   ", 42])
    def test_handles_bad_input(self, value):
        assert is_chart_request(value) is False


# --- 2. Explicit chart type ------------------------------------------------


class TestRequestedChartType:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("Show opportunity amount by region as a pie chart.", ChartType.PIE),
            ("Show it as a pie graph", ChartType.PIE),
            ("Compare amounts by region using a bar chart.", ChartType.BAR),
            ("Compare amounts by region using a bar graph.", ChartType.BAR),
            ("Show the monthly pipeline trend as a line chart.", ChartType.LINE),
            ("Show the trend as a line graph", ChartType.LINE),
            ("Show amount versus win rate as a scatter plot.", ChartType.SCATTER),
            ("Show amount versus win rate as a scatter chart.", ChartType.SCATTER),
        ],
    )
    def test_detects_explicit_type(self, question, expected):
        assert requested_chart_type(question) is expected

    @pytest.mark.parametrize(
        "question",
        ["chart it", "plot it", "show me a visualization", "make a chart"],
    )
    def test_returns_none_when_unspecified(self, question):
        assert requested_chart_type(question) is None


# --- 3. Directive stripping ------------------------------------------------


class TestStripChartDirective:
    @pytest.mark.parametrize(
        "question,must_not_contain",
        [
            ("How many opportunities did each owner close won? — and chart it.", "chart"),
            ("Compare opportunity amounts by region and chart it.", "chart"),
            ("Show opportunity amount by region as a pie chart.", "pie"),
            ("Show the monthly pipeline trend as a line chart.", "line chart"),
            ("Show the top 5 accounts by opportunity value as a chart.", "chart"),
        ],
    )
    def test_directive_is_removed(self, question, must_not_contain):
        assert must_not_contain.lower() not in strip_chart_directive(question).lower()

    def test_data_question_survives(self):
        cleaned = strip_chart_directive(
            "How many opportunities did each owner close won? — and chart it."
        )
        assert "owner" in cleaned
        assert "close won" in cleaned

    def test_plain_question_is_untouched(self):
        question = "What is the overall win rate?"
        assert strip_chart_directive(question) == question

    def test_directive_only_question_is_preserved(self):
        """Stripping must never leave an empty question for the model."""
        assert len(strip_chart_directive("chart it")) >= 5


# --- 4. Automatic type selection -------------------------------------------


class TestAutomaticSelection:
    def test_categorical_comparison_is_bar(self):
        assert select_chart_type(
            "Compare opportunity amounts by region", REGION_AMOUNTS
        ) is ChartType.BAR

    def test_ranking_is_bar(self):
        assert select_chart_type(
            "Show the top 5 accounts by opportunity value", OWNER_WINS
        ) is ChartType.BAR

    def test_assignment_example_is_bar(self):
        """The assignment's own example must pick a bar chart."""
        assert select_chart_type(
            "How many opportunities did each owner close won?", OWNER_WINS
        ) is ChartType.BAR

    def test_time_series_is_line(self):
        assert select_chart_type(
            "Show the pipeline trend over time", MONTHLY_TREND
        ) is ChartType.LINE

    def test_share_question_is_pie(self):
        assert select_chart_type(
            "Show the share of opportunities by stage", STAGE_SHARE
        ) is ChartType.PIE

    def test_relationship_is_scatter(self):
        assert select_chart_type(
            "Show the relationship between opportunity amount and win rate", NUMERIC_PAIR
        ) is ChartType.SCATTER

    def test_share_with_too_many_categories_is_bar(self):
        """Composition wording does not justify unreadable slices."""
        assert select_chart_type(
            "Show the share by account", MANY_CATEGORIES
        ) is ChartType.BAR

    def test_default_is_bar(self):
        assert select_chart_type("Amounts by region", REGION_AMOUNTS) is ChartType.BAR

    def test_selection_is_deterministic(self):
        first = select_chart_type("Compare amounts by region", REGION_AMOUNTS)
        for _ in range(5):
            assert select_chart_type("Compare amounts by region", REGION_AMOUNTS) is first


# --- 5. Explicit type vs data compatibility --------------------------------


class TestExplicitTypeCompatibility:
    def test_explicit_type_is_honoured(self):
        chart_type, note = resolve_chart_type(
            "Show opportunity amount by region as a pie chart.", REGION_AMOUNTS
        )
        assert chart_type is ChartType.PIE
        assert note is None

    def test_pie_on_time_series_falls_back_to_line(self):
        chart_type, note = resolve_chart_type(
            "Show monthly pipeline as a pie chart.", MONTHLY_TREND
        )
        assert chart_type is ChartType.LINE
        assert note and "line chart" in note.lower()

    def test_pie_with_too_many_categories_falls_back_to_bar(self):
        chart_type, note = resolve_chart_type(
            "Show share by account as a pie chart.", MANY_CATEGORIES
        )
        assert chart_type is ChartType.BAR
        assert note and "12" in note

    def test_pie_with_negative_values_falls_back_to_bar(self):
        negative = make_result(
            pd.DataFrame({"region": ["NA", "EMEA"], "delta": [100, -50]})
        )
        chart_type, note = resolve_chart_type("Show it as a pie chart.", negative)
        assert chart_type is ChartType.BAR
        assert note and "negative" in note.lower()

    def test_scatter_without_two_numeric_columns_falls_back_to_bar(self):
        chart_type, note = resolve_chart_type(
            "Show it as a scatter plot.", REGION_AMOUNTS
        )
        assert chart_type is ChartType.BAR
        assert note and "scatter" in note.lower()

    def test_fallback_note_explains_the_reason(self):
        _, note = resolve_chart_type("Show monthly pipeline as a pie chart.", MONTHLY_TREND)
        assert "because" in note.lower()


# --- 6. Rendering ----------------------------------------------------------


class TestChartRendering:
    @pytest.mark.parametrize(
        "chart_type,result",
        [
            (ChartType.BAR, REGION_AMOUNTS),
            (ChartType.LINE, MONTHLY_TREND),
            (ChartType.PIE, STAGE_SHARE),
            (ChartType.SCATTER, NUMERIC_PAIR),
        ],
    )
    def test_each_type_produces_a_png(self, charts_dir, chart_type, result):
        path, used, note = create_chart("Test question", result, chart_type=chart_type)
        assert used is chart_type
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000  # a real image, not an empty file

    def test_png_is_a_valid_image(self, charts_dir):
        path, _, _ = create_chart("Amounts by region", REGION_AMOUNTS)
        header = path.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", "file is not a valid PNG"

    def test_returns_path_type_and_note(self, charts_dir):
        path, chart_type, note = create_chart("Amounts by region", REGION_AMOUNTS)
        assert isinstance(path, Path)
        assert isinstance(chart_type, ChartType)
        assert note is None

    def test_charts_directory_is_created(self, tmp_path, monkeypatch):
        target = tmp_path / "nested" / "charts"
        monkeypatch.setattr(chart, "CHARTS_DIR", target)
        assert not target.exists()
        create_chart("Amounts by region", REGION_AMOUNTS)
        assert target.is_dir()

    def test_files_are_not_overwritten(self, charts_dir):
        paths = {create_chart("Amounts by region", REGION_AMOUNTS)[0] for _ in range(3)}
        assert len(paths) == 3
        assert all(path.exists() for path in paths)

    def test_many_categories_render(self, charts_dir):
        path, chart_type, _ = create_chart("Value by account", MANY_CATEGORIES)
        assert chart_type is ChartType.BAR
        assert path.exists()

    def test_grouped_result_renders(self, charts_dir):
        path, _, _ = create_chart("Closed won by owner", OWNER_WINS)
        assert path.exists()

    def test_uses_only_the_supplied_result(self, charts_dir, monkeypatch):
        """Charting must not touch the database or the model."""
        import database
        import llm

        def forbidden(*args, **kwargs):
            raise AssertionError("chart generation must not query or call the LLM")

        monkeypatch.setattr(database, "run_query", forbidden)
        monkeypatch.setattr(database, "execute_query", forbidden)
        monkeypatch.setattr(llm, "generate_sql", forbidden)
        monkeypatch.setattr(llm, "generate_answer", forbidden)

        path, _, _ = create_chart("Amounts by region", REGION_AMOUNTS)
        assert path.exists()


class TestChartTitleAndLabels:
    def test_title_and_axis_labels_are_set(self, charts_dir, monkeypatch):
        captured = {}
        real_subplots = chart.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(chart.plt, "subplots", spy)
        create_chart("Total amount by region", REGION_AMOUNTS, chart_type=ChartType.BAR)

        axes = captured["axes"]
        assert axes.get_title().strip()
        assert axes.get_xlabel().strip()
        assert axes.get_ylabel().strip()

    def test_title_has_no_chart_directive(self, charts_dir, monkeypatch):
        captured = {}
        real_subplots = chart.plt.subplots

        def spy(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["axes"] = axes
            return figure, axes

        monkeypatch.setattr(chart.plt, "subplots", spy)
        create_chart("Total amount by region and chart it.", REGION_AMOUNTS)
        assert "chart it" not in captured["axes"].get_title().lower()


# --- 7. Failure handling ---------------------------------------------------


class TestChartFailures:
    def test_empty_result_is_refused_cleanly(self, charts_dir):
        with pytest.raises(ChartError, match="no rows"):
            create_chart("Amounts by region", EMPTY)

    @pytest.mark.parametrize("bad", [None, "not a result", 42, {"rows": []}])
    def test_invalid_result_structure(self, charts_dir, bad):
        with pytest.raises(ChartError, match="QueryResult is required"):
            create_chart("Anything", bad)

    def test_result_without_numeric_column(self, charts_dir):
        text_only = make_result(
            pd.DataFrame({"region": ["NA", "EMEA"], "owner": ["A", "B"]})
        )
        with pytest.raises(ChartError):
            create_chart("Regions and owners", text_only)

    def test_scatter_without_two_numeric_columns_raises(self, charts_dir):
        with pytest.raises(ChartError, match="two numeric columns"):
            create_chart("Anything", REGION_AMOUNTS, chart_type=ChartType.SCATTER)

    def test_pie_with_negative_values_raises_when_forced(self, charts_dir):
        negative = make_result(pd.DataFrame({"region": ["NA", "EMEA"], "delta": [10, -5]}))
        with pytest.raises(ChartError, match="negative"):
            create_chart("Anything", negative, chart_type=ChartType.PIE)

    def test_render_failure_is_wrapped(self, charts_dir, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("simulated matplotlib failure")

        monkeypatch.setattr(chart.plt.Axes, "bar", boom)
        with pytest.raises(ChartError, match="Could not render"):
            create_chart("Amounts by region", REGION_AMOUNTS, chart_type=ChartType.BAR)


# --- 8. Terminal link ------------------------------------------------------


class TestTerminalLink:
    def test_emits_osc8_hyperlink(self, tmp_path):
        target = tmp_path / "chart.png"
        target.write_bytes(b"x")
        link = terminal_link(target, supports_links=True)
        assert link.startswith("\033]8;;file:///")
        assert "Open chart" in link
        assert link.endswith("\033]8;;\033\\")

    def test_falls_back_to_absolute_path(self, tmp_path):
        target = tmp_path / "chart.png"
        target.write_bytes(b"x")
        link = terminal_link(target, supports_links=False)
        assert link == str(target.resolve())
        assert "\033" not in link

    def test_custom_label(self, tmp_path):
        target = tmp_path / "chart.png"
        target.write_bytes(b"x")
        assert "View it" in terminal_link(target, "View it", supports_links=True)


# --- 9. Logging ------------------------------------------------------------


class TestChartLogging:
    def test_success_is_logged(self, charts_dir, caplog):
        with caplog.at_level("INFO", logger="chart"):
            create_chart("Amounts by region", REGION_AMOUNTS)
        messages = [record.getMessage() for record in caplog.records]
        assert any("Chart generated" in message for message in messages)
        assert any("type=bar" in message for message in messages)

    def test_incompatible_request_is_logged(self, caplog):
        with caplog.at_level("WARNING", logger="chart"):
            resolve_chart_type("Show monthly pipeline as a pie chart.", MONTHLY_TREND)
        assert any(
            "pie chart requested" in record.getMessage().lower()
            for record in caplog.records
        )

    def test_render_failure_is_logged(self, charts_dir, caplog, monkeypatch):
        def boom(*args, **kwargs):
            raise ValueError("simulated")

        monkeypatch.setattr(chart.plt.Axes, "bar", boom)
        with caplog.at_level("ERROR", logger="chart"):
            with pytest.raises(ChartError):
                create_chart("Amounts by region", REGION_AMOUNTS, chart_type=ChartType.BAR)
        assert any(
            "Chart generation failed" in record.getMessage() for record in caplog.records
        )
