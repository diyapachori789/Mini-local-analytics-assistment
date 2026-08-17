"""Chart generation from query results.

Charts are rendered from a :class:`QueryResult` that has already been produced by
DuckDB. This module never queries the database, never calls the language model,
and never invents a value: whatever is plotted came out of the same result the
natural-language answer was written from.

Chart intent and chart type are decided deterministically, by inspecting the
question text and the shape of the result. An LLM is deliberately not used for
that decision -- it would be slower, cost tokens, and be impossible to test.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import matplotlib

# Render without a display. Must be selected before pyplot is imported.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import (  # noqa: E402
    CHART_DPI,
    CHART_FIGURE_SIZE,
    CHART_HORIZONTAL_THRESHOLD,
    CHART_MAX_PIE_SLICES,
    CHARTS_DIR,
)
from database import QueryResult  # noqa: E402

logger = logging.getLogger(__name__)


class ChartError(RuntimeError):
    """Raised when a chart cannot be produced from a result."""


class ChartType(str, Enum):
    """Chart types this project supports."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"


class ChartDecision(str, Enum):
    """Internal reason a result should, or should not, be visualized."""

    AUTO_USEFUL = "auto_useful"
    USER_REQUESTED = "user_requested"
    NO_CHART = "no_chart"


@dataclass(frozen=True)
class ChartRecommendation:
    """A deterministic chart decision made without a model or database call."""

    decision: ChartDecision
    should_render: bool
    chart_type: Optional[ChartType] = None
    note: Optional[str] = None


# --- Intent detection ------------------------------------------------------

# Words that only appear when someone is explicitly asking to see a picture.
# Automatic usefulness is a separate, post-query decision below.
_CHART_INTENT_RE = re.compile(
    r"""
    \b(
        chart | charts | charted | chart\sit
      | graph | graphs | graphed
      | plot  | plots  | plotted
      | diagram
      | visuali[sz]e | visuali[sz]ation | visuali[sz]ations
      | bar\s?chart | pie\s?chart | line\s?chart | scatter\s?plot
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Explicit type requests. Order matters only in that each pattern is distinct.
_EXPLICIT_TYPE_PATTERNS = (
    (ChartType.PIE, re.compile(r"\bpie\s*(chart|graph|plot)?\b", re.IGNORECASE)),
    (ChartType.BAR, re.compile(r"\b(bar|column)\s*(chart|graph|plot)?\b", re.IGNORECASE)),
    (ChartType.LINE, re.compile(r"\bline\s*(chart|graph|plot)?\b", re.IGNORECASE)),
    (
        ChartType.SCATTER,
        re.compile(r"\bscatter\s*(plot|chart|graph|diagram)?\b", re.IGNORECASE),
    ),
)

# Phrases that are presentation instructions rather than part of the question.
# Removed before the question reaches either LLM call: "chart it" cannot be
# answered in SQL, and asking the model to satisfy it invites a refusal.
_DIRECTIVE_PATTERNS = (
    re.compile(
        r"[,;]?\s*(and\s+)?(also\s+)?(please\s+)?"
        r"(show|display|give|make|create|draw|generate|render|produce)\s+"
        r"(me\s+)?(it|this|that|them)?\s*"
        r"(as\s+)?(a|an|the)?\s*"
        r"(horizontal\s+|vertical\s+|stacked\s+)?"
        r"(bar|line|pie|scatter|column)?\s*"
        r"(chart|graph|plot|diagram|visuali[sz]ation)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"[,;]?\s*(and\s+)?(also\s+)?(please\s+)?"
        r"(chart|graph|plot|visuali[sz]e)\s+(it|this|that|them)\b",
        re.IGNORECASE,
    ),
    # "as a pie chart", "as a chart", "in a bar graph" - the type word is
    # optional, but the noun is not, so ordinary prose is left alone.
    re.compile(
        r"[,;]?\s*(and\s+)?(as|in|using|with)\s+(a|an)\s+"
        r"(horizontal\s+|vertical\s+|stacked\s+)?"
        r"(bar|line|pie|scatter|column)?\s*"
        r"(chart|graph|plot|diagram|visuali[sz]ation)\b",
        re.IGNORECASE,
    ),
    # A trailing bare directive, e.g. "... — and chart it." or "... and plot".
    re.compile(
        r"[,;—-]*\s*(and\s+)?(chart|graph|plot|visuali[sz]e)\s*[.!?]*\s*$",
        re.IGNORECASE,
    ),
)

# Question wording that signals what the user is actually looking for.
_TIME_WORDS_RE = re.compile(
    r"\b(trend|trends|over\s+time|monthly|month|quarterly|quarter|yearly|year|"
    r"weekly|daily|timeline|time\s+series|by\s+date|progression|growth)\b",
    re.IGNORECASE,
)
_SHARE_WORDS_RE = re.compile(
    r"\b(share|shares|proportion|proportions|percentage|percentages|composition|"
    r"breakdown|split|distribution|make\s?up|part\s+of|out\s+of\s+the\s+total)\b",
    re.IGNORECASE,
)
# Used to restore the right terminator after a trailing directive is removed.
_QUESTION_OPENERS = frozenset(
    {
        "how", "what", "which", "who", "whom", "whose", "when", "where", "why",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "should", "would", "will", "has", "have", "had",
    }
)

_RELATION_WORDS_RE = re.compile(
    r"\b(relationship|correlat\w*|versus|vs\.?|against|compared\s+with|"
    r"scatter|association)\b",
    re.IGNORECASE,
)

_COMPARISON_WORDS_RE = re.compile(
    r"\b(compare|comparison|across|by|per|each|top|bottom|rank\w*|highest|"
    r"lowest|most|least|breakdown|distribution|share|composition|split)\b",
    re.IGNORECASE,
)
_GROUPED_SQL_RE = re.compile(r"\bgroup\s+by\b", re.IGNORECASE)
_IDENTIFIER_COLUMN_RE = re.compile(r"(^id$|_id$|^identifier$)", re.IGNORECASE)

# Rendering hundreds of unrelated categories is technically possible but not
# useful. Time series and relationships tolerate more points than category
# labels, while automatic category charts remain deliberately compact.
_AUTO_MAX_CATEGORY_ROWS = 30
_AUTO_MAX_SERIES_ROWS = 250
_EXPLICIT_MAX_CATEGORY_ROWS = 100
_EXPLICIT_MAX_SERIES_ROWS = 500
_MAX_VISUAL_COLUMNS = 4


def is_chart_request(question: str) -> bool:
    """True when the question explicitly asks for a chart.

    This helper recognizes presentation wording only. Automatic usefulness is
    handled separately by :func:`recommend_chart` after a result exists.
    """
    if not isinstance(question, str) or not question.strip():
        return False
    return bool(_CHART_INTENT_RE.search(question))


def requested_chart_type(question: str) -> Optional[ChartType]:
    """Return the chart type the user named, or None if they did not name one."""
    if not isinstance(question, str):
        return None
    for chart_type, pattern in _EXPLICIT_TYPE_PATTERNS:
        if pattern.search(question):
            return chart_type
    return None


def strip_chart_directive(question: str) -> str:
    """Remove the charting instruction, leaving the data question behind.

    "How many deals did each owner close won? - and chart it."
        -> "How many deals did each owner close won?"

    Returns the original text if stripping would leave nothing meaningful.
    """
    if not isinstance(question, str):
        return question

    stripped = question
    for pattern in _DIRECTIVE_PATTERNS:
        stripped = pattern.sub(" ", stripped)

    # Removing a trailing clause leaves punctuation debris behind, e.g.
    # "... close won? — and chart it." becomes "... close won? — .". Strip every
    # trailing separator and terminator, then restore a single correct one.
    stripped = re.sub(r"\s+", " ", stripped).strip()
    # A dangling preposition can survive, e.g. "... value as a" once "chart" goes.
    stripped = re.sub(
        r"\s+(as|in|using|with)\s+(a|an|the)?\s*$", "", stripped, flags=re.IGNORECASE
    )
    stripped = stripped.strip(" ,;:.!?-—–")

    # If the directive was the whole question, keep the original rather than
    # sending an empty string to the model.
    if len(stripped) < 5:
        return question.strip()

    opening = stripped.split(" ", 1)[0].lower()
    stripped += "?" if opening in _QUESTION_OPENERS else "."
    return stripped


# --- Column classification -------------------------------------------------


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]


def _temporal_columns(frame: pd.DataFrame) -> list[str]:
    temporal = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_datetime64_any_dtype(series):
            temporal.append(name)
        elif (
            (series.dtype == object or pd.api.types.is_string_dtype(series))
            and len(series)
            and _looks_like_dates(series)
        ):
            temporal.append(name)
    return temporal


def _looks_like_dates(series: pd.Series) -> bool:
    """True when an object column actually holds dates."""
    sample = series.dropna().head(5)
    if sample.empty:
        return False
    import datetime as _dt

    if all(isinstance(value, (_dt.date, _dt.datetime)) for value in sample):
        return True
    # Strings such as "2025-01" or "2025-01-31".
    return all(
        isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", value.strip())
        for value in sample
    )


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(_numeric_columns(frame))
    temporal = set(_temporal_columns(frame))
    return [name for name in frame.columns if name not in numeric and name not in temporal]


def _meaningful_numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric measures while excluding obvious technical identifiers."""
    return [
        name
        for name in _numeric_columns(frame)
        if not _IDENTIFIER_COLUMN_RE.search(str(name).strip())
    ]


def _unsuitable_note(result: QueryResult) -> str:
    if result.is_empty:
        return "A meaningful chart could not be created because the query returned no rows."
    if len(result.frame) <= 1:
        return (
            "A meaningful chart could not be created because the returned data is a "
            "single value or single record."
        )
    return "A meaningful chart could not be created from the returned data."


def _explicit_shape_is_useful(result: QueryResult) -> bool:
    """Whether an explicit request has enough coherent data to visualize."""
    frame = result.frame
    row_count = len(frame)
    if result.is_empty or row_count <= 1 or len(frame.columns) > _MAX_VISUAL_COLUMNS:
        return False

    numeric = _meaningful_numeric_columns(frame)
    temporal = _temporal_columns(frame)
    categorical = _categorical_columns(frame)
    if temporal and numeric:
        return row_count <= _EXPLICIT_MAX_SERIES_ROWS
    if len(numeric) >= 2:
        return row_count <= _EXPLICIT_MAX_SERIES_ROWS
    if categorical and numeric:
        return row_count <= _EXPLICIT_MAX_CATEGORY_ROWS
    return False


def recommend_chart(question: str, result: QueryResult) -> ChartRecommendation:
    """Decide locally whether this one authoritative result merits a chart.

    Explicit presentation intent is respected when the result has a meaningful
    visual structure. Without an explicit request, both the question/query
    semantics and a compact result shape must support the visualization. This
    function performs no I/O and never calls the model or DuckDB.
    """
    explicit = is_chart_request(question)
    if not isinstance(result, QueryResult):
        return ChartRecommendation(ChartDecision.NO_CHART, False)

    frame = result.frame
    row_count = len(frame)
    if explicit:
        if not _explicit_shape_is_useful(result):
            return ChartRecommendation(
                ChartDecision.USER_REQUESTED,
                False,
                note=_unsuitable_note(result),
            )
        chart_type, note = resolve_chart_type(question, result)
        return ChartRecommendation(
            ChartDecision.USER_REQUESTED,
            True,
            chart_type=chart_type,
            note=note,
        )

    if (
        result.is_empty
        or row_count <= 1
        or len(frame.columns) > _MAX_VISUAL_COLUMNS
        or result.truncated
    ):
        return ChartRecommendation(ChartDecision.NO_CHART, False)

    numeric = _meaningful_numeric_columns(frame)
    temporal = _temporal_columns(frame)
    categorical = _categorical_columns(frame)
    question_text = question or ""
    grouped_query = bool(_GROUPED_SQL_RE.search(result.sql or ""))

    if temporal and numeric and row_count <= _AUTO_MAX_SERIES_ROWS:
        return ChartRecommendation(
            ChartDecision.AUTO_USEFUL, True, ChartType.LINE
        )

    if (
        len(numeric) >= 2
        and row_count <= _AUTO_MAX_SERIES_ROWS
        and _RELATION_WORDS_RE.search(question_text)
    ):
        return ChartRecommendation(
            ChartDecision.AUTO_USEFUL, True, ChartType.SCATTER
        )

    comparison_semantics = bool(_COMPARISON_WORDS_RE.search(question_text)) or grouped_query
    if (
        categorical
        and numeric
        and row_count <= _AUTO_MAX_CATEGORY_ROWS
        and comparison_semantics
    ):
        return ChartRecommendation(
            ChartDecision.AUTO_USEFUL,
            True,
            select_chart_type(question_text, result),
        )

    # A numeric ordered dimension can represent time even when the database
    # returns it as an integer year/quarter rather than a date-typed column.
    if (
        len(numeric) >= 2
        and row_count <= _AUTO_MAX_SERIES_ROWS
        and _TIME_WORDS_RE.search(question_text)
    ):
        return ChartRecommendation(
            ChartDecision.AUTO_USEFUL, True, ChartType.LINE
        )

    return ChartRecommendation(ChartDecision.NO_CHART, False)


# --- Chart type selection --------------------------------------------------


def select_chart_type(question: str, result: QueryResult) -> ChartType:
    """Choose the chart that fits the question and the shape of the result.

    Applied only when the user did not name a type. The order of these checks is
    the priority order: a time series is a line chart even if the wording also
    mentions comparison, because plotting dates as unordered categories loses the
    thing the question was about.
    """
    frame = result.frame
    numeric = _numeric_columns(frame)
    temporal = _temporal_columns(frame)
    categorical = _categorical_columns(frame)

    # Time series: an ordered progression, so the line carries meaning.
    if temporal and numeric:
        return ChartType.LINE
    if _TIME_WORDS_RE.search(question or "") and numeric and len(frame) > 1:
        # Wording says trend and there is something to trend, but no real date
        # column; only treat it as a line if the first column is ordered-looking.
        if categorical and len(frame) > 2:
            return ChartType.LINE
        if len(numeric) >= 2:
            return ChartType.LINE

    # Relationship between two measured quantities.
    if len(numeric) >= 2 and _RELATION_WORDS_RE.search(question or ""):
        return ChartType.SCATTER
    if len(numeric) >= 2 and not categorical and len(frame) > 2:
        return ChartType.SCATTER

    # Part-to-whole, but only when the slices would actually be readable.
    if (
        _SHARE_WORDS_RE.search(question or "")
        and categorical
        and numeric
        and 1 < len(frame) <= CHART_MAX_PIE_SLICES
        and (frame[numeric[0]] >= 0).all()
    ):
        return ChartType.PIE

    # Everything else is a comparison between categories.
    return ChartType.BAR


def resolve_chart_type(question: str, result: QueryResult) -> tuple[ChartType, Optional[str]]:
    """Decide the chart type, honouring an explicit request where it is sound.

    Returns the type plus a note for the user when a request had to be overruled.
    The data is never reshaped to satisfy an unsuitable request; the chart type
    changes instead.
    """
    explicit = requested_chart_type(question)
    if explicit is None:
        return select_chart_type(question, result), None

    frame = result.frame
    numeric = _numeric_columns(frame)
    temporal = _temporal_columns(frame)
    categorical = _categorical_columns(frame)

    if explicit is ChartType.PIE:
        if temporal:
            logger.warning("Pie chart requested for time-series data; using a line chart.")
            return ChartType.LINE, (
                "A line chart was used instead of a pie chart because the result is a "
                "time series, where slices of a whole would be misleading."
            )
        if not numeric or not categorical:
            logger.warning("Pie chart requested but the result has no category/value pair.")
            return ChartType.BAR, (
                "A bar chart was used instead of a pie chart because the result does "
                "not have a single category-and-value pair to divide into slices."
            )
        if len(frame) > CHART_MAX_PIE_SLICES:
            logger.warning(
                "Pie chart requested for %s categories; using a bar chart.", len(frame)
            )
            return ChartType.BAR, (
                f"A bar chart was used instead of a pie chart because {len(frame)} "
                "categories produce slices too thin to read."
            )
        if (frame[numeric[0]] < 0).any():
            logger.warning("Pie chart requested for data containing negative values.")
            return ChartType.BAR, (
                "A bar chart was used instead of a pie chart because negative values "
                "cannot be shown as a share of a whole."
            )

    if explicit is ChartType.SCATTER and len(numeric) < 2:
        logger.warning("Scatter requested but the result has fewer than two numeric columns.")
        return ChartType.BAR, (
            "A bar chart was used instead of a scatter plot because the result does "
            "not contain two numeric columns to plot against each other."
        )

    if explicit is ChartType.LINE and not numeric:
        logger.warning("Line chart requested but the result has no numeric column.")
        return ChartType.BAR, (
            "A bar chart was used instead of a line chart because the result has no "
            "numeric values to trace."
        )

    return explicit, None


# --- Rendering -------------------------------------------------------------


def _humanise(column: str) -> str:
    """Turn a column name into an axis label."""
    return column.replace("_", " ").strip().title()


def _slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return (slug[:limit].rstrip("_") or "chart")


def _unique_path(directory: Path, stem: str) -> Path:
    """Build a timestamped path, never overwriting an existing file."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = directory / f"{stem}_{stamp}.png"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{stamp}_{counter}.png"
        counter += 1
    return candidate


def _pick_axes(frame: pd.DataFrame, chart_type: ChartType) -> tuple[str, str]:
    """Choose which column is the label/x and which is the value/y."""
    numeric = _numeric_columns(frame)
    temporal = _temporal_columns(frame)
    categorical = _categorical_columns(frame)

    if chart_type is ChartType.SCATTER:
        if len(numeric) < 2:
            raise ChartError("A scatter plot needs two numeric columns.")
        return numeric[0], numeric[1]

    if not numeric:
        raise ChartError("The result has no numeric column to plot.")

    if chart_type is ChartType.LINE and temporal:
        return temporal[0], numeric[0]

    if categorical:
        return categorical[0], numeric[0]
    if temporal:
        return temporal[0], numeric[0]
    if len(numeric) >= 2:
        return numeric[0], numeric[1]

    raise ChartError("The result does not have a label column to plot against.")


def _validate_result(result: QueryResult) -> None:
    if not isinstance(result, QueryResult):
        raise ChartError("A QueryResult is required to draw a chart.")
    if result.is_empty:
        raise ChartError("There are no rows to chart.")
    if len(result.frame.columns) < 2 and len(_numeric_columns(result.frame)) < 1:
        raise ChartError("The result does not have enough columns to chart.")


def create_chart(
    question: str,
    result: QueryResult,
    chart_type: Optional[ChartType] = None,
) -> tuple[Path, ChartType, Optional[str]]:
    """Render a chart for a result and return its path, type and any fallback note.

    ``chart_type`` overrides detection entirely; leave it as ``None`` to honour
    the user's request or select automatically.
    """
    _validate_result(result)

    if chart_type is None:
        chart_type, note = resolve_chart_type(question, result)
    else:
        note = None

    frame = result.frame
    x_column, y_column = _pick_axes(frame, chart_type)

    if chart_type is ChartType.PIE and (frame[y_column] < 0).any():
        raise ChartError("A pie chart cannot show negative values.")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _unique_path(CHARTS_DIR, _slugify(strip_chart_directive(question)))

    title = strip_chart_directive(question).rstrip("?.! ")
    figure, axes = plt.subplots(figsize=CHART_FIGURE_SIZE)

    try:
        labels = frame[x_column]
        values = frame[y_column]

        if chart_type is ChartType.BAR:
            if len(frame) >= CHART_HORIZONTAL_THRESHOLD:
                axes.barh([str(value) for value in labels], values, color="#4C78A8")
                axes.set_xlabel(_humanise(y_column))
                axes.set_ylabel(_humanise(x_column))
                axes.invert_yaxis()
            else:
                axes.bar([str(value) for value in labels], values, color="#4C78A8")
                axes.set_xlabel(_humanise(x_column))
                axes.set_ylabel(_humanise(y_column))
                if max((len(str(value)) for value in labels), default=0) > 8:
                    plt.setp(axes.get_xticklabels(), rotation=30, ha="right")
            axes.grid(axis="x" if len(frame) >= CHART_HORIZONTAL_THRESHOLD else "y",
                      alpha=0.3)

        elif chart_type is ChartType.LINE:
            axes.plot(labels, values, marker="o", color="#4C78A8")
            axes.set_xlabel(_humanise(x_column))
            axes.set_ylabel(_humanise(y_column))
            axes.grid(alpha=0.3)
            if max((len(str(value)) for value in labels), default=0) > 6:
                plt.setp(axes.get_xticklabels(), rotation=30, ha="right")

        elif chart_type is ChartType.PIE:
            axes.pie(
                values,
                labels=[str(value) for value in labels],
                autopct="%1.1f%%",
                startangle=90,
                counterclock=False,
            )
            axes.axis("equal")
            axes.set_ylabel("")

        elif chart_type is ChartType.SCATTER:
            axes.scatter(labels, values, color="#4C78A8", alpha=0.75)
            axes.set_xlabel(_humanise(x_column))
            axes.set_ylabel(_humanise(y_column))
            axes.grid(alpha=0.3)

        else:  # pragma: no cover - ChartType is exhaustive
            raise ChartError(f"Unsupported chart type: {chart_type}")

        axes.set_title(title)
        figure.tight_layout()
        figure.savefig(output_path, dpi=CHART_DPI)
    except ChartError:
        raise
    except Exception as exc:
        logger.error("Chart generation failed: %s", exc)
        raise ChartError(f"Could not render the chart: {exc}") from exc
    finally:
        plt.close(figure)

    logger.info(
        "Chart generated: type=%s rows=%s file=%s",
        chart_type.value,
        result.row_count,
        output_path,
    )
    return output_path, chart_type, note


# --- Terminal link ---------------------------------------------------------


def terminal_link(path: Path, label: str = "Open chart", *, supports_links: bool = True) -> str:
    """Render a clickable OSC 8 hyperlink, or the absolute path as a fallback.

    Terminals that do not understand OSC 8 would print escape codes as noise, so
    callers pass ``supports_links=False`` when stdout is not an interactive
    terminal. The application never fails over a missing hyperlink.
    """
    absolute = Path(path).resolve()
    if not supports_links:
        return str(absolute)
    try:
        uri = absolute.as_uri()
    except ValueError:
        return str(absolute)
    return f"\033]8;;{uri}\033\\{label}\033]8;;\033\\"
