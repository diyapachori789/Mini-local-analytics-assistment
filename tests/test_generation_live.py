"""SQL generation regression suite (live).

These tests call the Groq API and cost quota, so they are deselected by default.
Run them explicitly after any prompt or model change:

    pytest -m live

Assertions target SQL *structure*, never exact string equality: an LLM may
legitimately word a query differently while remaining correct.
"""

from __future__ import annotations

import re

import pytest

import database
from config import GROQ_API_KEY
from llm import INVALID_QUESTION, generate_sql

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY is not configured"),
]

LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
STAR_RE = re.compile(r"SELECT\s+\*", re.IGNORECASE)
ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def limit_of(sql: str) -> int | None:
    match = LIMIT_RE.search(sql)
    return int(match.group(1)) if match else None


# (question, expected LIMIT)
TOP_N_CASES = [
    ("List the top 5 opportunities with the highest amount.", 5),
    ("Show the 3 largest deals by amount.", 3),
    ("Give me the first 10 opportunities by close date.", 10),
    ("Which single opportunity has the biggest amount?", 1),
    ("Show the bottom 4 opportunities by amount.", 4),
]

UNLIMITED_CASES = [
    "Find all opportunities created in 2025.",
    "What is the total won amount by region?",
    "How many opportunities are in each stage?",
]

STAR_CASES = [
    "Show every column for opportunities owned by D. Patel.",
    "Give me the full record for opportunity OPP-1000.",
]

NO_STAR_CASES = [
    "Show all opportunities owned by D. Patel.",
    "Show all opportunities in the EMEA region.",
    "List all closed won deals in Technology.",
]

REFUSAL_CASES = [
    "What is the capital of France?",
    "Delete all opportunities from the table.",
]


class TestTopN:
    """Regression anchors: 'top N' and 'N largest' previously omitted LIMIT."""

    @pytest.mark.parametrize("question,expected", TOP_N_CASES)
    def test_limit_is_applied(self, initialized_database, question, expected):
        sql = generate_sql(question)
        assert limit_of(sql) == expected, sql

    @pytest.mark.parametrize("question,expected", TOP_N_CASES)
    def test_limit_always_has_a_matching_order_by(
        self, initialized_database, question, expected
    ):
        """A LIMIT without ORDER BY returns arbitrary rows that look correct."""
        sql = generate_sql(question)
        assert ORDER_BY_RE.search(sql), sql


class TestNoLimit:
    @pytest.mark.parametrize("question", UNLIMITED_CASES)
    def test_no_limit_when_none_requested(self, initialized_database, question):
        sql = generate_sql(question)
        assert limit_of(sql) is None, sql


class TestColumnSelection:
    @pytest.mark.parametrize("question", STAR_CASES)
    def test_select_star_when_all_columns_requested(
        self, initialized_database, question
    ):
        assert STAR_RE.search(generate_sql(question))

    @pytest.mark.parametrize("question", NO_STAR_CASES)
    def test_no_select_star_for_row_scoped_questions(
        self, initialized_database, question
    ):
        """'all opportunities' scopes rows, not columns."""
        sql = generate_sql(question)
        assert not STAR_RE.search(sql), sql


class TestRefusals:
    @pytest.mark.parametrize("question", REFUSAL_CASES)
    def test_unanswerable_questions_are_refused(self, initialized_database, question):
        assert generate_sql(question) == INVALID_QUESTION


class TestGeneratedSqlIsExecutable:
    """Every generated statement must actually run against DuckDB."""

    @pytest.mark.parametrize(
        "question",
        [case[0] for case in TOP_N_CASES] + UNLIMITED_CASES + NO_STAR_CASES,
    )
    def test_executes_without_error(self, initialized_database, question):
        sql = generate_sql(question)
        database.execute_query(sql)
