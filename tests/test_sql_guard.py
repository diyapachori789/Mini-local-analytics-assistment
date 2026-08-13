"""SQL cleaning, masking and safety validation.

These are pure-function tests: no database, no API key, no network.
"""

from __future__ import annotations

import pytest

from sql_guard import (
    ALLOWED_PREFIXES,
    FORBIDDEN_KEYWORDS,
    INVALID_QUESTION,
    clean_sql,
    is_refusal,
    mask_literals_and_comments,
    validate_cleaned_sql,
    validate_sql,
)


class TestCleanSql:
    """clean_sql strips code fences and normalises whitespace."""

    @pytest.mark.parametrize(
        "raw",
        [
            "```sql\nSELECT 1;\n```",
            "```SQL\nSELECT 1;\n```",
            "```duckdb\nSELECT 1;\n```",
            "```\nSELECT 1;\n```",
            "```sql\nSELECT 1;",  # unbalanced fence
            "SELECT 1;",
            "   SELECT 1;   ",
        ],
    )
    def test_strips_fences_and_whitespace(self, raw):
        assert clean_sql(raw) == "SELECT 1;"

    def test_rejects_non_string(self):
        with pytest.raises(ValueError, match="must be a string"):
            clean_sql(None)

    def test_is_idempotent(self):
        once = clean_sql("```sql\nSELECT 1;\n```")
        assert clean_sql(once) == once


class TestMasking:
    """String literals, identifiers and comments are blanked before inspection."""

    @pytest.mark.parametrize(
        "sql,expected",
        [
            ("SELECT 'a;b' FROM t;", "SELECT '' FROM t;"),
            ("SELECT 'drop table' FROM t;", "SELECT '' FROM t;"),
            ('SELECT "col" FROM t;', 'SELECT "" FROM t;'),
            ("SELECT 'it''s' FROM t;", "SELECT '' FROM t;"),
        ],
    )
    def test_literals_are_masked(self, sql, expected):
        assert mask_literals_and_comments(sql) == expected

    def test_line_comment_removed(self):
        assert "DROP" not in mask_literals_and_comments("SELECT 1 -- DROP TABLE t\n;")

    def test_block_comment_removed(self):
        assert "DROP" not in mask_literals_and_comments("SELECT /* DROP */ 1;")

    def test_structure_outside_literals_survives(self):
        masked = mask_literals_and_comments("SELECT 1; DROP TABLE t;")
        assert masked.count(";") == 2
        assert "DROP" in masked


class TestValidateAccepts:
    """Legitimate read-only analytics queries must pass."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT region FROM opportunities;",
            "SELECT opportunity_id, amount FROM opportunities ORDER BY amount DESC LIMIT 5;",
            "WITH x AS (SELECT amount FROM opportunities) SELECT SUM(amount) FROM x;",
            "SELECT * FROM opportunities WHERE owner = 'D. Patel';",
            'SELECT "amount" FROM opportunities;',
            "SELECT notes FROM opportunities WHERE notes = 'it''s fine';",
        ],
    )
    def test_accepts(self, sql):
        assert validate_sql(sql) == sql

    @pytest.mark.parametrize(
        "sql",
        [
            # Regression: banned words inside a literal are data, not statements.
            "SELECT notes FROM opportunities WHERE notes ILIKE '%delete%';",
            "SELECT notes FROM opportunities WHERE notes ILIKE '%update%';",
            "SELECT notes FROM opportunities WHERE notes ILIKE '%drop table%';",
            # Regression: a semicolon inside a literal is not a separator.
            "SELECT notes FROM opportunities WHERE notes LIKE '%;%';",
        ],
    )
    def test_accepts_banned_words_inside_literals(self, sql):
        """VAL-008 / VAL-009: literal content must not trigger the keyword scan."""
        assert validate_sql(sql) == sql


class TestValidateRejects:
    """Anything that is not a single read-only statement must be refused."""

    @pytest.mark.parametrize(
        "sql,message",
        [
            ("", "empty"),
            ("SELECT region FROM opportunities", "exactly one statement"),
            ("SELECT 1; SELECT 2;", "exactly one statement"),
            ("DROP TABLE opportunities;", "disallowed statement"),
            ("INSERT INTO opportunities VALUES (1);", "disallowed statement"),
            ("UPDATE opportunities SET amount = 0;", "disallowed statement"),
            ("DELETE FROM opportunities;", "disallowed statement"),
            ("ALTER TABLE opportunities ADD COLUMN x INT;", "disallowed statement"),
            ("CREATE TABLE t AS SELECT 1;", "disallowed statement"),
            ("TRUNCATE opportunities;", "disallowed statement"),
            ("ATTACH 'evil.db' AS evil;", "disallowed statement"),
            ("COPY (SELECT 1) TO 'out.csv';", "disallowed statement"),
            ("INSTALL httpfs;", "disallowed statement"),
            ("PRAGMA database_list;", "disallowed statement"),
            ("EXPLAIN SELECT region FROM opportunities;", "start with SELECT or WITH"),
        ],
    )
    def test_rejects(self, sql, message):
        with pytest.raises(ValueError, match=message):
            validate_sql(sql)

    def test_rejects_statement_stacked_behind_a_comment(self):
        with pytest.raises(ValueError):
            validate_sql("SELECT 1 --x\n; DROP TABLE opportunities;")

    def test_unterminated_literal_fails_closed(self):
        """An unclosed quote must be refused, never silently accepted."""
        with pytest.raises(ValueError):
            validate_sql("SELECT * FROM opportunities WHERE notes = 'oops;")

    def test_error_messages_are_stable(self):
        """TESTING.md documents these strings; changing them breaks the doc."""
        with pytest.raises(ValueError, match=r"^Generated SQL is empty\.$"):
            validate_sql("")
        with pytest.raises(ValueError, match=r"^Generated SQL contains a disallowed statement\.$"):
            validate_sql("DROP TABLE opportunities;")
        with pytest.raises(ValueError, match=r"^Generated SQL must start with SELECT or WITH\.$"):
            validate_sql("EXPLAIN SELECT 1;")


class TestKeywordList:
    """The keyword tuple is the single source of truth for the regex."""

    @pytest.mark.parametrize("keyword", FORBIDDEN_KEYWORDS)
    def test_every_declared_keyword_is_actually_blocked(self, keyword):
        with pytest.raises(ValueError, match="disallowed statement"):
            validate_cleaned_sql(f"SELECT 1 {keyword} 2;")

    def test_allowed_prefixes(self):
        assert ALLOWED_PREFIXES == ("SELECT", "WITH")


class TestIsRefusal:
    """A refusal must survive harmless punctuation and quoting."""

    @pytest.mark.parametrize(
        "text",
        [
            "INVALID_QUESTION",
            "INVALID_QUESTION;",
            "INVALID_QUESTION.",
            "INVALID_QUESTION!",
            "  INVALID_QUESTION  ",
            '"INVALID_QUESTION"',
            "'INVALID_QUESTION'",
            "`INVALID_QUESTION`",
            "invalid_question",
            "INVALID_QUESTION\n",
        ],
    )
    def test_recognises_refusal(self, text):
        assert is_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "SELECT 1;",
            "",
            "INVALID",
            "THE ANSWER IS INVALID_QUESTION",
            "SELECT 'INVALID_QUESTION';",
        ],
    )
    def test_does_not_over_match(self, text):
        assert is_refusal(text) is False

    def test_non_string_is_not_a_refusal(self):
        assert is_refusal(None) is False

    def test_sentinel_value(self):
        assert INVALID_QUESTION == "INVALID_QUESTION"
