"""Friendly refusal replies: classification, determinism and safety.

Pure functions plus the service path. No Groq calls, no database writes.
"""

from __future__ import annotations

import pytest

import refusal
from refusal import RefusalCategory, classify_refusal, refusal_message


class TestMetadataCategory:
    @pytest.mark.parametrize(
        "question",
        [
            "Show all tables",
            "show tables",
            "List all tables",
            "Describe opportunities table",
            "describe the table",
            "Show schema",
            "what is the schema?",
            "What columns are in the database?",
            "which fields exist",
            "column names please",
            "what data types does it use",
            "explain the database structure",
            "SELECT * FROM information_schema.tables",
        ],
    )
    def test_structure_questions_are_metadata(self, question):
        assert classify_refusal(question) is RefusalCategory.METADATA

    def test_reply_redirects_to_business_questions(self):
        message = refusal_message("Show all tables")
        assert message in refusal._TEMPLATES[RefusalCategory.METADATA]
        # Must not leak the thing it is declining to reveal.
        for leaked in ("opportunity_id", "VARCHAR", "CREATE TABLE", "information_schema"):
            assert leaked not in message


class TestUnsafeSqlCategory:
    @pytest.mark.parametrize(
        "question",
        [
            "DROP TABLE opportunities",
            "DELETE FROM opportunities",
            "UPDATE opportunities SET amount = 0",
            "INSERT INTO opportunities VALUES (1)",
            "ATTACH database evil.db",
            "COPY opportunities TO 'out.csv'",
            "truncate the table",
            "please alter the table",
            "run PRAGMA database_list",
        ],
    )
    def test_admin_commands_are_unsafe_sql(self, question):
        assert classify_refusal(question) is RefusalCategory.UNSAFE_SQL

    def test_reply_offers_a_safe_alternative(self):
        message = refusal_message("DROP TABLE opportunities")
        assert message in refusal._TEMPLATES[RefusalCategory.UNSAFE_SQL]
        assert "read-only" in message or "safely" in message or "won't run" in message


class TestOutOfScopeCategory:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the weather today?",
            "Tell me a joke",
            "Who is the president?",
            "What is the capital of France?",
            "recommend a movie",
            "what is the bitcoin price",
            "who are you",
            "translate this to French",
        ],
    )
    def test_unrelated_questions_are_out_of_scope(self, question):
        assert classify_refusal(question) is RefusalCategory.OUT_OF_SCOPE

    def test_question_with_no_domain_words_is_out_of_scope(self):
        assert classify_refusal("purple monkey dishwasher") is RefusalCategory.OUT_OF_SCOPE

    def test_reply_names_the_domain(self):
        message = refusal_message("Tell me a joke")
        assert message in refusal._TEMPLATES[RefusalCategory.OUT_OF_SCOPE]


class TestUnsupportedCategory:
    @pytest.mark.parametrize(
        "question",
        [
            "opportunities ??? maybe",
            "pipeline the of by",
            "deals asdkjh",
            "show me opportunities in a way that cannot be computed",
        ],
    )
    def test_domain_questions_that_do_not_parse_are_unsupported(self, question):
        assert classify_refusal(question) is RefusalCategory.UNSUPPORTED

    @pytest.mark.parametrize("question", ["", "   ", None, 42])
    def test_empty_or_invalid_input_is_unsupported(self, question):
        assert classify_refusal(question) is RefusalCategory.UNSUPPORTED

    def test_single_template_is_used(self):
        assert len(refusal._TEMPLATES[RefusalCategory.UNSUPPORTED]) == 1
        assert refusal_message("deals asdkjh") == refusal._TEMPLATES[RefusalCategory.UNSUPPORTED][0]


class TestPriorityOrder:
    def test_metadata_beats_domain_words(self):
        """'describe the opportunities table' is about structure, not sales."""
        assert classify_refusal("describe the opportunities table") is RefusalCategory.METADATA

    def test_metadata_beats_unsafe_keyword(self):
        """'show schema' style questions are answered as metadata, not as admin SQL."""
        assert classify_refusal("show me the table schema") is RefusalCategory.METADATA

    def test_unsafe_beats_out_of_scope(self):
        assert classify_refusal("drop the weather table") is RefusalCategory.UNSAFE_SQL


class TestDeterminism:
    @pytest.mark.parametrize(
        "question",
        ["Show all tables", "DROP TABLE opportunities", "Tell me a joke", "deals asdkjh"],
    )
    def test_same_question_always_gives_the_same_reply(self, question):
        first = refusal_message(question)
        for _ in range(25):
            assert refusal_message(question) == first

    def test_selection_does_not_use_random(self):
        """A random pick would make these tests flake."""
        source = (refusal.__file__ or "")
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "import random" not in text
        assert "random." not in text

    def test_whitespace_and_case_do_not_change_the_reply(self):
        assert refusal_message("Show All Tables") == refusal_message("  show   all tables  ")

    def test_different_questions_can_select_different_templates(self):
        """Variation within a category, still deterministic."""
        metadata_questions = [
            "Show all tables",
            "Describe opportunities table",
            "What columns are in the database?",
            "show schema",
            "list all tables",
            "what fields exist",
            "table structure",
            "database metadata",
        ]
        chosen = {refusal_message(q) for q in metadata_questions}
        assert len(chosen) > 1, "all metadata questions produced an identical reply"
        assert chosen <= set(refusal._TEMPLATES[RefusalCategory.METADATA])


class TestReplyQuality:
    @pytest.mark.parametrize("category", list(RefusalCategory))
    def test_every_template_suggests_what_to_ask(self, category):
        """A refusal should always point at something that would work."""
        guidance = ("try", "ask", "i can help", "i can safely", "i'm here to help")
        for template in refusal._TEMPLATES[category]:
            assert len(template) > 60
            lowered = template.lower()
            assert any(phrase in lowered for phrase in guidance), template

    @pytest.mark.parametrize("category", list(RefusalCategory))
    def test_templates_expose_nothing_technical(self, category):
        for template in refusal._TEMPLATES[category]:
            lowered = template.lower()
            for leaked in ("select ", "duckdb", "groq", "traceback", "sql_guard",
                           "c:\\", "information_schema", "exception", "error code"):
                assert leaked not in lowered

    def test_every_category_has_at_least_one_reply(self):
        for category in RefusalCategory:
            assert refusal._TEMPLATES.get(category), category

    def test_total_template_count(self):
        """Ten replies, plus two for the follow-up case intent routing added."""
        total = sum(len(v) for v in refusal._TEMPLATES.values())
        assert total == 12


class TestGuardBlocksMetadataSql:
    """The guard must stop metadata reaching the database at all."""

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT table_name FROM information_schema.tables;",
            "SELECT column_name FROM information_schema.columns;",
            "SELECT * FROM duckdb_tables();",
            "SELECT * FROM duckdb_columns();",
            "SELECT * FROM pragma_table_info('opportunities');",
            "SELECT * FROM sqlite_master;",
            "WITH x AS (SELECT * FROM information_schema.tables) SELECT * FROM x;",
        ],
    )
    def test_metadata_sources_are_rejected(self, sql):
        import sql_guard

        with pytest.raises(ValueError, match="metadata"):
            sql_guard.validate_sql(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT region, COUNT(*) AS n FROM opportunities GROUP BY region;",
            "SELECT * FROM opportunities WHERE owner = 'D. Patel';",
            "SELECT notes FROM opportunities WHERE notes ILIKE '%schema%';",
        ],
    )
    def test_ordinary_business_queries_still_pass(self, sql):
        import sql_guard

        assert sql_guard.validate_sql(sql) == sql
