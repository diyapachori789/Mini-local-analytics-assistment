"""Question normalisation and input validation in llm.generate_sql.

None of these tests contact the Groq API: every case is rejected before the
request would be made, or exercises a pure helper.
"""

from __future__ import annotations

import pytest

import llm
from llm import normalize_question

NBSP = " "
ZWSP = "​"
BOM = "﻿"


class TestNormalizeQuestion:
    """Invisible characters must not silently corrupt a question."""

    def test_plain_text_is_unchanged(self):
        assert normalize_question("show all deals") == "show all deals"

    @pytest.mark.parametrize(
        "space",
        [
            " ",  # no-break space
            " ",  # figure space
            " ",  # thin space
            " ",  # narrow no-break space
            "　",  # ideographic space
        ],
    )
    def test_space_like_characters_become_real_spaces(self, space):
        """Regression: deleting these joined words into 'showalldeals'."""
        assert normalize_question(f"show{space}all{space}deals") == "show all deals"

    @pytest.mark.parametrize("zero_width", [BOM, ZWSP, "‌", "‍", "⁠"])
    def test_zero_width_characters_are_removed(self, zero_width):
        assert normalize_question(f"{zero_width}show all deals") == "show all deals"

    def test_zero_width_inside_a_word_is_removed_not_spaced(self):
        assert normalize_question(f"show{ZWSP}all deals") == "showall deals"

    def test_surrounding_whitespace_is_trimmed(self):
        assert normalize_question("  show all deals \n") == "show all deals"

    @pytest.mark.parametrize("blank", ["", "   ", BOM, NBSP, f"{BOM}{NBSP}  "])
    def test_effectively_blank_input_normalises_to_empty(self, blank):
        assert normalize_question(blank) == ""


class TestGenerateSqlInputValidation:
    """Bad input is rejected locally, before any API call."""

    @pytest.mark.parametrize("value", [None, 42, [], {}])
    def test_rejects_non_string(self, value):
        with pytest.raises(ValueError, match="Question must be a string"):
            llm.generate_sql(value)

    @pytest.mark.parametrize(
        "blank",
        ["", "   ", "\t\n", BOM, NBSP, f"{BOM}   ", f"{NBSP}{ZWSP}"],
    )
    def test_rejects_blank_question(self, blank):
        """A byte-order mark from piped input must not reach the API."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            llm.generate_sql(blank)

    def test_blank_question_makes_no_api_call(self, monkeypatch):
        """Guard the cost path: no client should ever be constructed."""

        def explode():
            raise AssertionError("API client was created for a blank question")

        monkeypatch.setattr(llm, "_get_client", explode)
        with pytest.raises(ValueError, match="Question cannot be empty"):
            llm.generate_sql(BOM)


class TestReExports:
    """llm re-exports the guard helpers; existing imports must keep working."""

    def test_helpers_are_importable_from_llm(self):
        assert llm.clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1;"
        assert llm.validate_sql("SELECT 1;") == "SELECT 1;"
        assert llm.is_refusal("INVALID_QUESTION;") is True
        assert llm.INVALID_QUESTION == "INVALID_QUESTION"
