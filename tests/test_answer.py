"""Phase 5: natural-language answer generation.

The Groq client is mocked throughout, so this file makes no API calls and
consumes no tokens. Live end-to-end coverage lives in test_generation_live.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

import llm
from config import ANSWER_MAX_ROWS, NO_DATA_ANSWER
from database import QueryResult


# --- Mock Groq client ------------------------------------------------------


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = FakeMessage(content)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content, *, no_choices=False, finish_reason="stop"):
        self.choices = [] if no_choices else [FakeChoice(content, finish_reason)]


class FakeCompletions:
    def __init__(self, content="An answer.", error=None, no_choices=False, finish_reason="stop"):
        self.content = content
        self.error = error
        self.no_choices = no_choices
        self.finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeResponse(
            self.content, no_choices=self.no_choices, finish_reason=self.finish_reason
        )


class FakeClient:
    def __init__(self, content="An answer.", error=None, no_choices=False, finish_reason="stop"):
        self.completions = FakeCompletions(content, error, no_choices, finish_reason)
        self.chat = type("Chat", (), {"completions": self.completions})()


@pytest.fixture
def fake_client(monkeypatch):
    """Install a mock Groq client and hand it back for inspection."""

    def install(content="An answer.", error=None, no_choices=False, finish_reason="stop"):
        client = FakeClient(content, error, no_choices, finish_reason)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        return client

    return install


@pytest.fixture
def no_client(monkeypatch):
    """Fail loudly if any API call is attempted."""

    def explode():
        raise AssertionError("an API call was made when none was expected")

    monkeypatch.setattr(llm, "_get_client", explode)


def make_result(frame, *, truncated=False, max_rows=1000, sql="SELECT 1;") -> QueryResult:
    return QueryResult(
        frame=frame,
        sql=sql,
        row_count=len(frame),
        truncated=truncated,
        max_rows=max_rows,
    )


SINGLE_NUMBER = make_result(pd.DataFrame({"total_amount": [5329008]}))
SINGLE_TEXT = make_result(pd.DataFrame({"region": ["LATAM"]}))
GROUPED = make_result(
    pd.DataFrame(
        {"region": ["NA", "LATAM", "EMEA", "APAC"], "total": [1801861, 1420697, 1266612, 839838]}
    )
)
EMPTY = make_result(pd.DataFrame(columns=["opportunity_id", "region"]))


# --- Result serialisation --------------------------------------------------


class TestFormatResultForAnswer:
    def test_single_numeric_result(self):
        text = llm.format_result_for_answer(SINGLE_NUMBER)
        assert "Columns: total_amount" in text
        assert "Total rows returned by the query: 1" in text
        assert "5329008" in text

    def test_single_text_result(self):
        text = llm.format_result_for_answer(SINGLE_TEXT)
        assert "LATAM" in text

    def test_multiple_rows_are_all_included(self):
        text = llm.format_result_for_answer(GROUPED)
        for region in ("NA", "LATAM", "EMEA", "APAC"):
            assert region in text
        assert "Total rows returned by the query: 4" in text

    def test_numbers_are_not_reformatted(self):
        """The model must receive exact values, not pre-formatted ones."""
        text = llm.format_result_for_answer(SINGLE_NUMBER)
        assert "5329008" in text
        assert "5,329,008" not in text

    def test_truncation_is_flagged(self):
        result = make_result(pd.DataFrame({"x": range(5)}), truncated=True, max_rows=5)
        text = llm.format_result_for_answer(result)
        assert "PARTIAL" in text
        assert "do not cover the whole dataset" in text

    def test_row_budget_is_applied_and_disclosed(self):
        big = make_result(pd.DataFrame({"x": range(ANSWER_MAX_ROWS + 25)}))
        text = llm.format_result_for_answer(big)
        assert f"Total rows returned by the query: {ANSWER_MAX_ROWS + 25}" in text
        assert f"only the first {ANSWER_MAX_ROWS}" in text
        # One header line, one column line, plus at most ANSWER_MAX_ROWS data rows.
        data_lines = text.split("Rows:\n")[1].splitlines()
        assert len(data_lines) == ANSWER_MAX_ROWS + 1

    def test_nulls_are_unambiguous(self):
        result = make_result(pd.DataFrame({"region": ["NA", None], "total": [1, None]}))
        text = llm.format_result_for_answer(result)
        assert "NULL" in text
        assert "nan" not in text.lower()

    def test_no_schema_or_sql_is_leaked(self):
        text = llm.format_result_for_answer(GROUPED)
        assert "SELECT" not in text.upper()
        assert "opportunities" not in text


# --- Answer generation -----------------------------------------------------


class TestGenerateAnswer:
    def test_single_numeric_answer(self, fake_client):
        fake_client("The total amount of closed-won opportunities is 5,329,008.")
        answer = llm.generate_answer("What is the total closed won amount?", SINGLE_NUMBER)
        assert answer == "The total amount of closed-won opportunities is 5,329,008."

    def test_single_text_answer(self, fake_client):
        fake_client("LATAM has the highest total opportunity amount.")
        answer = llm.generate_answer("Which region leads?", SINGLE_TEXT)
        assert "LATAM" in answer

    def test_multiple_rows_answer(self, fake_client):
        fake_client("NA leads with 1,801,861, followed by LATAM.")
        answer = llm.generate_answer("Totals by region?", GROUPED)
        assert "NA leads" in answer

    def test_question_is_sent_for_context(self, fake_client):
        client = fake_client()
        llm.generate_answer("Which region leads?", SINGLE_TEXT)
        user_message = client.completions.calls[0]["messages"][1]["content"]
        assert "Which region leads?" in user_message

    def test_result_is_sent(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        user_message = client.completions.calls[0]["messages"][1]["content"]
        assert "5329008" in user_message

    def test_uses_the_configured_model_and_bounds(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        call = client.completions.calls[0]
        assert call["model"] == llm.MODEL_NAME
        assert call["temperature"] == 0
        assert call["max_completion_tokens"] > 0

    def test_answer_is_stripped(self, fake_client):
        fake_client("  The total is 5,329,008.\n\n")
        assert llm.generate_answer("Total?", SINGLE_NUMBER) == "The total is 5,329,008."

    def test_system_prompt_forbids_sql_and_invention(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        system = client.completions.calls[0]["messages"][0]["content"]
        assert "Never output SQL" in system
        assert "Never invent" in system
        assert "Use ONLY" in system

    def test_schema_is_never_sent(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        sent = " ".join(
            message["content"] for message in client.completions.calls[0]["messages"]
        )
        assert "| column | type |" not in sent


class TestEmptyResult:
    def test_zero_rows_answered_without_an_api_call(self, no_client):
        """No data means nothing to summarise, so no request is made."""
        assert llm.generate_answer("Any deals in Antarctica?", EMPTY) == NO_DATA_ANSWER

    def test_no_data_message_is_the_configured_one(self):
        assert NO_DATA_ANSWER == "No matching records were found."


class TestTruncatedResult:
    def test_partial_flag_reaches_the_model(self, fake_client):
        client = fake_client("Across the rows shown, the largest is 251,867.")
        truncated = make_result(
            pd.DataFrame({"amount": [251867, 251832]}), truncated=True, max_rows=2
        )
        llm.generate_answer("Biggest deals?", truncated)
        user_message = client.completions.calls[0]["messages"][1]["content"]
        assert "PARTIAL" in user_message

    def test_prompt_forbids_implying_completeness(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        system = client.completions.calls[0]["messages"][0]["content"]
        assert "Never imply a partial result is the complete dataset" in system


class TestInvalidInput:
    @pytest.mark.parametrize("question", [None, 42, [], {}])
    def test_rejects_non_string_question(self, no_client, question):
        with pytest.raises(ValueError, match="Question must be a string"):
            llm.generate_answer(question, SINGLE_NUMBER)

    @pytest.mark.parametrize("question", ["", "   ", "﻿"])
    def test_rejects_blank_question(self, no_client, question):
        with pytest.raises(ValueError, match="Question cannot be empty"):
            llm.generate_answer(question, SINGLE_NUMBER)

    @pytest.mark.parametrize("result", [None, "not a result", 42, {"rows": []}])
    def test_rejects_invalid_result_structure(self, no_client, result):
        with pytest.raises(ValueError, match="QueryResult is required"):
            llm.generate_answer("Total?", result)


class TestApiFailures:
    def test_generic_api_failure_raises_runtime_error(self, fake_client):
        fake_client(error=Exception("connection reset"))
        with pytest.raises(RuntimeError, match="Answer generation failed"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_rate_limit_is_reported_clearly(self, fake_client):
        fake_client(
            error=Exception(
                "Error code: 429 - {'error': {'message': 'Rate limit reached', "
                "'code': 'rate_limit_exceeded'}}"
            )
        )
        with pytest.raises(RuntimeError, match="rate limit"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_rate_limit_detected_by_status_code(self, fake_client):
        error = Exception("something went wrong")
        error.status_code = 429
        fake_client(error=error)
        with pytest.raises(RuntimeError, match="rate limit"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_empty_content_is_rejected(self, fake_client):
        fake_client(content="   ")
        with pytest.raises(RuntimeError, match="empty answer"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_none_content_is_rejected(self, fake_client):
        fake_client(content=None)
        with pytest.raises(RuntimeError, match="empty answer"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_missing_choices_is_rejected(self, fake_client):
        fake_client(no_choices=True)
        with pytest.raises(RuntimeError, match="invalid response"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_rate_limit_helper(self):
        assert llm._is_rate_limit_error(Exception("Error code: 429")) is True
        assert llm._is_rate_limit_error(Exception("rate limit reached")) is True
        assert llm._is_rate_limit_error(Exception("connection reset")) is False


class TestTruncatedAnswer:
    """A half-finished sentence must not be presented as a complete answer."""

    def test_truncated_answer_is_flagged(self, fake_client):
        fake_client("The totals are: Atlas 604551, Cobalt 335", finish_reason="length")
        answer = llm.generate_answer("Totals per account?", GROUPED)
        assert "[Answer truncated" in answer

    def test_complete_answer_is_not_flagged(self, fake_client):
        fake_client("The total is 5,329,008.", finish_reason="stop")
        answer = llm.generate_answer("Total?", SINGLE_NUMBER)
        assert "truncated" not in answer.lower()

    def test_truncation_is_logged(self, fake_client, caplog):
        fake_client("cut off here", finish_reason="length")
        with caplog.at_level("WARNING", logger="llm"):
            llm.generate_answer("Total?", SINGLE_NUMBER)
        assert any("cut short" in record.getMessage() for record in caplog.records)


class TestGroundingRules:
    """The prompt must forbid the failure modes seen in live testing."""

    def test_prompt_forbids_derived_arithmetic(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        system = client.completions.calls[0]["messages"][0]["content"]
        assert "Do not calculate new numbers" in system

    def test_prompt_forbids_inventing_units(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        system = client.completions.calls[0]["messages"][0]["content"]
        assert "never '38%'" in system

    def test_prompt_forbids_self_ranking(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        system = client.completions.calls[0]["messages"][0]["content"]
        assert "Do not rank the rows yourself" in system

    def test_prompt_requires_decimal_precision(self, fake_client):
        client = fake_client()
        llm.generate_answer("Total?", SINGLE_NUMBER)
        system = client.completions.calls[0]["messages"][0]["content"]
        assert "24.64%, never 25%" in system


class TestAnswerMustNotBeSql:
    """An answer that is really a query is worse than no answer."""

    @pytest.mark.parametrize(
        "content",
        [
            "SELECT SUM(amount) FROM opportunities;",
            "WITH x AS (SELECT 1) SELECT * FROM x;",
            "```sql\nSELECT 1;\n```",
            "Here you go:\n```\nSELECT 1;\n```",
        ],
    )
    def test_sql_shaped_answers_are_rejected(self, fake_client, content):
        fake_client(content=content)
        with pytest.raises(RuntimeError, match="produced SQL"):
            llm.generate_answer("Total?", SINGLE_NUMBER)

    def test_prose_mentioning_select_is_allowed(self, fake_client):
        """The guard must not fire on ordinary English."""
        fake_client(content="A select group of four regions contributed the total.")
        answer = llm.generate_answer("Total?", SINGLE_NUMBER)
        assert answer.startswith("A select group")


class TestLogging:
    def test_start_and_success_are_logged_with_row_count(self, fake_client, caplog):
        fake_client("The total is 5,329,008.")
        with caplog.at_level("INFO", logger="llm"):
            llm.generate_answer("Total?", GROUPED)
        messages = [record.getMessage() for record in caplog.records]
        assert any("Answer generation started" in message for message in messages)
        assert any("rows_provided=4" in message for message in messages)
        assert any("Answer generation succeeded" in message for message in messages)

    def test_failure_is_logged(self, fake_client, caplog):
        fake_client(error=Exception("boom"))
        with caplog.at_level("ERROR", logger="llm"):
            with pytest.raises(RuntimeError):
                llm.generate_answer("Total?", SINGLE_NUMBER)
        assert any(
            "Answer generation failed" in record.getMessage() for record in caplog.records
        )

    def test_empty_result_path_is_logged(self, no_client, caplog):
        with caplog.at_level("INFO", logger="llm"):
            llm.generate_answer("Anything?", EMPTY)
        assert any(
            "without an API call" in record.getMessage() for record in caplog.records
        )

    def test_no_secret_in_log_messages(self, fake_client, caplog):
        fake_client("The total is 5,329,008.")
        with caplog.at_level("DEBUG", logger="llm"):
            llm.generate_answer("Total?", SINGLE_NUMBER)
        from config import GROQ_API_KEY

        if GROQ_API_KEY:
            for record in caplog.records:
                assert GROQ_API_KEY not in record.getMessage()
