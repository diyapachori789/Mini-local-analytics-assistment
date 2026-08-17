"""Deterministic output boundaries: schema identifiers and invented numbers.

The prompts already forbid both of these. These tests cover the part that does
not depend on the model complying, which is the part that has actually failed
in practice: a released model named a result column in prose, and volunteered a
difference it had worked out itself.

No Groq client is constructed anywhere here.
"""

from __future__ import annotations

import pandas as pd
import pytest

import analytics_service
import database
import llm
from database import QueryResult
from intent import Intent, QueryPlan, contains_schema_disclosure


@pytest.fixture(scope="module")
def identifiers(request):
    """The live column names, used the way the filter uses them."""
    database.initialize_database()
    return database.column_identifiers()


def make_result(frame: pd.DataFrame, **kwargs) -> QueryResult:
    return QueryResult(
        frame=frame,
        sql=kwargs.get("sql", "SELECT 1;"),
        row_count=len(frame),
        truncated=kwargs.get("truncated", False),
        max_rows=kwargs.get("max_rows", 1000),
    )


BY_REGION = make_result(
    pd.DataFrame({"region": ["NA", "APAC"], "win_rate_pct": [25.00, 21.40]})
)


class TestSchemaIdentifiersAreNotDisclosed:
    """Business language survives; implementation language does not."""

    @pytest.mark.parametrize(
        "answer",
        [
            "The column names are opportunity_id and amount.",
            "The columns are opportunity_id and amount.",
            "The is_won boolean field indicates whether a deal closed.",
            "The opportunity_id column stores the record identifier.",
            "columns include account_name and amount",
            "The created_date is a DATE field in the table.",
            "This figure comes from the win_rate_pct column.",
            "The fields are region, stage and owner.",
            "Each record has an amount column and a stage column.",
            "The dataset schema contains a notes attribute.",
        ],
    )
    def test_schema_disclosure_is_detected(self, answer, identifiers):
        assert contains_schema_disclosure(answer, identifiers) is True

    @pytest.mark.parametrize(
        "answer",
        [
            "The opportunity amount is $25,000.",
            "Pipeline represents active opportunities moving through the sales process.",
            "Your win rate is 23.67%.",
            "NA leads with the highest win rate, followed by EMEA.",
            "The largest deal is worth 248,177 and belongs to Acme Corp.",
            "Win rate measures how many deals were won out of those that reached a decision.",
            "The amount of revenue closed last quarter was strong.",
            "Deals in the negotiation stage are still open.",
            "Owners in the EMEA region closed more business.",
            "Closed Won means the deal was signed.",
        ],
    )
    def test_business_language_is_not_blocked(self, answer, identifiers):
        assert contains_schema_disclosure(answer, identifiers) is False

    def test_a_query_alias_is_caught_even_though_it_is_not_a_table_column(
        self, identifiers
    ):
        """win_rate_pct is invented per query, so no column list can hold it."""
        assert "win_rate_pct" not in identifiers
        assert contains_schema_disclosure("Taken from win_rate_pct.", identifiers) is True

    def test_the_filter_reads_the_live_schema_rather_than_a_copy(self, identifiers):
        assert "opportunity_id" in identifiers
        assert len(identifiers) >= 10

    def test_missing_identifiers_still_catch_pattern_based_leaks(self):
        """An unavailable schema must not disable the check entirely."""
        assert contains_schema_disclosure("The is_won field is a boolean.", frozenset())
        assert contains_schema_disclosure("The columns are a and b.", frozenset())


class TestConceptualAnswersAreSanitised:
    """The conceptual stage is the only one handed the real schema."""

    def _client(self, monkeypatch, content):
        class Fake:
            def __init__(self):
                self.chat = self

            @property
            def completions(self):
                return self

            def create(self, **kwargs):
                message = type("M", (), {"content": content, "reasoning": None})()
                choice = type("C", (), {"message": message, "finish_reason": "stop"})()
                return type("R", (), {"choices": [choice]})()

        monkeypatch.setattr(llm, "_get_client", Fake)

    def test_a_leaking_conceptual_answer_is_refused(self, monkeypatch, identifiers):
        self._client(monkeypatch, "Pipeline is derived from the stage column and is_won field.")
        with pytest.raises(RuntimeError, match="could not be returned safely"):
            llm.generate_conceptual_answer("What is pipeline?")

    def test_a_clean_conceptual_answer_is_returned_unchanged(self, monkeypatch):
        good = (
            "Pipeline represents opportunities that are still moving through the "
            "sales process and have not yet reached a final outcome."
        )
        self._client(monkeypatch, good)
        assert llm.generate_conceptual_answer("What is pipeline?") == good

    def test_a_leak_never_reaches_the_adapter(self, monkeypatch):
        """The refusal happens in llm, so no caller can forward the text."""
        def leaking(_question):
            raise RuntimeError("Conceptual answer could not be returned safely.")

        response = analytics_service.process_question(
            "What is pipeline?",
            plan_generator=lambda q: QueryPlan(intent=Intent.DATA_EXPLANATION),
            conceptual_answer_generator=leaking,
            query_runner=lambda sql: pytest.fail("no query for a conceptual answer"),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert "column" not in (response.answer or "").lower()


class TestNumericGrounding:
    """DuckDB calculates; the model explains. This checks the second half."""

    def test_values_present_in_the_result_are_allowed(self):
        answer = "NA is 25.00% and APAC is 21.40%."
        assert llm.unsupported_numbers(answer, BY_REGION) == []

    def test_a_derived_difference_is_rejected(self):
        answer = "NA is 25.00% and APAC is 21.40%. That is a difference of 3.6 points."
        assert llm.unsupported_numbers(answer, BY_REGION) == ["3.6"]

    @pytest.mark.parametrize(
        "written",
        ["25", "25.0", "25.00", "25.000"],
        ids=["integer", "one-dp", "two-dp", "three-dp"],
    )
    def test_formatting_differences_are_the_same_value(self, written):
        assert llm.unsupported_numbers(f"The rate is {written}%.", BY_REGION) == []

    def test_rounding_a_rate_to_two_decimals_is_allowed(self):
        result = make_result(pd.DataFrame({"win_rate_pct": [23.666667]}))
        assert llm.unsupported_numbers("The win rate is 23.67%.", result) == []

    def test_thousands_separators_are_allowed(self):
        result = make_result(pd.DataFrame({"total_amount": [248177]}))
        assert llm.unsupported_numbers("The total is 248,177.", result) == []

    def test_the_row_count_is_a_fact_about_the_result(self):
        assert llm.unsupported_numbers("Four regions were returned: 2 shown.", BY_REGION) == []

    def test_digits_inside_a_text_cell_may_be_repeated(self):
        result = make_result(pd.DataFrame({"opportunity_id": ["OPP-1003"], "amount": [500]}))
        assert llm.unsupported_numbers("OPP-1003 is worth 500.", result) == []

    def test_an_invented_total_is_rejected(self):
        result = make_result(pd.DataFrame({"region": ["NA", "EMEA"], "n": [80, 74]}))
        assert llm.unsupported_numbers("Together that is 154 deals.", result) == ["154"]

    def test_a_leaking_answer_is_replaced_with_a_grounded_summary(self, monkeypatch):
        class Fake:
            def __init__(self):
                self.chat = self

            @property
            def completions(self):
                return self

            def create(self, **kwargs):
                message = type("M", (), {
                    "content": "NA is 25.00% and APAC is 21.40%, a spread of 3.6 points.",
                    "reasoning": None})()
                choice = type("C", (), {"message": message, "finish_reason": "stop"})()
                return type("R", (), {"choices": [choice]})()

        monkeypatch.setattr(llm, "_get_client", Fake)
        answer = llm.generate_answer("Compare win rates by region", BY_REGION)
        assert "3.6" not in answer
        assert llm.unsupported_numbers(answer, BY_REGION) == []

    def test_the_replacement_costs_no_model_call_and_no_query(self, monkeypatch):
        """The result is already in hand, so recovery is local."""
        calls = []

        class Fake:
            def __init__(self):
                self.chat = self

            @property
            def completions(self):
                return self

            def create(self, **kwargs):
                calls.append(1)
                message = type("M", (), {
                    "content": "A spread of 3.6 points.", "reasoning": None})()
                choice = type("C", (), {"message": message, "finish_reason": "stop"})()
                return type("R", (), {"choices": [choice]})()

        monkeypatch.setattr(llm, "_get_client", Fake)
        monkeypatch.setattr(
            database, "run_query", lambda *a, **k: pytest.fail("must not re-query")
        )
        llm.generate_answer("Compare win rates by region", BY_REGION)
        assert len(calls) == 1, "no repair call may be made"

    def test_a_grounded_summary_states_only_what_the_result_holds(self):
        summary = llm.grounded_summary(BY_REGION)
        assert llm.unsupported_numbers(summary, BY_REGION) == []
        assert "3.6" not in summary


class TestSinglePersistencePath:
    """One authoritative write per browser turn."""

    def test_create_app_no_longer_accepts_a_legacy_saver(self):
        import inspect

        import web_app

        assert "history_saver" not in inspect.signature(web_app.create_app).parameters

    def test_web_app_has_no_second_persistence_function(self):
        import web_app

        assert not hasattr(web_app, "save_completed_history")

    def test_only_the_conversation_turn_is_written(self):
        source = open("web_app.py", encoding="utf-8").read()
        assert source.count("history_repository.save_conversation_turn") == 1
        assert "history_repository.save_history" not in source
        assert "legacy_history_saver" not in source

    def test_a_reply_severed_mid_number_is_replaced_not_flagged(self):
        """'14206' for 1420697 would mislead; a grounded summary will not."""
        result = make_result(pd.DataFrame({"region": ["LATAM"], "total": [1420697]}))
        assert llm.unsupported_numbers("LATAM totals 14206", result) == ["14206"]
