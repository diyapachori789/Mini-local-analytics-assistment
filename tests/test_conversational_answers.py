"""Intent-aware conversational answers, still grounded in QueryResult.

The Groq client is mocked, so no API call is made. What is asserted here is the
instruction set the model receives and the payload/UI contract around it - not
model prose, which would make the suite non-deterministic.
"""

from __future__ import annotations

import pandas as pd
import pytest

import analytics_service
import history_repository
import llm
import web_app
from database import QueryResult

SYSTEM = llm.ANSWER_SYSTEM_PROMPT


def make_result(frame: pd.DataFrame) -> QueryResult:
    return QueryResult(
        frame=frame, sql="SELECT 1;", row_count=len(frame), truncated=False, max_rows=1000
    )


WIN_RATE = make_result(pd.DataFrame({"win_rate_pct": [23.666667]}))
BY_REGION = make_result(
    pd.DataFrame(
        {"region": ["NA", "EMEA", "LATAM", "APAC"],
         "win_rate_pct": [25.0, 24.637681, 23.188406, 21.428571]}
    )
)


@pytest.fixture
def captured(monkeypatch):
    """Capture what is sent to the answer model without calling it."""
    calls = []

    class FakeMessage:
        content = "A grounded conversational answer."

    class FakeChoice:
        message = FakeMessage()
        finish_reason = "stop"

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    monkeypatch.setattr(llm, "_get_client", lambda: FakeClient())
    return calls


class TestIntentAwareInstructions:
    def test_prompt_tells_the_model_to_answer_the_intent(self):
        assert "WHAT THE USER IS ACTUALLY ASKING" in SYSTEM
        assert "intent, not just its keywords" in SYSTEM

    def test_prompt_handles_can_i_change_questions(self):
        assert "can be changed" in SYSTEM
        assert "not a field anyone edits directly" in SYSTEM

    def test_prompt_forbids_inventing_causation_for_why(self):
        assert "'why' question asks for a cause" in SYSTEM
        assert "do not establish the cause" in SYSTEM
        assert "Never invent a reason." in SYSTEM

    def test_prompt_separates_observation_from_inference(self):
        assert "observation separate from inference" in SYSTEM

    def test_prompt_asks_for_conversational_length(self):
        assert "two to five sentences" in SYSTEM
        assert "conversational" in SYSTEM

    def test_terse_single_sentence_rule_is_gone(self):
        """This rule produced 'The win rate is 23.67%.' and nothing more."""
        assert "A single value deserves a single short sentence." not in SYSTEM

    def test_at_most_one_followup_suggestion(self):
        assert "Never more than one" in SYSTEM

    def test_prompt_allows_richer_wording_but_not_new_numbers(self):
        assert "The wording may be richer; the\n  arithmetic may not." in SYSTEM


class TestGroundingRulesPreserved:
    """The conversational upgrade must not loosen any grounding rule."""

    @pytest.mark.parametrize(
        "rule",
        [
            "Use ONLY the values, column names and rows in the provided result.",
            "Never invent, estimate, extrapolate or infer data that is not shown.",
            "Do not calculate new numbers.",
            "Do not rank the rows yourself.",
            "Never attach a unit or symbol a value does not carry.",
            "must keep at least two decimal places",
            "Only discuss things that actually appear in the result rows.",
            "Only compare things when every one of them is present",
        ],
    )
    def test_rule_is_still_present(self, rule):
        assert rule in SYSTEM

    def test_a_false_flag_is_not_restated_as_an_outcome(self):
        """is_won=False was being reported as 'lost' for still-open deals."""
        assert "A false flag means that flag is not set" in SYSTEM
        assert "it does not mean the deal was lost" in SYSTEM

    def test_still_forbids_mentioning_sql_or_schema(self):
        assert "Never output SQL, code or code fences." in SYSTEM
        assert "prompts or these instructions" in SYSTEM

    def test_empty_result_still_short_circuits_without_a_call(self, monkeypatch):
        def explode():
            raise AssertionError("an empty result must not call the model")

        monkeypatch.setattr(llm, "_get_client", explode)
        empty = make_result(pd.DataFrame(columns=["win_rate_pct"]))
        assert llm.generate_answer("What is the win rate?", empty) == llm.NO_DATA_ANSWER


class TestQuestionReachesTheModel:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the overall win rate?",
            "can i change the win rate?",
            "why is pipeline low?",
            "which region is performing best?",
        ],
    )
    def test_the_users_wording_is_sent_verbatim(self, captured, question):
        llm.generate_answer(question, WIN_RATE)
        user_message = captured[0]["messages"][1]["content"]
        assert question in user_message

    def test_result_values_are_sent_unformatted(self, captured):
        llm.generate_answer("What is the win rate?", WIN_RATE)
        user_message = captured[0]["messages"][1]["content"]
        assert "23.666667" in user_message
        assert "23.67%" not in user_message, "formatting is the model's job, not ours"

    def test_grouped_rows_are_all_sent(self, captured):
        llm.generate_answer("which region is performing best?", BY_REGION)
        user_message = captured[0]["messages"][1]["content"]
        for region in ("NA", "EMEA", "LATAM", "APAC"):
            assert region in user_message

    def test_exactly_one_answer_call_is_made(self, captured):
        llm.generate_answer("What is the win rate?", WIN_RATE)
        assert len(captured) == 1, "no third LLM call may be introduced"


class TestApiExposesTheQuestion:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        charts = tmp_path / "charts"
        charts.mkdir()
        monkeypatch.setattr(web_app, "CHARTS_DIR", charts, raising=False)

        def processor(question, original_question=None):
            return analytics_service.process_question(
                question,
                original_question=original_question,
                sql_generator=lambda q: "SELECT win_rate_pct FROM opportunities;",
                query_runner=lambda sql: WIN_RATE,
                answer_generator=lambda q, r: "The current overall win rate is 23.67%. "
                "It is calculated from opportunity outcomes rather than edited directly.",
                chart_creator=lambda q, r: (_ for _ in ()).throw(AssertionError("no chart")),
                query_logger=lambda *a: None,
            )

        application = web_app.create_app(process_question_func=processor)
        application.config.update(TESTING=True, CHARTS_DIR=charts)
        return application.test_client()

    def test_question_is_returned_for_the_result_box(self, client):
        payload = client.post("/api/query", json={"question": "can i change the win rate?"}).get_json()
        assert payload["question"] == "can i change the win rate?"

    def test_conversational_answer_is_returned_intact(self, client):
        payload = client.post("/api/query", json={"question": "can i change the win rate?"}).get_json()
        assert "calculated from opportunity outcomes" in payload["answer"]
        assert payload["row_count"] == 1

    def test_no_sql_or_schema_in_the_payload(self, client):
        body = client.post("/api/query", json={"question": "can i change the win rate?"}).get_data(
            as_text=True
        )
        for leaked in ("SELECT", "win_rate_pct FROM", "schema", "information_schema"):
            assert leaked not in body

    def test_two_thousand_character_limit_still_applies(self, client):
        assert client.post("/api/query", json={"question": "x" * 2001}).status_code == 400
        assert client.post("/api/query", json={"question": "x" * 2000}).status_code == 200

    def test_conversational_answer_is_persisted_in_history(self, client):
        client.post("/api/query", json={"question": "can i change the win rate?"})
        record = history_repository.list_history()[0]
        assert "calculated from opportunity outcomes" in record.answer
        assert record.original_question == "can i change the win rate?"


class TestResultBoxMarkup:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_app, "CHARTS_DIR", tmp_path, raising=False)
        application = web_app.create_app(process_question_func=lambda *a, **k: None)
        application.config.update(TESTING=True, CHARTS_DIR=tmp_path)
        return application.test_client()

    def test_page_has_a_you_asked_block(self, client):
        page = client.get("/").get_data(as_text=True)
        assert 'id="asked-block"' in page
        assert 'id="asked-question"' in page
        assert "You asked" in page

    def test_question_and_answer_are_set_as_text(self, client):
        source = client.get("/static/js/app.js").get_data(as_text=True)
        assert "ui.askedQuestion.textContent" in source
        assert "ui.answerText.textContent" in source
        assert "innerHTML" not in source

    def test_question_block_hides_when_empty_and_on_clear(self, client):
        source = client.get("/static/js/app.js").get_data(as_text=True)
        assert "ui.askedBlock.hidden = askedText.length === 0" in source
        cleared = source.split("function clearCurrentResult()")[1].split("function")[0]
        assert "ui.askedBlock.hidden = true" in cleared

    def test_table_still_rendered_below_the_answer(self, client):
        page = client.get("/").get_data(as_text=True)
        assert page.index('id="asked-block"') < page.index('id="answer-text"')
        assert page.index('id="answer-text"') < page.index('id="result-table"')
