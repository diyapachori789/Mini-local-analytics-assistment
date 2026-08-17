"""Global intent routing: any wording, no metric-specific handling.

The first model call now returns a routing decision alongside its SQL, so a
question is no longer refused merely because it is not a direct lookup. These
tests pin the routing contract, the call budget and the safety boundary.

The model is mocked throughout, which is what makes the assertions
deterministic: what is being tested is how *this code* routes a decision, never
how the model words one. Whether the model classifies real paraphrases
consistently is a question about the model, and is measured by the live
evaluation set instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import analytics_service
import database
import intent as intent_module
import llm
import refusal
import sql_guard
import web_app
from config import CONVERSATION_CONTEXT_MAX_CHARS
from database import QueryResult, SqlValidationError
from intent import ANSWERABLE, Intent, QueryPlan, parse_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUPED_SQL = (
    "SELECT region, COUNT(*) AS opportunity_count FROM opportunities GROUP BY region;"
)


def make_result(frame: pd.DataFrame) -> QueryResult:
    return QueryResult(
        frame=frame, sql=GROUPED_SQL, row_count=len(frame), truncated=False, max_rows=1000
    )


REGION_ROWS = make_result(
    pd.DataFrame({"region": ["NA", "EMEA"], "opportunity_count": [80, 74]})
)


def plan_reply(intent: str, sql: str | None, response: str | None = None) -> str:
    """The literal JSON text the first call is expected to produce."""
    return json.dumps({"intent": intent, "sql": sql, "response": response})


class ScriptedClient:
    """A Groq stand-in that replies from a script and counts every call.

    Stage is detected from the system prompt rather than call order, so a test
    that makes them in an unexpected order still gets a coherent reply instead
    of a confusing mismatch.
    """

    def __init__(self, *, route: str, answer: str = "A grounded answer."):
        self.route = route
        self.answer = answer
        self.calls: list[str] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "routing and SQL layer" in system:
            stage, content = "route", self.route
        elif "You are Analytics Assistant" in system:
            stage, content = "conversation", self.answer
        elif "what a\nmeasure in their sales-opportunity data means" in system:
            stage, content = "conceptual", self.answer
        else:
            stage, content = "answer", self.answer
        self.calls.append(stage)

        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        return type("Response", (), {"choices": [choice]})()


def executable_source(module: str) -> str:
    """Module source with comments and string literals removed.

    What matters is whether the *logic* keys off a metric name. Prose in a
    docstring that explains the bug this design fixed is not hard coding, and a
    check that cannot tell the two apart would be measuring the wrong thing.
    """
    import tokenize

    kept: list[str] = []
    with open(PROJECT_ROOT / module, "rb") as handle:
        for token in tokenize.tokenize(handle.readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(token.string)
    return " ".join(kept).lower()


@pytest.fixture
def scripted(monkeypatch, initialized_database):
    """Install a scripted client and hand the test its call log.

    The live schema is required because stage one describes the real columns to
    the model; that is the whole reason routing generalises without a hand-kept
    list of supported metrics.
    """

    def install(route: str, answer: str = "A grounded answer.") -> ScriptedClient:
        client = ScriptedClient(route=route, answer=answer)
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        return client

    return install


# ---------------------------------------------------------------------------
# 1. Broad analytics questions route to a data query
# ---------------------------------------------------------------------------


class TestAnalyticsRouting:
    @pytest.mark.parametrize(
        "question",
        [
            "What is the overall win rate?",
            "Show pipeline by stage.",
            "How many opportunities are open?",
            "Top 5 accounts by opportunity value.",
            "Which owner has the most closed-won opportunities?",
            "Show monthly opportunity trend.",
        ],
    )
    def test_a_direct_question_becomes_a_data_query(self, scripted, question):
        scripted(plan_reply("DATA_QUERY", GROUPED_SQL))
        plan = llm.generate_plan(question)
        assert plan.intent is Intent.DATA_QUERY
        assert plan.sql == GROUPED_SQL
        assert plan.needs_data is True

    def test_a_data_query_without_sql_cannot_be_an_empty_success(self, scripted):
        """Labelled as a query but carrying nothing to run: refuse, don't no-op."""
        scripted(plan_reply("DATA_QUERY", None))
        assert llm.generate_plan("anything").intent is Intent.UNSUPPORTED


# ---------------------------------------------------------------------------
# 2. Explanatory analytics routes correctly, with supporting data
# ---------------------------------------------------------------------------


class TestExplanatoryRouting:
    @pytest.mark.parametrize(
        "question",
        [
            "How can I improve the win rate?",
            "What affects our win rate?",
            "Why is pipeline low?",
            "How can I change the open pipeline?",
            "Why is APAC performing worse?",
            "How is Acme Corp doing?",
            "What should I pay attention to?",
            "Which owner needs attention?",
        ],
    )
    def test_an_explanatory_question_still_retrieves_evidence(self, scripted, question):
        """The old design refused these. They now carry supporting SQL."""
        scripted(plan_reply("DATA_EXPLANATION", GROUPED_SQL))
        plan = llm.generate_plan(question)
        assert plan.intent is Intent.DATA_EXPLANATION
        assert plan.is_answerable is True
        assert plan.needs_data is True

    def test_a_definitional_question_needs_no_data(self, scripted):
        scripted(plan_reply("DATA_EXPLANATION", None))
        plan = llm.generate_plan("What does win rate mean?")
        assert plan.intent is Intent.DATA_EXPLANATION
        assert plan.is_answerable is True
        assert plan.needs_data is False


# ---------------------------------------------------------------------------
# 3. Paraphrases route through identical code
# ---------------------------------------------------------------------------


class TestParaphrasesAreNotSpecialCased:
    PARAPHRASES = {
        "win rate": [
            "How can I change the win rate?",
            "What would change the win rate?",
            "How does the win rate move?",
            "What can affect the win rate?",
            "Can this win rate be improved?",
        ],
        "pipeline": [
            "How can I increase the pipeline?",
            "What would grow open pipeline?",
            "Why does the pipeline look low?",
            "What moves our pipeline?",
        ],
        "region performance": [
            "Why is APAC behind?",
            "What is holding APAC back?",
            "How is APAC doing compared with the rest?",
        ],
        "account comparison": [
            "Compare Acme Corp with Summit Industries.",
            "How does Acme Corp stack up against Summit Industries?",
            "Acme Corp versus Summit Industries.",
        ],
    }

    @pytest.mark.parametrize("concept", sorted(PARAPHRASES))
    def test_every_paraphrase_of_a_concept_routes_the_same_way(self, scripted, concept):
        """Routing depends on the decision, never on how the sentence is built."""
        client = scripted(plan_reply("DATA_EXPLANATION", GROUPED_SQL))
        outcomes = {llm.generate_plan(q) for q in self.PARAPHRASES[concept]}
        assert len(outcomes) == 1, outcomes
        assert client.calls == ["route"] * len(self.PARAPHRASES[concept])

    def test_no_routing_module_mentions_a_specific_metric(self):
        """A metric name in routing logic is the bug this design removes.

        ``refusal.py`` is not checked here: its keyword lists only choose the
        wording of a refusal that has already been decided, and never decide
        whether a question is answerable.
        """
        metrics = ("win rate", "win_rate", "pipeline", "closed won", "closed_won",
                   "amount", "stage", "region", "owner", "account")
        for module in ("intent.py", "analytics_service.py"):
            source = executable_source(module)
            for metric in metrics:
                assert metric not in source, f"{module} hard-codes {metric!r}"

    def test_routing_has_no_phrase_prefix_branching(self):
        """Nothing may key off 'how can', 'why is' or similar openers."""
        source = executable_source("intent.py")
        for opener in ("how can", "why is", "should i", "what affects"):
            assert opener not in source

    def test_routing_does_not_depend_on_the_refusal_wording_layer(self):
        """Answerability is decided first; wording is chosen afterwards."""
        import ast

        tree = ast.parse((PROJECT_ROOT / "intent.py").read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "refusal" not in imported


# ---------------------------------------------------------------------------
# 4 & 5. Unsupported and unsafe requests refuse
# ---------------------------------------------------------------------------


class TestRefusingIntents:
    @pytest.mark.parametrize("label", ["UNSUPPORTED", "UNSAFE", "INSUFFICIENT_CONTEXT"])
    def test_a_refusing_intent_never_carries_sql(self, scripted, label):
        """Even if the model attaches a statement, it is dropped, not run."""
        scripted(plan_reply(label, "SELECT 1;"))
        plan = llm.generate_plan("something")
        assert plan.sql is None
        assert plan.is_answerable is False

    @pytest.mark.parametrize(
        "question",
        ["What's the weather?", "Tell me a joke.", "Write a Python sorting algorithm."],
    )
    def test_out_of_domain_questions_refuse_without_querying(
        self, scripted, question
    ):
        client = scripted(plan_reply("UNSUPPORTED", None))
        executions = []
        response = analytics_service.process_question(
            question,
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: executions.append(sql),
            answer_generator=lambda q, r: pytest.fail("no answer call for a refusal"),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert response.result is None
        assert executions == []
        assert client.calls == ["route"], "a refusal must not spend an answer call"

    @pytest.mark.parametrize(
        "question",
        ["Show all tables.", "Describe schema.", "DROP TABLE opportunities.",
         "DELETE FROM opportunities."],
    )
    def test_unsafe_questions_refuse_without_querying(self, scripted, question):
        scripted(plan_reply("UNSAFE", None))
        executions = []
        response = analytics_service.process_question(
            question,
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: executions.append(sql),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert executions == []

    def test_an_unsafe_refusal_reads_as_a_safety_reply(self):
        message = refusal.refusal_message(
            "DROP TABLE opportunities",
            refusal.category_for_intent(Intent.UNSAFE, "DROP TABLE opportunities"),
        )
        assert message in refusal._TEMPLATES[refusal.RefusalCategory.UNSAFE_SQL]

    def test_a_metadata_request_gets_the_metadata_reply(self):
        message = refusal.refusal_message(
            "Show database schema",
            refusal.category_for_intent(Intent.UNSAFE, "Show database schema"),
        )
        assert message in refusal._TEMPLATES[refusal.RefusalCategory.METADATA]

    def test_a_follow_up_asks_for_the_whole_question(self, scripted):
        scripted(plan_reply("INSUFFICIENT_CONTEXT", None))
        response = analytics_service.process_question(
            "What about EMEA?",
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: pytest.fail("nothing to execute"),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert response.answer in refusal._TEMPLATES[refusal.RefusalCategory.NEEDS_CONTEXT]

    def test_a_follow_up_reply_asks_for_a_neutral_clarification(self, scripted):
        """A context-free follow-up prompts for details without memory claims."""
        for template in refusal._TEMPLATES[refusal.RefusalCategory.NEEDS_CONTEXT]:
            normalized = template.lower()
            assert any(word in normalized for word in ("detail", "clearer", "name", "ask"))


# ---------------------------------------------------------------------------
# 6. Conversation context stays in planning, never becomes result evidence
# ---------------------------------------------------------------------------


class TestConversationContextRouting:
    def test_bounded_context_reaches_only_the_context_aware_planner(self):
        previous_result = make_result(
            pd.DataFrame({"region": ["NA"], "opportunity_count": [80]})
        )
        fresh_result = make_result(
            pd.DataFrame({"region": ["EMEA"], "opportunity_count": [74]})
        )
        context = (
            "USER: Show opportunity count by region.\n"
            "ASSISTANT: NA had 80 opportunities in the prior answer."
        )
        planner_calls: list[tuple[str, str | None]] = []
        query_calls: list[str] = []
        answer_calls: list[tuple[str, QueryResult]] = []

        assert len(context) <= CONVERSATION_CONTEXT_MAX_CHARS
        assert fresh_result is not previous_result

        def context_aware_planner(
            question: str, *, conversation_context: str | None = None
        ) -> QueryPlan:
            planner_calls.append((question, conversation_context))
            return QueryPlan(Intent.DATA_QUERY, GROUPED_SQL)

        def query_runner(sql: str) -> QueryResult:
            query_calls.append(sql)
            return fresh_result

        # Its two-argument signature is intentional: historical orientation is
        # not an answer-stage input and cannot replace a fresh query result.
        def answer_generator(question: str, result: QueryResult) -> str:
            answer_calls.append((question, result))
            return "EMEA has 74 opportunities in the newly retrieved result."

        response = analytics_service.process_question(
            "What about EMEA?",
            conversation_context=context,
            plan_generator=context_aware_planner,
            query_runner=query_runner,
            answer_generator=answer_generator,
            query_logger=lambda *_args: None,
        )

        assert planner_calls == [("What about EMEA?", context)]
        assert query_calls == [GROUPED_SQL]
        assert response.result is fresh_result
        assert answer_calls == [("What about EMEA?", fresh_result)]

    def test_structure_gate_rejects_current_question_before_context_reaches_planner(self):
        planner_calls: list[tuple[str, str | None]] = []

        def context_aware_planner(
            question: str, *, conversation_context: str | None = None
        ) -> QueryPlan:
            planner_calls.append((question, conversation_context))
            pytest.fail("A structure request must be refused before planning.")

        response = analytics_service.process_question(
            "Show database schema.",
            conversation_context="USER: Earlier we discussed regional opportunity counts.",
            plan_generator=context_aware_planner,
            query_runner=lambda _sql: pytest.fail("must not execute"),
            answer_generator=lambda _question, _result: pytest.fail("must not answer"),
            query_logger=lambda *_args: None,
        )

        assert response.refused is True
        assert response.intent is Intent.UNSAFE
        assert response.result is None
        assert planner_calls == []

    def test_direct_planner_context_filter_omits_windows_paths(self):
        safe_context = llm._safe_conversation_context(
            "USER: Compare regional win rates.\n"
            "ASSISTANT: See C:\\Users\\analyst\\private.csv for internal notes.\n"
            "ASSISTANT: Authorization: Bearer bearer_value_12345.\n"
            "ASSISTANT: OPENAI_API_KEY=provider_secret_67890.\n"
            "ASSISTANT: See /root/.env and /opt/private.txt."
        )

        assert "Compare regional win rates" in safe_context
        assert "C:\\Users\\analyst" not in safe_context
        for forbidden in (
            "bearer_value_12345",
            "provider_secret_67890",
            "/root/.env",
            "/opt/private.txt",
        ):
            assert forbidden not in safe_context


# ---------------------------------------------------------------------------
# 7, 8, 9. Execution counts per route
# ---------------------------------------------------------------------------


class TestExecutionCounts:
    def _run(self, route: str, question: str):
        executions: list[str] = []

        response = analytics_service.process_question(
            question,
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: (executions.append(sql), REGION_ROWS)[1],
            query_logger=lambda *a: None,
        )
        return response, executions

    def test_a_data_query_executes_exactly_once(self, scripted):
        client = scripted(plan_reply("DATA_QUERY", GROUPED_SQL))
        response, executions = self._run("DATA_QUERY", "opportunities by region")
        assert len(executions) == 1
        assert response.refused is False
        assert response.result is REGION_ROWS, "answer and table share one result"
        assert client.calls == ["route", "answer"]

    def test_an_explanation_needing_data_executes_exactly_once(self, scripted):
        client = scripted(plan_reply("DATA_EXPLANATION", GROUPED_SQL))
        response, executions = self._run("DATA_EXPLANATION", "why is one region behind?")
        assert len(executions) == 1
        assert response.intent is Intent.DATA_EXPLANATION
        assert client.calls == ["route", "answer"]

    def test_a_conceptual_answer_executes_no_query(self, scripted):
        client = scripted(plan_reply("DATA_EXPLANATION", None), answer="Here is what it means.")
        response, executions = self._run("DATA_EXPLANATION", "what does win rate mean?")
        assert executions == [], "a definitional question must not touch the database"
        assert response.refused is False
        assert response.result is None
        assert response.answer == "Here is what it means."
        assert client.calls == ["route", "conceptual"]

    def test_a_conceptual_answer_is_never_asked_for_figures(self, scripted):
        assert "State NO numbers" in llm.CONCEPTUAL_SYSTEM_PROMPT
        assert "you have no data" in llm.CONCEPTUAL_SYSTEM_PROMPT.lower()
        assert "Never name a column" in llm.CONCEPTUAL_SYSTEM_PROMPT

    def test_a_failed_conceptual_answer_refuses_rather_than_returning_nothing(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            llm, "generate_plan", lambda q: QueryPlan(intent=Intent.DATA_EXPLANATION)
        )

        def boom(_question):
            raise RuntimeError("provider unavailable")

        response = analytics_service.process_question(
            "what does that measure mean?",
            plan_generator=llm.generate_plan,
            conceptual_answer_generator=boom,
            query_runner=lambda sql: pytest.fail("nothing to execute"),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert response.answer is not None


# ---------------------------------------------------------------------------
# 9. Safety survives the new structure
# ---------------------------------------------------------------------------


class TestSafetyIsUnchanged:
    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE opportunities;",
            "DELETE FROM opportunities;",
            "SELECT * FROM information_schema.columns;",
            "SELECT * FROM duckdb_tables();",
            "ATTACH 'other.db' AS other;",
        ],
    )
    def test_a_structured_plan_cannot_smuggle_unsafe_sql(self, scripted, sql):
        """The intent label carries no authority. The guard decides."""
        scripted(plan_reply("DATA_QUERY", sql))
        with pytest.raises(ValueError):
            llm.generate_plan("anything")

    @pytest.mark.parametrize(
        "question",
        [
            "Show database schema.",
            "Show all tables.",
            "Describe schema",
            "list the tables",
            "What are the column names?",
            "Tell me the table structure.",
            "What data types does it use?",
            "show me the database metadata",
        ],
    )
    def test_a_structure_request_is_refused_before_any_model_call(self, question):
        """Live evaluation found this leak: the model answered 'Show database
        schema.' with SELECT *, which the guard allows, and the answer then read
        every column name out loud. Safety cannot depend on how it routes."""
        assert intent_module.is_structure_request(question) is True

        response = analytics_service.process_question(
            question,
            plan_generator=lambda q: pytest.fail("must not reach the model"),
            query_runner=lambda sql: pytest.fail("must not execute"),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert response.intent is Intent.UNSAFE
        assert response.result is None

    @pytest.mark.parametrize(
        "question",
        [
            "Show every field for OPP-1003.",
            "Describe the pipeline.",
            "Show all opportunities in the Negotiation stage.",
            "Which accounts have the most pipeline?",
            "How many opportunities are open?",
            "list the top 5 accounts by amount",
            "What is the win rate by region?",
        ],
    )
    def test_ordinary_questions_are_not_caught_by_the_structure_gate(self, question):
        """A safety gate that swallows business questions is its own bug."""
        assert intent_module.is_structure_request(question) is False

    def test_the_routing_prompt_makes_structure_requests_outrank_everything(self):
        prompt_source = (PROJECT_ROOT / "llm.py").read_text(encoding="utf-8")
        assert "This outranks every other intent." in prompt_source
        assert "returns whole records so the columns can be read off" in prompt_source

    def test_the_guard_still_rejects_what_it_always_rejected(self):
        with pytest.raises(ValueError):
            sql_guard.validate_sql("DROP TABLE opportunities;")
        with pytest.raises(ValueError):
            sql_guard.validate_sql("SELECT * FROM information_schema.tables;")

    def test_the_database_layer_revalidates_independently(self, initialized_database):
        """Even a plan whose SQL bypassed stage one is stopped at execution."""
        with pytest.raises(SqlValidationError):
            database.run_query("DROP TABLE opportunities;")

    def test_a_guard_rejection_becomes_a_refusal_not_a_crash(self):
        response = analytics_service.process_question(
            "something unsafe",
            plan_generator=lambda q: QueryPlan(
                intent=Intent.DATA_QUERY, sql="DROP TABLE opportunities;"
            ),
            query_runner=lambda sql: (_ for _ in ()).throw(
                SqlValidationError("Generated SQL contains a disallowed statement.")
            ),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
        assert "disallowed" not in (response.answer or "")


# ---------------------------------------------------------------------------
# 10. Real failures stay failures
# ---------------------------------------------------------------------------


class TestRealErrorsAreNotSwallowed:
    def test_a_provider_outage_still_raises(self, monkeypatch):
        def boom(_question):
            raise RuntimeError("Groq API request failed: 503")

        with pytest.raises(RuntimeError):
            analytics_service.process_question(
                "opportunities by region",
                plan_generator=boom,
                query_runner=lambda sql: pytest.fail("must not execute"),
                query_logger=lambda *a: None,
            )

    def test_a_database_error_still_raises(self):
        from database import SqlExecutionError

        def boom(_sql):
            raise SqlExecutionError("Referenced column does not exist.")

        with pytest.raises(SqlExecutionError):
            analytics_service.process_question(
                "bad column",
                plan_generator=lambda q: parse_plan(GROUPED_SQL),
                query_runner=boom,
                query_logger=lambda *a: None,
            )

    def test_an_answer_failure_still_preserves_the_result(self):
        def boom(_question, _result):
            raise RuntimeError("answer stage down")

        response = analytics_service.process_question(
            "opportunities by region",
            plan_generator=lambda q: parse_plan(GROUPED_SQL),
            query_runner=lambda sql: REGION_ROWS,
            answer_generator=boom,
            query_logger=lambda *a: None,
        )
        assert response.answer_fallback_used is True
        assert response.result is REGION_ROWS
        assert response.refused is False


# ---------------------------------------------------------------------------
# 11. The call budget did not grow
# ---------------------------------------------------------------------------


class TestCallBudget:
    @pytest.mark.parametrize(
        "route,expected",
        [
            (plan_reply("DATA_QUERY", GROUPED_SQL), ["route", "answer"]),
            (plan_reply("DATA_EXPLANATION", GROUPED_SQL), ["route", "answer"]),
            (plan_reply("DATA_EXPLANATION", None), ["route", "conceptual"]),
            (
                plan_reply(
                    "GENERAL_CONVERSATION",
                    None,
                ),
                ["route", "conversation"],
            ),
            (plan_reply("UNSUPPORTED", None), ["route"]),
            (plan_reply("UNSAFE", None), ["route"]),
            (plan_reply("INSUFFICIENT_CONTEXT", None), ["route"]),
        ],
    )
    def test_no_route_makes_a_third_call(self, scripted, route, expected):
        client = scripted(route)
        analytics_service.process_question(
            "a question",
            plan_generator=llm.generate_plan,
            query_runner=lambda sql: REGION_ROWS,
            query_logger=lambda *a: None,
        )
        assert client.calls == expected
        assert len(client.calls) <= 2, "the two-call budget is the whole design"

    def test_routing_is_folded_into_the_first_call_not_added_before_it(self, scripted):
        """Intent and SQL come back together, so routing costs no extra call."""
        client = scripted(plan_reply("DATA_QUERY", GROUPED_SQL))
        plan = llm.generate_plan("opportunities by region")
        assert client.calls == ["route"]
        assert plan.intent is Intent.DATA_QUERY and plan.sql is not None


# ---------------------------------------------------------------------------
# 13. Conversation persistence semantics
# ---------------------------------------------------------------------------


class TestConversationPersistenceSemantics:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        charts = tmp_path / "charts"
        charts.mkdir()
        monkeypatch.setattr(web_app, "CHARTS_DIR", charts, raising=False)

        def processor(question, original_question=None, conversation_context=None):
            return analytics_service.process_question(
                question,
                original_question=original_question,
                conversation_context=conversation_context,
                plan_generator=lambda q: parse_plan(GROUPED_SQL),
                query_runner=lambda sql: REGION_ROWS,
                answer_generator=lambda q, r: "NA leads on opportunity count.",
                query_logger=lambda *a: None,
            )

        application = web_app.create_app(process_question_func=processor)
        application.config.update(TESTING=True, CHARTS_DIR=charts)
        return application.test_client()

    def test_an_explanatory_answer_is_persisted_in_a_conversation(self, client):
        response = client.post("/api/query", json={"question": "why is one region ahead?"})
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["conversation_saved"] is True
        detail = client.get(f"/api/conversations/{payload['conversation_id']}").get_json()
        messages = detail["conversation"]["messages"]
        assert [message["role"] for message in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "why is one region ahead?"
        assert messages[1]["content"] == "NA leads on opportunity count."

    def test_no_intent_label_reaches_the_browser(self, client):
        body = client.post(
            "/api/query", json={"question": "why is one region ahead?"}
        ).get_data(as_text=True)
        for label in [member.value for member in Intent]:
            assert label not in body
        assert "intent" not in body.lower()

    def test_saved_conversation_still_hides_sql(self, client):
        payload = client.post(
            "/api/query", json={"question": "why is one region ahead?"}
        ).get_json()
        body = client.get(f"/api/conversations/{payload['conversation_id']}").get_data(
            as_text=True
        )
        assert "SELECT" not in body
        assert "opportunity_count FROM" not in body


# ---------------------------------------------------------------------------
# Parser robustness: a formatting slip must not become a refusal
# ---------------------------------------------------------------------------


class TestPlanParsing:
    def test_plain_sql_is_still_understood(self):
        plan = parse_plan(GROUPED_SQL)
        assert plan.intent is Intent.DATA_QUERY and plan.sql == GROUPED_SQL

    def test_the_legacy_sentinel_is_still_understood(self):
        assert parse_plan("INVALID_QUESTION").intent is Intent.UNSUPPORTED

    def test_a_fenced_json_object_is_understood(self):
        text = '```json\n{"intent": "DATA_QUERY", "sql": "%s"}\n```' % GROUPED_SQL
        assert parse_plan(text).sql == GROUPED_SQL

    def test_prose_around_the_object_is_tolerated(self):
        text = 'Here you go: {"intent": "DATA_QUERY", "sql": "%s"} hope that helps' % GROUPED_SQL
        assert parse_plan(text).sql == GROUPED_SQL

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("analytics_query", Intent.DATA_QUERY),
            ("conceptual_analytics", Intent.DATA_EXPLANATION),
            ("Data Explanation", Intent.DATA_EXPLANATION),
            ("conversation", Intent.GENERAL_CONVERSATION),
            ("out_of_scope", Intent.UNSUPPORTED),
            ("unsafe_sql", Intent.UNSAFE),
        ],
    )
    def test_label_synonyms_are_accepted(self, alias, expected):
        sql = None if expected not in ANSWERABLE else GROUPED_SQL
        assert parse_plan(plan_reply(alias, sql)).intent is expected

    def test_an_unknown_label_with_usable_sql_is_not_thrown_away(self):
        """A relabelled reply must not lose a perfectly good query."""
        plan = parse_plan(plan_reply("something_new", GROUPED_SQL))
        assert plan.intent is Intent.DATA_QUERY and plan.sql == GROUPED_SQL

    def test_a_conversation_route_never_retains_sql(self):
        plan = parse_plan(
            plan_reply(
                "GENERAL_CONVERSATION",
                "SELECT 1;",
            )
        )
        assert plan.intent is Intent.GENERAL_CONVERSATION
        assert plan.sql is None

    @pytest.mark.parametrize(
        "unsafe_response",
        [
            "```sql SELECT * FROM opportunities```",
            "DATA_QUERY",
            "Use API_KEY=not-a-real-secret-value",
            "Read C:/private/config.txt",
            "The current total is 300.",
        ],
    )
    def test_an_unsafe_conversation_reply_is_not_exposed(self, unsafe_response):
        assert intent_module.safe_conversation_response(unsafe_response) is None

    @pytest.mark.parametrize("junk", ["", "   ", "not json and not sql", None, "{", "[]"])
    def test_unintelligible_output_refuses_instead_of_raising(self, junk):
        assert parse_plan(junk).intent is Intent.UNSUPPORTED

    def test_a_null_sql_string_is_treated_as_no_sql(self):
        assert parse_plan('{"intent": "DATA_EXPLANATION", "sql": "null"}').sql is None

    def test_the_sentinel_inside_the_sql_field_is_not_executed(self):
        plan = parse_plan(plan_reply("DATA_QUERY", "INVALID_QUESTION"))
        assert plan.sql is None and plan.intent is Intent.UNSUPPORTED


class TestBackwardCompatibility:
    def test_generate_sql_still_returns_a_statement(self, scripted):
        scripted(plan_reply("DATA_QUERY", GROUPED_SQL))
        assert llm.generate_sql("opportunities by region") == GROUPED_SQL

    def test_generate_sql_still_returns_the_sentinel_when_there_is_nothing_to_run(
        self, scripted
    ):
        scripted(plan_reply("UNSUPPORTED", None))
        assert llm.generate_sql("the weather") == llm.INVALID_QUESTION

    def test_an_injected_sql_generator_still_works(self):
        response = analytics_service.process_question(
            "opportunities by region",
            sql_generator=lambda q: GROUPED_SQL,
            query_runner=lambda sql: REGION_ROWS,
            answer_generator=lambda q, r: "ok",
            query_logger=lambda *a: None,
        )
        assert response.refused is False and response.result is REGION_ROWS

    def test_an_injected_sentinel_still_refuses(self):
        response = analytics_service.process_question(
            "the weather",
            sql_generator=lambda q: llm.INVALID_QUESTION,
            query_runner=lambda sql: pytest.fail("must not execute"),
            query_logger=lambda *a: None,
        )
        assert response.refused is True
