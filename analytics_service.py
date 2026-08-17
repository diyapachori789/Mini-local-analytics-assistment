"""Shared orchestration for the CLI and local web application.

The service owns the single analytics workflow.  It executes one generated SQL
statement, then gives the resulting :class:`database.QueryResult` to both the
answer and optional chart paths.  Adapters are responsible only for rendering
the structured response.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

from chart import (
    ChartDecision,
    ChartError,
    ChartRecommendation,
    ChartType,
    create_chart,
    is_chart_request,
    recommend_chart,
    requested_chart_type,
    strip_chart_directive,
)
from config import NO_DATA_ANSWER
from database import QueryResult, SqlValidationError, run_query
from intent import (
    CONVERSATIONAL,
    Intent,
    QueryPlan,
    is_structure_request,
    parse_plan,
    numbers_in_context,
    safe_conversation_response,
)
from refusal import category_for_intent, refusal_message
from llm import (
    INVALID_QUESTION,
    absent_identifiers,
    extract_opportunity_ids,
    generate_answer,
    generate_conversation_answer,
    generate_conceptual_answer,
    generate_plan,
    is_comparison_question,
    normalize_question,
)

logger = logging.getLogger(__name__)

_CHART_DIRECTIVES = {
    "auto": "and chart it.",
    "bar": "and show it as a bar chart.",
    "line": "and show it as a line chart.",
    "pie": "and show it as a pie chart.",
    "scatter": "and show it as a scatter chart.",
}


@dataclass(frozen=True)
class AnalysisResponse:
    """The authoritative outcome of one analytics request.

    ``result`` remains internal adapter data and must not be serialized directly
    for a browser response because it includes the generated SQL statement.
    """

    original_question: str
    effective_question: str
    analytical_question: str
    answer: Optional[str]
    result: Optional[QueryResult]
    chart_requested: bool
    chart_path: Optional[Path]
    chart_type: Optional[ChartType]
    chart_note: Optional[str]
    answer_fallback_used: bool
    answer_error: Optional[Exception]
    chart_error: Optional[str]
    refused: bool
    elapsed_seconds: float
    # Internal routing state, kept for logs and tests. Adapters must not
    # serialize it: how a question was classified is not the user's concern.
    intent: Intent = Intent.DATA_QUERY
    # Likewise internal-only; public adapters expose chart output, not policy.
    chart_decision: ChartDecision = ChartDecision.NO_CHART


def adapt_question_for_chart(
    question: str,
    *,
    chart_requested: bool = False,
    chart_type: str = "auto",
) -> str:
    """Add deterministic chart wording only when the UI explicitly requests it.

    An explicitly typed chart type remains authoritative. A generic typed chart
    request can be refined by a specific UI preference without duplicating the
    analytics workflow; the chart module still owns compatibility and fallback.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")
    if not isinstance(chart_requested, bool):
        raise ValueError("chart_requested must be a boolean.")
    if not isinstance(chart_type, str) or chart_type not in _CHART_DIRECTIVES:
        raise ValueError("chart_type must be auto, bar, line, pie, or scatter.")

    normalized = normalize_question(question)
    if not normalized:
        raise ValueError("Question cannot be empty.")
    has_chart_intent = is_chart_request(normalized)
    typed_chart_type = requested_chart_type(normalized) if has_chart_intent else None

    # A user who explicitly requested a pie, line, bar, or scatter chart has
    # made the most specific choice. The UI selector must not override it.
    if typed_chart_type is not None:
        return normalized

    # Generic wording such as "chart it" expresses intent but no type. A
    # selected, specific UI type can safely refine that presentation request.
    if has_chart_intent:
        if chart_requested and chart_type != "auto":
            base_question = strip_chart_directive(normalized).rstrip().rstrip(".?!")
            return f"{base_question} {_CHART_DIRECTIVES[chart_type]}"
        return normalized

    if not chart_requested:
        return normalized

    base_question = normalized.rstrip().rstrip(".?!")
    return f"{base_question} {_CHART_DIRECTIVES[chart_type]}"


def incomplete_comparison_reason(question: str, result: QueryResult) -> Optional[str]:
    """Explain why a comparison chart would mislead, or None when it is fine.

    A chart of one entity built from a question that named several presents a
    single value as if it were the whole comparison - a pie chart of one slice
    reads as 100%. This only fires when the question states a comparison and
    names several identifiers, so ordinary grouped charts are untouched.
    """
    if result is None:
        return None
    if not is_comparison_question(question):
        return None

    named = extract_opportunity_ids(question)
    if len(named) < 2:
        return None

    missing = absent_identifiers(question, result)
    if not missing:
        return None

    returned = len(named) - len(missing)
    if returned <= 1:
        return (
            "A comparison chart was not generated because only "
            f"{returned} of the {len(named)} requested records was returned."
        )
    return None


def _log_query_details(sql: str, result: QueryResult) -> None:
    """Log generated SQL while keeping it out of presentation adapters."""
    if result.truncated:
        logger.info(
            "Answered using SQL: %s | rows=%s (capped at %s, the query matched more)",
            sql,
            result.row_count,
            result.max_rows,
        )
    else:
        logger.info("Answered using SQL: %s | rows=%s", sql, result.row_count)


def _plan_from_sql_generator(
    sql_generator: Callable[[str], str], question: str
) -> QueryPlan:
    """Adapt a string-returning generator to the plan contract.

    Existing callers and tests inject something that returns SQL or
    ``INVALID_QUESTION``. That contract is still honoured: the text is read with
    the same tolerant parser the model output goes through, so an injected
    generator routes exactly as an equivalent model reply would.
    """
    produced = sql_generator(question)
    if produced == INVALID_QUESTION:
        return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)
    return parse_plan(produced)


def _conceptual_response(
    *,
    explain_concept: Callable[[str], str],
    plan: QueryPlan,
    display_question: str,
    effective_question: str,
    analytical_question: str,
    started: float,
) -> AnalysisResponse:
    """Answer a question that needs no figures, so runs no query.

    There is no result, so there is no table and no chart. If the explanation
    cannot be produced there is nothing to fall back to - unlike the analytical
    path, which still has rows to show - so this refuses rather than returning
    an empty success.
    """
    try:
        answer = explain_concept(analytical_question)
    except (RuntimeError, ValueError) as exc:
        logger.error("Conceptual answer generation failed: %s", str(exc))
        answer = None
    except Exception as exc:
        logger.error("Unexpected error during conceptual answer generation: %s", str(exc))
        answer = None

    elapsed = perf_counter() - started
    if answer is None:
        return AnalysisResponse(
            original_question=display_question,
            effective_question=effective_question,
            analytical_question=analytical_question,
            answer=refusal_message(display_question),
            result=None,
            chart_requested=False,
            chart_path=None,
            chart_type=None,
            chart_note=None,
            answer_fallback_used=False,
            answer_error=None,
            chart_error=None,
            refused=True,
            elapsed_seconds=elapsed,
            intent=plan.intent,
            chart_decision=ChartDecision.NO_CHART,
        )

    logger.info(
        "Analytics request answered conceptually in %.3fs; no query was executed.",
        elapsed,
    )
    return AnalysisResponse(
        original_question=display_question,
        effective_question=effective_question,
        analytical_question=analytical_question,
        answer=answer,
        result=None,
        chart_requested=False,
        chart_path=None,
        chart_type=None,
        chart_note=None,
        answer_fallback_used=False,
        answer_error=None,
        chart_error=None,
        refused=False,
        elapsed_seconds=elapsed,
        intent=plan.intent,
        chart_decision=ChartDecision.NO_CHART,
    )


def _general_conversation_response(
    *,
    plan: QueryPlan,
    respond: Callable[..., str],
    conversation_context: Optional[str],
    display_question: str,
    effective_question: str,
    analytical_question: str,
    started: float,
) -> AnalysisResponse:
    """Return a valid social turn without SQL, rows, charts, or DuckDB."""
    answer_error: Optional[Exception] = None
    answer_fallback_used = False
    try:
        try:
            signature = inspect.signature(respond)
        except (TypeError, ValueError):
            produced = respond(analytical_question)
        else:
            accepts_context = (
                "conversation_context" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            produced = (
                respond(
                    analytical_question,
                    conversation_context=conversation_context,
                )
                if accepts_context
                else respond(analytical_question)
            )
        answer = safe_conversation_response(
            produced,
            grounded_numbers=numbers_in_context(conversation_context)
            | numbers_in_context(analytical_question),
        )
        if answer is None:
            raise RuntimeError("Conversational answer could not be returned safely.")
    except Exception as exc:
        logger.error("Conversational answer generation failed: %s", str(exc))
        answer = "Sorry, I had trouble with that one. Could you try rephrasing it?"
        answer_error = exc
        answer_fallback_used = True

    elapsed = perf_counter() - started
    logger.info(
        "Assistant request answered conversationally in %.3fs; no query was executed.",
        elapsed,
    )
    return AnalysisResponse(
        original_question=display_question,
        effective_question=effective_question,
        analytical_question=analytical_question,
        answer=answer,
        result=None,
        chart_requested=False,
        chart_path=None,
        chart_type=None,
        chart_note=None,
        answer_fallback_used=answer_fallback_used,
        answer_error=answer_error,
        chart_error=None,
        refused=False,
        elapsed_seconds=elapsed,
        # The route actually taken, not the family it belongs to: three intents
        # share this stage, and a log that flattened them would hide which.
        intent=plan.intent,
        chart_decision=ChartDecision.NO_CHART,
    )


def process_question(
    question: str,
    *,
    original_question: Optional[str] = None,
    conversation_context: Optional[str] = None,
    plan_generator: Optional[Callable[[str], QueryPlan]] = None,
    sql_generator: Optional[Callable[[str], str]] = None,
    query_runner: Optional[Callable[[str], QueryResult]] = None,
    answer_generator: Optional[Callable[[str, QueryResult], str]] = None,
    conversation_answer_generator: Optional[Callable[..., str]] = None,
    conceptual_answer_generator: Optional[Callable[[str], str]] = None,
    chart_creator: Optional[
        Callable[[str, QueryResult], tuple[Path, ChartType, Optional[str]]]
    ] = None,
    query_logger: Optional[Callable[[str, QueryResult], None]] = None,
) -> AnalysisResponse:
    """Process one effective question through the existing analytics backend.

    The injected callables keep the service independently testable and let the
    CLI retain its existing monkeypatchable compatibility surface.  The query
    runner is called once at most; the returned ``QueryResult`` is passed by
    identity to answer and chart generation.

    ``conversation_context`` is optional, bounded semantic orientation from the
    active persisted conversation. It is supplied to the default planning stage
    and, for GENERAL_CONVERSATION only, the schema-free conversational answer
    stage. It never reaches grounded result answering, charting, or DuckDB. The
    current question still passes the deterministic structure gate before either
    the planner or any context can influence routing.

    ``plan_generator`` is the current first stage: it routes the question and
    returns the SQL to run, if any.  ``sql_generator`` remains supported for
    callers that only produce a statement.  One-argument injected generators
    stay compatible, while context-aware test adapters can explicitly accept a
    ``conversation_context`` keyword.
    """
    started = perf_counter()
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    effective_question = normalize_question(question)
    if not effective_question:
        raise ValueError("Question cannot be empty.")

    display_question = original_question if isinstance(original_question, str) else question
    wants_chart = is_chart_request(effective_question)
    analytical_question = (
        strip_chart_directive(effective_question) if wants_chart else effective_question
    )

    if plan_generator is not None:

        def make_plan(q: str) -> QueryPlan:
            """Call an injected planner without mistaking its TypeError for a retry."""
            try:
                signature = inspect.signature(plan_generator)
            except (TypeError, ValueError):
                return plan_generator(q)
            accepts_context = (
                "conversation_context" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_context:
                return plan_generator(q, conversation_context=conversation_context)
            return plan_generator(q)

    elif sql_generator is not None:

        def make_plan(q: str) -> QueryPlan:
            return _plan_from_sql_generator(sql_generator, q)

    else:

        def make_plan(q: str) -> QueryPlan:
            return generate_plan(q, conversation_context=conversation_context)
    execute = query_runner or run_query
    answer_from_result = answer_generator or generate_answer
    respond_conversationally = conversation_answer_generator or generate_conversation_answer
    explain_concept = conceptual_answer_generator or generate_conceptual_answer
    render_chart = chart_creator or create_chart
    record_query = query_logger or _log_query_details

    def refuse(reason: str, intent: Intent = Intent.UNSUPPORTED) -> AnalysisResponse:
        """Build a refusal outcome. A refusal is a success with nothing to compute."""
        elapsed = perf_counter() - started
        logger.info(
            "Analytics request completed as a refusal in %.3fs (intent=%s, %s).",
            elapsed,
            intent.value,
            reason,
        )
        return AnalysisResponse(
            original_question=display_question,
            effective_question=effective_question,
            analytical_question=analytical_question,
            # Written locally from a template: no third model call, no quota.
            answer=refusal_message(
                display_question, category_for_intent(intent, display_question)
            ),
            result=None,
            chart_requested=False,
            chart_path=None,
            chart_type=None,
            chart_note=None,
            answer_fallback_used=False,
            answer_error=None,
            chart_error=None,
            refused=True,
            elapsed_seconds=elapsed,
            intent=intent,
            chart_decision=ChartDecision.NO_CHART,
        )

    logger.info("Analytics request started (chart_requested=%s).", wants_chart)

    if is_structure_request(analytical_question):
        # Decided here rather than by the model. A request for the shape of the
        # database can be answered with a query the guard has no reason to
        # reject, whose answer then reads out the column names - a leak built
        # from safe parts. Refusing first also spends no model call.
        return refuse("question asks for database structure", Intent.UNSAFE)

    try:
        plan = make_plan(analytical_question)
    except ValueError:
        # No usable statement could be produced for this question. That is the
        # question being unsupported, not the system failing, so it is refused
        # rather than raised. Real failures (RuntimeError from the provider,
        # database errors) still propagate untouched.
        return refuse("no valid SQL could be generated")

    if plan.intent in CONVERSATIONAL:
        # Greeting, meta-conversation, an ordinary off-topic question, or a
        # request too ambiguous to run: all answered from words alone, by the
        # same second call, touching no database.
        return _general_conversation_response(
            plan=plan,
            respond=respond_conversationally,
            conversation_context=conversation_context,
            display_question=display_question,
            effective_question=effective_question,
            analytical_question=analytical_question,
            started=started,
        )

    if not plan.is_answerable:
        return refuse("routed away from analytics", plan.intent)

    if plan.sql is None:
        # A conceptual question: it is answered from what this dataset records,
        # so nothing is executed and no result table or chart is produced. This
        # uses the answer-stage call, not an additional one.
        return _conceptual_response(
            explain_concept=explain_concept,
            plan=plan,
            display_question=display_question,
            effective_question=effective_question,
            analytical_question=analytical_question,
            started=started,
        )

    sql = plan.sql

    # This is the one and only analytical database execution in the workflow.
    try:
        result = execute(sql)
    except SqlValidationError:
        # The safety layer rejected the generated statement. That is the guard
        # doing its job on an unsupported request, not an outage.
        return refuse("generated SQL was refused by the safety guard", Intent.UNSAFE)

    record_query(sql, result)

    # This is a pure, post-query decision over the one authoritative result.
    # It cannot add a model call or a second analytical database execution.
    chart_recommendation: ChartRecommendation = recommend_chart(
        effective_question, result
    )

    answer: Optional[str]
    answer_error: Optional[Exception] = None
    answer_fallback_used = False
    if result.is_empty:
        # Keep the existing no-data behavior local and avoid constructing an
        # answer client or calling an injected answer adapter unnecessarily.
        answer = NO_DATA_ANSWER
    else:
        try:
            answer = answer_from_result(analytical_question, result)
        except (RuntimeError, ValueError) as exc:
            # Pass text through the existing redacting filter rather than an
            # exception object, whose deferred formatting could bypass it.
            logger.error("Answer generation failed, preserving the query result: %s", str(exc))
            answer = None
            answer_error = exc
            answer_fallback_used = True
        except Exception as exc:
            logger.error(
                "Unexpected error during answer generation; preserving result: %s",
                str(exc),
            )
            answer = None
            answer_error = exc
            answer_fallback_used = True

    chart_path: Optional[Path] = None
    chart_type: Optional[ChartType] = None
    chart_note: Optional[str] = None
    chart_error: Optional[str] = None
    if chart_recommendation.decision is not ChartDecision.NO_CHART:
        blocked_reason = incomplete_comparison_reason(analytical_question, result)
    else:
        blocked_reason = None

    if blocked_reason:
        # Skip only the chart. The answer and the result table still stand, and
        # nothing is re-queried or re-generated to reach this decision.
        logger.warning("Comparison chart suppressed: %s", blocked_reason)
        chart_note = blocked_reason
    elif chart_recommendation.should_render:
        try:
            chart_path, chart_type, chart_note = render_chart(effective_question, result)
        except ChartError as exc:
            logger.warning("Chart not generated: %s", exc)
            chart_error = str(exc)
        except Exception:
            logger.exception("Unexpected error while generating a chart.")
            chart_error = "an unexpected error occurred. See logs/app.log."
    elif chart_recommendation.decision is ChartDecision.USER_REQUESTED:
        chart_note = chart_recommendation.note

    elapsed = perf_counter() - started
    logger.info(
        "Analytics request completed (rows=%s, chart_generated=%s, elapsed=%.3fs).",
        result.row_count,
        chart_path is not None,
        elapsed,
    )
    return AnalysisResponse(
        original_question=display_question,
        effective_question=effective_question,
        analytical_question=analytical_question,
        answer=answer,
        result=result,
        chart_requested=chart_recommendation.decision is not ChartDecision.NO_CHART,
        chart_path=chart_path,
        chart_type=chart_type,
        chart_note=chart_note,
        answer_fallback_used=answer_fallback_used,
        answer_error=answer_error,
        chart_error=chart_error,
        refused=False,
        elapsed_seconds=elapsed,
        intent=plan.intent,
        chart_decision=chart_recommendation.decision,
    )
