from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
from groq import Groq

from config import (
    ANSWER_MAX_ROWS,
    ANSWER_MAX_TOKENS,
    LLM_MAX_RETRIES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
    NO_DATA_ANSWER,
    require_groq_api_key,
)
from database import QueryResult, get_schema
from intent import Intent, QueryPlan, parse_plan
from sql_guard import (
    ALLOWED_PREFIXES,
    INVALID_QUESTION,
    clean_sql,
    is_refusal,
    validate_cleaned_sql,
    validate_sql,
)

logger = logging.getLogger(__name__)

# clean_sql / validate_sql are re-exported so existing callers and tests that
# import them from this module keep working. The rules live in sql_guard.
__all__ = [
    "INVALID_QUESTION",
    "NO_DATA_ANSWER",
    "Intent",
    "QueryPlan",
    "clean_sql",
    "generate_answer",
    "generate_conceptual_answer",
    "generate_plan",
    "generate_sql",
    "is_refusal",
    "normalize_question",
    "validate_sql",
]

# Characters with no visual width: delete them outright.
_ZERO_WIDTH_CHARS = (
    0xFEFF,  # zero-width no-break space / byte-order mark
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0x2060,  # word joiner
)

# Characters that render as a space. These must become an ordinary space rather
# than be deleted: removing them silently joins words, turning a question pasted
# from a browser or word processor into "showalldeals".
_UNICODE_SPACES = (
    0x00A0,  # no-break space
    0x1680,  # ogham space mark
    0x2000,  # en quad
    0x2001,  # em quad
    0x2002,  # en space
    0x2003,  # em space
    0x2004,  # three-per-em space
    0x2005,  # four-per-em space
    0x2006,  # six-per-em space
    0x2007,  # figure space
    0x2008,  # punctuation space
    0x2009,  # thin space
    0x200A,  # hair space
    0x202F,  # narrow no-break space
    0x205F,  # medium mathematical space
    0x3000,  # ideographic space
)

_TEXT_NORMALISATION = {code_point: None for code_point in _ZERO_WIDTH_CHARS}
_TEXT_NORMALISATION.update({code_point: " " for code_point in _UNICODE_SPACES})


# Opportunity identifiers a user can name literally, e.g. "OPP-1003".
_OPPORTUNITY_ID_RE = re.compile(r"\bOPP-\d+\b", re.IGNORECASE)

# Wording that states an explicit comparison. Deliberately narrow: naming several
# values is not on its own a comparison, so this must be said, not inferred.
_COMPARISON_RE = re.compile(
    r"\b(compare[sd]?|comparison|versus|vs\.?|against|difference\s+between|"
    r"differences\s+between|side\s+by\s+side)\b",
    re.IGNORECASE,
)


def extract_opportunity_ids(question: str) -> list[str]:
    """Return every opportunity identifier named in a question, upper-cased.

    Order is preserved and duplicates removed, so a repeated identifier is only
    required once.
    """
    if not isinstance(question, str):
        return []
    seen: dict[str, None] = {}
    for match in _OPPORTUNITY_ID_RE.findall(question):
        seen.setdefault(match.upper(), None)
    return list(seen)


def is_comparison_question(question: str) -> bool:
    """True when the question explicitly asks for a comparison."""
    if not isinstance(question, str):
        return False
    return bool(_COMPARISON_RE.search(question))


def missing_identifiers(question: str, sql: str) -> list[str]:
    """Return identifiers named in the question that the SQL does not filter on.

    Compared against the SQL text rather than the result, so this catches a
    dropped identifier before anything is executed. An identifier that is present
    but matches no row is a data outcome, not a generation fault.
    """
    if not isinstance(sql, str):
        return extract_opportunity_ids(question)
    present = {value.upper() for value in _OPPORTUNITY_ID_RE.findall(sql)}
    return [value for value in extract_opportunity_ids(question) if value not in present]


def normalize_question(question: str) -> str:
    """Normalise a user question before it is sent to the model.

    Zero-width characters are removed. Characters that render as a space are
    converted to an ordinary space so that word boundaries survive.
    """
    return question.translate(_TEXT_NORMALISATION).strip()


# Lazily created Groq client. Building it on demand keeps this module importable
# without an API key so the SQL helpers can be unit-tested offline.
_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Return the shared Groq client, creating it on first use."""
    global _client
    if _client is None:
        _client = Groq(
            api_key=require_groq_api_key(),
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )
    return _client


def generate_plan(question: str) -> QueryPlan:
    """Route one question and, when data would help, return validated SQL.

    This is the first of the two model calls. It answers both questions the
    workflow needs at this point - what kind of request is this, and what should
    be retrieved - in a single request, so routing costs no extra quota.

    The returned plan is a proposal. Any SQL it carries has passed
    :mod:`sql_guard`, and the database layer validates it again before it runs.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    # Normalise invisible characters before the empty check: str.strip() does
    # not remove a byte-order mark, so piped input could otherwise reach the
    # API as an "empty" question.
    cleaned_question = normalize_question(question)
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    schema = get_schema()

    # Build a production-style prompt for routing and safe SQL generation.
    system_prompt = (
        "You are the routing and SQL layer of a sales-opportunity analytics\n"
        "assistant. You receive one business question. You decide what kind of\n"
        "request it is, and when data would help answer it you write the DuckDB\n"
        "query that retrieves that data.\n"
        "\n"
        "OUTPUT\n"
        "- Reply with one JSON object and nothing else: no prose, no explanation,\n"
        "  no markdown, no code fences, no comments.\n"
        '- Shape: {"intent": "<INTENT>", "sql": "<one DuckDB SELECT>" or null}\n'
        "- When present, sql is a single statement ending with one semicolon,\n"
        "  written as a one-line JSON string with the escaping JSON requires.\n"
        "- Never restate the question and never add a field of your own.\n"
        "\n"
        "INTENTS - choose exactly one\n"
        "- DATA_QUERY: the question asks for a figure, a list, a ranking, a\n"
        "  breakdown or a comparison. Write the SQL that answers it.\n"
        "- DATA_EXPLANATION: the question is about a measure or about this data\n"
        "  rather than a plain lookup - what it means, how it moves, what would\n"
        "  change it, why it looks the way it does, how something is doing, what\n"
        "  deserves attention. Almost always still write SQL: the current figures\n"
        "  are the evidence any explanation has to rest on. Use null only for a\n"
        "  purely definitional question where no figure would add anything.\n"
        "- INSUFFICIENT_CONTEXT: the question only makes sense as a follow-up to\n"
        "  an earlier turn you cannot see, and names nothing you could query on\n"
        "  its own. There is no conversation memory. sql is null.\n"
        "- UNSUPPORTED: the subject is not in this data at all. sql is null.\n"
        "- UNSAFE: the request would modify or administer the database, or asks\n"
        "  for its structure - tables, columns, schema, types, DDL. sql is null.\n"
        "  This outranks every other intent. A request to see the structure stays\n"
        "  UNSAFE however politely it is phrased, and even though you could write\n"
        "  a legal SELECT that exposes the same thing. Wanting to know what the\n"
        "  data is shaped like is not a business question. Never answer it with a\n"
        "  query that returns whole records so the columns can be read off.\n"
        "\n"
        "THE CENTRAL QUESTION\n"
        "- Do not ask yourself whether the sentence can be translated into SQL.\n"
        "  Ask what data would best support an answer, and retrieve that.\n"
        "- A question is not out of scope merely because it is phrased\n"
        "  conversationally, or asks how, why, whether, or what would happen.\n"
        "  If the schema holds figures that bear on it, it is answerable:\n"
        "  return DATA_EXPLANATION together with the SQL for those figures.\n"
        "- Questions of the form 'how can I change X', 'what affects X', 'why is\n"
        "  X low', 'how is X doing', 'should I worry about X' are questions about\n"
        "  a measure. Retrieve that measure's current value. Where the schema\n"
        "  offers a dimension that breaks it down, return the breakdown rather\n"
        "  than a single total: a breakdown is what makes an explanation useful.\n"
        "- Judge every question against the schema below, not against a list of\n"
        "  familiar phrasings. Two questions that mean the same thing must route\n"
        "  the same way however differently they are worded.\n"
        "- Reserve UNSUPPORTED for subjects this data does not describe. Never\n"
        "  use it for a question about something the schema contains.\n"
        "\n"
        "SAFETY - the statement must be read-only\n"
        "- It must begin with SELECT or WITH.\n"
        "- Never emit INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, EXEC,\n"
        "  CALL, GRANT, REVOKE or COPY, including inside CTEs and subqueries.\n"
        "\n"
        "SCHEMA\n"
        "- Use only the table and columns listed under DATABASE SCHEMA; the only table is opportunities.\n"
        "- Spell column names exactly as the schema spells them. Never invent, rename or guess a column.\n"
        "\n"
        "COLUMN SELECTION\n"
        "- Select the fewest columns that answer the question: the columns asked about, plus any\n"
        "  the question filters, groups or ranks by.\n"
        "- For row-level questions also include opportunity_id and account_name so rows are identifiable.\n"
        "- Use SELECT * ONLY when the question explicitly asks for every column, all fields, or the\n"
        "  full/entire record.\n"
        "- 'all opportunities' constrains which ROWS are returned, not how many columns. It is NOT\n"
        "  a request for SELECT *.\n"
        "- Alias every aggregate, for example SUM(amount) AS total_amount.\n"
        "- If the question asks for a share, percentage, proportion, rate or\n"
        "  breakdown, compute the whole ratio in SQL so the database produces the\n"
        "  figure, not the reader. A percentage is always 100.0 * part / whole:\n"
        "  never multiply a count by 100 without dividing by a total.\n"
        "  Share across groups:\n"
        "    SELECT stage, 100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct_of_total\n"
        "    FROM opportunities GROUP BY stage;\n"
        "  One overall figure:\n"
        "    SELECT 100.0 * SUM(CASE WHEN is_won THEN 1 ELSE 0 END) / COUNT(*)\n"
        "    AS win_rate_pct FROM opportunities;\n"
        "\n"
        "FILTERS - keep every value the user names\n"
        "- If the question names concrete values (opportunity ids, account names,\n"
        "  owners, regions, industries, stages), the filter must include ALL of\n"
        "  them. Never silently drop one.\n"
        "- Two or more named values of the same kind become IN (...) containing\n"
        "  every one of them, not a filter on the first value:\n"
        "    compare OPP-1003 to OPP-1014\n"
        "    -> WHERE opportunity_id IN ('OPP-1003', 'OPP-1014')\n"
        "- This applies however many are named. Three ids means three ids in the\n"
        "  IN list.\n"
        "- A comparison question returns one row or one group per compared thing.\n"
        "  Never collapse a comparison to a single entity unless the user asked\n"
        "  for exactly one.\n"
        "- Copy the named values verbatim, including their case and hyphens.\n"
        "\n"
        "ROW LIMITING - apply this rule literally\n"
        "- If the question names how many rows it wants, you MUST end with LIMIT <that number>.\n"
        "- These phrasings all REQUIRE a LIMIT: 'top N', 'bottom N', 'first N', 'last N',\n"
        "  'N highest', 'N lowest', 'N largest', 'N smallest', 'N biggest', 'N best', 'N worst',\n"
        "  'N most recent'. Each of them means LIMIT N.\n"
        "- A singular superlative ('the biggest deal', 'the most recent opportunity', 'which\n"
        "  account has the highest amount') means LIMIT 1.\n"
        "- Every ranking question needs an ORDER BY that matches the ranking, placed before the\n"
        "  LIMIT: 'top 5 by amount' becomes ORDER BY amount DESC LIMIT 5.\n"
        "- If the question names no row count and uses no superlative, omit LIMIT entirely.\n"
        "  Plain aggregations such as totals or counts per group are never limited.\n"
        "\n"
        "AMBIGUITY\n"
        "- If several readings are possible, choose the narrowest, most literal one.\n"
        "  Narrower never means dropping a value the user named: every named value\n"
        "  stays in the filter.\n"
        "- A vague but on-topic question is still answerable. Return the broadest\n"
        "  breakdown the schema supports rather than refusing.\n"
        "\n"
        f"DATABASE SCHEMA\n{schema}\n"
        "\n"
        "REFERENCE EXAMPLES - these show the expected shape only; always derive the answer from\n"
        "the actual question and the schema above.\n"
        "List the top 5 opportunities with the highest amount.\n"
        '{"intent": "DATA_QUERY", "sql": "SELECT opportunity_id, account_name, amount FROM opportunities ORDER BY amount DESC LIMIT 5;"}\n'
        "What is the total amount won by region?\n"
        '{"intent": "DATA_QUERY", "sql": "SELECT region, SUM(amount) AS total_amount FROM opportunities WHERE is_won = TRUE GROUP BY region ORDER BY total_amount DESC;"}\n'
        "Show every field for opportunities in the Negotiation stage.\n"
        '{"intent": "DATA_QUERY", "sql": "SELECT * FROM opportunities WHERE stage = \'Negotiation\';"}\n'
        "Show all opportunities in the Negotiation stage.\n"
        '{"intent": "DATA_QUERY", "sql": "SELECT opportunity_id, account_name, region, owner, amount, close_date FROM opportunities WHERE stage = \'Negotiation\';"}\n'
        "Compare OPP-1003 to OPP-1014.\n"
        '{"intent": "DATA_QUERY", "sql": "SELECT opportunity_id, account_name, region, stage, amount, is_won FROM opportunities WHERE opportunity_id IN (\'OPP-1003\', \'OPP-1014\');"}\n'
        "Compare Acme Labs and Vertex Labs.\n"
        '{"intent": "DATA_QUERY", "sql": "SELECT account_name, SUM(amount) AS total_amount, COUNT(*) AS deal_count FROM opportunities WHERE account_name IN (\'Acme Labs\', \'Vertex Labs\') GROUP BY account_name;"}\n'
        "How could we move that share of won deals?\n"
        '{"intent": "DATA_EXPLANATION", "sql": "SELECT stage, COUNT(*) AS opportunity_count, SUM(amount) AS total_amount FROM opportunities GROUP BY stage ORDER BY total_amount DESC;"}\n'
        "Why does one of the regions look behind the others?\n"
        '{"intent": "DATA_EXPLANATION", "sql": "SELECT region, COUNT(*) AS opportunity_count, SUM(amount) AS total_amount, 100.0 * SUM(CASE WHEN is_won THEN 1 ELSE 0 END) / COUNT(*) AS win_rate_pct FROM opportunities GROUP BY region ORDER BY win_rate_pct DESC;"}\n'
        "What should I pay attention to?\n"
        '{"intent": "DATA_EXPLANATION", "sql": "SELECT stage, COUNT(*) AS opportunity_count, SUM(amount) AS total_amount FROM opportunities GROUP BY stage ORDER BY total_amount DESC;"}\n'
        "What do you mean by that measure?\n"
        '{"intent": "DATA_EXPLANATION", "sql": null}\n'
        "And the other one?\n"
        '{"intent": "INSUFFICIENT_CONTEXT", "sql": null}\n'
        "What is the weather today?\n"
        '{"intent": "UNSUPPORTED", "sql": null}\n'
        "List the tables in the database.\n"
        '{"intent": "UNSAFE", "sql": null}'
    )

    def request_plan(user_content: str) -> QueryPlan:
        """Send one routing request and read the reply into a plan."""
        try:
            response = _get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=LLM_TEMPERATURE,
            )
        except Exception as exc:
            logger.error("Groq API request failed: %s", exc)
            raise RuntimeError(f"Groq API request failed: {exc}") from exc

        if response is None or not hasattr(response, "choices") or not response.choices:
            logger.error("Groq API returned a response without any choices.")
            raise RuntimeError("Groq API returned an invalid response.")

        message = response.choices[0].message
        if message is None or message.content is None:
            logger.error("Groq API returned an empty message content.")
            raise RuntimeError("Groq API returned an empty message content.")

        logger.info("Groq response received.")
        return parse_plan(message.content)

    logger.info("Groq request started (model=%s).", MODEL_NAME)
    plan = request_plan(cleaned_question)

    if not plan.is_answerable:
        # Normal operation, not a system fault: keep it off the console.
        logger.info("Question routed to %s: %s", plan.intent.value, cleaned_question)
        return plan

    if plan.sql is None:
        # A conceptual question that needs no data. Nothing to validate.
        logger.info("Question routed to %s with no supporting query.", plan.intent.value)
        return plan

    sql_text = plan.sql

    # Every identifier the user named must survive into the filter. A dropped id
    # silently narrows the question, which then produces a confident answer about
    # data that was never retrieved. Checked against the statement before it runs,
    # so this never causes a second execution.
    dropped = missing_identifiers(cleaned_question, sql_text)
    if dropped:
        logger.warning(
            "Generated SQL omitted %s of the identifiers named in the question; retrying once.",
            len(dropped),
        )
        retry_instruction = (
            f"{cleaned_question}\n\n"
            "The previous attempt left out these identifiers that the question "
            f"names: {', '.join(dropped)}. Rewrite the statement so the filter "
            "includes every named identifier, using "
            "opportunity_id IN (...) with all of them."
        )
        plan = request_plan(retry_instruction)
        if not plan.is_answerable or plan.sql is None:
            logger.info("Question could not be answered from the schema after a retry.")
            return QueryPlan(intent=Intent.UNSUPPORTED, sql=None)
        sql_text = plan.sql

        dropped = missing_identifiers(cleaned_question, sql_text)
        if dropped:
            # Refuse rather than run a query that answers a narrower question
            # than the one that was asked.
            logger.error(
                "Generated SQL still omitted %s named identifier(s) after a retry.",
                len(dropped),
            )
            raise ValueError(
                "The generated query left out "
                f"{', '.join(dropped)}, so it would not answer the question asked."
            )

    logger.debug("SQL generated: %s", sql_text)

    try:
        # Already cleaned by the parser, so validate directly and avoid cleaning
        # twice. The intent label carries no authority here: a statement is safe
        # because the guard says so, never because the model said it was.
        validated = validate_cleaned_sql(sql_text)
    except ValueError as exc:
        logger.error("SQL validation failed (%s) for generated SQL: %s", exc, sql_text)
        raise

    logger.info("SQL validation passed (intent=%s).", plan.intent.value)
    return QueryPlan(intent=plan.intent, sql=validated)


def generate_sql(question: str) -> str:
    """Generate a validated SQL SELECT statement from a natural-language question.

    Kept as the narrow, string-in/string-out view of :func:`generate_plan` for
    callers that only want a statement. Any routing outcome with no query to run
    collapses to ``INVALID_QUESTION``, which is exactly what this function has
    always returned for a question it could not translate.
    """
    plan = generate_plan(question)
    if plan.sql is None:
        return INVALID_QUESTION
    return plan.sql


# ---------------------------------------------------------------------------
# Answer generation
#
# The second LLM call. It receives the question and the rows that DuckDB
# actually returned, and turns them into a sentence. It never sees the schema,
# never produces SQL, and never touches the database.
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = (
    "You are a data analyst writing the final answer for a business user.\n"
    "You are given their question and the result of a query that has already\n"
    "been run to answer it.\n"
    "\n"
    "GROUNDING\n"
    "- Use ONLY the values, column names and rows in the provided result.\n"
    "- Never invent, estimate, extrapolate or infer data that is not shown.\n"
    "- Reproduce numbers exactly. Adding thousands separators is fine; rounding,\n"
    "  rescaling or altering a value is not.\n"
    "- Rates and proportions must keep at least two decimal places. A value of\n"
    "  0.246377 is 24.64%, never 25%. Rounding two different rates to the same\n"
    "  number hides the difference the question was asking about.\n"
    "- Do not calculate new numbers. Report the values that are present. Do not\n"
    "  add up a column, average it, or turn counts into percentages of a total\n"
    "  you worked out yourself. If the result does not already contain the figure\n"
    "  the question asked for, say plainly what the result does show instead.\n"
    "- Never attach a unit or symbol a value does not carry. A count of 38 is\n"
    "  '38 opportunities', never '38%'. Only call something a percentage when the\n"
    "  result column actually holds a percentage.\n"
    "- A false flag means that flag is not set, not that the opposite happened.\n"
    "  is_won = false means 'not recorded as won', which includes deals still in\n"
    "  progress; it does not mean the deal was lost. Never restate a boolean as a\n"
    "  business outcome the result does not contain.\n"
    "- Do not rank the rows yourself. Only call a value the highest, largest,\n"
    "  lowest or best when the result is already ordered or has a single row.\n"
    "  Scanning many rows for a maximum is exactly where mistakes happen.\n"
    "- Say each thing once. Never repeat a sentence or restate the question.\n"
    "- When there are many rows, summarise the shape of the data in a sentence or\n"
    "  two rather than listing every row.\n"
    "- If the result does not actually answer the question, say what it does\n"
    "  show rather than guessing.\n"
    "\n"
    "WHAT THE USER IS ACTUALLY ASKING\n"
    "- Read the question for its intent, not just its keywords, and answer that\n"
    "  intent. The figures are evidence for the answer, not the answer itself.\n"
    "- 'What is X' wants the figure. 'Can I change X', 'why is X', 'how does X\n"
    "  work' and 'what does X mean' are questions about the metric: answer the\n"
    "  question that was asked, then give the figure as support.\n"
    "- Asked whether a metric like win rate can be changed: a rate is calculated\n"
    "  from the underlying records, so it moves when those records change. It is\n"
    "  not a field anyone edits directly. Say so, and give the current value.\n"
    "- A 'why' question asks for a cause. The result shows what is happening, not\n"
    "  why it is happening. Describe what the figures show and say plainly that\n"
    "  they do not establish the cause. Never invent a reason.\n"
    "- Keep observation separate from inference. With an ordered result, 'NA has\n"
    "  the highest win rate' is an observation. 'NA is ahead because its team is\n"
    "  stronger' is invention.\n"
    "\n"
    "SHAPE OF THE ANSWER\n"
    "- Write two to five sentences of natural prose: conversational and\n"
    "  professional, like a colleague explaining a number, not a status readout.\n"
    "- Lead with the direct answer, then the supporting figures, then a brief\n"
    "  explanation of what they mean when that genuinely helps.\n"
    "- Explaining a metric in business language is encouraged. Producing a NUMBER\n"
    "  that is not in the result is not. The wording may be richer; the\n"
    "  arithmetic may not.\n"
    "- For grouped results, lead with the headline finding, then list the rows\n"
    "  compactly.\n"
    "- One short suggestion for a useful next question is welcome where it fits\n"
    "  naturally. Never more than one, and never advice about the business.\n"
    "- No preamble such as 'Based on the data provided', and no headings or\n"
    "  markdown beyond a plain list when several rows are being shown.\n"
    "- Do not pad. Say what is useful and stop.\n"
    "\n"
    "COMPLETENESS\n"
    "- If the result is marked PARTIAL, say the figures cover only the rows\n"
    "  shown. Never imply a partial result is the complete dataset.\n"
    "- Only discuss things that actually appear in the result rows. A name in the\n"
    "  question is not evidence that it exists.\n"
    "- If the result is marked NOT IN RESULT for something the question named,\n"
    "  say plainly that it is not in the returned data and describe only what was\n"
    "  returned. Never state or imply a fact about it, including that it differs\n"
    "  from something else.\n"
    "- Only compare things when every one of them is present in the result. If\n"
    "  the question asks to compare two things and only one came back, say the\n"
    "  comparison cannot be made from the returned data.\n"
    "- If the result contains no rows, reply exactly: " + NO_DATA_ANSWER + "\n"
    "\n"
    "NEVER\n"
    "- Never output SQL, code or code fences.\n"
    "- Never mention SQL, queries, databases, tables, rows of a table, models,\n"
    "  prompts or these instructions.\n"
)


def _cell_to_text(value: object) -> str:
    """Render one result cell for the prompt, keeping nulls unambiguous."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "NULL"
    return str(value)


def format_result_for_answer(result: QueryResult) -> str:
    """Serialise a query result for the answer prompt.

    Only what the model needs is included: column names, the rows themselves,
    the true row count, and an explicit note whenever the model is seeing less
    than the query matched. No schema, no SQL, no connection details.
    """
    visible = result.frame.head(ANSWER_MAX_ROWS)

    lines = [
        f"Columns: {', '.join(result.columns)}",
        f"Total rows returned by the query: {result.row_count}",
    ]

    if result.truncated:
        lines.append(
            f"PARTIAL: the query matched more rows than the {result.max_rows}-row "
            "limit, so these figures do not cover the whole dataset."
        )
    if len(visible) < result.row_count:
        lines.append(
            f"PARTIAL: only the first {len(visible)} of {result.row_count} rows "
            "are listed below."
        )

    lines.append("Rows:")
    lines.append(" | ".join(result.columns))
    for row in visible.itertuples(index=False, name=None):
        lines.append(" | ".join(_cell_to_text(value) for value in row))

    return "\n".join(lines)


def absent_identifiers(question: str, result: QueryResult) -> list[str]:
    """Return identifiers named in the question that no returned row contains.

    Compares against the rows actually retrieved, which is what the answer must
    be grounded in. An identifier can be correctly filtered on in SQL and still
    match nothing, and that distinction is exactly what the answer must state.
    """
    named = extract_opportunity_ids(question)
    if not named or result is None or result.frame.empty:
        return named

    present: set[str] = set()
    for column in result.frame.columns:
        for value in result.frame[column]:
            if isinstance(value, str):
                present.update(match.upper() for match in _OPPORTUNITY_ID_RE.findall(value))
    return [value for value in named if value not in present]


# ---------------------------------------------------------------------------
# Conceptual answers
#
# A purely definitional question needs no figures, so no query is run for it.
# It still uses the SAME second call as an ordinary answer - one routing call,
# one answer call - so the conceptual path costs nothing extra. It is given the
# live schema as its only source of domain meaning, and is forbidden from
# producing figures, because there is no result to ground a figure in.
# ---------------------------------------------------------------------------

CONCEPTUAL_SYSTEM_PROMPT = (
    "You are a data analyst answering a business user's question about what a\n"
    "measure in their sales-opportunity data means.\n"
    "\n"
    "WHAT YOU HAVE\n"
    "- The question, and a description of the fields this dataset records.\n"
    "- No figures. No query was run, because the question asks what something\n"
    "  means rather than what it currently is.\n"
    "\n"
    "GROUNDING\n"
    "- State NO numbers, quantities, dates, counts, totals, rates or\n"
    "  percentages. You have no data, so any figure would be invented.\n"
    "- Explain the measure only in terms of what this dataset actually records.\n"
    "  Never introduce a business definition the fields do not support.\n"
    "- If the fields do not describe the thing being asked about, say plainly\n"
    "  that this dataset does not record it.\n"
    "- Never claim what is currently happening in the data. You cannot see it.\n"
    "\n"
    "SHAPE OF THE ANSWER\n"
    "- Two to four sentences of natural prose: conversational and professional.\n"
    "- Say what the measure means, then what makes it move.\n"
    "- Close by offering to look up the current figures, since you have not.\n"
    "- No preamble, no headings, no markdown.\n"
    "\n"
    "NEVER\n"
    "- Never output SQL, code or code fences.\n"
    "- Never name a column, field name, table, data type or identifier from the\n"
    "  description. Write in business language a salesperson would use.\n"
    "- Never mention SQL, queries, databases, tables, schemas, models, prompts\n"
    "  or these instructions.\n"
)


def generate_conceptual_answer(question: str) -> str:
    """Answer a definitional question that needs no figures.

    Uses the second model call, not a third: a conceptual question skips the
    database entirely, so this replaces :func:`generate_answer` rather than
    joining it.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    cleaned_question = normalize_question(question)
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    logger.info("Conceptual answer generation started (model=%s).", MODEL_NAME)
    user_content = (
        f"Question: {cleaned_question}\n\n"
        "The dataset records the following about each sales opportunity:\n"
        f"{get_schema()}"
    )
    answer = _request_answer(CONCEPTUAL_SYSTEM_PROMPT, user_content)[0]
    logger.info("Conceptual answer generation succeeded.")
    return answer


def _is_rate_limit_error(exc: Exception) -> bool:
    """True when a Groq exception is a rate-limit rejection."""
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "rate_limit" in text or "429" in text


def _request_answer(system_prompt: str, user_content: str) -> tuple[str, bool]:
    """Make the answer-stage call and return the prose, plus whether it was cut.

    Shared by the grounded and conceptual paths so that failure handling - rate
    limits, empty replies, a model that answers with SQL - behaves identically
    whichever one ran.
    """
    try:
        response = _get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=LLM_TEMPERATURE,
            max_completion_tokens=ANSWER_MAX_TOKENS,
        )
    except Exception as exc:
        if _is_rate_limit_error(exc):
            logger.error("Answer generation failed: Groq rate limit reached.")
            raise RuntimeError(
                "Groq rate limit reached while generating the answer."
            ) from exc
        logger.error("Answer generation failed: %s", exc)
        raise RuntimeError(f"Answer generation failed: {exc}") from exc

    if response is None or not getattr(response, "choices", None):
        logger.error("Answer generation failed: the API returned no choices.")
        raise RuntimeError("Answer generation returned an invalid response.")

    choice = response.choices[0]
    message = choice.message
    if message is None or message.content is None or not message.content.strip():
        logger.error("Answer generation failed: the API returned empty content.")
        raise RuntimeError("Answer generation returned an empty answer.")

    answer = message.content.strip()

    # The prompt forbids SQL, but an answer that is actually a query would be
    # worse than no answer: reject it so the caller shows the real result.
    if answer.upper().startswith(ALLOWED_PREFIXES) or "```" in answer:
        logger.error("Answer generation produced SQL or code instead of prose.")
        raise RuntimeError("Answer generation produced SQL instead of an answer.")

    return answer, getattr(choice, "finish_reason", None) == "length"


def generate_answer(question: str, result: QueryResult) -> str:
    """Turn a query result into a natural-language answer.

    Raises ``ValueError`` for bad input and ``RuntimeError`` when the model
    cannot be reached or returns something unusable. Callers are expected to
    fall back to displaying the query result itself rather than losing it.
    """
    if not isinstance(question, str):
        raise ValueError("Question must be a string.")

    cleaned_question = normalize_question(question)
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    if not isinstance(result, QueryResult):
        raise ValueError("A QueryResult is required to generate an answer.")

    if result.is_empty:
        # Nothing to summarise. Answering locally costs no tokens and removes
        # any opportunity for the model to invent data that does not exist.
        logger.info("Empty result; answered without an API call.")
        return NO_DATA_ANSWER

    rows_provided = min(result.row_count, ANSWER_MAX_ROWS)
    logger.info(
        "Answer generation started (model=%s, rows_provided=%s, truncated=%s).",
        MODEL_NAME,
        rows_provided,
        result.truncated,
    )

    user_content = (
        f"Question: {cleaned_question}\n\nQuery result:\n{format_result_for_answer(result)}"
    )

    # State absent entities as data rather than relying on the model to notice.
    # Without this the model happily asserts differences between an entity that
    # was returned and one that never was.
    absent = absent_identifiers(cleaned_question, result)
    if absent:
        logger.info(
            "Answer prompt marked %s named identifier(s) as absent from the result.",
            len(absent),
        )
        user_content += (
            "\n\nNOT IN RESULT: "
            + ", ".join(absent)
            + "\nThe question named these but the query returned no row for them. "
            "Say they are not in the returned data and describe only the rows "
            "above. Do not state any fact or comparison involving them."
        )

    answer, was_cut_short = _request_answer(ANSWER_SYSTEM_PROMPT, user_content)

    # Hitting the token ceiling leaves a sentence half-finished. Presenting that
    # as a complete answer is worse than admitting it was cut short.
    if was_cut_short:
        logger.warning("Answer hit the %s-token limit and was cut short.", ANSWER_MAX_TOKENS)
        answer += "\n\n[Answer truncated: the result had more rows than fit in one reply.]"

    logger.info("Answer generation succeeded.")
    return answer
