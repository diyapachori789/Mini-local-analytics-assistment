# Analytics Assistant

A tiny, single-engine version of a SQL metrics engine for Salesforce-style opportunity
data. Ask a question in plain English; the assistant writes SQL, runs it against a local
DuckDB database, and answers in plain English with exact numbers from the data.

```
Question: What is the total amount of closed won opportunities?

Answer:
The total amount of closed won opportunities is $5,329,008.
```

Everything except the language-model calls runs locally. The dataset never leaves the
machine. Analytics planning receives the table schema and current question; grounded
answering receives only the question and returned rows. For an active follow-up, only
the bounded safe conversation window needed to resolve meaning is included, never as
data truth. A general conversational answer receives no schema or query result.

---

## What it does

- Loads `data/sample_opportunities.csv` (300 synthetic Salesforce opportunities) into DuckDB
- Accepts a plain-English business question
- Handles greetings, thanks, identity/capability questions, acknowledgements, and goodbyes naturally
- Uses an LLM to write DuckDB SQL from the live table schema
- Validates the SQL and refuses anything that is not a single read-only statement
- Executes it against DuckDB, so the numbers come from the data rather than the model
- Sends the result back to the LLM to phrase a natural-language answer
- Draws a matplotlib PNG when a visualization meaningfully improves the result, or when a compatible chart is requested
- Logs every generated query so any number can be re-checked

---

## Pipeline

```
Plain-English question
        ↓
 explicit chart intent  ──  deterministic presentation wording, if present
        ↓
   Groq call #1  ──  schema-aware routing + SQL generation
        ↓
    QueryPlan   ──  {intent, sql}   sql may be null
        ↓
   ┌────┴──────────────┬─────────────────────┐
   ↓                   ↓                     ↓
 sql_guard        no SQL needed          not answerable
 validate         (conceptual)           (unsupported/unsafe/
   ↓                   ↓                  needs context)
 DuckDB           Groq call #2                ↓
 executes once    explains, no figures   friendly refusal
   ↓                   ↓                  (no model call)
QueryResult       Answer
   ├──────────────────────────────┐
   ↓                              ↓
 Groq call #2                matplotlib
 answer grounded             PNG in charts/
 strictly in those rows      (only if useful/requested)
   ↓                              ↓
Natural-language answer     "Open chart" link
```

The answer and the chart are built from the **same** `QueryResult`. The query runs
once; nothing is recalculated, and no second model call is made for the chart.
Every branch above costs **at most two** model calls.

`GENERAL_CONVERSATION` is the additional result-free branch: routing is call #1,
and a schema-free conversational response is call #2. It never constructs SQL,
touches DuckDB, creates a `QueryResult`, or enters chart recommendation. Unsupported
and unsafe requests still use locally written scope/safety replies after routing;
deterministic metadata requests are refused before any model call.

### 1. The question is routed, and English becomes SQL

`llm.generate_plan()` builds a system prompt containing the **live schema**, read from
`information_schema` at runtime rather than hard-coded, so the prompt can never drift from
the table. That one call answers two things at once — what kind of request this is, and
what data would support an answer — so routing costs no extra quota. It returns a small
JSON object that `intent.py` parses into a `QueryPlan`:

```json
{"intent": "DATA_EXPLANATION", "sql": "SELECT stage, SUM(amount) ... ;"}
```

The prompt states the output contract, the routing rules, the read-only rules,
column-selection rules and row-limit rules, followed by worked examples. Generation runs at
`temperature=0` for repeatability.

**Why routing is a separate concept.** The prompt used to ask only "can this sentence be
turned into SQL?", with `INVALID_QUESTION` as the sole escape hatch. Anything that was not
a direct lookup fell through it, so `How can I change the win rate?` was refused while
`can i change the win rate?` happened to survive — a difference in phrasing, not in
meaning. The prompt now asks "what data would best support an answer?", and the intent
records why. Nothing in the routing code names a metric, a column or a phrase; the live
schema is the only description of what can be answered.

| Intent | Meaning | SQL | Outcome |
|---|---|---|---|
| `DATA_QUERY` | a figure, list, ranking, breakdown or comparison | yes | answer + table + optional chart |
| `DATA_EXPLANATION` | what a measure means, what moves it, why it looks as it does | usually | same, worded as an explanation |
| `DATA_EXPLANATION` | purely definitional | null | explanation with **no** figures, no query |
| `GENERAL_CONVERSATION` | greeting, thanks, identity, capabilities, acknowledgement, goodbye | null | brief natural reply; no result or chart |
| `INSUFFICIENT_CONTEXT` | only makes sense as a follow-up | null | asks for the whole question |
| `UNSUPPORTED` | not about this data | null | friendly refusal |
| `UNSAFE` | modify, administer, or reveal structure | null | safety refusal |

Intent values are internal. They are never serialized to the browser.

`llm.generate_sql()` remains as the string-returning view of the same call, and still
returns `INVALID_QUESTION` when there is nothing to run.

### 2. SQL is validated and kept read-only

`sql_guard.py` holds every safety rule in one dependency-free module. Before anything runs,
the statement must:

- begin with `SELECT` or `WITH`
- be exactly one statement, ending in a single semicolon
- contain none of 21 forbidden keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`,
  `ALTER`, `TRUNCATE`, `COPY`, `ATTACH`, `PRAGMA`, …)

Checks run against a **masked copy** of the statement, with string literals, quoted
identifiers and comments blanked out first. Without that, an ordinary text search such as
`WHERE notes ILIKE '%delete%'` would be rejected as a DELETE statement. Masking fails
closed: an unterminated quote swallows the rest of the statement, which is then rejected.

Three further layers back this up:

| Layer | Protection |
| --- | --- |
| `database.run_query()` | Re-validates independently, so a bug in a caller cannot become a write |
| DuckDB latch | Startup makes a best-effort attempt to switch `enable_external_access` off after load. When DuckDB accepts it, the setting is one-way; if it cannot be applied, startup logs a warning and the SQL guard plus database revalidation remain in force. |
| Answer step | Receives only the result rows; it never sees the schema and never produces SQL |

### 3. DuckDB executes the query

`database.run_query()` runs the validated statement **verbatim** and returns a
`QueryResult` with the rows, column names, row count, and a truncation flag.

Results are bounded by stopping the fetch after `MAX_RESULT_ROWS` (one extra row is read to
detect truncation) — never by rewriting the SQL. Appending a `LIMIT` would silently change a
query that already has its own `LIMIT` or `ORDER BY`; stopping the fetch cannot. A query
asking for 3 rows still returns exactly 3.

### 4. The result becomes an answer

`llm.generate_answer()` sends the original question plus the rows — column names, up to
`ANSWER_MAX_ROWS` rows, the true row count, and an explicit `PARTIAL` marker when the model
is seeing less than the query matched. It never receives the schema or the SQL, so it has
nothing to build a query from.

Safety properties of this step:

- An empty result is answered locally with `No matching records were found.` and **no API
  call** — nothing to summarise means no opportunity to invent data
- Values reach the model raw (`5329008`, not `5,329,008`); formatting is the model's job,
  accuracy is the code's
- An answer that is actually SQL is rejected, and the raw rows are shown instead
- If answer generation fails for any reason — API error, rate limit, empty response — the
  database result is still displayed. A failed second call never discards a successful query

### 5. Charts

Chart usefulness is decided locally after DuckDB returns the one authoritative
`QueryResult`. The decision combines generic question/query semantics with row count,
column types, and result shape. A scalar, a single entity, an empty result, or a large raw
table does not get an automatic chart merely because matplotlib could draw something.

Categorical comparisons normally become bars, temporal measures become lines,
part-to-whole results can become pies when the slice count and values are suitable, and
relationships between two meaningful numeric measures can become scatter plots. These
automatic decisions use no model call and no second query.

`is_chart_request()` separately detects explicit wording — *chart, graph, plot,
diagram, visualise* — so a natural-language request is still honoured when the returned
data supports a meaningful visualization. Internally the service distinguishes automatic,
explicit, and no-chart outcomes; those policy labels are never serialized to the browser.

The charting words are then stripped before the question reaches either LLM call.
*"…and chart it"* cannot be expressed in SQL, and leaving it in invites a refusal:

```
"How many opportunities did each owner close won? — and chart it."
        →  "How many opportunities did each owner close won?"
```

**Supported types:** bar, line, pie, scatter.

If the question names a type (*"as a pie chart"*, *"bar graph"*, *"scatter plot"*), that
type is used. Otherwise it is chosen from the question and the shape of the result:

| Chosen | When |
| --- | --- |
| **Line** | A date/time column is present, or the wording is about a trend over time |
| **Scatter** | Two numeric columns and relationship wording (*versus*, *correlation*) |
| **Pie** | Composition wording (*share*, *proportion*, *breakdown*), few categories, no negatives |
| **Bar** | Everything else — comparisons and rankings. Horizontal beyond 6 categories |

An explicit request is honoured wherever it is sound, but the **data is never reshaped to
satisfy an unsuitable one**. Asking for a pie chart of a time series produces a line chart
plus an explanation of why. The same applies to a pie chart with too many slices or with
negative values, and to a scatter plot without two numeric columns.

PNGs are written to `charts/`, created automatically, with a timestamped filename so an
existing chart is never overwritten:

```
charts/how_many_opportunities_did_each_owner_cl_20260810_235015.png
```

The file stays on disk after the app exits.

### 6. Persistent conversations

The browser groups turns into persistent conversations in `history.duckdb`, so
chats survive browser and Flask restarts, Docker recreation, and cleared browser
storage. The database is authoritative; JavaScript memory and `localStorage` are
not conversation storage. On refresh, the browser restores the most recently
updated saved chat before allowing another message; starting a different chat is
an explicit **+ New Chat** action.

`conversations` holds an opaque UUID, a locally generated safe title, and UTC
timestamps. `conversation_messages` holds ordered user/assistant prose plus safe
row, chart, refusal, success, and elapsed-time metadata. It deliberately excludes
generated SQL, schemas, prompts, API keys, raw provider payloads, absolute paths,
exception text, and full result rows.

Only the latest six safe user/assistant messages (2,400 characters total and 600
characters each) can reach routing as untrusted semantic orientation. This helps
resolve follow-ups but never supplies data truth: DuckDB is queried anew for current
figures. For `GENERAL_CONVERSATION`, the same bounded safe prose may orient the
schema-free conversational answer. It never reaches grounded result answering,
charting, or DuckDB, and no route adds a third model call.

Pure conversational turns are persisted as ordinary user/assistant pairs with nullable
analytics metadata. They show no row count, data toggle, chart, or “Analysis complete”
status. A social opening keeps the neutral **New chat** title; the first meaningful
analytics question supplies the stable local title without an extra model call.

Opening a saved chat loads messages only; it never calls Groq or reruns analytics.
Current-turn rows can be expanded inline, while old messages explain that detailed
rows were not persisted. Existing PNGs load by safely revalidated filename; a
missing PNG is reported as unavailable rather than regenerated.

The stored transcript remains complete, but a reopened view returns its newest 200
messages at once. If a chat is longer, the API marks it as truncated and the UI
shows an explicit notice rather than silently presenting a partial transcript.

Use **+ New Chat** to begin another persisted conversation. Deleting one chat or
**Delete all chats** requires confirmation and affects conversation records only,
never analytics data, charts, logs, or `.env`.

The CLI deliberately keeps its established simple compatibility model: each
completed CLI request is a standalone legacy history record. When conversation
storage is initialized, any previously unmapped legacy record is migrated into
its own one-turn conversation without changing or removing the original record.

#### Legacy query-history archive

**Where:** `history.duckdb` in the project root — a **separate** DuckDB file from
`analytics.duckdb`. That separation is deliberate:

- `initialize_database()` rebuilds `opportunities` from CSV on every start, and
  `analytics.duckdb` is disposable — deleting it is a supported way to reset the
  dataset. History stored there would be lost with it.
- Each database keeps its own single-writer lock, so history never contends with
  analytics.
- Analytics initialization makes a best-effort attempt to disable DuckDB external
  access. If that setting cannot be applied, startup logs a warning and continues;
  generated SQL is still protected by the shared SQL guard and database revalidation.

**Table `query_history` stores:** id (server-generated UUID), UTC timestamp, the
original question, the generated answer, row count, truncation flag, max rows,
chart requested/type/filename/note, answer-fallback flag, refused flag, success
flag, elapsed seconds, and an optional error code.

**Deliberately not stored:** generated SQL, the database schema, prompts, API
keys, stack traces or raw exception text, absolute paths, request headers, and
full result rows. History is a lightweight record of questions and answers, not a
cache of business data or an audit-query browser.

**Migration:** when conversation storage is initialized, each unmapped legacy
record is copied atomically into a one-turn conversation containing a user and
assistant message. `legacy_history_migrations` records the mapping, so repeated
initialization never duplicates conversations and the original rows remain
available.

**Charts:** only the bare filename is stored, never a path. On read the filename
is re-validated and checked against the charts directory; if the PNG has been
removed the record still loads and is shown as *Chart unavailable*. A missing
chart is never regenerated and never re-runs the query.

**Compatibility API:** `GET` and `DELETE /api/history` remain for the CLI and
older integrations. The browser uses `/api/conversations`; its **Delete all
chats** control does not delete legacy archive rows, charts, logs, analytics data,
or `.env`.

**Privacy boundary:** legacy archive rows are not treated as global model memory.
Only the bounded safe messages from the active saved conversation can provide
semantic orientation to the planner; opening either legacy history or a saved chat
never reruns an analysis automatically.

Browser `localStorage` now holds display preferences only (theme and selected
view). Any history left by an earlier build is discarded on load.

### 7. Questions it will not answer

Some questions cannot be answered from sales data. Those are **not errors** — the
request succeeded, there was simply nothing to compute. The assistant replies in
the normal result area with a friendly explanation and a suggestion, with no red
error styling, no table, and no chart.

| Category | Example | Reply points you toward |
| --- | --- | --- |
| Database structure | *"Show all tables"*, *"What columns are in the database?"* | business questions instead of schema |
| Unsafe or admin SQL | *"DROP TABLE opportunities"* | safe read-only analytics |
| Unrelated | *"What is the weather today?"* | the sales dataset |
| Unsupported phrasing | a malformed question | a concrete example that works |

Ten replies are written locally from templates. The category comes from the
wording; the specific reply comes from a stable hash of the question, so **the
same question always gives the same reply** while different questions in a
category vary. There is no third model call and no quota spent to say "I can't
do that", and `random` is not used, so tests never flake.

**Structure questions are also blocked at the database layer.** `sql_guard`
rejects `information_schema`, `duckdb_*()` catalog functions, `pragma_*()`,
`sqlite_master` and `pg_catalog`. Those are readable with an ordinary `SELECT`,
so the keyword ban alone did not stop them — without this rule, *"show all
tables"* could return the full `CREATE TABLE` definition. Internal code that
legitimately needs the schema reads it on a direct connection and never passes
through the validator.

Genuine failures — the model being unreachable, a timeout, a database error —
still surface as errors. Only unsupported questions become friendly replies.

---

## Technologies

| Component | Choice |
| --- | --- |
| Database | DuckDB 1.5.5 (embedded, local file) |
| LLM | Groq — `openai/gpt-oss-120b` |
| Charts | matplotlib (Agg backend, no display required) |
| Data handling | pandas |
| Config | python-dotenv |
| Tests | pytest |
| Language | Python 3.11+ (developed on 3.13) |

---

## Project structure

```
config.py            paths, model, limits, .env loading
sql_guard.py         SQL cleaning + read-only validation (no other dependencies)
database.py          DuckDB lifecycle, schema extraction, guarded execution
intent.py            routing decision: parses {intent, sql} into a QueryPlan
llm.py               routing/SQL plus one grounded, conceptual, or conversational answer
chart.py             chart intent, type selection, matplotlib rendering
history_repository.py legacy archive + persistent conversations (separate history.duckdb)
logging_config.py    rotating file + console logging, API-key redaction
app.py               command-line interface
data/                sample_opportunities.csv
charts/              generated PNGs
tests/               pytest suite (offline by default, live tests opt-in)
logs/                app.log — every generated query
TESTING.md           regression test document
```

---

## Setup

Requires Python 3.11+ and a Groq API key ([console.groq.com](https://console.groq.com)).

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b   # optional, this is the default
```

`.env` is git-ignored and the key is never logged — a redaction filter scrubs it from every
log record as a second line of defence.

---

## Running it

```bash
python app.py
```

Then type questions. A blank line, `exit`, or `quit` ends the session; `Ctrl+C` also works.

```
----------------------------------------
Database initialized successfully
Ask a business question. Blank line or 'exit' to quit.

Question: What is the win rate by region?

Answer:
Win rate by region: NA 25.00%, EMEA 24.64%, LATAM 23.19%, APAC 21.43%.

Question: exit
```

The SQL and row count are written to `logs/app.log` rather than the screen, so the answer
stays the focus while every number remains checkable.

### Asking for a chart

```
Question: How many opportunities did each owner close won? — and chart it.

Answer:
Each owner closed the following number of opportunities:
A. Rao closed 15 opportunities.
C. Mehta closed 19 opportunities.
B. Singh closed 9 opportunities.
E. Shah closed 15 opportunities.
D. Patel closed 13 opportunities.

Chart generated successfully:
Open chart
```

`Open chart` is a clickable OSC 8 terminal hyperlink pointing at the PNG. Clicking it
opens the image in the default handler. In terminals that do not support hyperlinks — and
whenever output is piped or redirected — the absolute path is printed instead:

```
Chart generated successfully:
C:\Projects\Mini Local Analytics\charts\how_many_opportunities_did_each_owner_cl_20260810_235015.png
```

The application never fails because hyperlinks are unsupported. There is no web server and
no browser UI.

### Asking for a specific chart type

```
Question: Show opportunity share by stage as a pie chart.

Answer:
Opportunity share by stage is 16.67% Qualification, 12.67% Prospecting,
23.67% Closed Won, 15.00% Proposal, 11.67% Negotiation, and 20.33% Closed Lost.

Chart generated successfully:
Open chart
```

A scalar or otherwise unsuitable result produces an answer without a PNG, even if charting
is technically possible. A comparison or trend can receive an automatic chart without
requiring special chart wording.

---

## Web UI

Phase 7 adds a local Flask dashboard alongside the CLI. It uses server-rendered
HTML, custom CSS, and vanilla JavaScript; there is no frontend build step and no
browser-side analytics calculation.

Install the runtime dependencies, then choose either entry point:

```bash
pip install -r requirements.txt

# Command-line assistant
python app.py

# Local browser dashboard
python web_app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) after starting the web UI.
The server binds only to loopback and starts DuckDB once, rather than on each
page request.

The browser presents the user-facing **Analytics Assistant** brand, an empty-chat welcome
screen with safe question suggestions, a conversation stream, and a question-only bottom
composer. There is no chart toggle or chart-type selector: users can ask naturally and the
backend decides locally whether a chart adds value. **+ New Chat** creates a blank persisted chat without
deleting previous ones. The sidebar groups saved chats by the browser's local date
and loading one is read-only: it fetches only saved messages.

Each submitted message follows the same shared backend service as the CLI: at
most one generated analytical SQL statement executes, and the resulting
`QueryResult` supplies the grounded answer, expandable current-turn table, and
optional existing matplotlib PNG. Charts appear inside the assistant message.
Historical messages never regenerate data; saved result rows are intentionally
unavailable after reload.

Greetings and other general conversational turns use the same assistant-message UI,
but execute no analytical SQL and display no table, row count, chart, or permanent
completion banner. Mixed messages such as a greeting followed by a data request still
take the analytics route, and deterministic metadata/security gates remain first.

The normal UI intentionally provides no raw SQL, schema browser, log viewer, CSV
upload, browser-stored transcript, or multi-user support.

CSV storage, DuckDB execution, SQL validation, result handling, and chart
rendering are local. The configured Groq API receives the information required
for SQL generation and answer phrasing. Result tables disclose when the fetch
limit truncated rows, and browser errors are intentionally concise.

## Example questions

All five below were verified against hand-written SQL on the 300-row dataset.

| # | Question | Verified answer |
| --- | --- | --- |
| 1 | What is the overall win rate? | 23.67% (71 of 300) |
| 2 | What is the win rate by region? | NA 25.00%, EMEA 24.64%, LATAM 23.19%, APAC 21.43% |
| 3 | Total open pipeline amount by stage. | Qualification 4,224,186 · Proposal 3,826,472 · Prospecting 3,574,616 · Negotiation 3,392,955 |
| 4 | Top 5 accounts by total opportunity value. | Vertex Labs 1,161,424 · Acme Labs 1,049,727 · Orion Industries 1,009,478 · Pinnacle Systems 954,086 · Summit Labs 895,588 |
| 5 | How many opportunities did each owner close won? | C. Mehta 19 · A. Rao 15 · E. Shah 15 · D. Patel 13 · B. Singh 9 |

Also handled: `What is the total amount of closed won opportunities?` → **5,329,008**.

### Chart examples

| Question | Chart |
| --- | --- |
| How many opportunities did each owner close won? — and chart it. | **Bar** (chosen automatically — a categorical ranking) |
| Show opportunity share by stage as a pie chart. | **Pie** (explicitly requested) |
| Show the total opportunity amount by close month as a line chart. | **Line** (explicitly requested) |
| Show the relationship between total amount and deal count per account as a scatter plot. | **Scatter** (explicitly requested) |
| Show monthly pipeline as a pie chart. | **Line**, with an explanation — a pie chart of a time series would mislead |

Out-of-scope questions are refused rather than guessed — *"What is the capital of France?"*
returns *"This question cannot be answered from the available data."*

---

## Running in Docker

**Prerequisite:** Docker Desktop with the WSL2 backend enabled (`wsl --status` should
report Default Version 2, and `docker info` should show `osType: linux`).

```powershell
docker compose up --build     # build and start
docker compose up -d          # start detached
docker compose logs -f web    # follow logs
docker compose down           # stop and remove
```

Then open **http://127.0.0.1:8000** — the same URL as the local run.

Inside the container Flask binds `0.0.0.0:8000` (a container must bind all
interfaces to receive its published port), but Compose publishes it as
`127.0.0.1:8000:8000`, so it is reachable **only from this machine**, never from
the network.

### Docker uses its own databases

This is the one behavioural difference worth knowing:

| Path | Used by |
| --- | --- |
| `analytics.duckdb`, `history.duckdb` (project root) | the **local** Python run |
| `docker-data/analytics.duckdb`, `docker-data/history.duckdb` | the **container** |

They are deliberately separate. DuckDB writes a `.wal` file *beside* its database,
so bind-mounting individual database files would leave that write-ahead log inside
the container and lose it on recreation. Mounting a directory keeps the database
and its `.wal` together on the host. It also means your existing local history was
never migrated or overwritten.

The container builds its own `opportunities` table from the same
`data/sample_opportunities.csv` on first start, so `docker-data/` needs no seeding.

### What persists

| Path | Mount | Survives `docker compose down` |
| --- | --- | --- |
| `./docker-data` | read-write | yes — both databases and their `.wal` files |
| `./charts` | read-write, shared with local | yes |
| `./logs` | read-write, shared with local | yes |
| `./data` | **read-only** | source CSV, never written |

Charts and logs are shared with the local run, so a chart generated in Docker is
visible to the local app and vice versa.

### API key

`GROQ_API_KEY` is supplied at runtime through Compose's `env_file: .env`. It is
**not** copied into the image — `.dockerignore` excludes `.env` from the build
context entirely.

> **Caution:** `docker compose config` prints the fully resolved configuration,
> including the value of `GROQ_API_KEY`. Avoid it in shared terminals or logs; use
> `docker compose config --no-interpolate` if you need to inspect the file.

### One instance only

DuckDB is embedded and file-based: exactly one process may hold a database file.
Do not add replicas, do not start a second container against the same
`docker-data/`, and do not run the local CLI or local Flask against the *same*
files as a running container. Because Docker uses `docker-data/` and local uses
the project root, the container and the local test suite **can** run at the same
time — but two containers cannot.

### Local Python is unaffected

Docker is optional. The `.venv` workflow is unchanged and requires no Docker:

```powershell
.\.venv\Scripts\python.exe app.py            # CLI
.\.venv\Scripts\python.exe -u web_app.py     # local web, still 127.0.0.1:8000
```

---

## Testing

```bash
pip install -r requirements-dev.txt

pytest                          # offline suite: no API key, no network, ~2s
pytest -m live                  # live SQL-generation regression (uses Groq quota)
pytest -m "live or not live"    # everything
```

The offline suite is the default. Live tests are deselected via `pyproject.toml` and skip
automatically when `GROQ_API_KEY` is absent, so the suite runs in CI without secrets. The
answer-generation tests use a mocked Groq client and consume no tokens.

Chart tests render real PNGs through matplotlib's Agg backend from mocked query results,
so they verify actual image output without a display and without touching the API.

Assertions target SQL *structure* (`contains LIMIT 5`, `does not contain SELECT *`) rather
than exact strings, because exact-match assertions on model output are brittle.

`TESTING.md` documents the full case list and a regression checklist to run after any prompt
change.

---

## Logging

Two rotating log files, both 1 MB × 3 backups, both written to `logs/`.

| File | Contents |
| --- | --- |
| `logs/app.log` | Everything: database initialization, every generated SQL statement with its row count, both LLM calls, validation failures, execution errors, and all web requests |
| `logs/web_app.log` | Web layer only: server startup with host and port, request start and completion, row count and chart outcome, history-persistence failures, rejected chart requests, and Werkzeug's request lines |

`web_app.log` is a filtered view for reviewing a browser session without reading
past CLI and analytics entries — `app.log` remains the complete record, so nothing
is moved out of it. It is created when `web_app.py` starts; the CLI does not write
to it.

The console stays at WARNING so normal output is clean; both files capture INFO.
This is why Werkzeug's `* Running on http://127.0.0.1:8000` banner appears in the
log files rather than the terminal — `web_app.py` prints the address explicitly at
startup instead.

**Never logged:** the API key, `.env` contents, request headers, raw prompts, full
result datasets, or provider payloads. A `SecretRedactingFilter` is attached to
every handler, including the web one, and replaces the key with `***REDACTED***`
if it ever appears in a message, argument, or exception. Stack traces go to the
log files only and are never returned to the browser.

---

## Notes on the model

Currently `openai/gpt-oss-120b`.

The project has now migrated twice. It started on `llama-3.3-70b-versatile`, moved to
`llama-3.1-8b-instant` on 2026-08-10 when the 70b daily token quota ran out, and moved to
`openai/gpt-oss-120b` on 2026-08-17 when Groq retired **both** Llama models. Every request
was failing with a model-not-found error, which the browser reported as
"The language model is unavailable. Please try again later."

Groq suggested two replacements. `qwen/qwen3.6-27b` was tested and rejected: it writes its
chain of thought into `message.content`, so the routing JSON arrives wrapped in `<think>`
blocks and the reply hits the token ceiling before the answer starts. Reading that would
mean loosening the plan parser, which is the one component that has to stay strict.
`openai/gpt-oss-120b` returns reasoning in a separate field and leaves `content` clean.

The prompts were tuned against the 70b model and transferred to the 8b model intact,
including the trickiest behaviour (turning *"top 5"* into `ORDER BY … LIMIT 5`). Re-run the
regression suite after any model change.

---

## What I learned

Putting the database between the model and the answer is what makes the numbers
trustworthy. The model decides *what to compute*; DuckDB decides *what the value is*. That
split means a wrong answer is almost always a wrong query, which is inspectable and fixable,
rather than a plausible-sounding number nobody can trace.

I also learned that prompt rules only work if they match how a model actually reads them.
The instruction *"Do not use LIMIT unless the user explicitly asks for a limited number of
rows"* failed on *"top 5"* — the phrase contains neither "limit" nor "rows". Rewriting the
same rule as a positive requirement with the trigger phrasings listed fixed it immediately.

The sharpest lesson came from charts. A question asking for a *share* returned raw counts,
and the model dutifully labelled a count of 38 as "38%". I tried three times to forbid that
in the answer prompt and failed each time — because I was fixing the wrong layer. The right
fix was upstream: make the SQL compute the percentage, so the number comes from the database
instead of the model. That is the whole thesis of the project, and I had to relearn it.

## What surprised me

Three things. First, that the safety filter was rejecting *legitimate* queries: scanning raw
SQL for banned keywords meant `WHERE notes ILIKE '%delete%'` looked like a DELETE statement,
so the fix was to validate a copy with string literals masked out. Second, how many bugs
were invisible to code review and only appeared when running the thing — a UTF-16 encoding
that made `pip install -r requirements.txt` fail outright, and a byte-order mark from piped
input that slipped past an "is the question empty?" check. Third, that a smaller, faster
model handled this task just as well as a model nine times its size once the prompt was
specific enough.

## What I would improve with more time

Prose summaries of wide results are the weakest part. A 49-row scatter question produces a
long, repetitive answer, because a small model asked to describe fifty rows will list them
rather than characterise them — the chart is the right output for that question, not the
paragraph. I would aggregate large results before they reach the answer step rather than
relying on the prompt to discourage listing. Beyond that I would add a retry loop that feeds
a DuckDB error back to the model so it can repair its own SQL, cache the schema instead of
querying it once per question, and reload the CSV only when it has actually changed so the
app could run on a read-only connection.

---

## Known limitations

- **Long results summarise poorly.** With many rows the answer tends to list rather than
  summarise, and can hit the token ceiling — the reply is then explicitly marked as
  truncated rather than presented as complete.
- **One process at a time.** DuckDB allows a single writer, so `python app.py` and the test
  suite cannot run simultaneously.
- **Dataset quirk.** 100 of the 300 sample rows have `close_date` earlier than
  `created_date`, so any sales-cycle-duration metric will produce negative values. This is a
  fixture artifact, not a query bug.
- **Terminal hyperlinks depend on the terminal.** VS Code and Windows Terminal support
  them; elsewhere the absolute path is printed instead.
