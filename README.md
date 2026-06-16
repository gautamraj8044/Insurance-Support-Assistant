# Insurance Support Assistant (v2)

A production-oriented insurance customer support chatbot, rebuilt with
**Groq (via langchain-groq)**, **LangGraph**, and a **Streamlit** chat UI.

## What changed from v1

- **LLM provider**: Gemini removed entirely; replaced with `ChatGroq`
  (default model: `llama-3.1-8b-instant`).
- **Frontend**: the old `input()`-based CLI is gone. The app is now a
  Streamlit chat app (`ui/app.py`).
- **Architecture**: the old supervisor-loop (which could re-route
  indefinitely and force-escalate on simple greetings) is replaced with a
  single linear flow: classify intent once, then route to exactly one
  response node. No loops, no iteration caps needed.
- **Intent classification**: a small, separate, fast LLM call classifies
  the user's message before any tool is considered.
- **Tool selection**: a single `database_agent` node owns all six DB
  tools (policy, auto policy, claims, billing, payment history,
  renewal) and decides which to call based on intent + available
  entities; FAQ questions never hit the database, greetings never call
  any tool.
- **Reliability**: every LLM call goes through `safe_invoke()`, which
  never raises - rate limits, timeouts, and API errors degrade to a
  friendly message instead of crashing the app.
- **Logging**: structured, ASCII-only logging to console + rotating file,
  with a short trace id attached to every line for a given user turn.

## Project structure

```
insurance_agent_v2/
├── setup.py                     # one-time DB + FAQ vector store setup
├── ui/
│   └── app.py                   # Streamlit chat app (entry point)
├── core/
│   ├── settings.py               # all configuration (typed, env-driven)
│   ├── logging_setup.py          # structured logging
│   ├── llm_client.py              # ChatGroq wrapper + safe_invoke()
│   ├── exceptions.py              # app-specific exception types
│   ├── tracing.py                 # trace ids + node timing + Phoenix Cloud init
│   └── conversation.py            # turn orchestration (Streamlit <-> graph)
├── graph/
│   ├── state.py                   # GraphState TypedDict
│   ├── entity_extraction.py       # regex ID extraction (policy/customer/claim)
│   ├── builder.py                 # compiles the LangGraph workflow
│   └── nodes/
│       ├── intent_classifier.py   # entry point: classifies intent
│       ├── greeting.py            # greeting -> direct response, no tools
│       ├── faq.py                 # FAQ -> ChromaDB RAG response
│       ├── database_agent.py      # account lookups via tool calling
│       └── misc_nodes.py          # clarification / escalation / general chat
├── tools/
│   ├── db_tools.py                 # SQLite query functions
│   └── faq_tool.py                 # ChromaDB FAQ search
├── prompts/
│   └── templates.py                # every prompt template
└── data/
    ├── generator.py                 # synthetic data generator
    ├── db_setup.py                  # SQLite schema + seeding
    └── faq_setup.py                 # loads FAQ dataset into ChromaDB
```

## Graph flow

```
START
  -> classify_intent
       -> greeting              -> greeting_node            -> END
       -> human_escalation      -> human_escalation_node     -> END
       -> clarification_needed  -> clarification_node        -> END
       -> faq                   -> faq_node                  -> END
       -> requires_tool=true    -> database_agent_node       -> END
       -> else                  -> general_conversation_node -> END
```

Each user turn does exactly one intent classification call, then exactly
one response-generation call (two if a database tool is invoked, to turn
raw tool output into a natural-language answer). There is no re-entrant
supervisor loop, so there's no risk of the old "max iterations reached ->
force escalate" failure mode on simple greetings.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY
```

### 3. Run one-time setup (creates SQLite DB + loads FAQ vectors)
```bash
python setup.py
```

### 4. Launch the app
```bash
streamlit run ui/app.py
```

## Running with Docker

```bash
docker compose up -d phoenix      # start Phoenix tracing (optional)
docker compose run --rm setup     # one-time DB + FAQ setup
docker compose up -d app          # start the Streamlit app
```

The app will be available at **http://localhost:8501**, and the Phoenix
tracing UI at **http://localhost:6006** (if started).

If you want tracing, set in `.env`:
```
ENABLE_TRACING=true
```
The `app` service automatically points at `http://phoenix:6006/v1/traces`
(the Phoenix container's name on the Docker network) regardless of what
`PHOENIX_COLLECTOR_ENDPOINT` is set to in `.env` - that variable is only
used when running the app outside Docker, where `phoenix` as a hostname
wouldn't resolve. No `PHOENIX_API_KEY` is needed for this local setup.

## Configuration reference

All settings are read once in `core/settings.py` from environment
variables (see `.env.example` for the full list). Key ones:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Main response-generation model |
| `GROQ_INTENT_MODEL` | same as `GROQ_MODEL` | Model used for intent classification |
| `GROQ_MAX_RETRIES` | `2` | SDK retry attempts before failing fast |
| `FAQ_TOP_K` | `3` | Number of FAQ matches retrieved per query |
| `DEBUG_MODE` | `false` | Reserved for future verbose-mode toggling |

## Observability and tracing

Two layers of observability are built in, and they work independently:

**1. Structured logging (always on, no setup needed).** Every node logs
its start/finish and duration via the `timed_node` decorator, and every
log line in a given user turn shares a short trace id (e.g.
`[trace=a1b2c3d4]`), so you can `grep` one turn's full path through the
graph in `logs/app.log`.

**2. Phoenix tracing (optional, requires setup).** Set `ENABLE_TRACING=true`
to send real OpenTelemetry spans to Phoenix, giving you a visual trace of
every LLM call (intent classification, greeting, FAQ, database agent)
including prompts, completions, and latency. Works in two modes:

### Local Phoenix via Docker (no API key needed)

This is the default and what `docker-compose.yml` is set up for:

```bash
docker compose up -d phoenix
```

Then in `.env`:
```
ENABLE_TRACING=true
```
That's it - no `PHOENIX_API_KEY` required for local/self-hosted Phoenix.
View traces at **http://localhost:6006**.

If running the app with `docker compose up -d app`, the endpoint is
already wired to the `phoenix` container automatically (see
`docker-compose.yml`). If running the app outside Docker (plain
`streamlit run ui/app.py`) while Phoenix runs in Docker, keep the default
`PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces` in `.env` -
Docker's port mapping (`-p 6006:6006`) makes it reachable at `localhost`
from the host machine.

### Phoenix Cloud instead

1. Sign up / log in at https://app.phoenix.arize.com (a different
   product from `app.arize.com` - keys are not interchangeable).
2. Go to **Settings > API Keys** and create a key.
3. Set in `.env`:
   ```
   ENABLE_TRACING=true
   PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com
   PHOENIX_API_KEY=your_phoenix_cloud_key
   ```

### Either mode

Install the optional tracing dependencies first:
```bash
pip install arize-phoenix-otel openinference-instrumentation-langchain
```
(Already included in `requirements.txt` and the Docker image.)

The sidebar will show "Phoenix tracing enabled" once it's working.
Traces appear under the `insurance-support-assistant` project
(configurable via `PHOENIX_PROJECT_NAME`).

If `ENABLE_TRACING=true` but a Cloud key is missing (and the endpoint
isn't local), or the optional packages aren't installed, the app logs a
warning and continues working normally - tracing is purely additive and
never blocks the chat functionality.


## Notes on tool selection rules

- FAQ tool is used for: coverage questions, claim procedures, renewal
  process, documentation requirements, payment methods, and general
  insurance questions - **never** touches the database.
- Database tool(s) are used for: policy details, customer records, claim
  status, premium due dates, payment history, and renewal status -
  **never** called for greetings or small talk.
- If required identifying information (policy number, customer ID, claim
  ID) is missing, the database agent asks for it directly rather than
  guessing or fabricating data.
