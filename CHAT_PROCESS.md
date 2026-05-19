# Chat API Process

## Overview

The chat API provides a natural language interface to the trained MASAC portfolio system. Users send queries in plain English; the system returns three side-by-side portfolio panels (Cooperative, Competitive, Mixed) with per-asset allocations, ESG metrics, and Sharpe statistics.

The chat layer is powered by **Google ADK** with **Gemini** as the underlying LLM. The system uses a **3-agent architecture**:

| Agent | Model | Role |
|---|---|---|
| `portfolio_advisor` | `gemini-2.5-flash` | Orchestrator — parses intent, calls tools, synthesises all output |
| `market_intelligence` | `gemini-2.5-flash-lite` | Sub-agent — live macro/sector/earnings research via Google Search |
| `esg_research` | `gemini-2.5-flash-lite` | Sub-agent — Bloomberg vs LESG ESG ratings, controversies, ΔESG context |

An **input rail** (Gemini Flash-Lite pre-check) runs before the advisor on every request to block off-topic, abusive, system-probing, and jailbreak messages. Session history is persisted to PostgreSQL via `DatabaseSessionService`.

Two endpoints are available:
- `POST /api/v1/chat` — blocking, returns full response when complete
- `POST /api/v1/chat/stream` — Server-Sent Events, emits live status + text chunks as the pipeline progresses

---

## End-to-End Flow

```
POST /api/v1/chat  (or /chat/stream)
  { "message": "...", "session_id": "optional-uuid" }
         │
         ▼
  ┌─────────────────────────────────────────┐
  │  INPUT RAIL (Gemini Flash-Lite)          │
  │  Classifies message into one of:        │
  │  relevant | off_topic | abusive |        │
  │  system_probe | jailbreak               │
  │                                         │
  │  Blocked? → canned response, stop.      │
  │  Relevant? → continue.                  │
  └─────────────────────────────────────────┘
         │
         ▼
  ChatService ensures ADK session exists
  (DatabaseSessionService — persists to PostgreSQL)
         │
         ▼
  ADK Runner dispatches to portfolio_advisor (Gemini 2.5 Flash)
         │
         ├── User wants market data?
         │     └── Calls AgentTool: market_intelligence (Gemini 2.5 Flash-Lite)
         │           Uses Google Search + url_context to fetch live macro data
         │
         ├── User wants ESG research?
         │     └── Calls AgentTool: esg_research (Gemini 2.5 Flash-Lite)
         │           Uses Google Search + url_context for Bloomberg/LESG data
         │
         └── User wants a portfolio?
               └── Calls tool: generate_portfolio(model, amount, max_assets)
                     │
                     ▼
               InferenceService.run(model, amount)
                 ├── Find latest completed training job for model
                 ├── Load frozen normalizer from training_normalizer_params
                 ├── Build 10N state vector from market_data + esg_scores
                 ├── Compute per-asset Sharpe/μESG/ΔESG from last 252 rows
                 └── For each topology (cooperative, competitive, mixed):
                       Load checkpoint → run 3 actors → softmax → trim to max_assets
         │
         ▼
  portfolio_advisor synthesises all sub-agent output + MASAC panels
  into a single institutional-quality response
         │
         ▼
  /chat   → ChatService.chat() returns { session_id, response, portfolio_result }
  /stream → SSE events: status → text_chunk × N → done
         │
         ▼
  Route persists user + assistant messages to chat_messages table
  Returns ChatResponse with panels (or null for conversational queries)
```

---

## Guardrails (Two Layers)

### Layer 1 — Input Rail

A fast **Gemini Flash-Lite** call runs before the advisor on every message. It classifies intent and short-circuits if the message is not relevant to the system's purpose:

| Category | Response |
|---|---|
| `relevant` | Passed through to the advisor |
| `off_topic` | "I'm a MASAC portfolio advisor… I'm not able to help with that topic." |
| `abusive` | "I'm not able to respond to that kind of message." |
| `system_probe` | "I'm not able to share information about my internal configuration." |
| `jailbreak` | "I'm a MASAC portfolio advisor… How can I assist you today?" |

The guard **fails open** — if the classification call itself fails (network error, quota), the message is treated as `relevant` so legitimate users are never blocked by infrastructure issues.

### Layer 2 — Instruction Guardrails

Explicit `GUARDRAILS` rules in the advisor's system prompt cover gray-area cases the input rail may not catch: subtle allocation-bypass attempts ("just tell me what % to put in Apple"), nuanced system probing framed as a legitimate question, and jailbreak prompts mixed with real financial questions.

---

## Streaming (SSE) — `/chat/stream`

The streaming endpoint returns `Content-Type: text/event-stream`. Each line is `data: {json}\n\n`.

### Event types

| `type` | When emitted | Fields |
|---|---|---|
| `status` | Pipeline stage change | `status` (`thinking`\|`calling_tool`), `agent`, `tool` (on `calling_tool`), `label`, `content` (`""`) |
| `text_chunk` | Partial text from any agent | `agent`, `label`, `content` (chunk text) |
| `done` | After all agents finish | `session_id`, `response` (full text), `portfolio_result` |
| `error` | Unhandled exception | `message` |

### Example stream for "Allocate $10M with Portfolio C"

```
data: {"type":"status","status":"thinking","agent":"portfolio_advisor","label":"Thinking...","content":""}
data: {"type":"status","status":"calling_tool","tool":"generate_portfolio","agent":"portfolio_advisor","label":"Calculating portfolio...","content":""}
data: {"type":"text_chunk","agent":"portfolio_advisor","label":"Thinking...","content":"Here are your Portfolio C results"}
data: {"type":"text_chunk","agent":"portfolio_advisor","label":"Thinking...","content":" across all three topologies:"}
data: {"type":"done","session_id":"...","response":"<full text>","portfolio_result":{...}}
```

### Example stream for "Given current rates, allocate $5M"

```
data: {"type":"status","status":"thinking","agent":"portfolio_advisor","label":"Thinking...","content":""}
data: {"type":"status","status":"calling_tool","tool":"market_intelligence","agent":"portfolio_advisor","label":"Fetching market intelligence...","content":""}
data: {"type":"text_chunk","agent":"market_intelligence","label":"Fetching market intelligence...","content":"Current Fed funds rate stands at..."}
data: {"type":"status","status":"calling_tool","tool":"generate_portfolio","agent":"portfolio_advisor","label":"Calculating portfolio...","content":""}
data: {"type":"text_chunk","agent":"portfolio_advisor","label":"Thinking...","content":"Based on the current macro environment..."}
data: {"type":"done","session_id":"...","response":"<full text>","portfolio_result":{...}}
```

**Frontend rendering:**
- `text_chunk` with `agent == "portfolio_advisor"` → stream into the main response area
- `text_chunk` with `agent == "market_intelligence"` or `"esg_research"` → collapsible "Research in progress" panel
- `status` events → update the status indicator with `label`
- `done` → replace streamed text with the authoritative `response`, attach portfolio panels

Consume with `fetch` + `ReadableStream` (not native `EventSource`) so the `Authorization: Bearer` header can be sent.

---

## Agent Architecture

### Portfolio Advisor (orchestrator)

Built per-request so tool closures can capture per-request `ChatService` state. Sub-agents are module-level singletons.

```python
# app/agents/portfolio_advisor.py
def build_portfolio_advisor(service: ChatService, username: str) -> LlmAgent:
    return LlmAgent(
        name="portfolio_advisor",
        model=cfg.adk_model,            # gemini-2.5-flash
        instruction=PORTFOLIO_ADVISOR_INSTRUCTION + GUARDRAILS,
        tools=[
            make_generate_portfolio(service),   # closure — captures service instance
            make_list_available_models(service), # closure — captures service instance
            AgentTool(agent=market_agent),       # delegates to market_intelligence sub-agent
            AgentTool(agent=esg_research_agent), # delegates to esg_research sub-agent
        ],
    )
```

### Market Intelligence (sub-agent singleton)

```python
# app/agents/market_intelligence.py
market_agent = LlmAgent(
    name="market_intelligence",
    model=cfg.adk_model_market,        # gemini-2.5-flash-lite
    instruction=MARKET_INTELLIGENCE_INSTRUCTION,
    tools=[google_search, url_context],
)
```

Scope: macro environment, interest rates, inflation, equity sector performance, earnings, geopolitical risks. ESG topics are explicitly out of scope — redirected to the ESG agent.

### ESG Research (sub-agent singleton)

```python
# app/agents/esg_research.py
esg_research_agent = LlmAgent(
    name="esg_research",
    model=cfg.adk_model_market,        # gemini-2.5-flash-lite
    instruction=ESG_RESEARCH_ANALYST_INSTRUCTION,
    tools=[google_search, url_context],
)
```

Scope: Bloomberg ESG vs LESG ESG ratings, score divergence (ΔESG), controversies, regulatory changes. Macro/market topics are explicitly out of scope — redirected to the market agent.

---

## Tools

### `generate_portfolio(portfolio_model, investment_amount, max_assets=3)`

Runs the full MASAC inference pipeline. Returns a compact JSON summary for the LLM to narrate. Also writes full trimmed panels into `service._portfolio_result` which the route reads after `chat()` returns.

- `portfolio_model`: `"A"`, `"B"`, or `"C"` — coerced to uppercase; invalid values → `"C"`
- `investment_amount`: total USD (e.g. `10000000.0`)
- `max_assets`: top N assets per topology panel (default 3, max 7)

**Fallback logic**: if the requested model has no completed training job, the tool silently retries with `"C"`. The LLM is not told unless it asks.

### `list_available_models()`

Queries `training_jobs WHERE status = "completed"` and returns job IDs, portfolio models, topologies, and best Sharpe scores. Called **only** when the user explicitly asks about training status — never before `generate_portfolio`.

### `market_intelligence` (AgentTool)

Delegates to the Market Intelligence sub-agent. Triggered when the user asks about macro, rates, sector, earnings, or geopolitical context. The sub-agent runs Google Search autonomously and returns a structured research summary.

### `esg_research` (AgentTool)

Delegates to the ESG Research sub-agent. Triggered when the user asks about Bloomberg/LESG score changes, ESG controversies, ΔESG drivers, or sustainability regulations. Returns a structured ESG intelligence report.

### Model routing rules (enforced by advisor instruction)

| User says | Model used |
|---|---|
| "model A" / "portfolio A" | `"A"` |
| "model B" / "portfolio B" | `"B"` |
| Anything else (including "best", "recommended", "default", unspecified) | `"C"` |

The advisor **never asks** which model to use and never explains routing logic unless asked.

---

## Session Management

### ADK sessions (conversation history)

Session history is maintained via `DatabaseSessionService` backed by the same PostgreSQL database. Sessions persist across server restarts and are scoped to `app_name="madrl_portfolio"` + `user_id` (the authenticated user's DB integer ID as a string).

```python
_session_service = DatabaseSessionService(db_url=cfg.postgres_dsn)
```

### Chat sessions (message history)

The route layer additionally persists messages to `chat_sessions` and `chat_messages` tables so users can retrieve history via the REST API.

| Endpoint | Description |
|---|---|
| `GET /api/v1/chat/sessions` | List all sessions for the authenticated user, ordered by last active |
| `GET /api/v1/chat/sessions/{session_id}` | Full session with message history |
| `PATCH /api/v1/chat/sessions/{session_id}` | Rename a session |
| `DELETE /api/v1/chat/sessions/{session_id}` | Delete session + all messages (also cleans up ADK session) |

Sessions are auto-named from the first message (truncated to 60 chars). DB persistence is failure-safe — persistence errors are logged but never break the chat response.

---

## State Vector Construction (Inference)

At inference time the system reads the **most recent available date** from `market_data` for all N ISINs from the job and constructs a single 10N state vector using the **frozen normalizer** fitted during Stage 3 of the training pipeline.

### Feature order (matches training exactly)

| Position   | Feature                 | DB Column                | Normalization                           |
| ---------- | ----------------------- | ------------------------ | --------------------------------------- |
| 0 … N-1    | Open                    | `market_data.open`       | Time-series min-max (frozen)            |
| N … 2N-1   | High                    | `market_data.high`       | Time-series min-max (frozen)            |
| 2N … 3N-1  | Low                     | `market_data.low`        | Time-series min-max (frozen)            |
| 3N … 4N-1  | Close                   | `market_data.close`      | Time-series min-max (frozen)            |
| 4N … 5N-1  | Volume                  | `market_data.volume`     | Time-series min-max (frozen)            |
| 5N … 6N-1  | RSI                     | `market_data.rsi`        | Time-series min-max (frozen)            |
| 6N … 7N-1  | MACD histogram          | `market_data.macd_hist`  | Time-series min-max (frozen)            |
| 7N … 8N-1  | Individual return R_i,t | `market_data.return_pct` | Time-series min-max (frozen)            |
| 8N … 9N-1  | ΔESG per stock          | `esg_scores.delta_esg`   | Cross-sectional (Stage 2, pre-computed) |
| 9N … 10N-1 | μESG per stock          | `esg_scores.mu_esg`      | Cross-sectional (Stage 2, pre-computed) |

The frozen normalizer clips values to [0, 1] — values slightly outside the training range are clamped, not rejected.

---

## Portfolio Weight Derivation

Each MASAC actor takes the 10N state vector and outputs N unnormalized allocation scores z ∈ ℝᴺ.

```
z_bloomberg = actor_bloomberg.deterministic_action(state)   # (N,) — mean output, no sampling
z_lesg      = actor_lesg.deterministic_action(state)        # (N,)
z_financial = actor_financial.deterministic_action(state)   # (N,)

z_joint = (z_bloomberg + z_lesg + z_financial) / 3          # equal-weight average
weights = softmax(z_joint)                                   # sums to 1.0, all ≥ 0
```

Deterministic mode uses the actor mean directly — no sampling noise. Identical to the inference path used in training's `_eval_validation`.

### Why weights differ across topologies

All three topologies receive the **same** 10N state vector. Differences emerge entirely from how each topology trained its agents under a different reward structure:

| Topology    | β in reward                 | Effect on high-ΔESG stocks                                             |
| ----------- | --------------------------- | ---------------------------------------------------------------------- |
| Cooperative | β > 0 (full shared penalty) | All agents penalized — suppresses high-ΔESG allocation                 |
| Competitive | β = 0 (no penalty)          | Bloomberg + Financial agents may independently favour high-ΔESG stocks |
| Mixed       | β × 0.5 (partial penalty)   | Intermediate — moderates the competitive risk appetite                 |

---

## Panel Output Per Asset

Each topology panel is a list sorted by weight descending:

```json
{
  "isin": "US0378331005",
  "company": "Apple Inc.",
  "sector": "Technology",
  "return_ann": 0.22,
  "risk": 0.12,
  "sharpe": 1.83,
  "mu_esg": 0.93,
  "delta_esg": 0.14,
  "weight": 0.4,
  "allocation": 4000000.0
}
```

| Field        | Meaning                                                                      |
| ------------ | ---------------------------------------------------------------------------- |
| `return_ann` | Annualised simple return = mean(daily return_pct) × 252                      |
| `risk`       | Annualised std = std(daily return_pct) × √252                                |
| `sharpe`     | `return_ann / (risk + ε)`, risk-free rate = 0                                |
| `mu_esg`     | Per-stock ESG consensus = (esg_b_norm + esg_l_norm) / 2 at most recent date  |
| `delta_esg`  | Per-stock ESG disagreement = \|esg_b_norm − esg_l_norm\| at most recent date |
| `weight`     | Portfolio weight ∈ [0, 1]; all N weights sum to 1.0                          |
| `allocation` | `weight × investment_amount` in USD                                          |

---

## API Reference

### `POST /api/v1/chat`

**Request:**
```json
{ "message": "I have $10,000,000 to allocate. Use Portfolio C.", "session_id": "optional-uuid" }
```

**Response (202):**
```json
{
  "session_id": "3f7a2b1c-...",
  "response": "Here are your Portfolio C recommendations across all three topologies...",
  "job_id": 16,
  "portfolio_model": "C",
  "panels": {
    "C_cooperative": [ { "isin": "...", "weight": 0.40, "allocation": 4000000.0 } ],
    "C_competitive": [ ... ],
    "C_mixed":       [ ... ]
  }
}
```

`panels` is `null` for purely conversational queries. Error: `503` if Gemini API is unavailable.

---

### `POST /api/v1/chat/stream`

**Request:** same as `/chat`.

**Response:** `text/event-stream` — see [Streaming section](#streaming-sse----chatstream) above.

---

### Session endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/chat/sessions` | List all sessions, ordered by last active |
| `GET` | `/api/v1/chat/sessions/{session_id}` | Session metadata + full message history |
| `PATCH` | `/api/v1/chat/sessions/{session_id}` | Rename a session |
| `DELETE` | `/api/v1/chat/sessions/{session_id}` | Delete session + all messages |

All session endpoints require authentication and enforce ownership (403 if session belongs to another user).

---

## Prerequisites

Before the chat API can generate portfolios:

1. At least one `.xlsx` file uploaded via `POST /api/v1/training/start`
2. The Celery training job must have reached `status = "completed"`
3. At least one row in `model_checkpoints` per topology (training ran ≥ 10,000 steps past warmup)
4. `training_normalizer_params` populated (8 × N rows for the job) — required for state vector construction

If no completed job exists for the requested model, the agent falls back to model C. If C also has no completed job, it returns an error message asking the user to train first.

---

## Configuration

| Setting | .env key | Default | Purpose |
|---|---|---|---|
| `adk_model` | `ADK_MODEL` | `gemini-2.5-flash` | Portfolio Advisor — orchestration & reasoning |
| `adk_model_market` | `ADK_MODEL_MARKET` | `gemini-2.5-flash-lite` | Market Intelligence + ESG Research sub-agents |
| `adk_model_guard` | `ADK_MODEL_GUARD` | `gemini-2.5-flash-lite` | Input rail classifier — off-topic/abuse/jailbreak gate |
| `google_api_key` | `GOOGLE_API_KEY` | _(required)_ | Google API key — injected into `os.environ` at startup |
| `model_store_path` | `MODEL_STORE_PATH` | `./model_store` | Filesystem root for `.pt` checkpoint files |

> `OTEL_SDK_DISABLED=true` is set in Python code at module load — do **not** add it to `.env` as it causes a Pydantic validation error (`Extra inputs are not permitted`).

---

## Implementation Notes

### Google API key injection

Pydantic-settings reads `.env` into the `Settings` object but does **not** populate `os.environ`. Google ADK reads `GOOGLE_API_KEY` directly from `os.environ`. The service sets it explicitly at module load:

```python
if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)
```

### Async generator — no break

```python
async for event in self._runner.run_async(...):
    if event.is_final_response() and event.content and event.content.parts:
        final_text = "".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text)
# No break — generator exhausts naturally
```

`break` inside an async generator causes `GeneratorExit` to propagate through OpenTelemetry context managers, raising `ValueError: Token was created in a different Context` on Python 3.12+. The generator is allowed to exhaust naturally. OTel is also disabled at module level: `os.environ.setdefault("OTEL_SDK_DISABLED", "true")`.

### RunConfig

```python
run_config = RunConfig(max_llm_calls=25)
```

Hard cap of 25 LLM calls per request — accounts for multi-agent pipelines where market_intelligence and esg_research each consume several calls internally.

### MASAC allocation constraint

**All portfolio weights come exclusively from the MASAC reinforcement learning engine.** The LLM agents are narrators and researchers — they are explicitly forbidden from predicting, suggesting, or adjusting any allocation weight. The advisor instruction contains a hard `SYSTEM IDENTITY` section reinforcing this boundary, and the guardrails catch allocation-bypass attempts.
