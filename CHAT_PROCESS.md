# Chat API Process

## Overview

The chat API provides a natural language interface to the trained MASAC portfolio system. Users send queries in plain English; the system returns three side-by-side portfolio panels (Cooperative, Competitive, Mixed) with per-asset allocations, ESG metrics, and Sharpe statistics.

The chat layer is powered by **Google ADK 1.33** with **Gemini** as the underlying LLM. The ADK agent parses user intent and calls structured tools that run the actual inference pipeline. Session history is maintained across multiple requests via an in-memory singleton.

---

## End-to-End Flow

```
POST /api/v1/chat
  { "message": "I have $10M. Use Portfolio C.", "session_id": "optional-uuid" }
         │
         ▼
  ChatService.chat(session_id, message)
    ├── Checks InMemorySessionService for existing session
    ├── Creates session if new (app_name="madrl_portfolio", user_id="madrl_user")
    └── Runs ADK Runner.run_async(RunConfig(max_llm_calls=10))
         │
         ▼
  ADK LLM (Gemini) parses intent
    ├── Identifies portfolio_model="C", investment_amount=10_000_000
    └── Calls tool: generate_portfolio("C", 10_000_000, max_assets=3)
         │
         ▼
  generate_portfolio tool → InferenceService.run("C", 10_000_000)
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 1: Find latest completed training job for portfolio_model  │
  │    SELECT * FROM training_jobs                                   │
  │    WHERE portfolio_model="C" AND status="completed"             │
  │    ORDER BY id DESC LIMIT 1                                     │
  │    If model A or B requested but not trained → silently fall     │
  │    back to C and continue                                        │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 2: Load frozen normalizer from DB                          │
  │    SELECT isin, feature_name, min_val, max_val                   │
  │    FROM training_normalizer_params WHERE job_id = ?              │
  │    Reconstructs 8 × N TimeSeriesNormalizer scalers (N dynamic)   │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 3: Build current 10N state vector                          │
  │    a. Query most recent date in market_data for all N ISINs      │
  │    b. Get OHLCV, RSI, macd_hist, return_pct from market_data     │
  │    c. Get delta_esg, mu_esg from esg_scores (Stage-2 computed)   │
  │    d. Apply frozen time-series normalizer to OHLCV+RSI+MACD+ret  │
  │    e. Append delta_esg, mu_esg (already cross-sectional normed)  │
  │    f. Concatenate into 10N vector:                               │
  │       [open(N)|high(N)|low(N)|close(N)|vol(N)|                   │
  │        rsi(N)|macd(N)|return(N)|delta_esg(N)|mu_esg(N)]         │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 4: Compute asset metrics from last 365 days                │
  │    For each ISIN (last 252 usable return rows):                   │
  │      ann_return = mean(return_pct) × 252                         │
  │      risk (σ)  = std(return_pct) × √252                         │
  │      sharpe    = ann_return / (risk + ε)                         │
  │      mu_esg    = most recent (esg_b_norm + esg_l_norm) / 2       │
  │      delta_esg = most recent |esg_b_norm − esg_l_norm|           │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 5: For each topology (cooperative, competitive, mixed):    │
  │    a. Load best checkpoint from model_checkpoints table          │
  │       (highest Sharpe for this job_id + topology)               │
  │       Falls back to default path if no checkpoint rows yet       │
  │    b. Instantiate MASAC(n_assets=N) and load .pt weights         │
  │    c. Run deterministic inference:                               │
  │         z_B = actor_bloomberg.deterministic_action(state) → (N,) │
  │         z_L = actor_lesg.deterministic_action(state)      → (N,) │
  │         z_F = actor_financial.deterministic_action(state) → (N,) │
  │         z_joint = (z_B + z_L + z_F) / 3                         │
  │         weights = softmax(z_joint) → (N,) summing to 1.0        │
  │    d. Build panel: sort by weight desc, trim to top max_assets   │
  │       allocation = weight × investment_amount                    │
  └──────┬──────────────────────────────────────────────────────────┘
         │
         ▼
  generate_portfolio stores full panels in service._portfolio_result
  Returns JSON summary (top max_assets per topology) to ADK LLM
         │
         ▼
  ADK LLM composes natural language response — 3 panels side by side
         │
         ▼
  ChatService.chat() returns:
    { session_id, response (text), portfolio_result }
         │
         ▼
  Route assembles ChatResponse:
    {
      "session_id": "...",
      "response": "Here are your Portfolio C recommendations...",
      "job_id": 16,
      "portfolio_model": "C",
      "panels": {
        "C_cooperative": [...],
        "C_competitive": [...],
        "C_mixed":       [...]
      }
    }
```

---

## Panel Key Format

Panel keys in the response follow the pattern `{MODEL}_{topology}`:

| Key | Meaning |
|---|---|
| `C_cooperative` | Portfolio C — Cooperative topology |
| `C_competitive` | Portfolio C — Competitive topology |
| `C_mixed` | Portfolio C — Mixed topology |

If a user somehow triggers multiple model calls in one session the key would be `A_cooperative`, etc. The `portfolio_model` field at the top level is set to `"ALL"` in that case.

---

## State Vector Construction (Inference)

At inference time the system reads the **most recent available date** from `market_data` for all N ISINs from the job and constructs a single 10N state vector using the **frozen normalizer** fitted during Stage 3 of the training pipeline.

### Feature order (matches training exactly)

| Position | Feature | DB Column | Normalization |
|---|---|---|---|
| 0 … N-1 | Open | `market_data.open` | Time-series min-max (frozen) |
| N … 2N-1 | High | `market_data.high` | Time-series min-max (frozen) |
| 2N … 3N-1 | Low | `market_data.low` | Time-series min-max (frozen) |
| 3N … 4N-1 | Close | `market_data.close` | Time-series min-max (frozen) |
| 4N … 5N-1 | Volume | `market_data.volume` | Time-series min-max (frozen) |
| 5N … 6N-1 | RSI | `market_data.rsi` | Time-series min-max (frozen) |
| 6N … 7N-1 | MACD histogram | `market_data.macd_hist` | Time-series min-max (frozen) |
| 7N … 8N-1 | Individual return R_i,t | `market_data.return_pct` | Time-series min-max (frozen) |
| 8N … 9N-1 | ΔESG per stock | `esg_scores.delta_esg` | Cross-sectional (Stage 2, pre-computed) |
| 9N … 10N-1 | μESG per stock | `esg_scores.mu_esg` | Cross-sectional (Stage 2, pre-computed) |

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

| Topology | β in reward | Effect on high-ΔESG stocks |
|---|---|---|
| Cooperative | β > 0 (full shared penalty) | All agents penalized — suppresses high-ΔESG allocation |
| Competitive | β = 0 (no penalty) | Bloomberg + Financial agents may independently favour high-ΔESG stocks |
| Mixed | β × 0.5 (partial penalty) | Intermediate — moderates the competitive risk appetite |

---

## Asset Count: max_assets

The `generate_portfolio` tool accepts a `max_assets: int = 3` parameter. The ADK agent sets this dynamically:

- **Default**: 3 (concentrated portfolio)
- **User-specified**: if the user says "top 5" or "7 assets", that number is used
- **Agent discretion**: the agent may increase beyond 3 (up to 5) when multiple assets show strong Sharpe and meaningful weight — but never exceeds 7 without an explicit user request

Internally, the full N-asset panel is computed and sorted by weight descending. Only the top `max_assets` rows are returned to the LLM for narration and stored in `panels`. The full panel is never exposed in the API response.

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
  "weight": 0.40,
  "allocation": 4000000.0
}
```

| Field | Meaning |
|---|---|
| `return_ann` | Annualised simple return = mean(daily return_pct) × 252 |
| `risk` | Annualised std = std(daily return_pct) × √252 |
| `sharpe` | `return_ann / (risk + ε)`, risk-free rate = 0 |
| `mu_esg` | Per-stock ESG consensus = (esg_b_norm + esg_l_norm) / 2 at most recent date |
| `delta_esg` | Per-stock ESG disagreement = \|esg_b_norm − esg_l_norm\| at most recent date |
| `weight` | Portfolio weight ∈ [0, 1]; all N weights sum to 1.0 |
| `allocation` | `weight × investment_amount` in USD |

---

## Google ADK Agent

### Identity & session

```python
_APP_NAME  = "madrl_portfolio"
_USER_ID   = "madrl_user"          # stable app-level identifier — NOT the session UUID
_session_service = InMemorySessionService()  # process-level singleton
```

`session_id` is the per-conversation UUID (client-supplied or server-generated). `user_id` is a fixed app constant. Both must match exactly between `create_session` and `run_async`.

### Tools

#### `generate_portfolio(portfolio_model, investment_amount, max_assets=3)`

Runs the full inference pipeline. Returns a compact JSON summary (top holdings per topology) for the LLM to narrate. Also writes the full trimmed panels into `service._portfolio_result` which the route reads after `chat()` returns.

- `portfolio_model`: `"A"`, `"B"`, or `"C"` — coerced to uppercase; invalid values → `"C"`
- `investment_amount`: total USD (e.g. `10000000.0`)
- `max_assets`: how many top assets to return per topology panel (default 3)

**Fallback logic**: if the requested model has no completed training job, the tool silently retries with `"C"` and logs a warning. The LLM is never told about the fallback unless it asks.

#### `list_available_models()`

Queries `training_jobs WHERE status = "completed"` and returns job IDs, portfolio models, topologies, and best Sharpe scores. Called **only** when the user explicitly asks about training status or model availability — never before `generate_portfolio`.

### Model routing rules (enforced by agent instruction)

| User says | Model used |
|---|---|
| "model A" / "portfolio A" | `"A"` |
| "model B" / "portfolio B" | `"B"` |
| Anything else (including "best", "recommended", "full", "default", unspecified) | `"C"` |

The agent **never asks** which model to use. It never explains the fallback logic unless directly asked.

### RunConfig

```python
run_config = RunConfig(max_llm_calls=10)
```

Hard cap of 10 LLM calls per request. Prevents the infinite tool-call loop that occurs when `gemini-2.5-flash-lite` returns mixed text+function_call parts — the ADK warning `"there are non-text parts in the response: ['function_call']"` is a known model quirk, capped here.

### Async generator — no break

```python
async for event in self._runner.run_async(...):
    if event.is_final_response() and event.content and event.content.parts:
        final_text = "".join(p.text for p in event.content.parts if hasattr(p, "text") and p.text)
# No break — generator exhausts naturally
```

`break` inside an async generator causes `GeneratorExit` to propagate through OpenTelemetry context managers, raising `ValueError: Token was created in a different Context` on Python 3.12+. The generator is allowed to exhaust naturally to avoid this. OTel is also disabled at module level: `os.environ.setdefault("OTEL_SDK_DISABLED", "true")`.

### Google API key injection

Pydantic-settings reads `.env` into the `Settings` object but does **not** populate `os.environ`. Google ADK reads `GOOGLE_API_KEY` directly from `os.environ`. The service sets it explicitly at module load:

```python
if cfg.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)
```

---

## API Reference

### `POST /api/v1/chat`

**Request:**
```json
{
  "message": "I have $10,000,000 to allocate. Use Portfolio C.",
  "session_id": "optional-existing-uuid"
}
```

- `session_id`: omit on first message — the server generates and returns one. Include on all follow-up messages to continue the same conversation.

**Response:**
```json
{
  "session_id": "3f7a2b1c-...",
  "response": "Here are your Portfolio C recommendations across all three topologies...",
  "job_id": 16,
  "portfolio_model": "C",
  "panels": {
    "C_cooperative": [ { "isin": "...", "weight": 0.40, "allocation": 4000000.0, ... } ],
    "C_competitive": [ ... ],
    "C_mixed":       [ ... ]
  }
}
```

- `panels` is `null` for purely conversational queries (no portfolio generated)
- `response` always contains the LLM's natural language reply
- Status code: `202 Accepted`
- Error: `503` if the Gemini API or ADK is unavailable

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
| `adk_model` | `ADK_MODEL` | `gemini-2.5-flash-lite` | Gemini model used by the ADK agent |
| `google_api_key` | `GOOGLE_API_KEY` | *(required)* | Google API key — set in `.env`, injected into `os.environ` at startup |
| `model_store_path` | `MODEL_STORE_PATH` | `./model_store` | Filesystem root for `.pt` checkpoint files |

> `OTEL_SDK_DISABLED=true` is set in Python code at module load — do **not** add it to `.env` as it causes a Pydantic validation error (`Extra inputs are not permitted`).
