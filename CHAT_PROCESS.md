# Chat API Process

## Overview

The chat API provides a natural language interface to the trained MASAC portfolio system. Users send queries in plain English; the system returns three side-by-side portfolio panels (Cooperative, Competitive, Mixed) with per-asset allocations, ESG metrics, and performance statistics.

The chat layer is powered by **Google ADK** (Agent Development Kit) with Gemini as the underlying LLM. The ADK agent parses user intent and calls structured tools that run the actual inference pipeline.

---

## End-to-End Flow

```
POST /api/v1/chat
  { "message": "I have $10M. Use Portfolio C.", "session_id": "optional-uuid" }
         │
         ▼
  ChatService.chat(session_id, message)
         │
         ▼
  Google ADK Runner
    ├── LLM parses intent → identifies portfolio_model="C", investment_amount=10_000_000
    └── Calls tool: generate_portfolio("C", 10_000_000)
         │
         ▼
  InferenceService.run(portfolio_model="C", investment_amount=10_000_000)
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 1: Find latest completed training job for portfolio_model  │
  │    SELECT * FROM training_jobs                                   │
  │    WHERE portfolio_model="C" AND status="completed"             │
  │    ORDER BY id DESC LIMIT 1                                     │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 2: Load frozen normalizer from DB                          │
  │    SELECT isin, feature_name, min_val, max_val                   │
  │    FROM training_normalizer_params WHERE job_id = ?              │
  │    Reconstructs 8 × N TimeSeriesNormalizer scalers               │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 3: Build current 10N state vector                          │
  │    a. Query most recent date in market_data for all N ISINs      │
  │    b. Get OHLCV, RSI, MACD histogram, return_pct for each ISIN   │
  │    c. Get delta_esg, mu_esg from esg_scores for same date        │
  │    d. Apply frozen normalizer to OHLCV + RSI + MACD + return     │
  │    e. Concatenate into 10N vector:                               │
  │       [open(N)|high(N)|low(N)|close(N)|vol(N)|                   │
  │        rsi(N)|macd(N)|return(N)|delta_esg(N)|mu_esg(N)]         │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 4: Compute asset metrics from recent 252 days              │
  │    For each ISIN:                                                │
  │      returns[] = last 252 daily return_pct values from DB        │
  │      ann_return = mean(returns) × 252                            │
  │      risk (σ)  = std(returns) × √252                            │
  │      sharpe    = ann_return / (risk + ε)                         │
  │      mu_esg    = most recent (esg_b_norm + esg_l_norm) / 2       │
  │      delta_esg = most recent |esg_b_norm − esg_l_norm|           │
  └──────┬──────────────────────────────────────────────────────────┘
         │
  ┌──────┴──────────────────────────────────────────────────────────┐
  │  Step 5: For each topology (cooperative, competitive, mixed):    │
  │    a. Load best checkpoint from model_checkpoints table          │
  │       (highest sharpe for this job_id + topology)               │
  │    b. Instantiate MASAC(n_assets=N) and load .pt weights         │
  │    c. Run deterministic inference:                               │
  │         z_B = actor_bloomberg(state)   → (N,)                   │
  │         z_L = actor_lesg(state)        → (N,)                   │
  │         z_F = actor_financial(state)   → (N,)                   │
  │         z_joint = (z_B + z_L + z_F) / 3                         │
  │         weights = softmax(z_joint)     → (N,) summing to 1.0    │
  │    d. Build panel: weight × investment_amount = allocation       │
  └──────┬──────────────────────────────────────────────────────────┘
         │
         ▼
  generate_portfolio tool returns JSON summary to ADK agent
         │
         ▼
  ADK LLM composes natural language response explaining the 3 panels
         │
         ▼
  ChatService returns:
    { session_id, response (text), portfolio_result (panels) }
         │
         ▼
  Route returns ChatResponse:
    {
      "session_id": "...",
      "response": "Here are your portfolio recommendations...",
      "job_id": 16,
      "portfolio_model": "C",
      "panels": {
        "cooperative": [...],
        "competitive": [...],
        "mixed":       [...]
      }
    }
```

---

## State Vector Construction (Inference)

At inference time the system reads the **most recent available date** from `market_data` for all N ISINs from the job and constructs a single 10N state vector using the **frozen normalizer** trained during Stage 3 of the training pipeline.

### Feature order (matches training exactly)

| Position | Feature | Source | Normalization |
|---|---|---|---|
| 0 … N-1 | Open | `market_data.open` | Time-series min-max (frozen) |
| N … 2N-1 | High | `market_data.high` | Time-series min-max (frozen) |
| 2N … 3N-1 | Low | `market_data.low` | Time-series min-max (frozen) |
| 3N … 4N-1 | Close | `market_data.close` | Time-series min-max (frozen) |
| 4N … 5N-1 | Volume | `market_data.volume` | Time-series min-max (frozen) |
| 5N … 6N-1 | RSI | `market_data.rsi` | Time-series min-max (frozen) |
| 6N … 7N-1 | MACD histogram | `market_data.macd_hist` | Time-series min-max (frozen) |
| 7N … 8N-1 | Individual return R_i,t | `market_data.return_pct` | Time-series min-max (frozen) |
| 8N … 9N-1 | ΔESG per stock | `esg_scores.delta_esg` | Cross-sectional (pre-computed in Stage 2) |
| 9N … 10N-1 | μESG per stock | `esg_scores.mu_esg` | Cross-sectional (pre-computed in Stage 2) |

The frozen normalizer clips values to [0, 1] — test-period values slightly outside the training range are clamped, not rejected.

---

## Portfolio Weight Derivation

Each MASAC actor takes the 10N state vector and outputs N unnormalized scores z ∈ ℝᴺ.

```
z_bloomberg = actor_bloomberg.deterministic_action(state)   # (N,)
z_lesg      = actor_lesg.deterministic_action(state)        # (N,)
z_financial = actor_financial.deterministic_action(state)   # (N,)

z_joint = (z_bloomberg + z_lesg + z_financial) / 3          # equal-weight average
weights = softmax(z_joint)                                   # sums to 1.0, all ≥ 0
```

This is identical to the training environment — deterministic mode uses the actor's mean output (no sampling noise).

### Why weights differ across topologies

All three topologies receive the **same** state vector. The differences come entirely from how each topology's agents were trained:

| Topology | β in reward | Effect on Energy-type stocks (high ΔESG) |
|---|---|---|
| Cooperative | β > 0 (full penalty shared) | All agents penalized → underweight high-ΔESG |
| Competitive | β = 0 (no penalty) | Bloomberg + Financial agents may favour high-ΔESG stocks independently |
| Mixed | β × 0.5 (partial penalty) | Intermediate — moderates the competitive risk appetite |

---

## Panel Output Per Asset

Each topology panel is a list of N assets, sorted by weight descending:

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
| `return_ann` | Annualised simple return = mean(daily returns) × 252 |
| `risk` | Annualised standard deviation = std(daily returns) × √252 |
| `sharpe` | `return_ann / risk` (risk-free rate = 0, consistent with training) |
| `mu_esg` | Per-stock ESG consensus = (esg_b_norm + esg_l_norm) / 2 at most recent date |
| `delta_esg` | Per-stock ESG disagreement = \|esg_b_norm − esg_l_norm\| at most recent date |
| `weight` | Portfolio weight ∈ [0, 1], all N weights sum to 1.0 |
| `allocation` | `weight × investment_amount` in USD |

---

## Google ADK Agent

### Tools available to the agent

#### `generate_portfolio(portfolio_model, investment_amount)`
Runs the full inference pipeline described above. Returns a JSON summary of the top holdings per topology for the LLM to narrate.

- `portfolio_model`: `"A"`, `"B"`, or `"C"`
- `investment_amount`: total USD amount (e.g. `10000000`)

#### `list_available_models()`
Queries `training_jobs` for all completed runs. Returns job IDs, portfolio models, topologies, and best Sharpe scores. Useful when the user asks "what models have been trained?"

### Session management

Conversation history is maintained via `InMemorySessionService` — a process-level singleton. Each `session_id` maps to a full conversation history. The ADK runner replays all prior turns on each new message, so the agent remembers context across multiple exchanges.

**Limitation**: history is lost on server restart. Production deployments should replace `InMemorySessionService` with a Redis- or DB-backed session store.

### Agent instruction summary

The agent knows:
- Portfolio A/B/C definitions and the reward function differences
- Cooperative / Competitive / Mixed topology mechanics
- How to interpret ΔESG (disagreement) and μESG (consensus) in plain language
- That results must be presented as three panels side by side

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

- `message`: any natural language query
- `session_id`: omit on first message — the server generates one. Include on follow-up messages to continue the conversation.

**Response:**
```json
{
  "session_id": "3f7a2b1c-...",
  "response": "Here are your three portfolio recommendations for Portfolio C...",
  "job_id": 16,
  "portfolio_model": "C",
  "panels": {
    "cooperative": [ { "isin": "...", "weight": 0.40, "allocation": 4000000, ... }, ... ],
    "competitive": [ ... ],
    "mixed":       [ ... ]
  }
}
```

- `panels` is `null` if the query was conversational (not a portfolio generation request)
- `response` always contains the LLM's natural language reply

### `GET /health` (existing)
No change — still returns `{"status": "ok"}`.

---

## Prerequisites

Before the chat API can generate portfolios:
1. At least one XLSX file must have been uploaded via `POST /api/v1/training/start`
2. The corresponding Celery training job must have reached `status = "completed"`
3. At least one row must exist in `model_checkpoints` for each topology (i.e. training ran for ≥ 10,000 steps past warmup)

If no completed job exists for the requested portfolio model, the agent returns a clear error message asking the user to train a model first.

---

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `adk_model` | `gemini-2.0-flash` | LLM used by the ADK agent |
| `google_api_key` | *(required)* | Google API key for Gemini access |
| `model_store_path` | `./model_store` | Filesystem root where `.pt` checkpoint files are stored |

Set `GOOGLE_API_KEY=your-key` in `.env` before starting the server.
