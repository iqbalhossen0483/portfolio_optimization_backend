# MADRL Portfolio System — Architecture & Design

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         MADRL Portfolio Platform                                         │
│                                                                                          │
│  ┌──────────┐  REST/SSE/WS ┌──────────────────────────────────────────────┐              │
│  │  Client  │◄────────────►│               FastAPI Gateway                │              │
│  │(UI / API)│              │  /auth  /training  /data  /chat  /ws         │              │
│  └──────────┘              └────────────────┬─────────────────────────────┘              │
│                                             │  JWT Bearer auth on all routes             │
│                            ┌────────────────┼────────────────────┐                       │
│                            │                │                    │                       │
│                  ┌─────────▼──────┐ ┌───────▼──────────────────┐ ┌──────────▼────────┐   │
│                  │ TrainingService│ │      ChatService         │ │ InferenceService  │   │
│                  │ (4-stage XLSX  │ │  Input Rail (Flash-Lite) │ │ (loads .pt model  │   │
│                  │  pipeline +    │ │  ┌─────────────────────┐ │ │  weights, builds  │   │
│                  │  Celery queue) │ │  │ portfolio_advisor   │ │ │  10N state vec,   │   │
│                  └────────┬───────┘ │  │   (Flash)           │ │ │  runs actors)     │   │
│                           │         │  │  ┌───────┐ ┌──────┐ │ │ │                   │   │
│                           │         │  │  │market │ │ esg  │ │ │ │                   │   │
│                           │         │  │  │intel. │ │resrch│ │ │ └───────────────────┘   │
│                           │         │  │  │(Lite) │ │(Lite)│ │ │                         │
│                           │         │  │  └───────┘ └──────┘ │ │                         │
│                           │         │  └─────────────────────┘ │                         │
│                           │         └──────────────────────────┘                         │
│         ┌─────────────────▼────────────────────────────────────┐                         │
│         │                      RL Engine (MASAC)               │                         │
│         │  3 Actor networks + 6 Critics (twin Q per agent)     │                         │
│         │  Shared ReplayBuffer (1M capacity)                   │                         │
│         │  3 Topologies: cooperative / competitive / mixed     │                         │
│         └──────────────────────────┬───────────────────────────┘                         │
│                                    │                                                     │
│         ┌──────────────────────────▼─────────────────────────┐                           │
│         │                   Storage Layer                    │                           │
│         │  PostgreSQL  ←→  Redis (cache + PubSub + sessions) │                           │
│         │  model_store/{job_id}/{topology}/*.pt              │                           │
│         └────────────────────────────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Authentication

Every API endpoint (except `/auth/register` and `/auth/login`) requires a JWT Bearer token.
The system has two roles: **user** and **admin**.

```
POST /api/v1/auth/register      — public — create account (role = "user" by default)
POST /api/v1/auth/login         — public — returns JWT access token (24h expiry, HS256)
GET  /api/v1/auth/me            — any user  — own profile
PUT  /api/v1/auth/me            — any user  — update email / username / password
GET  /api/v1/auth/users         — admin     — list all users
PUT  /api/v1/auth/users/{id}/role — admin   — promote / demote a user

Role-based access:

| Endpoint                          | user | admin |
|-----------------------------------|------|-------|
| POST /auth/register               | ✓    | ✓     |
| POST /auth/login                  | ✓    | ✓     |
| GET  /auth/me                     | ✓    | ✓     |
| PUT  /auth/me                     | ✓    | ✓     |
| GET  /auth/users                  | ✗    | ✓     |
| PUT  /auth/users/{id}/role        | ✗    | ✓     |
| POST /training/start              | ✗    | ✓     |
| GET  /training/{id}/status        | ✗    | ✓     |
| POST /training/{id}/stop          | ✗    | ✓     |
| GET  /data/assets                 | ✗    | ✓     |
| POST /chat                        | ✓    | ✓     |
| POST /chat/stream                 | ✓    | ✓     |
| GET  /chat/sessions               | ✓    | ✓     |
| GET  /chat/sessions/{id}          | ✓    | ✓     |
| PATCH /chat/sessions/{id}         | ✓    | ✓     |
| DELETE /chat/sessions/{id}        | ✓    | ✓     |
| WS   /ws/training/{id}?token=...  | ✗    | ✓     |
```

**JWT claims:** `{ sub: user_id, email, role, exp }`
**WebSocket auth:** token passed as query param `?token=<jwt>` (HTTP `Authorization` header
is not available over the WS handshake in browsers).

---

## 3. Interaction Topologies

All three game-theoretic topologies are trained sequentially per job and run at inference
time for every chat request, producing three side-by-side portfolio panels.

```
Training (per job): cooperative → competitive → mixed   (sequential)
Inference (chat):   cooperative + competitive + mixed    (all three, same state vector)
```

**Reward function by topology and model:**

| Topology    | β applied         | Portfolio A   | Portfolio B            | Portfolio C                      |
| ----------- | ----------------- | ------------- | ---------------------- | -------------------------------- |
| Cooperative | β (full)          | rₜ + α₁·ESG_B | rₜ + λ·(ESG_B − ESG_L) | rₜ + α₁·ESG_B_norm − β·ΔESGₜ     |
| Competitive | 0 (none)          | rₜ + α₁·ESG_B | rₜ + λ·(ESG_B − ESG_L) | rₜ + α₁·ESG_B_norm               |
| Mixed       | β × 0.5 (partial) | rₜ + α₁·ESG_B | rₜ + λ·(ESG_B − ESG_L) | rₜ + α₁·ESG_B_norm − 0.5·β·ΔESGₜ |

---

## 4. File Structure

```
madrl_portfolio/
├── ARCHITECTURE.md
├── CHAT_PROCESS.md
├── TRAINING_PROCESS.md
├── requirements.txt
├── requirements.md          (system specification)
├── app/
│   ├── main.py              — FastAPI factory; registers all routers; lifespan
│   ├── config.py            — Pydantic Settings (env-driven; JWT, MASAC, DB params)
│   ├── api/
│   │   ├── deps.py          — DI: get_db, get_redis, get_current_user, require_admin,
│   │   │                        get_ws_user, get_training_service, get_chat_service
│   │   └── routes/
│   │       ├── auth.py      — register, login, me (GET/PUT), users (admin)
│   │       ├── training.py  — POST /start (admin), GET /{id}/status, POST /{id}/stop (admin)
│   │       ├── data.py      — GET /assets (any user), GET /health
│   │       ├── chat.py      — POST /chat, POST /chat/stream (SSE), session CRUD endpoints
│   │       └── websocket.py — WS /ws/training/{job_id}?token=<jwt>
│   ├── core/
│   │   ├── database.py      — AsyncEngine, async_sessionmaker, create_tables()
│   │   └── security.py      — hash_password, verify_password, create_access_token,
│   │                           decode_token (HS256 via python-jose + passlib[bcrypt])
│   ├── agents/
│   │   ├── __init__.py           — exports market_agent, build_portfolio_advisor
│   │   ├── instructions.py       — PORTFOLIO_ADVISOR_INSTRUCTION, MARKET_INTELLIGENCE_INSTRUCTION,
│   │   │                            ESG_RESEARCH_ANALYST_INSTRUCTION (+ GUARDRAILS section)
│   │   ├── portfolio_advisor.py  — build_portfolio_advisor() factory (per-request, captures service)
│   │   ├── market_intelligence.py — market_agent singleton (Gemini Flash-Lite + google_search)
│   │   ├── esg_research.py       — esg_research_agent singleton (Gemini Flash-Lite + google_search)
│   │   └── tools/
│   │       ├── __init__.py       — exports make_generate_portfolio, make_list_available_models
│   │       └── portfolio_tools.py — tool closures that capture ChatService per-request state
│   ├── models/
│   │   ├── domain.py        — SQLAlchemy ORM: User, Asset, MarketData, ESGScore,
│   │   │                        TrainingJob, ModelCheckpoint, TrainingNormalizerParams,
│   │   │                        ChatSession, ChatMessage
│   │   └── schemas.py       — Pydantic schemas for all API request/response bodies
│   ├── services/
│   │   ├── training_service.py   — 4-stage ingestion pipeline; Celery dispatch
│   │   ├── chat_service.py       — input rail; ADK runner; stream_chat() SSE generator
│   │   └── inference_service.py  — loads checkpoint; builds state vector; runs actors
│   ├── data/
│   │   ├── sources/
│   │   │   ├── xlsx.py      — XLSXDataSource: parse files, compute return_pct + macd_hist
│   │   │   └── database.py  — DatabaseMarketDataSource + DatabaseESGDataSource
│   │   ├── preprocessing/
│   │   │   ├── normalizer.py — DataNormalizer: time-series min-max per asset;
│   │   │   │                    fit_transform, transform_market_only, to_param_records,
│   │   │   │                    load_from_db (reconstructs frozen scaler from DB)
│   │   │   └── indicators.py — compute_rsi, compute_macd_histogram (fallback only;
│   │   │                         DB path skips these — RSI from XLSX, MACD pre-computed)
│   │   └── pipeline.py      — DataPipeline.prepare(): aligns dates, drops MACD warmup
│   │                           rows, applies normalizer, assembles ProcessedDataset
│   ├── rl/
│   │   ├── networks.py      — ActorNetwork (10N → 256 → 256 → μ,log_σ, no tanh)
│   │   │                       CriticNetwork (33N → 256 → 256 → Q, twin)
│   │   ├── masac.py         — MASAC: owns 3 actors + 6 critics + 6 targets +
│   │   │                       3 log_alpha_t; select_actions, update, save, load
│   │   ├── replay_buffer.py — UniformReplayBuffer: capacity 1M, stores
│   │   │                       (obs, a^B, a^L, a^F, r^B, r^L, r^F, next_obs, done)
│   │   ├── environment.py   — MarketEnvironment: step(), reset(); reward A/B/C;
│   │   │                       _effective_beta() per topology
│   │   └── trainer.py       — TrainingOrchestrator: 500k-step loop; warmup; convergence
│   │                           check; checkpoint selection on val Sharpe; Redis publish
│   └── workers/
│       ├── tasks.py         — Celery task run_training_job: loads DB sources + frozen
│       │                       normalizer; passes val_dataset to TrainingOrchestrator
│       └── __init__.py
```

---

## 5. Data Pipeline — 4 Stages

### Stage 1 — XLSX Parsing & Raw Storage

```
POST /api/v1/training/start  (multipart: .xlsx files + form fields)
         │
         ▼
XLSXDataSource.parse_files(paths)
  ├── Sheet: Stock_ESG_Dataset
  ├── Columns: Date | ISIN | Company name | Sector | Open | High | Low | Close
  │            | Volume | RSI | Bloom. ESG (0-100) | LESG ESG (0-10)
  ├── Volume parsed: "10.5M" → 10,500,000  (K/M/B/T, case-insensitive)
  ├── RSI: read as-is from XLSX (NOT recomputed)
  ├── return_pct = df.groupby("isin")["close"].pct_change()   ← NULL for first row/ISIN
  ├── macd_hist  = MACD(close, fast=12, slow=26, signal=9)     ← NULL for first 25 rows/ISIN
  └── Dedup on (ISIN, Date) — safe to upload overlapping files
         │
         ▼
Bulk upsert to PostgreSQL (ON CONFLICT DO UPDATE):
  ├── assets                  (isin, name, sector)
  ├── market_data             (ohlcv, rsi, return_pct, macd_hist)
  └── esg_scores              (bloomberg_score, lesg_score — raw)
```

### Stage 2 — Cross-Sectional ESG Normalization

```
For each date t, across all N ISINs:
  ESG_B_norm(i,t) = (ESG_B(i,t) − min_i ESG_B(t)) / (max_i ESG_B(t) − min_i ESG_B(t))
  ESG_L_norm(i,t) = same formula for LESG
  delta_esg(i,t)  = |ESG_B_norm − ESG_L_norm|   ← ESG disagreement
  mu_esg(i,t)     = (ESG_B_norm + ESG_L_norm) / 2  ← ESG consensus

Implementation: df.groupby("date").transform(cs_norm) — one pandas pass over all N assets
Result: bulk UPDATE esg_scores (esg_b_norm, esg_l_norm, delta_esg, mu_esg)
```

Why cross-sectional: ESG scores are relative signals — meaningful only compared to peers
on the same day, not across time.

### Stage 3 — Time-Series Normalizer Fitting

```
For the training window only (first 80% of dates by default):
  For each ISIN i, for each of 8 features:
    open, high, low, close, volume, return_pct, rsi, macd_hist

  min_val(i, feature) = min over training window
  max_val(i, feature) = max over training window

Stored in: training_normalizer_params — 8 × N rows per job_id
Frozen after fit — val window and inference use these exact min/max values (no look-ahead)
```

### Stage 4 — MASAC Training (Celery Background)

```
Celery worker receives job_id + config
         │
         ▼
DatabaseMarketDataSource → SELECT market_data WHERE isin IN (...) AND date BETWEEN ...
DatabaseESGDataSource    → SELECT esg_scores   WHERE isin IN (...) AND date BETWEEN ...
DataNormalizer.load_from_db(job_id) → reconstructs frozen scaler from training_normalizer_params
DataPipeline.prepare()   → drops MACD warmup rows (first macd_slow=26 rows/ISIN),
                            applies frozen normalizer, assembles ProcessedDataset
         │
         ▼
For each topology in [cooperative, competitive, mixed]:
  TrainingOrchestrator.run()
    ├── 10,000 warmup steps: random actions, fill ReplayBuffer (no gradient updates)
    ├── Steps 10,000–500,000: actor inference → env.step → buffer.add → masac.update
    ├── Every 500 steps: publish metrics to Redis PubSub (WebSocket stream)
    ├── Every 10,000 steps: eval on 63-day val window (deterministic actions)
    │     if Sharpe > best_sharpe → save checkpoint to model_store/{job_id}/{topology}/
    └── Convergence: rolling std of mean entropy over 100 steps < 0.01 → stop early
         │
         ▼
model_store/{job_id}/{topology}/bloomberg.pt
model_store/{job_id}/{topology}/lesg.pt
model_store/{job_id}/{topology}/financial.pt

Each .pt: {actor, critic_1, critic_2, critic_1_target, critic_2_target, log_alpha_t} state dicts
```

---

## 6. Neural Network Architecture

### Actor Network — 3 total (one per agent)

```
Input:  state_t ∈ ℝ^(10N)
        [open(N) | high(N) | low(N) | close(N) | volume(N) |
         rsi(N)  | macd(N) | return(N) | ΔESG(N) | μESG(N)]
              │
   Linear(10N → 256) + ReLU
              │
   Linear(256 → 256) + ReLU
              │
        ┌─────┴─────┐
        ▼           ▼
  Linear(256→N)  Linear(256→N)
       μ_π          log σ²
        │           │
        └─────┬─────┘
              │
  Training: z ~ N(μ_π, σ²)   [stochastic — entropy-regularised]
  Inference: z = μ_π          [deterministic — mean output only]
              │
  (No tanh — direct to z_joint → Softmax)
```

### Critic Network — twin per agent, 6 total + 6 target copies

```
Input:  [obs_B(10N) | obs_L(10N) | obs_F(10N)] ++ [a_B(N) | a_L(N) | a_F(N)]
        = ℝ^(33N)   [CTDE: centralized view of all agents]
              │
   Linear(33N → 256) + ReLU
              │
   Linear(256 → 256) + ReLU
              │
   Linear(256 → 1) → Q-value (scalar)

Twin: Q = min(Q₁, Q₂)  — reduces overestimation (Clipped Double-Q)
Target: θ⁻ ← τ·θ + (1−τ)·θ⁻,  τ = 0.005
```

---

## 7. State Vector Construction

Used identically during training, validation, and inference.

| Position   | Feature        | Source                   | Normalization                         |
| ---------- | -------------- | ------------------------ | ------------------------------------- |
| 0 … N-1    | Open           | `market_data.open`       | Time-series min-max (frozen per ISIN) |
| N … 2N-1   | High           | `market_data.high`       | Time-series min-max (frozen per ISIN) |
| 2N … 3N-1  | Low            | `market_data.low`        | Time-series min-max (frozen per ISIN) |
| 3N … 4N-1  | Close          | `market_data.close`      | Time-series min-max (frozen per ISIN) |
| 4N … 5N-1  | Volume         | `market_data.volume`     | Time-series min-max (frozen per ISIN) |
| 5N … 6N-1  | RSI            | `market_data.rsi`        | Time-series min-max (frozen per ISIN) |
| 6N … 7N-1  | MACD histogram | `market_data.macd_hist`  | Time-series min-max (frozen per ISIN) |
| 7N … 8N-1  | Return Rᵢₜ     | `market_data.return_pct` | Time-series min-max (frozen per ISIN) |
| 8N … 9N-1  | ΔESG           | `esg_scores.delta_esg`   | Cross-sectional (Stage 2, pre-stored) |
| 9N … 10N-1 | μESG           | `esg_scores.mu_esg`      | Cross-sectional (Stage 2, pre-stored) |

N is always dynamic — derived from whatever ISINs are in the uploaded XLSX, never hardcoded.

---

## 8. MASAC Training Loop

```
Initialize per topology:
  3 Actor networks, 6 Critic networks, 6 Target critics
  1 shared ReplayBuffer (capacity 1,000,000)
  3 temperature params log_alpha_t (init 1.0, auto-tuned per agent)
  Target entropy H̄ = −N (negative n_assets)

Warmup — steps 0 to 10,000:
  Random actions → env.step → buffer.add   (no gradient updates)

Training — steps 10,000 to 500,000 (or convergence):
  Per step:
    1. actor_B.deterministic_action(obs) / sample_action(obs) → z_B  (N,)
    2. actor_L / actor_F similarly                              → z_L, z_F
    3. z_joint = (z_B + z_L + z_F) / 3
    4. weights = softmax(z_joint)  → env.step(weights)
    5. Rewards: r^B, r^L, r^F  (topology-specific β applies here)
    6. buffer.add(obs, a_B, a_L, a_F, r_B, r_L, r_F, next_obs, done)
    7. batch = buffer.sample(256)
    8. For each agent i in {B, L, F}:
         critic_loss: Bellman error on Q1ᵢ, Q2ᵢ
           TD target = rᵢ + γ·(min(Q1ᵢ_tgt, Q2ᵢ_tgt) − α_T·log π(a'|s'))
         actor_loss: −E[min(Q1ᵢ, Q2ᵢ) − α_T·log π(a|s)]
         alpha_loss: −E[log_α_T·(log π(a|s) + H̄)]
    9. Soft-update: θ⁻ ← 0.005·θ + 0.995·θ⁻
   10. Every 500 steps: Redis PubSub publish (step, entropy, rewards, losses, alpha_t)
   11. Every 10,000 steps: validate on 63-day held-out window
         if val_sharpe > best_sharpe → save checkpoint → DB row in model_checkpoints
   12. Convergence: rolling std of mean entropy (100 steps) < 0.01 → break

Episode management:
  length: 252 trading days (one year)
  reset:  on episode end or dataset exhaustion — env.reset() returns obs at t=0
  max:    500,000 steps total across all episodes
```

---

## 9. Chat Interface (Google ADK)

The chat endpoint is the only user-facing inference path. There is no separate `/portfolio/generate` endpoint. Two endpoints are available — blocking and streaming (SSE).

### 9.1 Agent Architecture

A **3-agent pipeline** runs on every relevant request:

| Agent                 | Model                   | Role                                                               |
| --------------------- | ----------------------- | ------------------------------------------------------------------ |
| `portfolio_advisor`   | `gemini-2.5-flash`      | Orchestrator — parses intent, calls tools, synthesises output      |
| `market_intelligence` | `gemini-2.5-flash-lite` | Sub-agent — live macro/sector/earnings research (Google Search)    |
| `esg_research`        | `gemini-2.5-flash-lite` | Sub-agent — Bloomberg vs LESG ratings, controversies, ΔESG context |

`portfolio_advisor` is rebuilt per-request (factory function) so tool closures can capture per-request `ChatService` state. Sub-agents are module-level singletons.

### 9.2 Guardrails (Two Layers)

**Layer 1 — Input Rail:** A fast Gemini Flash-Lite call classifies every message before the advisor runs:

| Category       | Action                                                      |
| -------------- | ----------------------------------------------------------- |
| `relevant`     | Pass through to advisor                                     |
| `off_topic`    | Canned redirect, skip advisor                               |
| `abusive`      | Professional decline, skip advisor                          |
| `system_probe` | "I'm not able to share configuration details", skip advisor |
| `jailbreak`    | Canned redirect, skip advisor                               |

Fails open — classification errors default to `relevant` so legitimate users are never blocked by infrastructure failures.

**Layer 2 — Instruction Guardrails:** A `GUARDRAILS` section in the advisor's system prompt covers gray-area cases: subtle allocation-bypass ("just tell me what % to put in Apple"), nuanced system probing framed as financial questions, and mixed jailbreak+real requests.

### 9.3 Request Flow

```
POST /api/v1/chat  (or /chat/stream)
  { "message": "I have $10M. Use Portfolio C.", "session_id": "optional-uuid" }
          │
          ▼
  Input Rail (Gemini Flash-Lite)
  Blocked? → canned response returned immediately
  Relevant? → continue
          │
          ▼
  DatabaseSessionService (PostgreSQL-backed, persists across restarts)
  app_name = "madrl_portfolio",  user_id = str(authenticated_user.id)
  session_id = per-conversation UUID (client-supplied or server-generated)
          │
          ▼
  Runner.run_async(RunConfig(max_llm_calls=25))
          │
          ▼
  portfolio_advisor (Gemini 2.5 Flash)
    Model routing (no user prompting):
      "model A" → portfolio_model="A"
      "model B" → portfolio_model="B"
      anything else → portfolio_model="C"
          │
    ├── AgentTool: market_intelligence (Gemini Flash-Lite + Google Search)
    ├── AgentTool: esg_research (Gemini Flash-Lite + Google Search)
    ├── Tool: generate_portfolio(portfolio_model, investment_amount, max_assets=3)
    │         → InferenceService.run() → 3 topology panels via MASAC actors
    └── Tool: list_available_models() (only when user explicitly asks)
          │
          ▼
  /chat   → full response returned when pipeline completes (202)
  /stream → SSE: status events → text_chunk events → done event
          │
          ▼
  Route persists user + assistant messages to chat_messages table (failure-safe)

ChatResponse:
  { session_id, response, job_id, portfolio_model,
    panels: { "C_cooperative": [...], "C_competitive": [...], "C_mixed": [...] } }
```

### 9.4 SSE Streaming Events (`/chat/stream`)

Each line: `data: {json}\n\n`

| `type`       | When                        | Key fields                                                                    |
| ------------ | --------------------------- | ----------------------------------------------------------------------------- |
| `status`     | Stage change                | `status` (`thinking`\|`calling_tool`), `agent`, `tool`, `label`, `content:""` |
| `text_chunk` | Partial text from any agent | `agent`, `label`, `content`                                                   |
| `done`       | All agents finished         | `session_id`, `response`, `portfolio_result`                                  |
| `error`      | Unhandled exception         | `message`                                                                     |

### 9.5 Session Persistence

ADK conversation history → `DatabaseSessionService` (PostgreSQL).
User-visible chat history → `chat_sessions` + `chat_messages` ORM tables (REST-queryable).
Sessions auto-named from first message (truncated to 60 chars). Persistence failure-safe.

**Key implementation notes:**

- `OTEL_SDK_DISABLED=true` set in Python code (NOT in `.env` — Pydantic rejects unknown env vars)
- `os.environ.setdefault("GOOGLE_API_KEY", cfg.google_api_key)` — pydantic-settings does not populate `os.environ`; ADK reads it directly from `os.environ`
- No `break` in the ADK async generator — `GeneratorExit` through OTel context managers raises `ValueError: Token was created in a different Context` on Python 3.12+; generator exhausts naturally
- `max_llm_calls=25` — accounts for multi-agent pipelines where sub-agents consume several calls internally

---

## 10. Storage Layer

### PostgreSQL Schema

```
users
  id, email (unique), username (unique), hashed_password, role ("user"|"admin"),
  is_active, created_at, updated_at

assets
  id, isin (unique), name, sector, created_at

market_data                                      ← unique (asset_id, date)
  id, asset_id → assets.id, date
  open, high, low, close, volume               ← raw from XLSX
  rsi                                          ← raw from XLSX (NOT recomputed)
  return_pct                                   ← computed: pct_change(close) per ISIN
  macd_hist                                    ← computed: MACD(12,26,9) per ISIN

esg_scores                                       ← unique (asset_id, date)
  id, asset_id → assets.id, date
  bloomberg_score, lesg_score                  ← raw (0-100 / 0-10)
  esg_b_norm, esg_l_norm                       ← cross-sectional [0,1] (Stage 2)
  delta_esg                                    ← |esg_b_norm − esg_l_norm|
  mu_esg                                       ← (esg_b_norm + esg_l_norm) / 2

training_jobs
  id, status, portfolio_model, topology, config_json (N, dates, hyperparams)
  current_step, best_sharpe, best_mu_esg
  error_message, started_at, completed_at, created_at

model_checkpoints                                ← unique best per (job_id, topology)
  id, job_id → training_jobs.id, topology
  step, path (filesystem), sharpe, mu_esg, entropy, saved_at

training_normalizer_params                       ← unique (job_id, isin, feature_name)
  id, job_id → training_jobs.id, isin
  feature_name  ("open"|"high"|"low"|"close"|"volume"|"rsi"|"return_pct"|"macd_hist")
  min_val, max_val
  8 × N rows per job; N is dynamic — derived from XLSX contents

chat_sessions                                    ← one row per conversation
  id, user_id → users.id, session_id (UUID, unique), name
  created_at, updated_at

chat_messages                                    ← CASCADE delete with chat_sessions
  id, chat_session_id → chat_sessions.id, role ("user"|"assistant"), content
  created_at
```

### Redis Usage

```
pubsub:training:{job_id}        — step metrics streamed to WS clients during training
training:snapshot:{job_id}      — TTL 1h — last published message (for late WS subscribers)
stop:{job_id}                   — set by POST /training/{job_id}/stop → Celery checks between topologies
redis://localhost:6379/0        — main (session cache, snapshots)
redis://localhost:6379/1        — Celery broker
redis://localhost:6379/2        — Celery result backend
```

### Model Store (Filesystem)

```
model_store/{job_id}/{topology}/bloomberg.pt
model_store/{job_id}/{topology}/lesg.pt
model_store/{job_id}/{topology}/financial.pt

Each .pt contains state dicts for:
  actor, critic_1, critic_2, critic_1_target, critic_2_target, log_alpha_t
```

---

## 11. API Reference

### Auth Endpoints (public)

```
POST /api/v1/auth/register
Body:     { "email": "...", "username": "...", "password": "..." }
Response: UserProfile (201)

POST /api/v1/auth/login
Body:     { "email": "...", "password": "..." }
Response: { "access_token": "...", "token_type": "bearer", "role": "user"|"admin" }
```

### Auth Endpoints (protected — Bearer token required)

```
GET  /api/v1/auth/me                → UserProfile
PUT  /api/v1/auth/me                → UserProfile (email/username/password update)
GET  /api/v1/auth/users             → UserListResponse  [admin only]
PUT  /api/v1/auth/users/{id}/role   → UserProfile       [admin only]
```

### Training Endpoints

```
POST /api/v1/training/start         [admin]  multipart/form-data
  files:            .xlsx files (sheet: Stock_ESG_Dataset)
  portfolio_model:  A | B | C
  topology:         cooperative | competitive | mixed | all
  train_start/end, val_start/end:   YYYY-MM-DD (auto-split 80/20 if omitted)
  hyperparams_json: '{"alpha_1":0.5,"alpha_2":0.5,"alpha_3":0.01,"beta":0.3,"lam":0.4}'
Response: { "job_id": int, "status": "queued", "message": "..." }  (202)

GET  /api/v1/training/{job_id}/status   [any user]
Response: { job_id, status, step, max_steps, progress_pct, best_sharpe, best_mu_esg, ... }

POST /api/v1/training/{job_id}/stop     [admin]
Response: { "job_id": int, "stop_requested": true }
```

### Data Endpoints

```
GET  /api/v1/data/assets?sector=Technology   [any user]
Response: { "assets": [{ "isin", "name", "sector" }], "total": N }

GET  /api/v1/data/health    [public]
Response: { "status": "ok", "database": "connected" }
```

### Chat Endpoints

```
POST /api/v1/chat   [any user]   (202)
Body:     { "message": "I have $10M. Use Portfolio C.", "session_id": null }
Response: {
  "session_id": "uuid",
  "response":   "...",
  "job_id":     16,
  "portfolio_model": "C",
  "panels": {
    "C_cooperative": [{ isin, company, sector, return_ann, risk, sharpe, mu_esg, delta_esg, weight, allocation }],
    "C_competitive": [...],
    "C_mixed":       [...]
  }
}
panels is null for conversational queries. Error 503 if Gemini API unavailable.

POST /api/v1/chat/stream   [any user]   (200)  text/event-stream
Body: same as /chat
Stream: data: {json}\n\n  — event types: status | text_chunk | done | error
  status:     { type, status, agent, tool?, label, content:"" }
  text_chunk: { type, agent, label, content }
  done:       { type, session_id, response, portfolio_result }
  error:      { type, message }

GET  /api/v1/chat/sessions          [any user]   (200)
Response: { "sessions": [...], "total": N }  — ordered by last active

GET  /api/v1/chat/sessions/{id}     [any user]   (200)
Response: { id, session_id, name, created_at, updated_at, messages: [...] }
Errors: 403 (not owner), 404 (not found)

PATCH /api/v1/chat/sessions/{id}    [any user]   (200)
Body:     { "name": "New name" }
Response: ChatSessionInfo

DELETE /api/v1/chat/sessions/{id}   [any user]   (204)
Effect: deletes session + all messages; best-effort ADK session cleanup
Errors: 403 (not owner), 404 (not found)
```

### WebSocket

```
WS /ws/training/{job_id}?token=<jwt>
  → 4001 close if token invalid
  → snapshot delivered immediately on connect (late subscriber support)

Server messages:
  { "type": "warmup",    "step": 500, "warmup_total": 10000, "message": "..." }
  { "type": "step",      "step": 12000, "entropy": 3.1, "entropy_rolling_std": 0.04,
    "reward_bloomberg": 0.012, "reward_lesg": 0.009, "reward_financial": 0.015,
    "loss_actor": 0.23, "loss_critic": 0.41, "alpha_t": 0.87 }
  { "type": "converged", "step": 234100, "final_sharpe": 1.44, "mu_esg": 0.71,
    "message": "Training converged at step 234100" }
  { "type": "error",     "message": "..." }
```

---

## 12. Configuration

All settings live in `.env` and are loaded via Pydantic Settings into `app/config.py`.

| Key                         | Default                                                           | Purpose                                                     |
| --------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------- |
| `POSTGRES_DSN`              | `postgresql+asyncpg://madrl:madrl@localhost:5432/madrl_portfolio` | Async PostgreSQL DSN                                        |
| `REDIS_URL`                 | `redis://localhost:6379/0`                                        | Main Redis (cache + pub/sub)                                |
| `CELERY_BROKER_URL`         | `redis://localhost:6379/1`                                        | Celery broker                                               |
| `CELERY_RESULT_BACKEND`     | `redis://localhost:6379/2`                                        | Celery result backend                                       |
| `JWT_SECRET_KEY`            | _(change in production)_                                          | HS256 signing key                                           |
| `JWT_ALGORITHM`             | `HS256`                                                           | Token signing algorithm                                     |
| `JWT_EXPIRE_MINUTES`        | `1440`                                                            | Token lifetime (24h)                                        |
| `ADK_MODEL`                 | `gemini-2.5-flash`                                                | Portfolio Advisor — orchestration & reasoning               |
| `ADK_MODEL_MARKET`          | `gemini-2.5-flash-lite`                                           | Market Intelligence + ESG Research sub-agents               |
| `ADK_MODEL_GUARD`           | `gemini-2.5-flash-lite`                                           | Input rail classifier — off-topic / jailbreak gate          |
| `GOOGLE_API_KEY`            | _(required for chat)_                                             | Injected into `os.environ` at startup                       |
| `MODEL_STORE_PATH`          | `./model_store`                                                   | Filesystem root for `.pt` checkpoint files                  |
| `MASAC_MAX_STEPS`           | `500000`                                                          | Maximum training steps per topology                         |
| `MASAC_WARMUP_STEPS`        | `10000`                                                           | Random-action warmup before gradient updates                |
| `MASAC_BATCH_SIZE`          | `256`                                                             | Transitions sampled per gradient update                     |
| `MASAC_GAMMA`               | `0.99`                                                            | Discount factor                                             |
| `MASAC_TAU`                 | `0.005`                                                           | Soft target update rate                                     |
| `MASAC_LR_ACTOR`            | `3e-4`                                                            | Actor learning rate                                         |
| `MASAC_LR_CRITIC`           | `3e-4`                                                            | Critic + temperature learning rate                          |
| `MASAC_HIDDEN_SIZE`         | `256`                                                             | Hidden layer width for all networks                         |
| `MASAC_CONVERGENCE_EPSILON` | `0.01`                                                            | Entropy rolling std threshold for early stop                |
| `MASAC_CONVERGENCE_WINDOW`  | `100`                                                             | Window size for convergence check                           |
| `MACD_FAST`                 | `12`                                                              | MACD fast EMA period                                        |
| `MACD_SLOW`                 | `26`                                                              | MACD slow EMA period (also = warmup rows dropped per ISIN)  |
| `MACD_SIGNAL`               | `9`                                                               | MACD signal EMA period                                      |
| `RSI_PERIOD`                | `14`                                                              | RSI window (normalization only — RSI values come from XLSX) |
| `VALIDATION_WINDOW_DAYS`    | `63`                                                              | Val window for checkpoint selection (~1 quarter)            |

> `OTEL_SDK_DISABLED=true` is set in Python code at module load — do **not** put it in `.env`
> (Pydantic rejects unknown env vars with `Extra inputs are not permitted`).

---

## 13. Infrastructure

```
Docker Compose services:
  api       FastAPI + uvicorn                 port 8000
  worker    Celery worker (solo pool, Windows-compatible)
  postgres  PostgreSQL 15                     port 5432
  redis     Redis 7                           port 6379
  flower    Celery monitoring UI              port 5555

Alembic migration required after any domain.py change:
  alembic revision --autogenerate -m "description"
  alembic upgrade head
```
