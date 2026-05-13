# MADRL Portfolio System — Full Architecture & Design

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MADRL Portfolio Platform                           │
│                                                                             │
│  ┌──────────┐   REST/WS   ┌──────────────────────────────────────────────┐ │
│  │  Client  │◄───────────►│           FastAPI Gateway                    │ │
│  │ (UI/API) │             │  /portfolio  /training  /data  /ws           │ │
│  └──────────┘             └──────────────┬───────────────────────────────┘ │
│                                          │                                  │
│              ┌───────────────────────────▼──────────────────────────────┐  │
│              │              ADK Orchestration Layer                      │  │
│              │                                                           │  │
│              │  ┌─────────────────────────────────────────────────────┐ │  │
│              │  │          PortfolioOrchestratorAgent (ADK)            │ │  │
│              │  │                                                       │ │  │
│              │  │   ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │ │  │
│              │  │   │ Bloomberg    │  │   LESG       │  │Financial │  │ │  │
│              │  │   │ ESGAgent     │  │  ESGAgent    │  │  Agent   │  │ │  │
│              │  │   │ (ADK)        │  │  (ADK)       │  │  (ADK)   │  │ │  │
│              │  │   │ α₁·ESG^(B)  │  │  α₂·ESG^(L) │  │  α₃≈0   │  │ │  │
│              │  │   └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │ │  │
│              │  │          │                  │               │        │ │  │
│              │  │          └──────────────────┴───────────────┘        │ │  │
│              │  │                        │ z_joint = avg(z^B,z^L,z^F)  │ │  │
│              │  │                   Softmax(z_joint) → weights          │ │  │
│              │  └─────────────────────────────────────────────────────┘ │  │
│              │                                                           │  │
│              │  ┌─────────────────┐  ┌─────────────────────────────┐    │  │
│              │  │  Training Agent │  │   Data Ingestion Agent       │    │  │
│              │  │  (ADK)          │  │   (ADK)                     │    │  │
│              │  └─────────────────┘  └─────────────────────────────┘    │  │
│              └───────────────────────────┬───────────────────────────────┘  │
│                                          │                                  │
│         ┌────────────────────────────────▼──────────────────────────────┐  │
│         │                      Service Layer                             │  │
│         │   PortfolioService  │  TrainingService  │  InferenceService   │  │
│         └────────┬───────────────────┬──────────────────────┬───────────┘  │
│                  │                   │                       │             │
│    ┌─────────────▼───┐  ┌────────────▼────────┐  ┌──────────▼──────────┐  │
│    │   RL Engine     │  │   Data Pipeline     │  │   Storage Layer     │  │
│    │                 │  │                     │  │                     │  │
│    │  ┌───────────┐  │  │  ┌──────────────┐  │  │  ┌───────────────┐  │  │
│    │  │   MASAC   │  │  │  │  Market API  │  │  │  │  PostgreSQL   │  │  │
│    │  │ Algorithm │  │  │  │  ESG API     │  │  │  │  (Portfolio,  │  │  │
│    │  ├───────────┤  │  │  ├──────────────┤  │  │  │   Training,   │  │  │
│    │  │ 3 Actors  │  │  │  │  Normalizer  │  │  │  │   Results)    │  │  │
│    │  │ 6 Critics │  │  │  │  RSI/MACD    │  │  │  └───────────────┘  │  │
│    │  ├───────────┤  │  │  └──────────────┘  │  │  ┌───────────────┐  │  │
│    │  │  Replay   │  │  └────────────────────┘  │  │     Redis     │  │  │
│    │  │  Buffer   │  │                          │  │  (Cache/PubSub│  │  │
│    │  └───────────┘  │                          │  │   /Sessions)  │  │  │
│    │                 │  ┌──────────────────────┐ │  └───────────────┘  │  │
│    │  Market         │  │  Celery Workers      │ │  ┌───────────────┐  │  │
│    │  Environment    │  │  (Background Train)  │ │  │  Model Store  │  │  │
│    └─────────────────┘  └──────────────────────┘ │  │  (File/S3)   │  │  │
│                                                   │  └───────────────┘  │  │
│                                                   └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Interaction Topologies — Parallel Execution

All three game-theoretic topologies run simultaneously for every query. Each produces
an independent portfolio recommendation returned side-by-side to the user.

```
User Query (Portfolio C, $10M)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│              PortfolioOrchestratorAgent                  │
│                                                         │
│  spawn_topology("cooperative")  ─────────────────────► │─► Panel 1
│  spawn_topology("competitive")  ─────────────────────► │─► Panel 2
│  spawn_topology("mixed")        ─────────────────────► │─► Panel 3
│                                                         │
│  [all three run concurrently via asyncio.gather]        │
└─────────────────────────────────────────────────────────┘
```

**Reward function by topology:**

| Topology    | β applied?           | Portfolio A             | Portfolio B           | Portfolio C                           |
|-------------|----------------------|-------------------------|-----------------------|---------------------------------------|
| Cooperative | β > 0 (full penalty) | rₜ + αᵢ·ESGᵢ           | rₜ ± λ·ΔESG_signed   | rₜ + αᵢ·ESGᵢ − β·ΔESGₜ              |
| Competitive | β = 0                | rₜ + αᵢ·ESGᵢ           | rₜ ± λ·ΔESG_signed   | rₜ + αᵢ·ESGᵢ                         |
| Mixed       | 0 < β_partial < β    | rₜ + αᵢ·ESGᵢ           | rₜ ± λ·ΔESG_signed   | rₜ + αᵢ·ESGᵢ − β_partial·ΔESGₜ      |

---

## 3. Component Architecture

### 3.1 FastAPI Gateway

```
app/
├── main.py                         # App factory, lifespan events
├── config.py                       # Pydantic Settings (env-driven)
└── api/
    ├── deps.py                     # DI: db, redis, services
    └── routes/
        ├── portfolio.py            # POST /portfolio/generate
        │                           # GET  /portfolio/{id}
        │                           # GET  /portfolio/{id}/comparison
        ├── training.py             # POST /training/start
        │                           # GET  /training/{job_id}/status
        │                           # POST /training/{job_id}/stop
        ├── data.py                 # POST /data/ingest
        │                           # GET  /data/assets
        │                           # GET  /data/health
        └── websocket.py            # WS  /ws/training/{job_id}
                                    # WS  /ws/portfolio/{session_id}
```

### 3.2 ADK Agent Layer

```
agents/
├── base.py                         # BasePortfolioAgent (ADK Agent subclass)
│                                   # — shared tools: fetch_state, log_decision
├── bloomberg_agent.py              # BloombergESGAgent
│                                   # — tool: compute_bloomberg_scores(state) → z^(B)
│                                   # — reward: rₜ + α₁·ESG^(B) [− β·ΔESGₜ if coop]
├── lesg_agent.py                   # LESGAgent
│                                   # — tool: compute_lesg_scores(state) → z^(L)
│                                   # — reward: rₜ + α₂·ESG^(L) [− β·ΔESGₜ if coop]
├── financial_agent.py              # FinancialAgent
│                                   # — tool: compute_financial_scores(state) → z^(F)
│                                   # — reward: rₜ  (α₃ ≈ 0)
└── portfolio_orchestrator.py       # PortfolioOrchestratorAgent
                                    # — sub_agents: [Bloomberg, LESG, Financial]
                                    # — tool: aggregate_scores(z^B, z^L, z^F) → weights
                                    # — tool: run_topology(mode, portfolio_model) → panel
                                    # — tool: generate_comparison() → 3 panels
```

### 3.3 RL Engine (MASAC)

```
rl/
├── networks.py                     # ActorNetwork, CriticNetwork (PyTorch)
│                                   # Input: 10N features | Output: μ_π, log σ²
├── masac.py                        # MASACAgent: owns 1 Actor + 2 Critics each
│                                   # 3 agents × (1 Actor + 2 Critics + 2 targets)
│                                   # = 3 Actors + 12 networks total
├── replay_buffer.py                # UniformReplayBuffer, capacity=1M
│                                   # stores (s,a^B,a^L,a^F,r^B,r^L,r^F,s')
├── environment.py                  # MarketEnvironment (Gym-compatible)
│                                   # — step(actions) → (obs, rewards, done, info)
│                                   # — reset() → initial_state
│                                   # — topology: "cooperative"|"competitive"|"mixed"
└── trainer.py                      # TrainingOrchestrator
                                    # — runs 500k steps max
                                    # — early stop: rolling std entropy < 0.01
                                    # — emits events to Redis PubSub
```

### 3.4 Data Pipeline

```
data/
├── sources/
│   ├── market.py                   # MarketDataSource
│   │                               # — fetch_ohlcv(isin, start, end) → DataFrame
│   │                               # — supports: yfinance, Bloomberg API, Alpha Vantage
│   └── esg.py                      # ESGDataSource
│                                   # — fetch_bloomberg_esg(isin, date) → score (0-100)
│                                   # — fetch_lesg_esg(isin, date) → score (0-10)
├── preprocessing/
│   ├── normalizer.py               # MinMaxNormalizer
│   │                               # — cross_sectional(esg_matrix) → normed per-day
│   │                               # — time_series(feature_matrix, window) → normed per-asset
│   │                               # — freeze/unfreeze for train/test split
│   └── indicators.py               # TechnicalIndicators
│                                   # — rsi(prices, period=14) → Series
│                                   # — macd_histogram(prices, 12, 26, 9) → Series
└── pipeline.py                     # DataPipeline
                                    # orchestrates source → preprocess → validate
                                    # enforces train/test split discipline
```

### 3.5 Storage Layer

```
PostgreSQL schema:
├── assets          (isin, sector, name, created_at)
├── market_data     (asset_id, date, open, high, low, close, volume)
├── esg_scores      (asset_id, date, bloomberg_score, lesg_score)
├── training_jobs   (id, status, model_type, topology, started_at, config_json)
├── model_checkpoints (job_id, step, path, sharpe, mu_esg, entropy, saved_at)
├── portfolios      (id, job_id, topology, model_type, allocation_json, metrics_json)
└── query_results   (id, query_json, cooperative_id, competitive_id, mixed_id, created_at)

Redis usage:
├── cache:market:{isin}:{date}      TTL 1h  — raw OHLCV
├── cache:esg:{isin}:{date}         TTL 24h — ESG scores
├── cache:state:{isin}:{window}     TTL 6h  — normalized state
├── pubsub:training:{job_id}        — step metrics streamed to WS clients
└── session:{session_id}            TTL 1h  — active portfolio sessions
```

---

## 4. Data Flow: End-to-End

### 4.1 Training Data Flow

```
Market API → fetch_ohlcv()
                │
                ▼
ESG APIs   → fetch_bloomberg_esg() + fetch_lesg_esg()
                │
                ▼
        ┌───────────────────────────────────┐
        │         DataPipeline              │
        │                                   │
        │  1. Cross-sectional ESG norm:     │
        │     ESG_norm(i,t) across N assets │
        │     per day t  (no look-ahead)    │
        │                                   │
        │  2. Compute per-stock:            │
        │     ΔESGᵢₜ = |ESG^B - ESG^L|     │
        │     μESGᵢₜ = (ESG^B + ESG^L)/2   │
        │                                   │
        │  3. Time-series norm (per asset,  │
        │     training window W only):      │
        │     Close, OHLCV, Rᵢₜ            │
        │     RSI(14), MACD histogram(26)   │
        │     Freeze min/max before test    │
        │                                   │
        │  4. Warm-up: skip first 26 days   │
        └───────────────────────────────────┘
                │
                ▼
        State vector per asset: 10N features
        [OHLCV(5N), RSI(N), MACD(N), Rᵢₜ(N), ΔESGᵢ(N), μESGᵢ(N)]
                │
                ▼
         MarketEnvironment → ReplayBuffer → MASAC Training
```

### 4.2 Inference (Portfolio Generation) Flow

```
User Request
    │
    ▼
PortfolioOrchestratorAgent.generate_comparison(
    model=PortfolioC, amount=$10M, assets=[...]
)
    │
    ├─ asyncio.gather([
    │     run_topology("cooperative"),
    │     run_topology("competitive"),
    │     run_topology("mixed")
    │  ])
    │
    │  Each topology:
    │    1. Load trained actor weights (from model store)
    │    2. Fetch + preprocess current market state
    │    3. BloombergAgent.compute_bloomberg_scores(state) → z^(B)
    │    4. LESGAgent.compute_lesg_scores(state)          → z^(L)
    │    5. FinancialAgent.compute_financial_scores(state) → z^(F)
    │    6. z_joint = (z^B + z^L + z^F) / 3
    │    7. weights = softmax(z_joint)
    │    8. Compute portfolio metrics (return, σ, Sharpe, μESG, ΔESG)
    │    9. Scale to user's allocation amount
    │
    ▼
ComparisonResponse {
    cooperative: Panel,   ← β > 0, full shared penalty
    competitive: Panel,   ← β = 0, no shared penalty
    mixed: Panel          ← 0 < β_partial < β
}
```

---

## 5. Reward Function Implementation Matrix

### Portfolio A — ESG Consensus Baseline

```python
# Agent 1 (Bloomberg):  R = r_t + α₁ · ESG_t^(B)
# Agent 2 (LESG):       R = r_t + α₂ · ESG_t^(L)
# Agent 3 (Financial):  R = r_t  (α₃ ≈ 0)
# β = 0 in all topologies for Portfolio A

def reward_A(r_t, esg_portfolio, alpha, topology, beta=0.0):
    penalty = -beta * delta_esg_t if topology == "cooperative" else 0.0
    return r_t + alpha * esg_portfolio + penalty
```

### Portfolio B — Signed Disagreement

```python
# Agent 1 (Bloomberg):  R = r_t + λ · (ESG_t^(B) − ESG_t^(L))
# Agent 2 (LESG):       R = r_t + λ · (ESG_t^(L) − ESG_t^(B))
# Agent 3 (Financial):  R = r_t
# Note: degenerate case when ESG_t^(B) == ESG_t^(L) → all agents get r_t only

def reward_B_bloomberg(r_t, esg_B, esg_L, lam):
    return r_t + lam * (esg_B - esg_L)

def reward_B_lesg(r_t, esg_B, esg_L, lam):
    return r_t + lam * (esg_L - esg_B)
```

### Portfolio C — Full Model (Consensus + Uncertainty Penalty)

```python
# Agent 1 (Bloomberg):  R = r_t + α₁·ESG_t^(B) − β·ΔESGₜ
# Agent 2 (LESG):       R = r_t + α₂·ESG_t^(L) − β·ΔESGₜ
# Agent 3 (Financial):  R = r_t − β·ΔESGₜ         (α₃ ≈ 0)
# β=0 in Competitive; β>0 in Cooperative/Mixed

def reward_C(r_t, esg_portfolio, alpha, delta_esg_t, beta, topology):
    effective_beta = beta if topology in ("cooperative", "mixed") else 0.0
    if topology == "mixed":
        effective_beta *= 0.5  # partial penalty
    return r_t + alpha * esg_portfolio - effective_beta * delta_esg_t
```

---

## 6. Neural Network Architecture

### Actor Network (per agent, 3 total)

```
Input: state_t ∈ ℝ^(10N)
  [OHLCV(5N) | RSI(N) | MACD(N) | R_i_t(N) | ΔESG_i(N) | μESG_i(N)]
                │
       Linear(10N → 256) + ReLU
                │
       Linear(256 → 256)  + ReLU
                │
         ┌──────┴──────┐
         ▼             ▼
   Linear(256→N)  Linear(256→N)
       μ_π             log σ²
         │             │
         └──────┬──────┘
                │
        z ~ N(μ_π, σ²)   [at training]
        z = μ_π           [at inference]
                │
        (No tanh — direct to Softmax)
```

### Critic Network (twin per agent, 6 total + 6 targets)

```
Input: [all_obs ∈ ℝ^(3·10N)] ++ [all_actions ∈ ℝ^(3·N)]
       = ℝ^(33N)   [CTDE: centralized view]
                │
       Linear(33N → 256) + ReLU
                │
       Linear(256 → 256) + ReLU
                │
       Linear(256 → 1)
                │
            Q-value (scalar)

Twin critics: Q = min(Q₁, Q₂)  [reduces overestimation bias]
Target nets:  θ⁻ ← τθ + (1-τ)θ⁻,  τ = 0.005
```

---

## 7. MASAC Training Loop

```
Initialize:
  - 3 Actor networks (one per agent)
  - 6 Critic networks (twin Q per agent)
  - 6 Target critic networks
  - 1 ReplayBuffer (capacity=1M)
  - 3 temperature parameters α_T (init=1.0, auto-tuned)
  - Target entropy H̄ = −N

Warmup (first 10,000 steps):
  - Random actions only, fill replay buffer
  - Skip first 26 trading days per episode (MACD stabilisation)

Per step:
  1. Each actor computes z^(i) from local obs s_t (decentralized)
  2. Sample from N(μ_π, σ²)  →  z^(i)_sampled
  3. z_joint = mean([z^B, z^L, z^F])
  4. weights = softmax(z_joint)
  5. Environment step: (r^B, r^L, r^F, s_{t+1}) = env.step(weights)
  6. Store (s, a^B, a^L, a^F, r^B, r^L, r^F, s') in replay buffer
  7. Sample batch (size=256) from replay buffer
  8. For each agent i ∈ {B, L, F}:
       a. Critic update: minimize Bellman error on Q₁ᵢ, Q₂ᵢ
          TD target = rᵢ + γ·(min(Q₁ᵢ_target, Q₂ᵢ_target) - α_T·log π(a'|s'))
       b. Actor update: maximize E[min(Q₁ᵢ, Q₂ᵢ) - α_T·log π(a|s)]
       c. Temperature update: minimize L(α_T) = E[-α_T·(log π + H̄)]
  9. Soft-update target networks: θ⁻ ← 0.005·θ + 0.995·θ⁻
  10. Check convergence: rolling std of mean entropy (100 steps) < 0.01

Episode management:
  - Episode length: 252 trading days
  - Reset: weights → equal (1/N each)
  - No early termination on drawdown
  - Max: 500,000 steps total

Hyperparameter search:
  - Grid search α₁, α₂ ∈ [0.1, 1.0], β ∈ [0.1, 1.0], λ ∈ [0.1, 1.0]
  - Validation metric: Sharpe Ratio (primary), μESG (secondary constraint)
  - Validation window: 63 trading days (rolling out-of-sample)
```

---

## 8. API Specification

### REST Endpoints

```
POST /api/v1/portfolio/generate
Body: {
  "assets": ["ISIN1", "ISIN2", ...],
  "portfolio_model": "A" | "B" | "C",
  "allocation_amount": 10000000,
  "hyperparams": {
    "alpha_1": 0.5, "alpha_2": 0.5, "alpha_3": 0.01,
    "beta": 0.3, "lambda": 0.4
  },
  "date": "2024-01-15"
}
Response: {
  "query_id": "uuid",
  "cooperative":  { ...Panel },
  "competitive":  { ...Panel },
  "mixed":        { ...Panel }
}

Panel schema:
{
  "topology": "cooperative",
  "portfolio": [
    {
      "isin": "US03783...",
      "sector": "Tech",
      "weight": 0.40,
      "allocation": 4000000,
      "return_ann": 0.22,
      "risk_ann": 0.12,
      "sharpe": 1.83,
      "mu_esg": 0.93,
      "delta_esg": 0.14
    }, ...
  ],
  "aggregate_metrics": {
    "portfolio_return": 0.19,
    "portfolio_risk": 0.14,
    "portfolio_sharpe": 1.36,
    "portfolio_mu_esg": 0.72,
    "portfolio_delta_esg": 0.30
  },
  "strategic_summary": "..."
}

POST /api/v1/training/start
Body: {
  "portfolio_model": "C",
  "topology": "cooperative" | "competitive" | "mixed" | "all",
  "assets": [...],
  "train_start": "2018-01-01",
  "train_end": "2022-12-31",
  "val_start": "2023-01-01",
  "val_end": "2023-12-31",
  "hyperparams": { ... }
}
Response: { "job_id": "uuid", "status": "queued" }

GET /api/v1/training/{job_id}/status
Response: {
  "job_id": "uuid",
  "status": "running" | "completed" | "failed",
  "step": 42300,
  "max_steps": 500000,
  "entropy_rolling_std": 0.023,
  "best_sharpe": 1.41,
  "best_mu_esg": 0.68
}

GET /api/v1/portfolio/{id}/comparison
Response: { "cooperative": Panel, "competitive": Panel, "mixed": Panel }

GET /api/v1/data/assets?sector=Tech
Response: { "assets": [{ "isin": "...", "sector": "...", "name": "..." }] }
```

### WebSocket Events

```
WS /ws/training/{job_id}

Server → Client (every step):
{ "type": "step", "step": 1500, "entropy": 3.2, "reward_B": 0.012, "reward_L": 0.009, "reward_F": 0.015 }

Server → Client (convergence):
{ "type": "converged", "step": 234100, "final_sharpe": 1.44, "mu_esg": 0.71 }

Server → Client (error):
{ "type": "error", "message": "Training diverged: NaN loss" }

WS /ws/portfolio/{session_id}

Client → Server:  { "action": "update_weights", "hyperparams": { "beta": 0.6 } }
Server → Client:  { "type": "recomputed", "cooperative": Panel, ... }
```

---

## 9. Infrastructure & Deployment

```
docker-compose services:
├── api          FastAPI (uvicorn, 4 workers)     port 8000
├── worker       Celery worker (training jobs)     —
├── beat         Celery beat (scheduled tasks)     —
├── postgres     PostgreSQL 15                     port 5432
├── redis        Redis 7                           port 6379
├── flower       Celery monitoring                 port 5555
└── prometheus   Metrics scraping                 port 9090

Environment variables (via .env):
POSTGRES_DSN, REDIS_URL,
BLOOMBERG_API_KEY, LESG_API_KEY,
MODEL_STORE_PATH,
CELERY_BROKER_URL,
ADK_MODEL (e.g. "gemini-2.0-flash" or "claude-sonnet-4-6"),
GOOGLE_API_KEY / ANTHROPIC_API_KEY
```

---

## 10. File Structure

```
madrl_portfolio/
├── ARCHITECTURE.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── portfolio.py
│   │       ├── training.py
│   │       ├── data.py
│   │       └── websocket.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bloomberg_agent.py
│   │   ├── lesg_agent.py
│   │   ├── financial_agent.py
│   │   └── portfolio_orchestrator.py
│   ├── rl/
│   │   ├── __init__.py
│   │   ├── networks.py
│   │   ├── masac.py
│   │   ├── replay_buffer.py
│   │   ├── environment.py
│   │   └── trainer.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── market.py
│   │   │   └── esg.py
│   │   ├── preprocessing/
│   │   │   ├── __init__.py
│   │   │   ├── normalizer.py
│   │   │   └── indicators.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── domain.py
│   │   └── schemas.py
│   └── services/
│       ├── __init__.py
│       ├── portfolio_service.py
│       ├── training_service.py
│       └── inference_service.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_normalizer.py
    ├── test_masac.py
    ├── test_agents.py
    └── test_api.py
```
