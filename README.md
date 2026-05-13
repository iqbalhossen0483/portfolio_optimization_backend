# MADRL Portfolio System

A **Multi-Agent Deep Reinforcement Learning** portfolio optimization system that resolves ESG rating disagreement between Bloomberg and LESG through a three-agent MASAC framework. The system runs three game-theoretic topologies — Cooperative, Competitive, and Mixed — concurrently for every query and returns side-by-side portfolio panels for comparison.

---

## Architecture at a Glance

```
User Query
    │
    ▼
FastAPI  ──►  PortfolioOrchestratorAgent (Google ADK)
                  │         │         │
             Bloomberg   LESG      Financial
             ESGAgent    ESGAgent   Agent
                  │         │         │
                  └────┬────┘         │
                  z_joint = avg(z^B, z^L, z^F)
                  Softmax → portfolio weights
                  │
          ┌───────┼───────┐
     Cooperative Competitive Mixed    ← three panels returned simultaneously
```

**Stack:** FastAPI · Google ADK · PyTorch MASAC · PostgreSQL · Redis · Celery

---

## Prerequisites

| Requirement             | Minimum Version               |
| ----------------------- | ----------------------------- |
| Python                  | 3.11                          |
| Docker & Docker Compose | Docker 24                     |
| Git                     | any                           |
| NVIDIA GPU _(optional)_ | CUDA 12.x for faster training |

---

## Quick Start — Docker (Recommended)

### 1. Clone and configure

```bash
git clone <repo-url> madrl_portfolio
cd madrl_portfolio

cp .env.example .env
```

Open `.env` and fill in at minimum:

```env
GOOGLE_API_KEY=your_google_api_key_here   # required for ADK agents
```

All other values default to the Docker service names and are pre-wired in `docker-compose.yml`.

### 2. Start all services

```bash
docker compose up --build
```

This starts:

| Service                     | URL                         |
| --------------------------- | --------------------------- |
| **API**                     | http://localhost:8000       |
| **Swagger docs**            | http://localhost:8000/docs  |
| **ReDoc**                   | http://localhost:8000/redoc |
| **Flower** (Celery monitor) | http://localhost:5555       |
| **Prometheus**              | http://localhost:9090       |
| PostgreSQL                  | localhost:5432              |
| Redis                       | localhost:6379              |

### 3. Verify the API is running

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"1.0.0"}
```

---

## Local Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
Services are split between `uv run` (Python processes) and Docker (infrastructure).

### Service map

| Service | How to run | URL |
|---|---|---|
| **FastAPI** | `uv run` | http://localhost:8000 |
| **Celery worker** | `uv run` | — |
| **Flower** | `uv run` | http://localhost:5555 |
| **PostgreSQL** | Docker | localhost:5432 |
| **Redis** | Docker | localhost:6379 |
| **Prometheus** | Docker | http://localhost:9090 |

> Prometheus runs in Docker but scrapes the `uv`-hosted API on your machine via
> `host.docker.internal:8000` — no special config needed.

---

### 1. Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install all dependencies

```bash
uv sync
uv sync --group dev      # include dev/test dependencies
```

`uv sync` reads `pyproject.toml`, creates `.venv` automatically, installs all
packages, and registers the `app` package in editable mode — no separate
`pip install -e .` step needed.

### 3. Configure environment

```bash
cp .env.example .env
```

Minimum required `.env` settings for local development:

```env
POSTGRES_DSN=postgresql+asyncpg://madrl:madrl@localhost:5432/madrl_portfolio
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
GOOGLE_API_KEY=your_google_api_key_here
MODEL_STORE_PATH=./model_store
```

### 4. Start infrastructure services (Docker)

```bash
docker compose up postgres redis prometheus -d
```

This starts PostgreSQL, Redis, and Prometheus. Prometheus will immediately begin
scraping http://host.docker.internal:8000/metrics — it retries automatically
until the API is up.

### 5. Start the API server

Open a terminal in the project root and run:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 6. Start the Celery worker

Open a second terminal in the project root:

```bash
# Windows
uv run celery -A celery_app worker --loglevel=info --pool=solo

# macOS / Linux
uv run celery -A celery_app worker --loglevel=info --concurrency=2
```

> **Windows note:** The default `prefork` pool uses `os.fork()` which is not available
> on Windows. Use `--pool=solo` for local development.

### 7. Start Flower (Celery monitor)

Open a third terminal in the project root:

```bash
uv run celery -A celery_app flower --port=5555
```

### All services at a glance

| Terminal | Command | Opens |
|---|---|---|
| 1 | `uv run uvicorn app.main:app --reload --port 8000` | API + Swagger |
| 2 | `uv run celery -A celery_app worker --loglevel=info --pool=solo` | Worker |
| 3 | `uv run celery -A celery_app flower --port=5555` | Flower UI |
| Docker | `docker compose up postgres redis prometheus -d` | DB + Cache + Metrics |

---

## Running the System

### Step 1 — Ingest market and ESG data

```bash
curl -X POST http://localhost:8000/api/v1/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "assets": ["AAPL", "XOM", "JNJ", "HSBA.L"],
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "sources": ["market", "bloomberg", "lesg"]
  }'
```

> **Note:** Without API keys for Bloomberg and LESG, the system uses deterministic synthetic ESG data automatically. Set `BLOOMBERG_API_KEY` and `LESG_API_KEY` in `.env` to switch to live data.

---

### Step 2 — Train MASAC agents

```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_model": "C",
    "topology": "all",
    "assets": ["AAPL", "XOM", "JNJ", "HSBA.L"],
    "train_start": "2018-01-01",
    "train_end": "2022-12-31",
    "val_start": "2023-01-01",
    "val_end": "2023-12-31",
    "hyperparams": {
      "alpha_1": 0.5,
      "alpha_2": 0.5,
      "alpha_3": 0.01,
      "beta": 0.3,
      "lam": 0.4
    }
  }'
# Returns: {"job_id": "uuid", "status": "queued"}
```

**Portfolio models:**

| Model | Description                                                         |
| ----- | ------------------------------------------------------------------- |
| `A`   | ESG consensus — tests whether Bloomberg + LESG consensus adds alpha |
| `B`   | Signed disagreement — each agent bets its own ESG source is correct |
| `C`   | Full model — consensus + uncertainty penalty (recommended)          |

**Topology options:** `cooperative` · `competitive` · `mixed` · `all` (trains all three)

---

### Step 3 — Monitor training progress

#### Poll via REST

```bash
curl http://localhost:8000/api/v1/training/{job_id}/status
```

#### Stream via WebSocket

```bash
# Using wscat (npm install -g wscat)
wscat -c ws://localhost:8000/ws/training/{job_id}
```

Messages streamed every 500 steps:

```json
{"type": "step", "step": 12500, "entropy": 2.8, "entropy_rolling_std": 0.045, "reward_bloomberg": 0.018, "reward_lesg": 0.014, "reward_financial": 0.021}
{"type": "converged", "step": 234100, "final_sharpe": 1.44, "mu_esg": 0.71}
```

Training stops automatically when the rolling standard deviation of mean policy entropy (100-step window) falls below **0.01**, or at **500,000 steps** maximum.

---

### Step 4 — Generate a portfolio

```bash
curl -X POST http://localhost:8000/api/v1/portfolio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "assets": ["AAPL", "XOM", "JNJ", "HSBA.L"],
    "portfolio_model": "C",
    "allocation_amount": 10000000,
    "hyperparams": {
      "alpha_1": 0.5,
      "alpha_2": 0.5,
      "alpha_3": 0.01,
      "beta": 0.3,
      "lam": 0.4
    }
  }'
```

The response returns **three independent panels** simultaneously:

```json
{
  "query_id": "uuid",
  "cooperative": {
    "topology": "cooperative",
    "portfolio": [
      {"isin": "AAPL", "sector": "Tech", "weight": 0.40, "allocation": 4000000,
       "return_ann": 0.22, "risk_ann": 0.12, "sharpe": 1.83, "mu_esg": 0.93, "delta_esg": 0.14}
    ],
    "aggregate_metrics": {"portfolio_sharpe": 1.36, "portfolio_mu_esg": 0.72, ...},
    "strategic_summary": "Cooperative mode: shared ESG ambiguity penalty β=0.30 ..."
  },
  "competitive": { ... },
  "mixed": { ... }
}
```

#### Retrieve a previous comparison

```bash
curl http://localhost:8000/api/v1/portfolio/{query_id}/comparison
```

---

## Hyperparameter Reference

| Parameter | Default | Range        | Role                                                 |
| --------- | ------- | ------------ | ---------------------------------------------------- |
| `alpha_1` | `0.5`   | `[0.1, 1.0]` | Bloomberg ESG weight in reward (Portfolios A, C)     |
| `alpha_2` | `0.5`   | `[0.1, 1.0]` | LESG ESG weight in reward (Portfolios A, C)          |
| `alpha_3` | `0.01`  | `≈ 0`        | Financial agent ESG bias (negligible by design)      |
| `beta`    | `0.3`   | `[0.1, 1.0]` | Shared ambiguity penalty strength (Portfolio C only) |
| `lam`     | `0.4`   | `[0.1, 1.0]` | Signed disagreement sensitivity (Portfolio B only)   |

Hyperparameters are selected via grid search on the validation period using **Sharpe Ratio** as the primary metric and **μESG** as a secondary constraint. They are not learned — they encode investor preference before training.

---

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ -v --cov=app --cov-report=term-missing

# Individual test modules
uv run pytest tests/test_normalizer.py -v    # Data normalization (no-leakage checks)
uv run pytest tests/test_masac.py -v         # Actor/Critic networks, replay buffer, update step
uv run pytest tests/test_environment.py -v   # Reward functions, topology β differences
```

Expected output summary:

```
tests/test_normalizer.py   ......   6 passed
tests/test_masac.py        ......   6 passed
tests/test_environment.py  ......   6 passed
```

---

## Project Structure

```
madrl_portfolio/
├── app/
│   ├── main.py                     # FastAPI app factory + lifespan
│   ├── config.py                   # Pydantic Settings (env-driven)
│   ├── agents/
│   │   ├── bloomberg_agent.py      # ADK agent — Bloomberg ESG perspective
│   │   ├── lesg_agent.py           # ADK agent — LESG ESG perspective
│   │   ├── financial_agent.py      # ADK agent — pure financial return
│   │   └── portfolio_orchestrator.py  # Top-level ADK orchestrator
│   ├── rl/
│   │   ├── networks.py             # ActorNetwork, CriticNetwork (PyTorch)
│   │   ├── masac.py                # MASAC algorithm (3 agents, 6 critics)
│   │   ├── replay_buffer.py        # 1M-capacity uniform replay buffer
│   │   ├── environment.py          # MarketEnvironment (all models + topologies)
│   │   └── trainer.py              # Training loop + Redis streaming
│   ├── data/
│   │   ├── pipeline.py             # End-to-end data orchestration
│   │   ├── sources/market.py       # OHLCV fetcher (yfinance / Bloomberg)
│   │   ├── sources/esg.py          # ESG score fetcher (Bloomberg / LESG / stub)
│   │   ├── preprocessing/normalizer.py   # Cross-sectional + time-series normalization
│   │   └── preprocessing/indicators.py  # RSI(14), MACD histogram(12/26/9)
│   ├── api/routes/
│   │   ├── portfolio.py            # POST /portfolio/generate
│   │   ├── training.py             # POST /training/start, GET /training/{id}/status
│   │   ├── data.py                 # POST /data/ingest, GET /data/assets
│   │   └── websocket.py            # WS /ws/training/{id}, WS /ws/portfolio/{id}
│   ├── services/
│   │   ├── portfolio_service.py    # Portfolio generation business logic
│   │   └── training_service.py     # Training job management
│   ├── models/
│   │   ├── domain.py               # SQLAlchemy ORM models
│   │   └── schemas.py              # Pydantic request/response schemas
│   └── workers/tasks.py            # Celery training tasks
├── tests/
│   ├── test_normalizer.py
│   ├── test_masac.py
│   └── test_environment.py
├── ARCHITECTURE.md                 # Full system design with diagrams
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable                | Default                                                           | Description                                  |
| ----------------------- | ----------------------------------------------------------------- | -------------------------------------------- |
| `POSTGRES_DSN`          | `postgresql+asyncpg://madrl:madrl@localhost:5432/madrl_portfolio` | Async PostgreSQL connection string           |
| `REDIS_URL`             | `redis://localhost:6379/0`                                        | Redis for caching and PubSub                 |
| `CELERY_BROKER_URL`     | `redis://localhost:6379/1`                                        | Celery message broker                        |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2`                                        | Celery result storage                        |
| `GOOGLE_API_KEY`        | _(required)_                                                      | Google AI API key for ADK agents             |
| `ADK_MODEL`             | `gemini-2.0-flash`                                                | ADK LLM model for agents                     |
| `BLOOMBERG_API_KEY`     | _(optional)_                                                      | Bloomberg ESG API — uses stub data if empty  |
| `LESG_API_KEY`          | _(optional)_                                                      | LESG API — uses stub data if empty           |
| `MODEL_STORE_PATH`      | `./model_store`                                                   | Directory for trained actor/critic weights   |
| `DEBUG`                 | `false`                                                           | Enables SQLAlchemy query logging             |
| `MASAC_MAX_STEPS`       | `500000`                                                          | Maximum training steps per topology          |
| `MASAC_HIDDEN_SIZE`     | `256`                                                             | Hidden layer width for Actor and Critic MLPs |

---

## Key Design Decisions

**Why three topologies run concurrently?**
The same normalized state vector is fed to all three topologies. The allocation differences emerge entirely from the reward structure — specifically how the shared ambiguity penalty `β · ΔESGₜ` is applied (full / zero / partial). Running them in parallel lets users observe the direct impact of game-theoretic framing on portfolio construction.

**Why no tanh on actor output?**
Portfolio weights are produced by Softmax over the joint score vector `z_joint`. Softmax accepts unbounded real inputs directly, and tanh squashing would distort the score magnitudes without providing any benefit for the sum-to-one constraint.

**Why cross-sectional normalization for ESG but time-series for OHLCV?**
ESG scores from different agencies (Bloomberg 0–100, LESG 0–10) must be harmonized to a comparable scale _before_ computing `ΔESGᵢₜ` and `μESGᵢₜ`. Cross-sectional normalization (same-day peer ranking) achieves this with zero temporal look-ahead. OHLCV/RSI/MACD are normalized per-asset over the training window and frozen before the test window to prevent data leakage.

---

## Stopping and Cleanup

```bash
# Stop all containers (preserve data volumes)
docker compose down

# Stop and remove all data volumes (full reset)
docker compose down -v

# Stop a running training job
curl -X POST http://localhost:8000/api/v1/training/{job_id}/stop
```
