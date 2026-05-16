# MADRL Portfolio System

Multi-Agent Deep Reinforcement Learning portfolio optimisation that resolves ESG rating disagreement between Bloomberg and LESG through a three-agent MASAC framework. Three game-theoretic topologies — Cooperative, Competitive, and Mixed — run concurrently and return side-by-side portfolio panels for direct comparison.

**Stack:** FastAPI · PyTorch MASAC · PostgreSQL · Redis · Celery · Google ADK · Docker

---

## Table of Contents

1. [Architecture](#architecture)
2. [Prerequisites](#prerequisites)
3. [Getting Started — Production](#getting-started--production)
4. [Getting Started — Development](#getting-started--development)
5. [Database Migrations](#database-migrations)
6. [API Usage](#api-usage)
   - [Training Workflow](#training-workflow)
   - [Monitor Training](#monitor-training)
   - [Generate Portfolio](#generate-portfolio)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Design Decisions](#design-decisions)
10. [Testing](#testing)
11. [Operations](#operations)

---

## Architecture

```
POST /training/start (.xlsx files)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1  Parse XLSX → compute return_pct + macd_hist       │
│           → bulk upsert: assets, market_data, esg_scores    │
├─────────────────────────────────────────────────────────────┤
│  Stage 2  Cross-sectional ESG normalisation per date        │
│           esg_b_norm, esg_l_norm, delta_esg, mu_esg         │
│           → update esg_scores                               │
├─────────────────────────────────────────────────────────────┤
│  Stage 3  Fit time-series normaliser on training window     │
│           → insert training_normalizer_params (8 × N rows)  │
└─────────────────────────────────────────────────────────────┘
         │
         └── Celery ──► Stage 4: MASAC Training
                              │
                    ┌─────────┼─────────┐
               Cooperative Competitive Mixed
                    │         │         │
               Bloomberg   LESG    Financial
               ESGAgent  ESGAgent   Agent
                    │         │         │
                    └────┬────┘─────────┘
                    z_joint = avg(z^B, z^L, z^F)
                    Softmax → portfolio weights
```

**State vector per timestep:** `10N` features — `[OHLCV(5N) | RSI(N) | MACD(N) | Return(N) | ΔESG(N) | μESG(N)]`
**N** (number of assets) and **T** (timesteps) are always dynamic — derived from the uploaded XLSX, never hardcoded.

| Topology | β penalty | Behaviour |
|---|---|---|
| Cooperative | full β | Penalises ESG-ambiguous assets — conservative, ESG-aligned |
| Competitive | β = 0 | Each agent maximises its own ESG source — divergent weights |
| Mixed | partial β | Balanced between cooperation and competition |

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker & Docker Compose | Docker 24+ |
| Python | 3.11+ (dev only) |
| Git | any |
| NVIDIA GPU | optional — CUDA 12.x for faster training |

---

## Getting Started — Production

### 1. Clone and configure

```bash
git clone <repo-url> madrl_portfolio
cd madrl_portfolio

cp .env.example .env.docker
```

Edit `.env.docker` — minimum required:

```env
POSTGRES_DSN=postgresql+asyncpg://<user>:<pass>@host.docker.internal:5432/madrl_portfolio
REDIS_PASSWORD=your_redis_password
GOOGLE_API_KEY=your_google_api_key
```

### 2. Build and start

```bash
docker compose up --build -d
```

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Flower (Celery monitor) | http://localhost:5555 |
| Prometheus | http://localhost:9090 |

### 3. Apply database migrations

```bash
# Run inside the api container
docker compose exec api alembic upgrade head
```

### 4. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","version":"1.0.0"}
```

---

## Getting Started — Development

Development mode mounts `./app` as a live volume — **no image rebuild after code changes**. Uvicorn reloads automatically on every `.py` save. `debugpy` is available on port `5678` for IDE attachment.

Two files handle this:

| File | Purpose |
|---|---|
| `Dockerfile.development` | Dev image: BuildKit pip cache, `watchfiles`, `debugpy` |
| `docker-compose.dev.yml` | Overrides: live volume, `--reload`, debug port, no resource limits |

### 1. Start dev stack

```bash
# First time — builds the dev image
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# All subsequent starts — no rebuild needed
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

| Service | URL |
|---|---|
| API + Swagger | http://localhost:8000/docs |
| Flower | http://localhost:5555 |
| Prometheus | http://localhost:9090 |
| debugpy (IDE attach) | localhost:5678 |

### 2. Apply migrations

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec api alembic upgrade head
```

### 3. Live reload

```
Save any file in app/  →  watchfiles detects change  →  uvicorn reloads  →  new code live
```

> **Windows / Docker Desktop:** `WATCHFILES_FORCE_POLLING=true` is pre-set in `Dockerfile.development`. WSL2 bind mounts do not reliably deliver inotify events — polling ensures reload always works.

### Rebuild reference

| What changed | Action |
|---|---|
| Any `app/` file | Nothing — volume mount, instant |
| `requirements.txt` | `docker compose … up --build` — pip cache on host makes it fast |
| `Dockerfile.development` | `docker compose … up --build` |
| Base image / system deps | `docker compose … up --build --no-cache` |

### VS Code remote debugger

Add to `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach",
      "type": "debugpy",
      "request": "attach",
      "connect": { "host": "localhost", "port": 5678 },
      "pathMappings": [
        { "localRoot": "${workspaceFolder}/app", "remoteRoot": "/app/app" }
      ]
    }
  ]
}
```

Set a breakpoint → press **F5**. Attaches to the running container without restarting it.

---

## Database Migrations

Migrations are managed with **Alembic**. The `alembic/env.py` reads `POSTGRES_DSN` from `app/config.py` directly — no URL in `alembic.ini`. Alembic runs in async mode (asyncpg).

### Common commands

```bash
# Apply all pending migrations (run after every pull)
alembic upgrade head

# Generate a new migration after changing app/models/domain.py
alembic revision --autogenerate -m "describe what changed"

# Roll back one migration
alembic downgrade -1

# Roll back everything (empty schema)
alembic downgrade base

# Check current revision
alembic current

# View full history
alembic history
```

> PostgreSQL must be running before any migration command.

---

## API Usage

Full interactive docs at **http://localhost:8000/docs**.

### Training Workflow

Training is a 4-stage pipeline triggered by a single multipart `POST`. Stages 1–3 run synchronously; Stage 4 (MASAC) runs in Celery and returns immediately with a `job_id`.

#### Upload XLSX and start training

```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -F "files=@stock_esg_dataset.xlsx" \
  -F "portfolio_model=C" \
  -F "topology=all" \
  -F "train_start=2018-01-01" \
  -F "train_end=2022-12-31" \
  -F "val_start=2023-01-01" \
  -F "val_end=2023-12-31" \
  -F 'hyperparams_json={"alpha_1":0.5,"alpha_2":0.5,"alpha_3":0.01,"beta":0.3,"lam":0.4}'
```

```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "queued",
  "message": "Stages 1-3 complete. MASAC training queued."
}
```

**Upload multiple files** — different ISINs or date ranges are merged and deduplicated on `(ISIN, Date)`:

```bash
curl -X POST http://localhost:8000/api/v1/training/start \
  -F "files=@dataset_2018_2020.xlsx" \
  -F "files=@dataset_2021_2024.xlsx" \
  -F "portfolio_model=C" \
  -F "topology=all"
  # omit dates → auto-split 80/20 from file contents
```

**XLSX format** — sheet name must be `Stock_ESG_Dataset`:

| Column | Type | Notes |
|---|---|---|
| `Date` | date | Trading date |
| `ISIN` | string | Asset identifier — N derived from distinct ISINs |
| `Company name` | string | |
| `Sector` | string | |
| `Open`, `High`, `Low`, `Close` | float | Raw OHLCV — stored as-is |
| `Volume` | string / float | Accepts `10.5M`, `2.3K`, `1B`, `1T` or plain number |
| `RSI` | float | RSI value |
| `Bloom. ESG (0-100)` | float | Bloomberg ESG score |
| `LESG ESG (0-10)` | float | LESG ESG score |

**Portfolio models:**

| Model | Reward structure |
|---|---|
| `A` | ESG consensus: `α₁·ESG_B_norm + α₂·ESG_L_norm + financial_return` |
| `B` | Signed disagreement: each agent bets its own ESG source is correct |
| `C` | Full model: consensus + `β·ΔESGᵢₜ` uncertainty penalty **(recommended)** |

---

### Monitor Training

#### Poll status

```bash
curl http://localhost:8000/api/v1/training/{job_id}/status
```

```json
{
  "job_id": "3fa85f64-...",
  "status": "running",
  "step": 45000,
  "max_steps": 500000,
  "progress_pct": 9.0,
  "best_sharpe": 1.31,
  "best_mu_esg": 0.68,
  "elapsed_seconds": 183.4
}
```

**Status values:** `queued` → `running` → `completed` / `failed` / `stopped`

#### Stream via WebSocket

```bash
# npm install -g wscat
wscat -c ws://localhost:8000/ws/training/{job_id}
```

```json
{"type":"step","step":12500,"entropy":2.8,"entropy_rolling_std":0.045,"reward_bloomberg":0.018,"reward_lesg":0.014,"reward_financial":0.021}
{"type":"converged","step":234100,"final_sharpe":1.44,"mu_esg":0.71}
```

Training stops when the 100-step rolling std of mean policy entropy drops below **0.01**, or at **500,000 steps** maximum.

#### Stop a running job

```bash
curl -X POST http://localhost:8000/api/v1/training/{job_id}/stop
```

The worker finishes the current topology cleanly before exiting.

---

### Generate Portfolio

```bash
curl -X POST http://localhost:8000/api/v1/portfolio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "assets": ["GB0002875804", "US0378331005", "US30231G1022", "GB0005405286"],
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

Returns three independent panels simultaneously:

```json
{
  "query_id": "uuid",
  "cooperative": {
    "topology": "cooperative",
    "portfolio": [
      {
        "isin": "GB0002875804",
        "sector": "Financials",
        "weight": 0.40,
        "allocation": 4000000,
        "return_ann": 0.22,
        "risk_ann": 0.12,
        "sharpe": 1.83,
        "mu_esg": 0.93,
        "delta_esg": 0.14
      }
    ],
    "aggregate_metrics": {
      "portfolio_sharpe": 1.36,
      "portfolio_mu_esg": 0.72,
      "portfolio_delta_esg": 0.11,
      "portfolio_return": 0.19,
      "portfolio_risk": 0.14
    },
    "strategic_summary": "Cooperative mode: shared ESG ambiguity penalty β=0.30 ..."
  },
  "competitive": { "..." },
  "mixed": { "..." }
}
```

Retrieve a stored comparison:

```bash
curl http://localhost:8000/api/v1/portfolio/{query_id}/comparison
```

---

## Configuration

### Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `POSTGRES_DSN` | yes | — | Async PostgreSQL DSN (`postgresql+asyncpg://...`) |
| `REDIS_URL` | yes | — | Redis URL for cache and PubSub |
| `REDIS_PASSWORD` | yes | — | Redis auth password |
| `CELERY_BROKER_URL` | yes | — | Celery broker (Redis DB 1) |
| `CELERY_RESULT_BACKEND` | yes | — | Celery results (Redis DB 2) |
| `GOOGLE_API_KEY` | yes | — | Google AI key for ADK agents |
| `ADK_MODEL` | no | `gemini-2.0-flash` | ADK LLM model |
| `BLOOMBERG_API_KEY` | no | `""` | Bloomberg ESG API — stub data if empty |
| `LESG_API_KEY` | no | `""` | LESG API — stub data if empty |
| `MODEL_STORE_PATH` | no | `./model_store` | Directory for trained actor/critic weights |
| `DEBUG` | no | `false` | Enables SQLAlchemy query logging |
| `MASAC_MAX_STEPS` | no | `500000` | Maximum training steps per topology |
| `MASAC_BATCH_SIZE` | no | `256` | Replay buffer sample size |
| `MASAC_HIDDEN_SIZE` | no | `256` | Hidden layer width for Actor and Critic MLPs |
| `MASAC_LR_ACTOR` | no | `3e-4` | Actor learning rate |
| `MASAC_LR_CRITIC` | no | `3e-4` | Critic learning rate |

### Hyperparameters

Hyperparameters are **not learned** — they encode investor preference before training. Selected via grid search on the validation window using Sharpe as primary metric and μESG as secondary.

| Parameter | Default | Range | Role |
|---|---|---|---|
| `alpha_1` | `0.5` | `[0.0, 1.0]` | Bloomberg ESG weight in reward (Portfolios A, C) |
| `alpha_2` | `0.5` | `[0.0, 1.0]` | LESG ESG weight in reward (Portfolios A, C) |
| `alpha_3` | `0.01` | `[0.0, 0.1]` | Financial agent ESG bias — kept near 0 by design |
| `beta` | `0.3` | `[0.0, 1.0]` | Ambiguity penalty strength `β·ΔESGᵢₜ` (Portfolio C, Cooperative topology) |
| `lam` | `0.4` | `[0.0, 1.0]` | Signed disagreement sensitivity (Portfolio B) |

---

## Project Structure

```
madrl_portfolio/
├── app/
│   ├── main.py                        # FastAPI factory, lifespan, middleware, routers
│   ├── config.py                      # Pydantic Settings — all env-var bindings
│   │
│   ├── api/
│   │   ├── deps.py                    # DB session, Redis, service dependencies
│   │   └── routes/
│   │       ├── portfolio.py           # POST /portfolio/generate, GET /portfolio/{id}/comparison
│   │       ├── training.py            # POST /training/start (multipart), GET/POST /training/{id}/*
│   │       ├── data.py                # GET /data/assets, POST /data/ingest
│   │       └── websocket.py           # WS /ws/training/{id}, WS /ws/portfolio/{id}
│   │
│   ├── services/
│   │   ├── training_service.py        # 4-stage XLSX pipeline + Celery job lifecycle
│   │   └── portfolio_service.py       # Portfolio generation orchestration
│   │
│   ├── agents/                        # Google ADK agents
│   │   ├── base.py
│   │   ├── bloomberg_agent.py         # Bloomberg ESG perspective
│   │   ├── lesg_agent.py              # LESG ESG perspective
│   │   ├── financial_agent.py         # Pure financial return
│   │   └── portfolio_orchestrator.py  # Coordinates all three
│   │
│   ├── rl/                            # PyTorch MASAC engine
│   │   ├── masac.py                   # 3 actors, 6 critics, shared replay buffer
│   │   ├── networks.py                # ActorNetwork, CriticNetwork
│   │   ├── environment.py             # MarketEnvironment (all models + topologies)
│   │   ├── replay_buffer.py           # 1M-capacity uniform buffer
│   │   └── trainer.py                 # Training loop + Redis PubSub streaming
│   │
│   ├── data/
│   │   ├── pipeline.py                # Fetch → preprocess → assemble state tensors
│   │   ├── sources/
│   │   │   ├── xlsx.py                # Stage 1 — XLSX parser, return_pct + macd_hist computation
│   │   │   ├── database.py            # Stage 4 — DB-backed sources, pre-computed feature reads
│   │   │   ├── market.py              # Legacy — yfinance OHLCV fetcher
│   │   │   └── esg.py                 # Legacy — Bloomberg / LESG / stub ESG fetcher
│   │   └── preprocessing/
│   │       ├── normalizer.py          # Cross-sectional (ESG) + time-series (OHLCV) normalisation
│   │       └── indicators.py          # RSI, MACD histogram computation
│   │
│   ├── models/
│   │   ├── domain.py                  # SQLAlchemy ORM — assets, market_data, esg_scores,
│   │   │                              #   training_jobs, training_normalizer_params, ...
│   │   └── schemas.py                 # Pydantic request/response schemas with examples
│   │
│   ├── core/
│   │   └── database.py                # Async engine, session factory, create_tables
│   │
│   └── workers/
│       └── tasks.py                   # Celery training task — DB path + legacy yfinance path
│
├── alembic/
│   ├── env.py                         # Async Alembic env (reads POSTGRES_DSN from settings)
│   └── versions/                      # Auto-generated migration scripts
│
├── tests/
│   ├── test_normalizer.py
│   ├── test_masac.py
│   └── test_environment.py
│
├── alembic.ini
├── docker-compose.yml                 # Production
├── docker-compose.dev.yml             # Dev overrides (live reload, debugpy, no limits)
├── Dockerfile                         # Production image
├── Dockerfile.development             # Dev image (BuildKit cache, watchfiles, debugpy)
├── requirements.txt
├── pyproject.toml
└── .env.example
```

### Layer boundaries

| Layer | Folder | Owns | Never touches |
|---|---|---|---|
| HTTP | `api/routes/` | URL paths, status codes, serialisation | Business logic |
| Business logic | `services/` | Orchestration, business rules | HTTP, raw SQL |
| Agents | `agents/` | Per-agent ESG/financial decisions | Training loop, HTTP |
| RL engine | `rl/` | MASAC algorithm, neural nets, environment | Agent coordination |
| Data pipeline | `data/` | Fetching, preprocessing, feature assembly | Portfolio decisions |
| Data contracts | `models/` | DB shapes, API schema validation | Logic of any kind |
| Infrastructure | `core/` | Connection pool, session management | Application logic |
| Background jobs | `workers/` | Long-running async Celery tasks | Synchronous handling |

---

## Design Decisions

**Why three topologies run concurrently?**
The same normalised state vector feeds all three topologies. Allocation differences emerge purely from how the ambiguity penalty `β·ΔESGₜ` is applied — full, zero, or partial. Running them in parallel makes the game-theoretic effect directly observable without confounding variables.

**Why cross-sectional normalisation for ESG but time-series for OHLCV?**
Bloomberg (0–100) and LESG (0–10) are on incompatible scales. Cross-sectional min-max per trading day harmonises them with zero temporal look-ahead — only same-day peer values are used. OHLCV/RSI/MACD are normalised per-asset over the training window and frozen before the validation window, preventing data leakage.

**Why no tanh on actor output?**
Portfolio weights are produced by Softmax over `z_joint`. Softmax accepts unbounded real inputs and enforces the sum-to-one constraint natively. tanh squashing would distort score magnitudes with no benefit.

**Why staged DB persistence?**
Stage 2 (ESG normalisation) depends on Stage 1 being complete across all N assets — you cannot normalise cross-sectionally until every asset's score for that date is in the DB. Stage 3 (normaliser fit) depends on Stage 2. Staged persistence makes each dependency explicit and recoverable.

---

## Testing

```bash
# Activate venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate         # macOS / Linux

# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Individual suites
pytest tests/test_normalizer.py -v    # No-leakage normalisation checks
pytest tests/test_masac.py -v         # Actor/Critic networks, replay buffer, update step
pytest tests/test_environment.py -v   # Reward functions, topology β differences
```

---

## Operations

```bash
# View logs
docker compose logs -f api
docker compose logs -f worker

# Restart a single service (e.g. after a config change)
docker compose restart api

# Stop — preserve volumes (data intact)
docker compose down

# Full reset — destroy all data
docker compose down -v

# Rebuild after code or requirements change (production)
docker compose up --build -d

# Switch from dev back to production
docker compose -f docker-compose.yml up --build -d
```
