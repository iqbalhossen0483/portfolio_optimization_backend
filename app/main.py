"""
FastAPI application factory.
Lifespan: creates DB tables on startup; closes Redis pool on shutdown.
"""
from __future__ import annotations
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.core.database import create_tables
from app.api.routes import training, data, websocket, chat, auth, admin

log = structlog.get_logger(__name__)
cfg = get_settings()

# ── Tag metadata — each group gets a description in Swagger UI ────────────────
TAGS_METADATA = [
    {
        "name": "training",
        "description": (
            "**MASAC training jobs.**\n\n"
            "Upload one or more `.xlsx` files containing historical OHLCV + ESG data. "
            "The system runs a 4-stage pipeline:\n\n"
            "1. **Stage 1** — Parse XLSX → compute `return_pct` + `macd_hist` from Close "
            "→ bulk-upsert to `assets`, `market_data`, `esg_scores`\n"
            "2. **Stage 2** — Cross-sectional ESG normalisation per date across all N ISINs "
            "→ store `esg_b_norm`, `esg_l_norm`, `delta_esg`, `mu_esg` in `esg_scores`\n"
            "3. **Stage 3** — Fit time-series min/max normaliser on training window "
            "→ store frozen params in `training_normalizer_params` (8 × N rows)\n"
            "4. **Stage 4** — MASAC training reads ALL pre-computed features from DB via "
            "`DatabaseMarketDataSource` + `DatabaseESGDataSource` (no external API calls). "
            "Runs asynchronously in Celery."
        ),
    },
    {
        "name": "data",
        "description": (
            "**Asset and data management.**\n\n"
            "Query ingested assets. Assets are populated via `POST /training/start` with XLSX files."
        ),
    },
    {
        "name": "websocket",
        "description": (
            "**Real-time WebSocket streams.**\n\n"
            "- `WS /ws/training/{job_id}` — live MASAC step metrics via Redis PubSub"
        ),
    },
    {
        "name": "chat",
        "description": (
            "**Natural language portfolio advisor.**\n\n"
            "Send plain-English queries to the Google ADK agent. "
            "It parses your intent, runs MASAC inference across all three topologies, "
            "and returns three side-by-side portfolio panels with per-asset weights and allocations."
        ),
    },
    {
        "name": "auth",
        "description": (
            "**Authentication & user management.**\n\n"
            "- `POST /auth/register` — create an account (role: `user` by default)\n"
            "- `POST /auth/login` — obtain a JWT Bearer token\n"
            "- `GET /auth/me` — get your own profile\n"
            "- `PUT /auth/me` — update email, username, or password\n"
            "- `GET /auth/users` — *(admin)* list all users\n"
            "- `PUT /auth/users/{id}/role` — *(admin)* promote/demote a user"
        ),
    },
    {
        "name": "admin",
        "description": (
            "**Admin dashboard.**\n\n"
            "- `GET /admin/dashboard` — asset, job, and user counts; training status\n"
            "- `GET /admin/assets` — keyset-paginated asset list with market-data count"
        ),
    },
    {
        "name": "system",
        "description": "Health check and system status.",
    },
]

_DESCRIPTION = """\
## Multi-Agent Deep Reinforcement Learning Portfolio Optimisation

Resolves ESG rating disagreement between **Bloomberg** (0–100) and **LESG** (0–10) through
a three-agent **MASAC** framework. Three game-theoretic topologies run concurrently and
return side-by-side portfolio panels for direct comparison.

### Key concepts

| Term | Meaning |
|---|---|
| **MASAC** | Multi-Agent Soft Actor-Critic — 3 actors, 6 critics (2 per agent), shared replay buffer |
| **CTDE** | Centralised Training, Decentralised Execution — critics see all agents, actors see local obs only |
| **Portfolio A** | ESG consensus — Bloomberg + LESG weighted average |
| **Portfolio B** | Signed disagreement — each agent bets its own ESG source is correct |
| **Portfolio C** | Full model — consensus + uncertainty penalty β · ΔESGᵢₜ (recommended) |
| **State vector** | 10N features per timestep: [OHLCV(5N) \\| RSI(N) \\| MACD(N) \\| Return(N) \\| ΔESG(N) \\| μESG(N)] |
| **N** | Number of assets — always dynamic, derived from uploaded XLSX, never hardcoded |

### Data pipeline stages

```
POST /training/start  (multipart: .xlsx files + form fields)
     │
     ├─ Stage 1: XLSX → market_data + esg_scores (return_pct & macd_hist computed)
     ├─ Stage 2: Cross-sectional ESG norm → esg_scores (esg_b_norm, delta_esg, mu_esg)
     ├─ Stage 3: Time-series normaliser fit → training_normalizer_params (8 × N rows)
     └─ Stage 4: MASAC training (Celery) — pure DB reads, no external APIs
```

### XLSX file format  *(sheet name: `Stock_ESG_Dataset`)*

| Column | Type | Notes |
|---|---|---|
| Date | date | Trading date |
| ISIN | string | Asset identifier — N is derived from unique ISINs |
| Company name | string | — |
| Sector | string | — |
| Open / High / Low / Close | float | Raw OHLCV — stored as-is |
| Volume | string / float | Accepts `10.5M`, `2.3K`, `1B`, `1T` or plain number |
| RSI | float | RSI value |
| Bloom. ESG (0-100) | float | Bloomberg ESG score |
| LESG ESG (0-10) | float | LESG ESG score |
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting MADRL Portfolio System", version=cfg.app_version)
    await create_tables()
    log.info("Database tables created/verified")
    yield
    log.info("Shutting down")
    from app.api.deps import _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MADRL Portfolio System",
        version=cfg.app_version,
        description=_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        contact={
            "name": "MADRL Portfolio",
            "email": "uni.soton.uk@gmail.com",
        },
        license_info={
            "name": "Proprietary",
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    Instrumentator().instrument(app).expose(app)

    api_prefix = "/api/v1"
    app.include_router(auth.router,      prefix=api_prefix)
    app.include_router(training.router,  prefix=api_prefix)
    app.include_router(data.router,      prefix=api_prefix)
    app.include_router(chat.router,      prefix=api_prefix)
    app.include_router(admin.router,     prefix=api_prefix)
    app.include_router(websocket.router)

    @app.get(
        "/health",
        tags=["system"],
        summary="System health check",
        description="Returns `ok` when the API process is running. Does not probe DB or Redis.",
        response_description="Service is up",
    )
    async def health():
        return {"status": "ok", "version": cfg.app_version}

    return app


app = create_app()
