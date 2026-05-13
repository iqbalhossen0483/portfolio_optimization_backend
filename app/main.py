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
from app.api.routes import portfolio, training, data, websocket

log = structlog.get_logger(__name__)
cfg = get_settings()


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
        title=cfg.app_name,
        version=cfg.app_version,
        description=(
            "Multi-Agent Deep Reinforcement Learning portfolio optimization system. "
            "Runs three ESG-aware MASAC agents under Cooperative, Competitive, and Mixed "
            "game-theoretic topologies to generate side-by-side portfolio comparisons."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics
    Instrumentator().instrument(app).expose(app)

    # Routers
    api_prefix = "/api/v1"
    app.include_router(portfolio.router, prefix=api_prefix)
    app.include_router(training.router,  prefix=api_prefix)
    app.include_router(data.router,      prefix=api_prefix)
    app.include_router(websocket.router)

    @app.get("/health", tags=["system"])
    async def health():
        return {"status": "ok", "version": cfg.app_version}

    return app


app = create_app()
