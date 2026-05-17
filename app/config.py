from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application
    app_name: str = "MADRL Portfolio System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    postgres_dsn: str = "postgresql+asyncpg://madrl:madrl@localhost:5432/madrl_portfolio"
    postgres_pool_size: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_market: int = 3600       # 1h for OHLCV
    redis_ttl_esg: int = 86400         # 24h for ESG scores
    redis_ttl_state: int = 21600       # 6h for normalized state
    redis_ttl_session: int = 3600      # 1h for sessions

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Market Data APIs
    bloomberg_api_key: str = ""
    lesg_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # ADK
    adk_model: str = "gemini-2.0-flash"
    google_api_key: str = ""

    # Model storage
    model_store_path: str = "./model_store"

    # MASAC default hyperparameters
    masac_gamma: float = 0.99
    masac_tau: float = 0.005
    masac_lr_actor: float = 3e-4
    masac_lr_critic: float = 3e-4
    masac_lr_alpha: float = 3e-4
    masac_batch_size: int = 256
    masac_buffer_capacity: int = 1_000_000
    masac_warmup_steps: int = 10_000
    masac_max_steps: int = 500_000
    masac_episode_length: int = 252
    masac_convergence_epsilon: float = 0.01
    masac_convergence_window: int = 100
    masac_initial_alpha_t: float = 1.0
    masac_hidden_size: int = 256

    # Technical indicator parameters
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # warmup = cfg.macd_slow (computed inline where needed — not hardcoded here)

    # Validation window
    validation_window_days: int = 63   # 1 calendar quarter


@lru_cache
def get_settings() -> Settings:
    return Settings()
