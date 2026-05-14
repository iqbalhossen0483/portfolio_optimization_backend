# Plan: Full Staged XLSX Ingestion → PostgreSQL + DB-Driven Training

## Context
The system currently uses yfinance (prototype) for market data and a synthetic stub for ESG scores.
The real app must:
1. Accept `.xlsx` files at `POST /training/start`
2. Apply ALL equations from requirements.md in dependency order (staged)
3. Persist EVERY stage to PostgreSQL ("every operation must be sync with database")
4. Train MASAC from the fully pre-processed DB data — NO external API calls for training data

**Scale**: Production dataset = up to 10 years × ~252 trading days × N assets.
N is **never fixed or hardcoded** — it is derived at runtime from whatever ISINs are present in the uploaded XLSX files. T is likewise dynamic. All computations must be vectorized (pandas/numpy) over the actual (N, T) shape — no loops over individual assets or dates for bulk operations.

**XLSX confirmed format** (sheet: `Stock_ESG_Dataset`):
```
Date | ISIN | Company name | Sector | Open | High | Low | Close | Volume | RSI | Bloom. ESG (0-100) | LESG ESG (0-10)
```
Volume stored as string with M-suffix: `"10.5M"` → must be parsed to float.

**What is raw data (read from XLSX — no recomputation):**
- OHLCV (Open, High, Low, Close, Volume)
- RSI — taken directly from the XLSX column as-is
- Bloomberg ESG (0-100)
- LESG ESG (0-10)

**What must be computed (not in XLSX — derived from Close):**
- `return_pct`: individual return `R_{i,t} = (Close_t - Close_{t-1}) / Close_{t-1}`
- `macd_hist`: MACD histogram from Close using params `(cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)`

**What must NOT be computed from scratch:** RSI is already provided by the data source. We only normalize it.

---

## Processing Axis — Per-ISIN vs. Cross-Asset

All data processing is keyed by ISIN. One ISIN can have thousands of rows (e.g. 10 years × 252 trading days = ~2,520 rows). The axis of each operation:

| Operation | Axis | Why |
|---|---|---|
| Returns `R_{i,t}` | **Per ISIN** — grouped by ISIN, sorted by date | Consecutive rows of same asset |
| MACD histogram | **Per ISIN** — EMA computed over each ISIN's own Close series | Each asset has independent price history |
| RSI normalization (min-max) | **Per ISIN** — min/max of RSI over training window for THIS ISIN | Asset-specific range |
| OHLCV normalization (min-max) | **Per ISIN** — min/max over training window for THIS ISIN | Asset-specific price scale |
| Sharpe ratio, risk σ | **Per ISIN** — annualized return/std computed from each asset's return series | Per-asset risk metrics |
| ESG normalization | **Cross-sectional (per date, across all N ISINs)** | Must compare all N assets on same day |
| ΔeSG, μeSG | **Per ISIN per date** — derived from the cross-sectional ESG norms | Per-asset values after cross-sectional step |

**All `groupby("isin")` operations are vectorized** — pandas processes all N ISINs in one pass, not a Python loop.

---

## Dependency-Ordered Processing Stages

Requirements.md mandates a strict calculation order (some values depend on earlier ones):

```
Stage 1 — Parse & store raw data (independent)
  XLSX → assets, market_data(OHLCV + RSI from file), esg_scores(raw Bloomberg + LESG)
  + compute from Close: return_pct, MACD histogram(cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
  RSI is NOT computed — taken directly from XLSX RSI column
  → stored in: market_data (existing OHLCV + new columns: rsi, return_pct, macd_hist)

Stage 2 — Cross-sectional ESG normalization (independent of training window)
  ESG_B_norm(i,t) = (ESG_B(i,t) - min_i ESG_B(t)) / (max_i ESG_B(t) - min_i ESG_B(t))
  ESG_L_norm(i,t) = same formula for LESG
  delta_esg(i,t)  = |ESG_B_norm - ESG_L_norm|   ← requires stage 2 complete first
  mu_esg(i,t)     = (ESG_B_norm + ESG_L_norm) / 2
  → stored in: esg_scores (new columns)

Stage 3 — Time-series normalization params (per training job, training window only)
  For each asset i, for each feature f over training window W:
    min_f(i) = min_{t in W} feature_f(i,t)
    max_f(i) = max_{t in W} feature_f(i,t)
  Features normalized (8 total per asset):
    open, high, low, close, volume  ← from XLSX raw
    return_pct                       ← computed from Close
    rsi                              ← from XLSX raw; normalized here (NOT recomputed)
    macd_hist                        ← computed from Close; normalized here
  These params are FROZEN after fit — reused for val window (no look-ahead)
  → stored in: training_normalizer_params (new table, linked to job_id)

Stage 4 — MASAC training (reads ALL stages from DB)
  Reads pre-computed features from market_data + esg_scores
  Applies frozen normalizer params from training_normalizer_params
  Assembles 10N state vectors in memory: [OHLCV(5N)|RSI(N)|MACD(N)|Return(N)|ΔESG(N)|μESG(N)]
  No API calls, no recomputation — pure DB reads + in-memory normalization application
```

---

## Database Schema Changes

### Modify `market_data` (add 3 columns):
```python
return_pct:  Mapped[float | None] = mapped_column(Float, nullable=True)  # computed: (Close_t - Close_{t-1}) / Close_{t-1}; NULL for first row per ISIN
rsi:         Mapped[float | None] = mapped_column(Float, nullable=True)  # raw from XLSX — NOT computed; taken as-is from data provider
macd_hist:   Mapped[float | None] = mapped_column(Float, nullable=True)  # computed from Close using cfg params; NULL for first cfg.macd_slow-1 rows
```
_NULLs only for MACD warmup rows (first `cfg.macd_slow - 1` rows per ISIN). RSI has no NULLs — it comes from the XLSX already computed._

**Fix required in `config.py`**: remove the hardcoded `macd_warmup_days: int = 26` field. Replace all usages with the dynamic expression:

```python
effective_warmup = cfg.macd_slow  # only MACD drives the warmup; RSI is from raw data
```

Changing `macd_slow` in `.env` automatically adjusts the warmup — no code changes needed.

### Modify `esg_scores` (add 4 columns):
```python
esg_b_norm:  Mapped[float | None] = mapped_column(Float, nullable=True)  # cross-sectional [0,1]
esg_l_norm:  Mapped[float | None] = mapped_column(Float, nullable=True)  # cross-sectional [0,1]
delta_esg:   Mapped[float | None] = mapped_column(Float, nullable=True)  # |ESG_B_norm - ESG_L_norm|
mu_esg:      Mapped[float | None] = mapped_column(Float, nullable=True)  # (ESG_B_norm + ESG_L_norm) / 2
```

### New table `training_normalizer_params`:
```python
class TrainingNormalizerParams(Base):
    __tablename__ = "training_normalizer_params"
    __table_args__ = (UniqueConstraint("job_id", "isin", "feature_name"),)

    id:           Mapped[str]   # UUID PK
    job_id:       Mapped[str]   # FK → training_jobs.id
    isin:         Mapped[str]   # which asset (N assets per job — N is dynamic)
    feature_name: Mapped[str]   # "open"|"high"|"low"|"close"|"volume"|"return_pct"|"rsi"|"macd_hist"
    min_val:      Mapped[float]
    max_val:      Mapped[float]
```
_Stores 8 × N rows per training job. N is dynamic — derived from XLSX contents, not hardcoded._

---

## Files to Create

### 1. `app/data/sources/xlsx.py` — NEW
`XLSXDataSource` — pure data extraction and indicator computation. No DB.

```python
COLUMN_MAP = {
    "Date": "date", "ISIN": "isin", "Company name": "name",
    "Sector": "sector", "Open": "open", "High": "high",
    "Low": "low", "Close": "close", "Volume": "volume",
    "Bloom. ESG (0-100)": "bloomberg_score",
    "LESG ESG (0-10)": "lesg_score",
    # "RSI" column intentionally excluded — ignored, recomputed from Close
}
```

**`parse_files(paths: list[str])`** → `ParsedXLSX` dataclass:
- Reads each `.xlsx` file with `pd.read_excel(path, sheet_name="Stock_ESG_Dataset")`
- Concatenates and **deduplicates on `(ISIN, Date)`** — handles multiple overlapping files
- Parses Volume — handles ALL suffix variants (case-insensitive, optional spaces):
  - `K` or `k` → × 1,000
  - `M` or `m` → × 1,000,000
  - `B` or `b` → × 1,000,000,000
  - `T` or `t` → × 1,000,000,000,000
  - Numeric already (`float`/`int`) → passthrough
  - No suffix → parse as plain float
  - Fallback: `NaN` (never raise on bad data — log a warning)
  - Implementation: `str(v).strip().upper()` → check suffix → strip → `float(body) * multiplier`
- Parses Date → `pd.Timestamp`
- **RSI**: read directly from the `RSI` column in XLSX — NO computation. Store as-is.
- **Vectorized computation for derived features** (only two things need to be computed from Close):
  - `return_pct` = `df.groupby("isin")["close"].pct_change()` — shape `(T×N,)`; NaN for first row per ISIN
  - `macd_hist` = `df.groupby("isin")["close"].apply(lambda s: ta.trend.MACD(s, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal).macd_diff())` — NaN for first `cfg.macd_slow - 1` rows per ISIN; params from config, never hardcoded
- All N assets processed in one pass over the long-format DataFrame (not looped)
- Returns:
  - `assets`: `list[dict(isin, name, sector)]` — deduplicated, N entries
  - `market_df`: `DataFrame[isin, date, open, high, low, close, volume, return_pct, rsi, macd_hist]` — shape `(T×N, 10)`
  - `esg_df`: `DataFrame[isin, date, bloomberg_score, lesg_score]` — shape `(T×N, 4)`
  - `n_assets`: int (N, derived at parse time)
  - `n_timesteps`: int (T, derived as number of unique dates after dedup)

**`derive_date_split(all_dates)`** — static method:
- Input: sorted array of unique trading dates (any length T)
- Split at 80% mark: `split_idx = int(T * 0.8)`
- Returns `(train_start, train_end, val_start, val_end)` as `date` objects

### 2. `app/data/sources/database.py` — NEW
DB-backed sources that return pre-computed features for the training loop.

**`DatabaseMarketDataSource`**
- Implements same interface as `MarketDataSource`
- `async fetch_ohlcv_batch(isins: list[str], start, end)` → `dict[str, DataFrame]`
  - Single SQL query with `WHERE asset.isin IN (...) AND date BETWEEN start AND end`
  - No loop over individual ISINs — fetches all N assets in one query
  - Pivots result to `dict[isin → DataFrame[date, open, high, low, close, volume, return_pct, rsi, macd_hist]]`
  - The pipeline detects these extra columns and skips recomputation

**`DatabaseESGDataSource`**
- Implements same interface as `ESGDataSource`
- `async fetch_esg_batch(isins: list[str], start, end)` → `dict[str, DataFrame]`
  - Single SQL query for all N ISINs at once
  - Returns `dict[isin → DataFrame[date, bloomberg_score, lesg_score, esg_b_norm, esg_l_norm, delta_esg, mu_esg]]`
  - The pipeline detects these extra columns and skips ESG normalization

Both create a short-lived async engine per call (safe for Celery worker subprocesses). For production-scale (N×T up to millions of rows), the single-query approach with `IN (...)` clause handles any N efficiently via index on `asset.isin` + `market_data.date`.

---

## Files to Modify

### 3. `app/models/domain.py`
Add new columns to `MarketData` and `ESGScore` (see schema changes above).
Add new `TrainingNormalizerParams` model.

### 4. `app/data/pipeline.py` — KEY CHANGE
Modify `prepare()` to detect pre-computed features and skip recomputation:

```python
# ── 3. Technical indicators ───────────────────────────────────────────
# DB source returns DataFrames with pre-computed columns → use them directly
has_precomputed_indicators = "return_pct" in ohlcv_arrays and \
                             "rsi" in ohlcv_arrays and \
                             "macd_hist" in ohlcv_arrays
if has_precomputed_indicators:
    raw_returns   = ohlcv_arrays["return_pct"]   # (T, N) — already computed
    rsi_raw       = ohlcv_arrays["rsi"]           # (T, N) — already computed
    macd_hist_raw = ohlcv_arrays["macd_hist"]     # (T, N) — already computed
else:
    raw_returns   = compute_returns(close_)
    rsi_raw       = compute_rsi_matrix(close_, cfg.rsi_period)
    macd_hist_raw = compute_macd_histogram_matrix(close_, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

# ── 5. Normalization ──────────────────────────────────────────────────
# DB source returns ESG with cross-sectional normalization already applied
has_precomputed_esg_norm = all(
    col in esg_arrays for col in ("esg_b_norm", "esg_l_norm", "delta_esg", "mu_esg")
)
if has_precomputed_esg_norm:
    # ESG cross-sectional normalization already done in Stage 2 — use directly
    normed = normalizer.fit_transform_market_only(market_data)  # new method — only normalizes OHLCV/indicators
    normed["esg_b_norm"] = esg_arrays["esg_b_norm"]
    normed["esg_l_norm"] = esg_arrays["esg_l_norm"]
    normed["delta_esg"]  = esg_arrays["delta_esg"]
    normed["mu_esg"]     = esg_arrays["mu_esg"]
else:
    # Original path — normalizer handles everything
    normed = normalizer.fit_transform(market_data, bloomberg_raw, lesg_raw)
```

Also add `fit_transform_market_only()` to `DataNormalizer` — same as `fit_transform()` but skips ESG cross-sectional step.

### 5. `app/data/preprocessing/normalizer.py`
Add:
- `fit_transform_market_only(market_data)` — normalizes only OHLCV/indicators (skips cross-sectional ESG)
- `transform_market_only(market_data)` — for val window reuse
- `to_param_records(job_id: str, isins: list[str]) → list[dict]` — exports min/max as DB records

### 6. `app/services/training_service.py`
Add `start_training_from_xlsx()` method — executes all 4 stages:

```python
async def start_training_from_xlsx(self, tmp_paths, portfolio_model, topology,
                                    train_start, train_end, val_start, val_end,
                                    hyperparams) -> str:
    job_id = str(uuid.uuid4())

    # ── Stage 1: Parse XLSX + compute indicators (vectorized over all N assets) ──
    parsed = XLSXDataSource().parse_files(tmp_paths)
    # N and T are derived dynamically from parsed data — never hardcoded
    isins = [a["isin"] for a in parsed.assets]  # length = N (whatever the XLSX contains)

    await self._upsert_assets(parsed.assets)
    await self._upsert_market_data(parsed.market_df)  # bulk, any (T×N) shape
    await self._upsert_esg_raw(parsed.esg_df)         # bulk, any (T×N) shape

    # ── Stage 2: Cross-sectional ESG normalization (vectorized per-date, all N) ──
    await self._compute_esg_normalization(isins)

    # ── Derive/validate date ranges ────────────────────────────────────────────
    if None in (train_start, train_end, val_start, val_end):
        train_start, train_end, val_start, val_end = \
            XLSXDataSource.derive_date_split(parsed.all_dates())

    # ── Stage 3: Fit time-series normalizer, persist params for all N assets ───
    market_src = DatabaseMarketDataSource(cfg.postgres_dsn)
    esg_src    = DatabaseESGDataSource(cfg.postgres_dsn)
    pipeline   = DataPipeline(market_src, esg_src)
    train_ds   = await pipeline.prepare(isins, train_start, train_end, fit=True)
    param_records = train_ds.normalizer.to_param_records(job_id, isins)
    await self._upsert_normalizer_params(param_records)  # 8×N rows, N is dynamic

    config = {
        "data_source": "database",
        "assets": isins,                              # N ISINs, derived from XLSX
        "n_assets": len(isins),                       # N — dynamic
        "train_start": str(train_start), "train_end": str(train_end),
        "val_start": str(val_start),     "val_end": str(val_end),
        "portfolio_model": portfolio_model,
        "topologies": ["cooperative","competitive","mixed"] if topology=="all" else [topology],
        "hyperparams": hyperparams,
    }
    await self._create_job_record(job_id, config)
    run_training_job.delay(job_id, config)
    return job_id
```

**Helper methods** (all use SQLAlchemy Core bulk operations, NOT ORM `add()` loops):

- `_upsert_assets(assets: list[dict])` — `INSERT ... ON CONFLICT (isin) DO UPDATE SET name, sector` — single statement for all N assets
- `_upsert_market_data(df: DataFrame)` — convert DataFrame to list of dicts, `executemany` bulk INSERT with `ON CONFLICT (asset_id, date) DO UPDATE` (updates indicators if file re-uploaded); handles any T×N rows
- `_upsert_esg_raw(df: DataFrame)` — same pattern for ESG; handles any T×N rows
- `_compute_esg_normalization(isins: list[str])` — loads all ESG data into pandas DataFrame, applies `.groupby("date")` cross-sectional min-max normalization (vectorized over N per date, over all T dates), bulk UPDATE back to DB
- `_upsert_normalizer_params(records: list[dict])` — INSERT `8×N` rows (N dynamic) with `ON CONFLICT DO UPDATE`

### 7. `app/api/routes/training.py`
Replace JSON body endpoint with **multipart form**:

```python
@router.post("/start", status_code=202)
async def start_training(
    files:            list[UploadFile] = File(..., description="One or more .xlsx data files"),
    portfolio_model:  str  = Form(...),
    topology:         str  = Form(default="all"),
    train_start:      str | None = Form(default=None),
    train_end:        str | None = Form(default=None),
    val_start:        str | None = Form(default=None),
    val_end:          str | None = Form(default=None),
    hyperparams_json: str = Form(default="{}"),
    service:          TrainingService = Depends(get_training_service),
) -> TrainingJobResponse:
    # validate portfolio_model ∈ {A,B,C}, topology ∈ valid set
    # parse hyperparams_json → HyperParams
    # save files to tempfile.mkdtemp(), get tmp_paths
    # call service.start_training_from_xlsx(tmp_paths, ...)
    # cleanup temp files in finally block
```

### 8. `app/workers/tasks.py`
In `_async_run_training()`, branch on `config["data_source"]`:

```python
if config.get("data_source") == "database":
    from app.data.sources.database import DatabaseMarketDataSource, DatabaseESGDataSource
    market_src = DatabaseMarketDataSource(cfg.postgres_dsn)
    esg_src    = DatabaseESGDataSource(cfg.postgres_dsn)
    # Normalizer: load frozen params from DB for this job
    normalizer = await DataNormalizer.load_from_db(job_id, cfg.postgres_dsn)
else:
    # backward compat: yfinance + stub (non-XLSX path)
    market_src = MarketDataSource(redis_client, ttl=cfg.redis_ttl_market)
    esg_src    = ESGDataSource(bloomberg_api_key=cfg.bloomberg_api_key, ...)
    normalizer = None  # pipeline fits normalizer fresh
```

When `data_source == "database"`: pass pre-loaded normalizer into `pipeline.prepare()` so it skips the fit step and uses the DB-loaded params.

Add `DataNormalizer.load_from_db(job_id, dsn)` — reads `training_normalizer_params` table, reconstructs min/max arrays, returns a frozen `DataNormalizer` instance.

### 9. `requirements.txt`
```diff
+ openpyxl==3.1.5          # pd.read_excel() backend for .xlsx files
+ python-multipart==0.0.20  # FastAPI multipart/form-data file upload support
```

---

## Complete Data Flow After Change

```
POST /training/start  (multipart: files + form fields)
         │
         ▼
TrainingService.start_training_from_xlsx()
         │
  ┌──────────────────────────────────────────────────┐
  │  Stage 1: XLSXDataSource.parse_files()           │
  │   XLSX → OHLCV + ESG (raw)                       │
  │   Compute: return_pct, RSI(14), MACD(12,26,9)    │
  │   → INSERT to assets, market_data, esg_scores    │
  └──────────────────────────────────────────────────┘
         │
  ┌──────────────────────────────────────────────────┐
  │  Stage 2: _compute_esg_normalization()           │
  │   For each date T across all N ISINs:            │
  │     esg_b_norm = cross-sectional min-max         │
  │     esg_l_norm = cross-sectional min-max         │
  │     delta_esg  = |esg_b_norm - esg_l_norm|       │
  │     mu_esg     = (esg_b_norm + esg_l_norm) / 2   │
  │   → UPDATE esg_scores                            │
  └──────────────────────────────────────────────────┘
         │
  ┌──────────────────────────────────────────────────┐
  │  Stage 3: DataPipeline.prepare() [fit only]      │
  │   Read pre-computed features from DB             │
  │   Fit time-series min/max over training window   │
  │   → INSERT to training_normalizer_params         │
  └──────────────────────────────────────────────────┘
         │
         └─── Celery: run_training_job(job_id, config)
                            │
  ┌─────────────────────────▼────────────────────────┐
  │  Stage 4: MASAC Training                         │
  │   DatabaseMarketDataSource → pre-computed OHLCV, │
  │     return_pct, rsi, macd_hist from DB           │
  │   DatabaseESGDataSource → esg_b_norm, esg_l_norm │
  │     delta_esg, mu_esg from DB                    │
  │   DataNormalizer (loaded from DB) → apply frozen │
  │     time-series normalization to OHLCV features  │
  │   Assemble state vectors (10N) → TrainingEnv     │
  │   → MASAC trains 500k steps max                  │
  └──────────────────────────────────────────────────┘
```

---

## Files Summary

| File | Action |
|---|---|
| `app/models/domain.py` | MODIFY — add 3 cols to `market_data`, 4 cols to `esg_scores`, new `TrainingNormalizerParams` model |
| `app/data/sources/xlsx.py` | CREATE — XLSX parser + indicator computation |
| `app/data/sources/database.py` | CREATE — DB-backed market + ESG sources |
| `app/data/preprocessing/normalizer.py` | MODIFY — add `fit_transform_market_only`, `to_param_records`, `load_from_db` |
| `app/data/pipeline.py` | MODIFY — detect pre-computed features, skip recomputation when available; replace `cfg.macd_warmup_days` with dynamic `max(cfg.macd_slow, cfg.rsi_period)` |
| `app/config.py` | MODIFY — remove hardcoded `macd_warmup_days = 26`; warmup is now computed inline from existing `macd_slow` + `rsi_period` settings |
| `app/services/training_service.py` | MODIFY — add `start_training_from_xlsx()` with all stages |
| `app/api/routes/training.py` | MODIFY — multipart form endpoint |
| `app/workers/tasks.py` | MODIFY — DB data sources + load frozen normalizer from DB |
| `requirements.txt` | MODIFY — add `openpyxl`, `python-multipart` |

**Not changed:** `rl/` (MASAC, trainer, environment, replay buffer), `agents/`, `routes/` (portfolio, data, websocket), `services/portfolio_service.py`, `config.py`

---

## Alembic Migration Note
The domain.py changes (new columns + new table) require an Alembic migration. A new migration script must be generated:
```
alembic revision --autogenerate -m "add computed features columns and normalizer params table"
alembic upgrade head
```

---

## Verification

1. **Upload test**: POST `synthetic_stock_esg_dataset.xlsx` → `POST /training/start` (multipart, portfolio_model=C, no dates → auto-split)
2. **N and T are dynamic**: After upload, `SELECT COUNT(DISTINCT isin) FROM assets` shows actual N from file; `SELECT COUNT(DISTINCT date) FROM market_data` shows actual T — these must match exactly what's in the XLSX, no hardcoded values
3. **Stage 1 check**: `SELECT isin, COUNT(*) FROM market_data GROUP BY isin` — each ISIN should have T rows; `return_pct` non-NULL for all but first row per ISIN; `rsi` non-NULL after row 13 per ISIN; `macd_hist` non-NULL after row 25 per ISIN
4. **Stage 2 check**: `SELECT COUNT(*) FROM esg_scores WHERE esg_b_norm IS NULL` = 0 (every row has cross-sectional normalized ESG + ΔeSG + μESG)
5. **Stage 3 check**: `SELECT COUNT(*) FROM training_normalizer_params WHERE job_id = '<id>'` = 8 × N (exactly 8 feature rows per asset — N is whatever the XLSX contained)
6. **Dynamic N state vector**: `config_json.n_assets` in `training_jobs` reflects actual N; MASAC actor/critic dimensions (10N, 33N) must match this N
7. **Celery worker**: logs show DatabaseMarketDataSource path; no yfinance/stub calls
8. **Multi-file test**: Upload two XLSX files covering different ISINs → N increases correctly; covering same ISINs + overlapping dates → deduplication leaves no duplicate rows
9. **10-year scale test**: Load large multi-year file → verify bulk insert completes without row-by-row ORM loop; check `training_normalizer_params` still has exactly 8×N rows
