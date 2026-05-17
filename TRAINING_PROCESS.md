# MADRL Portfolio System — Training Process

End-to-end explanation of everything that happens from file upload to a trained model ready for inference.

---

## Overview

The system trains a **Multi-Agent Deep Reinforcement Learning (MADRL)** model to build ESG-aware investment portfolios. Three AI agents (Bloomberg, LESG, Financial) compete and cooperate across three game-theoretic topologies (cooperative, competitive, mixed). The process is divided into four sequential stages, each building on the previous.

```
POST /training/start  (XLSX file + form fields)
         │
    Stage 1 ── Parse XLSX → store raw data + derived indicators in PostgreSQL
         │
    Stage 2 ── Cross-sectional ESG normalization → update esg_scores table
         │
    Stage 3 ── Fit time-series normalizer → store min/max params in DB
         │
    Stage 4 ── Celery background task: MASAC training loop (up to 500,000 steps)
         │
    Model weights saved to model_store/{job_id}/{topology}/
```

---

## Stage 1 — XLSX Parsing and Raw Data Storage

**Trigger:** API receives `POST /api/v1/training/start` as multipart form with one or more `.xlsx` files.

**What happens:**

1. `XLSXDataSource.parse_files(paths)` reads every file, expecting sheet `Stock_ESG_Dataset` with columns:
   ```
   Date | ISIN | Company name | Sector | Open | High | Low | Close | Volume | RSI | Bloom. ESG (0-100) | LESG ESG (0-10)
   ```
2. Files are concatenated and **deduplicated on `(ISIN, Date)`** — safe to upload overlapping files.
3. Volume strings are parsed: `"10.5M"` → `10,500,000`. Handles K/M/B/T suffixes, case-insensitive.
4. **RSI is read directly from the XLSX column** — it is NOT recomputed. The data provider already computed it.
5. Two indicators are computed from Close:
   - `return_pct = (Close_t - Close_{t-1}) / Close_{t-1}` — grouped per ISIN, so each asset's first row is NULL
   - `macd_hist` = MACD histogram from Close using `macd_fast=2, macd_slow=3, macd_signal=2` (set in `.env`) — first `macd_slow - 1` rows per ISIN are NULL (warmup)
6. The parsed data is upserted into three tables:
   - `assets (isin, name, sector)` — one row per unique company; `ON CONFLICT DO UPDATE`
   - `market_data (asset_id, date, open, high, low, close, volume, rsi, return_pct, macd_hist)` — T×N rows; re-uploading the same file updates existing rows
   - `esg_scores (asset_id, date, bloomberg_score, lesg_score)` — raw ESG scores; normalized columns filled in Stage 2

**Why this design:**

- Storing raw data + derived indicators in one pass means the training worker never needs to call yfinance or any external API — it reads everything from the database.
- Bulk upsert (single SQL statement with `executemany`) handles any T×N size without row-by-row ORM loops.
- Deduplication at parse time prevents training on duplicate rows.

---

## Stage 2 — Cross-Sectional ESG Normalization

**What happens:**

For every trading date `t`, across all N assets in the job:

```
ESG_B_norm(i, t) = (ESG_B(i, t) - min_i ESG_B(t)) / (max_i ESG_B(t) - min_i ESG_B(t))
ESG_L_norm(i, t) = same formula for LESG
delta_esg(i, t)  = |ESG_B_norm(i, t) - ESG_L_norm(i, t)|
mu_esg(i, t)     = (ESG_B_norm(i, t) + ESG_L_norm(i, t)) / 2
```

Implementation:

1. Load all ESG rows for all N assets in one SQL query
2. `df.groupby("date").transform(cs_norm)` — vectorized, processes all N assets per date in one pandas pass
3. Bulk UPDATE the `esg_scores` table with four new columns: `esg_b_norm, esg_l_norm, delta_esg, mu_esg`

**Why cross-sectional (per date, across assets) instead of per-asset over time:**

- ESG scores are **relative** signals. An asset with Bloomberg score 70 is only meaningful compared to its peers on the same day. If every asset has score 70 on the same day, all get `esg_b_norm = 0.5`.
- Time-series normalization of ESG would mix the "good ESG period" vs "bad ESG period" signal — which is not what we want. We want to know: is this asset better or worse than its peers _today_.

**What delta_esg and mu_esg mean:**

- `delta_esg` = ESG disagreement between Bloomberg and LESG providers. High value = the two providers disagree on this asset's ESG quality. The cooperative topology penalizes this.
- `mu_esg` = ESG consensus. Average of both providers' normalized scores. Used as the portfolio-level ESG quality metric.

---

## Stage 3 — Time-Series Normalizer Fitting

**What happens:**

For the training window only (e.g., 80% of dates), for each asset ISIN, for each of 8 market features:

```
features: open, high, low, close, volume, return_pct, rsi, macd_hist
```

Compute:

```
min_val(isin, feature) = min value of that feature for that asset over the training window
max_val(isin, feature) = max value of that feature for that asset over the training window
```

These 8 × N parameters are stored in the `training_normalizer_params` table, linked to the job ID.

**Why store normalizer params in the database:**

- After training completes, the API server may restart. If normalizer params were only in memory, inference would be broken.
- A frozen normalizer ensures no look-ahead bias: the validation window and inference use the **exact same min/max values** that were computed on the training window. Refitting on inference data would leak future information.
- Storing per-job means multiple models (A/B/C) can coexist with their own independent normalizers.

**Why per-asset normalization (not global):**

- A stock trading at \$500 and a stock at $5 need independent scaling. Global normalization would collapse the lower-priced stock's variation to near zero.

**Date split logic:**

- If dates are not supplied in the API call, the system automatically splits: first 80% → training window, last 20% → validation window.

---

## Stage 4 — MASAC Training (Celery Background Task)

The Celery worker receives `job_id` and `config`. All data comes from the database — no XLSX re-reading, no external API calls.

### 4.1 Data Loading from Database

```
DatabaseMarketDataSource → reads market_data for N ISINs over training window
DatabaseESGDataSource    → reads esg_scores (including Stage 2 columns) for N ISINs
DataPipeline.prepare()   → aligns on common dates, removes MACD warmup rows, applies frozen normalizer
```

**Warmup removal:** The first `macd_slow` rows per ISIN have `macd_hist = NULL`. These rows are dropped from both train and validation datasets. With the default `macd_slow = 3`, the first 3 rows per ISIN are discarded.

**State vector assembly:** For each timestep `t`, a `10N`-dimensional observation vector is built:

```
state[t] = [
    open_norm(t, 1..N),     ← N values: normalized opening prices for all assets
    high_norm(t, 1..N),     ← N values
    low_norm(t, 1..N),      ← N values
    close_norm(t, 1..N),    ← N values
    volume_norm(t, 1..N),   ← N values
    rsi_norm(t, 1..N),      ← N values: normalized RSI
    macd_hist_norm(t, 1..N),← N values: normalized MACD histogram
    return_pct_norm(t, 1..N),← N values: normalized daily returns
    delta_esg(t, 1..N),     ← N values: ESG disagreement (from Stage 2)
    mu_esg(t, 1..N),        ← N values: ESG consensus (from Stage 2)
]
```

### 4.2 Three Agents

The system trains three MASAC agents simultaneously, each with its own actor network and twin critics:

| Agent                 | Objective                                |
| --------------------- | ---------------------------------------- |
| **BloombergESGAgent** | Maximise Bloomberg ESG-weighted returns  |
| **LESGAgent**         | Maximise LESG ESG-weighted returns       |
| **FinancialAgent**    | Maximise pure financial returns (Sharpe) |

Each agent outputs an N-dimensional allocation score vector `z ∈ ℝᴺ`. The three vectors are averaged and passed through Softmax to produce portfolio weights that sum to 1.0:

```
z_joint = (z_Bloomberg + z_LESG + z_Financial) / 3
weights = Softmax(z_joint)
```

**Why three agents instead of one:**

- A single agent with a combined objective would trade off ESG against returns in a fixed way, controlled by hyperparameters chosen before training.
- Three agents in a game-theoretic setting naturally find the Pareto frontier — they compete and cooperate, and the aggregated result reflects the balance of their objectives at inference time without retraining.

### 4.3 Three Topologies

Each training run iterates through all three topologies sequentially (cooperative → competitive → mixed):

| Topology        | Beta (β)     | ESG penalty | Meaning                                                                                                                   |
| --------------- | ------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Cooperative** | β (e.g. 0.3) | Full        | All agents penalised for high ESG disagreement. Favours stocks both Bloomberg and LESG agree are high-quality.            |
| **Competitive** | 0            | None        | No shared penalty. Each agent optimises its own private objective. May allocate more to high-return/controversial stocks. |
| **Mixed**       | β/2          | Partial     | Compromise between the two extremes.                                                                                      |

The effective beta is applied inside the reward function (Portfolio Model C):

```
reward_Bloomberg = r_t + α₁ · ESG_B_norm - β_eff · delta_esg
reward_LESG      = r_t + α₂ · ESG_L_norm - β_eff · delta_esg
reward_Financial = r_t + α₃ · (ESG_B_norm + ESG_L_norm)/2 - β_eff · delta_esg
```

### 4.4 Training Loop

```
max_steps: 500,000
warmup:    10,000 steps (random actions, no gradient updates)
batch_size: 256
convergence: rolling std of mean entropy over 100 steps < 0.01
```

**Warmup (steps 0–10,000):**

- Random portfolio weights are chosen instead of using the actor networks.
- These transitions fill the Replay Buffer with diverse experiences.
- No gradient updates happen — the actors and critics are untouched.
- Reason: gradient updates on an empty or near-empty buffer would overfit to the first few transitions and destabilize training.

**After warmup (steps 10,000–500,000):**
Per step:

1. Actor networks select allocation scores from the current observation
2. Environment steps forward one day: computes portfolio return and per-agent rewards
3. Transition `(obs, actions, rewards, next_obs, done)` is stored in the Replay Buffer
4. MASAC update runs: sample 256 transitions, update critics and actors
5. Every 500 steps: publish metrics to Redis PubSub → WebSocket → training monitor HTML
6. Every 10,000 steps: evaluate on validation window; if best Sharpe so far, save checkpoint

**Episode reset:**
The environment has a 252-step episode length (one trading year). When the agent reaches the end of the dataset, or completes 252 steps, it resets to a random start position within the dataset. With 14 rows of data, the agent sees the same 14 days ~35,000 times — each time with slightly different policy weights, learning from the reward signal.

**Convergence check:**
If the rolling standard deviation of mean entropy across all three agents over the last 100 steps drops below 0.01, training stops early. This signals that the agents' policies have stabilised — they are no longer exploring and their action distributions are consistent.

### 4.5 MASAC Update (Centralized Training, Decentralized Execution)

Each agent has:

- **1 Actor** — decentralized: takes only the shared `10N` observation, outputs `N` allocation scores
- **2 Critics + 2 target Critics** — centralized: take the joint observation (`3 × 10N = 30N`) and all three agents' actions (`3 × N = 3N`) as input. Total critic input: `33N` dimensions.

**Why twin critics:** The minimum of two Q-value estimates reduces overestimation bias (Clipped Double-Q trick from TD3/SAC).

**Why centralized critics:** In multi-agent settings, if each critic only sees its own agent's action, it cannot distinguish between "my action was bad" and "another agent's action caused the bad outcome." Centralized critics with joint inputs eliminate this credit assignment problem.

**Temperature auto-tuning:** Each agent has a learnable temperature parameter `α_T` that controls exploration. Target entropy is set to `-N` (negative number of assets). When the policy becomes too deterministic (entropy < -N), `α_T` increases to encourage more exploration. This eliminates the need to manually tune exploration.

**Soft target update:** Critic target networks are updated at rate `τ = 0.005`:

```
target = (1 - τ) × target + τ × online
```

This prevents the training target from changing too fast, which would destabilize learning.

### 4.6 Checkpoint Saving

Every 10,000 steps, the validation Sharpe ratio is computed over a 63-day rolling window. If it exceeds the previous best, model weights are saved to:

```
model_store/{job_id}/{topology}/bloomberg.pt
model_store/{job_id}/{topology}/lesg.pt
model_store/{job_id}/{topology}/financial.pt
```

Each `.pt` file contains: `actor`, `critic`, `critic_target`, `log_alpha_t` state dicts.

Only the best-Sharpe checkpoint is kept per topology — no bloat.

### 4.7 Database Status Updates

Throughout training:

- `status = "running"` — set when training begins
- `current_step` — updated after each topology completes
- `best_sharpe, best_mu_esg` — updated after each topology
- `status = "completed"` — set when all topologies finish
- `status = "failed", error_message` — set on any uncaught exception (training still marked failed in Celery too)

---

## Real-Time Monitoring

**Redis PubSub:** During training, the worker publishes to channel `pubsub:training:{job_id}` every 500 steps. Messages include: step, entropy, rewards per agent, actor/critic losses, temperature.

**WebSocket:** The API subscribes to this channel and streams messages to any connected browser via `ws://localhost:8000/ws/training/{job_id}`.

**Snapshot key:** Every published message is also stored in `training:snapshot:{job_id}` (TTL 1 hour). When a browser connects _after_ training has finished, it immediately receives the last known state instead of seeing nothing.

**Training monitor:** Open `training_monitor.html` in any browser, enter the job ID, click Connect.

---

## Inference (Portfolio Generation)

After training, call `POST /api/v1/portfolio/generate` with a list of ISINs.

```
1. Find the best completed training job for the requested portfolio model (highest Sharpe)
2. Load frozen normalizer from training_normalizer_params (DB) — exact same min/max as training
3. Fetch pre-computed features from market_data + esg_scores for the validation window
4. Apply frozen normalizer → assemble state vectors (no look-ahead)
5. Load saved actor weights into PortfolioOrchestratorAgent
6. Run all three topologies concurrently
7. Each topology: Bloomberg/LESG/Financial actors produce z vectors → average → Softmax → weights
8. Compute aggregate metrics (return, risk, Sharpe, μESG, ΔESG)
9. Return three-panel comparison response + persist to portfolio_results table
```

The inference window uses the validation dates from the original training job — data the model never saw during training.

---

## Configuration Reference

All parameters live in `.env` and can be changed without code modifications:

| Parameter                   | Default | Effect                                                               |
| --------------------------- | ------- | -------------------------------------------------------------------- |
| `MASAC_MAX_STEPS`           | 500,000 | Maximum training steps per topology                                  |
| `MASAC_WARMUP_STEPS`        | 10,000  | Steps of random exploration before gradient updates                  |
| `MASAC_BATCH_SIZE`          | 256     | Transitions sampled per gradient update                              |
| `MASAC_CONVERGENCE_EPSILON` | 0.01    | Entropy rolling std threshold for early stopping                     |
| `MASAC_CONVERGENCE_WINDOW`  | 100     | Steps to measure entropy stability over                              |
| `MASAC_GAMMA`               | 0.99    | Discount factor (how much future rewards matter)                     |
| `MASAC_TAU`                 | 0.005   | Soft target update rate                                              |
| `MASAC_LR_ACTOR`            | 3e-4    | Actor learning rate                                                  |
| `MASAC_LR_CRITIC`           | 3e-4    | Critic learning rate                                                 |
| `MASAC_HIDDEN_SIZE`         | 256     | Hidden layer width for all networks                                  |
| `MACD_FAST`                 | 2       | MACD fast EMA period                                                 |
| `MACD_SLOW`                 | 3       | MACD slow EMA period (also = warmup rows removed per ISIN)           |
| `MACD_SIGNAL`               | 2       | MACD signal EMA period                                               |
| `RSI_PERIOD`                | 14      | RSI period (used for normalization only — RSI values come from XLSX) |

---

## Data Flow Summary

```
XLSX file
  └─ XLSXDataSource.parse_files()
        ├─ assets table           (isin, name, sector)
        ├─ market_data table      (OHLCV + RSI from file + return_pct + macd_hist computed)
        └─ esg_scores table       (bloomberg_score, lesg_score raw)
                │
                ▼
        Stage 2: cross-sectional ESG norm (per date, all N assets)
                │
                └─ esg_scores updated: esg_b_norm, esg_l_norm, delta_esg, mu_esg
                │
                ▼
        Stage 3: time-series normalizer fitting (training window only)
                │
                └─ training_normalizer_params: 8 × N rows (min/max per feature per asset)
                │
                ▼
        Celery task: Stage 4 MASAC training
                ├─ DatabaseMarketDataSource → reads pre-computed OHLCV + indicators
                ├─ DatabaseESGDataSource   → reads pre-computed ESG norms
                ├─ DataNormalizer.load_from_db() → reconstructs frozen scaler
                ├─ DataPipeline.prepare()  → assembles state vectors (10N each)
                │
                ├─ topology: cooperative  → MASAC trains → best checkpoint saved
                ├─ topology: competitive  → MASAC trains → best checkpoint saved
                └─ topology: mixed        → MASAC trains → best checkpoint saved
                │
                ▼
        model_store/{job_id}/{topology}/{bloomberg,lesg,financial}.pt
                │
                ▼
        POST /portfolio/generate → loads weights → runs inference → returns 3-panel comparison
```
