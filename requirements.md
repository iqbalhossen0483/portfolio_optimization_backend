# AI Agent Workflow: Multi-Agent Deep Reinforcement Learning (MADRL) System

## 1. Data Architecture & Preprocessing

### A. Raw Input Data (Source: Market APIs & ESG Databases)

- **Asset Identifiers:** **ISIN** (International Securities Identification Number) mapped to industry sectors.
- **Control Variables (Price & Volume History):** **OHLCV Data** (Open, High, Low, Close, Volume).
  - _Note on OHLCV integration:_ The daily **Returns** are calculated _only_ using the Closing prices. The full OHLCV array acts as the environmental control variables to provide deeper representations of market liquidity, volatility, and intraday price action.
- **Technical Indicators (State Features):** OHLCV data is used to generate dynamic momentum indicators — specifically **MACD** (Moving Average Convergence Divergence) and **RSI** (Relative Strength Index). These serve as the active state features so the RL agents can effectively evaluate real-time market trends.
- **Divergent ESG Signals:**
  - $ESG^{(B)}$: Bloomberg Overall ESG Score (Scaled 0–100).
  - $ESG^{(L)}$: LESG Overall ESG Score (Scaled 0–10).
    _(Note: The study gives equal 50/50 baseline importance to both sources)._

### B. Data Normalization & Standardization

All state features are normalized using **Min-Max Normalization (0.0–1.0)**. However, the normalization axis differs by feature type — this distinction is critical for avoiding data leakage and preserving reward meaningfulness:

1. **ESG Scores — Cross-sectional normalization (across assets at time $t$):**

   $$ESG\_norm^{(B)}(i,t) = \frac{ESG^{(B)}(i,t) - \min_i ESG^{(B)}(t)}{\max_i ESG^{(B)}(t) - \min_i ESG^{(B)}(t)}$$

   The min and max are taken across all $N$ assets on the same day $t$. This ensures Bloomberg (0–100) and LESG (0–10) are harmonized on the same relative scale before computing $\Delta ESG_t$ and $\mu ESG_t$. Cross-sectional normalization introduces no temporal look-ahead — only same-day peer values are used.

2. **OHLCV Data & Technical Indicators — Time-series normalization (per asset, over training window):**

   $$Close\_norm(i,t) = \frac{Close(i,t) - \min_{\tau \in W} Close(i,\tau)}{\max_{\tau \in W} Close(i,\tau) - \min_{\tau \in W} Close(i,\tau)}$$

   The min and max are computed over the **training window $W$ only** and **frozen** before being applied to the test window. A value of 0.9 means "near this stock's historical high"; 0.1 means "near its historical low." Recomputing min/max using test-period data would introduce look-ahead bias. The same time-series procedure applies to RSI and MACD.

| Feature                                | Normalization Axis                              | Min/Max Source                         |
| -------------------------------------- | ----------------------------------------------- | -------------------------------------- |
| ESG (Bloomberg, LESG)                  | Cross-sectional (across $N$ assets at time $t$) | All assets on day $t$                  |
| Individual return $R_{i,t}$            | Time-series (per asset)                         | Training window only — frozen for test |
| OHLCV (Open, High, Low, Close, Volume) | Time-series (per asset)                         | Training window only — frozen for test |
| RSI                                    | Time-series (per asset)                         | Training window only — frozen for test |
| MACD histogram                         | Time-series (per asset)                         | Training window only — frozen for test |

_Crucially, ESG normalization (Step 1) must be completed BEFORE calculating $\Delta ESG_t$ and $\mu ESG_t$, so the disagreement and consensus signals reflect harmonized, comparable values._

---

## 2. Mathematical Framework (The Markov Game)

The environment is modeled as a Markov Game: $G = (S, \{A_i\}, P, \{R_i\})$.
As we apply a Markov Game for the multi-agent system, the interaction topology can be defined as:

- **Cooperative:** Team Game (Maximize a shared joint reward).
- **Competitive:** Non-cooperative Game (No shared penalty; each agent maximizes its own private reward independently).
- **Mixed:** General-sum Game (Arbitrary, often conflicting agent goals).

_All three interaction modes run in parallel for every user query. Each produces an independent portfolio recommendation, which the system returns to the user simultaneously for side-by-side comparison. No topology's output is merged into another._

_The key mechanical difference between topologies is the treatment of the shared ambiguity penalty $\beta \cdot \Delta ESG_t$ in the reward function: in **Cooperative** mode $\beta > 0$ — the penalty is applied to the shared joint reward so all agents collectively bear the cost of high ESG disagreement; in **Competitive** mode $\beta = 0$ — the shared penalty is removed and each agent maximizes only its own private reward; in **Mixed** mode a partial penalty applies, producing outcomes intermediate between the two extremes._

### Phase 1: Asset-Level Metrics (Internal Logic)

1.  **Individual Return:** $R_{i,t} = \frac{Close_{i,t} - Close_{i,t-1}}{Close_{i,t-1}}$
2.  **Individual Risk:** $\sigma_i = \sqrt{\frac{\sum (R_{i,t} - E(R_i))^2}{n-1}}$ — display metric only, computed from the out-of-sample validation window. Does not enter the agent observation vector or the reward function.
3.  **Per-stock ESG Disagreement ($\Delta ESG_{i,t}$):** $|ESG_{i,t}^{(B)} - ESG_{i,t}^{(L)}|$. _(Calculated post-normalization)_. This per-stock value enters the agent observation vector.
4.  **Per-stock ESG Consensus ($\mu ESG_{i,t}$):** $\frac{ESG_{i,t}^{(B)} + ESG_{i,t}^{(L)}}{2}$. _(Calculated post-normalization)_. This per-stock value enters the agent observation vector.
5.  **Portfolio-level ESG Disagreement ($\Delta ESG_t$):** $\sum_i w_{i,t} \cdot \Delta ESG_{i,t}$, where $w_{i,t}$ are the current Softmax portfolio weights. This portfolio-weighted scalar is the quantity used in the reward function — it converts the per-stock disagreement vector into a single scalar penalty term.
6.  **Portfolio-level ESG Scores ($ESG_t^{(B)}$, $ESG_t^{(L)}$):** $ESG_t^{(B)} = \sum_i w_{i,t} \cdot ESG^{(B)}_\text{norm}(i,t)$ and equivalently $ESG_t^{(L)} = \sum_i w_{i,t} \cdot ESG^{(L)}_\text{norm}(i,t)$. These portfolio-weighted scalars are used in the reward function for Portfolios A and C — the same aggregation pattern as Item 5.

### Phase 2: Agent Design (Divergent Perspectives)

The framework utilizes three specialized agents observing the same global market state ($s_t$). Each agent's observation vector includes: normalized OHLCV ($5N$), RSI ($N$), MACD histogram ($N$), $R_{i,t}$ (normalized individual return, $N$-vector), $\Delta ESG_{i,t}$ (per-stock disagreement, $N$-vector), and $\mu ESG_{i,t}$ (per-stock consensus, $N$-vector) — totalling $10N$ features. The portfolio-level scalars $\Delta ESG_t$ and $ESG_t^{(i)}$ defined in Phase 1 Items 5–6 are used in the reward function only — they do not enter the observation vector.

- **Agent 1 (Bloomberg):** Maximizes performance weighted by $ESG^{(B)}$ — its private ESG signal.
- **Agent 2 (LESG):** Maximizes performance weighted by $ESG^{(L)}$ — its private ESG signal.
- **Agent 3 (Financial):** Maximizes raw portfolio simple return ($r_t$) with negligible ESG bias ($\alpha_3 \approx 0$). Acts as the pure financial performance anchor in the multi-agent negotiation.

**Action Space:** Each agent outputs a continuous vector of unnormalized allocation scores $z_t^{(i)} \in \mathbb{R}^N$. The three agents' score vectors are combined via equal-weight averaging to form a single joint score vector:

$$z_t^{joint} = \frac{z_t^{(B)} + z_t^{(L)} + z_t^{(F)}}{3}$$

This joint score vector is passed through a single **Softmax** activation function to produce the final investable portfolio weight distribution:

$$w_i = \frac{e^{z_i^{joint}}}{\sum_j e^{z_j^{joint}}}$$

All weights are non-negative and sum to 1.0 (long-only portfolio). An agent that strongly favors a stock outputs a high score for it — averaging ensures each agent has equal voice in the final allocation. The resulting weight vector $[w_1, w_2, ..., w_N]$ is the portfolio submitted to the market.

### Phase 3: Learning Algorithms — MASAC (Multi-Agent Soft Actor-Critic)

The system uses **MASAC**, a multi-agent extension of the Soft Actor-Critic algorithm. MASAC is an **Actor-Critic** method: it combines a policy network (Actor) that decides actions, and a value network (Critic) that evaluates them. The key innovation over standard Actor-Critic is the **maximum-entropy objective** — agents are rewarded not only for financial performance but also for maintaining exploratory, diverse behavior, which prevents premature convergence to suboptimal ESG-financial trade-off policies.

#### Maximum-Entropy Objective

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t \left(r_t + \alpha_T \cdot H\!\left(\pi(\cdot \mid s_t)\right)\right)\right]$$

- $\gamma$ = **Discount factor** — scalar $\in (0, 1]$ weighting future rewards relative to immediate rewards. Default value: $\gamma = 0.99$.
- $H(\pi(\cdot|s_t))$ = **Policy entropy** — measures how spread-out the agent's action distribution is. Higher entropy = more exploration.
- $\alpha_T$ = **Temperature parameter** — auto-tuned during training (see Temperature Tuning below). Uses subscript $T$ to distinguish from the ESG weight hyperparameters $\alpha_i$ in the reward function.

#### Actor Network (Policy)

- **Input:** Each agent's local observation vector — normalized OHLCV ($5N$), RSI ($N$), MACD histogram ($N$), $R_{i,t}$ (normalized individual return, $N$), $\Delta ESG_{i,t}$ (per-stock, $N$), $\mu ESG_{i,t}$ (per-stock, $N$) — total input dimension: $10N$.
- **Output:** Two separate output heads: (1) policy mean $\mu_\pi \in \mathbb{R}^N$; (2) log-variance $\log \sigma^2 \in \mathbb{R}^N$ (log-variance is predicted for numerical stability). A score $z_t \sim \mathcal{N}(\mu_\pi, \sigma^2)$ is sampled and passed **directly** through Softmax — no tanh squashing is applied, as Softmax accepts unbounded real-valued inputs and the action space requires only non-negativity and sum-to-one, both satisfied by Softmax.
- **Execution:** Each Actor operates independently on its own local observation only (decentralized). Deterministic at inference (mean action $\mu_\pi$ used directly).
- **Architecture:** MLP — 2 hidden layers × 256 units each, ReLU activations. Shared trunk splits into two output heads at the final layer.

#### Critic Network (Twin Q-Functions)

- **Input:** The concatenated observations and actions of **all** agents simultaneously — the Critic has a centralized global view during training (CTDE paradigm). Input dimension: $3 \times 10N + 3 \times N = 33N$ (three agents × $10N$ observations plus three agents × $N$ action vectors).
- **Output:** A scalar Q-value — expected cumulative discounted reward for the joint state-action pair, specific to that agent's reward function.
- **Per-agent critics:** Each agent owns its own separate twin critic ($Q_1^{(i)}$, $Q_2^{(i)}$). Because each agent has a distinct reward function ($R_t^{(B)}$, $R_t^{(L)}$, $R_t^{(F)}$), a shared critic cannot simultaneously approximate three different value functions. This yields **6 Q-networks total** (3 agents × 2 twin critics each), plus 6 corresponding **target networks** (soft-updated).
- **Twin Critics:** The **minimum** of $Q_1^{(i)}$ and $Q_2^{(i)}$ is used at each update step — reduces Q-value overestimation bias.
- **Architecture:** MLP — 2 hidden layers × 256 units each, ReLU activations, scalar output.

#### Centralized Training, Decentralized Execution (CTDE)

> _During **training**: each Critic observes the full global state and all agents' actions, enabling accurate joint Q-value estimation with coordination signals. During **execution**: each Actor acts solely on its own local observation — agents are independently deployable with no inter-agent communication required at runtime._

#### Replay Buffer

| Parameter                    | Value                                                                                     |
| :--------------------------- | :---------------------------------------------------------------------------------------- |
| Buffer capacity              | 1,000,000 transitions                                                                     |
| Sampling strategy            | Uniform random                                                                            |
| Minimum fill before training | 10,000 transitions                                                                        |
| Batch size                   | 256                                                                                       |
| Stored per transition        | $(s_t,\ a_t^{(B)},\ a_t^{(L)},\ a_t^{(F)},\ r_t^{(B)},\ r_t^{(L)},\ r_t^{(F)},\ s_{t+1})$ |

#### Training Loop

| Parameter              | Value                                                                                                                                                                                                    |
| :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Time step              | 1 trading day                                                                                                                                                                                            |
| Episode length         | 252 trading days (1 calendar year)                                                                                                                                                                       |
| Episode initialization | Portfolio weights reset to equal-weight ($w_i = 1/N$ for all $i$) at episode start                                                                                                                       |
| Terminal state         | End of episode only — no early termination on drawdown                                                                                                                                                   |
| Training budget        | Maximum 500,000 environment steps                                                                                                                                                                        |
| Gradient updates       | 1 update per environment step (after warmup)                                                                                                                                                             |
| Target network update  | Soft update: $\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$, $\tau = 0.005$                                                                                                                     |
| Optimizer              | Adam, learning rate $= 3 \times 10^{-4}$ for Actor, Critic, and temperature                                                                                                                              |
| Convergence criterion  | Training stops early when the rolling standard deviation of mean policy entropy (averaged across all three agents) over the last 100 steps falls below $\varepsilon = 0.01$, or at 500,000 steps maximum |

#### Temperature Tuning ($\alpha_T$)

The temperature $\alpha_T$ is auto-tuned by minimising:

$$\mathcal{L}(\alpha_T) = \mathbb{E}_{a \sim \pi}\left[-\alpha_T \cdot \log \pi(a|s) - \alpha_T \cdot \bar{H}\right]$$

- **Target entropy:** $\bar{H} = -N$ (negative number of assets in the portfolio universe)
- **Update rule:** $\alpha_T$ is updated with Adam after each gradient step
- **Initial value:** $\alpha_T = 1.0$ (decays automatically toward the target entropy constraint)

#### Technical Indicator Parameters

| Indicator | Parameters                                                                                                                                                                                    | Minimum lookback required        |
| :-------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------- |
| RSI       | Period = 14 trading days                                                                                                                                                                      | 14 days                          |
| MACD      | Fast EMA = 12, Slow EMA = 26, Signal = 9 — **histogram only** (MACD line − signal line) enters the state vector; the histogram is normalized independently per asset over the training window | 26 days (slow EMA stabilisation) |

The first 26 trading days of each training window are consumed as warm-up — no RL updates occur during this period.

---

## 3. Reward Logic & Trade-off Phenomenon

The general reward structure for **Portfolios A and C** is:

$$R_t^{(i)} = r_t + \alpha_i \cdot ESG_t^{(i)} - \beta \cdot \Delta ESG_t$$

_Portfolio B uses a distinct per-agent signed-disagreement formulation — see Portfolio Model Definitions below._

Where:

- $r_t$ = **Portfolio simple return**: $r_t = \sum_i w_i \cdot R_{i,t}$, where $R_{i,t}$ is the individual asset simple return defined in Phase 1 and $w_i$ is the Softmax portfolio weight. This is a raw, sign-preserving signal — positive values indicate a gain, negative values indicate a loss. $r_t$ is **not** Min-Max scaled; preserving sign is essential so the agent can distinguish losses from gains in the reward signal.
- $\alpha_i \cdot ESG_t^{(i)}$ = **Agent-specific ESG bias** (divergent signal — the source of multi-agent tension):
  - **Agent 1:** $ESG_t^{(i)} = ESG_t^{(B)}$ (uses cross-sectionally normalized Bloomberg score as its private reward signal).
  - **Agent 2:** $ESG_t^{(i)} = ESG_t^{(L)}$ (uses cross-sectionally normalized LESG score as its private reward signal).
  - **Agent 3:** $\alpha_3 \approx 0$ (Financial agent has negligible ESG bias; focuses purely on maximizing $r_t$).
- $-\beta \cdot \Delta ESG_t$ = **Shared ambiguity penalty** (penalizes all agents equally for high disagreement between agencies; $\beta = 0$ in Competitive mode).

**Hyperparameters $\alpha_i$ and $\beta$** are not learned by the algorithm — they are design choices fixed before training that encode investor preference. Because $ESG_t^{(i)}$ and $\Delta ESG_t$ are cross-sectionally normalized to $[0,1]$ while $r_t$ is a raw simple portfolio return (typically $\pm 0.01$ to $\pm 0.05$ per day), $\alpha_i$ and $\beta$ must be calibrated to prevent ESG terms from overwhelming the financial signal.

| Parameter  | Agent                   | Suggested Range | Role                                                                                                                                                                              |
| :--------- | :---------------------- | :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\alpha_1$ | Bloomberg               | $[0.1,\ 1.0]$   | Weight on $ESG^{(B)}$ relative to $r_t$ (Portfolios A, C)                                                                                                                         |
| $\alpha_2$ | LESG                    | $[0.1,\ 1.0]$   | Weight on $ESG^{(L)}$ relative to $r_t$ (Portfolios A, C)                                                                                                                         |
| $\alpha_3$ | Financial               | $\approx 0$     | Negligible ESG bias                                                                                                                                                               |
| $\beta$    | All (Cooperative/Mixed) | $[0.1,\ 1.0]$   | Strength of shared disagreement penalty (Portfolio C)                                                                                                                             |
| $\lambda$  | All                     | $[0.1,\ 1.0]$   | Signed disagreement sensitivity (Portfolio B only) — Agent 1 is rewarded where $ESG^{(B)}>ESG^{(L)}$, Agent 2 where $ESG^{(L)}>ESG^{(B)}$; distinct from discount factor $\gamma$ |

**Tuning method:** Grid search over the suggested ranges, evaluated on the validation period. Model selection uses Sharpe Ratio as the primary metric and $\mu ESG$ as a secondary constraint.

### Portfolio Model Definitions

Three distinct portfolio models are produced by varying the reward function parameters. Each model runs under all three interaction topologies (Cooperative, Competitive, Mixed), yielding **3 × 3 = 9 portfolio configurations** per training run. For a given user query specifying a portfolio model, the system returns three side-by-side panels — one per topology.

| Model           | Full Reward Function                                                                                                                            | Parameter Set                                               | Hypothesis                                                                                         |
| :-------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| **Portfolio A** | $R_t^{(i)} = r_t + \alpha_i \cdot ESG_t^{(i)}$                                                                                                  | $\beta = 0$; $\alpha_1, \alpha_2 > 0$; $\alpha_3 \approx 0$ | Does ESG consensus signal add alpha beyond pure financial return?                                  |
| **Portfolio B** | $R_t^{(B)} = r_t + \lambda \cdot (ESG_t^{(B)} - ESG_t^{(L)})$; $R_t^{(L)} = r_t + \lambda \cdot (ESG_t^{(L)} - ESG_t^{(B)})$; $R_t^{(F)} = r_t$ | $\alpha_i = 0$; $\lambda > 0$                               | Is ESG disagreement a tradeable signal when each agent bets its own source is correct?             |
| **Portfolio C** | $R_t^{(i)} = r_t + \alpha_i \cdot ESG_t^{(i)} - \beta \cdot \Delta ESG_t$                                                                       | $\alpha_1, \alpha_2 > 0$; $\alpha_3 \approx 0$; $\beta > 0$ | Does combining ESG consensus with uncertainty penalisation maximise risk-adjusted ESG performance? |

Portfolio A tests ESG consensus in isolation. Portfolio B introduces genuine multi-agent tension via signed disagreement: Bloomberg Agent rewards stocks where $ESG^{(B)} > ESG^{(L)}$ (betting Bloomberg is correct); LESG Agent rewards the opposite; Financial Agent remains source-agnostic. Note that $ESG^{(B)} - ESG^{(L)} = -(ESG^{(L)} - ESG^{(B)})$, so agents 1 and 2 hold directly opposing views on every stock.

Two design properties of Portfolio B must be understood by implementers:

1. **Degenerate agreement case:** When the portfolio-weighted aggregate scores are equal — $ESG_t^{(B)} = ESG_t^{(L)}$, i.e. $\sum_i w_{i,t} \cdot ESG^{(B)}_\text{norm}(i,t) = \sum_i w_{i,t} \cdot ESG^{(L)}_\text{norm}(i,t)$ — both signed terms in the reward equal zero and all three agents receive $r_t$ only, collapsing to a pure financial baseline. Note: individual stocks with $ESG^{(B)}_\text{norm}(i) = ESG^{(L)}_\text{norm}(i)$ contribute zero to the signed difference but do not alone trigger this collapse — only when the portfolio-weighted sum agrees does the full degenerate condition apply. The Financial Agent's score is the tie-breaking signal at such points.

2. **Partial cancellation under aggregation:** In $z^{joint} = (z^{(B)} + z^{(L)} + z^{(F)})/3$, Agents 1 and 2 contribute opposing ESG-driven scores that partially cancel. The net portfolio signal for Portfolio B is therefore driven primarily by Agent 3 (financial return) plus whatever asymmetry exists in how the ESG disagreement shapes each agent's full value function during training. Portfolio B is not a symmetric-cancellation baseline — each agent trains a different Q-function — but the action-level ESG contributions cancel more than in any other portfolio model.

Portfolio C is the full model combining both signals.

---

_Why agent-specific ESG signals matter:_ If all agents used the same consensus signal ($\mu ESG_t$), they would receive identical rewards and converge to the same policy, eliminating the benefit of the multi-agent architecture. The divergent $ESG_t^{(i)}$ signals force each agent to negotiate from genuinely different perspectives, producing a richer and more robust joint portfolio decision. $\mu ESG_t$ remains available as an observed state feature but does not enter the reward function directly.

### Understanding the Trade-Off

The reward function creates a direct trade-off: ESG disagreement penalises the reward signal, but high-disagreement stocks may also carry higher financial returns. The algorithm handles this dynamically:

_(The values below are stylized for illustration — scaled to show the trade-off mechanism clearly, not representative of raw daily simple-return magnitudes.)_

- If a stock yields a strong simple return ($r_t = +0.04$) but has high ESG disagreement causing a penalty ($-\beta \cdot \Delta ESG_t = -0.025$, e.g. $\beta=0.5$, $\Delta ESG_t=0.05$), the net reward is still positive ($+0.015$). The agent **increases the stock's allocation weight**.
- However, if the return is modest ($r_t = +0.01$) and the penalty remains ($-0.025$), the net reward becomes negative ($-0.015$). The agent **decreases the stock's allocation weight**.
- _Conclusion:_ The AI continuously calculates whether the financial return is large enough to justify the "Ambiguity Risk." The exact crossover point is determined by the calibrated values of $\alpha_i$ and $\beta$.

---

## 4. Implementation Example: Data Sheets

To illustrate how the system transforms real-world data into actionable agent decisions, we examine a 4-stock sample universe.

### Table 1: Raw Market Data Input

_This is the external data fetched via APIs before processing. The AI uses the full OHLCV matrix to understand daily price action, liquidity, and volatility._

| Date       | ISIN           | Company name | Sector  | Open    | High    | Low     | **Close**   | **Volume** | RSI | Bloom. ESG (0-100) | LESG ESG (0-10) |
| :--------- | :------------- | :----------- | :------ | :------ | :------ | :------ | :---------- | :--------- | :-- | :----------------- | :-------------- |
| 01-01-2010 | **US03783...** | Apple        | Tech    | $148.50 | $151.20 | $148.00 | **$150.00** | **50.5M**  | 65  | 90                 | 9.0             |
| 02-01-2010 | **US03783...** | Apple        | Tech    | $148.50 | $151.20 | $148.00 | **$150.00** | **50.5M**  | 65  | 90                 | 9.0             |
| 03-01-2010 | **US03783...** | Apple        | Tech    | $148.50 | $151.20 | $148.00 | **$150.00** | **50.5M**  | 65  | 90                 | 9.0             |
| 01-01-2010 | **US30303...** | Microsoft    | Energy  | $43.00  | $45.50  | $42.80  | **$45.00**  | **12.2M**  | 78  | 95                 | 4.5             |
| 02-01-2010 | **US30303...** | Microsoft    | Energy  | $43.00  | $45.50  | $42.80  | **$45.00**  | **12.2M**  | 78  | 95                 | 4.5             |
| 03-01-2010 | **US30303...** | Microsoft    | Energy  | $43.00  | $45.50  | $42.80  | **$45.00**  | **12.2M**  | 78  | 95                 | 4.5             |
| 01-01-2010 | **US47816...** | Tesla        | Health  | $208.00 | $211.50 | $207.00 | **$210.00** | **8.1M**   | 55  | 80                 | 7.8             |
| 02-01-2010 | **US47816...** | Tesla        | Health  | $208.00 | $211.50 | $207.00 | **$210.00** | **8.1M**   | 55  | 80                 | 7.8             |
| 03-01-2010 | **US47816...** | Tesla        | Health  | $208.00 | $211.50 | $207.00 | **$210.00** | **8.1M**   | 55  | 80                 | 7.8             |
| 01-01-2010 | **GB00071...** | Alibaba      | Finance | $106.50 | $107.00 | $104.50 | **$105.00** | **25.0M**  | 45  | 60                 | 6.0             |
| 02-01-2010 | **GB00071...** | Alibaba      | Finance | $106.50 | $107.00 | $104.50 | **$105.00** | **25.0M**  | 45  | 60                 | 6.0             |
| 03-01-2010 | **GB00071...** | Alibaba      | Finance | $106.50 | $107.00 | $104.50 | **$105.00** | **25.0M**  | 45  | 60                 | 6.0             |

_(Note: MACD is omitted from this single-day snapshot as it requires multi-day historical windows to compute. It is included in the full normalized state vector during training.)_

### Table 2: Final Processed Datasheet (Agent Calculation Layer)

_After Min-Max normalization (0.0–1.0) applied uniformly to all state features, the RL agents read a unified **"State Space Vector"** instead of raw dollars and share counts._

_Important note on OHLCV normalization: As specified in Section 1.B, OHLCV features are normalized time-series-wise — per asset, over the training window. The normalized values therefore reflect where each stock's current price sits within **its own** historical range, not relative to other stocks. Energy's low OHLCV values [0.01...0.02] indicate that its current price is near the bottom of Energy's own historical trading range during this window — not that it has a low price compared to other stocks. Energy's high normalized return (0.95) is computed separately from its price position. The normalization formula applied per asset is:_

$$\text{Norm}(x_{i,t}) = \frac{x_{i,t} - \min_{\tau \in W} x_{i,\tau}}{\max_{\tau \in W} x_{i,\tau} - \min_{\tau \in W} x_{i,\tau}}$$

_where $i$ = asset, $t$ = current day, $W$ = training window (per-asset, time-series axis)._

| Date       | ISIN           | Company   | Sector  | Norm. Return (state feature) | **Norm. OHLCV Vector** (State Space) | **Norm. RSI** | **Norm. $ESG^{(B)}$** | **Norm. $ESG^{(L)}$** | **Consensus ($\mu ESG$)** | **Disagreement ($\Delta ESG$)** | Weight — Cooperative Mode |
| :--------- | :------------- | :-------- | :------ | :--------------------------- | :----------------------------------- | :------------ | :-------------------- | :-------------------- | :------------------------ | :------------------------------ | :------------------------ |
| 01-01-2010 | **US03783...** | Apple     | Tech    | 0.65                         | `[0.64, 0.66, 0.64, 0.65, 1.00]`     | **0.65**      | **0.86**              | **1.00**              | **0.93**                  | **0.14**                        | **40%**                   |
| 02-01-2010 | **US03783...** | Apple     | Tech    | 0.65                         | `[0.64, 0.66, 0.64, 0.65, 1.00]`     | **0.65**      | **0.86**              | **1.00**              | **0.93**                  | **0.14**                        | **40%**                   |
| 03-01-2010 | **US03783...** | Apple     | Tech    | 0.65                         | `[0.64, 0.66, 0.64, 0.65, 1.00]`     | **0.65**      | **0.86**              | **1.00**              | **0.93**                  | **0.14**                        | **40%**                   |
| 01-01-2010 | **US30303...** | Microsoft | Energy  | 0.95                         | `[0.01, 0.03, 0.00, 0.02, 0.25]`     | **0.78**      | **1.00**              | **0.00**              | **0.50**                  | **1.00**                        | **15%**                   |
| 02-01-2010 | **US30303...** | Microsoft | Energy  | 0.95                         | `[0.01, 0.03, 0.00, 0.02, 0.25]`     | **0.78**      | **1.00**              | **0.00**              | **0.50**                  | **1.00**                        | **15%**                   |
| 03-01-2010 | **US30303...** | Microsoft | Energy  | 0.95                         | `[0.01, 0.03, 0.00, 0.02, 0.25]`     | **0.78**      | **1.00**              | **0.00**              | **0.50**                  | **1.00**                        | **15%**                   |
| 01-01-2010 | **US47816...** | Tesla     | Health  | 0.35                         | `[0.98, 1.00, 0.98, 0.99, 0.10]`     | **0.55**      | **0.57**              | **0.73**              | **0.65**                  | **0.16**                        | **35%**                   |
| 02-01-2010 | **US47816...** | Tesla     | Health  | 0.35                         | `[0.98, 1.00, 0.98, 0.99, 0.10]`     | **0.55**      | **0.57**              | **0.73**              | **0.65**                  | **0.16**                        | **35%**                   |
| 03-01-2010 | **US47816...** | Tesla     | Health  | 0.35                         | `[0.98, 1.00, 0.98, 0.99, 0.10]`     | **0.55**      | **0.57**              | **0.73**              | **0.65**                  | **0.16**                        | **35%**                   |
| 01-01-2010 | **GB00071...** | Alibaba   | Finance | 0.15                         | `[0.45, 0.46, 0.43, 0.44, 0.60]`     | **0.45**      | **0.00**              | **0.33**              | **0.17**                  | **0.33**                        | **10%**                   |
| 02-01-2010 | **GB00071...** | Alibaba   | Finance | 0.15                         | `[0.45, 0.46, 0.43, 0.44, 0.60]`     | **0.45**      | **0.00**              | **0.33**              | **0.17**                  | **0.33**                        | **10%**                   |
| 03-01-2010 | **GB00071...** | Alibaba   | Finance | 0.15                         | `[0.45, 0.46, 0.43, 0.44, 0.60]`     | **0.45**      | **0.00**              | **0.33**              | **0.17**                  | **0.33**                        | **10%**                   |

_(Note on Table 2 state logic: The Agent reads the `Norm. OHLCV Vector` to understand the environment. For Apple (US03783...), the Close (0.65) is higher than the Open (0.64), and Volume is maxed at 1.00 — indicating high-liquidity upward momentum. For Tesla (US47816...), the high absolute price generates normalized OHLCV values near 1.0, while volume at 0.10 signals lower liquidity. The Norm. $ESG^{(B)}$ and Norm. $ESG^{(L)}$ columns are cross-sectionally normalized on the same day $t$ across all four stocks — μESG and ΔESG are derived directly from these two columns.)_

_(Weights shown are for the Cooperative topology. Competitive and Mixed topology outputs are presented in Section 5.)_

---

## 5. Agent Output: User Query & Logic Breakdown

**User Query:**
_"I have $10,000,000 to allocate. Generate an optimal portfolio using the 'Consensus & Uncertainty' model (Portfolio C) that maximizes financial returns while strictly managing the disagreement between Bloomberg and LESG ratings."_

The system runs all three interaction topologies simultaneously. The user receives three independent portfolio recommendations side by side and can compare them directly. No response is merged into another.

---

### Panel 1 — Cooperative Mode (Team Negotiation)

_Agents share a joint reward signal and negotiate toward collective consensus. All three agents are penalized equally by the shared ΔESG ambiguity term._

_Return = annualised simple return; Risk = annualised standard deviation; Sharpe = Return / Risk ($r_f = 0$). In the live system these are computed from a rolling out-of-sample validation window of **63 trading days** (one calendar quarter), not the training window. Values shown here are illustrative._

| ISIN           | Sector  | Return (ann.) | Risk ($\sigma$) | Sharpe Ratio | $\mu ESG$ | $\Delta ESG$ | Weight  | Allocation     |
| :------------- | :------ | :------------ | :-------------- | :----------- | :-------- | :----------- | :------ | :------------- |
| **US03783...** | Tech    | 0.22          | 0.12            | 1.83         | 0.93      | 0.14         | **40%** | **$4,000,000** |
| **US47816...** | Health  | 0.14          | 0.15            | 0.93         | 0.65      | 0.16         | **35%** | **$3,500,000** |
| **US30303...** | Energy  | 0.38          | 0.28            | 1.36         | 0.50      | 1.00         | **15%** | **$1,500,000** |
| **GB00071...** | Finance | 0.09          | 0.10            | 0.90         | 0.17      | 0.33         | **10%** | **$1,000,000** |

> _Strategic Summary: Tech dominates with the highest Sharpe (1.83) and lowest ESG disagreement (ΔESG=0.14). Energy carries the highest raw return (0.38) but maximum disagreement (ΔESG=1.00) — the shared penalty $-\beta \cdot \Delta ESG = -\beta \cdot 1.00$ makes all agents collectively bear the full ambiguity cost, suppressing it to 15%._

---

### Panel 2 — Competitive Mode (Non-cooperative Negotiation)

_Each agent maximizes its own private objective; there is no cooperative penalty mechanism ($\beta = 0$)._

| ISIN           | Sector  | Return (ann.) | Risk ($\sigma$) | Sharpe Ratio | $\mu ESG$ | $\Delta ESG$ | Weight  | Allocation     |
| :------------- | :------ | :------------ | :-------------- | :----------- | :-------- | :----------- | :------ | :------------- |
| **US03783...** | Tech    | 0.22          | 0.12            | 1.83         | 0.93      | 0.14         | **35%** | **$3,500,000** |
| **US47816...** | Health  | 0.14          | 0.15            | 0.93         | 0.65      | 0.16         | **25%** | **$2,500,000** |
| **US30303...** | Energy  | 0.38          | 0.28            | 1.36         | 0.50      | 1.00         | **30%** | **$3,000,000** |
| **GB00071...** | Finance | 0.09          | 0.10            | 0.90         | 0.17      | 0.33         | **10%** | **$1,000,000** |

> _Strategic Summary: Energy allocation rises to 30% — Bloomberg Agent ($ESG^{(B)}_\text{norm}(\text{Energy},t)=1.00$, maximum per-stock normalized score) and Financial Agent (highest return 0.38) independently reach aligned strategies favouring Energy. LESG Agent ($ESG^{(L)}_\text{norm}(\text{Energy},t)=0.00$, minimum per-stock normalized score) opposes the position, but with no shared penalty mechanism its objection is diluted in the aggregation step. The portfolio accepts maximum ambiguity risk (ΔESG=1.00) in exchange for the highest available return._

---

### Panel 3 — Mixed Mode (General-Sum Negotiation)

_Agents pursue partly shared, partly conflicting goals. Partial ESG cooperation between Bloomberg and LESG Agents moderates the Financial Agent's risk appetite — producing an intermediate outcome between Panels 1 and 2._

| ISIN           | Sector  | Return (ann.) | Risk ($\sigma$) | Sharpe Ratio | $\mu ESG$ | $\Delta ESG$ | Weight  | Allocation     |
| :------------- | :------ | :------------ | :-------------- | :----------- | :-------- | :----------- | :------ | :------------- |
| **US03783...** | Tech    | 0.22          | 0.12            | 1.83         | 0.93      | 0.14         | **38%** | **$3,800,000** |
| **US47816...** | Health  | 0.14          | 0.15            | 0.93         | 0.65      | 0.16         | **30%** | **$3,000,000** |
| **US30303...** | Energy  | 0.38          | 0.28            | 1.36         | 0.50      | 1.00         | **22%** | **$2,200,000** |
| **GB00071...** | Finance | 0.09          | 0.10            | 0.90         | 0.17      | 0.33         | **10%** | **$1,000,000** |

> _Strategic Summary: Energy receives 22% — above Cooperative (15%) because the penalty is only partial, but well below Competitive (30%) because Bloomberg and LESG Agents retain partial coordination against Energy's maximum disagreement score (ΔESG=1.00)._

---

### Behind-the-Scenes Logic (Why the Three Panels Differ)

The same normalized state vector — the same ISIN data, same OHLCV, same ESG scores — is fed into all three topologies. The allocation differences emerge entirely from how the agents **negotiate** with each other under each reward structure:

1.  **State Observation (Shared Input):** All three topologies read the same normalized RSI, OHLCV, μESG, and ΔESG values. The Energy stock's RSI of 0.78 and return of 0.95 are visible to every agent in every mode.
2.  **Reward Structure Divergence:** In Cooperative mode, the $-\beta \cdot \Delta ESG$ penalty is applied to the shared joint reward — all agents feel the full cost of backing Energy (ΔESG=1.00, maximum). In Competitive mode, no shared penalty exists; the Bloomberg Agent's private reward from $ESG^{(B)}_\text{norm}(\text{Energy},t)=1.00$ (maximum per-stock normalized Bloomberg score) and the Financial Agent's high return signal independently favour Energy, while the LESG Agent's $ESG^{(L)}_\text{norm}(\text{Energy},t)=0.00$ (minimum per-stock normalized LESG score) is diluted in aggregation.
3.  **Score Aggregation → Softmax Output:** Each agent outputs its own allocation score vector ($z_t^{(B)}$, $z_t^{(L)}$, $z_t^{(F)}$). These are equal-weight averaged into a single joint score vector $z_t^{joint} = (z_t^{(B)} + z_t^{(L)} + z_t^{(F)}) / 3$, which is passed through one **Softmax layer** to produce the final portfolio weights $[w_1, w_2, w_3, w_4]$ summing to 1.0. Different negotiation dynamics → different per-agent score vectors → different joint averages → different final weight distributions.
4.  **User Comparison:** The user reads all three panels side by side. The difference in Energy weight (15% vs 30% vs 22%) directly illustrates the impact of ESG agency disagreement on portfolio construction under each game-theoretic framework.
