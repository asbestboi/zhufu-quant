# Project Spec: Personal Quant Research — Step 2: Backtesting Engine

> Hand-off document for Claude Code. This extends the **existing `zhufu-quant` repo** built in Step 1 (data pipeline). Read the Step 1 code and README first, reuse its loading API, and ask before deviating from the stack or scope.

## 1. Context

Roadmap:

1. Data pipeline — **done** (`qr.load.get_bars`, DuckDB + Parquet, validation)
2. **Backtesting engine** ← this spec
3. Strategy research and validation (walk-forward, out-of-sample, overfitting controls)
4. Paper trading, then small live trading via a broker API (Interactive Brokers or Alpaca)

The design borrows good ideas from QUANTAXIS (a unified account format shared by backtest, paper, and live trading; a Python front end with an optional Rust core) but deliberately avoids its heavy infrastructure (no MongoDB, RabbitMQ, web servers, or microservices). This is a single-user research tool.

**Priorities, in order:** correctness → no look-ahead bias → realistic costs → reproducibility → speed.

## 2. Goal

Two engines sharing one data layer, one cost model, one account format, and one metrics module:

- **Vectorized engine** — fast, for screening many ideas. Strategy outputs target portfolio weights per day; engine computes returns, costs, and metrics.
- **Event-driven engine** — realistic, for validating promising ideas. Strategy reacts to bars one at a time and submits orders; engine simulates fills, cash, positions, splits, and dividends.

For simple strategies, both engines must produce nearly identical results (see parity tests). This cross-check is one of the most valuable correctness guarantees in the project.

## 3. Tech Stack

Same as Step 1, plus:

- **numpy / pandas** for the vectorized engine (polars optional internally)
- **matplotlib** or **plotly** for reports (self-contained HTML output)
- **pyyaml + pydantic** for backtest configs
- Optional later: **Rust + PyO3 + maturin** for the event-loop core (see §12)

## 4. Repository Additions

```
src/qr/backtest/
├── __init__.py
├── config.py          # pydantic models for backtest configs
├── calendar.py        # trading-day handling, rebalance schedules
├── costs.py           # commission, spread, slippage, borrow models
├── account.py         # unified account format (QIFI-inspired)
├── vectorized.py      # weights-based engine
├── events.py          # event types: Bar, Order, Fill, CorporateAction
├── broker_sim.py      # simulated broker: order matching, fills
├── event_engine.py    # event loop
├── strategy.py        # Strategy base classes for both engines
├── metrics.py         # performance + risk statistics
├── report.py          # HTML tearsheet
├── registry.py        # experiment log: every run recorded
└── strategies/        # reference strategies (see §10)
    ├── buy_and_hold.py
    ├── sma_crossover.py
    └── xs_momentum.py
configs/backtests/
└── *.yaml             # one config per experiment
reports/               # gitignored, generated tearsheets
```

## 5. Core Rules Against Look-Ahead Bias

These are non-negotiable and must be enforced in code, not just documented:

- A decision made using data up to the **close of day t** can only be executed **at the open of day t+1** (default) or at the close of t+1. Never at the close of t.
- Strategies only receive data up to the current timestamp. The engine passes a **truncated view**, never the full DataFrame.
- Any rolling statistic (moving averages, volatility, z-scores) must use only past data. Provide helpers that make this the default.
- Universe membership must be decided with information available at that time (respect each symbol's first available date; no trading a symbol before it has data).

## 6. Unified Account Format (`account.py`)

Inspired by QUANTAXIS's QIFI. One serializable structure used by both engines now and by paper/live trading later.

```python
@dataclass
class AccountState:
    timestamp: datetime
    cash: float
    positions: dict[str, Position]   # symbol -> Position(quantity, avg_cost, market_value)
    open_orders: dict[str, Order]
    equity: float                     # cash + market value of positions
    gross_exposure: float
    net_exposure: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
```

- Every fill produces an immutable `Trade` record (id, symbol, side, quantity, price, fees, timestamp, order_id).
- Account state can be snapshotted to JSON at any time and restored exactly.
- The vectorized engine produces a daily series of simplified account snapshots so reports and metrics work identically for both engines.

## 7. Cost Model (`costs.py`)

Configurable per asset class. All costs are applied in **both** engines.

- **Commission:** per-share, per-trade minimum, and percentage-of-value modes (default modeled on Interactive Brokers tiered pricing; values in config, not hard-coded).
- **Spread:** half-spread cost as basis points, configurable per symbol or asset class.
- **Slippage:** basis points, plus an optional volume-participation model (cost grows with order size relative to average daily volume).
- **Short borrow fee:** annualized rate on short market value, accrued daily.
- **Participation cap:** orders larger than a configurable % of average daily volume are partially filled or rejected (configurable).

Every report must show results **before and after costs**.

## 8. Vectorized Engine (`vectorized.py`)

```python
class WeightStrategy(ABC):
    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame: ...
    # returns a (date x symbol) DataFrame of target weights, computed only from past data
```

- Uses **total-return adjusted** prices from `qr.load.get_bars`.
- Engine shifts weights by one bar internally (decision at t → holding from t+1), so strategies cannot accidentally peek.
- Supports rebalance schedules: daily, weekly, monthly, or a custom date list. Between rebalances, weights drift with prices.
- Computes turnover from weight changes and applies costs from `costs.py`.
- Constraints: max gross leverage, max weight per symbol, long-only flag.
- Target: a 10-year daily backtest over 50 symbols runs in under 2 seconds.

## 9. Event-Driven Engine (`event_engine.py`, `broker_sim.py`)

```python
class EventStrategy(ABC):
    def on_start(self, ctx: Context) -> None: ...
    def on_bar(self, ctx: Context, bars: dict[str, Bar]) -> None: ...
    def on_fill(self, ctx: Context, trade: Trade) -> None: ...
    def on_end(self, ctx: Context) -> None: ...
# Context exposes: account state, history(symbol, n) (past only), order methods
```

- Uses **unadjusted** prices plus corporate actions from Step 1: splits change position quantity and cost basis; dividends are credited as cash (and debited on short positions). This matches what a real broker does.
- Order types: market (fills at next bar open), limit (fills if next bar's range crosses the limit, at the limit price or better), market-on-close.
- Helper: `ctx.order_target_weight(symbol, w)` and `ctx.order_target_quantity(symbol, q)`.
- Fills respect the cost model and participation cap.
- Deterministic: same config + same data = identical output, bit for bit.
- Design the event loop behind a small interface (`EngineCore`) so a Rust implementation can replace it later without changing strategies.

## 10. Reference Strategies

Implement each for **both** engines where practical:

- **Buy and hold** (single symbol and equal-weight basket)
- **SMA crossover** (e.g. 50/200 on SPY, long or flat)
- **Cross-sectional momentum** (rank by 12-1 month return, hold top N, monthly rebalance)

These are not expected to be profitable after costs. They exist to validate the engines.

## 11. Metrics and Reporting

`metrics.py` — computed from the daily equity curve and trade list:

- Total return, CAGR, annualized volatility
- Sharpe, Sortino, Calmar ratios
- Max drawdown, drawdown duration, time underwater
- Turnover (annualized), average holding period, exposure over time
- Hit rate, average win/loss, profit factor (event engine)
- Beta and correlation to a benchmark (default SPY), alpha
- **Probabilistic Sharpe Ratio** (López de Prado), with a note that it doesn't account for multiple testing (Step 3 will add the Deflated Sharpe Ratio using the registry)
- Rolling 1-year Sharpe and volatility

`report.py` — one self-contained HTML tearsheet per run: equity curve vs. benchmark (log scale), drawdown chart, monthly returns heatmap, rolling metrics, before/after cost comparison, metrics table, config used, and run metadata.

## 12. Experiment Registry (`registry.py`)

Every backtest run is logged to DuckDB automatically, **including failed and discarded ideas**:

```sql
CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id TEXT PRIMARY KEY, created_at TIMESTAMP, strategy TEXT, engine TEXT,
  config_json TEXT, config_hash TEXT, git_commit TEXT, git_dirty BOOLEAN,
  data_snapshot TEXT,      -- latest ingested_at of data used
  start_date DATE, end_date DATE,
  sharpe DOUBLE, cagr DOUBLE, max_drawdown DOUBLE, turnover DOUBLE,
  report_path TEXT, notes TEXT
);
```

Why this matters: the number of ideas tried is needed later to correct for overfitting (Deflated Sharpe Ratio). Without a full log, a lucky backtest looks like skill.

## 13. CLI Additions

```
qr backtest run configs/backtests/sma_spy.yaml     # run + write report + log to registry
qr backtest compare RUN_ID RUN_ID ...              # side-by-side metrics table
qr backtest list [--strategy NAME]                 # recent runs from the registry
```

Example config:

```yaml
name: sma_crossover_spy
engine: vectorized        # or: event
strategy: sma_crossover
params: { fast: 50, slow: 200 }
universe: [SPY]
start: 2012-01-01
end: 2024-12-31
initial_cash: 10000
benchmark: SPY
execution: next_open
rebalance: daily
costs:
  commission_model: ibkr_tiered
  spread_bps: 1
  slippage_bps: 2
constraints: { long_only: true, max_gross_leverage: 1.0 }
```

## 14. Testing Requirements

All offline, using fixtures from Step 1 plus small synthetic datasets.

- **Hand-calculated cases:** a 5-day synthetic series with known trades; assert exact cash, positions, fees, and equity each day.
- **Look-ahead test:** run a strategy, then randomly perturb all data **after** date t and rerun; every decision and equity value up to t must be unchanged.
- **Engine parity test:** buy-and-hold and SMA crossover produce equity curves within a small tolerance across both engines (differences only from rounding to whole shares).
- **Benchmark sanity:** buy-and-hold SPY with zero costs matches SPY's total-return series from `get_bars`.
- **Zero-signal test:** a strategy that never trades ends with exactly the starting cash.
- **Corporate actions:** a fixture with a 2:1 split and a dividend; event engine positions, cash, and equity match hand-calculated values.
- **Cost tests:** each commission, spread, slippage, and borrow model against hand-calculated values.
- **Determinism:** identical runs produce identical outputs.
- **Metrics:** each metric checked against a known series (e.g. constant-return series has known Sharpe and zero drawdown).

`ruff`, `mypy`, and `pytest` must all pass.

## 15. Acceptance Criteria

- [ ] `qr backtest run` works for all three reference strategies with both engines
- [ ] Each run writes an HTML tearsheet and a registry row
- [ ] Look-ahead, parity, benchmark-sanity, and corporate-action tests pass
- [ ] Reports show before- and after-cost results
- [ ] Vectorized 10-year / 50-symbol daily backtest runs in under 2 seconds
- [ ] README updated: engine overview, how to write a strategy for each engine, config reference, known limitations
- [ ] `notebooks/02_backtest_walkthrough.ipynb` runs SMA crossover on both engines and compares results

## 16. Optional Stretch: Rust Core

Only after everything above passes:

- Implement `EngineCore` in Rust (`rust/qr_core/`, built with maturin, exposed via PyO3).
- Python remains the strategy and research layer; Rust handles the event loop, order matching, and account updates.
- **Required:** the Rust core must pass the exact same test suite, plus a parity test against the Python core on all reference strategies.
- Python core stays as the fallback and reference implementation.
- NautilusTrader (Python + Rust, PyO3) is a good open-source reference for this architecture.

## 17. Out of Scope

- Walk-forward optimization, parameter search, purged cross-validation, Deflated Sharpe (Step 3)
- Intraday data, tick-level simulation, order book modeling
- Futures, options, margin calls, FX conversion (single base currency, USD)
- Live or paper broker connections (Step 4 — but the account format must be reusable there)
- Machine learning models

## 18. Working Style

- Build in this order: account + costs → vectorized engine → metrics + report → registry → event engine → parity tests → CLI.
- Commit in small, logical steps; keep code readable over clever.
- Docstrings and type hints on all public functions.
- When unsure, choose the more conservative and more pessimistic assumption (higher costs, later fills) and note it in the README under "Design decisions". A backtest should be harder to pass than real trading, not easier.
