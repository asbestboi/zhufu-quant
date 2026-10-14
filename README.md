# zhufu-quant

[![CI](https://github.com/asbestboi/zhufu-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/asbestboi/zhufu-quant/actions/workflows/ci.yml)
![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What this is

A personal quantitative research toolkit, built step by step as a learning project. The
Python package is called `qr`. So far it has two parts:

- A **market data pipeline**: it downloads daily OHLCV bars plus splits and dividends for
  a configurable list of stocks and ETFs, checks them for errors, stores them locally in
  DuckDB (with an immutable Parquet snapshot of every download), and offers a small
  Python API and CLI for loading them, adjusted or not.
- A **backtester** with two engines on top of that data: a fast vectorized one for
  screening ideas and a realistic event-driven one for validating them, with shared
  costs, metrics, HTML reports, and a log of every experiment. See
  [Backtesting](#backtesting).

Later steps (strategy research, paper trading) will build on both.

Priorities, in order: correctness, no look-ahead bias, realistic costs, reproducibility.
Speed and features come after.

> **Disclaimer.** This project is for research and education only. Nothing in it is
> financial advice, and it comes with no warranty of any kind (see [LICENSE](LICENSE)).
> Market data is downloaded by you, to your machine, from third-party providers whose
> terms of use apply; this repository does not contain or redistribute their data.

## Roadmap

| Step | What | Status |
| --- | --- | --- |
| 1 | Market data pipeline: ingest, validate, store, load | Done |
| 1.1 | Pipeline polish: CI, fail-fast ingestion, cross-checking, named universes | Done |
| 2 | Backtesting engine (vectorized and event-driven) | Done |
| 3 | Strategy research, starting with a pairs-trading study | Planned |
| 4 | Paper trading, then possibly live trading | Planned |

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync                # install everything (add --extra crypto for the ccxt source)
uv run qr init         # create data/market.duckdb and its tables
uv run qr ingest       # download the default universe (about a minute the first time)
uv run qr status       # see what you have
```

On Windows, if `uv` is not on your PATH but was installed with pip, use `py -m uv` wherever
this page says `uv`.

Data lives in `./data` (gitignored). Set `QR_DATA_DIR` to move it and `QR_CONFIG` to use
a different universe file; every CLI command also takes `--data-dir`, and the ones that
read the universe take `--config`. The CLI reads a `.env` file from the folder it is run
in (real environment variables win); see `.env.example`.

## CLI

| Command | What it does |
| --- | --- |
| `qr init` | Create the database and tables. Safe to repeat. |
| `qr ingest [--symbols X,Y \| --universe NAME]` | Incremental download, then quality checks. Exits non-zero if any symbol failed or came back incomplete, or a source was unreachable. |
| `qr validate [--symbols X,Y \| --universe NAME]` | Re-run the quality checks and print a summary of open findings. Exits non-zero if any finding has severity `error`. |
| `qr status [--universe NAME]` | Per-symbol coverage: first/last date, row count, open errors and warnings. |
| `qr crosscheck [--symbols X,Y \| --universe NAME] [--days 365]` | Compare stored closes and splits with a second source (Tiingo). See [Cross-checking](#cross-checking-against-a-second-source). |
| `qr export SYMBOL --out f.csv [--adjustment none\|split\|total_return]` | Dump bars to CSV (total-return adjusted by default). |
| `qr backtest run CONFIG.yaml [--notes TEXT] [--reports-dir DIR]` | Run a backtest, write its HTML tearsheet, log it to the registry. Exits non-zero (and logs the run as failed) if it could not run. |
| `qr backtest list [--strategy NAME] [--limit 20]` | Recent runs from the registry. |
| `qr backtest compare RUN_ID RUN_ID ...` | Side-by-side metrics of logged runs; unique id prefixes are enough. |

Add `-v` before the command for debug logs: `qr -v ingest`. `--universe` takes the name
of a group from the config's `groups:` section; using it together with `--symbols` is an
error.

### Running it daily

Ingestion is incremental and idempotent, so run it whenever you like. It only stores
completed sessions, so schedule it after the US close (21:00 UTC in winter, 20:00 in
summer) and leave Yahoo some time to settle.

#### Linux and macOS

With cron, on weekdays at 23:30 UTC:

```
30 23 * * 1-5  cd /path/to/zhufu-quant && uv run qr ingest >> data/ingest.log 2>&1
```

Or as a systemd timer (`~/.config/systemd/user/`):

```ini
# qr-ingest.service
[Service]
Type=oneshot
WorkingDirectory=/path/to/zhufu-quant
ExecStart=/usr/bin/env uv run qr ingest

# qr-ingest.timer
[Timer]
OnCalendar=Mon..Fri 23:30 UTC
Persistent=true
[Install]
WantedBy=timers.target
```

Enable with `systemctl --user enable --now qr-ingest.timer`.

#### Windows

Two PowerShell scripts in `scripts/windows/` set up Task Scheduler:

```powershell
# Register a task that runs on weekdays at 23:30 local time (pick another with -Time 22:45)
powershell -ExecutionPolicy Bypass -File scripts\windows\register_task.ps1

Start-ScheduledTask -TaskName qr-daily-ingest    # try it right now
Get-Content logs\ingest.log -Tail 40             # see what it did

# Remove the task again
powershell -ExecutionPolicy Bypass -File scripts\windows\register_task.ps1 -Remove
```

The task runs `run_daily_ingest.ps1`, which changes to the repo folder, runs
`uv run qr ingest` and then `uv run qr validate`, and appends all output to
`logs\ingest.log` (gitignored). It runs as you and only while you are logged on, so it
needs no administrator rights; if the PC was off at the scheduled time it runs as soon as
it can afterwards. Running the register script again replaces the task, which is how you
change the time. Pick a time after the US close (16:00 New York): 23:30 suits Europe.

## Loading data in Python

`qr.load` is the only module later steps should import (the backtester reads all its
data through it). Besides the functions below it has `get_corporate_actions(symbols,
end=None)` (splits and dividends as a table), `get_asset_classes(symbols)`, and
`data_version(symbols, end=None)` (when the newest of those bars was last written).

```python
from datetime import date, datetime
from qr.load import get_bars, get_returns, list_symbols

bars = get_bars("SPY")  # total-return adjusted, full history
bars = get_bars(["SPY", "TLT"], start=date(2020, 1, 1), end=date(2024, 12, 31))
raw = get_bars("AAPL", adjustment="none")  # exactly as traded
wide = bars["close"].unstack("symbol")  # date x symbol table

rets = get_returns(["SPY", "TLT"], kind="simple")  # wide table of daily total returns
list_symbols(active_only=True)
```

### `get_bars(symbols, start=None, end=None, adjustment="total_return", as_of=None)`

Returns a DataFrame indexed by a `(date, symbol)` MultiIndex, sorted by date then symbol,
with float64 columns `open, high, low, close, volume`.

- `date` is a timezone-naive `datetime64[ns]` at midnight, meaning "the trading session
  on that calendar day" (exchange-local for equities, UTC for crypto).
- `start` and `end` are both inclusive.
- `adjustment`:
  - `"none"`: prices and volumes as traded.
  - `"split"`: backward-adjusted for splits (prices divided, volumes multiplied).
  - `"total_return"`: also adjusted for cash dividends, so close-to-close changes are
    total returns. Volumes are split-adjusted only.
- Adjustment is **anchored at the last returned bar** and only uses corporate actions
  dated on or before `end`. With `end` set, the newest bar equals its unadjusted value
  and nothing that happened later influences the result. Adjusted price *levels*
  therefore depend on `end`; returns do not.
- Unknown symbols raise `KeyError`; a missing database raises `qr.storage.StorageError`.

### Point-in-time loading (`as_of`)

`get_bars("SPY", as_of=datetime(2026, 11, 1))` rebuilds the data from the raw snapshots
downloaded at or before that moment (naive datetimes are UTC; aware ones are converted).
Each bar takes its value from the latest such snapshot, and adjustment uses the corporate
actions known in that snapshot.

Limits:

- It only goes back to **your first download**. The history fetched that day is whatever
  the provider believed then, including any corrections it had already applied.
- It needs the files under `data/raw/` to still exist. Do not delete them.
- Its resolution is the time of each download, not the time the provider published.
- The instruments table is not versioned: which symbols exist and their active flag are
  always the current ones.

### `get_returns(symbols, start=None, end=None, kind="log")`

Wide (date x symbol) DataFrame of daily close-to-close **total** returns, `"log"` or
`"simple"`. Each symbol's return is computed across its own consecutive bars, is NaN on
its first bar and on days it has no bar, and rows where every symbol is NaN are dropped.

### `list_symbols(active_only=False)`

Sorted list of symbols in the store.

## Configuration

`config/universe.yaml` lists the instruments and settings. It is validated with pydantic
and any problem (unknown key, duplicate symbol, bad date, a crypto symbol on the Yahoo
source, ...) stops the command with a message naming the field.

```yaml
start_date: 2010-01-01
sources:
  yahoo:  { request_delay_seconds: 1.0, max_retries: 5 }
  crypto: { enabled: false, exchange: kraken }
ingest:
  max_consecutive_failures: 3      # see "Failing fast" below
crosscheck:
  close_tolerance: 0.005           # 0.5%
  request_delay_seconds: 0.5
  max_retries: 2
validation:
  outlier_return_threshold: 0.25
  stale_after_trading_days: 5
  max_zero_volume_streak: 3
  revision_tolerance: 0.0001
universe:
  - { symbol: SPY, source: yahoo, asset_class: etf }
  - { symbol: KO,  source: yahoo, asset_class: equity }
  - { symbol: PEP, source: yahoo, asset_class: equity }
  - { symbol: BTC/USD, source: crypto, asset_class: crypto }
groups:
  core:  [SPY]
  pairs: [KO, PEP]
```

Symbols whose source is disabled are skipped, not failed. Removing a symbol from the file
keeps its data and marks it inactive.

### Named universes

`groups:` gives names to subsets of the universe. Every symbol in a group must also be
defined under `universe:`, otherwise the config is rejected with a message naming the
group and the missing symbols. Use a group from the CLI with `--universe`:

```
uv run qr ingest --universe pairs
uv run qr status --universe core
```

or from Python:

```python
from qr.config import universe_symbols
from qr.load import get_bars

bars = get_bars(universe_symbols("pairs"))
```

The default config defines `core` (SPY, QQQ, IWM, TLT, GLD, AAPL, MSFT) and `pairs` (KO,
PEP, XOM, CVX, V, MA, HD, LOW, GS, MS). A plain `qr ingest` downloads everything in
`universe:` whether or not it belongs to a group.

## Cross-checking against a second source

Yahoo is an unofficial, unguaranteed source, so `qr crosscheck` asks a second provider,
[Tiingo](https://www.tiingo.com), for the same days and reports where the two disagree.

1. Create a free Tiingo account and copy your token from
   <https://www.tiingo.com/account/api/token>.
2. Put `TIINGO_API_KEY=<your token>` in a file called `.env` in the repo folder (it is
   gitignored; `.env.example` shows the format).
3. Run `uv run qr crosscheck` (add `--universe pairs`, `--symbols KO,PEP`, or `--days 90`).

Without a key the command explains these steps and exits without an error.

For each symbol it compares the **unadjusted close** on every day both sources have, and
the **split events**, over the last `--days` calendar days (default 365). A close that
differs by more than `crosscheck.close_tolerance` (0.5%) and any split that only one
source has, or with a different ratio, is written to `quality_issues` with
`check_name = 'crosscheck'` and severity `warning`, so it shows up in `qr validate` and
`qr status`. The command prints one line per symbol: days compared, mismatches, and the
worst difference with its date. It exits non-zero only if Tiingo could not be read.

Things to know:

- Tiingo is a witness, **not** a data source for the store. Nothing it returns is written
  to `daily_bars`, and a mismatch does not say which provider is wrong.
- Each response is saved as a raw snapshot under `data/raw/tiingo/`, like Yahoo's.
- Running it again for the same window replaces that window's findings, so a difference
  that a provider has since fixed disappears.
- The free tier allows roughly 50 requests an hour and 1,000 a day; one symbol is one
  request. A rejected key or an exhausted allowance stops the run at once.

## Backtesting

Two engines share one data layer, one cost model, one account format, and one metrics
module (`qr/backtest/`):

| | Vectorized engine | Event-driven engine |
| --- | --- | --- |
| Purpose | Screen many ideas quickly | Validate the promising ones realistically |
| A strategy is | A table of target weights per day | Code that reacts to each bar and sends orders |
| Prices | Total-return adjusted | As traded, with splits and cash dividends applied to the account |
| Positions | Dollars (fractional shares) | Shares (whole by default), with cost basis and realized P&L |
| Orders | Implied by the weights | Market, limit, market-on-close |
| Speed | 10 years of 50 symbols in under a second | Seconds for a few symbols; a daily rebalance of 50 takes much longer |

Because they are independent implementations, they check each other: on the reference
strategies their equity curves agree to within the effect of rounding to whole shares
(tested in `tests/test_bt_parity_lookahead.py`). If you write a strategy for both and
they disagree, one of them has a bug.

```
uv run qr backtest run configs/backtests/sma_spy.yaml         # run, write report, log
uv run qr backtest run configs/backtests/sma_spy_event.yaml   # the same idea, event engine
uv run qr backtest list                                       # recent runs
uv run qr backtest compare 20261013T2134 20261013T2135        # side by side (id prefixes)
```

`run` prints the metrics after costs, before costs, and for the benchmark, writes a
tearsheet to `reports/` (gitignored), and adds a row to the registry.

### No look-ahead, enforced in code

- **A decision made on the close of day t trades on day t+1**, at the open
  (`execution: next_open`, the default) or the close (`next_close`), never on day t. The
  vectorized engine applies the one-bar delay itself; the event engine only matches
  orders submitted on earlier sessions.
- **Orders are sized with what was known at the decision.** A target weight becomes a
  share count using the equity and closes of day t. The fill happens at a different
  price, so the weight actually reached differs slightly from the target, exactly as it
  would in real trading.
- **Strategies only see the past.** An event strategy gets one day of bars and a
  `ctx.history()` that stops at the current bar. A weight strategy is given the whole
  table once, so the engine verifies it: it reruns the strategy on data cut off at a few
  dates (`lookahead_checks`, default 3) and raises `LookAheadError` if the weights for
  those dates change.
- **Rolling helpers are past-only**: `sma`, `past_return`, `rolling_volatility`, and
  `zscore` in `qr.backtest.strategy` need a full window and never center it.
- **No symbol is traded before its first bar**, whatever weight a strategy gives it.

### Writing a strategy for the vectorized engine

Return a (date x symbol) table of weights. The row for a date is the decision made on
that date's close.

```python
import pandas as pd
from qr.backtest.strategies import register
from qr.backtest.strategy import WeightStrategy, closes, sma


class AboveAverage(WeightStrategy):
    """Hold each symbol while it closes above its moving average."""

    def __init__(self, window: int = 100) -> None:  # config `params` arrive here
        self.window = window

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices = closes(bars)  # bars has the shape get_bars returns
        return (prices > sma(prices, self.window)) / prices.shape[1]


register("above_average", vectorized=AboveAverage)
```

Weights are fractions of equity: 0.5 is half the account, negative is short, NaN is no
position. Use ratios of prices, never price levels: adjusted levels shift whenever a later
dividend is paid, ratios do not.

### Writing a strategy for the event-driven engine

Override the hooks you need. Anything submitted in `on_bar` is executed on the next
session.

```python
from qr.backtest.account import Trade
from qr.backtest.events import Bar
from qr.backtest.strategy import Context, EventStrategy


class AboveAverageEvents(EventStrategy):
    def __init__(self, window: int = 100) -> None:
        self.window = window

    def on_bar(self, ctx: Context, bars: dict[str, Bar]) -> None:
        if not ctx.is_rebalance:  # the config's `rebalance` schedule
            return
        for symbol in ctx.symbols:
            prices = ctx.history(symbol, self.window)  # past only, adjusted
            if len(prices) < self.window:
                continue
            above = prices.iloc[-1] > prices.mean()
            ctx.order_target_weight(symbol, 1 / len(ctx.symbols) if above else 0.0)

    def on_fill(self, ctx: Context, trade: Trade) -> None: ...  # also on_start, on_end
```

`Context` offers `now`, `symbols`, `cash`, `equity`, `account` (a full `AccountState`),
`position(symbol)`, `history(symbol, n, field="close", adjusted=True)`, `is_rebalance`,
`open_orders()`, and the order methods `order(symbol, quantity)` (signed; a market order,
or market-on-close under `execution: next_close`), `order_limit(symbol, quantity, price,
tif="gtc"|"day")`, `order_target_quantity`, `order_target_weight`, and `cancel`.

Register both versions under one name with `register("name", vectorized=..., event=...)`
before running, or call the engines directly:

```python
from qr.backtest.config import load_backtest_config
from qr.backtest.runner import run_backtest

result = run_backtest(load_backtest_config("configs/backtests/sma_spy.yaml"))
result.equity, result.gross_equity, result.benchmark  # daily curves
result.metrics["sharpe"], result.net.trades, result.net.accounts
```

### Backtest config reference

One YAML file per experiment in `configs/backtests/`. Unknown keys are rejected.

```yaml
name: sma_crossover_spy     # used in the report file name and the registry
engine: vectorized          # vectorized | event
strategy: sma_crossover     # buy_and_hold | sma_crossover | xs_momentum | pairs | your own
params: { fast: 50, slow: 200 }
universe: [SPY]             # a list, or the name of a group in config/universe.yaml
start: 2012-01-01           # first session traded; earlier bars warm up indicators
end: 2024-12-31             # nothing after this date is read
initial_cash: 10000
benchmark: SPY              # or null
execution: next_open        # next_open | next_close
rebalance: daily            # daily | weekly | monthly | [2020-01-02, 2020-07-01]
whole_shares: true          # event engine; false allows fractions (crypto)
risk_free_rate: 0.0         # annual, for Sharpe, Sortino, and alpha
lookahead_checks: 3         # vectorized engine; 0 switches the check off
notes: ""
costs:
  commission_model: ibkr_tiered   # ibkr_tiered | ibkr_fixed | per_share | percent | none
  commission_per_share: 0.0035    # these four override the preset's numbers
  commission_min_per_order: 0.35
  commission_max_pct: 0.01        # cap, as a fraction of trade value
  commission_pct: 0.0             # fraction of trade value
  spread_bps: 1                   # half-spread: the cost of one trade
  slippage_bps: 2
  impact_bps: 0                   # extra slippage for an order as big as a day's volume
  impact_exponent: 1.0            # scaled by (order / daily volume) ** exponent
  borrow_rate: 0.01               # annual, on short market value, per calendar day / 360
  max_participation: null         # e.g. 0.1: at most 10% of average daily dollar volume
  participation_action: partial   # partial (cut the order) | reject
  adv_window: 20                  # sessions in the volume average
  asset_classes:                  # the same keys, per asset class ...
    crypto: { commission_model: percent, commission_pct: 0.0026, spread_bps: 5 }
  symbols:                        # ... and per symbol
    TLT: { spread_bps: 0.5 }
constraints:
  long_only: false
  max_gross_leverage: 1.0
  max_weight: null                # per symbol, as a fraction of equity
  min_trade_weight: 0.0           # skip rebalancing trades below this share of equity
```

`weekly` and `monthly` decide on the last session of the week or month and trade on the
first session of the next; the first session of the backtest is always a decision date.
Presets: `ibkr_tiered` is 0.0035 a share, 0.35 minimum, 1% maximum; `ibkr_fixed` is
0.005, 1.00, 1%; `per_share` 0.005; `percent` 0.1%.

### Costs

All of them apply in both engines, and every report shows the result **before and after
costs** (the "before" run is the same strategy with every cost at zero).

- **Commission** `per_share * shares + pct * value`, capped at `max_pct * value`, then
  raised to the per-order minimum.
- **Spread and slippage** in basis points of the traded value. The event engine charges
  them through the fill price (a buy pays more, a sell receives less); a limit order that
  is filled at its limit pays neither.
- **Short borrow** accrues daily on the short market value.
- **Participation cap** against the average dollar volume of the previous `adv_window`
  sessions.

### The account format

`qr.backtest.account.AccountState` is one serializable snapshot (timestamp, cash,
positions, open orders, equity, exposures, realized and unrealized P&L, fees) used by
both engines and meant for paper and live trading later. `state.to_json()` and
`AccountState.from_json()` round-trip exactly. Every fill produces an immutable `Trade`.
The vectorized engine has no shares or cost basis, so it produces the same daily table
with the P&L columns empty; `result.net.snapshot(date)` gives any day as an
`AccountState`.

### Metrics and the tearsheet

Total return, CAGR, annualized volatility, Sharpe, Sortino, Calmar, maximum drawdown,
longest drawdown, time underwater, annualized turnover, average holding period, gross and
net exposure, beta, correlation and alpha against the benchmark, rolling one-year Sharpe
and volatility, and the **Probabilistic Sharpe Ratio**. The event engine adds hit rate,
average win and loss, and profit factor from realized P&L.

The PSR is the probability that the true Sharpe ratio is above zero given this one
sample's length, skew, and fat tails. It does **not** account for multiple testing: try
enough ideas and one will look good by luck. Step 3 will add the Deflated Sharpe Ratio,
which needs the registry below.

Each run writes one self-contained HTML file (charts are inline SVG; nothing is loaded
from the network): equity against the benchmark on a log scale, drawdowns, a monthly
returns heatmap, rolling Sharpe and volatility, exposure, the before/after cost
comparison, the metrics table, the config, and the run's metadata.

### The experiment registry

Every `qr backtest run` is logged to `data/backtests.duckdb`: run id, time, strategy,
engine, the full config and its hash, git commit and whether the tree was dirty, the
version of the data read, the period, Sharpe, CAGR, maximum drawdown, turnover, the
report path, and notes. **Failed runs are logged too**, with status `failed` and the
error as notes. Do not delete rows for ideas that did not work: the number of things
tried is exactly what an honest significance test needs.

## Research studies

Step 3 uses the pipeline and the backtester to study trading ideas. Each study is a set
of notebooks that explain every step in plain language, with the reusable code in
`src/qr/research/` (tested in `tests/`).

### Pairs trading: KO / PEP and four comparison pairs

Does a simple pairs strategy on Coca-Cola and PepsiCo make money after costs, without
hindsight? The notebooks live in [`notebooks/pairs/`](notebooks/pairs/) and will be
published once the research is finished (`notebooks/` is gitignored until then):

1. `01_intuition`: what a spread is, correlation, and why "it always comes back" is hindsight
2. `02_cointegration`: correlation vs. cointegration, Engle-Granger, half-life, multiple testing
3. `03_signals`: rolling hedge ratio, z-score, entry/exit/stop rules, the cost of hindsight
4. `04_honest_backtest`: train 2010–2018, validation 2019–2022, test 2023 onwards (run once)
5. `05_findings`: the results in plain language, and what to try next

Code: `qr.research.pairs` (hedge ratios, spreads, cointegration tests, signals, leg
weights) and the `pairs` strategy for the vectorized engine, with costs in
`configs/backtests/pairs_ko_pep.yaml`.

**Result: no.** 36 parameter combinations were tried on 2010–2018 KO / PEP. The chosen
one made about 1% a year after costs in training and validation (Sharpe 0.26 and 0.20),
then lost 1.4% a year in the test (Sharpe −0.14), against 22% a year for SPY. With the
same parameters, no comparison pair made money after costs in all three periods; which
pair "worked" changed from period to period like noise.

## How it works

```
sources/ (yahoo, crypto)  ->  ingest  ->  data/raw/{source}/{symbol}/{timestamp}.parquet
                                 |
                                 v
                           market.duckdb  (unadjusted bars, corporate actions, logs)
                                 |
              validate  <--------+-------->  load (adjust at read time)
                                 ^
sources/tiingo  ->  crosscheck --+   (writes findings only, never bars)
```

- **Ingestion** (`qr/ingest.py`). First run fetches from `start_date`; later runs re-fetch
  from five trading sessions before the last stored bar to catch late corrections. Each
  download is written to a raw Parquet snapshot before anything else happens. If an
  overlapping value differs from what is stored by more than `revision_tolerance`, a
  `revision` warning is recorded and then the new value is stored. One symbol failing
  never stops the run; every symbol's outcome goes to `ingestion_log` under the run's id.
- **Failing fast** (also `qr/ingest.py`). A provider that is down would otherwise cost
  about half a minute of retries per symbol. So before a run each source gets one cheap
  test request (for Yahoo: five days of SPY, no retries); if that fails, the source's
  symbols are logged as `skipped` and the command exits non-zero within seconds. During
  a run, failed downloads are counted per source and any success resets the count. When
  `ingest.max_consecutive_failures` (default 3) symbols in a row have failed, the source
  is tested again: if the test fails too it is declared down and its remaining symbols
  are skipped; if the test succeeds, the failures were about those tickers and the run
  carries on.
- **Storage** (`qr/storage.py`). Five tables: `instruments`, `daily_bars` (unadjusted),
  `corporate_actions`, `ingestion_log`, `quality_issues`. Upserts only touch rows whose
  values changed, so re-running is a no-op and `ingested_at` means "when this value last
  changed".
- **Validation** (`qr/validate.py`). Findings go to `quality_issues`:

  | Check | Severity | Meaning |
  | --- | --- | --- |
  | `ohlc_consistency` | error | high below, or low above, the open/close |
  | `non_positive_price` | error | a price is zero or negative |
  | `negative_volume` | error | volume below zero |
  | `missing_values` | error | a NaN in a stored bar |
  | `duplicate_bar` | error | a provider response repeated a date |
  | `gap` | warning | expected sessions with no bar (NYSE calendar; every day for crypto) |
  | `unexpected_session` | warning | a bar on a day the market was closed |
  | `outlier_return` | warning | daily move above the threshold that no recorded split explains |
  | `stale_data` | warning | an active symbol's last bar is too many sessions old |
  | `zero_volume_streak` | warning | too many consecutive zero-volume days |
  | `revision` | warning | the provider changed a value we had already stored |
  | `crosscheck` | warning | the second source disagrees on a close or a split (written by `qr crosscheck`) |

  State checks are recomputed on every run, so the table shows the current state.
  `revision` and `duplicate_bar` are events and are kept as history. `crosscheck`
  findings are only touched by `qr crosscheck`.
- **Adjustment** (`qr/adjust.py`). Standard backward adjustment, computed when loading.
  A split of ratio `r` divides earlier prices by `r`; a dividend `D` multiplies earlier
  prices by `1 - D / previous close`.

## Known limitations and biases

Read this before trusting any backtest built on this data.

- **Survivorship bias.** The universe is a hand-picked list of tickers that exist today.
  Companies that went bankrupt, were acquired, or were delisted are simply absent, which
  makes any strategy tested on this list look better than it would have been. The schema
  keeps delisted symbols (`is_active`), but nothing here *finds* them. A point-in-time
  index-constituents source is a possible later step; until then, treat results on
  single stocks with suspicion and prefer broad ETFs.
- **Look-ahead bias.** Adjusted prices are computed with knowledge of later splits and
  dividends, and providers quietly revise history. Both leak future information into the
  past. Mitigations here: prices are stored unadjusted, adjustment never uses actions
  after `end`, revisions are logged rather than silently applied, and `as_of` reproduces
  what was known at a past download. None of this helps for the period before your first
  download.
- **Data source reliability.** `yfinance` is an unofficial scraper of Yahoo Finance. It
  can break without notice, be rate-limited, or return wrong, missing, or late data, and
  it comes with no accuracy guarantee. That is why every download is validated and
  snapshotted, and why `qr crosscheck` exists. Two free sources agreeing is still not
  proof: they may share an upstream vendor and its mistakes.
- **Yahoo's "unadjusted" prices are not unadjusted.** Yahoo serves prices, volumes, and
  dividend amounts already adjusted for every later split. `qr` undoes this with Yahoo's
  own split list (AAPL closes at about 645 the day before its 2014 split, not 23). If
  Yahoo lists a split before it has re-based its history, or the reverse, prices will be
  off by the split ratio for a while; the `outlier_return` and `revision` checks exist
  to catch that. Un-adjusting also leaves float noise around the sixth significant digit
  (a 2.65 dividend may read 2.650004).
- **Daily bars only**, with no intraday data, no bid/ask, and no auction prices. Yahoo's
  volume is consolidated and gets revised for days afterwards.
- **Other corporate actions are not modelled**: spin-offs, mergers, rights issues, ticker
  changes, and special distributions that Yahoo does not report as a dividend.
- **Crypto.** Kraken only serves its most recent 720 daily candles whatever start date is
  asked for, so history there begins about two years back. Use another exchange (e.g.
  `binance`) for a longer history. Crypto "closes" are simply 00:00 UTC.

### Limitations of the backtester

- **Daily bars are all it knows.** A market order fills at the official open or close
  plus the modelled costs. There is no queue, no intraday path, and no auction imbalance.
  A limit order either traded through or did not, judged from the day's high and low.
- **An order fills once.** What the participation cap or the leverage limit cuts off is
  cancelled, not worked over the following days.
- **Costs are a model.** Spread and slippage are flat basis points (plus an optional
  volume term), not measured from quotes. Exchange and regulatory fees, stamp duties,
  and taxes are not included; margin interest and interest on idle cash are zero.
- **Shorting is simplified**: one borrow rate, always available, no locates, no recalls,
  no margin calls. Gross exposure is limited when an order is filled, not afterwards, so
  a position can drift above the limit as prices move.
- **Dividends are paid on the ex-date** rather than the later payment date, and a split
  that leaves a fraction of a share keeps the fraction instead of paying cash.
- **The vectorized engine holds fractional shares** and reinvests dividends in the stock
  that paid them at once (it uses total-return prices); the event engine receives cash
  and reinvests on the next rebalance. This is the main reason the two differ slightly.
- **A daily rebalance pays the minimum commission on every tiny trade.** That is what a
  broker would charge, but nobody would send those orders. Use `min_trade_weight` or a
  weekly or monthly schedule.
- **Survivorship bias is not handled.** A symbol cannot be traded before its first bar,
  but the universe is still a list of today's survivors (see the first point above).
- **One currency (USD), no futures, options, or intraday data.**
- **The reference strategies are test fixtures.** Their results, and those of anything
  else run on a hand-picked universe over one period, are not evidence of skill.

## Design decisions

Where the spec left room, the more conservative, correctness-first option was chosen.

- **Truly unadjusted storage.** See the Yahoo point above. Storing as-traded prices is
  what makes the split logic testable and lets the step 2 event-driven engine apply
  splits and dividends the way a broker does.
- **Only completed sessions are stored.** The fetch window ends at the last session whose
  closing bell has passed (yesterday UTC for crypto), so a half-formed bar is never saved
  and later "revised".
- **Partial responses count as failures.** If a response lacks expected sessions, repeats
  a date, or contains incomplete rows, the usable rows are stored but the symbol is
  logged as `partial` and `qr ingest` exits non-zero. On a first run coverage is checked
  from the first bar the provider returned, since the listing date is unknown.
- **Revisions are recorded, then applied.** The newest provider value wins in the
  database; the old value survives in the earlier raw snapshot and in the `revision`
  finding's details.
- **Corporate actions are never deleted.** If a provider drops one, ours stays.
- **Dividend convention.** The factor `1 - D / previous close` is the common vendor
  convention (it reproduces Yahoo's "Adj Close" to about one part in a million on the
  default universe). It treats the dividend as reinvested at the theoretical ex-dividend
  price, so the adjusted return on an ex-date differs from the cash-accounting return
  `(close + D) / previous close - 1` by a second-order amount. Capital-gain distributions
  from funds are treated as dividends.
- **Raw snapshots hold what was stored, plus the provider's originals.** Each snapshot is
  the normalised frame the source returned; for Yahoo it also keeps the untouched values
  in `yahoo_*` columns. Corporate actions go in a sibling `*.actions.parquet` file. Names
  are UTC timestamps, and a file is never overwritten.
- **One calendar for equities and ETFs.** Gap and staleness checks use the NYSE calendar
  from `pandas_market_calendars`. A non-US listing would need its own calendar.
- **OHLC checks allow one part in a million**, because Yahoo's prices are single-precision
  floats and disagree with each other at that level.
- **Timestamps are naive UTC** everywhere in the database and snapshot names.
- **Extra modules beyond the spec's layout**: `frames.py` (the canonical column layout),
  `calendars.py` (trading sessions), and `snapshots.py` (raw Parquet files).
- **A down source is detected by asking it, not by reading error messages.** Yahoo
  answers a dead connection and a mistyped ticker with the same "no timezone found"
  error, so the type of an error is a poor guide. Instead every download failure counts
  towards the breaker, and only a failed test request for a ticker that certainly exists
  (SPY) declares the source down. Three bad tickers in a row therefore cost one extra
  request rather than aborting the run. Errors raised after a successful download (a
  bug, a full disk) never count.
- **A skipped source fails the run.** Symbols skipped because their source is *disabled*
  are fine; symbols skipped because their source is *down* make `qr ingest` exit non-zero.
- **The connectivity test is a real data request**, made through `yfinance` like any
  other, so it also notices rate limiting and a broken `yfinance`, not just a dead
  network. With Yahoo unreachable the full default universe fails in 10 to 35 seconds.
- **Cross-check findings are warnings and never change stored data.** With two sources
  there is no way to tell which is right, so the tool only reports. Merging sources or
  promoting Tiingo to primary is deliberately out of scope.
- **Cross-checks compare unadjusted closes**, since adjusted prices differ between
  vendors by convention. Days only one source has are counted in the summary but are not
  findings (the gap check already covers our side). Dividends are not compared.
- **Tiingo's corporate actions come from its price rows** (it has no separate endpoint
  on the free tier), so its snapshots hold the actions inside the fetched window only.
- **No new dependencies for Tiingo or `.env`.** The Tiingo client uses the standard
  library's `urllib` with the token in a header, and `.env` is read by a few lines in
  `qr/config.py` instead of `python-dotenv`.
- **Group membership is not stored in the database.** Groups are a view over the config;
  `--universe` just expands to a list of symbols.

### Backtester

When unsure, the more pessimistic assumption was taken: a backtest should be harder to
pass than real trading.

- **Orders are sized at the decision, not at the fill.** The usual vectorized shortcut,
  "the portfolio has exactly the target weights at the open", sizes trades with a price
  the decision could not know. On noisy opens that quietly earns a rebalancing premium:
  on synthetic data it added about 0.4% a year to a daily-rebalanced basket. Both engines
  instead fix the share count on the decision close and let the fill price fall where it
  may; a purchase the gap made unaffordable is cut to the leverage limit.
- **Limit orders must be traded through.** A bar whose low only touches a buy limit does
  not fill it. A limit order that gaps through its price fills at the open, which is
  better than the limit, and then pays spread and slippage like a market order (but
  never more than its limit).
- **Costs are on by default**: IBKR tiered commissions, 1 bp of half-spread, 2 bp of
  slippage, and a 1% borrow rate. `spread_bps` means the half-spread, the cost of one
  trade, as the spec defines it.
- **The commission minimum beats the 1% cap** when the two conflict on a very small order.
- **No volume history means no fill** for any symbol with a participation cap or volume
  impact configured: on its first bar the average volume is unknown.
- **Borrow fees accrue per calendar day on a 360-day year**, so a weekend costs three
  days.
- **Sells are matched before buys** within an open or a close, so a switch between two
  holdings can pay for itself. Everything submitted during a session, in `on_bar` or
  `on_fill`, waits for the next one.
- **The before-costs run keeps the participation cap.** It is the same strategy with
  costs at zero, so the two curves differ by costs and nothing else.
- **Event-engine history is adjusted point-in-time.** `ctx.history()` rescales the stored
  adjusted series so that today's value is the traded price, which removes every
  corporate action after the current bar.
- **Weights on a symbol without data are dropped, not redistributed.** The engine will
  not guess how a strategy wants its weights renormalized.
- **Runs are logged to their own database**, `data/backtests.duckdb`, not to
  `market.duckdb`: research code never opens the market data for writing, and a backtest
  cannot collide with the scheduled ingest's write lock. The table has two columns
  beyond the spec's, `name` and `status`.
- **Failed runs are returned, not raised**, by `runner.execute`, after being logged.
- **Zero-risk ratios are NaN.** A flat or perfectly steady curve has no Sharpe ratio;
  reporting infinity or zero would both be wrong.
- **Realized P&L is before commissions** and includes dividends; fees are tracked
  separately, so `equity = initial cash + realized + unrealized - fees` always holds.
- **Reports use matplotlib and inline SVG**, not plotly: a self-contained plotly page
  carries megabytes of JavaScript per report. The charts are static.
- **Extra modules beyond the spec's layout**: `data.py` (the bundle of bars and actions
  both engines read), `result.py` (the tables both engines return), and `runner.py`
  (config to result to report to registry). Indicator helpers live in `strategy.py`.
- **`EngineCore`** (in `event_engine.py`) is the interface the spec's optional Rust core
  would implement: stepping time, the account, the order book, and price history.
  `PythonCore` is the reference. The Rust core itself is not built.

### Pairs-trading study

- **Notebooks 01–03 only look at 2010–2018.** The spec plots the full history in the
  first notebooks, but a chart of the validation or test years seen early biases every
  later choice, so they stay hidden until notebook 04.
- **Log prices are rebased to the first bar** (`log(price / first price)`).
  Back-adjusted price levels change whenever a later dividend is paid, which would leak
  future dividends into a spread built from levels; ratios of adjusted prices do not.
- **The z-score re-hedges the whole window with today's hedge ratio**
  (`rolling_spread_zscore`). Subtracting the rolling mean of a spread built with a
  hedge ratio that changes every day mixes different spreads, and the result would
  depend on the price level.
- **The hedge ratio is frozen while a trade is open**, and a leg is only rebalanced
  once it has drifted by 5% of equity (`min_trade_weight: 0.05`). Re-hedging every day
  would pay the minimum commission on tiny trades.
- **After a stop-out, no new trade until the z-score is back near zero**, so a spread
  that has just broken out is not bought again on the way back.
- **Costs above the spec's 5 bps per leg**: IBKR commission on top of 2 bps half-spread
  and 3 bps slippage, plus 0.5% a year borrow on the short leg.
- **Selection rule fixed in advance**: highest after-cost training Sharpe among
  combinations with at least 10 trades. **No adjustment after validation**, although the
  spec allowed a small one.
- **The test period runs once.** The last cell of notebook 04 saves its results to
  `reports/pairs/test_results.json`; run again, it shows those, and it refuses to run
  with different parameters. Notebook 05 only reads that file.
- **Hit rate and holding period come from the spread position** (the vectorized engine
  does not track cost basis): a trade's return is the account's change from the close
  before it was opened to the close of the day it was closed, costs included.

## Development

```
uv run pytest                   # offline: every network call is replaced by a fake
uv run pytest --run-network     # also hit Yahoo for one symbol
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pre-commit install       # once: run the checks on every commit
```

Tests use small static CSVs in `tests/fixtures/` and seeded synthetic price series, and
never touch the network.

Continuous integration (`.github/workflows/ci.yml`) runs the lint, format, type, and test
commands above on Ubuntu and Windows with Python 3.12 and 3.13 for every push and pull
request. The pre-commit hooks run ruff plus a few hygiene checks and refuse to commit
`.env` or `*.duckdb` files. See [CONTRIBUTING.md](CONTRIBUTING.md).
