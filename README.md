# zhufu-quant

Personal quantitative research toolkit. The Python package is called `qr`.

This is **step 1 of 4: the market data pipeline**. It downloads daily OHLCV bars plus
splits and dividends for a configurable universe, validates them, stores them locally in
DuckDB (with immutable Parquet snapshots of every download), and exposes a small Python
API and CLI for loading them. Later steps (backtesting, strategy research, paper/live
trading) import this as a library; their specs are in `docs/specs/`.

Priorities, in order: correctness, reproducibility, avoiding data biases. Speed and
features come after.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync                # install everything (add --extra crypto for the ccxt source)
uv run qr init         # create data/market.duckdb and its tables
uv run qr ingest       # download the default universe (about 20 seconds)
uv run qr status       # see what you have
```

Data lives in `./data` (gitignored). Set `QR_DATA_DIR` to move it and `QR_CONFIG` to use
a different universe file; every CLI command also takes `--data-dir`, and the ones that
read the universe take `--config`. See `.env.example`.

## CLI

| Command | What it does |
| --- | --- |
| `qr init` | Create the database and tables. Safe to repeat. |
| `qr ingest [--symbols X,Y]` | Incremental download, then quality checks. Exits non-zero if any symbol failed or came back incomplete. |
| `qr validate [--symbols X,Y]` | Re-run the quality checks and print a summary of open findings. Exits non-zero if any finding has severity `error`. |
| `qr status` | Per-symbol coverage: first/last date, row count, open errors and warnings. |
| `qr export SYMBOL --out f.csv [--adjustment none\|split\|total_return]` | Dump bars to CSV (total-return adjusted by default). |

Add `-v` before the command for debug logs: `qr -v ingest`.

### Running it daily

Ingestion is incremental and idempotent, so run it whenever you like. It only stores
completed sessions, so schedule it after the US close (21:00 UTC in winter, 20:00 in
summer) and leave Yahoo some time to settle. With cron, on weekdays at 23:30 UTC:

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

Enable with `systemctl --user enable --now qr-ingest.timer`. On Windows, a Task Scheduler
task running `uv run qr ingest` in the repo folder does the same job.

## Loading data in Python

`qr.load` is the only module later steps should import.

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
validation:
  outlier_return_threshold: 0.25
  stale_after_trading_days: 5
  max_zero_volume_streak: 3
  revision_tolerance: 0.0001
universe:
  - { symbol: SPY, source: yahoo, asset_class: etf }
  - { symbol: BTC/USD, source: crypto, asset_class: crypto }
```

Symbols whose source is disabled are skipped, not failed. Removing a symbol from the file
keeps its data and marks it inactive.

## How it works

```
sources/ (yahoo, crypto)  ->  ingest  ->  data/raw/{source}/{symbol}/{timestamp}.parquet
                                 |
                                 v
                           market.duckdb  (unadjusted bars, corporate actions, logs)
                                 |
              validate  <--------+-------->  load (adjust at read time)
```

- **Ingestion** (`qr/ingest.py`). First run fetches from `start_date`; later runs re-fetch
  from five trading sessions before the last stored bar to catch late corrections. Each
  download is written to a raw Parquet snapshot before anything else happens. If an
  overlapping value differs from what is stored by more than `revision_tolerance`, a
  `revision` warning is recorded and then the new value is stored. One symbol failing
  never stops the run; every symbol's outcome goes to `ingestion_log` under the run's id.
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

  State checks are recomputed on every run, so the table shows the current state.
  `revision` and `duplicate_bar` are events and are kept as history.
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
  snapshotted. Do not trade real money on a signal you have not cross-checked against a
  second source.
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

## Development

```
uv run pytest                   # offline: every network call is replaced by a fake
uv run pytest --run-network     # also hit Yahoo for one symbol
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Tests use small static CSVs in `tests/fixtures/`. `notebooks/01_explore_data.ipynb` loads
SPY, plots adjusted against unadjusted closes, and shows the validation summary; open it
with `uv run --with jupyterlab jupyter lab`.
