# zhufu-quant

[![CI](https://github.com/asbestboi/zhufu-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/asbestboi/zhufu-quant/actions/workflows/ci.yml)
![Python 3.12 | 3.13](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What this is

A personal quantitative research toolkit, built step by step as a learning project. The
Python package is called `qr`. So far it is a **market data pipeline**: it downloads daily
OHLCV bars plus splits and dividends for a configurable list of stocks and ETFs, checks
them for errors, stores them locally in DuckDB (with an immutable Parquet snapshot of
every download), and offers a small Python API and CLI for loading them, adjusted or not.
Later steps (backtesting, strategy research, paper trading) will import it as a library.

Priorities, in order: correctness, reproducibility, avoiding data biases. Speed and
features come after.

> **Disclaimer.** This project is for research and education only. Nothing in it is
> financial advice, and it comes with no warranty of any kind (see [LICENSE](LICENSE)).
> Market data is downloaded by you, to your machine, from third-party providers whose
> terms of use apply; this repository does not contain or redistribute their data.

## Roadmap

| Step | What | Status |
| --- | --- | --- |
| 1 | Market data pipeline: ingest, validate, store, load | Done |
| 1.1 | Pipeline polish: CI, fail-fast ingestion, cross-checking, named universes | Done |
| 2 | Backtesting engine (vectorized and event-driven) | Planned |
| 3 | Strategy research, starting with a pairs-trading study | Planned |
| 4 | Paper trading, then possibly live trading | Planned |

The hand-off specs for each step are in [`docs/specs/`](docs/specs).

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

## Development

```
uv run pytest                   # offline: every network call is replaced by a fake
uv run pytest --run-network     # also hit Yahoo for one symbol
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pre-commit install       # once: run the checks on every commit
```

Tests use small static CSVs in `tests/fixtures/` and never touch the network.
`notebooks/01_explore_data.ipynb` loads SPY, plots adjusted against unadjusted closes, and
shows the validation summary; it is committed with its outputs so the charts show on
GitHub. Open it with `uv run --with jupyterlab jupyter lab`.

Continuous integration (`.github/workflows/ci.yml`) runs the lint, format, type, and test
commands above on Ubuntu and Windows with Python 3.12 and 3.13 for every push and pull
request. The pre-commit hooks run ruff plus a few hygiene checks and refuse to commit
`.env` or `*.duckdb` files. See [CONTRIBUTING.md](CONTRIBUTING.md).
