# Project Spec: Personal Quant Research — Step 1: Market Data Pipeline

> Hand-off document for Claude Code. Build everything below inside the existing `zhufu-quant` repository (currently only README and LICENSE). The Python package is named `qr`. Ask before deviating from the stack or scope.

## 1. Context

I'm a CS student building a solo, self-funded quantitative trading research setup. This is **step 1 of a larger roadmap**:

1. **Data pipeline** ← this spec
2. Backtesting engine (vectorized + event-driven)
3. Strategy research and validation (walk-forward, out-of-sample)
4. Paper trading, then small live trading via a broker API (Interactive Brokers or Alpaca)

Everything later depends on this layer, so **correctness, reproducibility, and avoiding data biases matter more than speed or features.** Design so later steps can import this as a library.

## 2. Goal

A reliable, incremental pipeline that downloads **daily OHLCV price data** (plus splits and dividends) for a configurable universe of instruments, validates it, stores it locally in an analytics-friendly format, and exposes a clean Python API and CLI for loading it.

## 3. Tech Stack

- **Python 3.12+**, managed with **uv** (`pyproject.toml`, lockfile committed)
- **pandas** for data handling (keep loading functions able to return pandas DataFrames; polars optional internally)
- **DuckDB** as the local store, with raw data also persisted as **Parquet** files
- **yfinance** as the first data source (free, unofficial — treat as unreliable and validate everything)
- **ccxt** as an optional second source for crypto daily bars
- **typer** for the CLI
- **pydantic** for config models and validation
- **pytest** for tests; **ruff** for lint/format; **mypy** for type checking
- **loguru** or stdlib `logging` for structured logs

## 4. Repository Structure

```
zhufu-quant/
├── pyproject.toml
├── README.md
├── .gitignore              # must ignore data/, .env, *.duckdb
├── .env.example            # placeholders for future API keys
├── config/
│   └── universe.yaml       # instruments to track + settings
├── data/                   # gitignored
│   ├── raw/                # immutable Parquet snapshots per source/symbol/download
│   └── market.duckdb
├── src/qr/
│   ├── __init__.py
│   ├── config.py           # load + validate universe.yaml
│   ├── sources/
│   │   ├── base.py         # abstract DataSource interface
│   │   ├── yahoo.py
│   │   └── crypto.py       # ccxt-based, optional
│   ├── storage.py          # DuckDB schema, upserts, reads
│   ├── ingest.py           # incremental download orchestration
│   ├── validate.py         # data quality checks
│   ├── adjust.py           # split/dividend adjustment logic
│   ├── load.py             # public API used by later steps
│   └── cli.py
├── tests/
│   ├── fixtures/           # small static CSV/Parquet samples, no network
│   └── test_*.py
└── notebooks/
    └── 01_explore_data.ipynb
```

## 5. Data Sources

### Source interface
Every source implements a common abstract interface so new providers (Polygon, Alpaca, IBKR) can be added later without touching the rest of the code:

```python
class DataSource(ABC):
    name: str
    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame: ...  # splits, dividends
```

### Yahoo (yfinance)
- Download **unadjusted** OHLCV plus splits and dividends separately. Do **not** rely on Yahoo's pre-adjusted prices as the source of truth — compute adjustments ourselves in `adjust.py`.
- Handle rate limiting: retry with exponential backoff, configurable delay between symbols.
- Treat empty or partial responses as errors to log, never silent successes.

### Crypto (ccxt, optional)
- Daily candles from one configurable exchange (e.g. Binance or Kraken).
- Paginate correctly to get full history.
- No corporate actions; 7-day weeks (validation must account for this).

## 6. Configuration (`config/universe.yaml`)

```yaml
start_date: 2010-01-01
timezone: UTC
sources:
  yahoo:
    request_delay_seconds: 1.0
    max_retries: 5
  crypto:
    enabled: false
    exchange: kraken
universe:
  - { symbol: SPY,  source: yahoo, asset_class: etf }
  - { symbol: QQQ,  source: yahoo, asset_class: etf }
  - { symbol: IWM,  source: yahoo, asset_class: etf }
  - { symbol: TLT,  source: yahoo, asset_class: etf }
  - { symbol: GLD,  source: yahoo, asset_class: etf }
  - { symbol: AAPL, source: yahoo, asset_class: equity }
  - { symbol: MSFT, source: yahoo, asset_class: equity }
  - { symbol: BTC/USD, source: crypto, asset_class: crypto }
```

Validate with pydantic; fail fast with a clear error on bad config.

## 7. Storage Schema (DuckDB)

```sql
CREATE TABLE IF NOT EXISTS instruments (
  symbol        TEXT PRIMARY KEY,
  source        TEXT NOT NULL,
  asset_class   TEXT NOT NULL,
  first_date    DATE,
  last_date     DATE,
  is_active     BOOLEAN DEFAULT TRUE,   -- keep delisted symbols, never delete
  updated_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS daily_bars (
  symbol     TEXT NOT NULL,
  date       DATE NOT NULL,
  open       DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
  volume     DOUBLE,
  source     TEXT NOT NULL,
  ingested_at TIMESTAMP NOT NULL,
  PRIMARY KEY (symbol, date)
);  -- UNADJUSTED prices

CREATE TABLE IF NOT EXISTS corporate_actions (
  symbol  TEXT NOT NULL,
  date    DATE NOT NULL,
  action  TEXT NOT NULL,   -- 'split' | 'dividend'
  value   DOUBLE NOT NULL, -- split ratio or cash dividend per share
  PRIMARY KEY (symbol, date, action)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
  run_id      TEXT, symbol TEXT, started_at TIMESTAMP, finished_at TIMESTAMP,
  rows_written INTEGER, status TEXT, message TEXT
);

CREATE TABLE IF NOT EXISTS quality_issues (
  symbol TEXT, date DATE, check_name TEXT, severity TEXT, details TEXT, detected_at TIMESTAMP
);
```

Upserts must be idempotent: re-running an ingest never duplicates rows.

## 8. Ingestion Behaviour

- **Incremental:** for each symbol, fetch from `last_date - 5 trading days` (overlap to catch late corrections) to today. First run fetches from `start_date`.
- **Raw snapshots:** before upserting, write the raw response to `data/raw/{source}/{symbol}/{YYYYMMDD_HHMMSS}.parquet`. Raw files are never modified. This makes results reproducible and lets us detect when a provider silently revises history.
- **Revision detection:** if overlapping rows differ from stored values beyond a small tolerance, log a `quality_issues` row (severity `warning`) instead of silently overwriting without trace.
- **Fault isolation:** one failing symbol must not abort the whole run. Log it and continue; exit with a non-zero code at the end if anything failed.
- Every run writes to `ingestion_log` with a unique `run_id`.

## 9. Validation Checks (`validate.py`)

Run automatically after ingestion and via CLI. Record findings in `quality_issues`.

- `high >= max(open, close)` and `low <= min(open, close)`
- No non-positive prices; no negative volume
- No duplicate `(symbol, date)`
- **Gap detection:** missing trading days vs. the expected calendar (use `pandas_market_calendars` for exchanges; every day for crypto)
- **Outlier returns:** flag daily absolute returns above a configurable threshold (e.g. 25%) that do **not** coincide with a known split
- **Stale data:** flag symbols whose `last_date` is older than N trading days
- **Zero-volume streaks** longer than a configurable number of days

## 10. Adjustments (`adjust.py`)

- Compute **split-adjusted** and **total-return (split + dividend) adjusted** series from unadjusted bars plus `corporate_actions`, using the standard backward-adjustment method.
- Keep unadjusted data untouched in storage; adjustment happens at load time.
- Unit tests with hand-computed expected values for a known split and a known dividend.

## 11. Public Loading API (`load.py`)

This is what steps 2–4 will import:

```python
def get_bars(
    symbols: list[str] | str,
    start: date | None = None,
    end: date | None = None,
    adjustment: Literal["none", "split", "total_return"] = "total_return",
    as_of: datetime | None = None,   # see below
) -> pd.DataFrame: ...               # MultiIndex (date, symbol) or long format, documented

def get_returns(symbols, start=None, end=None, kind: Literal["simple", "log"] = "log") -> pd.DataFrame: ...
def list_symbols(active_only: bool = False) -> list[str]: ...
```

**Point-in-time support (`as_of`):** when given, return only data that had been ingested by that timestamp (using `ingested_at` / raw snapshots). This guards against look-ahead bias later on. Implement at least a basic version and document its limits.

## 12. CLI (`qr`)

```
qr init                      # create DB + tables
qr ingest [--symbols X,Y]    # incremental download
qr validate [--symbols X,Y]  # run quality checks, print summary table
qr status                    # per-symbol coverage: first/last date, row count, open issues
qr export SYMBOL --out f.csv # dump adjusted bars
```

## 13. Bias Awareness (document in README)

The README must have a short section explaining known limitations of this dataset:

- **Survivorship bias:** a hand-picked list of today's tickers excludes delisted companies, which inflates backtest results. Note this clearly; a future step may add a point-in-time constituents source.
- **Look-ahead bias:** adjusted prices and revised data can leak future information; point-in-time loading mitigates this partly.
- **Data source reliability:** yfinance is unofficial and may break or return wrong data; validation exists for this reason.

## 14. Testing Requirements

- Tests must **run offline** using fixtures; mock all network calls.
- Cover: config validation, idempotent upserts, incremental date logic, every validation check (with a fixture that triggers it), split and dividend adjustment math, revision detection, and `get_bars` with each adjustment mode.
- One optional integration test marked `@pytest.mark.network` that hits Yahoo for a single symbol (skipped by default).
- `ruff check`, `ruff format --check`, `mypy src`, and `pytest` should all pass.

## 15. Acceptance Criteria

- [ ] `uv sync && uv run qr init && uv run qr ingest` populates the DB for the default universe
- [ ] Running `qr ingest` twice in a row adds no duplicate rows
- [ ] `qr status` shows coverage per symbol; `qr validate` prints a readable issue summary
- [ ] `get_bars("SPY", adjustment="total_return")` returns a clean, documented DataFrame
- [ ] Raw snapshots exist for every download
- [ ] All tests, lint, and type checks pass
- [ ] README explains setup, CLI usage, the loading API, and the bias limitations
- [ ] `notebooks/01_explore_data.ipynb` loads SPY, plots adjusted vs. unadjusted close, and shows the validation summary

## 16. Out of Scope (for now)

- Intraday / tick data
- Paid data providers (keep the interface ready for them)
- Fundamentals, news, or alternative data
- Backtesting, strategies, or any order execution
- Scheduling — but add a short README note showing how to run `qr ingest` daily via cron or a systemd timer

## 17. Working Style

- Commit in small, logical steps with clear messages.
- Prefer simple, readable code over clever abstractions; this is a research codebase I need to understand fully.
- Add docstrings to public functions and type hints everywhere.
- When a design decision is ambiguous, pick the more conservative, correctness-first option and note it in the README under "Design decisions".
