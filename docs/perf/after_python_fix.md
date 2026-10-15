# After the Python data-path fix (step 2.5, part A)

Same machine and method as [`baseline.md`](baseline.md): Intel Core i7-8650U, 16 GB RAM,
Arch Linux, Python 3.13.16; `uv run python scripts/bench/bench.py`; each time is one
simulation with costs. Measured on 2026-10-15 on the branch `fast-data-path`.

**What changed** (event engine only; the vectorized engine is untouched):

- `ctx.history()` returns a read-only NumPy array instead of a pandas Series
  (`ctx.history_df()` keeps the old form).
- A session's bars are a mapping that builds a `Bar` only when a symbol is looked up; the
  broker marks only the symbols held.
- Session datetimes are converted once per run instead of once per step.
- The reference strategies use the arrays directly.

**Results are identical**: trades, account rows, positions, costs, and orders of every
event-engine reference run, the 500-symbol stress case, and three sweep runs were saved
before the change and compared after it with `check_exact=True`: no differences.

## Before and after

| Case | Engine | Before | After | Speedup |
| --- | --- | ---: | ---: | ---: |
| Buy and hold, 17 symbols | vectorized | 0.072 s | 0.067 s | (unchanged) |
| Buy and hold, 17 symbols | event | 0.358 s | 0.349 s | 1.0× |
| SMA crossover, 17 symbols | vectorized | 0.220 s | 0.205 s | (unchanged) |
| SMA crossover, 17 symbols | event | 4.613 s | 1.369 s | **3.4×** |
| Momentum, 17 symbols | vectorized | 0.069 s | 0.070 s | (unchanged) |
| Momentum, 17 symbols | event | 0.473 s | 0.179 s | **2.6×** |
| Stress: momentum, 500 symbols × 15 years | event | 10.9 s | 3.09 s | **3.5×** |
| Sweep: 200 SMA crossovers on SPY | event | 79.4 s | 33.4 s | **2.4×** (0.40 → 0.17 s a run) |

Reference cases: median of 5 runs. Stress and sweep: one run each. The vectorized rows did
not change and show the run-to-run noise (a few percent). Buy and hold barely moves: it
trades monthly and never calls `history()`, so it had little boundary cost to remove.

One `history("SPY", 200)` call in the middle of a run: **36.1 µs before, 5.8 µs after**
(3.4 µs for unadjusted prices, which are a view). `history_df` costs about 42 µs.

## Where the time goes now

Share of the samples inside `run_event` (py-spy, 200 Hz):

| | Stress (500 symbols) | Sweep (SPY, 200 runs) |
| --- | ---: | ---: |
| Core setup (`PythonCore.__init__`) | **45%** | 3% |
| …turning bars into arrays (`MarketData.panel`: pandas `unstack`) | 29% | <1% |
| …grouping corporate actions (`_group_actions`) | 14% | 1% |
| Strategy `on_bar` (Python) | 30% | **50%** |
| …`ctx.history()` | 22% | 20% |
| …NumPy `mean` in the strategy | – | 15% |
| …`order_target_weight` | 1% | 11% |
| `step()`: order matching, fills, account (`SimBroker.run_session`) | 15% | 18% |
| `step()`: everything else | 1% | 9% |
| Account snapshot for the results (`account_state`) | 2% | 7% |
| Building result tables (`run_event` itself, DataFrames) | 7% | 12% |

In the sweep the panels are built once and reused by all 200 runs, which is why setup is
small there; the stress case builds them once for one run.

Top 10 functions by self time, stress (876 samples inside `run_event`):

| Self | Function |
| ---: | --- |
| 8.8% | `history` (`qr/backtest/event_engine.py`) |
| 8.3% | `_make_selectors` (pandas `unstack`, in setup) |
| 7.1% | `_wrapfunc` (numpy `searchsorted`, in `history`) |
| 6.3% | `SessionBars.__getitem__` (`qr/backtest/event_engine.py`) |
| 6.2% | `compress_group_index` (pandas `unstack`, in setup) |
| 5.3% | `on_bar` (`qr/backtest/strategies/xs_momentum.py`) |
| 4.8% | `remove_unused_levels` (pandas `unstack`, in setup) |
| 3.9% | `__init__` (`<string>`: `Bar` and `Fill` dataclasses) |
| 3.5% | `_window` (`qr/backtest/event_engine.py`) |
| 3.4% | `get_group_index` (pandas `unstack`, in setup) |

Top 10 functions by self time, sweep (9,697 samples inside `run_event`):

| Self | Function |
| ---: | --- |
| 9.7% | `_mean` (numpy, the strategy's moving averages) |
| 8.0% | `run_event` (`qr/backtest/event_engine.py`) |
| 7.5% | `history` (`qr/backtest/event_engine.py`) |
| 7.2% | `_wrapfunc` (numpy `searchsorted`, in `history`) |
| 5.4% | `_count_reduce_items` (numpy, inside `mean`) |
| 5.0% | `step` (`qr/backtest/event_engine.py`) |
| 4.6% | `_window` (`qr/backtest/event_engine.py`) |
| 3.8% | `SessionBars.__getitem__` (`qr/backtest/event_engine.py`) |
| 3.5% | `__init__` (`<string>`: `Bar` and `Fill` dataclasses) |
| 3.3% | `on_bar` (`qr/backtest/strategies/sma_crossover.py`) |

## What this means for a Rust core

The parts a Rust core would take over, and the most any core could save on them (Amdahl's
law: if a share *p* of the time disappears completely, the run is at most 1 / (1 − *p*)
times faster):

| What moves to Rust | Stress: share → best case | Sweep: share → best case |
| --- | ---: | ---: |
| Event loop, matching, fills, account (`step`, `account_state`) | 18% → 1.22× | 34% → 1.52× |
| …plus building the result tables | 25% → 1.34× | 46% → 1.85× |
| …plus the one-time setup (array store, corporate actions) | 70% → 3.3× | 49% → 1.96× |

The strategy (30–50%) stays in Python whatever the core is written in, and every
`history()` call and callback still crosses between Python and the core, so a real Rust
core would land below these ceilings. The biggest single item in the stress case, the
setup, is pandas `unstack` and grouping 30,000 corporate actions: plain data preparation
that could also be made much cheaper in Python.
