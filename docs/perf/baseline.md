# Performance baseline (step 2.5, phase 1)

Measured on 2026-10-15, before any Rust, on the branch `rust-core` at `main` d76ba19.

- **Machine:** Intel Core i7-8650U (4 cores / 8 threads, 1.9 GHz, laptop), 16 GB RAM,
  Arch Linux, Python 3.13.16
- **Script:** `uv run python scripts/bench/bench.py` (see its docstring). Each time is
  one simulation with costs; loading data, building the synthetic market, and computing
  metrics are not included. Reference cases: min and median of 5 runs. Stress and sweep:
  one run.
- **Profiler:** `py-spy record --format raw --rate 200`, summarized with
  `scripts/bench/top_functions.py --within run_event` (only samples inside the event
  engine count, so data loading and market generation are excluded).

## Timings

| Case | Engine | Symbols | Sessions | Min | Median |
| --- | --- | ---: | ---: | ---: | ---: |
| Buy and hold, 17 symbols, 2012–2024 | vectorized | 17 | 3,270 | 0.069 s | 0.072 s |
| Buy and hold, 17 symbols, 2012–2024 | event | 17 | 3,270 | 0.353 s | 0.358 s |
| SMA crossover, 17 symbols, 2012–2024 | vectorized | 17 | 3,270 | 0.213 s | 0.220 s |
| SMA crossover, 17 symbols, 2012–2024 | event | 17 | 3,270 | 4.580 s | 4.613 s |
| Momentum, 17 symbols, 2012–2024 | vectorized | 17 | 3,270 | 0.066 s | 0.069 s |
| Momentum, 17 symbols, 2012–2024 | event | 17 | 3,270 | 0.465 s | 0.473 s |
| Stress: momentum, 500 synthetic symbols × 15 years | event | 500 | 3,780 | 10.9 s | – |
| Sweep: 200 SMA crossovers on SPY | event | 1 | 3,270 each | 79.4 s (0.40 s per run) | – |

The SMA crossover is the slow event case: it rebalances daily, and each day it asks for
200 days of history for every symbol. Buy and hold and momentum rebalance monthly.

## Where the event engine's time goes

Share of the samples inside `run_event`, grouped by what the code is doing:

| | Stress (500 symbols) | Sweep (SPY, 200 runs) | SMA, 17 symbols |
| --- | ---: | ---: | ---: |
| Strategy `on_bar` (Python, stays Python) | 44% | 66% | 90% |
| …of which `ctx.history()` building a pandas Series | 31% | 39% | 52% |
| `PythonCore.step()` | 41% | 24% | ~6% |
| …building `Bar` objects for every symbol (`step` lines 204–211 plus `Bar.__init__`) | ~37% | ~3% | – |
| …looking up the session's `datetime` in a pandas index (`step` line 193) | <1% | 9% | – |
| …order matching (`SimBroker.run_session`) | 3.3% | 4.1% | <2% |
| Account snapshots (`account_state`) | 0.4% | 3.8% | <2% |
| Core setup (`PythonCore.__init__`: panels, grouping actions) | 11% | 1% | – |

### Top 10 functions by self time

Stress case (3,153 samples inside `run_event`):

| Self | Function |
| ---: | --- |
| 24.3% | `step` (`qr/backtest/event_engine.py`) |
| 12.7% | `__init__` (`<string>`: the `Bar` dataclass constructor) |
| 6.1% | `history` (`qr/backtest/event_engine.py`) |
| 3.2% | `Series.__init__` (pandas) |
| 3.2% | `on_bar` (`qr/backtest/strategies/xs_momentum.py`) |
| 3.1% | `_wrapfunc` (numpy `searchsorted`) |
| 2.6% | `run_session` (`qr/backtest/broker_sim.py`) |
| 2.2% | `_make_selectors` (pandas `unstack`, in core setup) |
| 2.0% | `sanitize_array` (pandas, building a Series) |
| 1.8% | `run_event` (`qr/backtest/event_engine.py`) |

Sweep (22,057 samples inside `run_event`):

| Self | Function |
| ---: | --- |
| 8.2% | `step` (`qr/backtest/event_engine.py`) |
| 7.2% | `history` (`qr/backtest/event_engine.py`) |
| 5.9% | `_mean` (numpy: the strategy's moving averages) |
| 5.1% | `_box_func` (pandas: making `Timestamp`s from a `DatetimeIndex`) |
| 4.6% | `to_numpy` (pandas: the strategy turning the Series back into an array) |
| 3.8% | `_wrapfunc` (numpy `searchsorted`) |
| 3.8% | `run_event` (`qr/backtest/event_engine.py`) |
| 3.5% | `on_bar` (`qr/backtest/strategies/sma_crossover.py`) |
| 3.5% | `Series.__init__` (pandas) |
| 3.1% | `_count_reduce_items` (numpy, inside `mean`) |

### One `history()` call, taken apart

`core.history("SPY", 200)` in the middle of a run, averaged over 20,000 calls:

| Part | Time |
| --- | ---: |
| The whole call | 36.1 µs |
| The numpy slice and rescaling that is the actual work | 1.3 µs |
| Slicing the `DatetimeIndex` for the result's index | 4.9 µs |
| Building the `pd.Series` | 20.7 µs |

About 95% of a `history()` call is building pandas objects that most strategies turn
straight back into arrays (`sma_crossover` calls `.to_numpy()` on the result).

## Conclusion

The parts the spec expected to port (order matching in `broker_sim`, account and
position updates) take **under 10% of the time** in every case measured. Porting them
to Rust would save at most that much, whatever Rust's speed. Most of the time is
Python object construction at the boundary with strategies: pandas Series for
`history()`, a `Bar` object per symbol per session, a pandas `Timestamp` per session.
Strategies stay in Python, so a Rust core would still have to build those same Python
objects for them.
