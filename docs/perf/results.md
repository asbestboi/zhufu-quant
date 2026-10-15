# Results: baseline, Python fix, Rust core (step 2.5)

All numbers from the same machine: Intel Core i7-8650U (4 cores, 1.9 GHz laptop), 16 GB
RAM, Arch Linux, Python 3.13.16, Rust 1.99, `qr_core` built with
`maturin develop --release`. Script: `uv run python scripts/bench/bench.py [--core rust]`.
Each time is one simulation with costs; loading data, building the synthetic market, and
computing metrics are not timed.

- **Baseline**: before any change ([`baseline.md`](baseline.md)).
- **Part A**: Python core after the data-path fix
  ([`after_python_fix.md`](after_python_fix.md)), re-measured in the same session as the
  Rust core (within a few percent of the first Part A measurement).
- **Part B**: the same Python code on the Rust core (`engine_core: rust`).

## Timings

| Case (event engine) | Baseline | Part A (Python) | Part B (Rust) | Rust vs. Part A | Rust vs. baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy and hold, 17 symbols, 2012–2024 | 0.358 s | 0.369 s | 0.122 s | 3.0× | 2.9× |
| SMA crossover, 17 symbols, 2012–2024 | 4.613 s | 1.378 s | 0.538 s | 2.6× | 8.6× |
| Momentum, 17 symbols, 2012–2024 | 0.473 s | 0.167 s | 0.082 s | 2.0× | 5.8× |
| Stress: momentum, 500 symbols × 15 years, first run | 10.9 s | 2.91 s | 1.97 s | 1.5× | 5.5× |
| Stress, repeated runs (median of 3) | – | 1.92 s | 1.06 s | 1.8× | – |
| Sweep: 200 SMA crossovers on SPY | 79.4 s | 32.9 s | 14.4 s | 2.3× | 5.5× |

Reference cases: median of 5 runs. Stress and sweep: one run unless noted. The first
stress run includes turning the market into NumPy arrays (pandas `unstack`), which later
runs on the same data reuse; that part is the same Python code for both cores, which is
why the Rust advantage is larger on repeated runs.

The vectorized engine is not part of the Rust core and did not change (0.07 s, 0.21 s,
and 0.07 s for the three 17-symbol cases throughout).

**Results are identical on both cores**: every reference strategy and the 500-symbol
stress market give the same trades, account rows, positions, costs, and orders, compared
with `check_exact=True` (`tests/test_bt_rust_core.py`), and the whole event-engine test
suite passes on both.

## Where the time goes on the Rust core

The sweep under py-spy (Python frames only; time inside Rust shows up in the Python
function that called it):

| | Share of `run_event` |
| --- | ---: |
| Strategy `on_bar` | 44% |
| …NumPy `mean` (the strategy's moving averages) | 26% |
| …`order_target_weight` (sizing in Rust, order objects in Python) | 7% |
| …`ctx.history()` (a call into Rust that returns a view) | 5% |
| `run_event` itself (collecting result rows) | 13% |
| `RustCore.step` (one call into Rust, then `Fill`/`Trade` objects) | 12% |
| `account_state` (an `AccountState` object every session) | 10% |
| Result DataFrames | 8% |
| `RustCore.__init__` (copying the bars into Rust) | 8% |

`history()` went from 20% of the sweep on the Python core to 5%.

## Conclusion

The Rust core makes the event engine **1.5× to 3× faster** than the fixed Python core on
these benchmarks, and **3× to 9× faster** than where it started. Most of the total gain
from the baseline came from the Python fix in part A (2.4× to 3.5× on its own); Rust
roughly doubled it again.

The part A report estimated that Rust could make the sweep at most 1.85× faster. It came
out at 2.3×. The estimate counted every `history()` call and every order-sizing call as
strategy time, which stays in Python; but those calls go into the core, and the Rust core
answers them several times faster. The estimate's main point still holds: what is left is
mostly Python (the strategy's own NumPy work, the per-session result objects, and the
DataFrames at the end), and a faster core cannot remove it. Going further would mean
changing the strategy interface, which this step was not allowed to do.

## Floating-point notes

Both cores use `f64` (Python's `float`) everywhere, and the Rust code does every
calculation in the same order as the Python code, so no tolerance is needed. Three places
needed care:

- **`sum()`**: since Python 3.12 the built-in `sum()` uses Neumaier compensated
  summation, not left-to-right addition. The account's totals (market value, gross
  exposure, unrealized P&L) are `sum()`s, so the Rust account uses `py_sum`, a port of
  CPython's algorithm. Plain addition gave equity that differed in the last bits on the
  multi-position strategies.
- **`max()` and `min()`**: Python's built-ins and Rust's `f64::max`/`f64::min` disagree
  when one side is NaN; the Rust core uses `py_max`/`py_min`, which copy Python.
- **Position order**: positions are kept in the order they were opened, like a Python
  `dict`, because the order of the terms changes a floating-point sum.

`x ** y` (volume impact) is the C library's `pow` in both languages; parity was tested on
Linux, and CI runs the parity tests on Windows too.
