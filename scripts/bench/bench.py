"""Time the backtest engines: the baseline for the Rust core (step 2.5, phase 1).

Run from the repo root (the reference cases read ./data):

    uv run python scripts/bench/bench.py                 # every case
    uv run python scripts/bench/bench.py --case stress   # one case (e.g. under py-spy)
    uv run python scripts/bench/bench.py --json out.json # also save the numbers
    uv run python scripts/bench/bench.py --core rust     # event engine on the Rust core

Each timing is one simulation (``run_vectorized`` or ``run_event``) with costs: what
``run_backtest`` does twice, once with and once without costs. Loading data, building the
synthetic market, and computing metrics are not timed. Cases:

* ``reference``: buy-and-hold, SMA crossover, and momentum on the 17 symbols of
  ``config/universe.yaml`` (groups ``core`` and ``pairs``), 2012-2024, both engines.
* ``stress``: 500 synthetic symbols x 15 years of daily bars, momentum, event engine.
* ``sweep``: 200 SMA-crossover parameter pairs on SPY, event engine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from qr.backtest.config import BacktestConfig, load_backtest_config
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData, load_market_data
from qr.backtest.event_engine import run_event
from qr.backtest.result import EngineOutput
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import EventStrategy, WeightStrategy
from qr.backtest.vectorized import run_vectorized
from qr.config import universe_symbols

STRESS_SYMBOLS = 500
STRESS_YEARS = 15
SWEEP_FAST = list(range(10, 110, 5))  # 20 values
SWEEP_SLOW = list(range(120, 320, 20))  # 10 values: 200 pairs


def simulate(config: BacktestConfig, data: MarketData) -> EngineOutput:
    """One simulation with costs, built the way ``run_backtest`` builds it."""
    costs = CostModel(config.costs, data.symbols, data.asset_classes)
    strategy = build_strategy(config.strategy, config.engine, config.params)
    if isinstance(strategy, WeightStrategy):
        return run_vectorized(strategy, data, config, costs)
    assert isinstance(strategy, EventStrategy)
    return run_event(strategy, data, config, costs)


def timed(function: Callable[[], Any], repeat: int) -> list[float]:
    seconds = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        seconds.append(time.perf_counter() - start)
    return seconds


def with_changes(config: BacktestConfig, **changes: Any) -> BacktestConfig:
    return BacktestConfig.model_validate(config.model_dump() | changes)


# --------------------------------------------------------------------------- cases


def reference_cases(repeat: int) -> list[dict[str, Any]]:
    symbols = [*universe_symbols("core"), *universe_symbols("pairs")]
    data = load_market_data(symbols, date(2024, 12, 31))
    rows = []
    for name in ("buy_and_hold_spy", "sma_spy", "xs_momentum"):
        base = load_backtest_config(f"configs/backtests/{name}.yaml")
        for engine in ("vectorized", "event"):
            config = with_changes(base, engine=engine, universe=symbols)
            seconds = timed(lambda c=config: simulate(c, data), repeat)
            rows.append(row(f"{base.strategy} / 17 symbols", engine, seconds, config, data))
    return rows


def synthetic_market(symbols: int, years: int, seed: int = 0) -> MarketData:
    """Random-walk stocks with quarterly dividends; every 50th has a 2:1 split.

    The corporate actions are there so the event engine exercises its split and
    dividend code, not just order matching.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=years * 252)
    days = len(dates)
    frames, actions = {}, {}
    for i in range(symbols):
        log_close = np.log(rng.uniform(20, 200)) + np.cumsum(rng.normal(0.0003, 0.015, days))
        close = np.exp(log_close)
        events = []
        if i % 50 == 0:
            split_at = days // 2
            close[split_at:] /= 2.0
            events.append((dates[split_at], "split", 2.0))
        for k in range(63, days, 63):
            dividend = round(0.004 * close[k - 1], 4)
            close[k:] -= dividend
            events.append((dates[k], "dividend", dividend))
        close = np.maximum(close, 1.0)
        open_ = close * np.exp(rng.normal(0.0, 0.004, days))
        frames[f"S{i:03d}"] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": np.maximum(open_, close) * 1.003,
                "low": np.minimum(open_, close) * 0.997,
                "close": close,
                "volume": rng.uniform(5e5, 5e6, days).round(),
            }
        )
        actions[f"S{i:03d}"] = pd.DataFrame(events, columns=["date", "action", "value"])
    return MarketData.from_frames(frames, actions)


def stress_case(repeat: int) -> list[dict[str, Any]]:
    data = synthetic_market(STRESS_SYMBOLS, STRESS_YEARS)
    base = load_backtest_config("configs/backtests/xs_momentum_event.yaml")
    config = with_changes(
        base,
        name="stress",
        universe=list(data.symbols),
        start=data.dates[0].date(),
        end=data.dates[-1].date(),
        benchmark=None,
        params={"lookback": 252, "skip": 21, "top_n": 50},
        constraints={"long_only": True, "max_gross_leverage": 1.0},
    )
    seconds = timed(lambda: simulate(config, data), repeat)
    label = f"xs_momentum / {STRESS_SYMBOLS} synthetic x {STRESS_YEARS}y"
    return [row(label, "event", seconds, config, data)]


def sweep_case(repeat: int) -> list[dict[str, Any]]:
    base = load_backtest_config("configs/backtests/sma_spy_event.yaml")
    data = load_market_data(["SPY"], base.end)
    configs = [
        with_changes(base, params={"fast": fast, "slow": slow})
        for fast in SWEEP_FAST
        for slow in SWEEP_SLOW
    ]

    def sweep() -> None:
        for config in configs:
            simulate(config, data)

    seconds = timed(sweep, repeat)
    out = row(f"sma_crossover sweep / SPY, {len(configs)} runs", "event", seconds, base, data)
    out["per_run_s"] = out["median_s"] / len(configs)
    return [out]


def row(
    label: str, engine: str, seconds: list[float], config: BacktestConfig, data: MarketData
) -> dict[str, Any]:
    sessions = len(data.sessions(config.start, config.end))
    return {
        "case": label,
        "engine": engine,
        "symbols": len(data.symbols),
        "sessions": sessions,
        "repeat": len(seconds),
        "min_s": min(seconds),
        "median_s": statistics.median(seconds),
    }


CASES: dict[str, Callable[[int], list[dict[str, Any]]]] = {
    "reference": reference_cases,
    "stress": stress_case,
    "sweep": sweep_case,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=list(CASES), action="append")
    parser.add_argument("--repeat", type=int, default=None, help="runs per timing")
    parser.add_argument("--json", help="also write the results to this file")
    parser.add_argument(
        "--core", choices=["python", "rust"], default="python", help="event-engine core"
    )
    args = parser.parse_args()
    # run_event picks the core from this variable; fail rather than silently fall back.
    os.environ["QR_ENGINE_CORE"] = args.core
    os.environ["QR_REQUIRE_RUST_CORE"] = "1" if args.core == "rust" else ""

    results = []
    for name in args.case or list(CASES):
        # Reference runs are short, so repeat them; the long cases run once by default.
        repeat = args.repeat or (5 if name == "reference" else 1)
        for result in CASES[name](repeat):
            results.append(result)
            print(
                f"{result['case']:45s} {result['engine']:10s} "
                f"{result['symbols']:4d} sym {result['sessions']:5d} sessions  "
                f"min {result['min_s']:8.3f}s  median {result['median_s']:8.3f}s",
                flush=True,
            )
    if args.json:
        meta = {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.platform(),
            "event_core": args.core,
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"meta": meta, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
