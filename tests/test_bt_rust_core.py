"""The Rust event-engine core: choosing it, falling back without it, and exact parity.

The parity tests are the contract: on every reference strategy and on a 500-symbol stress
market, the Rust core must give the same trades, account rows, positions, costs, and
orders as the Python core, bit for bit (``check_exact=True``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from qr.backtest import rust_core
from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData
from qr.backtest.event_engine import PythonCore, run_event
from qr.backtest.result import EngineOutput
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import EventStrategy

from .bt_helpers import make_config, synthetic_market

needs_rust = pytest.mark.skipif(
    not rust_core.rust_available() and os.environ.get(rust_core.REQUIRE_ENV_VAR) != "1",
    reason="the Rust core (qr_core) is not built",
)


# --------------------------------------------------------------------------- choosing a core


def config(**overrides: Any) -> BacktestConfig:
    data = synthetic_market(2, 30)
    return make_config(data, engine="event", **overrides)


def test_python_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(rust_core.CORE_ENV_VAR, raising=False)
    assert rust_core.select_core(config()) is PythonCore


def test_the_environment_variable_overrides_the_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(rust_core.CORE_ENV_VAR, "python")
    assert rust_core.select_core(config(engine_core="rust")) is PythonCore
    monkeypatch.setenv(rust_core.CORE_ENV_VAR, "fortran")
    with pytest.raises(ValueError, match="must be 'python' or 'rust'"):
        rust_core.select_core(config())


def test_without_the_module_rust_falls_back_to_python_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(rust_core, "_qr_core", None)
    monkeypatch.delenv(rust_core.CORE_ENV_VAR, raising=False)
    monkeypatch.delenv(rust_core.REQUIRE_ENV_VAR, raising=False)
    with caplog.at_level(logging.WARNING, logger="qr.backtest.rust_core"):
        assert rust_core.select_core(config(engine_core="rust")) is PythonCore
    assert "not installed" in caplog.text and "maturin develop" in caplog.text
    # A whole run still works, on the Python core.
    data = synthetic_market(2, 60)
    cfg = make_config(data, engine="event", engine_core="rust")
    strategy = build_strategy("buy_and_hold", "event", {})
    assert isinstance(strategy, EventStrategy)
    run_event(strategy, data, cfg, CostModel(cfg.costs, data.symbols))


def test_requiring_rust_without_the_module_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rust_core, "_qr_core", None)
    monkeypatch.setenv(rust_core.REQUIRE_ENV_VAR, "1")
    with pytest.raises(RuntimeError, match="not installed"):
        rust_core.select_core(config(engine_core="rust"))


def test_the_core_does_not_change_the_config_hash() -> None:
    assert config(engine_core="rust").config_hash() == config(engine_core="python").config_hash()
    assert config(engine_core="rust").to_json() != config(engine_core="python").to_json()


# --------------------------------------------------------------------------- parity


def stress_market(symbols: int = 500, years: int = 15, seed: int = 0) -> MarketData:
    """Random-walk stocks with quarterly dividends; every 50th has a 2:1 split.

    The same market as the stress case in ``scripts/bench/bench.py``.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=years * 252)
    days = len(dates)
    frames, actions = {}, {}
    for i in range(symbols):
        close = np.exp(np.log(rng.uniform(20, 200)) + np.cumsum(rng.normal(3e-4, 0.015, days)))
        events = []
        if i % 50 == 0:
            close[days // 2 :] /= 2.0
            events.append((dates[days // 2], "split", 2.0))
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


def run_on(core: str, data: MarketData, config: BacktestConfig) -> EngineOutput:
    factory = rust_core.RustCore if core == "rust" else PythonCore
    strategy = build_strategy(config.strategy, "event", config.params)
    assert isinstance(strategy, EventStrategy)
    return run_event(strategy, data, config, CostModel(config.costs, data.symbols), factory)


def assert_identical(a: EngineOutput, b: EngineOutput) -> None:
    for name in ("accounts", "positions", "trades", "costs", "orders"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert a.final_state == b.final_state


REFERENCE = [
    ("buy_and_hold", {}, "monthly"),
    ("sma_crossover", {"fast": 5, "slow": 20}, "daily"),
    ("xs_momentum", {"lookback": 60, "skip": 5, "top_n": 3}, "weekly"),
]


@needs_rust
@pytest.mark.parametrize(("strategy", "params", "rebalance"), REFERENCE)
def test_reference_strategies_are_identical_on_both_cores(
    strategy: str, params: dict[str, Any], rebalance: str
) -> None:
    data = synthetic_market(6, 400, seed=3, dividend=0.4, split_on=200)
    config = make_config(
        data,
        engine="event",
        strategy=strategy,
        params=params,
        rebalance=rebalance,
        costs={"impact_bps": 20, "max_participation": 0.05},  # every cost rule switched on
        constraints={"long_only": False, "max_gross_leverage": 1.0},
    )
    python, rust = run_on("python", data, config), run_on("rust", data, config)
    assert len(python.trades) > 10
    assert_identical(python, rust)


@needs_rust
def test_the_stress_market_is_identical_on_both_cores() -> None:
    data = stress_market()
    config = make_config(
        data,
        engine="event",
        strategy="xs_momentum",
        params={"lookback": 252, "skip": 21, "top_n": 50},
        rebalance="monthly",
        costs={"commission_model": "ibkr_tiered", "spread_bps": 1, "slippage_bps": 2},
        constraints={"long_only": True, "max_gross_leverage": 1.0},
    )
    python, rust = run_on("python", data, config), run_on("rust", data, config)
    assert len(python.trades) > 1000
    assert_identical(python, rust)


@needs_rust
def test_rust_runs_are_deterministic() -> None:
    data = synthetic_market(6, 300, seed=5, dividend=0.3)
    config = make_config(
        data, engine="event", strategy="sma_crossover", params={"fast": 5, "slow": 20}
    )
    first, second = run_on("rust", data, config), run_on("rust", data, config)
    assert_identical(first, second)
    # Bit for bit: compare the raw bytes of every float, not just their values.
    assert first.equity.to_numpy().tobytes() == second.equity.to_numpy().tobytes()


# --------------------------------------------------------------------------- the bar store


@needs_rust
def test_history_views_of_the_rust_store_cannot_be_made_writable() -> None:
    data = synthetic_market(2, 30)
    config = make_config(data, engine="event")
    core = rust_core.RustCore(data, config, CostModel(config.costs, data.symbols))
    for _ in range(10):
        core.step()
    view = core.history("S00", 5, adjusted=False)
    assert not view.flags.owndata  # a view of the Rust store, not a copy
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = 0.0
    # Unlike a view of a NumPy array, a view of Rust memory cannot be switched back.
    with pytest.raises(ValueError):
        view.flags.writeable = True
    expected = data.panel("close")[5:10, 0]
    assert view.tolist() == expected.tolist()


@needs_rust
def test_a_view_keeps_the_rust_store_alive() -> None:
    data = synthetic_market(1, 30)
    config = make_config(data, engine="event")
    core = rust_core.RustCore(data, config, CostModel(config.costs, data.symbols))
    core.step()
    view = core.history("S00", 1, adjusted=False)
    expected = view.tolist()
    del core  # the array's base object still holds the engine
    assert view.tolist() == expected
