"""The two cross-checks the engines are built around.

* **Parity**: the vectorized and the event-driven engine are independent implementations
  (adjusted prices and dollar positions against as-traded prices, shares, splits, and
  cash dividends). On simple strategies they must tell the same story.
* **Look-ahead**: changing the data after a date must not change anything before it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from qr.backtest.data import MarketData
from qr.backtest.result import EngineOutput

from .bt_helpers import make_config, run, synthetic_bars, synthetic_market

# Every test here runs on both event-engine cores (see the engine_core fixture).
pytestmark = pytest.mark.usefixtures("engine_core")

STRATEGIES: list[tuple[str, dict[str, Any], str]] = [
    ("buy_and_hold", {}, "daily"),
    ("sma_crossover", {"fast": 10, "slow": 40}, "daily"),
]


def both_engines(data: MarketData, **config: Any) -> tuple[EngineOutput, EngineOutput]:
    base = {"initial_cash": 1_000_000, **config}
    vectorized = run(data, make_config(data, engine="vectorized", **base))
    event = run(data, make_config(data, engine="event", **base))
    return vectorized, event


def worst_gap(vectorized: EngineOutput, event: EngineOutput) -> float:
    return float((vectorized.equity / event.equity - 1.0).abs().max())


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize(("strategy", "params", "rebalance"), STRATEGIES)
def test_engines_agree_without_costs(strategy: str, params: dict[str, Any], rebalance: str) -> None:
    data = synthetic_market(1, 750, seed=3)
    vectorized, event = both_engines(data, strategy=strategy, params=params, rebalance=rebalance)
    # The only difference is whole shares: at most one share of about 100 in a million.
    assert worst_gap(vectorized, event) < 3e-4
    assert len(event.trades) > 0
    if strategy == "sma_crossover":
        assert len(event.trades) > 4  # it really did go in and out


@pytest.mark.parametrize(("strategy", "params", "rebalance"), STRATEGIES)
def test_engines_agree_with_costs(strategy: str, params: dict[str, Any], rebalance: str) -> None:
    data = synthetic_market(1, 750, seed=3)
    costs = {"commission_model": "ibkr_tiered", "spread_bps": 1, "slippage_bps": 2}
    vectorized, event = both_engines(
        data, strategy=strategy, params=params, rebalance=rebalance, costs=costs,
        constraints={"min_trade_weight": 0.001},
    )  # fmt: skip
    assert worst_gap(vectorized, event) < 3e-4
    paid_v, paid_e = vectorized.costs.to_numpy().sum(), event.costs.to_numpy().sum()
    assert paid_e > 0
    assert paid_v == pytest.approx(paid_e, rel=0.05)


def test_engines_agree_through_splits_and_dividends() -> None:
    """Adjusted prices on one side, real splits and cash dividends on the other."""
    frame, actions = synthetic_bars(750, seed=5, dividend=0.6, split_on=400)
    assert set(actions["action"]) == {"split", "dividend"}
    data = MarketData.from_frames({"AAA": frame}, {"AAA": actions})
    for strategy, params, rebalance in STRATEGIES:
        vectorized, event = both_engines(
            data, strategy=strategy, params=params, rebalance=rebalance
        )
        # Dividends sit in cash until they buy a whole share, so the gap is a little wider.
        assert worst_gap(vectorized, event) < 1e-3, strategy
    assert event.final_state is not None
    assert event.final_state.realized_pnl > 0  # dividends were received in cash


def test_engines_agree_on_a_monthly_momentum_portfolio() -> None:
    data = synthetic_market(8, 600, seed=11)
    vectorized, event = both_engines(
        data,
        strategy="xs_momentum",
        params={"lookback": 120, "skip": 10, "top_n": 3},
        rebalance="monthly",
        start=data.dates[130].date(),
    )
    assert worst_gap(vectorized, event) < 2e-3
    # Both hold the same three names at the end.
    held_v = set(vectorized.positions.columns[vectorized.positions.iloc[-1] > 0])
    held_e = set(event.positions.columns[event.positions.iloc[-1] > 0])
    assert held_v == held_e and len(held_v) == 3


# --------------------------------------------------------------------------- look-ahead


def perturbed_after(data: MarketData, cut: pd.Timestamp, seed: int = 99) -> MarketData:
    """A copy of ``data`` with every bar after ``cut`` replaced by different prices."""
    rng = np.random.default_rng(seed)
    frames = {}
    for symbol in data.symbols:
        own = data.raw.xs(symbol, level="symbol").reset_index()
        later = own["date"] > cut
        shock = rng.uniform(0.5, 1.5, int(later.sum()))
        for column in ("open", "high", "low", "close"):
            own.loc[later, column] = own.loc[later, column].to_numpy() * shock
        own.loc[later, "volume"] = own.loc[later, "volume"].to_numpy() * rng.uniform(0.1, 3.0)
        frames[symbol] = own
    return MarketData.from_frames(frames)


@pytest.mark.parametrize("engine", ["vectorized", "event"])
@pytest.mark.parametrize(
    ("strategy", "params", "rebalance"),
    [
        ("sma_crossover", {"fast": 5, "slow": 20}, "daily"),
        ("xs_momentum", {"lookback": 60, "skip": 5, "top_n": 2}, "weekly"),
        ("buy_and_hold", {}, "monthly"),
    ],
)
def test_changing_the_future_does_not_change_the_past(
    engine: str, strategy: str, params: dict[str, Any], rebalance: str
) -> None:
    data = synthetic_market(4, 300, seed=7)
    cut = data.dates[200]
    config = make_config(
        data,
        engine=engine,
        strategy=strategy,
        params=params,
        rebalance=rebalance,
        costs={"impact_bps": 20},  # default costs plus one that depends on volume
        start=data.dates[70].date(),
    )
    original = run(data, config)
    changed = run(perturbed_after(data, cut), config)

    assert not original.equity.equals(changed.equity)  # the future really is different
    for name in ("accounts", "positions", "costs"):
        before, after = getattr(original, name).loc[:cut], getattr(changed, name).loc[:cut]
        pd.testing.assert_frame_equal(before, after, check_exact=True)
    # Every trade up to the cut, and every decision made up to it.
    trades = [o.trades[o.trades["date"] <= cut].reset_index(drop=True) for o in (original, changed)]
    pd.testing.assert_frame_equal(trades[0], trades[1], check_exact=True)
    assert len(trades[0]) > 0
    if engine == "vectorized":
        assert original.weights is not None and changed.weights is not None
        pd.testing.assert_frame_equal(
            original.weights.loc[:cut], changed.weights.loc[:cut], check_exact=True
        )
    else:
        assert original.orders is not None and changed.orders is not None
        stamp = cut.isoformat()
        orders = [
            o[o["submitted_at"] <= stamp][["id", "symbol", "side", "quantity", "type"]]
            for o in (original.orders, changed.orders)
        ]
        pd.testing.assert_frame_equal(orders[0], orders[1], check_exact=True)
