"""Pairs-trading research functions and the ``pairs`` strategy, on synthetic data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from qr.backtest.data import MarketData
from qr.backtest.strategies import build_strategy
from qr.backtest.strategies.pairs import PairsWeights
from qr.research.pairs import (
    engle_granger,
    half_life,
    hedge_ratio_ols,
    log_prices,
    pair_signals,
    pair_weights,
    rolling_cointegration_pvalue,
    rolling_hedge_ratio,
    rolling_spread_zscore,
    rolling_zscore,
    round_trip_returns,
    round_trips,
    spread,
)
from tests.bt_helpers import make_config, run

DATES = pd.bdate_range("2015-01-02", periods=1500)


def random_walk(seed: int, n: int = len(DATES), scale: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0.0, scale, n)), index=DATES[:n])


def ar1(phi: float, n: int, seed: int, scale: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = phi * values[i - 1] + rng.normal(0.0, scale)
    return pd.Series(values, index=pd.bdate_range(DATES[0], periods=n))


def cointegrated_pair(beta: float = 1.5, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """``x`` is a random walk; ``y = beta * x + stationary noise``."""
    x = random_walk(seed)
    y = beta * x + ar1(0.9, len(x), seed + 100)
    return y.rename("y"), x.rename("x")


# --------------------------------------------------------------------------- hedge ratio


def test_hedge_ratio_recovers_known_beta() -> None:
    y, x = cointegrated_pair(beta=1.5)
    assert hedge_ratio_ols(y, x) == pytest.approx(1.5, abs=0.05)


def test_rolling_hedge_ratio_recovers_beta_and_is_shifted_by_one_day() -> None:
    y, x = cointegrated_pair(beta=0.8, seed=3)
    beta = rolling_hedge_ratio(y, x, 252)
    assert beta.iloc[:252].isna().all()  # day 252 is the first with 252 earlier days
    assert beta.iloc[252] == pytest.approx(hedge_ratio_ols(y.iloc[:252], x.iloc[:252]))
    assert beta.dropna().median() == pytest.approx(0.8, abs=0.1)


def test_log_prices_start_at_zero_and_ignore_the_anchor() -> None:
    closes = pd.DataFrame({"a": [np.nan, 10.0, 11.0, 12.1], "b": [5.0, 5.0, 4.0, 2.0]})
    out = log_prices(closes)
    assert out["a"].iloc[1] == 0.0 and out["b"].iloc[0] == 0.0
    assert out["a"].iloc[3] == pytest.approx(math.log(1.21))
    # Scaling every price (as a later dividend does to back-adjusted levels) changes nothing.
    pd.testing.assert_frame_equal(log_prices(closes * 0.97), out)


# --------------------------------------------------------------------------- no look-ahead


ROLLING = {
    "hedge_ratio": lambda y, x: rolling_hedge_ratio(y, x, 120),
    "zscore": lambda y, x: rolling_zscore(spread(y, x, 1.2), 60),
    "spread_zscore": lambda y, x: rolling_spread_zscore(y, x, rolling_hedge_ratio(y, x, 120), 60),
    "coint_pvalue": lambda y, x: rolling_cointegration_pvalue(y, x, 120, 20),
    "signals": lambda y, x: pair_signals(rolling_zscore(spread(y, x, 1.2), 60), 1.5, 0.3, 3.0),
}


@pytest.mark.parametrize("name", list(ROLLING))
def test_rolling_functions_never_use_future_data(name: str) -> None:
    y, x = cointegrated_pair(beta=1.2, seed=7)
    cut = 800
    before = ROLLING[name](y, x)
    noise = np.random.default_rng(99).normal(0.0, 0.5, len(y) - cut - 1)
    y2, x2 = y.copy(), x.copy()
    y2.iloc[cut + 1 :] += noise
    x2.iloc[cut + 1 :] -= noise
    after = ROLLING[name](y2, x2)
    upto = before.index <= y.index[cut]
    pd.testing.assert_series_equal(before[upto], after[upto])
    assert not before[~upto].equals(after[~upto])  # the perturbation did reach later rows


def test_spread_zscore_equals_plain_zscore_with_a_constant_beta() -> None:
    y, x = cointegrated_pair(beta=1.1, seed=5)
    plain = rolling_zscore(spread(y, x, 1.1), 60)
    pd.testing.assert_series_equal(rolling_spread_zscore(y, x, 1.1, 60), plain)


# --------------------------------------------------------------------------- half-life


def test_half_life_recovers_known_value() -> None:
    phi = 0.95  # half-life ln(0.5) / ln(0.95) = 13.5 days
    s = ar1(phi, 20_000, seed=11)
    assert half_life(s) == pytest.approx(math.log(0.5) / math.log(phi), rel=0.1)


def test_half_life_of_a_trending_series_is_infinite() -> None:
    assert half_life(pd.Series(np.exp(np.linspace(0.0, 1.0, 100)))) == math.inf


# --------------------------------------------------------------------------- cointegration


def test_engle_granger_separates_cointegrated_from_independent() -> None:
    y, x = cointegrated_pair(beta=1.5, seed=1)
    found = engle_granger(y, x)
    assert found["pvalue"] < 0.01
    assert set(found["critical_values"]) == {"1%", "5%", "10%"}
    assert found["observations"] == len(y)
    independent = engle_granger(random_walk(21), random_walk(22))
    assert independent["pvalue"] > 0.2


def test_rolling_cointegration_pvalue_is_indexed_by_window_end() -> None:
    y, x = cointegrated_pair(seed=2)
    pvalues = rolling_cointegration_pvalue(y, x, window=250, step=50)
    assert pvalues.index[0] == y.index[249]
    assert (pvalues.index[1:] - pvalues.index[:-1]).size == len(pvalues) - 1
    assert len(pvalues) == len(range(249, len(y), 50))
    assert pvalues.between(0, 1).all()


# --------------------------------------------------------------------------- signals


def test_signal_state_machine_enters_exits_and_stops() -> None:
    z = pd.Series([0.0, 1.5, 2.1, 1.0, 0.4, -2.5, -3.0, -4.5, -2.5, -0.2, -2.2, np.nan, -2.2, 0.6])
    expected = [0, 0, -1, -1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    # 2.1: short (above entry). 0.4: exit (inside 0.5). -2.5: long. -4.5: stopped out.
    # -2.5 right after the stop: blocked. -0.2: unblocked. -2.2: long. NaN: flat.
    assert pair_signals(z, entry=2.0, exit=0.5, stop=4.0).tolist() == expected


def test_signal_does_not_enter_beyond_the_stop_and_exits_when_crossing_the_mean() -> None:
    z = pd.Series([4.2, 3.0, -0.8, 5.0])
    assert pair_signals(z, entry=2.0, exit=0.0, stop=4.0).tolist() == [0, -1, 0, 0]


def test_signal_rejects_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError):
        pair_signals(pd.Series([0.0]), entry=1.0, exit=1.5, stop=3.0)


# --------------------------------------------------------------------------- weights


def test_long_spread_is_long_y_short_x_in_hedge_proportions() -> None:
    signal = pd.Series([0, 1, 1, 0, -1])
    beta = pd.Series([0.5, 0.8, 2.0, 2.0, 1.0])
    weights = pair_weights(signal, beta, gross=1.0, names=("KO", "PEP"))
    assert list(weights.columns) == ["KO", "PEP"]
    # Day 1 opens a long spread with beta 0.8: KO +1/1.8, PEP -0.8/1.8.
    assert weights.iloc[1].tolist() == pytest.approx([1 / 1.8, -0.8 / 1.8])
    # Day 2 keeps the beta of the day the trade opened.
    assert weights.iloc[2].tolist() == pytest.approx([1 / 1.8, -0.8 / 1.8])
    assert weights.iloc[3].tolist() == [0.0, 0.0]
    assert weights.iloc[4].tolist() == pytest.approx([-0.5, 0.5])
    assert weights.abs().sum(axis=1).iloc[[1, 2, 4]].tolist() == pytest.approx([1.0] * 3)


def test_weights_without_a_hedge_ratio_are_zero() -> None:
    weights = pair_weights(pd.Series([1, 1]), pd.Series([np.nan, 1.0]), gross=2.0)
    assert weights.to_numpy().tolist() == [[0.0, 0.0], [0.0, 0.0]]


# --------------------------------------------------------------------------- round trips


def test_round_trips_and_their_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    signal = pd.Series([0, 1, 1, 0, -1, 1, 1, 1], index=dates)
    trips = round_trips(signal)
    assert trips["side"].tolist() == [1, -1, 1]
    assert trips["days"].tolist() == [2, 1, 3]
    assert trips["opened"].tolist() == [dates[1], dates[4], dates[5]]
    assert pd.isna(trips["closed"].iloc[-1])
    equity = pd.Series([100.0, 100, 101, 102, 102, 101, 103, 104], index=dates)
    # First trip: decided at the close of day 1, closed by the fill on day 4.
    returns = round_trip_returns(trips, equity)
    assert returns.tolist() == pytest.approx([102 / 100 - 1, 103 / 102 - 1, 104 / 101 - 1])


# --------------------------------------------------------------------------- strategy


def pair_market(days: int = 900) -> MarketData:
    """Two symbols whose log prices are cointegrated, as bars the engines accept."""
    rng = np.random.default_rng(4)
    log_x = np.cumsum(rng.normal(0.0003, 0.012, days))
    noise = np.zeros(days)
    for i in range(1, days):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0.0, 0.01)
    log_y = 0.9 * log_x + noise
    dates = pd.bdate_range("2016-01-04", periods=days)
    frames = {}
    for name, path in (("YY", 50 * np.exp(log_y)), ("XX", 80 * np.exp(log_x))):
        frames[name] = pd.DataFrame(
            {
                "date": dates,
                "open": path,
                "high": path * 1.001,
                "low": path * 0.999,
                "close": path,
                "volume": 1e6,
            }
        )
    return MarketData.from_frames(frames, {})


def test_pairs_strategy_runs_in_the_vectorized_engine_without_looking_ahead() -> None:
    data = pair_market()
    params = {
        "y": "YY",
        "x": "XX",
        "beta_window": 120,
        "z_window": 40,
        "entry": 1.5,
        "exit": 0.3,
        "stop": 4.0,
    }
    config = make_config(
        data,
        engine="vectorized",
        strategy="pairs",
        params=params,
        lookahead_checks=10,
        execution="next_open",
        constraints={"max_gross_leverage": 1.0},
    )
    output = run(data, config)  # raises LookAheadError if the weights use the future
    assert len(output.trades) > 0
    weights = output.weights
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-9).all()
    long_days = weights["YY"] > 0
    assert long_days.any() and (weights.loc[long_days, "XX"] < 0).all()


def test_pairs_strategy_is_registered_and_validates_params() -> None:
    strategy = build_strategy("pairs", "vectorized", {"y": "KO", "x": "PEP"})
    assert isinstance(strategy, PairsWeights)
    with pytest.raises(ValueError, match="no implementation for the event engine"):
        build_strategy("pairs", "event", {"y": "KO", "x": "PEP"})
    with pytest.raises(ValueError):
        PairsWeights("KO", "KO")


def test_fixed_beta_param_replaces_the_rolling_hedge_ratio() -> None:
    data = pair_market(400)
    strategy = PairsWeights("YY", "XX", beta=0.9)
    assert (strategy.hedge_ratio(data.adjusted) == 0.9).all()
