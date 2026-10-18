"""Computing target weights as of a date, on the real engine, from a real DuckDB store."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from qr.live.decide import DecisionError, target_weights

from ..bt_helpers import synthetic_bars
from .live_store import seed_store


def _three_symbol_store(data_dir: Path) -> tuple[Path, list[str], date]:
    symbols = ["AAA", "BBB", "CCC"]
    bars = {
        s: synthetic_bars(400, seed=i, drift=0.0003 * (i + 1))[0] for i, s in enumerate(symbols)
    }
    seed_store(data_dir, bars)
    as_of = bars["AAA"]["date"].iloc[-1].date()
    return data_dir, symbols, as_of


def test_buy_and_hold_targets_equal_weight(tmp_path: Path) -> None:
    data_dir, symbols, as_of = _three_symbol_store(tmp_path)
    weights = target_weights("buy_and_hold", {}, symbols, as_of, data_dir=data_dir)
    assert weights == {s: pytest.approx(1 / 3) for s in symbols}


def test_xs_momentum_targets_the_top_ranked_symbols(tmp_path: Path) -> None:
    data_dir, symbols, as_of = _three_symbol_store(tmp_path)
    lookback, skip, top_n = 120, 5, 2
    weights = target_weights(
        "xs_momentum", {"lookback": lookback, "skip": skip, "top_n": top_n}, symbols, as_of,
        data_dir=data_dir,
    )  # fmt: skip

    # Reproduce XsMomentumEvents.on_bar's own ranking formula independently, from the
    # same raw closes (there are no corporate actions, so unadjusted == adjusted here).
    closes = {
        s: synthetic_bars(400, seed=i, drift=0.0003 * (i + 1))[0]["close"].to_numpy()
        for i, s in enumerate(symbols)
    }
    momentum = {s: c[-1 - skip] / c[-1 - lookback] - 1.0 for s, c in closes.items()}
    winners = sorted(momentum, key=lambda s: -momentum[s])[:top_n]

    assert weights.keys() == set(winners)
    assert all(w == pytest.approx(1 / top_n) for w in weights.values())


def test_unregistered_strategy_is_a_decision_error(tmp_path: Path) -> None:
    data_dir, symbols, as_of = _three_symbol_store(tmp_path)
    with pytest.raises(DecisionError, match="unknown strategy"):
        target_weights("not_a_real_strategy", {}, symbols, as_of, data_dir=data_dir)


def test_vectorized_only_strategy_is_a_decision_error(tmp_path: Path) -> None:
    data_dir, symbols, as_of = _three_symbol_store(tmp_path)
    with pytest.raises(DecisionError, match="event"):
        target_weights("pairs", {}, symbols, as_of, data_dir=data_dir)


def test_no_session_on_as_of_is_a_decision_error(tmp_path: Path) -> None:
    data_dir, symbols, as_of = _three_symbol_store(tmp_path)
    from datetime import timedelta

    with pytest.raises(DecisionError):
        target_weights("buy_and_hold", {}, symbols, as_of + timedelta(days=3650), data_dir=data_dir)
