"""Walk-forward folds, selection without look-ahead, stitching, logging, and the CLI."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qr import storage
from qr.backtest import registry
from qr.backtest.config import load_backtest_config
from qr.backtest.strategies import STRATEGIES, register
from qr.backtest.strategy import WeightStrategy, closes
from qr.cli import app
from qr.config import db_path
from qr.research import walkforward as wf_module
from qr.research.splits import WALK_FORWARD
from qr.research.walkforward import (
    ANCHORED,
    ROLLING,
    Fold,
    best_sharpe,
    expand_grid,
    make_folds,
    walk_forward,
)

from .conftest import AFTER_JAN_16

runner = CliRunner()


# ------------------------------------------------------------------ folds


def test_anchored_folds() -> None:
    folds = make_folds(date(2005, 1, 1), date(2019, 12, 31), ANCHORED, 5, 12, 12)
    assert len(folds) == 10
    assert folds[0] == Fold(0, date(2005, 1, 1), date(2009, 12, 31), date(2010, 1, 1),
                            date(2010, 12, 31))  # fmt: skip
    assert folds[-1] == Fold(9, date(2005, 1, 1), date(2018, 12, 31), date(2019, 1, 1),
                             date(2019, 12, 31))  # fmt: skip
    assert all(f.train_start == date(2005, 1, 1) for f in folds)


def test_rolling_folds_and_the_embargo() -> None:
    folds = make_folds(date(2010, 1, 1), date(2014, 12, 31), ROLLING, 2, 6, 6, embargo_days=5)
    assert [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds] == [
        (date(2010, 1, 1), date(2011, 12, 26), date(2012, 1, 1), date(2012, 6, 30)),
        (date(2010, 7, 1), date(2012, 6, 25), date(2012, 7, 1), date(2012, 12, 31)),
        (date(2011, 1, 1), date(2012, 12, 26), date(2013, 1, 1), date(2013, 6, 30)),
        (date(2011, 7, 1), date(2013, 6, 25), date(2013, 7, 1), date(2013, 12, 31)),
        (date(2012, 1, 1), date(2013, 12, 26), date(2014, 1, 1), date(2014, 6, 30)),
        (date(2012, 7, 1), date(2014, 6, 25), date(2014, 7, 1), date(2014, 12, 31)),
    ]


def test_bad_fold_settings() -> None:
    with pytest.raises(ValueError, match="unknown walk-forward mode"):
        make_folds(date(2010, 1, 1), date(2020, 1, 1), "sideways", 2, 12, 12)
    with pytest.raises(ValueError, match="no complete fold"):
        make_folds(date(2010, 1, 1), date(2012, 6, 1), ANCHORED, 2, 12, 12)
    with pytest.raises(ValueError, match="at least 1"):
        make_folds(date(2010, 1, 1), date(2020, 1, 1), ANCHORED, 0, 12, 12)


def test_grid_and_default_objective() -> None:
    assert expand_grid(None) == [{}]
    assert expand_grid({"a": [1, 2], "b": ["x"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]
    with pytest.raises(ValueError, match="non-empty list"):
        expand_grid({"a": []})
    tried = [({"a": 1}, {"sharpe": float("nan")}), ({"a": 2}, {"sharpe": 0.5}),
             ({"a": 3}, {"sharpe": 0.5}), ({"a": 4}, {"sharpe": -1.0})]  # fmt: skip
    assert best_sharpe(tried) == 1  # NaN last, ties to the first


# ------------------------------------------------------------------ a known answer

# One symbol whose drift flips sign every calendar year: up, up, down, down, up, up.
YEARS = [2010, 2011, 2012, 2013, 2014, 2015]
DRIFT = {2010: 1, 2011: 1, 2012: -1, 2013: -1, 2014: 1, 2015: 1}
SPLIT = """
name: regimes
segments:
  train:      { start: 2010-01-01, end: 2013-12-31 }
  validation: { start: 2014-01-01, end: 2015-12-31 }
  test:       { start: 2016-01-01, end: null, sealed: true }
embargo_days: 0
"""
UNIVERSE = """
start_date: 2010-01-01
universe:
  - { symbol: TRD, source: yahoo, asset_class: equity }
"""


class Side(WeightStrategy):
    """All in (``side=1``) or all short (``side=-1``), every day. Records what it saw."""

    seen: list[pd.Timestamp] = []  # noqa: RUF012 - shared on purpose, reset per test

    def __init__(self, side: int = 1) -> None:
        self.side = side

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        prices = closes(bars)
        Side.seen.append(prices.index.max())
        return pd.DataFrame(float(self.side), index=prices.index, columns=prices.columns)


@pytest.fixture
def regimes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "configs" / "splits").mkdir(parents=True)
    (tmp_path / "configs" / "splits" / "regimes.yaml").write_text(SPLIT, encoding="utf-8")
    (tmp_path / "universe.yaml").write_text(UNIVERSE, encoding="utf-8")
    monkeypatch.setenv("QR_CONFIG", str(tmp_path / "universe.yaml"))
    monkeypatch.setenv("QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.delenv("QR_SPLITS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    dates = pd.bdate_range("2009-06-01", "2016-06-30")
    rng = np.random.default_rng(1)
    drift = np.array([DRIFT.get(d.year, 1) * 0.002 for d in dates])
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, len(dates))))
    frame = pd.DataFrame({"date": dates, "open": close, "high": close * 1.001,
                          "low": close * 0.999, "close": close, "volume": 1e6})  # fmt: skip
    with storage.connect(db_path(tmp_path / "data")) as con:
        storage.init_schema(con)
        storage.upsert_instrument(con, "TRD", "yahoo", "equity", True, AFTER_JAN_16)
        storage.upsert_bars(con, "TRD", "yahoo", frame, AFTER_JAN_16)
    register("side", vectorized=Side)
    Side.seen = []
    yield tmp_path
    STRATEGIES.pop("side", None)


def side_config(regimes: Path) -> Path:
    path = regimes / "side.yaml"
    path.write_text(
        "name: side\nengine: vectorized\nstrategy: side\nuniverse: [TRD]\n"
        "split: regimes\nsegment: train\nbenchmark: TRD\n"
        "costs: { commission_model: none, spread_bps: 0, slippage_bps: 0, borrow_rate: 0 }\n"
        "constraints: { long_only: false }\n",
        encoding="utf-8",
    )
    return path


def test_picks_the_known_best_side_each_fold_without_looking_ahead(
    regimes: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[date] = []
    real = wf_module.load_market_data

    def spy(symbols: list[str], end: date, **kwargs: object):  # type: ignore[no-untyped-def]
        loads.append(end)
        return real(symbols, end, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(wf_module, "load_market_data", spy)
    config = load_backtest_config(side_config(regimes))
    # Rolling one-year windows: each fold trains on one year, tests on the next.
    result = walk_forward(config, "regimes", ROLLING, 1, 12, 12, grid={"side": [1, -1]})
    assert [f.fold.test_start.year for f in result.folds] == [2011, 2012, 2013, 2014, 2015]
    # The best side on a year is that year's drift.
    assert [f.chosen["side"] for f in result.folds] == [DRIFT[y] for y in YEARS[:5]]
    assert result.changes == 2 and not result.unstable
    # Data for selection never reached past the fold's train end.
    for i, fold in enumerate(result.folds):
        assert loads[2 * i] == fold.fold.train_end and loads[2 * i + 1] == fold.fold.test_end
    assert max(Side.seen) <= pd.Timestamp(result.folds[-1].fold.test_end)
    assert all(f.test.label == WALK_FORWARD for f in result.folds)
    # Losing years are the ones where the regime flipped after the choice.
    test_sharpes = [f.test.metrics["sharpe"] for f in result.folds]
    assert [s > 0 for s in test_sharpes] == [True, False, True, False, True]


def test_stitched_curve_and_logging(regimes: Path) -> None:
    config = load_backtest_config(side_config(regimes))
    result = walk_forward(config, "regimes", ANCHORED, 2, 12, 12, grid={"side": [1, -1]})
    assert len(result.folds) == 4 and result.runs_logged == 4 * 3
    # The stitched curve chains the test windows, so it never jumps between them.
    first, second = result.folds[0].test.equity, result.folds[1].test.equity
    growth = result.equity.loc[second.index] / result.equity.loc[first.index[-1]]
    np.testing.assert_allclose(growth.to_numpy(), (second / config.initial_cash).to_numpy())
    assert result.equity.index.is_monotonic_increasing and result.equity.index.is_unique
    assert result.equity.index[-1] <= pd.Timestamp("2015-12-31")  # the sealed test is untouched
    # Every run is in the registry, hidden from the normal list.
    assert registry.read_runs().empty
    logged = registry.read_runs(include_walkforward=True)
    assert len(logged) == 12 and set(logged["walkforward_id"]) == {result.walkforward_id}
    assert sorted(logged["fold"].unique().tolist()) == [0, 1, 2, 3]
    assert (logged["result_label"] == WALK_FORWARD).sum() == 4
    assert logged["data_hash"].notna().all()


def test_without_a_grid_the_config_params_run_in_every_fold(regimes: Path) -> None:
    config = load_backtest_config(side_config(regimes))
    result = walk_forward(config, "regimes", ANCHORED, 2, 12, 12, log=False)
    assert [f.chosen for f in result.folds] == [{}] * 4 and result.runs_logged == 0
    assert registry.read_runs(include_walkforward=True).empty


def test_cli_walkforward(regimes: Path) -> None:
    path = side_config(regimes)
    grid = regimes / "grid.yaml"
    grid.write_text("side: [1, -1]\n", encoding="utf-8")
    ran = runner.invoke(
        app, ["backtest", "walkforward", str(path), "--mode", "rolling", "--train-years", "1",
              "--grid", str(grid)],
    )  # fmt: skip
    assert ran.exit_code == 0, ran.output
    assert "changed in" in ran.output and "Runs logged: 15" in ran.output
    page = next((regimes / "reports").glob("*_walkforward.html")).read_text(encoding="utf-8")
    assert "out-of-sample (walk-forward)" in page and "side=-1" in page
    bad = runner.invoke(app, ["backtest", "walkforward", str(path), "--mode", "sideways"])
    assert bad.exit_code == 1 and "unknown walk-forward mode" in bad.output
