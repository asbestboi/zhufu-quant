"""Execution assumptions and the cost stress test (research-discipline part 2)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qr.backtest import registry
from qr.backtest.assumptions import build_assumptions
from qr.backtest.config import BacktestConfig, CostConfig
from qr.backtest.costs import CostModel
from qr.backtest.runner import execute, run_backtest
from qr.backtest.stress import (
    ABOVE,
    INSIDE,
    NONE,
    CostSensitivity,
    StressPoint,
    break_even,
    is_fragile,
)
from qr.cli import app

from .bt_helpers import make_config, synthetic_market
from .test_bt_cli import config_text, project, write

pytestmark = pytest.mark.usefixtures("engine_core")
__all__ = ["project"]  # the fixture, imported for this module's tests

runner = CliRunner()
COSTS = {"commission_model": "ibkr_tiered", "spread_bps": 1, "slippage_bps": 2}


def market() -> object:
    return synthetic_market(4, 400, seed=3)


def momentum(data: object, engine: str, **overrides: object) -> BacktestConfig:
    settings: dict[str, object] = {
        "engine": engine,
        "strategy": "xs_momentum",
        "params": {"lookback": 40, "skip": 5, "top_n": 2},
        "rebalance": "weekly",
        "costs": COSTS,
        **overrides,
    }
    return make_config(data, **settings)  # type: ignore[arg-type]


# ------------------------------------------------------------------ multipliers


@pytest.mark.parametrize("engine", ["vectorized", "event"])
def test_zero_x_equals_before_costs_and_one_x_is_unchanged(engine: str) -> None:
    data = market()
    config = momentum(data, engine)
    plain = run_backtest(config, data, cost_stress=False)  # type: ignore[arg-type]
    stressed = run_backtest(config, data)  # type: ignore[arg-type]
    assert stressed.metrics == plain.metrics  # 1x: the run itself, untouched
    pd.testing.assert_series_equal(stressed.equity, plain.equity)
    assert stressed.stress_metrics[0.0] == plain.gross_metrics  # 0x: before costs
    assert stressed.cost_sensitivity is not None
    assert [p.multiplier for p in stressed.cost_sensitivity.points] == [0.0, 1.0, 2.0]
    two = stressed.stress_metrics[2.0]
    assert two["total_costs"] > plain.metrics["total_costs"] > 0
    assert two["cagr"] < plain.metrics["cagr"]


def test_a_multiplier_of_one_leaves_the_config_hash_alone() -> None:
    data = market()
    config = momentum(data, "vectorized")
    explicit = momentum(data, "vectorized", cost_multiplier=1.0, cost_stress=[0, 1, 2, 3])
    assert explicit.config_hash() == config.config_hash()
    assert momentum(data, "vectorized", cost_multiplier=2.0).config_hash() != config.config_hash()
    # The hash of a config written before these options existed is unchanged.
    old = config.model_dump(mode="json", exclude={"engine_core", "cost_multiplier", "cost_stress"})
    import hashlib

    text = json.dumps(old, sort_keys=True, separators=(",", ":"))
    assert config.config_hash() == hashlib.sha256(text.encode()).hexdigest()[:16]


def test_bad_stress_lists_are_refused() -> None:
    data = market()
    with pytest.raises(ValueError, match="zero or more"):
        momentum(data, "vectorized", cost_stress=[-1, 1])
    with pytest.raises(ValueError, match="repeats"):
        momentum(data, "vectorized", cost_stress=[1, 1])


def test_costs_scale_linearly_on_a_fixed_trade_list() -> None:
    # Hand-checked: IBKR tiered is 0.0035 USD/share, at least 0.35, at most 1% of value.
    # 200 shares at 50: commission 0.70; spread 1 bp of 10,000 = 1.00; slippage 2 bps = 2.00.
    # 10 shares at 50:  commission 0.035 -> the 0.35 minimum; spread 0.05; slippage 0.10.
    # 3 shares at 0.5:  commission 0.0105 -> the minimum 0.35 is above the 1% cap (0.015),
    #                   and the minimum wins; spread 0.00015; slippage 0.0003.
    config = CostConfig(**COSTS)  # type: ignore[arg-type]
    symbols = ["A", "B", "C"]
    shares = np.array([200.0, 10.0, 3.0])
    prices = np.array([50.0, 50.0, 0.5])
    expected = {
        "commission": [0.70, 0.35, 0.35],
        "spread": [1.00, 0.05, 0.00015],
        "slippage": [2.00, 0.10, 0.0003],
    }
    for multiplier in (0.0, 0.5, 1.0, 2.0, 3.7):
        costs = CostModel(config, symbols, multiplier=multiplier).trade_costs(shares, prices)
        for part, values in expected.items():
            np.testing.assert_allclose(getattr(costs, part), np.array(values) * multiplier)
        one = CostModel(config, symbols, multiplier=multiplier).order_costs("A", 200, 50.0)
        assert one.total == pytest.approx(3.70 * multiplier)
    model = CostModel(config, symbols, multiplier=2.0)
    assert model.borrow_fee("A", 36_000.0, 1) == pytest.approx(36_000 * 0.01 / 360 * 2)
    with pytest.raises(ValueError, match="zero or more"):
        CostModel(config, symbols, multiplier=-1.0)


# ------------------------------------------------------------------ verdicts


def point(multiplier: float, cagr: float, sharpe: float = 1.0) -> StressPoint:
    return StressPoint(multiplier, cagr, sharpe, -0.1, 1.0, 100.0 * multiplier)


def test_break_even_is_interpolated_between_stress_points() -> None:
    # CAGR 10% at 0x, 4% at 1x, -4% at 2x: zero halfway between 1x and 2x.
    found = break_even([point(2, -0.04), point(0, 0.10), point(1, 0.04)])
    assert found is not None and found.where == INSIDE
    assert found.multiplier == pytest.approx(1.5)
    assert found.describe() == "1.50x"
    # Exactly zero at a stress point is the break-even.
    exact = break_even([point(0, 0.05), point(1, 0.0), point(2, -0.05)])
    assert exact is not None and exact.multiplier == pytest.approx(1.0)


def test_break_even_outside_the_tested_range() -> None:
    above = break_even([point(0, 0.10), point(1, 0.08), point(2, 0.06)])
    assert above is not None and above.where == ABOVE and above.multiplier is None
    assert above.describe() == "above 2x"
    none = break_even([point(0, -0.01), point(1, -0.03), point(2, -0.05)])
    assert none is not None and none.where == NONE
    assert "not positive" in none.describe()
    assert break_even([point(1, math.nan)]) is None


def test_fragile_flag_both_ways() -> None:
    assert is_fragile([point(1, 0.05, 1.0), point(2, 0.03, 0.6)]) is False
    assert is_fragile([point(1, 0.05, 1.0), point(2, 0.03, 0.5)]) is False  # exactly half
    assert is_fragile([point(1, 0.05, 1.0), point(2, 0.01, 0.49)]) is True
    assert is_fragile([point(1, 0.01, 0.2), point(2, -0.01, -0.1)]) is True
    assert is_fragile([point(1, 0.05, 1.0), point(2, 0.0, math.nan)]) is True
    assert is_fragile([point(0, 0.05), point(1, 0.04)]) is None  # no 2x point


def test_sensitivity_round_trips_through_json() -> None:
    sensitivity = CostSensitivity.from_points([point(2, -0.02), point(0, 0.10), point(1, 0.04)])
    raw = json.loads(json.dumps(sensitivity.to_dict()))
    assert CostSensitivity.from_dict(raw) == sensitivity


# ------------------------------------------------------------------ assumptions


@pytest.mark.parametrize("engine", ["vectorized", "event"])
def test_assumptions_are_complete_for_both_engines(engine: str) -> None:
    data = market()
    result = run_backtest(momentum(data, engine), data)  # type: ignore[arg-type]
    assumptions = result.assumptions
    assert assumptions is not None
    sections = assumptions["sections"]
    assert list(sections) == ["Engine", "Costs", "Portfolio", "Data"]
    required = {
        "Engine": ["Engine", "Execution timing", "Fill prices", "Rebalance"],
        "Costs": ["Commission model", "Commission", "Spread", "Slippage",
                  "Participation cap", "Short borrow", "Cost multiplier", "Cost stress"],
        "Portfolio": ["Direction", "Gross leverage", "Weight per symbol", "Smallest trade",
                      "Share rounding", "Corporate actions"],
        "Data": ["Adjustment", "As of", "Source", "Universe"],
    }  # fmt: skip
    for title, items in required.items():
        assert all(item in sections[title] for item in items), title
        assert all(isinstance(v, str) and v for v in sections[title].values())
    assert sections["Engine"]["Execution timing"] == (
        "signal at the close of t, filled at the open of t+1"
    )
    assert "0.0035 USD per share" in sections["Costs"]["Commission"]
    assert sections["Costs"]["Spread"].startswith("1 bps")
    assert sections["Costs"]["Cost stress"] == "0x, 1x, 2x"
    rounding = "whole shares" if engine == "event" else "fractional shares"
    assert sections["Portfolio"]["Share rounding"] == rounding
    warnings = " ".join(assumptions["warnings"])
    assert "survivorship" in warnings
    assert "1.5 years" in warnings  # 400 business days
    json.dumps(assumptions)  # JSON-ready


def test_assumptions_change_with_the_config() -> None:
    data = market()
    base = momentum(data, "event")
    changed = momentum(
        data,
        "event",
        execution="next_close",
        rebalance="monthly",
        costs={**COSTS, "slippage_bps": 5, "impact_bps": 10, "max_participation": 0.1,
               "symbols": {"S01": {"spread_bps": 4}}},
        constraints={"long_only": True, "min_trade_weight": 0.02},
        cost_multiplier=1.5,
    )  # fmt: skip
    a = build_assumptions(base, data.symbols, {})  # type: ignore[attr-defined]
    b = build_assumptions(changed, data.symbols, {})  # type: ignore[attr-defined]
    assert a != b
    engine, costs, portfolio = (b["sections"][k] for k in ("Engine", "Costs", "Portfolio"))
    assert engine["Execution timing"].endswith("close of t+1")
    assert "last session of each month" in engine["Rebalance"]
    assert costs["Slippage"].startswith("fixed 5 bps") and "10 bps x" in costs["Slippage"]
    assert costs["Participation cap"].startswith("10%")
    assert costs["Cost multiplier"] == "1.5x"
    assert "spread 4 bps" in costs["Costs for S01"]
    assert "Costs for S00" not in costs
    assert portfolio["Direction"] == "long only"
    assert "0.02 of equity" in portfolio["Smallest trade"]


def test_warnings_for_few_trades_and_limit_orders() -> None:
    from qr.backtest.strategy import Context, EventStrategy

    class OneLimitBuy(EventStrategy):
        def on_bar(self, ctx: Context, bars: object) -> None:
            if not ctx.open_orders("S00") and not ctx.position("S00"):
                ctx.order_limit("S00", 10, 1000.0)  # far above the price: fills next open

    data = market()
    config = make_config(data, engine="event", costs=COSTS)  # type: ignore[arg-type]
    from .bt_helpers import run

    result = run_backtest(config, data, cost_stress=False)  # type: ignore[arg-type]
    output = run(data, config, OneLimitBuy())  # type: ignore[arg-type]
    result.net = output
    result.metrics = {**result.metrics, "trades": float(len(output.trades))}
    warnings = " ".join(build_assumptions(config, data.symbols, {}, result=result)["warnings"])  # type: ignore[attr-defined]
    assert "Limit orders were filled from daily" in warnings
    assert "Only 1 trades" in warnings


# ------------------------------------------------------------------ registry, report, CLI


def test_stress_runs_are_logged_as_children_without_reports(project: Path) -> None:
    path = write(project, "mom", config_text("xs_momentum", "event"))
    from qr.backtest.config import load_backtest_config

    outcome = execute(load_backtest_config(path))
    assert outcome.error is None
    parent = outcome.record
    runs = registry.read_runs()
    assert runs["run_id"].tolist() == [parent.run_id]  # stress runs hidden by default
    children = registry.read_stress_runs(parent.run_id)
    assert children["run_id"].tolist() == [f"{parent.run_id}-x0", f"{parent.run_id}-x2"]
    assert children["cost_multiplier"].tolist() == [0.0, 2.0]
    assert children["report_path"].isna().all()
    assert (children["parent_run_id"] == parent.run_id).all()
    assert children["config_hash"].iloc[1] != parent.config_hash
    every = registry.read_runs(include_stress=True)
    assert len(every) == 3
    assert len(list((project / "reports").glob("*.html"))) == 1

    row = runs.iloc[0]
    sensitivity = CostSensitivity.from_dict(json.loads(row["cost_sensitivity_json"]))
    assert [p.multiplier for p in sensitivity.points] == [0.0, 1.0, 2.0]
    assert bool(row["fragile"]) == sensitivity.fragile
    assert json.loads(row["assumptions_json"])["sections"]["Engine"]["Engine"].startswith("event")
    html = Path(row["report_path"]).read_text(encoding="utf-8")
    assert "<h2>Cost sensitivity</h2>" in html and "Break-even cost multiplier" in html
    assert "<h2>Assumptions</h2>" in html and "survivorship" in html
    assert json.loads(row["assumptions_json"])["sections"]["Data"]["Source"] == "yahoo"


def test_cli_run_and_show(project: Path) -> None:
    path = write(project, "mom", config_text("xs_momentum", "vectorized"))
    ran = runner.invoke(app, ["backtest", "run", path])
    assert ran.exit_code == 0, ran.output
    assert "Cost stress: break-even" in ran.output
    run_id = registry.read_runs().iloc[0]["run_id"]
    shown = runner.invoke(app, ["backtest", "show", run_id])
    assert shown.exit_code == 0, shown.output
    assert "Cost sensitivity" in shown.output and "Assumptions: Engine" in shown.output
    assert "survivorship" in shown.output

    skipped = runner.invoke(app, ["backtest", "run", path, "--no-cost-stress"])
    assert skipped.exit_code == 0, skipped.output
    assert "Cost stress" not in skipped.output
    newest = registry.read_runs().iloc[0]
    assert registry.read_stress_runs(newest["run_id"]).empty
    assert pd.isna(newest["cost_sensitivity_json"])
    shown = runner.invoke(app, ["backtest", "show", newest["run_id"]])
    assert "Cost sensitivity: not recorded" in shown.output


def test_an_empty_stress_list_turns_it_off(project: Path) -> None:
    path = write(project, "mom", config_text("buy_and_hold", "vectorized", cost_stress="[]"))
    from qr.backtest.config import load_backtest_config

    outcome = execute(load_backtest_config(path))
    assert outcome.result is not None and outcome.result.cost_sensitivity is None
    assert registry.read_stress_runs(outcome.record.run_id).empty
