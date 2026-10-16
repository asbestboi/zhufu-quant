"""The assumptions behind a run, written out from what actually ran.

Every report and registry row carries this. It is generated from the resolved config, the
resolved cost parameters, and the engine code path that ran (which engine and core,
whether limit orders were sent), never typed by hand, so it cannot drift from the run it
describes.

The result is a JSON-ready dictionary: ``sections`` (title -> ordered ``{item: text}``)
and ``warnings`` (plain sentences). Warnings are raised automatically for:

* a hand-picked universe (survivorship bias): always, because every universe here is a
  fixed list chosen today;
* limit orders filled from daily bars (the order of prices within a day is unknown);
* fewer than 30 trades;
* a period shorter than 3 years.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from qr.backtest.config import BacktestConfig
from qr.backtest.costs import BORROW_DAY_COUNT, NO_CAP, CostParams, resolve_params
from qr.backtest.result import BacktestResult

MIN_TRADES = 30
MIN_YEARS = 3.0

Assumptions = dict[str, Any]


def _bps(rate: float) -> str:
    return f"{rate * 1e4:g} bps"


def _commission(p: CostParams) -> str:
    parts = []
    if p.per_share:
        parts.append(f"{p.per_share:g} USD per share")
    if p.pct:
        parts.append(f"{p.pct * 100:g}% of value")
    text = " + ".join(parts) or "none"
    if p.max_pct < NO_CAP:
        text += f", capped at {p.max_pct * 100:g}% of value"
    if p.min_per_order:
        text += f", at least {p.min_per_order:g} USD per order"
    return text


def _slippage(p: CostParams) -> str:
    text = f"fixed {_bps(p.slippage)} of value"
    if p.impact:
        text += f" + {_bps(p.impact)} x (order / average daily dollar volume)^{p.impact_exponent:g}"
    return text


def _participation(p: CostParams, adv_window: int) -> str:
    if not math.isfinite(p.max_participation):
        return "none"
    action = "refused" if p.reject_over_cap else "cut down to the cap"
    return (
        f"{p.max_participation * 100:g}% of the {adv_window}-session average daily dollar "
        f"volume; larger orders are {action}"
    )


def _execution(config: BacktestConfig) -> str:
    fill = "open" if config.execution == "next_open" else "close"
    return f"signal at the close of t, filled at the {fill} of t+1"


def _fill_prices(config: BacktestConfig) -> str:
    field = "open" if config.execution == "next_open" else "close"
    if config.engine == "vectorized":
        return (
            f"total-return adjusted {field}; spread and slippage charged as costs on the "
            "value traded; per-share commission on the unadjusted price"
        )
    return (
        f"unadjusted {field} of t+1, made worse by spread and slippage; limit orders at "
        "the limit, or at the open when it is better"
    )


def _rebalance(config: BacktestConfig) -> str:
    if isinstance(config.rebalance, list):
        return f"{len(config.rebalance)} given dates (each moved to the next session if needed)"
    if config.rebalance == "daily":
        return "every session"
    period = "week" if config.rebalance == "weekly" else "month"
    return f"decided on the last session of each {period} (and the first session)"


def _corporate_actions(config: BacktestConfig) -> str:
    if config.engine == "vectorized":
        return (
            "total-return adjusted prices: dividends are reinvested in the same symbol on "
            "the ex-date, and splits do not show"
        )
    return (
        "on the ex-date, splits restate share counts and open orders, and cash dividends "
        "are credited on shares held (shorts pay them)"
    )


def _limits(config: BacktestConfig) -> dict[str, str]:
    c = config.constraints
    return {
        "Direction": "long only" if c.long_only else "long and short",
        "Gross leverage": f"at most {c.max_gross_leverage:g}x equity",
        "Weight per symbol": "no limit" if c.max_weight is None else f"at most {c.max_weight:g}",
        "Smallest trade": (
            "any size"
            if c.min_trade_weight == 0
            else f"rebalancing trades under {c.min_trade_weight:g} of equity are skipped"
        ),
        "Share rounding": (
            "whole shares"
            if config.engine == "event" and config.whole_shares
            else "fractional shares"
        ),
    }


def _sent_limit_orders(result: BacktestResult | None) -> bool:
    orders = None if result is None else result.net.orders
    return orders is not None and len(orders) > 0 and bool((orders["type"] == "limit").any())


def build_assumptions(
    config: BacktestConfig,
    symbols: Sequence[str],
    asset_classes: Mapping[str, str],
    *,
    result: BacktestResult | None = None,
    engine_core: str | None = None,
    sources: Mapping[str, str] | None = None,
) -> Assumptions:
    """The assumptions of a run of ``config`` over ``symbols``.

    ``result`` supplies what is only known after the run (trades, limit orders, the
    period actually covered); ``engine_core`` the core that ran; ``sources`` the data
    source of each symbol, when known.
    """
    costs = config.costs
    default_class = "equity"
    # The config's own numbers, before any asset-class or symbol override.
    base = resolve_params(costs, "", "\0no asset class")
    engine = config.engine if engine_core is None else f"{config.engine} ({engine_core} core)"
    sections: dict[str, dict[str, str]] = {
        "Engine": {
            "Engine": engine,
            "Execution timing": _execution(config),
            "Fill prices": _fill_prices(config),
            "Rebalance": _rebalance(config),
        },
        "Costs": {
            "Commission model": costs.commission_model,
            "Commission": _commission(base),
            "Spread": f"{_bps(base.spread)} of value per trade (half the bid-ask spread)",
            "Slippage": _slippage(base),
            "Participation cap": _participation(base, costs.adv_window),
            "Short borrow": (
                f"{base.borrow_rate * 100:g}% a year on short value, {BORROW_DAY_COUNT:g}-day year"
            ),
            "Cost multiplier": f"{config.cost_multiplier:g}x",
            "Cost stress": (", ".join(f"{m:g}x" for m in sorted(config.cost_stress)) or "not run"),
        },
        "Portfolio": {**_limits(config), "Corporate actions": _corporate_actions(config)},
        "Data": {
            "Adjustment": (
                "total return (dividends and splits) for prices the strategy sees and trades"
                if config.engine == "vectorized"
                else "unadjusted for trading; split and dividend adjusted in Context.history"
            ),
            "As of": "the store when the run started",
            "Source": ", ".join(sorted(set((sources or {}).values()))) or "not recorded",
            "Universe": f"{len(symbols)} symbols: {', '.join(symbols)}",
        },
    }
    # Symbols whose costs differ from the base, so a reader sees every number used.
    overrides = {
        symbol: resolve_params(costs, symbol, asset_classes.get(symbol, default_class))
        for symbol in symbols
    }
    for symbol, params in overrides.items():
        if params != base:
            sections["Costs"][f"Costs for {symbol}"] = (
                f"commission {_commission(params)}; spread {_bps(params.spread)}; "
                f"slippage {_slippage(params)}"
            )

    warnings = [
        f"Hand-picked universe ({len(symbols)} symbols chosen today): symbols that were "
        "delisted, merged, or fell out of favour are missing, which flatters results "
        "(survivorship bias)."
    ]
    if _sent_limit_orders(result):
        warnings.append(
            "Limit orders were filled from daily open, high, low, and close: the order of "
            "prices within the day is unknown, so some fills may be optimistic."
        )
    if result is not None:
        trades = result.metrics.get("trades", math.nan)
        if math.isfinite(trades) and trades < MIN_TRADES:
            warnings.append(
                f"Only {int(trades)} trades: too few to tell skill from luck "
                f"(fewer than {MIN_TRADES})."
            )
        sessions = result.equity.index
        years = (sessions[-1] - sessions[0]).days / 365.25 if len(sessions) else 0.0
    else:
        years = (config.end - config.start).days / 365.25
    if years < MIN_YEARS:
        warnings.append(
            f"The period covers {years:.1f} years, shorter than {MIN_YEARS:g}: one market "
            "regime can dominate the result."
        )
    return {"sections": sections, "warnings": warnings}
