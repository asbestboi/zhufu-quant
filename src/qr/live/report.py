"""The monthly paper-trading report: what a strategy's live cycle decided, sent, and
ended up holding in one calendar month, read straight from the journal.

:func:`build_report`/:func:`render_text` are the plain journal summary (what was
decided, sent, and filled). :func:`build_tracking_report` adds the rest of the paper-
trading forward test's monthly tracking report: the missed-decision reliability tally,
the ledger-vs-shadow-backtest consistency check, and (with a
:class:`~qr.live.config.BenchmarkConfig`) the candidate's shadow performance against a
benchmark -- never traded, informative only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from qr.live import journal, schedule, shadow
from qr.live.config import LiveConfig


@dataclass(frozen=True)
class MonthlyReport:
    """Everything journaled for ``strategy`` in one calendar month."""

    strategy: str
    month: str  # "YYYY-MM"
    decisions: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    alerts: pd.DataFrame


def build_report(strategy: str, month: str, data_dir: Path | str | None = None) -> MonthlyReport:
    """Everything journaled for ``strategy`` in calendar month ``month`` ("YYYY-MM")."""
    decisions = journal.read_decisions(strategy, data_directory=data_dir)
    if not decisions.empty:
        decisions = decisions[decisions["decision_date"].astype(str).str.startswith(month)]
    decision_ids = list(decisions["decision_id"])

    empty_orders = journal.read_orders("", data_directory=data_dir).iloc[0:0]
    empty_fills = journal.read_fills("", data_directory=data_dir).iloc[0:0]
    orders = (
        pd.concat(
            [journal.read_orders(did, data_directory=data_dir) for did in decision_ids],
            ignore_index=True,
        )
        if decision_ids
        else empty_orders
    )
    fills = (
        pd.concat(
            [journal.read_fills(did, data_directory=data_dir) for did in decision_ids],
            ignore_index=True,
        )
        if decision_ids
        else empty_fills
    )

    alerts = journal.read_alerts(data_directory=data_dir)
    if not alerts.empty:
        alerts = alerts[alerts["decision_id"].isin(decision_ids)]

    return MonthlyReport(
        strategy=strategy, month=month, decisions=decisions, orders=orders, fills=fills,
        alerts=alerts,
    )  # fmt: skip


def render_text(report: MonthlyReport) -> str:
    """A plain-text summary of a :class:`MonthlyReport`, for the terminal or a log file."""
    lines = [f"# {report.strategy} -- {report.month}", ""]
    if report.decisions.empty:
        lines.append("No decisions recorded this month.")
        return "\n".join(lines)

    for _, decision in report.decisions.iterrows():
        lines.append(f"## {decision['decision_id']}  [{decision['status']}]")
        # "equity" in the journal is the managed capital (LiveConfig.capital_limit) this
        # decision was sized against, not the paper account's total equity.
        capital = decision["equity"]
        if pd.notna(capital):
            lines.append(f"  managed capital: ${capital:,.2f}")
        weights = json.loads(decision["target_weights_json"])
        for symbol, weight in sorted(weights.items()):
            lines.append(f"  target {symbol}: {weight:.2%}")
        did = decision["decision_id"]
        did_orders = report.orders[report.orders["decision_id"] == did]
        for _, order in did_orders.iterrows():
            lines.append(
                f"  order {order['symbol']} {order['side']} {order['quantity']:g} "
                f"@ ~{order['planned_price']:.2f} [{order['status']}]"
            )
        did_fills = report.fills[report.fills["decision_id"] == did]
        for _, fill in did_fills.iterrows():
            lines.append(
                f"  fill {fill['symbol']} {fill['side']} {fill['quantity']:g} @ {fill['price']:.2f}"
            )
        lines.append("")

    if not report.alerts.empty:
        lines.append("## Alerts")
        for _, alert in report.alerts.iterrows():
            lines.append(f"  [{alert['severity']}] {alert['decision_id']}: {alert['message']}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- section 8


@dataclass(frozen=True)
class ReliabilityTally:
    """How many of this strategy's decisions (so far, cumulatively) were missed.

    ``due`` counts every decision the journal knows of, whether it has a row (planned,
    submitted, completed, expired) or was never even created (a missed month-end nobody
    ran -- :mod:`qr.live.schedule` alerts those too, by decision id, with no row to show).
    """

    due: int
    missed: int
    missed_decision_ids: list[str]


def reliability_tally(strategy: str, data_dir: Path | str | None = None) -> ReliabilityTally:
    decisions = journal.read_decisions(strategy, data_directory=data_dir)
    decision_ids = set(decisions["decision_id"]) if not decisions.empty else set()
    alerts = journal.read_alerts(data_directory=data_dir)
    missed_ids: set[str] = set()
    if not alerts.empty:
        tagged = alerts[alerts["message"].str.startswith(schedule.MISSED_TAG)]
        for message in tagged["message"]:
            # "[missed] {decision_id}: ..." -- did itself may contain ':', message's own
            # separator does not, so split once from the right of the known prefix.
            rest = message[len(schedule.MISSED_TAG) + 1 :]
            missed_ids.add(rest.rsplit(": ", 1)[0])
    return ReliabilityTally(
        due=len(decision_ids | missed_ids),
        missed=len(missed_ids),
        missed_decision_ids=sorted(missed_ids),
    )


def _ledger_positions(
    strategy: str, capital_limit: float, data_dir: Path | str | None = None
) -> dict[str, float]:
    """Every position implied by this strategy's fills so far (see `qr.live.shadow`)."""
    _, quantities = shadow.ledger_state(strategy, capital_limit, data_dir=data_dir)
    return quantities


@dataclass(frozen=True)
class ConsistencyResult:
    """Whether the ledger's own positions match a shadow backtest, in whole shares."""

    ok: bool
    mismatches: dict[str, tuple[float, float]]  # symbol -> (ledger qty, shadow qty)


def consistency_check(
    config: LiveConfig,
    start: date,
    end: date,
    data_dir: Path | str | None = None,
    tolerance: float = 1e-6,
) -> ConsistencyResult:
    """Compare the ledger's positions through ``end`` against a shadow backtest's."""
    ledger = _ledger_positions(config.strategy, config.capital_limit, data_dir)
    result = shadow.run_shadow(
        config.strategy, config.params, config.universe, start, end,
        initial_cash=config.capital_limit, data_dir=data_dir,
    )  # fmt: skip
    shadow_positions = (
        {symbol: p.quantity for symbol, p in result.final_state.positions.items()}
        if result.final_state is not None
        else {}
    )
    mismatches = {}
    for symbol in set(ledger) | set(shadow_positions):
        ledger_qty, shadow_qty = ledger.get(symbol, 0.0), shadow_positions.get(symbol, 0.0)
        if abs(ledger_qty - shadow_qty) > tolerance:
            mismatches[symbol] = (ledger_qty, shadow_qty)
    return ConsistencyResult(ok=not mismatches, mismatches=mismatches)


@dataclass(frozen=True)
class TrackingReport:
    """Section 8's monthly tracking report: the journal summary plus the forward
    test's own reliability tally, consistency check, and (optional) benchmark shadow."""

    base: MonthlyReport
    reliability: ReliabilityTally
    consistency: ConsistencyResult | None
    candidate_return: float | None  # this month's ledger return
    benchmark_return: float | None  # this month's shadow benchmark return, if configured
    cumulative_candidate_return: float | None
    cumulative_benchmark_return: float | None


def build_tracking_report(
    config: LiveConfig, month: str, data_dir: Path | str | None = None
) -> TrackingReport:
    """Everything :func:`build_report` has, plus the forward test's own figures.

    ``month``'s decision (if any) anchors the consistency check and the shadow
    comparison; both run from this strategy's very first journaled decision through
    that date, so returns are measured over the same span the ledger has actually lived.
    """
    base = build_report(config.strategy, month, data_dir=data_dir)
    reliability = reliability_tally(config.strategy, data_dir=data_dir)

    all_decisions = journal.read_decisions(config.strategy, data_directory=data_dir)
    if all_decisions.empty or base.decisions.empty:
        return TrackingReport(base, reliability, None, None, None, None, None)

    first_date = pd.Timestamp(all_decisions["decision_date"].min()).date()
    this_date = pd.Timestamp(base.decisions["decision_date"].max()).date()

    consistency = consistency_check(config, first_date, this_date, data_dir=data_dir)
    ledger_curve = shadow.ledger_equity_curve(
        config.strategy, config.capital_limit, data_dir=data_dir
    )
    candidate_return = _recent_return(ledger_curve, this_date)
    cumulative_candidate = shadow.period_return(
        ledger_curve, pd.Timestamp(first_date), pd.Timestamp(this_date)
    )

    benchmark_return = cumulative_benchmark = None
    if config.benchmark is not None:
        benchmark_result = shadow.run_shadow(
            config.benchmark.strategy, config.benchmark.params, config.universe,
            first_date, this_date, initial_cash=config.capital_limit, data_dir=data_dir,
        )  # fmt: skip
        benchmark_return = _recent_return(benchmark_result.equity, this_date)
        cumulative_benchmark = shadow.period_return(
            benchmark_result.equity, pd.Timestamp(first_date), pd.Timestamp(this_date)
        )

    return TrackingReport(
        base, reliability, consistency, candidate_return, benchmark_return,
        cumulative_candidate, cumulative_benchmark,
    )  # fmt: skip


def _recent_return(equity: pd.Series, this_date: date) -> float | None:
    """The return from the observation just before ``this_date`` to ``this_date`` itself."""
    through = equity.loc[: pd.Timestamp(this_date)]
    if len(through) < 2:
        return None
    return float(through.iloc[-1] / through.iloc[-2] - 1.0)


def render_tracking_markdown(report: TrackingReport) -> str:
    """Section 8's monthly tracking report, as Markdown."""
    lines = [render_text(report.base), "", "## Forward-test tracking", ""]
    rel = report.reliability
    lines.append(
        f"- Reliability: {rel.missed} missed of {rel.due} decisions due so far "
        f"(primary criterion: at most 1 of 12) -- {', '.join(rel.missed_decision_ids) or 'none'}"
    )
    if report.consistency is not None:
        verdict = "match" if report.consistency.ok else "MISMATCH (bug to investigate)"
        lines.append(f"- Consistency vs shadow backtest: {verdict}")
        for symbol, (ledger_qty, shadow_qty) in sorted(report.consistency.mismatches.items()):
            lines.append(f"    {symbol}: ledger {ledger_qty:g} vs shadow {shadow_qty:g}")
    if report.candidate_return is not None:
        lines.append(f"- This month's return: {report.candidate_return:.2%}")
    if report.cumulative_candidate_return is not None:
        lines.append(f"- Cumulative return: {report.cumulative_candidate_return:.2%}")
    if report.benchmark_return is not None:
        lines.append(f"- Benchmark this month (informative only): {report.benchmark_return:.2%}")
    if report.cumulative_benchmark_return is not None:
        lines.append(
            f"- Benchmark cumulative (informative only): {report.cumulative_benchmark_return:.2%}"
        )
    return "\n".join(lines)


def render_tracking_html(report: TrackingReport) -> str:
    """Section 8's monthly tracking report, as a small standalone HTML fragment."""
    import html

    body = html.escape(render_tracking_markdown(report))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(report.base.strategy)} -- {html.escape(report.base.month)}</title>"
        "</head><body><pre style='font-family:monospace;white-space:pre-wrap'>"
        f"{body}</pre></body></html>"
    )
