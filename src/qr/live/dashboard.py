"""A self-contained local HTML dashboard for one live strategy.

Regenerated after every `qr live run` / `qr live approve` (see `qr.cli`) -- a plain
static file refreshed on disk, not a server. The page itself reloads every
`AUTO_REFRESH_SECONDS` while left open, so a browser tab left on it picks up the next
regeneration (at most once a day, since the strategy only ever rebalances once a month
and the underlying bars are daily) without a manual refresh. No network calls and no
external assets of any kind: every chart is a hand-drawn inline SVG, so the file always
opens correctly straight from disk, offline, forever.

If the config sets ``backtest_reference``, a second chart shows that already-disclosed
historical window (e.g. the candidate's own sealed test) next to its benchmark -- real
curves to look at before the live one has any history, never re-run or extended, and
never a new look at data the study hasn't already seen.
"""

from __future__ import annotations

import html
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from qr import calendars
from qr.config import data_dir as resolve_data_dir
from qr.live import journal, shadow
from qr.live.config import LiveConfig
from qr.live.cycle import ASSET_CLASS, is_month_end
from qr.live.decide import DecisionError, target_weights

_COLOURS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
AUTO_REFRESH_SECONDS = 300


def dashboard_path(config: LiveConfig, data_dir: Path | str | None = None) -> Path:
    """Where this strategy's dashboard file lives, next to the journal."""
    return resolve_data_dir(data_dir) / f"dashboard_{config.name}.html"


def _svg_chart(series: dict[str, pd.Series], width: int = 760, height: int = 320) -> str:
    """Several named series on a shared time axis, as one inline SVG line chart."""
    clean = {name: s.dropna() for name, s in series.items() if not s.dropna().empty}
    if not clean:
        return "<p>No equity history yet.</p>"
    all_dates = sorted({pd.Timestamp(d).date() for s in clean.values() for d in s.index})
    x0, x1 = all_dates[0], all_dates[-1]
    span_days = max((x1 - x0).days, 1)
    all_values = [v for s in clean.values() for v in s.to_numpy()]
    y_min, y_max = min(all_values), max(all_values)
    y_range = (y_max - y_min) or 1.0
    pad = 44

    def x_of(d: date) -> float:
        return pad + (d - x0).days / span_days * (width - 2 * pad)

    def y_of(v: float) -> float:
        return height - pad - (v - y_min) / y_range * (height - 2 * pad)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{width}px;background:#fff" role="img" '
        f'aria-label="Equity curve">',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        'stroke="#999"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#999"/>',
    ]
    for i, (name, s) in enumerate(clean.items()):
        colour = _COLOURS[i % len(_COLOURS)]
        points = " ".join(
            f"{x_of(pd.Timestamp(d).date()):.1f},{y_of(float(v)):.1f}" for d, v in s.items()
        )
        parts.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{points}"/>'
        )
        parts.append(
            f'<text x="{pad + 8}" y="{pad + 2 + 14 * i}" fill="{colour}" font-size="12" '
            f'font-family="monospace">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _next_month_end(after: date) -> date | None:
    """The next month-end session strictly after ``after``, if the calendar has one soon."""
    window = calendars.sessions(ASSET_CLASS, after, after + timedelta(days=65))
    for raw in window:
        day = pd.Timestamp(raw).date()
        if day > after and is_month_end(day):
            return day
    return None


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<p>None yet.</p>"
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def build_dashboard_html(config: LiveConfig, data_dir: Path | str | None = None) -> str:
    """Everything the spec asked for in one page: equity curve, positions, cash,
    orders, fills, and a month-by-month table -- all read fresh from the journal."""
    from qr.live import report  # local import: avoids a report <-> dashboard cycle

    decisions = journal.read_decisions(config.strategy, data_directory=data_dir)
    as_of = pd.Timestamp.now()
    cash, quantities = shadow.ledger_state(config.strategy, config.capital_limit, data_dir=data_dir)

    from qr.load import get_bars
    from qr.storage import StorageError

    try:
        universe_bars = get_bars(
            config.universe, end=date.today(), adjustment="none", data_dir=data_dir
        )
    except StorageError:
        universe_bars = pd.DataFrame()
    prices: dict[str, float] = {}
    latest_session: date | None = None
    if not universe_bars.empty:
        closes = universe_bars["close"].groupby(level="symbol").last()
        prices = {s: float(closes[s]) for s in quantities if s in closes.index}
        latest_session = pd.Timestamp(universe_bars.index.get_level_values("date").max()).date()
    equity = cash + sum(quantities[s] * prices.get(s, 0.0) for s in quantities)

    preview_weights: dict[str, float] = {}
    preview_error: str | None = None
    next_decision: date | None = None
    if latest_session is not None:
        try:
            preview_weights = target_weights(
                config.strategy, config.params, config.universe, latest_session,
                data_dir=data_dir,
            )  # fmt: skip
        except DecisionError as exc:
            preview_error = str(exc)
        next_decision = (
            latest_session if is_month_end(latest_session) else _next_month_end(latest_session)
        )

    ledger_curve = shadow.ledger_equity_curve(
        config.strategy, config.capital_limit, data_dir=data_dir
    )
    series = {f"{config.strategy} (ledger)": ledger_curve}
    if not decisions.empty:
        first_date = pd.Timestamp(decisions["decision_date"].min()).date()
        last_date = pd.Timestamp(decisions["decision_date"].max()).date()
        candidate_shadow = shadow.run_shadow(
            config.strategy, config.params, config.universe, first_date, last_date,
            initial_cash=config.capital_limit, data_dir=data_dir,
        )  # fmt: skip
        series[f"{config.strategy} (shadow)"] = candidate_shadow.equity
        if config.benchmark is not None:
            benchmark_shadow = shadow.run_shadow(
                config.benchmark.strategy, config.benchmark.params, config.universe,
                first_date, last_date, initial_cash=config.capital_limit, data_dir=data_dir,
            )  # fmt: skip
            series[f"{config.benchmark.strategy} (shadow)"] = benchmark_shadow.equity

    reference_chart: str | None = None
    if config.backtest_reference is not None:
        ref = config.backtest_reference
        reference_series = {}
        candidate_ref = shadow.run_shadow(
            config.strategy, config.params, config.universe, ref.start, ref.end,
            initial_cash=config.capital_limit, data_dir=data_dir,
        )  # fmt: skip
        reference_series[f"{config.strategy} (backtest)"] = candidate_ref.equity
        if config.benchmark is not None:
            benchmark_ref = shadow.run_shadow(
                config.benchmark.strategy, config.benchmark.params, config.universe,
                ref.start, ref.end, initial_cash=config.capital_limit, data_dir=data_dir,
            )  # fmt: skip
            reference_series[f"{config.benchmark.strategy} (backtest)"] = benchmark_ref.equity
        reference_chart = (
            f'<p class="meta">{html.escape(ref.label)}, {ref.start.isoformat()} to '
            f"{ref.end.isoformat()} -- already-disclosed historical backtest, not live "
            "paper money. Shown for context: this is where the candidate and benchmark "
            "diverged in the past, not a forecast of what happens next.</p>"
            + _svg_chart(reference_series)
        )
    reference_section = (
        f"<h2>{html.escape(config.backtest_reference.label)}</h2>{reference_chart}"
        if reference_chart and config.backtest_reference is not None
        else ""
    )

    position_rows = []
    for symbol, qty in sorted(quantities.items()):
        value = qty * prices.get(symbol, 0.0)
        weight = f"{value / config.capital_limit:.1%}" if config.capital_limit else "-"
        position_rows.append([symbol, f"{qty:g}", f"{value:,.2f}", weight])

    orders = journal.read_orders_for_strategy(config.strategy, data_directory=data_dir)
    order_rows = []
    if not orders.empty:
        for _, row in orders.sort_values("submitted_at").iterrows():
            order_rows.append(
                [
                    str(row["decision_id"]),
                    str(row["symbol"]),
                    str(row["side"]),
                    f"{row['quantity']:g}",
                    f"{row['planned_price']:.2f}",
                    str(row["status"]),
                ]
            )

    fills = journal.read_fills_for_strategy(config.strategy, data_directory=data_dir)
    fill_rows = []
    if not fills.empty:
        for _, row in fills.sort_values("filled_at").iterrows():
            fill_rows.append(
                [
                    str(row["decision_id"]),
                    str(row["symbol"]),
                    str(row["side"]),
                    f"{row['quantity']:g}",
                    f"{row['price']:.2f}",
                    f"{row['commission']:.2f}",
                    str(row["filled_at"]),
                ]
            )

    month_rows = []
    if not decisions.empty:
        months = sorted({f"{pd.Timestamp(d):%Y-%m}" for d in decisions["decision_date"]})
        for month in months:
            tracking = report.build_tracking_report(config, month, data_dir=data_dir)
            candidate = (
                f"{tracking.candidate_return:.2%}" if tracking.candidate_return is not None else "-"
            )
            benchmark = (
                f"{tracking.benchmark_return:.2%}" if tracking.benchmark_return is not None else "-"
            )
            consistency = (
                "match" if tracking.consistency and tracking.consistency.ok
                else "MISMATCH" if tracking.consistency else "-"
            )  # fmt: skip
            month_rows.append(
                [month, candidate, benchmark, consistency,
                 f"{tracking.reliability.missed}/{tracking.reliability.due}"]
            )  # fmt: skip

    preview_rows = [[symbol, f"{weight:.1%}"] for symbol, weight in sorted(preview_weights.items())]
    if preview_error is not None:
        preview_html = f"<p>Can't compute a preview yet: {html.escape(preview_error)}</p>"
    elif latest_session is None:
        preview_html = "<p>No market data yet.</p>"
    else:
        is_real = next_decision == latest_session
        label = "this month's real decision" if is_real else "not a real decision"
        note = (
            ""
            if is_real
            else f" Next scheduled decision: "
            f"{next_decision.isoformat() if next_decision else 'unknown'}."
        )
        preview_html = (
            f'<p class="meta">As of the {latest_session.isoformat()} close -- '
            f"{html.escape(label)}.{html.escape(note)}</p>"
            + _table(["symbol", "target weight"], preview_rows)
        )

    chart = _svg_chart(series)
    strategy_html = html.escape(config.strategy)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{AUTO_REFRESH_SECONDS}">
<title>{html.escape(config.name)} -- paper trading dashboard</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #111; }}
h1 {{ font-size: 1.3rem; }}
h2 {{ font-size: 1.05rem; margin-top: 2rem; }}
table.data {{ border-collapse: collapse; margin-top: 0.5rem; }}
table.data th, table.data td {{ border: 1px solid #ddd; padding: 4px 10px; font-size: 0.9rem; }}
table.data th {{ background: #f3f4f6; text-align: left; }}
.meta {{ color: #555; font-size: 0.9rem; }}
</style></head>
<body>
<h1>{html.escape(config.name)} ({strategy_html})</h1>
<p class="meta">Generated {as_of:%Y-%m-%d %H:%M}, refreshes every {AUTO_REFRESH_SECONDS // 60}
minutes while open. Broker: {html.escape(config.broker)}.
Cash: ${cash:,.2f}. Equity: ${equity:,.2f}.</p>

<h2>Live preview</h2>
{preview_html}
{reference_section}
<h2>Equity curve</h2>
{chart}

<h2>Current positions</h2>
{_table(["symbol", "quantity", "market value", "weight"], position_rows)}

<h2>Month by month</h2>
{_table(["month", "candidate return", "benchmark return", "consistency", "missed/due"], month_rows)}

<h2>Orders</h2>
{_table(["decision", "symbol", "side", "quantity", "planned price", "status"], order_rows)}

<h2>Fills</h2>
{_table(["decision", "symbol", "side", "quantity", "price", "commission", "filled at"], fill_rows)}
</body></html>"""


def write_dashboard(config: LiveConfig, data_dir: Path | str | None = None) -> Path:
    """Build and write this strategy's dashboard, overwriting any previous one."""
    path = dashboard_path(config, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_dashboard_html(config, data_dir), encoding="utf-8")
    return path
