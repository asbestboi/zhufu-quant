"""The HTML tearsheet: one self-contained file per run, with charts inlined as SVG.

Nothing is loaded from the network and nothing else needs to be kept next to the file.
Colors follow one rule throughout: blue is the strategy after costs, orange the same
strategy before costs, green the benchmark.
"""

from __future__ import annotations

import html
import io
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from qr.backtest import metrics as m
from qr.backtest.result import BacktestResult

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
NET = "#2a78d6"
GROSS = "#eb6834"
BENCHMARK = "#1baf7a"
NEGATIVE = "#e34948"
NEUTRAL = "#f0efec"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_RC = {
    "font.family": "sans-serif",
    "font.size": 9,
    "text.color": TEXT,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "svg.hashsalt": "qr",  # stable element ids, so the same run writes the same file
}

# name, label, kind: how each metric is shown in the table
METRIC_ROWS: list[tuple[str, str, str]] = [
    ("total_return", "Total return", "pct"),
    ("cagr", "CAGR", "pct"),
    ("volatility", "Volatility (annualized)", "pct"),
    ("sharpe", "Sharpe ratio", "num"),
    ("sortino", "Sortino ratio", "num"),
    ("calmar", "Calmar ratio", "num"),
    ("max_drawdown", "Maximum drawdown", "pct"),
    ("drawdown_duration_days", "Longest drawdown (days)", "int"),
    ("time_underwater", "Time underwater", "pct"),
    ("psr", "Probabilistic Sharpe ratio", "pct"),
    ("beta", "Beta to benchmark", "num"),
    ("correlation", "Correlation to benchmark", "num"),
    ("alpha", "Alpha (annualized)", "pct"),
    ("turnover", "Turnover (annualized)", "num"),
    ("average_holding_days", "Average holding period (sessions)", "int"),
    ("average_gross_exposure", "Average gross exposure", "pct"),
    ("average_net_exposure", "Average net exposure", "pct"),
    ("trades", "Trades", "int"),
    ("total_costs", "Total costs", "money"),
    ("closed_trades", "Closing trades", "int"),
    ("hit_rate", "Hit rate", "pct"),
    ("average_win", "Average win", "money"),
    ("average_loss", "Average loss", "money"),
    ("profit_factor", "Profit factor", "num"),
]


def format_metric(value: float | None, kind: str) -> str:
    """A metric as text; anything missing or not computable is a dash."""
    if value is None or not math.isfinite(value):
        return "-"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "int":
        return f"{value:,.0f}"
    if kind == "money":
        return f"{value:,.2f}"
    return f"{value:.2f}"


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Calendar-month returns as a (year x month) table; the first month starts at day one."""
    month_end = equity.groupby(pd.DatetimeIndex(equity.index).to_period("M")).last()
    returns = month_end.pct_change()
    returns.iloc[0] = month_end.iloc[0] / equity.iloc[0] - 1.0
    table = pd.DataFrame(
        {
            "year": pd.PeriodIndex(returns.index).year,
            "month": pd.PeriodIndex(returns.index).month,
            "r": returns.to_numpy(),
        }
    )
    return table.pivot(index="year", columns="month", values="r").reindex(columns=range(1, 13))


# --------------------------------------------------------------------------- charts


def _figure(height: float) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(9.6, height), facecolor=SURFACE)
    axes = figure.add_subplot()
    axes.set_facecolor(SURFACE)
    axes.grid(True, axis="y", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.tick_params(length=0)
    return figure, axes


def _svg(figure: Figure) -> str:
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", metadata={"Date": None})
    text = buffer.getvalue()
    text = text[text.index("<svg") :]
    # The metadata block only says which tool drew the chart.
    return re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.DOTALL)


def _end_labels(axes: Axes, series: list[tuple[str, pd.Series, str]]) -> None:
    """Name each line at its right end, nudged apart so the labels do not overlap."""
    low, high = axes.get_ylim()
    log = axes.get_yscale() == "log"

    def position(value: float) -> float:
        if log:
            return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
        return (value - low) / (high - low)

    placed: list[float] = []
    ends = [(name, s.dropna(), color) for name, s, color in series if len(s.dropna())]
    for name, line, color in sorted(ends, key=lambda item: item[1].iloc[-1]):
        y = position(float(line.iloc[-1]))
        if placed and y - placed[-1] < 0.07:
            y = placed[-1] + 0.07
        placed.append(y)
        axes.annotate(
            name,
            xy=(1.0, y),
            xycoords="axes fraction",
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
            fontsize=8.5,
        )
        axes.plot([1.0], [y], marker="o", markersize=5, color=color, clip_on=False,
                  transform=axes.transAxes)  # fmt: skip


def equity_chart(result: BacktestResult) -> str:
    """Equity after costs, before costs, and the benchmark, on a log scale."""
    figure, axes = _figure(3.6)
    lines = [("Before costs", result.gross_equity, GROSS), ("After costs", result.equity, NET)]
    if result.benchmark is not None:
        lines.insert(0, (f"Benchmark ({result.config.benchmark})", result.benchmark, BENCHMARK))
    for name, line, color in lines:
        axes.plot(line.index, line.to_numpy(), color=color, linewidth=1.6, label=name)
    if all((line.dropna() > 0).all() for _, line, _ in lines):
        axes.set_yscale("log")
    axes.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes.set_title("Account value (log scale)")
    axes.legend(loc="upper left", frameon=False, ncols=len(lines))
    _end_labels(axes, lines)
    return _svg(figure)


def drawdown_chart(result: BacktestResult) -> str:
    """Drawdown after costs, with the benchmark's for scale."""
    figure, axes = _figure(2.4)
    lines = [("After costs", m.drawdowns(result.equity), NET)]
    if result.benchmark is not None:
        lines.insert(0, (f"Benchmark ({result.config.benchmark})", m.drawdowns(result.benchmark),
                         BENCHMARK))  # fmt: skip
    own = lines[-1][1]
    axes.fill_between(own.index, own.to_numpy(), 0.0, color=NET, alpha=0.12, linewidth=0)
    for name, line, color in lines:
        axes.plot(line.index, line.to_numpy(), color=color, linewidth=1.4, label=name)
    axes.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes.set_title("Drawdown from the previous peak")
    if len(lines) > 1:
        axes.legend(loc="lower left", frameon=False, ncols=len(lines))
    return _svg(figure)


def heatmap_chart(equity: pd.Series) -> str:
    """Monthly returns after costs as a year-by-month grid, every cell labelled."""
    table = monthly_returns(equity)
    year_end = equity.groupby(pd.DatetimeIndex(equity.index).year).last()
    yearly = year_end.pct_change()
    yearly.iloc[0] = year_end.iloc[0] / equity.iloc[0] - 1.0
    values = table.to_numpy(dtype="float64")
    figure = Figure(figsize=(9.6, 0.36 * len(table) + 0.9), facecolor=SURFACE)
    axes = figure.add_subplot()
    finite = np.abs(values[np.isfinite(values)])
    bound = max(float(finite.max()) if len(finite) else 0.0, 1e-9)
    polarity = LinearSegmentedColormap.from_list("polarity", [NEGATIVE, NEUTRAL, NET])
    colors = polarity.with_extremes(bad=SURFACE)
    axes.imshow(
        np.ma.masked_invalid(values),
        cmap=colors,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
        aspect="auto",
    )
    for row in range(values.shape[0]):
        for column in range(12):
            value = values[row, column]
            if math.isfinite(value):
                strong = abs(value) > 0.6 * bound
                axes.text(column, row, f"{value:+.1%}", ha="center", va="center", fontsize=7.5,
                          color="#ffffff" if strong else TEXT)  # fmt: skip
        axes.text(12.05, row, f"{yearly.iloc[row]:+.1%}", ha="center", va="center", fontsize=7.5,
                  color=TEXT, fontweight="bold")  # fmt: skip
    axes.set_xlim(-0.5, 12.6)
    axes.set_xticks(range(13), [*MONTHS, "Year"])
    axes.set_yticks(range(len(table)), [str(y) for y in table.index])
    axes.set_xticks(np.arange(-0.5, 12), minor=True)
    axes.set_yticks(np.arange(-0.5, len(table)), minor=True)
    axes.grid(which="minor", color=SURFACE, linewidth=2)
    axes.tick_params(which="both", length=0)
    axes.xaxis.tick_top()
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_title("Monthly returns after costs", pad=22)
    return _svg(figure)


def _line_chart(title: str, lines: list[tuple[str, pd.Series, str]], percent: bool) -> str:
    figure, axes = _figure(2.4)
    for name, line, color in lines:
        axes.plot(line.index, line.to_numpy(), color=color, linewidth=1.4, label=name)
    if percent:
        axes.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes.set_title(title)
    if len(lines) > 1:
        axes.legend(loc="upper left", frameon=False, ncols=len(lines))
    return _svg(figure)


def rolling_charts(result: BacktestResult) -> list[str]:
    """Rolling one-year Sharpe ratio and volatility, one chart each (never a shared axis)."""
    rate = result.config.risk_free_rate / m.PERIODS_PER_YEAR
    pairs = [("After costs", m.daily_returns(result.equity), NET)]
    if result.benchmark is not None:
        name = f"Benchmark ({result.config.benchmark})"
        pairs.insert(0, (name, m.daily_returns(result.benchmark), BENCHMARK))
    if len(pairs[-1][1]) < m.ROLLING_WINDOW:
        return []
    return [
        _line_chart(
            "Rolling one-year Sharpe ratio",
            [(n, m.rolling_sharpe(r - rate), c) for n, r, c in pairs],
            percent=False,
        ),
        _line_chart(
            "Rolling one-year volatility (annualized)",
            [(n, m.rolling_volatility(r), c) for n, r, c in pairs],
            percent=True,
        ),
    ]


def exposure_chart(result: BacktestResult) -> str:
    """Gross and net exposure as a fraction of equity."""
    accounts = result.net.accounts
    return _line_chart(
        "Exposure (share of equity)",
        [
            ("Gross", accounts["gross_exposure"] / accounts["equity"], NET),
            ("Net", accounts["net_exposure"] / accounts["equity"], GROSS),
        ],
        percent=True,
    )


# --------------------------------------------------------------------------- page

_CSS = f"""
body {{ margin: 0; background: {SURFACE}; color: {TEXT};
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 980px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 15px; margin: 36px 0 8px; }}
h3 {{ font-size: 13px; margin: 20px 0 6px; }}
code {{ font: 12px ui-monospace, "Cascadia Mono", Consolas, monospace; }}
p.sub, p.note {{ color: {MUTED}; margin: 0 0 8px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin: 20px 0 8px; }}
.tile {{ border: 1px solid {GRID}; border-radius: 8px; padding: 10px 12px; }}
.tile .label {{ color: {MUTED}; font-size: 12px; }}
.tile .value {{ font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.tile .aside {{ color: {MUTED}; font-size: 12px; font-variant-numeric: tabular-nums; }}
figure {{ margin: 16px 0; overflow-x: auto; }}
figure svg {{ width: 100%; height: auto; min-width: 640px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 5px 10px; border-bottom: 1px solid {GRID}; text-align: right;
  white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: {MUTED}; font-weight: 600; font-size: 12px; }}
i.key {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; }}
pre {{ background: #f4f3f0; border-radius: 8px; padding: 12px; overflow-x: auto;
  font-size: 12px; }}
"""


def _tile(label: str, value: str, aside: str) -> str:
    return (
        f'<div class="tile"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{value}</div><div class="aside">{aside}</div></div>'
    )


def _metrics_table(result: BacktestResult) -> str:
    columns: list[tuple[str, str, Mapping[str, float]]] = [
        ("After costs", NET, result.metrics),
        ("Before costs", GROSS, result.gross_metrics),
    ]
    if result.benchmark is not None:
        columns.append((f"Benchmark ({result.config.benchmark})", BENCHMARK,
                        result.benchmark_metrics))  # fmt: skip
    head = "".join(
        f'<th><i class="key" style="background:{color}"></i>{html.escape(name)}</th>'
        for name, color, _ in columns
    )
    rows = []
    for key, label, kind in METRIC_ROWS:
        if not any(math.isfinite(values.get(key, math.nan)) for _, _, values in columns):
            continue
        cells = "".join(
            f"<td>{format_metric(values.get(key), kind)}</td>" for *_, values in columns
        )
        rows.append(f"<tr><td>{html.escape(label)}</td>{cells}</tr>")
    return f"<table><tr><th>Metric</th>{head}</tr>{''.join(rows)}</table>"


def _cost_table(result: BacktestResult) -> str:
    costs = result.net.costs.sum()
    final_gross, final_net = result.gross_equity.iloc[-1], result.equity.iloc[-1]
    rows = [
        ("Final value before costs", f"{final_gross:,.2f}"),
        ("Final value after costs", f"{final_net:,.2f}"),
        ("Difference", f"{final_net - final_gross:,.2f}"),
        ("Commission paid", f"{costs['commission']:,.2f}"),
        ("Spread paid", f"{costs['spread']:,.2f}"),
        ("Slippage paid", f"{costs['slippage']:,.2f}"),
        ("Short borrow fees paid", f"{costs['borrow']:,.2f}"),
    ]
    body = "".join(f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows)
    return f"<table><tr><th>Before and after costs</th><th>USD</th></tr>{body}</table>"


def _provenance_section(
    provenance: Mapping[str, Any] | None, per_symbol: Mapping[str, str] | None
) -> str:
    if not provenance and not per_symbol:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td><code>{html.escape(str(value))}</code></td></tr>"
        for key, value in (provenance or {}).items()
    )
    symbols = "".join(
        f"<tr><td>{html.escape(symbol)}</td><td><code>{html.escape(digest)}</code></td></tr>"
        for symbol, digest in sorted((per_symbol or {}).items())
    )
    hashes = (
        f"""<h3>Data hash per symbol</h3>
<div class="scroll"><table><tr><th>Symbol</th><th>SHA-256</th></tr>{symbols}</table></div>"""
        if symbols
        else ""
    )
    return f"""<h2>Provenance</h2>
<p class="note">Which data and which environment produced this result. The data hash covers
the unadjusted bars and corporate actions of every symbol read (the benchmark included),
full history up to the end date. <code>qr backtest verify RUN_ID</code> checks whether that
data is still the same.</p>
<div class="scroll"><table><tr><th>Field</th><th>Value</th></tr>{rows}</table></div>
{hashes}"""


def render_report(
    result: BacktestResult,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    per_symbol: Mapping[str, str] | None = None,
) -> str:
    """The complete tearsheet as one HTML string.

    ``metadata`` fills the Run table; ``provenance`` and ``per_symbol`` (data hash per
    symbol) fill the Provenance section, which is left out when both are empty.
    """
    config = result.config
    net, gross = result.metrics, result.gross_metrics
    with matplotlib.rc_context(_RC):  # type: ignore[arg-type]
        charts = [
            equity_chart(result),
            drawdown_chart(result),
            heatmap_chart(result.equity),
            *rolling_charts(result),
            exposure_chart(result),
        ]
    figures = "".join(f"<figure>{chart}</figure>" for chart in charts)

    def before(key: str, kind: str) -> str:
        return f"before costs {format_metric(gross.get(key), kind)}"

    tiles = "".join(
        [
            _tile("Total return", format_metric(net.get("total_return"), "pct"),
                  before("total_return", "pct")),
            _tile("CAGR", format_metric(net.get("cagr"), "pct"), before("cagr", "pct")),
            _tile("Sharpe ratio", format_metric(net.get("sharpe"), "num"),
                  before("sharpe", "num")),
            _tile("Maximum drawdown", format_metric(net.get("max_drawdown"), "pct"),
                  before("max_drawdown", "pct")),
            _tile("Total costs", format_metric(net.get("total_costs"), "money"),
                  f"{format_metric(net.get('trades'), 'int')} trades"),
        ]
    )  # fmt: skip
    sessions = result.equity.index
    meta_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in (metadata or {}).items()
    )
    config_text = html.escape(json.dumps(config.model_dump(mode="json"), indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(config.name)}</title>
<style>{_CSS}</style>
</head>
<body><main>
<h1>{html.escape(config.name)}</h1>
<p class="sub">{html.escape(config.strategy)} · {config.engine} engine ·
{sessions[0].date()} to {sessions[-1].date()} · {len(sessions):,} sessions ·
starting cash {config.initial_cash:,.0f}</p>
<div class="tiles">{tiles}</div>
<p class="note">Headline numbers are after costs. A backtest is a simulation of the past
under simplifying assumptions, not a forecast.</p>
{figures}
<h2>Before and after costs</h2>
<div class="scroll">{_cost_table(result)}</div>
<h2>Metrics</h2>
<div class="scroll">{_metrics_table(result)}</div>
<p class="note">The Probabilistic Sharpe Ratio is the chance that the true Sharpe ratio is
above zero given this one sample. It does not account for multiple testing: after trying
many ideas, the best one looks good by luck. The registry keeps every run so that step 3
can apply the Deflated Sharpe Ratio.</p>
<h2>Run</h2>
<div class="scroll"><table><tr><th>Field</th><th>Value</th></tr>{meta_rows}</table></div>
{_provenance_section(provenance, per_symbol)}
<h2>Config</h2>
<pre>{config_text}</pre>
</main></body>
</html>
"""


def write_report(
    result: BacktestResult,
    path: Path | str,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    per_symbol: Mapping[str, str] | None = None,
) -> Path:
    """Write the tearsheet to ``path`` (parent folders are created) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(result, metadata, provenance, per_symbol), encoding="utf-8")
    return target
