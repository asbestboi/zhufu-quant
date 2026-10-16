"""Walk-forward evaluation: choose parameters on the past, test them on what came next.

The unsealed part of a split (train + validation, never a sealed segment) is cut into
folds. In each fold:

1. the candidate parameter sets (a grid, or just the config's own params) are backtested
   on the fold's **train window**, with data loaded only up to that window's end;
2. one is chosen by a stated objective (default: the highest Sharpe after costs at 1x;
   NaN ranks last, ties go to the first in grid order);
3. the chosen set is backtested once on the fold's **test window**, the out-of-sample
   part.

The test windows are stitched into one out-of-sample equity curve.

Windows:

* **anchored** (expanding): every train window starts at the beginning of the range and
  grows by ``step_months`` per fold;
* **rolling**: every train window is ``train_years`` long and moves with the test window.

The first test window starts ``train_years`` after the range start; each is
``test_months`` long, and the next starts ``step_months`` later. Only test windows that
end inside the range are used. Each train window ends ``embargo_days`` calendar days
before its test window starts (by default the split's embargo), so no training
observation sits right next to the test data. Within a fold the test window is
continuous, so the stitched curve has no gaps.

Every backtest a walk-forward makes is logged to the registry with its
``walkforward_id`` and ``fold`` (without cost stress or a tearsheet), so the count of runs
is never lost. They are reported separately from a study's trials.
"""

from __future__ import annotations

import html
import itertools
import json
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from qr import storage
from qr.backtest import provenance, registry
from qr.backtest.config import BacktestConfig
from qr.backtest.data import MarketData, load_market_data
from qr.backtest.metrics import equity_metrics
from qr.backtest.result import BacktestResult
from qr.config import ConfigError
from qr.load import get_bars
from qr.research.splits import IN_SAMPLE, WALK_FORWARD, Split, load_split

ANCHORED = "anchored"
ROLLING = "rolling"
MODES = (ANCHORED, ROLLING)

Candidate = tuple[dict[str, Any], Mapping[str, float]]
# Picks one candidate (parameters, train metrics) by index.
Objective = Callable[[Sequence[Candidate]], int]


def _risk_free_label(config: BacktestConfig) -> str:
    from qr.backtest.runner import risk_free_label  # runner imports qr.research

    return risk_free_label(config)


def best_sharpe(candidates: Sequence[Candidate]) -> int:
    """The default objective: highest Sharpe after costs; NaN last; ties to the first."""
    best, best_value = 0, -math.inf
    for index, (_, metrics) in enumerate(candidates):
        value = metrics.get("sharpe", math.nan)
        if math.isfinite(value) and value > best_value:
            best, best_value = index, value
    return best


@dataclass(frozen=True)
class Fold:
    """One train window and the test window right after it."""

    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def _add_months(day: date, months: int) -> date:
    return (pd.Timestamp(day) + pd.DateOffset(months=months)).date()


def make_folds(
    start: date,
    end: date,
    mode: str,
    train_years: int,
    test_months: int,
    step_months: int,
    embargo_days: int = 0,
) -> list[Fold]:
    """The folds of a walk-forward over ``[start, end]`` (see the module docstring)."""
    if mode not in MODES:
        raise ValueError(f"unknown walk-forward mode '{mode}' (use anchored or rolling)")
    if train_years < 1 or test_months < 1 or step_months < 1:
        raise ValueError("train_years, test_months, and step_months must be at least 1")
    if embargo_days < 0:
        raise ValueError("embargo_days must be zero or more")
    folds: list[Fold] = []
    test_start = _add_months(start, 12 * train_years)
    while True:
        test_end = _add_months(test_start, test_months) - timedelta(days=1)
        if test_end > end:
            break
        train_start = start if mode == ANCHORED else _add_months(test_start, -12 * train_years)
        train_end = test_start - timedelta(days=1 + embargo_days)
        if train_end <= train_start:
            raise ValueError("the embargo leaves no training data in a fold")
        folds.append(Fold(len(folds), train_start, train_end, test_start, test_end))
        test_start = _add_months(test_start, step_months)
    if not folds:
        raise ValueError(
            f"no complete fold fits between {start} and {end} with {train_years} train "
            f"years and {test_months} test months"
        )
    return folds


def expand_grid(grid: Mapping[str, Sequence[Any]] | None) -> list[dict[str, Any]]:
    """Every combination of a ``{param: [values]}`` grid, in grid order; ``[{}]`` if none."""
    if not grid:
        return [{}]
    names = list(grid)
    for name in names:
        if not isinstance(grid[name], list | tuple) or not grid[name]:
            raise ValueError(f"grid parameter '{name}' needs a non-empty list of values")
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*grid.values())]


@dataclass
class FoldResult:
    """What happened in one fold."""

    fold: Fold
    chosen: dict[str, Any]
    train_metrics: dict[str, float]
    test: BacktestResult
    candidates: int


@dataclass
class WalkForwardResult:
    """All folds, the stitched out-of-sample curve, and how stable the choice was."""

    walkforward_id: str
    config: BacktestConfig
    split: Split
    mode: str
    train_years: int
    test_months: int
    step_months: int
    embargo_days: int
    folds: list[FoldResult]
    equity: pd.Series
    benchmark: pd.Series | None
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]
    runs_logged: int = 0
    label: str = WALK_FORWARD
    report_path: Path | None = None
    grid: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def changes(self) -> int:
        """In how many folds the chosen parameters differ from the fold before."""
        chosen = [f.chosen for f in self.folds]
        return sum(1 for a, b in itertools.pairwise(chosen) if a != b)

    @property
    def unstable(self) -> bool:
        """The choice changed in more than half of the fold-to-fold steps."""
        steps = len(self.folds) - 1
        return steps > 0 and self.changes > steps / 2


def _stitch(curves: Sequence[pd.Series], initial_cash: float) -> pd.Series:
    """Chain each window's growth onto the end of the previous one."""
    pieces, level = [], initial_cash
    for curve in curves:
        scaled = curve / initial_cash * level
        pieces.append(scaled)
        level = float(scaled.iloc[-1])
    stitched: pd.Series = pd.concat(pieces)
    return stitched


def _window_config(
    config: BacktestConfig, params: Mapping[str, Any], start: date, end: date
) -> BacktestConfig:
    return config.model_copy(
        update={
            "params": {**config.params, **params},
            "start": start,
            "end": end,
            "split": None,
            "segment": None,
            "evaluate_from": None,
            "open_end": False,
            "cost_stress": [],
        }
    )


class _Logger:
    """Logs every walk-forward backtest to the registry, with provenance."""

    def __init__(self, walkforward_id: str, data_dir: Path | str | None, symbols: list[str]):
        self.walkforward_id = walkforward_id
        self.data_dir = data_dir
        self.symbols = symbols
        self.count = 0
        self.commit, self.dirty = registry.git_state()
        self._fingerprints: dict[date, Any] = {}

    def log(self, config: BacktestConfig, result: BacktestResult, fold: int, label: str) -> None:
        if config.end not in self._fingerprints:
            used = provenance.data_symbols(config, self.symbols)
            self._fingerprints[config.end] = provenance.run_fingerprint(
                used, config.end, data_dir=self.data_dir
            )
        data = self._fingerprints[config.end]
        environment = provenance.environment(config)
        created = storage.utcnow()
        metrics = result.metrics

        def number(key: str) -> float | None:
            value = metrics.get(key)
            return None if value is None or not math.isfinite(value) else float(value)

        registry.log_run(
            registry.RunRecord(
                run_id=registry.new_run_id(created),
                created_at=created,
                strategy=config.strategy,
                engine=config.engine,
                config_json=config.to_json(),
                config_hash=config.config_hash(),
                git_commit=self.commit,
                git_dirty=self.dirty,
                data_snapshot=None,
                start_date=config.start,
                end_date=config.end,
                sharpe=number("sharpe"),
                cagr=number("cagr"),
                max_drawdown=number("max_drawdown"),
                turnover=number("turnover"),
                report_path=None,
                notes=f"walk-forward {self.walkforward_id} fold {fold} ({label})",
                name=config.name,
                status=registry.OK,
                data_hash=data.hash,
                data_per_symbol_json=json.dumps(data.per_symbol, sort_keys=True),
                data_rows=data.rows,
                python_version=environment.python_version,
                uv_lock_sha256=environment.uv_lock_sha256,
                engine_core=environment.engine_core,
                rust_core_version=environment.rust_core_version,
                os=environment.os,
                cost_multiplier=config.cost_multiplier,
                study_id=config.study_id,
                result_label=label,
                risk_free=_risk_free_label(config),
                walkforward_id=self.walkforward_id,
                fold=fold,
            ),
            self.data_dir,
        )
        self.count += 1


def walk_forward(
    config: BacktestConfig,
    split: Split | str,
    mode: str = ANCHORED,
    train_years: int = 5,
    test_months: int = 12,
    step_months: int = 12,
    grid: Mapping[str, Sequence[Any]] | None = None,
    objective: Objective = best_sharpe,
    *,
    embargo_days: int | None = None,
    data_dir: Path | str | None = None,
    universe_config: Path | str | None = None,
    log: bool = True,
) -> WalkForwardResult:
    """Run a walk-forward of ``config`` over ``split``'s unsealed segments.

    ``grid`` maps parameter names to the values to try (merged into ``config.params``);
    without one, the config's params are evaluated in every fold. ``objective`` picks a
    candidate from ``(params, train metrics)`` pairs. ``embargo_days`` defaults to the
    split's. With ``log`` every backtest is written to the registry.
    """
    from qr.backtest.runner import (  # runner imports qr.research
        load_risk_free_returns,
        resolve_universe,
        run_backtest,
    )

    split = load_split(split) if isinstance(split, str) else split
    start, end = split.unsealed_range()
    embargo = split.embargo_days if embargo_days is None else embargo_days
    folds = make_folds(start, end, mode, train_years, test_months, step_months, embargo)
    for fold in folds:  # by construction, but a sealed segment must never be touched
        for name, segment in split.segments.items():
            if segment.sealed and fold.test_end >= segment.start:
                raise ConfigError(f"fold {fold.index} reaches sealed segment '{name}'")
    candidates = expand_grid(grid)
    symbols = resolve_universe(config, universe_config)
    walkforward_id = f"wf-{uuid.uuid4().hex[:10]}"
    logger = _Logger(walkforward_id, data_dir, symbols) if log else None

    def market(until: date) -> tuple[MarketData, pd.Series | None]:
        data = load_market_data(symbols, until, data_dir=data_dir)
        bench = None
        if config.benchmark:
            bench = get_bars(config.benchmark, end=until, data_dir=data_dir)["close"]
            bench = bench.droplevel("symbol")
        return data, bench

    results: list[FoldResult] = []
    for fold in folds:
        train_data, train_bench = market(fold.train_end)  # nothing after the train window
        tried: list[Candidate] = []
        for params in candidates:
            window = _window_config(config, params, fold.train_start, fold.train_end)
            outcome = run_backtest(
                window, train_data, train_bench, cost_stress=False, data_dir=data_dir
            )
            if logger:
                logger.log(window, outcome, fold.index, IN_SAMPLE)
            tried.append((params, outcome.metrics))
        pick = objective(tried)
        chosen, train_metrics = tried[pick]
        test_data, test_bench = market(fold.test_end)
        window = _window_config(config, chosen, fold.test_start, fold.test_end)
        test = run_backtest(window, test_data, test_bench, cost_stress=False, data_dir=data_dir)
        test.label = WALK_FORWARD
        if logger:
            logger.log(window, test, fold.index, WALK_FORWARD)
        results.append(FoldResult(fold, dict(chosen), dict(train_metrics), test, len(tried)))

    equity = _stitch([r.test.equity for r in results], config.initial_cash)
    benchmark = None
    if all(r.test.benchmark is not None for r in results):
        curves = [r.test.benchmark for r in results if r.test.benchmark is not None]
        benchmark = _stitch(curves, config.initial_cash)
    # The stitched curve is measured against the same risk-free rate as every fold.
    rate: float | pd.Series = config.risk_free_rate
    if config.risk_free is not None:
        rate = load_risk_free_returns(config, data_dir, end=folds[-1].test_end)
    return WalkForwardResult(
        walkforward_id=walkforward_id,
        config=config,
        split=split,
        mode=mode,
        train_years=train_years,
        test_months=test_months,
        step_months=step_months,
        embargo_days=embargo,
        folds=results,
        equity=equity,
        benchmark=benchmark,
        metrics=equity_metrics(equity, rate),
        benchmark_metrics=equity_metrics(benchmark, rate) if benchmark is not None else {},
        runs_logged=logger.count if logger else 0,
        grid={k: list(v) for k, v in (grid or {}).items()},
    )


# --------------------------------------------------------------------------- report


def render_walkforward(result: WalkForwardResult) -> str:
    """The walk-forward report as one HTML page."""
    from qr.backtest import report as tearsheet  # matplotlib, only when a report is made

    def params_text(params: Mapping[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in params.items()) or "(config params)"

    rows = "".join(
        f"<tr><td>{f.fold.index}</td><td>{f.fold.train_start} to {f.fold.train_end}</td>"
        f"<td>{f.fold.test_start} to {f.fold.test_end}</td>"
        f"<td>{html.escape(params_text(f.chosen))}</td>"
        f"<td>{tearsheet.format_metric(f.train_metrics.get('sharpe'), 'num')}</td>"
        f"<td>{tearsheet.format_metric(f.test.metrics.get('sharpe'), 'num')}</td>"
        f"<td>{tearsheet.format_metric(f.test.metrics.get('cagr'), 'pct')}</td>"
        f"<td>{tearsheet.format_metric(f.test.metrics.get('max_drawdown'), 'pct')}</td></tr>"
        for f in result.folds
    )
    metric_rows = "".join(
        f"<tr><td>{html.escape(label)}</td>"
        f"<td>{tearsheet.format_metric(result.metrics.get(key), kind)}</td>"
        f"<td>{tearsheet.format_metric(result.benchmark_metrics.get(key), kind)}</td></tr>"
        for key, label, kind in tearsheet.METRIC_ROWS
        if key in result.metrics
    )
    steps = max(len(result.folds) - 1, 0)
    stability = f"The chosen parameters changed in {result.changes} of {steps} steps between folds."
    if result.unstable:
        stability = f'<span class="flag">Unstable:</span> {stability} A choice that keeps '
        stability += "changing suggests the in-sample ranking is mostly noise."
    chart = _curve_chart(result)
    config = result.config
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(config.name)} walk-forward</title>
<style>{tearsheet._CSS}</style>
</head>
<body><main>
<h1>{html.escape(config.name)}: walk-forward</h1>
<p class="sub"><span class="label">{html.escape(result.label)}</span> ·
{html.escape(config.strategy)} · {config.engine} engine · split {html.escape(result.split.name)}
· {result.mode}, {result.train_years} train years, {result.test_months} test months,
step {result.step_months} months, embargo {result.embargo_days} days ·
{len(result.folds)} folds · {result.runs_logged} runs logged as
<code>{result.walkforward_id}</code></p>
<p class="note">Risk-free rate for every Sharpe (the selection included):
<strong>{html.escape(_risk_free_label(config))}</strong>.</p>
<p class="note">Each fold chose its parameters on its train window only, then ran them
once on the next window. Only the stitched test windows are out of sample. Sealed
segments were not used.</p>
<figure>{chart}</figure>
<h2>Folds</h2>
<div class="scroll"><table><tr><th>Fold</th><th>Train</th><th>Test</th><th>Chosen</th>
<th>Train Sharpe</th><th>Test Sharpe</th><th>Test CAGR</th><th>Test max drawdown</th></tr>
{rows}</table></div>
<p>{stability}</p>
<h2>Stitched out-of-sample curve</h2>
<div class="scroll"><table><tr><th>Metric</th><th>Walk-forward</th>
<th>Benchmark ({html.escape(str(config.benchmark))})</th></tr>{metric_rows}</table></div>
<h2>Grid</h2>
<pre>{html.escape(json.dumps(result.grid, indent=2))}</pre>
</main></body>
</html>
"""


def _curve_chart(result: WalkForwardResult) -> str:
    import matplotlib

    from qr.backtest import report as tearsheet

    lines = [("Walk-forward (out of sample)", result.equity, tearsheet.NET)]
    if result.benchmark is not None:
        lines.append((f"Benchmark {result.config.benchmark}", result.benchmark,
                      tearsheet.BENCHMARK))  # fmt: skip
    with matplotlib.rc_context(tearsheet._RC):  # type: ignore[arg-type]
        return tearsheet._line_chart("Stitched out-of-sample equity", lines, percent=False)


def write_walkforward_report(result: WalkForwardResult, reports_dir: Path | str) -> Path:
    """Write the report next to the tearsheets and return its path."""
    target = Path(reports_dir) / f"{result.walkforward_id}_{result.config.name}_walkforward.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_walkforward(result), encoding="utf-8")
    result.report_path = target
    return target
