"""Run a backtest from a config: load data, simulate, measure, report, and log.

:func:`run_backtest` is the pure part (config and data in, result out) and is what a
notebook should call. :func:`execute` adds the side effects the CLI needs: it writes the
tearsheet and logs the run to the registry, also when the run fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qr import storage
from qr.backtest import registry
from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData, load_market_data
from qr.backtest.event_engine import run_event
from qr.backtest.metrics import compute_metrics, equity_metrics
from qr.backtest.report import write_report
from qr.backtest.result import BacktestResult, EngineOutput
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import EventStrategy, WeightStrategy
from qr.backtest.vectorized import run_vectorized
from qr.config import universe_symbols
from qr.load import data_version, get_bars

DEFAULT_REPORTS_DIR = Path("reports")


@dataclass(frozen=True)
class RunOutcome:
    """What :func:`execute` did: the registry row, and the result unless the run failed."""

    record: registry.RunRecord
    result: BacktestResult | None
    error: Exception | None = None


def resolve_universe(
    config: BacktestConfig, universe_config: Path | str | None = None
) -> list[str]:
    """The config's symbols; a group name is looked up in ``config/universe.yaml``."""
    if isinstance(config.universe, str):
        return universe_symbols(config.universe, universe_config)
    return list(config.universe)


def _simulate(config: BacktestConfig, data: MarketData, free: bool) -> EngineOutput:
    costs = CostModel(config.costs, data.symbols, data.asset_classes, free=free)
    # A fresh strategy per simulation: the two runs must not share any state.
    strategy = build_strategy(config.strategy, config.engine, config.params)
    if config.engine == "vectorized":
        assert isinstance(strategy, WeightStrategy)
        return run_vectorized(strategy, data, config, costs)
    assert isinstance(strategy, EventStrategy)
    return run_event(strategy, data, config, costs)


def benchmark_curve(
    closes: pd.Series, sessions: pd.DatetimeIndex, initial_cash: float
) -> pd.Series | None:
    """``initial_cash`` held in the benchmark from the first session, on ``sessions``.

    ``closes`` are total-return adjusted. Days the benchmark has no bar carry its last
    value. None when the benchmark has no price on the first session.
    """
    aligned = closes.reindex(closes.index.union(sessions)).ffill().reindex(sessions)
    if len(aligned) == 0 or not aligned.iloc[0] > 0:
        return None
    return pd.Series(aligned / aligned.iloc[0] * initial_cash, name="benchmark")


def run_backtest(
    config: BacktestConfig,
    data: MarketData | None = None,
    benchmark_closes: pd.Series | None = None,
    *,
    data_dir: Path | str | None = None,
    universe_config: Path | str | None = None,
) -> BacktestResult:
    """Run ``config`` with and without costs and compute the metrics of both.

    Data comes from the local store unless ``data`` is given (and, for the benchmark,
    ``benchmark_closes``: a date-indexed series of total-return closes).
    """
    snapshot = None
    if data is None:
        symbols = resolve_universe(config, universe_config)
        data = load_market_data(symbols, config.end, data_dir=data_dir)
        used = [*symbols, *([config.benchmark] if config.benchmark else [])]
        snapshot = data_version(list(dict.fromkeys(used)), config.end, data_dir=data_dir)
        if config.benchmark and benchmark_closes is None:
            bars = get_bars(config.benchmark, end=config.end, data_dir=data_dir)
            benchmark_closes = bars["close"].droplevel("symbol")

    net = _simulate(config, data, free=False)
    gross = _simulate(config, data, free=True)
    benchmark = None
    if benchmark_closes is not None:
        benchmark = benchmark_curve(
            benchmark_closes, pd.DatetimeIndex(net.equity.index), config.initial_cash
        )
    rate = config.risk_free_rate
    return BacktestResult(
        config=config,
        net=net,
        gross=gross,
        benchmark=benchmark,
        metrics=compute_metrics(net, benchmark, rate),
        gross_metrics=compute_metrics(gross, benchmark, rate),
        benchmark_metrics=equity_metrics(benchmark, rate) if benchmark is not None else {},
        data_snapshot=snapshot,
    )


def _number(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def execute(
    config: BacktestConfig,
    *,
    data_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    universe_config: Path | str | None = None,
    notes: str | None = None,
) -> RunOutcome:
    """Run ``config``, write its tearsheet, and log it to the registry.

    A run that raises is logged too, with status ``failed`` and the error as its notes,
    and the exception is returned in the outcome rather than raised: a failed idea is
    still an idea that was tried.
    """
    created = storage.utcnow()
    run_id = registry.new_run_id(created)
    commit, dirty = registry.git_state()
    result: BacktestResult | None = None
    error: Exception | None = None
    report: Path | None = None
    try:
        result = run_backtest(config, data_dir=data_dir, universe_config=universe_config)
        report = Path(reports_dir or DEFAULT_REPORTS_DIR) / f"{run_id}_{config.name}.html"
        write_report(
            result,
            report,
            {
                "Run id": run_id,
                "Created (UTC)": f"{created:%Y-%m-%d %H:%M:%S}",
                "Config hash": config.config_hash(),
                "Git commit": f"{commit or 'unknown'}{' (uncommitted changes)' if dirty else ''}",
                "Data last written (UTC)": result.data_snapshot or "unknown",
                "Notes": notes or config.notes or "",
            },
        )
    except Exception as exc:
        error = exc

    metrics = result.metrics if result is not None else {}
    said = notes or config.notes
    record = registry.RunRecord(
        run_id=run_id,
        created_at=created,
        strategy=config.strategy,
        engine=config.engine,
        config_json=config.to_json(),
        config_hash=config.config_hash(),
        git_commit=commit,
        git_dirty=dirty,
        data_snapshot=(
            None
            if result is None or result.data_snapshot is None
            else result.data_snapshot.isoformat(sep=" ")
        ),
        start_date=config.start,
        end_date=config.end,
        sharpe=_number(metrics.get("sharpe")),
        cagr=_number(metrics.get("cagr")),
        max_drawdown=_number(metrics.get("max_drawdown")),
        turnover=_number(metrics.get("turnover")),
        report_path=None if report is None or error is not None else str(report),
        notes=said
        if error is None
        else f"{type(error).__name__}: {error}" + (f" | {said}" if said else ""),
        name=config.name,
        status=registry.OK if error is None else registry.FAILED,
    )
    registry.log_run(record, data_dir)
    return RunOutcome(record=record, result=result if error is None else None, error=error)
