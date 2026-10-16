"""Run a backtest from a config: load data, simulate, measure, report, and log.

:func:`run_backtest` is the pure part (config and data in, result out) and is what a
notebook should call. :func:`execute` adds the side effects the CLI needs: it writes the
tearsheet and logs the run to the registry, also when the run fails.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from qr import storage
from qr.backtest import provenance, registry
from qr.backtest.assumptions import build_assumptions
from qr.backtest.config import BacktestConfig
from qr.backtest.costs import CostModel
from qr.backtest.data import MarketData, load_market_data
from qr.backtest.event_engine import run_event
from qr.backtest.metrics import compute_metrics, equity_metrics
from qr.backtest.report import write_report
from qr.backtest.result import BacktestResult, EngineOutput
from qr.backtest.strategies import build_strategy
from qr.backtest.strategy import EventStrategy, WeightStrategy
from qr.backtest.stress import CostSensitivity, StressPoint
from qr.backtest.vectorized import run_vectorized
from qr.config import db_path as storage_db_path
from qr.config import universe_symbols
from qr.fingerprint import DataFingerprint
from qr.load import data_version, get_bars
from qr.research import holdout
from qr.research.splits import IN_SAMPLE, load_split

DEFAULT_REPORTS_DIR = Path("reports")


@dataclass(frozen=True)
class RunOutcome:
    """What :func:`execute` did: the registry row, and the result unless the run failed."""

    record: registry.RunRecord
    result: BacktestResult | None
    error: Exception | None = None
    # Set when the config already ran on its sealed segment: the stored result, shown
    # instead of running again (``result`` is then None and ``record`` the stored run).
    sealed: holdout.SealedResult | None = None


def result_label(config: BacktestConfig) -> str:
    """What a run of ``config`` is: in-sample, validation, or sealed test."""
    if config.split is None or config.segment is None:
        return IN_SAMPLE
    return load_split(config.split).label(config.segment)


def resolve_universe(
    config: BacktestConfig, universe_config: Path | str | None = None
) -> list[str]:
    """The config's symbols; a group name is looked up in ``config/universe.yaml``."""
    if isinstance(config.universe, str):
        return universe_symbols(config.universe, universe_config)
    return list(config.universe)


def _simulate(config: BacktestConfig, data: MarketData, free: bool) -> EngineOutput:
    costs = CostModel(
        config.costs, data.symbols, data.asset_classes, free=free,
        multiplier=config.cost_multiplier,
    )  # fmt: skip
    # A fresh strategy per simulation: the two runs must not share any state.
    strategy = build_strategy(config.strategy, config.engine, config.params)
    if config.engine == "vectorized":
        assert isinstance(strategy, WeightStrategy)
        return run_vectorized(strategy, data, config, costs)
    assert isinstance(strategy, EventStrategy)
    return run_event(strategy, data, config, costs)


def _evaluated(config: BacktestConfig, output: EngineOutput) -> EngineOutput:
    """``output`` from ``config.evaluate_from`` on (a split's embargo), else unchanged."""
    if config.evaluate_from is None:
        return output
    cut = output.since(config.evaluate_from)
    if cut.accounts.empty:
        raise ValueError(f"no sessions from evaluate_from {config.evaluate_from} to {config.end}")
    return cut


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
    cost_stress: bool = True,
    engine_core: str | None = None,
) -> BacktestResult:
    """Run ``config`` with and without costs and compute the metrics of both.

    Data comes from the local store unless ``data`` is given (and, for the benchmark,
    ``benchmark_closes``: a date-indexed series of total-return closes).

    Unless ``cost_stress`` is False, the run is also simulated at each multiplier of
    ``config.cost_stress`` other than 1 (see :mod:`qr.backtest.stress`). The result
    carries the run's assumptions either way; ``engine_core`` is the core that ran, for
    them.
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

    net = _evaluated(config, _simulate(config, data, free=False))
    gross = _evaluated(config, _simulate(config, data, free=True))
    benchmark = None
    if benchmark_closes is not None:
        benchmark = benchmark_curve(
            benchmark_closes, pd.DatetimeIndex(net.equity.index), config.initial_cash
        )
    rate = config.risk_free_rate
    result = BacktestResult(
        config=config,
        net=net,
        gross=gross,
        benchmark=benchmark,
        metrics=compute_metrics(net, benchmark, rate),
        gross_metrics=compute_metrics(gross, benchmark, rate),
        benchmark_metrics=equity_metrics(benchmark, rate) if benchmark is not None else {},
        data_snapshot=snapshot,
        label=result_label(config),
    )
    if cost_stress and config.cost_stress:
        points = []
        for multiplier in config.cost_stress:
            if multiplier == 1.0:
                metrics = result.metrics
            else:
                scaled = config.model_copy(
                    update={"cost_multiplier": config.cost_multiplier * multiplier}
                )
                stressed = _evaluated(scaled, _simulate(scaled, data, free=False))
                metrics = compute_metrics(stressed, benchmark, rate)
                result.stress_metrics[float(multiplier)] = metrics
            points.append(StressPoint.from_metrics(multiplier, metrics))
        result.cost_sensitivity = CostSensitivity.from_points(points)
    sources = None
    if snapshot is not None:  # read from the store: the instruments table knows sources
        sources = data_sources(data.symbols, data_dir)
    result.assumptions = build_assumptions(
        config, data.symbols, data.asset_classes,
        result=result, engine_core=engine_core, sources=sources,
    )  # fmt: skip
    return result


def data_sources(symbols: list[str], data_dir: Path | str | None = None) -> dict[str, str]:
    """The data source of each of ``symbols`` that the store knows."""
    path = storage_db_path(data_dir)
    if not path.exists():
        return {}
    with storage.connect(path, read_only=True) as con:
        instruments = storage.read_instruments(con)
    known = instruments[instruments["symbol"].isin(symbols)]
    return dict(zip(known["symbol"], known["source"], strict=True))


def _number(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def provenance_rows(
    commit: str | None,
    dirty: bool | None,
    data: DataFingerprint | None,
    environment: provenance.Environment | None,
    data_written: object,
) -> dict[str, str]:
    """The report's Provenance table: where the result came from, one line per item."""
    unknown = "not recorded"
    rows = {
        "Git commit": f"{commit or 'unknown'}{' (uncommitted changes)' if dirty else ''}",
        "Data hash (SHA-256)": data.hash if data else unknown,
        "Data rows (bars)": f"{data.rows:,}" if data else unknown,
        "Data dates": f"{data.first_date} to {data.last_date}" if data and data.rows else unknown,
        "Data as of": "the store when the run started",
        "Data last written (UTC)": str(data_written or "unknown"),
    }
    if environment is not None:
        rows |= {
            "Python": environment.python_version,
            "uv.lock SHA-256": environment.uv_lock_sha256 or "no uv.lock found",
            "Engine core": environment.engine_core or "none (vectorized engine)",
            "Rust core version": environment.rust_core_version or "not used",
            "Operating system": environment.os,
        }
    return rows


def execute(
    config: BacktestConfig,
    *,
    data_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    universe_config: Path | str | None = None,
    notes: str | None = None,
    cost_stress: bool = True,
    unseal_reason: str | None = None,
) -> RunOutcome:
    """Run ``config``, write its tearsheet, and log it to the registry.

    A run that raises is logged too, with status ``failed`` and the error as its notes,
    and the exception is returned in the outcome rather than raised: a failed idea is
    still an idea that was tried.

    Each cost-stress run other than 1x is logged as a row of its own with
    ``parent_run_id`` set to this run, and gets no tearsheet. ``cost_stress=False``
    skips them.

    A config on a sealed segment goes through the holdout rules first
    (:mod:`qr.research.holdout`): the first run is stored; the same config again returns
    the stored result without running (``outcome.sealed``); a different one raises
    :class:`~qr.research.holdout.HoldoutError` before anything is run or logged, unless
    ``unseal_reason`` is given.
    """
    sealed_segment = (
        config.split is not None
        and config.segment is not None
        and load_split(config.split).segment(config.segment).sealed
    )
    previous: holdout.SealedResult | None = None
    if sealed_segment:
        assert config.split is not None and config.segment is not None
        stored = holdout.check(
            config.study_id, config.split, config.segment, config.config_hash(),
            unseal_reason, data_dir,
        )  # fmt: skip
        if stored is not None:
            rows = registry.read_runs(data_dir, run_ids=[stored.run_id])
            if rows.empty:
                raise holdout.HoldoutError(
                    f"the sealed result of run {stored.run_id} is stored, but the run is "
                    "missing from the registry"
                )
            return RunOutcome(
                record=registry.record_from_row(rows.iloc[0]), result=None, sealed=stored
            )
        previous = holdout.find(config.study_id or "", config.split, config.segment, data_dir)
    reused = unseal_reason.strip() if previous is not None and unseal_reason else None
    created = storage.utcnow()
    run_id = registry.new_run_id(created)
    commit, dirty = registry.git_state()
    result: BacktestResult | None = None
    error: Exception | None = None
    report: Path | None = None
    environment: provenance.Environment | None = None
    data: DataFingerprint | None = None
    try:
        environment = provenance.environment(config)
        universe = resolve_universe(config, universe_config)
        data = provenance.run_fingerprint(
            provenance.data_symbols(config, universe), config.end, data_dir=data_dir
        )
        result = run_backtest(
            config,
            data_dir=data_dir,
            universe_config=universe_config,
            cost_stress=cost_stress,
            engine_core=environment.engine_core,
        )
        result.holdout_reused = reused
        report = Path(reports_dir or DEFAULT_REPORTS_DIR) / f"{run_id}_{config.name}.html"
        write_report(
            result,
            report,
            {
                "Run id": run_id,
                "Created (UTC)": f"{created:%Y-%m-%d %H:%M:%S}",
                "Config hash": config.config_hash(),
                "Notes": notes or config.notes or "",
            },
            provenance_rows(commit, dirty, data, environment, result.data_snapshot),
            data.per_symbol,
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
        data_hash=data.hash if data else None,
        data_per_symbol_json=json.dumps(data.per_symbol, sort_keys=True) if data else None,
        data_rows=data.rows if data else None,
        data_as_of=None,
        python_version=environment.python_version if environment else None,
        uv_lock_sha256=environment.uv_lock_sha256 if environment else None,
        engine_core=environment.engine_core if environment else None,
        rust_core_version=environment.rust_core_version if environment else None,
        os=environment.os if environment else None,
        cost_multiplier=config.cost_multiplier,
        **_stress_columns(result if error is None else None),
        split_name=config.split,
        segment=config.segment,
        study_id=config.study_id,
        result_label=result_label(config),
        unseal_reason=reused,
    )
    registry.log_run(record, data_dir)
    if sealed_segment and error is None and result is not None:
        assert config.split is not None and config.segment is not None
        if previous is None:
            holdout.store(
                config.study_id or "", config.split, config.segment, config.config_hash(),
                record.data_hash, run_id, result.metrics, created, data_dir,
            )  # fmt: skip
        elif reused is not None:
            holdout.record_override(
                config.study_id or "", config.split, config.segment, run_id,
                config.config_hash(), previous.config_hash, reused, created, data_dir,
            )  # fmt: skip
    if error is None and result is not None:
        for multiplier, stressed in result.stress_metrics.items():
            registry.log_run(_stress_record(record, config, multiplier, stressed), data_dir)
    return RunOutcome(record=record, result=result if error is None else None, error=error)


def _stress_columns(result: BacktestResult | None) -> dict[str, Any]:
    """The registry columns a run gets from its assumptions and cost stress."""
    if result is None:
        return {}
    sensitivity = result.cost_sensitivity
    found = None if sensitivity is None else sensitivity.break_even
    return {
        "assumptions_json": None if result.assumptions is None else json.dumps(result.assumptions),
        "cost_sensitivity_json": None if sensitivity is None else json.dumps(sensitivity.to_dict()),
        "break_even_multiplier": None if found is None else found.multiplier,
        "fragile": None if sensitivity is None else sensitivity.fragile,
    }


def _stress_record(
    parent: registry.RunRecord,
    config: BacktestConfig,
    multiplier: float,
    metrics: dict[str, float],
) -> registry.RunRecord:
    """The registry row of ``parent``'s stress run at ``multiplier``."""
    scaled = config.model_copy(update={"cost_multiplier": config.cost_multiplier * multiplier})
    return replace(
        parent,
        run_id=registry.stress_run_id(parent.run_id, multiplier),
        config_json=scaled.to_json(),
        config_hash=scaled.config_hash(),
        sharpe=_number(metrics.get("sharpe")),
        cagr=_number(metrics.get("cagr")),
        max_drawdown=_number(metrics.get("max_drawdown")),
        turnover=_number(metrics.get("turnover")),
        report_path=None,
        notes=f"cost stress {multiplier:g}x of {parent.run_id}",
        parent_run_id=parent.run_id,
        cost_multiplier=scaled.cost_multiplier,
        assumptions_json=None,
        cost_sensitivity_json=None,
        break_even_multiplier=None,
        fragile=None,
    )
