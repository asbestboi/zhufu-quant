"""Parameter sweeps, with every trial counted.

Searching parameters finds lucky results, so a sweep logs every trial it runs, and a
study's trial count is the number of everything ever tried under its ``study_id``.

A sweep spec (``configs/sweeps/*.yaml``)::

    name: sma_spy_grid
    base: ../backtests/sma_spy_split.yaml   # relative to this file; must name a split
    study_id: sma_spy_v1
    segment: train                          # the default; a sealed segment is refused
    grid:                                   # every combination, in this order ...
      fast: [10, 20, 50]
      slow: [100, 200]
    # ... or a random search instead of a grid:
    # random: { n: 50, seed: 42, ranges: { fast: { low: 5, high: 60, int: true },
    #                                      slow: [100, 150, 200] } }

``qr sweep run SPEC [--workers N]`` runs every trial on the segment and logs each as a
registry row with ``sweep_id``, ``study_id``, ``trial_index``, its parameters, and its
status. A trial that fails is logged as ``failed`` with the error, and still counts.
Trials compute screening metrics only: no tearsheet and no cost stress.

With ``--workers`` the trials run in a pool of fresh ("spawned") processes, which find
plugin strategies through their entry points. Results are put back in trial order,
and each trial gets the same seed (``trial_seed``) however it is run, so a parallel
sweep logs exactly what a serial one does.

**Parameter stability** (grids only): the best trial's Sharpe against the median Sharpe
of its grid neighbours (every point one step away in one or more parameters). The best
trial is flagged as a **spike** when that median is below ``STABILITY_SHARE`` of its
Sharpe, or when its Sharpe is not positive.
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing
import random
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qr import storage
from qr.backtest import provenance, registry
from qr.backtest.config import BacktestConfig, backtest_config_from_dict
from qr.config import ConfigError
from qr.research.splits import load_split

STABILITY_SHARE = 0.75  # the neighbours' median must keep this share of the best Sharpe

SWEEPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps (
  sweep_id TEXT PRIMARY KEY, created_at TIMESTAMP, name TEXT, study_id TEXT,
  split_name TEXT, segment TEXT, kind TEXT, spec_json TEXT, base_config_hash TEXT,
  trials INTEGER
);
"""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Range(_Model):
    """A numeric range for a random search: uniform in ``[low, high]``."""

    low: float
    high: float
    int: bool = False

    @model_validator(mode="after")
    def _check(self) -> Range:
        if self.high < self.low:
            raise ValueError(f"range high {self.high} is below low {self.low}")
        return self


class RandomSearch(_Model):
    n: int = Field(ge=1)
    seed: int = 0
    ranges: dict[str, Range | list[Any]] = Field(min_length=1)


class SweepSpec(_Model):
    """A validated sweep file."""

    name: str = Field(min_length=1)
    base: str = Field(min_length=1)
    study_id: str = Field(min_length=1)
    segment: str = "train"
    grid: dict[str, list[Any]] | None = None
    random: RandomSearch | None = None

    @model_validator(mode="after")
    def _check(self) -> SweepSpec:
        if (self.grid is None) == (self.random is None):
            raise ValueError("give either grid or random")
        if self.grid is not None:
            empty = [k for k, v in self.grid.items() if not v]
            if not self.grid or empty:
                raise ValueError(f"every grid parameter needs values: {', '.join(empty)}")
        return self

    @property
    def kind(self) -> str:
        return "grid" if self.grid is not None else "random"


@dataclass(frozen=True)
class Sweep:
    """A sweep ready to run: the spec, its base config on the segment, and the trials."""

    spec: SweepSpec
    base: BacktestConfig
    trials: list[dict[str, Any]]
    seed: int


def load_sweep(path: Path | str) -> Sweep:
    """Read a sweep file, resolve its base config onto the segment, and list its trials."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"sweep spec not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        spec = SweepSpec.model_validate(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{resolved}: invalid sweep\n{problems}") from exc
    base_path = (resolved.parent / spec.base).resolve()
    base_raw = (
        yaml.safe_load(base_path.read_text(encoding="utf-8")) if base_path.is_file() else None
    )
    if not isinstance(base_raw, dict):
        raise ConfigError(f"{resolved}: base config not found or not a mapping: {base_path}")
    if "split" not in base_raw:
        raise ConfigError(f"{resolved}: the base config must name a split (and a segment)")
    split = load_split(str(base_raw["split"]))
    if split.segment(spec.segment).sealed:
        raise ConfigError(
            f"{resolved}: segment '{spec.segment}' of split '{split.name}' is sealed; a sweep "
            "searches parameters and must never see the test"
        )
    # The sweep's segment replaces the base config's own, and the study is the sweep's.
    base_raw = {**base_raw, "segment": spec.segment, "study_id": spec.study_id}
    base = backtest_config_from_dict(base_raw, str(base_path))
    seed = spec.random.seed if spec.random is not None else 0
    return Sweep(spec, base, list_trials(spec), seed)


def list_trials(spec: SweepSpec) -> list[dict[str, Any]]:
    """Every trial's parameters, in order: grid order, or the random draws."""
    if spec.grid is not None:
        names = list(spec.grid)
        return [dict(zip(names, v, strict=True)) for v in itertools.product(*spec.grid.values())]
    assert spec.random is not None
    rng = random.Random(spec.random.seed)
    trials = []
    for _ in range(spec.random.n):
        trial: dict[str, Any] = {}
        for name, space in spec.random.ranges.items():
            if isinstance(space, list):
                trial[name] = rng.choice(space)
            elif space.int:
                trial[name] = rng.randint(int(space.low), int(space.high))
            else:
                trial[name] = rng.uniform(space.low, space.high)
        trials.append(trial)
    return trials


def trial_seed(sweep_seed: int, index: int) -> int:
    """The seed of trial ``index``: the same whichever process runs it."""
    return (sweep_seed * 1_000_003 + index) % 2**31


# --------------------------------------------------------------------------- running


@dataclass(frozen=True)
class TrialOutcome:
    """One trial: its metrics and return moments, or the error that stopped it."""

    index: int
    params: dict[str, Any]
    seed: int
    metrics: dict[str, float]
    moments: tuple[int, float, float] | None  # observations, skew, kurtosis
    error: str | None


def _run_trial(
    config_json: str, index: int, params: dict[str, Any], seed: int,
    data_dir: str | None, universe_config: str | None,
) -> TrialOutcome:  # fmt: skip
    """Run one trial. Module level, so that a process pool can pickle it."""
    from qr.backtest.runner import excess_returns, run_backtest
    from qr.research.deflated import return_moments

    random.seed(seed)
    np.random.seed(seed)
    try:
        config = BacktestConfig.model_validate_json(config_json)
        config = config.model_copy(update={"params": {**config.params, **params}})
        result = run_backtest(
            config, data_dir=data_dir, universe_config=universe_config, cost_stress=False
        )
        moments = return_moments(excess_returns(result))
        return TrialOutcome(
            index, params, seed, result.metrics,
            (moments.observations, moments.skew, moments.kurtosis), None,
        )  # fmt: skip
    except Exception as exc:  # a failed trial is still a trial
        return TrialOutcome(index, params, seed, {}, None, f"{type(exc).__name__}: {exc}")


@dataclass
class SweepResult:
    """A finished sweep."""

    sweep_id: str
    sweep: Sweep
    outcomes: list[TrialOutcome]
    n_trials: int  # the whole study's count after this sweep

    def frame(self) -> pd.DataFrame:
        """One row per trial, in trial order: parameters, status, and headline metrics."""
        rows = []
        for o in self.outcomes:
            rows.append(
                {"trial": o.index, **o.params, "status": "failed" if o.error else "ok",
                 **{k: o.metrics.get(k, math.nan)
                    for k in ("sharpe", "cagr", "max_drawdown", "turnover", "trades")},
                 "error": o.error or ""}
            )  # fmt: skip
        return pd.DataFrame(rows)


def run_sweep(
    sweep: Sweep,
    *,
    workers: int = 1,
    data_dir: Path | str | None = None,
    universe_config: Path | str | None = None,
) -> SweepResult:
    """Run every trial of ``sweep`` and log each one (and the sweep) to the registry."""
    created = storage.utcnow()
    sweep_id = f"sw-{created:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    config_json = sweep.base.to_json()
    data = None if data_dir is None else str(data_dir)
    universe = None if universe_config is None else str(universe_config)
    jobs = [
        (config_json, i, params, trial_seed(sweep.seed, i), data, universe)
        for i, params in enumerate(sweep.trials)
    ]
    if workers > 1:
        # "spawn", not fork: forking a process that holds DuckDB's threads can deadlock,
        # and it is what Windows does anyway, so every platform behaves the same.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            outcomes = list(pool.map(_run_trial, *zip(*jobs, strict=True)))
    else:
        outcomes = [_run_trial(*job) for job in jobs]
    outcomes.sort(key=lambda o: o.index)
    _log(sweep, sweep_id, created, outcomes, data_dir, universe_config)
    return SweepResult(sweep_id, sweep, outcomes, count_trials(sweep.spec.study_id, data_dir))


def _log(
    sweep: Sweep,
    sweep_id: str,
    created: datetime,
    outcomes: Sequence[TrialOutcome],
    data_dir: Path | str | None,
    universe_config: Path | str | None,
) -> None:
    from qr.backtest.runner import resolve_universe, result_label, risk_free_label

    base = sweep.base
    commit, dirty = registry.git_state()
    environment = provenance.environment(base)
    symbols = resolve_universe(base, universe_config)
    try:
        data = provenance.run_fingerprint(
            provenance.data_symbols(base, symbols), base.end, data_dir=data_dir
        )
    except Exception:
        data = None
    label = result_label(base)
    for outcome in outcomes:
        config = base.model_copy(update={"params": {**base.params, **outcome.params}})

        def number(key: str, metrics: Mapping[str, float] = outcome.metrics) -> float | None:
            value = metrics.get(key)
            return None if value is None or not math.isfinite(value) else float(value)

        moments = outcome.moments
        registry.log_run(
            registry.RunRecord(
                run_id=f"{sweep_id}-t{outcome.index:04d}",
                created_at=created,
                strategy=config.strategy,
                engine=config.engine,
                config_json=config.to_json(),
                config_hash=config.config_hash(),
                git_commit=commit,
                git_dirty=dirty,
                data_snapshot=None,
                start_date=config.start,
                end_date=config.end,
                sharpe=number("sharpe"),
                cagr=number("cagr"),
                max_drawdown=number("max_drawdown"),
                turnover=number("turnover"),
                report_path=None,
                notes=outcome.error or f"sweep {sweep.spec.name} trial {outcome.index}",
                name=config.name,
                status=registry.FAILED if outcome.error else registry.OK,
                data_hash=data.hash if data else None,
                data_per_symbol_json=json.dumps(data.per_symbol, sort_keys=True) if data else None,
                data_rows=data.rows if data else None,
                python_version=environment.python_version,
                uv_lock_sha256=environment.uv_lock_sha256,
                engine_core=environment.engine_core,
                rust_core_version=environment.rust_core_version,
                os=environment.os,
                cost_multiplier=config.cost_multiplier,
                split_name=config.split,
                segment=config.segment,
                study_id=config.study_id,
                result_label=label,
                sweep_id=sweep_id,
                trial_index=outcome.index,
                params_json=json.dumps(outcome.params, sort_keys=True),
                trial_seed=outcome.seed,
                risk_free=risk_free_label(config),
                n_returns=None if moments is None else moments[0],
                return_skew=None
                if moments is None or not math.isfinite(moments[1])
                else moments[1],
                return_kurtosis=None
                if moments is None or not math.isfinite(moments[2])
                else moments[2],
            ),
            data_dir,
        )
    with storage.connect(registry.registry_path(data_dir)) as con:
        con.execute(SWEEPS_SCHEMA)
        con.execute(
            "INSERT INTO sweeps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [sweep_id, created, sweep.spec.name, sweep.spec.study_id, base.split, base.segment,
             sweep.spec.kind, sweep.spec.model_dump_json(), base.config_hash(), len(outcomes)],
        )  # fmt: skip


# --------------------------------------------------------------------------- counting


def count_trials(study_id: str, data_dir: Path | str | None = None) -> int:
    """Every trial ever run under ``study_id``: all sweeps (failed trials too) and runs.

    Cost-stress runs (same parameters, other costs) and walk-forward runs (reported
    separately) are not trials. A sealed result shown again is not a run at all.
    """
    return len(study_trials(study_id, data_dir))


def study_trials(study_id: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """The registry rows that count as trials of ``study_id``."""
    runs = registry.read_runs(data_dir)  # without stress and walk-forward runs
    if runs.empty:
        return runs
    return runs[runs["study_id"] == study_id].reset_index(drop=True)


def read_sweep(
    sweep_id: str, data_dir: Path | str | None = None
) -> tuple[dict[str, Any], pd.DataFrame]:
    """A logged sweep's row (as a dict) and its trials, in trial order."""
    path = registry.registry_path(data_dir)
    if not path.exists():
        raise ValueError(f"no sweep '{sweep_id}': the registry is empty")
    with storage.connect(path, read_only=True) as con:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        if "sweeps" not in tables:
            raise ValueError(f"no sweep '{sweep_id}'")
        found = con.execute("SELECT * FROM sweeps WHERE starts_with(sweep_id, ?)", [sweep_id]).df()
    if len(found) != 1:
        raise ValueError(f"no sweep '{sweep_id}'" if found.empty else f"'{sweep_id}' is ambiguous")
    row = {str(k): v for k, v in found.iloc[0].to_dict().items()}
    runs = registry.read_runs(data_dir)
    trials: pd.DataFrame = runs[runs["sweep_id"] == row["sweep_id"]].sort_values("trial_index")
    return row, trials.reset_index(drop=True)


# --------------------------------------------------------------------------- stability


@dataclass(frozen=True)
class Stability:
    """The best trial against its grid neighbours."""

    best_index: int
    best_sharpe: float
    neighbours: int
    neighbour_median: float
    spike: bool


def grid_neighbours(grid: Mapping[str, Sequence[Any]], index: int) -> list[int]:
    """Trial indices one step away from ``index`` in one or more grid parameters."""
    sizes = [len(v) for v in grid.values()]
    position = np.unravel_index(index, sizes)
    out = []
    for step in itertools.product((-1, 0, 1), repeat=len(sizes)):
        if not any(step):
            continue
        moved = [p + s for p, s in zip(position, step, strict=True)]
        if all(0 <= m < n for m, n in zip(moved, sizes, strict=True)):
            out.append(int(np.ravel_multi_index(moved, sizes)))
    return sorted(out)


def stability(grid: Mapping[str, Sequence[Any]], sharpes: Sequence[float]) -> Stability | None:
    """The parameter-stability check of a grid sweep; None when no trial has a Sharpe."""
    finite = [(i, s) for i, s in enumerate(sharpes) if math.isfinite(s)]
    if not finite:
        return None
    best, best_sharpe = max(finite, key=lambda pair: (pair[1], -pair[0]))
    around = [sharpes[i] for i in grid_neighbours(grid, best) if math.isfinite(sharpes[i])]
    median = float(np.median(around)) if around else math.nan
    spike = best_sharpe <= 0 or not (median >= STABILITY_SHARE * best_sharpe)
    return Stability(best, best_sharpe, len(around), median, spike)


# --------------------------------------------------------------------------- report


@dataclass(frozen=True)
class SweepSummary:
    """Everything a sweep report shows, read back from the registry."""

    sweep: dict[str, Any]
    trials: pd.DataFrame  # trial rows, in trial order, with a column per parameter
    grid: dict[str, list[Any]] | None
    n_trials: int  # the study's count
    stability: Stability | None
    best_index: int | None  # the trial with the highest Sharpe
    deflated: Any = None  # its Deflated Sharpe Ratio against every trial of the study
    psr: float = math.nan  # its Probabilistic Sharpe Ratio (against zero)


def summarize_sweep(sweep_id: str, data_dir: Path | str | None = None) -> SweepSummary:
    """Read a logged sweep and compute its stability check."""
    row, trials = read_sweep(sweep_id, data_dir)
    spec = json.loads(str(row["spec_json"]))
    params = pd.DataFrame([json.loads(str(p)) for p in trials["params_json"]])
    table = pd.concat([trials.reset_index(drop=True), params], axis=1)
    grid = spec.get("grid")
    sharpes = [float(v) if pd.notna(v) else math.nan for v in trials["sharpe"]]
    found = stability(grid, sharpes) if grid else None
    finite = [(i, v) for i, v in enumerate(sharpes) if math.isfinite(v)]
    best = max(finite, key=lambda pair: (pair[1], -pair[0]))[0] if finite else None
    deflated, psr = None, math.nan
    if best is not None:
        from qr.research.deflated import moments_from_row, probabilistic_sharpe, study_deflated

        moments = moments_from_row({str(k): v for k, v in trials.iloc[best].to_dict().items()})
        if moments is not None:
            deflated = study_deflated(str(row["study_id"]), moments, data_dir)
            psr = probabilistic_sharpe(moments, 0.0)
    return SweepSummary(
        row, table, grid, count_trials(str(row["study_id"]), data_dir), found, best, deflated, psr
    )


def heatmap_frame(summary: SweepSummary) -> pd.DataFrame | None:
    """Sharpe by the two parameters of a two-parameter grid; None otherwise."""
    if not summary.grid or len(summary.grid) != 2:
        return None
    rows, columns = list(summary.grid)
    frame = summary.trials.pivot_table(
        index=rows, columns=columns, values="sharpe", aggfunc="first"
    )
    return frame.reindex(index=summary.grid[rows], columns=summary.grid[columns])


def render_sweep(summary: SweepSummary) -> str:
    """The sweep report as one HTML page."""
    import html as html_module

    from qr.backtest import report as tearsheet

    escape = html_module.escape
    names = (
        list(summary.grid)
        if summary.grid
        else sorted({k for p in summary.trials["params_json"] for k in json.loads(str(p))})
    )
    columns = ["Trial", *names, "Status", "Sharpe", "CAGR", "Max drawdown", "Turnover"]
    head = "".join(f"<th>{escape(n)}</th>" for n in columns)
    best = summary.best_index
    body = []
    for row in summary.trials.itertuples(index=False):
        mark = ' style="font-weight:600"' if row.trial_index == best else ""
        cells = [str(row.trial_index), *(escape(str(getattr(row, n))) for n in names),
                 escape(str(row.status)),
                 tearsheet.format_metric(_float(row.sharpe), "num"),
                 tearsheet.format_metric(_float(row.cagr), "pct"),
                 tearsheet.format_metric(_float(row.max_drawdown), "pct"),
                 tearsheet.format_metric(_float(row.turnover), "num")]  # fmt: skip
        body.append(f"<tr{mark}>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    failed = int((summary.trials["status"] == registry.FAILED).sum())
    sweep = summary.sweep
    chart = _heatmap_svg(summary)
    study = escape(str(sweep["study_id"]))
    segment = f"{escape(str(sweep['segment']))} of {escape(str(sweep['split_name']))}"
    best_sharpe = _best(summary)
    found = summary.deflated
    dsr = (
        ""
        if found is None
        else tearsheet._tile(
            "Deflated Sharpe (DSR)",
            tearsheet.format_metric(found.dsr, "pct"),
            f"best trial; PSR {tearsheet.format_metric(summary.psr, 'pct')}; luck alone reaches "
            f"Sharpe {found.benchmark_annualized:.2f} in {found.n_trials} trials",
        )
    )
    tiles = (
        tearsheet._tile(
            "Trials in study", f"{summary.n_trials:,}", f"study {study}, every sweep and run"
        )
        + tearsheet._tile(
            "Best Sharpe",
            tearsheet.format_metric(best_sharpe, "num"),
            f"best of {len(summary.trials)} in this sweep",
        )
        + dsr
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(str(sweep["name"]))} sweep</title>
<style>{tearsheet._CSS}</style>
</head>
<body><main>
<h1>{escape(str(sweep["name"]))}: sweep</h1>
<p class="sub"><span class="label">{segment}</span>
· {escape(str(sweep["kind"]))} · {len(summary.trials)} trials ({failed} failed) ·
<code>{escape(str(sweep["sweep_id"]))}</code></p>
<div class="tiles">{tiles}</div>
<p class="note">Risk-free rate for every Sharpe and the DSR:
<strong>{escape(risk_free_of(summary))}</strong>.</p>
<p class="note">The best of many trials looks good by luck alone. Count every trial: the
study's total is what the Deflated Sharpe Ratio uses. DSR is the probability that the
best trial's true Sharpe is above what that many worthless trials would reach by luck.</p>
{_stability_html(summary)}
{f"<figure>{chart}</figure>" if chart else ""}
<h2>Trials</h2>
<div class="scroll"><table><tr>{head}</tr>{"".join(body)}</table></div>
<h2>Spec</h2>
<pre>{escape(json.dumps(json.loads(str(sweep["spec_json"])), indent=2))}</pre>
</main></body>
</html>
"""


def risk_free_of(summary: SweepSummary) -> str:
    """The risk-free rate(s) the sweep's trials were measured with."""
    labels = {str(v) for v in summary.trials.get("risk_free", pd.Series(dtype=object)).dropna()}
    return ", ".join(sorted(labels)) or "not recorded"


def _float(value: Any) -> float:
    return math.nan if value is None or pd.isna(value) else float(value)


def _best(summary: SweepSummary) -> float:
    values = [v for v in summary.trials["sharpe"] if pd.notna(v)]
    return float(max(values)) if values else math.nan


def _stability_html(summary: SweepSummary) -> str:
    found = summary.stability
    if found is None:
        return "<p>Parameter stability: not applicable (random search, or no Sharpe).</p>"
    verdict = '<span class="flag">lone spike</span>' if found.spike else "a plateau, not a spike"
    return (
        f"<p>Parameter stability: best trial {found.best_index} (Sharpe {found.best_sharpe:.2f})"
        f" against the median of its {found.neighbours} grid neighbours "
        f"({found.neighbour_median:.2f}, needs at least {STABILITY_SHARE:.0%} of the best): "
        f"{verdict}.</p>"
    )


def _heatmap_svg(summary: SweepSummary) -> str | None:
    frame = heatmap_frame(summary)
    if frame is None:
        return None
    import matplotlib

    from qr.backtest import report as tearsheet

    with matplotlib.rc_context(tearsheet._RC):  # type: ignore[arg-type]
        figure, axes = tearsheet._figure(3.0)
        values = frame.to_numpy(dtype="float64")
        image = axes.imshow(values, cmap="RdYlGn", aspect="auto")
        axes.set_xticks(range(len(frame.columns)), [str(c) for c in frame.columns])
        axes.set_yticks(range(len(frame.index)), [str(i) for i in frame.index])
        axes.set_xlabel(str(frame.columns.name))
        axes.set_ylabel(str(frame.index.name))
        for (i, j), value in np.ndenumerate(values):
            if math.isfinite(value):
                axes.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axes, label="Sharpe after costs")
        axes.set_title("Train Sharpe by parameter")
        return tearsheet._svg(figure)


def write_sweep_report(summary: SweepSummary, reports_dir: Path | str) -> Path:
    """Write the report and return its path."""
    target = Path(reports_dir) / f"{summary.sweep['sweep_id']}_{summary.sweep['name']}_sweep.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_sweep(summary), encoding="utf-8")
    return target
