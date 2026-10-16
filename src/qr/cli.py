"""The ``qr`` command line interface."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from qr import crosscheck as crosscheck_module
from qr import snapshots, storage
from qr.adjust import Adjustment
from qr.backtest.config import load_backtest_config
from qr.backtest.provenance import per_symbol_hashes, verify_run
from qr.backtest.registry import read_runs
from qr.backtest.report import METRIC_ROWS, format_metric
from qr.backtest.runner import DEFAULT_REPORTS_DIR, execute, risk_free_label
from qr.backtest.stress import CostSensitivity
from qr.config import (
    ConfigError,
    UniverseConfig,
    db_path,
    load_config,
    load_env_file,
    raw_dir,
)
from qr.ingest import FAILED, PARTIAL, run_ingest
from qr.load import get_bars
from qr.research.holdout import HoldoutError, SealedResult
from qr.research.sweeps import SweepSummary
from qr.sources.base import SourceError
from qr.storage import StorageError
from qr.validate import summarize, validate_symbols

app = typer.Typer(help="Personal quant research: market data pipeline and backtesting.")
console = Console()

ConfigOption = Annotated[
    Path | None, typer.Option("--config", help="Universe config (default: config/universe.yaml).")
]
DataDirOption = Annotated[
    Path | None, typer.Option("--data-dir", help="Data directory (default: ./data).")
]
SymbolsOption = Annotated[
    str | None, typer.Option("--symbols", help="Comma-separated symbols (default: all).")
]
UniverseOption = Annotated[
    str | None, typer.Option("--universe", help="A named group from the config's `groups:`.")
]


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
) -> None:
    """Set up logging and read ``.env`` for every command.

    Run without a command, ``qr`` opens the interactive menu in a terminal and prints this
    help everywhere else (scripts, schedulers, CI), so it never waits for input there.
    """
    load_env_file()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party chatter is only useful when debugging.
    for noisy in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbose else logging.CRITICAL)
    if ctx.invoked_subcommand is None:
        _menu_or_help(ctx, no_banner=False)


def _menu_or_help(ctx: typer.Context, no_banner: bool) -> None:
    """Open the menu in an interactive terminal; otherwise print the help and exit 0."""
    from qr import tui  # imported here: the menu's libraries are only needed in a terminal

    if not tui.is_interactive():
        root = ctx.find_root()
        root.command.get_help(root)  # Typer prints the help itself
        raise typer.Exit(code=0)
    tui.run(no_banner=no_banner)


@app.command()
def menu(
    ctx: typer.Context,
    no_banner: Annotated[
        bool, typer.Option("--no-banner", help="Skip the logo (or set QR_NO_BANNER=1).")
    ] = False,
) -> None:
    """Open the interactive menu (also what `qr` alone does in a terminal)."""
    _menu_or_help(ctx, no_banner)


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=1)


def _config(path: Path | None) -> UniverseConfig:
    try:
        return load_config(path)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc


def _symbols(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def _selection(cfg: UniverseConfig, symbols: str | None, universe: str | None) -> list[str] | None:
    """The symbols a command should act on; None means everything."""
    if symbols is not None and universe is not None:
        raise _fail("use either --symbols or --universe, not both")
    if universe is not None:
        try:
            return cfg.group(universe)
        except ConfigError as exc:
            raise _fail(str(exc)) from exc
    return _symbols(symbols)


def _date(value: Any) -> str:
    return "-" if pd.isna(value) else str(pd.Timestamp(value).date())


def _print_issue_summary(issues: pd.DataFrame) -> None:
    summary = summarize(issues)
    if summary.empty:
        console.print("[green]No quality issues.[/green]")
        return
    table = Table(title="Quality issues")
    for column in ("symbol", "check", "severity", "count", "first date", "last date"):
        table.add_column(column, justify="right" if column == "count" else "left")
    for row in summary.itertuples():
        colour = "red" if row.severity == "error" else "yellow"
        table.add_row(
            str(row.symbol),
            str(row.check_name),
            f"[{colour}]{row.severity}[/{colour}]",
            str(row.count),
            _date(row.first_date),
            _date(row.last_date),
        )
    console.print(table)
    errors = int((issues["severity"] == "error").sum())
    console.print(f"{len(issues)} findings: {errors} errors, {len(issues) - errors} warnings")


@app.command()
def init(data_dir: DataDirOption = None) -> None:
    """Create the database and its tables."""
    path = db_path(data_dir)
    with storage.connect(path) as con:
        storage.init_schema(con)
    console.print(f"Database ready at {path}")


@app.command()
def ingest(
    symbols: SymbolsOption = None,
    universe: UniverseOption = None,
    config: ConfigOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Incrementally download bars and corporate actions, then run quality checks."""
    cfg = _config(config)
    report = run_ingest(cfg, data_dir, _selection(cfg, symbols, universe))

    table = Table(title=f"Ingest run {report.run_id}")
    for column in ("symbol", "status", "rows written", "message"):
        table.add_column(column, justify="right" if column == "rows written" else "left")
    colours = {FAILED: "red", PARTIAL: "red", "ok": "green", "skipped": "dim"}
    for result in report.results:
        colour = colours[result.status]
        table.add_row(
            result.symbol,
            f"[{colour}]{result.status}[/{colour}]",
            str(result.rows_written),
            result.message,
        )
    console.print(table)
    with storage.connect(db_path(data_dir), read_only=True) as con:
        _print_issue_summary(storage.read_quality_issues(con))
    for source, reason in report.sources_down.items():
        console.print(
            f"[red]error:[/red] could not reach the '{source}' source, so its symbols were "
            f"skipped.\n  {reason}\n  Check your internet connection, VPN, or proxy and try "
            "again later; the provider may also be rate limiting you. Stored data is untouched."
        )
    if report.failed:
        raise _fail("one or more symbols failed, were incomplete, or were skipped")


@app.command()
def validate(
    symbols: SymbolsOption = None,
    universe: UniverseOption = None,
    config: ConfigOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Run quality checks on stored data and print a summary of all open findings."""
    cfg = _config(config)
    selected = _selection(cfg, symbols, universe)
    try:
        with storage.connect(db_path(data_dir)) as con:
            storage.init_schema(con)
            validate_symbols(con, cfg, selected)
            issues = storage.read_quality_issues(con, selected)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    _print_issue_summary(issues)
    if (issues["severity"] == "error").any():
        raise typer.Exit(code=1)


@app.command()
def status(
    universe: UniverseOption = None, config: ConfigOption = None, data_dir: DataDirOption = None
) -> None:
    """Show per-symbol coverage: first/last date, row count, and open issues."""
    # The config is only needed to look up a named universe.
    members = _selection(_config(config), None, universe) if universe is not None else None
    try:
        with storage.connect(db_path(data_dir), read_only=True) as con:
            coverage = storage.read_coverage(con)
    except StorageError as exc:
        raise _fail(str(exc)) from exc
    if members is not None:
        coverage = coverage[coverage["symbol"].isin(members)]
    if coverage.empty:
        console.print("No instruments yet. Run `qr ingest`.")
        return
    table = Table(title="Coverage")
    for column in ("symbol", "source", "class", "active", "first date", "last date"):
        table.add_column(column)
    for column in ("rows", "errors", "warnings"):
        table.add_column(column, justify="right")
    for row in coverage.to_dict("records"):
        table.add_row(
            str(row["symbol"]),
            str(row["source"]),
            str(row["asset_class"]),
            "yes" if row["is_active"] else "no",
            _date(row["first_date"]),
            _date(row["last_date"]),
            f"{int(row['n_rows']):,}",
            str(int(row["errors"])),
            str(int(row["warnings"])),
        )
    console.print(table)


@app.command()
def crosscheck(
    symbols: SymbolsOption = None,
    universe: UniverseOption = None,
    days: Annotated[int, typer.Option("--days", min=1, help="Calendar days to compare.")] = 365,
    config: ConfigOption = None,
    data_dir: DataDirOption = None,
) -> None:
    """Compare stored closes and splits with a second source (Tiingo)."""
    cfg = _config(config)
    selected = _selection(cfg, symbols, universe)
    api_key = os.environ.get("TIINGO_API_KEY", "").strip()
    if not api_key:
        console.print(
            "Cross-checking needs a Tiingo API key, and TIINGO_API_KEY is not set.\n"
            "  1. Create a free account at https://www.tiingo.com\n"
            "  2. Copy your token from https://www.tiingo.com/account/api/token\n"
            "  3. Put TIINGO_API_KEY=<your token> in a file called .env in this folder\n"
            "     (see .env.example), then run `qr crosscheck` again.\n"
            "Nothing was checked."
        )
        return
    try:
        report = crosscheck_module.run_crosscheck(cfg, data_dir, selected, days, api_key=api_key)
    except SourceError as exc:
        raise _fail(f"could not use the comparison source: {exc}") from exc

    table = Table(title=f"Cross-check against {report.source}, {report.start} to {report.end}")
    table.add_column("symbol")
    table.add_column("status")
    for column in ("days compared", "mismatches", "worst difference"):
        table.add_column(column, justify="right")
    table.add_column("note")
    colours = {"ok": "green", "mismatch": "yellow", "failed": "red", "skipped": "dim"}
    for result in report.results:
        colour = colours[result.status]
        worst = (
            f"{result.worst_difference:.3%} on {result.worst_date}" if result.days_compared else "-"
        )
        note = result.message
        if result.unmatched_days:
            note = f"{result.unmatched_days} days in only one source"
        table.add_row(
            result.symbol,
            f"[{colour}]{result.status}[/{colour}]",
            str(result.days_compared),
            str(result.mismatches),
            worst,
            note,
        )
    console.print(table)
    total = sum(r.mismatches for r in report.results)
    if total:
        console.print(
            f"{total} mismatches recorded as `crosscheck` warnings; see `qr validate`. "
            "A mismatch means the sources disagree, not which one is right."
        )
    if report.failed:
        raise _fail("one or more symbols could not be cross-checked")


@app.command()
def export(
    symbol: Annotated[str, typer.Argument(help="Symbol to export.")],
    out: Annotated[Path, typer.Option("--out", help="CSV file to write.")],
    adjustment: Annotated[
        str, typer.Option("--adjustment", help="none | split | total_return")
    ] = "total_return",
    data_dir: DataDirOption = None,
) -> None:
    """Write a symbol's (adjusted) bars to a CSV file."""
    if adjustment not in ("none", "split", "total_return"):
        raise _fail(f"unknown adjustment '{adjustment}' (use none, split, or total_return)")
    mode: Adjustment = adjustment  # type: ignore[assignment]
    try:
        bars = get_bars(symbol, adjustment=mode, data_dir=data_dir)
    except (KeyError, StorageError) as exc:
        raise _fail(str(exc).strip("'\"")) from exc
    if bars.empty:
        raise _fail(f"no bars stored for {symbol}")
    frame = bars.reset_index()
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote {len(frame):,} rows ({adjustment}) to {out}")


# --------------------------------------------------------------------------- snapshots

snapshots_app = typer.Typer(help="The raw snapshot files and their hashes.", no_args_is_help=True)
app.add_typer(snapshots_app, name="snapshots")


@snapshots_app.command("backfill")
def snapshots_backfill(data_dir: DataDirOption = None) -> None:
    """Hash raw snapshot files that are not in the manifest yet and record them."""
    raw_root = raw_dir(data_dir)
    try:
        with storage.connect(db_path(data_dir)) as con:
            storage.init_schema(con)
            added, already = snapshots.backfill_manifest(con, raw_root)
    except OSError as exc:
        raise _fail(str(exc)) from exc
    console.print(
        f"Added {added:,} snapshot files to the manifest "
        f"({already:,} were already listed) from {raw_root}"
    )


# --------------------------------------------------------------------------- backtest

backtest_app = typer.Typer(help="Run and compare backtests.", no_args_is_help=True)
app.add_typer(backtest_app, name="backtest")

ReportsDirOption = Annotated[
    Path | None, typer.Option("--reports-dir", help="Where tearsheets go (default: ./reports).")
]

# registry column, heading, and how to show it
_RUN_METRICS = [
    ("sharpe", "Sharpe", "num"),
    ("cagr", "CAGR", "pct"),
    ("max_drawdown", "max drawdown", "pct"),
    ("turnover", "turnover", "num"),
]


def _run_date(value: Any) -> str:
    return "-" if pd.isna(value) else f"{pd.Timestamp(value):%Y-%m-%d %H:%M}"


@backtest_app.command("run")
def backtest_run(
    config_file: Annotated[Path, typer.Argument(help="Backtest config (YAML).")],
    notes: Annotated[str | None, typer.Option("--notes", help="A remark for the registry.")] = None,
    config: ConfigOption = None,
    data_dir: DataDirOption = None,
    reports_dir: ReportsDirOption = None,
    engine: Annotated[
        str | None,
        typer.Option("--engine", help="Override the config's engine: vectorized | event."),
    ] = None,
    engine_core: Annotated[
        str | None,
        typer.Option(
            "--engine-core",
            help="Event-engine core for this run: python | rust (overrides the config "
            "and QR_ENGINE_CORE).",
        ),
    ] = None,
    no_cost_stress: Annotated[
        bool,
        typer.Option(
            "--no-cost-stress",
            help="Skip the runs at the config's other cost multipliers (cost_stress).",
        ),
    ] = False,
    unseal_reason: Annotated[
        str | None,
        typer.Option(
            "--unseal-reason",
            help="Run a sealed segment again with a different config. The reason is "
            "recorded permanently and the report is marked as reusing the holdout.",
        ),
    ] = None,
) -> None:
    """Run a backtest, write its HTML tearsheet, and log it to the registry.

    The same config is also run at each multiplier of its `cost_stress` (default 0x and
    2x besides 1x) for the report's Cost sensitivity table.
    """
    try:
        backtest_config = load_backtest_config(config_file)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    if engine is not None:
        if engine not in ("vectorized", "event"):
            raise _fail(f"unknown engine '{engine}' (use vectorized or event)")
        backtest_config = backtest_config.model_copy(update={"engine": engine})
    if engine_core is not None:
        if engine_core not in ("python", "rust"):
            raise _fail(f"unknown engine core '{engine_core}' (use python or rust)")
        backtest_config = backtest_config.model_copy(update={"engine_core": engine_core})
        # The environment variable outranks the config, so set it for this run too.
        os.environ["QR_ENGINE_CORE"] = engine_core
    try:
        outcome = execute(
            backtest_config,
            data_dir=data_dir,
            reports_dir=reports_dir,
            universe_config=config,
            notes=notes,
            cost_stress=not no_cost_stress,
            unseal_reason=unseal_reason,
        )
    except (HoldoutError, ConfigError) as exc:
        raise _fail(str(exc)) from exc
    if outcome.sealed is not None:
        _print_sealed(outcome.sealed)
        return
    if outcome.error is not None or outcome.result is None:
        console.print(f"Logged as failed run {outcome.record.run_id}.")
        raise _fail(str(outcome.error).strip("'\""))

    result = outcome.result
    table = Table(title=f"{backtest_config.name} ({backtest_config.engine} engine)")
    table.add_column("metric")
    columns = [("after costs", result.metrics), ("before costs", result.gross_metrics)]
    if result.benchmark is not None:
        columns.append((f"benchmark {backtest_config.benchmark}", result.benchmark_metrics))
    for heading, _ in columns:
        table.add_column(heading, justify="right")
    for key, label, kind in METRIC_ROWS:
        values = [metrics.get(key) for _, metrics in columns]
        if any(value is not None and not pd.isna(value) for value in values):
            table.add_row(label, *(format_metric(value, kind) for value in values))
    console.print(table)
    console.print(f"Risk-free rate: {risk_free_label(backtest_config)}")
    if result.cost_sensitivity is not None:
        console.print(f"Cost stress: {_stress_verdict(result.cost_sensitivity)}")
    console.print(f"Run id:  {outcome.record.run_id}")
    console.print(f"Report:  {outcome.record.report_path}")


def _print_sealed(sealed: SealedResult) -> None:
    """The stored result of a sealed segment, shown instead of running it again."""
    console.print(
        f"[yellow]Sealed segment already run[/yellow] for study {sealed.study_id} "
        f"({sealed.split_name}/{sealed.segment}) on {sealed.sealed_at:%Y-%m-%d %H:%M} UTC, "
        f"run {sealed.run_id}, with this same config. Showing the stored result; nothing "
        "was run."
    )
    table = Table(title="Sealed result (after costs)", show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, label, kind in METRIC_ROWS:
        value = sealed.metrics.get(key)
        if value is not None:
            table.add_row(label, format_metric(value, kind))
    console.print(table)


def _stress_verdict(sensitivity: CostSensitivity) -> str:
    """One line: break-even multiplier and the fragile flag."""
    found = sensitivity.break_even
    flag = {
        True: "[red]fragile[/red] (Sharpe at 2x below 0 or below half of 1x)",
        False: "[green]not fragile[/green]",
        None: "fragility not determined (needs 1x and 2x)",
    }[sensitivity.fragile]
    return f"break-even {found.describe() if found else '-'}; {flag}"


def _print_assumptions(raw: object) -> None:
    """The Assumptions section of a run, from its registry JSON."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        console.print("Assumptions: not recorded (run from before they were).")
        return
    assumptions = json.loads(str(raw))
    for warning in assumptions.get("warnings", []):
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for title, items in assumptions.get("sections", {}).items():
        table = Table(title=f"Assumptions: {title}", show_header=False)
        table.add_column("item")
        table.add_column("assumption", overflow="fold")
        for key, value in items.items():
            table.add_row(str(key), str(value))
        console.print(table)


def _print_sensitivity(raw: object) -> None:
    """The Cost sensitivity table of a run, from its registry JSON."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        console.print("Cost sensitivity: not recorded (no stress runs for this run).")
        return
    sensitivity = CostSensitivity.from_dict(json.loads(str(raw)))
    table = Table(title="Cost sensitivity")
    for column in ("costs", "CAGR", "Sharpe", "max drawdown", "turnover", "costs paid"):
        table.add_column(column, justify="left" if column == "costs" else "right")
    for point in sensitivity.points:
        table.add_row(
            f"{point.multiplier:g}x",
            format_metric(point.cagr, "pct"),
            format_metric(point.sharpe, "num"),
            format_metric(point.max_drawdown, "pct"),
            format_metric(point.turnover, "num"),
            format_metric(point.costs_paid, "money"),
        )
    console.print(table)
    console.print(f"Cost stress: {_stress_verdict(sensitivity)}")


@backtest_app.command("list")
def backtest_list(
    strategy: Annotated[
        str | None, typer.Option("--strategy", help="Only runs of this strategy.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="How many runs to show.")] = 20,
    data_dir: DataDirOption = None,
) -> None:
    """Show recent runs from the registry, newest first."""
    runs = read_runs(data_dir, strategy=strategy, limit=limit)
    if runs.empty:
        console.print("No backtest runs logged yet. Run `qr backtest run CONFIG`.")
        return
    table = Table(title="Backtest runs")
    for column in ("run id", "created (UTC)", "name", "strategy", "engine", "status"):
        table.add_column(column)
    for _, heading, _ in _RUN_METRICS:
        table.add_column(heading, justify="right")
    for row in runs.to_dict("records"):
        colour = "green" if row["status"] == "ok" else "red"
        table.add_row(
            str(row["run_id"]),
            _run_date(row["created_at"]),
            str(row["name"]),
            str(row["strategy"]),
            str(row["engine"]),
            f"[{colour}]{row['status']}[/{colour}]",
            *(format_metric(row[key], kind) for key, _, kind in _RUN_METRICS),
        )
    console.print(table)


def _one_run(runs: pd.DataFrame, wanted: str) -> dict[str, Any]:
    """The single registry row whose id starts with ``wanted``; fails otherwise."""
    matches = runs[runs["run_id"].str.startswith(wanted)]
    if len(matches) != 1:
        problem = "no run" if matches.empty else f"{len(matches)} runs"
        raise _fail(f"{problem} with an id starting with '{wanted}' (see `qr backtest list`)")
    return {str(k): v for k, v in matches.iloc[0].to_dict().items()}


def _shown(value: Any, missing: str = "not recorded") -> str:
    return (
        missing if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)
    )


@backtest_app.command("show")
def backtest_show(
    run_id: Annotated[str, typer.Argument(help="Run id (a unique prefix is enough).")],
    data_dir: DataDirOption = None,
) -> None:
    """Show one run: its full resolved config, provenance, and metrics."""
    run = _one_run(read_runs(data_dir, run_ids=[run_id]), run_id)

    summary = Table(title=f"Run {run['run_id']}", show_header=False)
    summary.add_column("field")
    summary.add_column("value", overflow="fold")
    status_colour = "green" if run["status"] == "ok" else "red"
    for label, value in (
        ("name", run["name"]),
        ("strategy", run["strategy"]),
        ("engine", run["engine"]),
        ("status", f"[{status_colour}]{run['status']}[/{status_colour}]"),
        ("created (UTC)", _run_date(run["created_at"])),
        ("period", f"{_date(run['start_date'])} to {_date(run['end_date'])}"),
        ("result", _shown(run["result_label"], "not recorded")),
        ("split / segment", f"{_shown(run['split_name'], '-')} / {_shown(run['segment'], '-')}"),
        ("study", _shown(run["study_id"], "-")),
        ("risk-free rate", _shown(run["risk_free"])),
        ("holdout reused", _shown(run["unseal_reason"], "no")),
        ("config hash", run["config_hash"]),
        ("report", _shown(run["report_path"], "none")),
        ("notes", _shown(run["notes"], "")),
    ):
        summary.add_row(label, str(value))
    console.print(summary)

    metrics = Table(title="Metrics (after costs)", show_header=False)
    metrics.add_column("metric")
    metrics.add_column("value", justify="right")
    for key, heading, kind in _RUN_METRICS:
        metrics.add_row(heading, format_metric(run[key], kind))
    study = run["study_id"]
    if isinstance(study, str) and study:
        from qr.research.deflated import moments_from_row, study_deflated
        from qr.research.sweeps import count_trials

        metrics.add_row(f"Trials in study {study}", str(count_trials(study, data_dir)))
        moments = moments_from_row(run)
        if moments is not None:
            found = study_deflated(study, moments, data_dir)
            metrics.add_row("Deflated Sharpe (DSR)", format_metric(found.dsr, "pct"))
    console.print(metrics)

    dirty = run["git_dirty"]
    has_dirty = dirty is not None and not pd.isna(dirty)
    origin = Table(title="Provenance", show_header=False)
    origin.add_column("field")
    origin.add_column("value", overflow="fold")
    for label, value in (
        (
            "git commit",
            _shown(run["git_commit"], "unknown")
            + (" (uncommitted changes)" if has_dirty and bool(dirty) else ""),
        ),
        ("data hash", _shown(run["data_hash"])),
        ("data rows (bars)", _shown(run["data_rows"])),
        (
            "data as of",
            _shown(run["data_as_of"], "the store when the run started")
            if not pd.isna(run["data_hash"])
            else "not recorded",
        ),
        ("data last written (UTC)", _shown(run["data_snapshot"])),
        ("python", _shown(run["python_version"])),
        ("uv.lock sha256", _shown(run["uv_lock_sha256"])),
        (
            "engine core",
            _shown(run["engine_core"], "none (vectorized engine)")
            if not pd.isna(run["python_version"])
            else "not recorded",
        ),
        ("rust core version", _shown(run["rust_core_version"], "not used")),
        ("operating system", _shown(run["os"])),
    ):
        origin.add_row(label, value)
    console.print(origin)
    hashes = per_symbol_hashes(run["data_per_symbol_json"])
    if hashes:
        per_symbol = Table(title="Data hash per symbol")
        per_symbol.add_column("symbol")
        per_symbol.add_column("sha256")
        for symbol, digest in sorted(hashes.items()):
            per_symbol.add_row(symbol, digest)
        console.print(per_symbol)

    _print_sensitivity(run["cost_sensitivity_json"])
    _print_assumptions(run["assumptions_json"])

    console.print("[bold]Config (resolved)[/bold]")
    console.print_json(str(run["config_json"]))


@backtest_app.command("verify")
def backtest_verify(
    run_id: Annotated[str, typer.Argument(help="Run id (a unique prefix is enough).")],
    data_dir: DataDirOption = None,
) -> None:
    """Check whether the data under a run is still exactly what it used.

    Exit code 0 when it matches, 1 when the data changed or cannot be checked.
    """
    run = _one_run(read_runs(data_dir, run_ids=[run_id]), run_id)
    try:
        outcome = verify_run(run, data_dir)
    except (ValueError, StorageError) as exc:
        raise _fail(str(exc)) from exc
    if outcome.matches:
        console.print(
            f"[green]match[/green]: run {run['run_id']} read data with hash "
            f"{outcome.recorded_hash[:16]}..., and the store still has exactly that data."
        )
        return
    console.print(
        f"[red]data changed[/red]: run {run['run_id']} used data with hash "
        f"{outcome.recorded_hash[:16]}..., now {(outcome.current_hash or 'unavailable')[:16]}..."
    )
    for symbol in outcome.changed:
        why = "no longer in the store" if symbol in outcome.missing else "different bars or actions"
        console.print(f"  {symbol}: {why}")
    console.print(
        "Rerunning the config now would read different data than this run did. "
        "See `qr validate` for recorded provider revisions."
    )
    raise typer.Exit(code=1)


@backtest_app.command("compare")
def backtest_compare(
    run_ids: Annotated[list[str], typer.Argument(help="Run ids (unique prefixes are enough).")],
    data_dir: DataDirOption = None,
) -> None:
    """Show the logged metrics of several runs side by side."""
    runs = read_runs(data_dir, run_ids=run_ids)
    chosen = [_one_run(runs, wanted) for wanted in run_ids]
    table = Table(title="Backtest comparison")
    table.add_column("")
    for run in chosen:
        table.add_column(str(run["run_id"]), justify="right")
    for field in ("name", "strategy", "engine", "status"):
        table.add_row(field, *(str(run[field]) for run in chosen))
    table.add_row(
        "period", *(f"{_date(r['start_date'])} to {_date(r['end_date'])}" for r in chosen)
    )
    for key, heading, kind in _RUN_METRICS:
        table.add_row(heading, *(format_metric(run[key], kind) for run in chosen))
    table.add_row("config hash", *(str(run["config_hash"]) for run in chosen))
    table.add_row("git commit", *(str(run["git_commit"] or "-")[:10] for run in chosen))
    table.add_row("data written", *(str(run["data_snapshot"] or "-")[:16] for run in chosen))
    console.print(table)
    for run in chosen:
        if run["report_path"]:
            console.print(f"{run['run_id']}: {run['report_path']}")


@backtest_app.command("walkforward")
def backtest_walkforward(
    config_file: Annotated[Path, typer.Argument(help="Backtest config (YAML).")],
    split: Annotated[
        str | None,
        typer.Option("--split", help="Split name or file (default: the config's split)."),
    ] = None,
    mode: Annotated[str, typer.Option("--mode", help="anchored | rolling.")] = "anchored",
    train_years: Annotated[int, typer.Option("--train-years")] = 5,
    test_months: Annotated[int, typer.Option("--test-months")] = 12,
    step_months: Annotated[int, typer.Option("--step-months")] = 12,
    grid: Annotated[
        Path | None,
        typer.Option("--grid", help="YAML mapping of parameter -> list of values to try."),
    ] = None,
    embargo_days: Annotated[
        int | None,
        typer.Option("--embargo-days", help="Gap before each test window (default: split's)."),
    ] = None,
    config: ConfigOption = None,
    data_dir: DataDirOption = None,
    reports_dir: ReportsDirOption = None,
) -> None:
    """Walk-forward: choose parameters on each train window, test them on the next one.

    Uses only the split's unsealed segments. Every backtest is logged to the registry
    under one walk-forward id; the report shows each fold's choice and the stitched
    out-of-sample curve.
    """
    from qr.research.walkforward import walk_forward, write_walkforward_report

    try:
        backtest_config = load_backtest_config(config_file)
        split_name = split or backtest_config.split
        if split_name is None:
            raise ConfigError("give --split, or set split in the config")
        values = None
        if grid is not None:
            raw = yaml.safe_load(grid.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ConfigError(f"{grid}: expected a mapping of parameter -> values")
            values = raw
        result = walk_forward(
            backtest_config, split_name, mode, train_years, test_months, step_months, values,
            embargo_days=embargo_days, data_dir=data_dir, universe_config=config,
        )  # fmt: skip
    except (ConfigError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    path = write_walkforward_report(result, reports_dir or DEFAULT_REPORTS_DIR)

    table = Table(title=f"{backtest_config.name}: walk-forward ({mode})")
    for column in ("fold", "train", "test", "chosen", "train Sharpe", "test Sharpe", "test CAGR"):
        table.add_column(
            column, justify="right" if "Sharpe" in column or "CAGR" in column else "left"
        )
    for fold in result.folds:
        table.add_row(
            str(fold.fold.index),
            f"{fold.fold.train_start} to {fold.fold.train_end}",
            f"{fold.fold.test_start} to {fold.fold.test_end}",
            ", ".join(f"{k}={v}" for k, v in fold.chosen.items()) or "(config params)",
            format_metric(fold.train_metrics.get("sharpe"), "num"),
            format_metric(fold.test.metrics.get("sharpe"), "num"),
            format_metric(fold.test.metrics.get("cagr"), "pct"),
        )
    console.print(table)
    steps = max(len(result.folds) - 1, 0)
    colour = "red" if result.unstable else "green"
    console.print(
        f"Chosen parameters changed in [{colour}]{result.changes} of {steps}[/{colour}] steps. "
        f"Stitched out of sample: Sharpe {format_metric(result.metrics.get('sharpe'), 'num')}, "
        f"CAGR {format_metric(result.metrics.get('cagr'), 'pct')}, max drawdown "
        f"{format_metric(result.metrics.get('max_drawdown'), 'pct')}."
    )
    console.print(f"Risk-free rate: {risk_free_label(backtest_config)}")
    console.print(f"Runs logged: {result.runs_logged} (walk-forward id {result.walkforward_id})")
    console.print(f"Report:  {path}")


holdout_app = typer.Typer(help="Sealed test results, one per study.", no_args_is_help=True)
app.add_typer(holdout_app, name="holdout")


@holdout_app.command("list")
def holdout_list(data_dir: DataDirOption = None) -> None:
    """Every sealed result, and every recorded override."""
    from qr.research import holdout

    results = holdout.read_results(data_dir)
    if results.empty:
        console.print("No sealed results yet.")
    else:
        table = Table(title="Sealed results")
        for column in ("study", "split / segment", "sealed at (UTC)", "run", "config hash",
                       "Sharpe", "CAGR"):  # fmt: skip
            table.add_column(column)
        for row in results.itertuples():
            metrics = json.loads(str(row.metrics_json))
            table.add_row(
                str(row.study_id),
                f"{row.split_name} / {row.segment}",
                _run_date(row.sealed_at),
                str(row.run_id),
                str(row.config_hash),
                format_metric(metrics.get("sharpe"), "num"),
                format_metric(metrics.get("cagr"), "pct"),
            )
        console.print(table)
    overrides = holdout.read_overrides(data_dir)
    if not overrides.empty:
        table = Table(title="[red]Holdout overrides[/red] (sealed segments run again)")
        for column in ("study", "split / segment", "when (UTC)", "run", "reason"):
            table.add_column(column, overflow="fold")
        for row in overrides.itertuples():
            table.add_row(str(row.study_id), f"{row.split_name} / {row.segment}",
                          _run_date(row.created_at), str(row.run_id), str(row.reason))  # fmt: skip
        console.print(table)


@holdout_app.command("import-pairs")
def holdout_import_pairs(
    path: Annotated[Path, typer.Argument(help="The pairs study's sealed results file.")] = Path(
        "reports/pairs/test_results.json"
    ),
    data_dir: DataDirOption = None,
) -> None:
    """Copy the pairs study's sealed test results into the holdout table, without re-running."""
    from qr.research.pairs import import_sealed_results

    if not path.is_file():
        raise _fail(f"sealed results file not found: {path}")
    try:
        imported = import_sealed_results(path, data_dir)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    if imported:
        console.print(f"Imported {len(imported)} sealed results: {', '.join(imported)}")
    else:
        console.print("Already imported; nothing changed.")


sweep_app = typer.Typer(help="Parameter sweeps, with every trial counted.", no_args_is_help=True)
app.add_typer(sweep_app, name="sweep")


@sweep_app.command("run")
def sweep_run(
    spec_file: Annotated[Path, typer.Argument(help="Sweep spec (YAML).")],
    workers: Annotated[
        int, typer.Option("--workers", help="Run trials in this many processes.")
    ] = 1,
    config: ConfigOption = None,
    data_dir: DataDirOption = None,
    reports_dir: ReportsDirOption = None,
) -> None:
    """Run every trial of a sweep on its segment and log each one to the registry."""
    from qr.research.sweeps import load_sweep, run_sweep, summarize_sweep, write_sweep_report

    try:
        sweep = load_sweep(spec_file)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    console.print(
        f"Sweep {sweep.spec.name}: {len(sweep.trials)} trials on {sweep.base.segment} of "
        f"{sweep.base.split} ({sweep.base.start} to {sweep.base.end}), study {sweep.spec.study_id}"
    )
    result = run_sweep(sweep, workers=workers, data_dir=data_dir, universe_config=config)
    summary = summarize_sweep(result.sweep_id, data_dir)
    path = write_sweep_report(summary, reports_dir or DEFAULT_REPORTS_DIR)
    _print_sweep(summary)
    console.print(f"Report:  {path}")


@sweep_app.command("show")
def sweep_show(
    sweep_id: Annotated[str, typer.Argument(help="Sweep id (a unique prefix is enough).")],
    data_dir: DataDirOption = None,
) -> None:
    """A sweep's trials, its heatmap (two-parameter grids), and the stability check."""
    from qr.research.sweeps import summarize_sweep

    try:
        summary = summarize_sweep(sweep_id, data_dir)
    except ValueError as exc:
        raise _fail(str(exc)) from exc
    _print_sweep(summary)


def _print_sweep(summary: SweepSummary) -> None:
    from qr.research.sweeps import STABILITY_SHARE, heatmap_frame

    sweep, trials = summary.sweep, summary.trials
    names = (
        list(summary.grid)
        if summary.grid
        else sorted({k for p in trials["params_json"] for k in json.loads(str(p))})
    )
    table = Table(title=f"Sweep {sweep['sweep_id']} ({sweep['name']})")
    for column in ("trial", *names, "status", "Sharpe", "CAGR", "max drawdown"):
        table.add_column(column, justify="left" if column == "status" else "right")
    for row in trials.itertuples(index=False):
        colour = "red" if row.status == "failed" else "green"
        table.add_row(
            str(row.trial_index), *(str(getattr(row, n)) for n in names),
            f"[{colour}]{row.status}[/{colour}]",
            format_metric(_maybe(row.sharpe), "num"), format_metric(_maybe(row.cagr), "pct"),
            format_metric(_maybe(row.max_drawdown), "pct"),
        )  # fmt: skip
    console.print(table)
    heat = heatmap_frame(summary)
    if heat is not None:
        grid = Table(title=f"Sharpe: {heat.index.name} (rows) x {heat.columns.name} (columns)")
        grid.add_column(str(heat.index.name))
        for column in heat.columns:
            grid.add_column(str(column), justify="right")
        for index, values in heat.iterrows():
            grid.add_row(str(index), *(format_metric(_maybe(v), "num") for v in values))
        console.print(grid)
    found = summary.stability
    if found is not None:
        verdict = "[red]lone spike[/red]" if found.spike else "[green]plateau[/green]"
        console.print(
            f"Stability: best trial {found.best_index} (Sharpe {found.best_sharpe:.2f}) vs the "
            f"median of {found.neighbours} neighbours {found.neighbour_median:.2f} "
            f"(needs {STABILITY_SHARE:.0%}): {verdict}"
        )
    console.print(
        f"[bold]Trials in study {sweep['study_id']}: {summary.n_trials}[/bold] "
        "(every sweep, failed trials, and single runs)"
    )
    from qr.research.sweeps import risk_free_of

    console.print(f"Risk-free rate: {risk_free_of(summary)}")
    found = summary.deflated
    if found is not None and summary.best_index is not None:
        console.print(
            f"Best trial {summary.best_index}: Sharpe "
            f"{format_metric(_maybe(trials.iloc[summary.best_index]['sharpe']), 'num')}, PSR "
            f"{format_metric(summary.psr, 'pct')}, [bold]DSR {format_metric(found.dsr, 'pct')}"
            f"[/bold] (luck alone reaches Sharpe {found.benchmark_annualized:.2f} in "
            f"{found.n_trials} trials)"
        )


def _maybe(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


if __name__ == "__main__":
    app()
