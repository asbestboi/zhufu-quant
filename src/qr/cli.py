"""The ``qr`` command line interface."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from qr import crosscheck as crosscheck_module
from qr import storage
from qr.adjust import Adjustment
from qr.backtest.config import load_backtest_config
from qr.backtest.registry import read_runs
from qr.backtest.report import METRIC_ROWS, format_metric
from qr.backtest.runner import execute
from qr.config import ConfigError, UniverseConfig, db_path, load_config, load_env_file
from qr.ingest import FAILED, PARTIAL, run_ingest
from qr.load import get_bars
from qr.sources.base import SourceError
from qr.storage import StorageError
from qr.validate import summarize, validate_symbols

app = typer.Typer(
    help="Personal quant research: market data pipeline and backtesting.", no_args_is_help=True
)
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


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show debug logs.")] = False,
) -> None:
    """Set up logging and read ``.env`` for every command."""
    load_env_file()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party chatter is only useful when debugging.
    for noisy in ("yfinance", "peewee", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if verbose else logging.CRITICAL)


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
) -> None:
    """Run a backtest, write its HTML tearsheet, and log it to the registry."""
    try:
        backtest_config = load_backtest_config(config_file)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    outcome = execute(
        backtest_config,
        data_dir=data_dir,
        reports_dir=reports_dir,
        universe_config=config,
        notes=notes,
    )
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
    console.print(f"Run id:  {outcome.record.run_id}")
    console.print(f"Report:  {outcome.record.report_path}")


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


@backtest_app.command("compare")
def backtest_compare(
    run_ids: Annotated[list[str], typer.Argument(help="Run ids (unique prefixes are enough).")],
    data_dir: DataDirOption = None,
) -> None:
    """Show the logged metrics of several runs side by side."""
    runs = read_runs(data_dir, run_ids=run_ids)
    chosen: list[dict[str, Any]] = []
    for wanted in run_ids:
        matches = runs[runs["run_id"].str.startswith(wanted)]
        if len(matches) != 1:
            problem = "no run" if matches.empty else f"{len(matches)} runs"
            raise _fail(f"{problem} with an id starting with '{wanted}' (see `qr backtest list`)")
        chosen.append({str(k): v for k, v in matches.iloc[0].to_dict().items()})
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


if __name__ == "__main__":
    app()
