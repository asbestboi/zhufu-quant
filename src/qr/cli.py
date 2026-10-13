"""The ``qr`` command line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from qr import storage
from qr.adjust import Adjustment
from qr.config import ConfigError, UniverseConfig, db_path, load_config, load_env_file
from qr.ingest import FAILED, PARTIAL, run_ingest
from qr.load import get_bars
from qr.storage import StorageError
from qr.validate import summarize, validate_symbols

app = typer.Typer(help="Personal quant research: market data pipeline.", no_args_is_help=True)
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


if __name__ == "__main__":
    app()
