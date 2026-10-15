"""Provenance: which data and which environment a backtest result came from.

Every logged run records

* the **data fingerprint** (``qr.load.fingerprint``) of everything it read: the
  universe's symbols plus the benchmark, full history up to the config's ``end`` (warm-up
  bars before ``start`` are data the strategy used), with total-return adjustment, from
  the store as it was when the run started (``as_of`` is None);
* the **environment**: Python version, the SHA-256 of ``uv.lock`` (which pins every
  package version), the engine core that actually ran, the Rust module's version when it
  ran, the operating system, and the git commit with its dirty flag.

``qr backtest verify`` recomputes the fingerprint with the same rule and the run's
``as_of`` and says whether the data under the result is still the same.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from qr.adjust import Adjustment
from qr.backtest.config import BacktestConfig
from qr.backtest.rust_core import CORE_ENV_VAR, rust_available
from qr.fingerprint import DataFingerprint
from qr.load import fingerprint, list_symbols
from qr.snapshots import file_sha256

# The fingerprint rule for runs. Changing it would make every stored hash unverifiable.
ADJUSTMENT: Adjustment = "total_return"
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/qr/backtest/ -> the repository


@dataclass(frozen=True)
class Environment:
    """What a run ran on, besides the data."""

    python_version: str
    uv_lock_sha256: str | None
    engine_core: str | None  # None for the vectorized engine, which has no core
    rust_core_version: str | None
    os: str


def data_symbols(config: BacktestConfig, universe: list[str]) -> list[str]:
    """Every symbol a run of ``config`` over ``universe`` reads, benchmark included."""
    return list(dict.fromkeys([*universe, *([config.benchmark] if config.benchmark else [])]))


def run_fingerprint(
    symbols: list[str],
    end: Any,
    as_of: datetime | None = None,
    *,
    data_dir: Path | str | None = None,
) -> DataFingerprint:
    """The fingerprint of a run's data, by the rule in this module's docstring."""
    return fingerprint(symbols, None, end, ADJUSTMENT, as_of, data_dir=data_dir)


def uv_lock_sha256(root: Path | None = None) -> str | None:
    """SHA-256 of ``uv.lock`` in the project (or ``root``); None when there is none."""
    path = (root or PROJECT_ROOT) / "uv.lock"
    return file_sha256(path) if path.is_file() else None


def core_used(config: BacktestConfig) -> str | None:
    """The event-engine core a run of ``config`` uses, after the fallback rules.

    Mirrors ``rust_core.select_core`` without logging: the Rust core when it is wanted
    (``$QR_ENGINE_CORE`` or the config) and installed, else Python.
    """
    if config.engine != "event":
        return None
    wanted = os.environ.get(CORE_ENV_VAR) or config.engine_core
    return "rust" if wanted == "rust" and rust_available() else "python"


def rust_core_version() -> str | None:
    """The installed ``qr_core`` package version, if it is installed."""
    try:
        return metadata.version("qr_core")
    except metadata.PackageNotFoundError:
        return None


def environment(config: BacktestConfig) -> Environment:
    """The environment of a run of ``config`` started now."""
    core = core_used(config)
    return Environment(
        python_version=platform.python_version(),
        uv_lock_sha256=uv_lock_sha256(),
        engine_core=core,
        rust_core_version=rust_core_version() if core == "rust" else None,
        os=platform.platform(),
    )


# --------------------------------------------------------------------------- verify


@dataclass(frozen=True)
class Verification:
    """The outcome of re-fingerprinting a run's data.

    ``changed`` lists symbols whose data differs (or that can no longer be read);
    ``missing`` is the subset no longer in the store.
    """

    recorded_hash: str
    current_hash: str | None
    changed: list[str]
    missing: list[str]
    current: DataFingerprint | None

    @property
    def matches(self) -> bool:
        """Whether the data is exactly what the run used."""
        return self.current_hash == self.recorded_hash and not self.changed


def per_symbol_hashes(value: Any) -> dict[str, str]:
    """The per-symbol hashes stored with a run (``{}`` when none were recorded)."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return {}
    parsed = json.loads(str(value))
    return {str(symbol): str(digest) for symbol, digest in parsed.items()}


def verify_run(run: dict[str, Any], data_dir: Path | str | None = None) -> Verification:
    """Recompute the fingerprint of a logged run's symbols and dates, with its ``as_of``.

    ``run`` is a registry row. Raises ``ValueError`` when the run has no recorded hash.
    """
    recorded = run.get("data_hash")
    if recorded is None or pd.isna(recorded):
        raise ValueError(
            f"run {run['run_id']} has no data hash (it was logged before hashes were "
            "recorded, or failed before loading data), so there is nothing to verify"
        )
    stored = per_symbol_hashes(run.get("data_per_symbol_json"))
    as_of = run.get("data_as_of")
    moment = None if as_of is None or pd.isna(as_of) else pd.Timestamp(as_of).to_pydatetime()
    end = pd.Timestamp(run["end_date"]).date()

    symbols = sorted(stored)
    missing: list[str] = []
    current: DataFingerprint | None
    try:
        current = run_fingerprint(symbols, end, moment, data_dir=data_dir)
    except KeyError:
        known = set(list_symbols(data_dir=data_dir))
        missing = [s for s in symbols if s not in known]
        present = [s for s in symbols if s in known]
        current = run_fingerprint(present, end, moment, data_dir=data_dir) if present else None
    now = current.per_symbol if current is not None else {}
    changed = sorted(s for s in symbols if now.get(s) != stored[s])
    return Verification(
        recorded_hash=str(recorded),
        current_hash=None if missing or current is None else current.hash,
        changed=changed,
        missing=missing,
        current=current,
    )
