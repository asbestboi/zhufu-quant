"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from qr import storage
from qr.backtest.rust_core import CORE_ENV_VAR, REQUIRE_ENV_VAR, rust_available
from qr.config import UniverseConfig
from qr.frames import empty_actions, normalize_actions, normalize_bars
from qr.sources.base import DataSource, SourceError

FIXTURES = Path(__file__).parent / "fixtures"

# A moment after the close of 2024-01-16, the last session in bars_clean.csv.
AFTER_JAN_16 = datetime(2024, 1, 17, 0, 0, 0)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-network", action="store_true", help="run tests that hit providers")
    parser.addoption("--run-ibkr", action="store_true", help="run tests against a real IB Gateway")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skips = {
        "network": pytest.mark.skip(reason="needs --run-network"),
        "ibkr": pytest.mark.skip(reason="needs --run-ibkr"),
    }
    for keyword, skip in skips.items():
        if config.getoption(f"--run-{keyword}"):
            continue
        for item in items:
            if keyword in item.keywords:
                item.add_marker(skip)


@pytest.fixture(params=["python", "rust"])
def engine_core(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run a test once per event-engine core, by setting ``QR_ENGINE_CORE``.

    Modules opt in with ``pytestmark = pytest.mark.usefixtures("engine_core")``. Without
    the compiled Rust module the Rust run is skipped, unless ``QR_REQUIRE_RUST_CORE=1``
    (set in CI), which makes a missing module a failure.
    """
    core: str = request.param
    if core == "rust" and not rust_available():
        if os.environ.get(REQUIRE_ENV_VAR) == "1":
            pytest.fail("QR_REQUIRE_RUST_CORE=1 but the qr_core module is not installed")
        pytest.skip("the Rust core (qr_core) is not built")
    monkeypatch.setenv(CORE_ENV_VAR, core)
    return core


def load_bars(name: str) -> pd.DataFrame:
    return normalize_bars(pd.read_csv(FIXTURES / name))


def load_actions(name: str) -> pd.DataFrame:
    return normalize_actions(pd.read_csv(FIXTURES / name))


class FakeSource(DataSource):
    """An offline DataSource serving frames held in memory, and recording every request."""

    name = "yahoo"

    def __init__(
        self,
        bars: dict[str, pd.DataFrame] | None = None,
        actions: dict[str, pd.DataFrame] | None = None,
        name: str = "yahoo",
    ) -> None:
        self.name = name
        self.bars = bars or {}
        self.actions = actions or {}
        self.calls: list[tuple[str, date, date]] = []
        self.probes = 0
        self.dies_after: int | None = None  # unreachable once this many fetches were made

    @property
    def dead(self) -> bool:
        return self.dies_after is not None and len(self.calls) >= self.dies_after

    def probe(self) -> None:
        self.probes += 1
        if self.dead:
            raise SourceError("connection refused")

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if self.dead:
            self.calls.append((symbol, start, end))
            raise SourceError("connection refused")
        self.calls.append((symbol, start, end))
        if symbol not in self.bars:
            raise SourceError(f"no data for {symbol}")
        frame = self.bars[symbol]
        in_range = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
        frame = frame[in_range].reset_index(drop=True)
        if frame.empty:
            raise SourceError(f"empty response for {symbol}")
        return frame

    def fetch_corporate_actions(self, symbol: str) -> pd.DataFrame:
        return self.actions.get(symbol, empty_actions())


class Clock:
    """A settable stand-in for ``storage.utcnow``."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clean_bars() -> pd.DataFrame:
    return load_bars("bars_clean.csv")


@pytest.fixture
def dirty_bars() -> pd.DataFrame:
    return load_bars("bars_dirty.csv")


@pytest.fixture
def split_dividend_bars() -> pd.DataFrame:
    return load_bars("bars_split_dividend.csv")


@pytest.fixture
def split_dividend_actions() -> pd.DataFrame:
    return load_actions("actions_split_dividend.csv")


@pytest.fixture
def make_config() -> Callable[..., UniverseConfig]:
    """Build a config for the given yahoo symbols (plus any overrides)."""

    def build(*symbols: str, **overrides: object) -> UniverseConfig:
        raw: dict[str, object] = {
            "start_date": "2024-01-01",
            "sources": {"yahoo": {"request_delay_seconds": 0, "max_retries": 0}},
            "universe": [
                {"symbol": s, "source": "yahoo", "asset_class": "equity"} for s in symbols
            ],
        }
        raw.update(overrides)
        return UniverseConfig.model_validate(raw)

    return build


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def con(tmp_path: Path) -> Iterator[storage.Connection]:
    with storage.connect(tmp_path / "test.duckdb") as connection:
        storage.init_schema(connection)
        yield connection
