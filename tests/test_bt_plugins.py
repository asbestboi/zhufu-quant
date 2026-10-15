"""Strategies from other packages, found through the ``qr.strategies`` entry points."""

from __future__ import annotations

from collections.abc import Iterator
from importlib import metadata
from typing import Any

import pandas as pd
import pytest

from qr.backtest import strategies
from qr.backtest.strategies import STRATEGIES, build_strategy, register
from qr.backtest.strategy import WeightStrategy


class PluginWeights(WeightStrategy):
    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def target_weights(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()


calls: list[str] = []


def register_all() -> None:
    calls.append("register_all")
    register("from_plugin", vectorized=PluginWeights)


def broken() -> None:
    raise RuntimeError("missing dependency")


def entry(name: str, target: str) -> metadata.EntryPoint:
    return metadata.EntryPoint(name, f"{__name__}:{target}", strategies.ENTRY_POINT_GROUP)


@pytest.fixture
def plugins(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[metadata.EntryPoint]]:
    """Installed entry points, as a list the test fills in."""
    installed: list[metadata.EntryPoint] = []

    def entry_points(**selection: Any) -> list[metadata.EntryPoint]:
        assert selection == {"group": strategies.ENTRY_POINT_GROUP}
        return installed

    monkeypatch.setattr(strategies.metadata, "entry_points", entry_points)
    monkeypatch.setattr(strategies, "_plugins_loaded", False)
    calls.clear()
    yield installed
    STRATEGIES.pop("from_plugin", None)


def test_unknown_name_loads_plugins_once(plugins: list[metadata.EntryPoint]) -> None:
    plugins.append(entry("mine", "register_all"))
    built = build_strategy("from_plugin", "vectorized", {"weight": 0.5})
    assert isinstance(built, PluginWeights) and built.weight == 0.5
    build_strategy("from_plugin", "vectorized", {})
    with pytest.raises(ValueError, match="unknown strategy 'nope'"):
        build_strategy("nope", "vectorized", {})
    assert calls == ["register_all"]


def test_built_in_names_do_not_load_plugins(plugins: list[metadata.EntryPoint]) -> None:
    plugins.append(entry("mine", "register_all"))
    build_strategy("buy_and_hold", "vectorized", {})
    assert calls == []


def test_a_broken_plugin_is_a_clear_error(plugins: list[metadata.EntryPoint]) -> None:
    plugins.append(entry("bad", "broken"))
    with pytest.raises(ValueError, match=r"plugin 'bad'.*failed to load: missing dependency"):
        build_strategy("from_plugin", "vectorized", {})
