from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from qr.config import ConfigError, load_config, load_env_file, universe_symbols

REPO_CONFIG = Path(__file__).parents[1] / "config" / "universe.yaml"

VALID = """
start_date: 2015-06-01
universe:
  - { symbol: SPY, source: yahoo, asset_class: etf }
  - { symbol: BTC/USD, source: crypto, asset_class: crypto }
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_default_config_in_repo_is_valid() -> None:
    config = load_config(REPO_CONFIG)
    assert config.start_date == date(2010, 1, 1)
    assert [i.symbol for i in config.universe][:2] == ["SPY", "QQQ"]
    assert config.sources.yahoo.max_retries == 5
    assert not config.source_enabled("crypto")
    assert config.source_enabled("yahoo")


def test_defaults_are_filled_in(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, VALID))
    assert config.timezone == "UTC"
    assert config.sources.yahoo.request_delay_seconds == 1.0
    assert config.validation.outlier_return_threshold == 0.25
    assert config.instrument("SPY") is not None
    assert config.instrument("NOPE") is None


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("start_date: 2015-01-01\nuniverse: []\n", "universe"),
        (VALID + "  - { symbol: SPY, source: yahoo, asset_class: etf }\n", "duplicate symbols"),
        (VALID.replace("source: yahoo", "source: bloomberg"), "universe.0.source"),
        (VALID.replace("asset_class: etf", "asset_class: crypto"), "cannot serve"),
        (VALID + "typo_key: 1\n", "typo_key"),
        (VALID.replace("2015-06-01", "not-a-date"), "start_date"),
        (VALID.replace("2015-06-01", "2999-01-01"), "in the future"),
        (VALID + "sources: { yahoo: { max_retries: -1 } }\n", "sources.yahoo.max_retries"),
        ("- just\n- a list\n", "expected a mapping"),
        ("start_date: [unclosed\n", "invalid YAML"),
    ],
)
def test_invalid_config_fails_with_clear_error(tmp_path: Path, text: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_config(write(tmp_path, text))


GROUPS = VALID + "groups:\n  macro: [SPY]\n  all: [BTC/USD, SPY]\n"


def test_groups(tmp_path: Path) -> None:
    path = write(tmp_path, GROUPS)
    config = load_config(path)
    assert config.group("all") == ["BTC/USD", "SPY"]  # config order is kept
    assert universe_symbols("macro", path) == ["SPY"]
    with pytest.raises(ConfigError, match=r"unknown universe 'nope' \(available: all, macro\)"):
        config.group("nope")
    assert load_config(write(tmp_path, VALID)).groups == {}


def test_group_start_dates(tmp_path: Path) -> None:
    text = VALID + (
        "groups:\n"
        "  macro: [SPY]\n"
        "  long: { start_date: 2003-01-01, symbols: [SPY] }\n"
        "  longer: { start_date: 2001-01-01, symbols: [SPY] }\n"
        "  late: { start_date: 2020-01-01, symbols: [BTC/USD] }\n"
    )
    config = load_config(write(tmp_path, text))
    assert config.group("long") == ["SPY"]
    assert config.groups["macro"] == ["SPY"]  # list-form groups are unchanged
    assert config.start_date_for("SPY") == date(2001, 1, 1)  # the earliest group wins
    assert config.start_date_for("BTC/USD") == date(2020, 1, 1)
    assert load_config(write(tmp_path, VALID)).start_date_for("SPY") == date(2015, 6, 1)


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("{ start_date: 2003-01-01 }", "missing 'symbols'"),
        ("{ symbols: [SPY], start: 2003-01-01 }", "unknown keys start"),
        ("{ start_date: 2999-01-01, symbols: [SPY] }", "in the future"),
        ("{ start_date: soon, symbols: [SPY] }", "group_start_dates"),
    ],
)
def test_invalid_group_start_dates(tmp_path: Path, group: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_config(write(tmp_path, VALID + f"groups:\n  long: {group}\n"))


def test_group_start_dates_cannot_be_set_directly(tmp_path: Path) -> None:
    text = VALID + "group_start_dates: { core: 2003-01-01 }\n"
    with pytest.raises(ConfigError, match="derived"):
        load_config(write(tmp_path, text))


def test_default_config_groups() -> None:
    assert universe_symbols("core", REPO_CONFIG)[0] == "SPY"
    pairs = ["KO", "PEP", "XOM", "CVX", "V", "MA", "HD", "LOW", "GS", "MS"]
    assert universe_symbols("pairs", REPO_CONFIG) == pairs
    etf = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "TIP", "GLD", "DBC", "SHY"]
    assert universe_symbols("etf", REPO_CONFIG) == etf
    config = load_config(REPO_CONFIG)
    assert config.start_date_for("SHY") == date(2003, 1, 1)
    assert config.start_date_for("KO") == config.start_date


@pytest.mark.parametrize(
    ("groups", "expected"),
    [
        ("groups: { pairs: [SPY, KO, PEP] }", "group 'pairs' lists symbols.*universe: KO, PEP"),
        ("groups: { pairs: [SPY, SPY] }", "group 'pairs' repeats symbols: SPY"),
        ("groups: [SPY]", "groups"),
    ],
)
def test_invalid_groups_fail_with_clear_error(tmp_path: Path, groups: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_config(write(tmp_path, VALID + groups + "\n"))


def test_load_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    lines = ["# comment", "", "QR_TEST_A=one", "export QR_TEST_B = 'two words'", "QR_TEST_C=file"]
    env.write_text("\n".join([*lines, "nonsense"]), encoding="utf-8")
    monkeypatch.setenv("QR_TEST_A", "placeholder")  # so monkeypatch restores all three
    monkeypatch.setenv("QR_TEST_B", "placeholder")
    monkeypatch.delenv("QR_TEST_A")
    monkeypatch.delenv("QR_TEST_B")
    monkeypatch.setenv("QR_TEST_C", "environment")

    load_env_file(env)
    assert os.environ["QR_TEST_A"] == "one"
    assert os.environ["QR_TEST_B"] == "two words"
    assert os.environ["QR_TEST_C"] == "environment"  # the real environment wins
    load_env_file(tmp_path / "absent.env")  # a missing file is fine
