from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from qr.config import ConfigError, load_config

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
