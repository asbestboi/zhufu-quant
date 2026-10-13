"""Load and validate ``config/universe.yaml`` and resolve local data paths."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_PATH = Path("config/universe.yaml")
DEFAULT_DATA_DIR = Path("data")

SourceName = Literal["yahoo", "crypto"]
AssetClass = Literal["etf", "equity", "crypto"]


class ConfigError(Exception):
    """Raised when the universe config is missing or invalid."""


class _Model(BaseModel):
    # Unknown keys are almost always typos; refuse them instead of ignoring them.
    model_config = ConfigDict(extra="forbid", frozen=True)


class YahooSettings(_Model):
    request_delay_seconds: float = Field(default=1.0, ge=0)
    max_retries: int = Field(default=5, ge=0)


class CryptoSettings(_Model):
    enabled: bool = False
    exchange: str = "kraken"


class SourcesConfig(_Model):
    yahoo: YahooSettings = YahooSettings()
    crypto: CryptoSettings = CryptoSettings()


class ValidationSettings(_Model):
    outlier_return_threshold: float = Field(default=0.25, gt=0)
    stale_after_trading_days: int = Field(default=5, ge=1)
    max_zero_volume_streak: int = Field(default=3, ge=0)
    revision_tolerance: float = Field(default=1e-4, ge=0)


class Instrument(_Model):
    symbol: str = Field(min_length=1)
    source: SourceName
    asset_class: AssetClass

    @model_validator(mode="after")
    def _source_matches_asset_class(self) -> Instrument:
        if (self.source == "crypto") != (self.asset_class == "crypto"):
            raise ValueError(
                f"{self.symbol}: source '{self.source}' cannot serve asset_class "
                f"'{self.asset_class}'"
            )
        return self


class UniverseConfig(_Model):
    start_date: date
    timezone: str = "UTC"
    sources: SourcesConfig = SourcesConfig()
    validation: ValidationSettings = ValidationSettings()
    universe: list[Instrument] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> UniverseConfig:
        symbols = [i.symbol for i in self.universe]
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ValueError(f"duplicate symbols in universe: {', '.join(duplicates)}")
        if self.start_date > date.today():
            raise ValueError(f"start_date {self.start_date} is in the future")
        return self

    def source_enabled(self, source: str) -> bool:
        """Whether instruments of ``source`` should be ingested."""
        return self.sources.crypto.enabled if source == "crypto" else True

    def instrument(self, symbol: str) -> Instrument | None:
        """Return the configured instrument for ``symbol``, if any."""
        return next((i for i in self.universe if i.symbol == symbol), None)


def config_path(path: Path | str | None = None) -> Path:
    """Resolve the config file: explicit argument, then ``QR_CONFIG``, then the default."""
    return Path(path or os.environ.get("QR_CONFIG") or DEFAULT_CONFIG_PATH)


def data_dir(path: Path | str | None = None) -> Path:
    """Resolve the data directory: explicit argument, then ``QR_DATA_DIR``, then ``./data``."""
    return Path(path or os.environ.get("QR_DATA_DIR") or DEFAULT_DATA_DIR)


def db_path(data_directory: Path | str | None = None) -> Path:
    """Path of the DuckDB file inside the data directory."""
    return data_dir(data_directory) / "market.duckdb"


def raw_dir(data_directory: Path | str | None = None) -> Path:
    """Directory holding the immutable raw Parquet snapshots."""
    return data_dir(data_directory) / "raw"


def load_config(path: Path | str | None = None) -> UniverseConfig:
    """Read and validate the universe config, raising :class:`ConfigError` on any problem."""
    resolved = config_path(path)
    if not resolved.is_file():
        raise ConfigError(f"config file not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved}: expected a mapping at the top level")
    try:
        return UniverseConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{resolved}: invalid config\n{problems}") from exc
