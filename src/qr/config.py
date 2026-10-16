"""Load and validate ``config/universe.yaml`` and resolve local data paths."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

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


class IngestSettings(_Model):
    # A source is declared down after this many symbols in a row fail to download.
    max_consecutive_failures: int = Field(default=3, ge=1)


class CrosscheckSettings(_Model):
    close_tolerance: float = Field(default=0.005, gt=0)  # relative difference on the close
    request_delay_seconds: float = Field(default=0.5, ge=0)
    max_retries: int = Field(default=2, ge=0)


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
    """The validated ``config/universe.yaml``.

    A group is either a list of symbols or ``{start_date: ..., symbols: [...]}``. The
    second form gives the group's symbols their own history start (see
    :meth:`start_date_for`); ``group_start_dates`` is filled from it and cannot be set
    directly.
    """

    start_date: date
    timezone: str = "UTC"
    sources: SourcesConfig = SourcesConfig()
    validation: ValidationSettings = ValidationSettings()
    ingest: IngestSettings = IngestSettings()
    crosscheck: CrosscheckSettings = CrosscheckSettings()
    universe: list[Instrument] = Field(min_length=1)
    groups: dict[str, list[str]] = Field(default_factory=dict)
    group_start_dates: dict[str, date] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _split_group_start_dates(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        if "group_start_dates" in raw:
            raise ValueError("group_start_dates is derived; set start_date inside a group")
        groups = raw.get("groups")
        if not isinstance(groups, dict):
            return raw
        members: dict[str, Any] = {}
        starts: dict[str, Any] = {}
        for name, value in groups.items():
            if isinstance(value, dict):
                unknown = sorted(set(value) - {"symbols", "start_date"})
                if unknown:
                    raise ValueError(f"group '{name}': unknown keys {', '.join(unknown)}")
                if "symbols" not in value:
                    raise ValueError(f"group '{name}': missing 'symbols'")
                members[name] = value["symbols"]
                if value.get("start_date") is not None:
                    starts[name] = value["start_date"]
            else:
                members[name] = value
        return {**raw, "groups": members, "group_start_dates": starts}

    @model_validator(mode="after")
    def _check(self) -> UniverseConfig:
        symbols = [i.symbol for i in self.universe]
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ValueError(f"duplicate symbols in universe: {', '.join(duplicates)}")
        if self.start_date > date.today():
            raise ValueError(f"start_date {self.start_date} is in the future")
        for name, members in self.groups.items():
            undefined = [s for s in members if s not in symbols]
            if undefined:
                raise ValueError(
                    f"group '{name}' lists symbols that are not defined in universe: "
                    f"{', '.join(undefined)}"
                )
            repeated = sorted({s for s in members if members.count(s) > 1})
            if repeated:
                raise ValueError(f"group '{name}' repeats symbols: {', '.join(repeated)}")
        for name, start in self.group_start_dates.items():
            if start > date.today():
                raise ValueError(f"group '{name}': start_date {start} is in the future")
        return self

    def start_date_for(self, symbol: str) -> date:
        """The date ``symbol``'s history should start from.

        The earliest ``start_date`` of the groups that list the symbol and set one, so a
        symbol shared by several studies gets the longest history any of them needs.
        Symbols in no such group use the global ``start_date``.
        """
        starts = [
            start for name, start in self.group_start_dates.items() if symbol in self.groups[name]
        ]
        return min(starts) if starts else self.start_date

    def source_enabled(self, source: str) -> bool:
        """Whether instruments of ``source`` should be ingested."""
        return self.sources.crypto.enabled if source == "crypto" else True

    def instrument(self, symbol: str) -> Instrument | None:
        """Return the configured instrument for ``symbol``, if any."""
        return next((i for i in self.universe if i.symbol == symbol), None)

    def group(self, name: str) -> list[str]:
        """Symbols of the named group, in the order the config lists them."""
        if name not in self.groups:
            known = ", ".join(sorted(self.groups)) or "none defined"
            raise ConfigError(f"unknown universe '{name}' (available: {known})")
        return list(self.groups[name])


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


def universe_symbols(name: str, path: Path | str | None = None) -> list[str]:
    """Symbols of the named universe (a ``groups:`` entry in the config file)."""
    return load_config(path).group(name)


def load_env_file(path: Path | str = ".env") -> None:
    """Copy ``KEY=VALUE`` lines from a ``.env`` file into the process environment.

    Variables that are already set win, so the real environment overrides the file.
    Blank lines and ``#`` comments are ignored; a missing file is fine.
    """
    env_file = Path(path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
