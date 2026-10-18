"""The paper-trading config: one YAML file per live strategy, validated with pydantic.

Deliberately small for Part 1: a broker type beyond ``sim`` (IBKR, paper only) and the
scheduling/approval fields of the spec's Part 4 are added when those parts are built, not
speculatively here. No strategy-specific mapping or UCITS translation belongs in this
public package at all; this config only names a registered ``qr`` strategy and the
symbols it decides on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qr.config import ConfigError

BrokerName = Literal["sim"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LiveConfig(_Model):
    """Everything that defines one paper-trading strategy's monthly cycle."""

    name: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    universe: list[str] = Field(min_length=1)
    rebalance: Literal["monthly"] = "monthly"
    broker: BrokerName = "sim"
    initial_cash: float = Field(default=100_000.0, gt=0)
    cash_buffer: float = Field(default=0.01, ge=0, lt=1)
    min_trade_weight: float = Field(default=0.02, ge=0, lt=1)
    max_single_order_value: float = Field(default=0.5, gt=0)
    max_total_turnover: float = Field(default=1.2, gt=0)
    max_gross_leverage: float = Field(default=1.0, gt=0)
    max_price_age_days: float = Field(default=4.0, gt=0)
    max_price_jump: float = Field(default=0.20, gt=0)
    # Where `KILL` is looked for (qr.live.safety), and where this config's own relative
    # paths (none yet) would resolve. Defaults to the current directory.
    working_dir: Path = Path()

    @model_validator(mode="after")
    def _check(self) -> LiveConfig:
        symbols = self.universe
        duplicates = sorted({s for s in symbols if symbols.count(s) > 1})
        if duplicates:
            raise ValueError(f"universe repeats symbols: {', '.join(duplicates)}")
        return self


def load_live_config(path: Path | str) -> LiveConfig:
    """Read and validate a live-trading YAML file, raising :class:`ConfigError`."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"live config not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved}: expected a mapping at the top level")
    try:
        return LiveConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{resolved}: invalid live config\n{problems}") from exc
