"""The paper-trading config: one YAML file per live strategy, validated with pydantic.

The scheduling/approval fields of the spec's Part 4 are added when that part is built,
not speculatively here. No strategy-specific mapping or UCITS translation belongs in
this public package at all; this config only names a registered ``qr`` strategy and the
symbols it decides on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qr.config import ConfigError

BrokerName = Literal["sim", "ibkr"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IBKRConfig(_Model):
    """Connection settings for ``qr.live.ibkr`` (IB Gateway or TWS, paper port only).

    No credentials here, ever -- those stay with IB Gateway itself (see the README).
    """

    host: str = "127.0.0.1"
    port: int = Field(default=4002, gt=0, le=65535)  # 4002: IB Gateway paper; 7497: TWS paper
    client_id: int = Field(default=1, ge=0)
    # How far through the bid/ask mid a day limit order is priced, to help it fill
    # without chasing the market: 0.3% through the mid, in the order's direction.
    limit_buffer: float = Field(default=0.003, ge=0, lt=1)


class LiveConfig(_Model):
    """Everything that defines one paper-trading strategy's monthly cycle."""

    name: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    universe: list[str] = Field(min_length=1)
    rebalance: Literal["monthly"] = "monthly"
    broker: BrokerName = "sim"
    ibkr: IBKRConfig = Field(default_factory=IBKRConfig)
    initial_cash: float = Field(default=100_000.0, gt=0)  # sim broker's starting cash only
    # The account's *managed* capital, in its base currency: qr.live plans, checks, and
    # reports against this number, never the account's total equity. A paper account is
    # usually funded with far more than this; capital_limit is what keeps it from ever
    # being asked to invest more than the slice it was actually meant to manage.
    capital_limit: float = Field(default=10_000.0, gt=0)
    cash_buffer: float = Field(default=0.01, ge=0, lt=1)
    min_trade_weight: float = Field(default=0.02, ge=0, lt=1)
    max_single_order_value: float = Field(default=0.5, gt=0)
    max_total_turnover: float = Field(default=1.2, gt=0)
    max_gross_leverage: float = Field(default=1.0, gt=0)
    max_price_age_days: float = Field(default=4.0, gt=0)
    max_price_jump: float = Field(default=0.20, gt=0)
    # How long an order is allowed to work before it's cancelled outright, never chased
    # (qr.live.cycle's order-lifecycle poller). Anchored to the decision's first order
    # submission, so it survives a restart rather than restarting on every resume.
    order_window_minutes: float = Field(default=30.0, gt=0)
    order_poll_interval_seconds: float = Field(default=5.0, gt=0)
    # How long a single qr.live.cycle.run_cycle call will actively poll before returning
    # control (so a CLI invocation never blocks for the full order_window_minutes); a
    # later call resumes polling the same orders, never resubmitting them.
    order_poll_timeout_seconds: float = Field(default=60.0, gt=0)
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
