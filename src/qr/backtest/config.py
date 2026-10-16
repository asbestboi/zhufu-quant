"""Backtest configs: one YAML file per experiment, validated with pydantic."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qr.config import ConfigError

EngineName = Literal["vectorized", "event"]
EngineCoreName = Literal["python", "rust"]
Execution = Literal["next_open", "next_close"]
RebalanceFrequency = Literal["daily", "weekly", "monthly"]
CommissionModel = Literal["ibkr_tiered", "ibkr_fixed", "per_share", "percent", "none"]
ParticipationAction = Literal["partial", "reject"]

DEFAULT_COST_STRESS = (0.0, 1.0, 2.0)

# Options added after configs were first hashed, with the value that leaves a run as it
# was. They are left out of the hash at that value, so older hashes still match.
_HASH_NEUTRAL: dict[str, Any] = {
    "cost_multiplier": 1.0,
    "split": None,
    "segment": None,
    "study_id": None,
    "evaluate_from": None,
    "open_end": False,
    "risk_free": None,
}
# Options that change what is reported about a run, not the run itself.
_NOT_HASHED = {"engine_core", "cost_stress"}


class _Model(BaseModel):
    # Unknown keys are almost always typos; refuse them instead of ignoring them.
    model_config = ConfigDict(extra="forbid", frozen=True)


class CostOverride(_Model):
    """Cost settings for one asset class or symbol. Anything left out is inherited."""

    commission_model: CommissionModel | None = None
    commission_per_share: float | None = Field(default=None, ge=0)
    commission_min_per_order: float | None = Field(default=None, ge=0)
    commission_max_pct: float | None = Field(default=None, gt=0)
    commission_pct: float | None = Field(default=None, ge=0)
    spread_bps: float | None = Field(default=None, ge=0)
    slippage_bps: float | None = Field(default=None, ge=0)
    impact_bps: float | None = Field(default=None, ge=0)
    impact_exponent: float | None = Field(default=None, gt=0)
    borrow_rate: float | None = Field(default=None, ge=0)
    max_participation: float | None = Field(default=None, gt=0)
    participation_action: ParticipationAction | None = None


class CostConfig(_Model):
    """Trading costs. See ``costs.py`` for how each number is used.

    ``commission_model`` picks a preset; the four ``commission_*`` numbers override the
    preset's values one by one. ``spread_bps`` is the *half*-spread, the cost of one
    trade. ``impact_bps`` is the extra slippage of an order as large as the average daily
    dollar volume, scaled by ``(order / volume) ** impact_exponent`` for other sizes.
    ``borrow_rate`` is an annual rate on short market value. ``max_participation`` caps
    an order at that fraction of average daily volume; the excess is cut off
    (``partial``) or the whole order refused (``reject``).
    """

    commission_model: CommissionModel = "ibkr_tiered"
    commission_per_share: float | None = Field(default=None, ge=0)
    commission_min_per_order: float | None = Field(default=None, ge=0)
    commission_max_pct: float | None = Field(default=None, gt=0)
    commission_pct: float | None = Field(default=None, ge=0)
    spread_bps: float = Field(default=1.0, ge=0)
    slippage_bps: float = Field(default=2.0, ge=0)
    impact_bps: float = Field(default=0.0, ge=0)
    impact_exponent: float = Field(default=1.0, gt=0)
    borrow_rate: float = Field(default=0.01, ge=0)
    max_participation: float | None = Field(default=None, gt=0)
    participation_action: ParticipationAction = "partial"
    adv_window: int = Field(default=20, ge=1)
    asset_classes: dict[str, CostOverride] = Field(default_factory=dict)
    symbols: dict[str, CostOverride] = Field(default_factory=dict)


class Constraints(_Model):
    """Portfolio limits enforced by both engines."""

    long_only: bool = False
    max_gross_leverage: float = Field(default=1.0, gt=0)
    max_weight: float | None = Field(default=None, gt=0)  # per symbol, as a fraction of equity
    # Rebalancing trades smaller than this fraction of equity are not sent (closing a
    # position always is). Stops daily top-ups of a few dollars paying the minimum fee.
    min_trade_weight: float = Field(default=0.0, ge=0, lt=1)


class RiskFreeConfig(_Model):
    """Use a symbol's daily total return as the risk-free rate (e.g. ``{symbol: SHY}``)."""

    symbol: str = Field(min_length=1)


class BacktestConfig(_Model):
    """Everything that defines one backtest run."""

    name: str = Field(min_length=1)
    engine: EngineName
    # Event engine only: run its loop, matching, and account in Python or in the Rust
    # module `qr_core` (same results; falls back to Python if the module is missing).
    engine_core: EngineCoreName = "python"
    strategy: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    # A list of symbols, or the name of a group from config/universe.yaml.
    universe: list[str] | str
    start: date
    end: date
    initial_cash: float = Field(default=10_000.0, gt=0)
    benchmark: str | None = "SPY"
    execution: Execution = "next_open"
    rebalance: RebalanceFrequency | list[date] = "daily"
    costs: CostConfig = CostConfig()
    constraints: Constraints = Constraints()
    whole_shares: bool = True  # event engine only; the vectorized engine is fractional
    risk_free_rate: float = 0.0  # annual, for Sharpe, Sortino, and alpha
    # Instead of the constant: a symbol's daily total return (dividends included) as the
    # risk-free series, for Sharpe, Sortino, PSR, alpha, DSR, and the fragile flag.
    risk_free: RiskFreeConfig | None = None
    lookahead_checks: int = Field(default=3, ge=0)  # vectorized engine, see vectorized.py
    notes: str = ""
    # Scales commission, spread, slippage (and impact), and borrow together; 0 is free.
    cost_multiplier: float = Field(default=1.0, ge=0)
    # Multipliers of the config's costs that every run is also evaluated at (see
    # qr.backtest.stress). 1 is the run itself; an empty list turns the stress test off.
    cost_stress: list[float] = Field(default_factory=lambda: list(DEFAULT_COST_STRESS))
    # A split from configs/splits and one of its segments; the loader fills start, end,
    # evaluate_from, and open_end from them (see qr.research.splits).
    split: str | None = None
    segment: str | None = None
    # The study the run belongs to: required on a sealed segment, counted in part 4.
    study_id: str | None = None
    # Metrics start here; the strategy still trades from `start` (a split's embargo).
    evaluate_from: date | None = None
    # The segment runs to the latest data: `end` is the day the run was made, and is
    # left out of the hash so that the same experiment hashes the same on another day.
    open_end: bool = False

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if isinstance(self.universe, list):
            if not self.universe:
                raise ValueError("universe is empty")
            repeated = sorted({s for s in self.universe if self.universe.count(s) > 1})
            if repeated:
                raise ValueError(f"universe repeats symbols: {', '.join(repeated)}")
        if self.evaluate_from is not None and not self.start <= self.evaluate_from <= self.end:
            raise ValueError(
                f"evaluate_from {self.evaluate_from} is outside {self.start} to {self.end}"
            )
        if self.risk_free is not None and self.risk_free_rate != 0:
            raise ValueError("give either risk_free_rate or risk_free, not both")
        if (self.split is None) != (self.segment is None):
            raise ValueError("give both split and segment, or neither")
        if any(m < 0 for m in self.cost_stress):
            raise ValueError("cost_stress multipliers must be zero or more")
        if len(set(self.cost_stress)) != len(self.cost_stress):
            raise ValueError("cost_stress repeats a multiplier")
        return self

    def to_json(self) -> str:
        """Canonical JSON: the same config always gives the same string."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        """Short hash of the canonical JSON, to spot reruns of one config in the registry.

        ``engine_core`` is left out: both cores give identical results, so a run on either
        is the same experiment (and hashes from before the option existed still match).
        ``cost_stress`` only adds evaluations next to the run, and ``cost_multiplier`` is
        left out while it is 1, for the same reason.
        """
        experiment = self.model_dump(mode="json", exclude=_NOT_HASHED)
        for name, neutral in _HASH_NEUTRAL.items():
            if experiment.get(name) == neutral:
                del experiment[name]
        if self.open_end:
            del experiment["end"]
        text = json.dumps(experiment, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_backtest_config(path: Path | str) -> BacktestConfig:
    """Read and validate a backtest YAML file, raising ``ConfigError`` on any problem."""
    resolved = Path(path)
    if not resolved.is_file():
        raise ConfigError(f"backtest config not found: {resolved}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{resolved}: expected a mapping at the top level")
    return backtest_config_from_dict(raw, str(resolved))


def backtest_config_from_dict(raw: dict[str, Any], source: str = "config") -> BacktestConfig:
    """Validate a backtest config given as a mapping (a split is resolved first)."""
    from qr.research.splits import apply_split  # here: qr.research imports this module

    try:
        raw = apply_split(raw)
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    try:
        return BacktestConfig.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{source}: invalid backtest config\n{problems}") from exc
