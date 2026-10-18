"""A small LiveConfig builder shared by the cycle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qr.live.config import LiveConfig


def make_live_config(
    universe: list[str], working_dir: Path, strategy: str = "buy_and_hold", **overrides: Any
) -> LiveConfig:
    raw: dict[str, Any] = {
        "name": "test",
        "strategy": strategy,
        "universe": universe,
        "working_dir": working_dir,
        "cash_buffer": 0.0,
        "min_trade_weight": 0.0,
    }
    raw.update(overrides)
    return LiveConfig.model_validate(raw)
