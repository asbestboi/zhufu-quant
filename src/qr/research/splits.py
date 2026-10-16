"""Reusable time splits: train, validation, and a sealed test, defined once per study.

A split lives in ``configs/splits/<name>.yaml`` (or ``$QR_SPLITS_DIR``)::

    name: default_2010
    segments:
      train:      { start: 2010-01-01, end: 2018-12-31 }
      validation: { start: 2019-01-01, end: 2022-12-31 }
      test:       { start: 2023-01-01, end: null, sealed: true }   # null = latest data
    embargo_days: 5

Rules, checked when the file is loaded:

* segments are listed in time order and do not overlap;
* only the last segment may be open-ended (``end: null``);
* sealed segments come after every unsealed one, so the unsealed part of a split is one
  unbroken stretch of history (what walk-forward runs over);
* ``embargo_days`` must be given (0 is allowed, and says so explicitly).

**The embargo** is a number of calendar days at the start of every segment but the first
that is left out of that segment's metrics. The strategy still trades through them using
earlier history; only the evaluation starts later. Adjacent segments are therefore fine.
Walk-forward applies the same number differently (see ``walkforward.py``).

**Warm-up**: a backtest of a segment may read history from before the segment's start,
never after its end. The engine enforces the end bound: data after ``end`` is never
loaded.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from qr.config import ConfigError

DEFAULT_SPLITS_DIR = Path("configs/splits")

# What a result is, for labels on every report.
IN_SAMPLE = "in-sample"
VALIDATION = "validation"
WALK_FORWARD = "out-of-sample (walk-forward)"
SEALED_TEST = "sealed test"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Segment(_Model):
    """One stretch of history. ``end`` None means up to the latest data."""

    start: date
    end: date | None = None
    sealed: bool = False


class Split(_Model):
    """A named, validated set of segments."""

    name: str = Field(min_length=1)
    segments: dict[str, Segment] = Field(min_length=1)
    embargo_days: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Split:
        items = list(self.segments.items())
        for index, (name, segment) in enumerate(items):
            if segment.end is not None and segment.end < segment.start:
                raise ValueError(f"segment '{name}' ends before it starts")
            if segment.end is None and index != len(items) - 1:
                raise ValueError(f"segment '{name}' is open-ended but is not the last one")
            if index:
                previous_name, previous = items[index - 1]
                assert previous.end is not None  # only the last segment may be open
                if segment.start <= previous.end:
                    raise ValueError(
                        f"segment '{name}' starts on {segment.start}, not after "
                        f"'{previous_name}' ends ({previous.end}): segments must be in "
                        "time order and must not overlap"
                    )
                if previous.sealed and not segment.sealed:
                    raise ValueError(
                        f"segment '{name}' is not sealed but follows sealed segment "
                        f"'{previous_name}': sealed segments must come last"
                    )
        return self

    def segment(self, name: str) -> Segment:
        """The segment called ``name``."""
        if name not in self.segments:
            known = ", ".join(self.segments)
            raise ConfigError(f"split '{self.name}' has no segment '{name}' (it has: {known})")
        return self.segments[name]

    def evaluate_from(self, name: str) -> date | None:
        """The first day that counts in ``name``'s metrics; None when nothing is cut."""
        if not self.embargo_days or name == next(iter(self.segments)):
            return None
        return self.segment(name).start + timedelta(days=self.embargo_days)

    def label(self, name: str) -> str:
        """What a result on segment ``name`` is: in-sample, validation, or sealed test."""
        if self.segment(name).sealed:
            return SEALED_TEST
        return IN_SAMPLE if name == next(iter(self.segments)) else VALIDATION

    def unsealed_range(self) -> tuple[date, date]:
        """First start and last end of the unsealed segments (what walk-forward may use)."""
        open_ = [s for s in self.segments.values() if not s.sealed]
        if not open_:
            raise ConfigError(f"split '{self.name}' has no unsealed segment")
        end = open_[-1].end
        if end is None:
            raise ConfigError(f"split '{self.name}': the last unsealed segment has no end")
        return open_[0].start, end


def splits_dir() -> Path:
    """Where split files live: ``$QR_SPLITS_DIR``, else ``configs/splits``."""
    return Path(os.environ.get("QR_SPLITS_DIR") or DEFAULT_SPLITS_DIR)


def load_split(name_or_path: str | Path) -> Split:
    """Load a split by name (``configs/splits/<name>.yaml``) or by path."""
    candidate = Path(name_or_path)
    path = (
        candidate
        if candidate.suffix in (".yaml", ".yml")
        else splits_dir() / f"{name_or_path}.yaml"
    )
    if not path.is_file():
        raise ConfigError(f"split not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    if "embargo_days" not in raw:
        raise ConfigError(f"{path}: embargo_days is missing (write 0 if there is none)")
    try:
        split = Split.model_validate(raw)
    except ValidationError as exc:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"{path}: invalid split\n{problems}") from exc
    if candidate.suffix not in (".yaml", ".yml") and split.name != str(name_or_path):
        raise ConfigError(f"{path}: name is '{split.name}', expected '{name_or_path}'")
    return split


def apply_split(raw: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    """A raw backtest config with its ``split`` and ``segment`` turned into dates.

    Fills ``start`` and ``end`` from the segment (an open end becomes ``today``, which
    reads up to the latest stored data, and sets ``open_end``), and ``evaluate_from``
    from the embargo. A config that names a split must not also give its own dates.
    """
    split_name, segment_name = raw.get("split"), raw.get("segment")
    if split_name is None and segment_name is None:
        return raw
    if split_name is None or segment_name is None:
        raise ConfigError("give both split and segment, or neither")
    # Keys at their empty default (as in a model_dump) are not a clash.
    clashing = [
        key
        for key in ("start", "end", "evaluate_from", "open_end")
        if raw.get(key) not in (None, False)
    ]
    if clashing:
        raise ConfigError(
            f"a config with a split takes its dates from it; remove {', '.join(clashing)}"
        )
    split = load_split(str(split_name))
    segment = split.segment(str(segment_name))
    out = dict(raw)
    out["start"] = segment.start
    out["end"] = segment.end if segment.end is not None else (today or date.today())
    out["open_end"] = segment.end is None
    found = split.evaluate_from(str(segment_name))
    if found is not None:
        out["evaluate_from"] = found
    return out
