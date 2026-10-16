"""Sweeps, trial counting, and the parameter-stability check (part 4)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qr.backtest import registry
from qr.backtest.config import load_backtest_config
from qr.backtest.runner import execute
from qr.cli import app
from qr.config import ConfigError
from qr.research.sweeps import (
    count_trials,
    grid_neighbours,
    list_trials,
    load_sweep,
    run_sweep,
    stability,
    summarize_sweep,
    trial_seed,
)

from .test_research_splits import config_text, study, write

__all__ = ["study"]  # the fixture
runner = CliRunner()

GRID = """
name: mom_grid
base: base.yaml
study_id: mom_v1
grid:
  lookback: [20, 40]
  skip: [5, 10, 30]
"""


def spec(study: Path, text: str = GRID, name: str = "sweep") -> Path:
    write(study, "base", config_text("train"))
    return write(study, name, text)


# ------------------------------------------------------------------ specs


def test_grid_trials_in_order(study: Path) -> None:
    sweep = load_sweep(spec(study))
    assert sweep.trials == [
        {"lookback": 20, "skip": 5}, {"lookback": 20, "skip": 10}, {"lookback": 20, "skip": 30},
        {"lookback": 40, "skip": 5}, {"lookback": 40, "skip": 10}, {"lookback": 40, "skip": 30},
    ]  # fmt: skip
    assert sweep.base.segment == "train" and sweep.base.study_id == "mom_v1"
    assert str(sweep.base.start) == "2015-01-01"


def test_random_search_is_deterministic(study: Path) -> None:
    text = """
name: mom_random
base: base.yaml
study_id: mom_v1
random: { n: 5, seed: 7, ranges: { lookback: { low: 20, high: 60, int: true },
                                   top_n: [1, 2], noise: { low: 0.0, high: 1.0 } } }
"""
    sweep = load_sweep(spec(study, text))
    again = load_sweep(spec(study, text))
    assert sweep.trials == again.trials and len(sweep.trials) == 5
    assert all(20 <= t["lookback"] <= 60 and isinstance(t["lookback"], int) for t in sweep.trials)
    assert all(t["top_n"] in (1, 2) and 0 <= t["noise"] <= 1 for t in sweep.trials)
    assert list_trials(sweep.spec) == sweep.trials
    assert trial_seed(7, 3) == trial_seed(7, 3) != trial_seed(7, 4)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (GRID.replace("study_id: mom_v1\n", ""), "study_id"),
        (GRID + "random: { n: 2, ranges: { a: [1] } }\n", "either grid or random"),
        (GRID.replace("grid:\n  lookback: [20, 40]\n  skip: [5, 10, 30]\n", ""),
         "either grid or random"),
        (GRID.replace("[5, 10, 30]", "[]"), "needs values"),
    ],
)  # fmt: skip
def test_bad_specs(study: Path, text: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_sweep(spec(study, text))


def test_a_sealed_segment_is_refused(study: Path) -> None:
    with pytest.raises(ConfigError, match="is sealed"):
        load_sweep(spec(study, GRID + "segment: test\n"))


def test_the_base_must_name_a_split(study: Path) -> None:
    path = spec(study)
    dated = config_text("train").replace(
        "split: tiny\nsegment: train\n", "start: 2015-01-01\nend: 2015-06-30\n"
    )
    write(study, "base", dated)
    with pytest.raises(ConfigError, match="must name a split"):
        load_sweep(path)


# ------------------------------------------------------------------ running and counting


def test_every_trial_is_logged_and_failures_count(study: Path) -> None:
    result = run_sweep(load_sweep(spec(study)))
    frame = result.frame()
    assert frame["trial"].tolist() == list(range(6))
    # skip 30 with lookback 20 is invalid (skip must be below lookback): it fails.
    assert frame["status"].tolist() == ["ok", "ok", "failed", "ok", "ok", "ok"]
    assert "skip" in frame.loc[2, "error"]
    runs = registry.read_runs()
    trials = runs[runs["sweep_id"] == result.sweep_id].sort_values("trial_index")
    assert trials["trial_index"].tolist() == list(range(6))
    assert trials["status"].tolist() == frame["status"].tolist()
    assert json.loads(trials.iloc[3]["params_json"]) == {"lookback": 40, "skip": 5}
    assert trials["report_path"].isna().all()  # screening only: no tearsheets
    assert registry.read_stress_runs(trials.iloc[0]["run_id"]).empty  # and no cost stress
    ok = trials[trials["status"] == "ok"]
    assert ok["n_returns"].notna().all() and ok["return_kurtosis"].notna().all()
    assert (trials["study_id"] == "mom_v1").all() and (trials["segment"] == "train").all()
    assert result.n_trials == 6 == count_trials("mom_v1")


def test_trials_count_across_sweeps_and_single_runs(study: Path) -> None:
    run_sweep(load_sweep(spec(study)))
    smaller = GRID.replace("name: mom_grid", "name: mom_grid_2").replace("[5, 10, 30]", "[5]")
    run_sweep(load_sweep(spec(study, smaller, "sweep2")))
    assert count_trials("mom_v1") == 6 + 2
    # A single run tagged with the study counts; its stress runs do not.
    text = config_text("validation", study_id="mom_v1")
    single = load_backtest_config(write(study, "single", text))
    outcome = execute(single)
    assert outcome.result is not None and outcome.result.study_trials == 9
    assert count_trials("mom_v1") == 9
    assert count_trials("another_study") == 0
    page = Path(str(outcome.record.report_path)).read_text(encoding="utf-8")
    assert "Trials in study" in page and ">9<" in page


def test_parallel_and_serial_sweeps_log_the_same(study: Path) -> None:
    sweep = load_sweep(spec(study))
    serial = run_sweep(sweep)
    parallel = run_sweep(sweep, workers=3)
    assert [o.index for o in parallel.outcomes] == list(range(6))
    for a, b in zip(serial.outcomes, parallel.outcomes, strict=True):
        assert (a.params, a.seed, a.error) == (b.params, b.seed, b.error)
        pd.testing.assert_series_equal(pd.Series(a.metrics), pd.Series(b.metrics))
        assert a.moments == b.moments
    runs = registry.read_runs()

    def logged(sweep_id: str) -> pd.DataFrame:
        rows = runs[runs["sweep_id"] == sweep_id].sort_values("trial_index")
        return rows[["trial_index", "params_json", "status", "sharpe", "cagr", "trial_seed"]]

    pd.testing.assert_frame_equal(
        logged(serial.sweep_id).reset_index(drop=True),
        logged(parallel.sweep_id).reset_index(drop=True),
    )


# ------------------------------------------------------------------ stability


def test_grid_neighbours() -> None:
    grid = {"a": [1, 2, 3], "b": [1, 2, 3]}
    assert grid_neighbours(grid, 4) == [0, 1, 2, 3, 5, 6, 7, 8]  # the centre
    assert grid_neighbours(grid, 0) == [1, 3, 4]  # a corner
    assert grid_neighbours(grid, 1) == [0, 2, 3, 4, 5]  # an edge
    assert grid_neighbours({"n": [8, 10, 12]}, 0) == [1]
    assert grid_neighbours({"n": [8, 10, 12]}, 1) == [0, 2]


def test_stability_flags_a_lone_spike() -> None:
    grid = {"a": [1, 2, 3], "b": [1, 2, 3]}
    plateau = [0.8, 0.9, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9, 0.8]
    found = stability(grid, plateau)
    assert found is not None and found.best_index == 4 and not found.spike
    assert found.neighbour_median == pytest.approx(0.85)
    spike = [0.1, 0.2, 0.1, 0.2, 1.0, 0.2, 0.1, 0.2, 0.1]
    assert stability(grid, spike).spike  # type: ignore[union-attr]
    losing = [-0.5, -0.4, -0.5, -0.4, -0.1, -0.4, -0.5, -0.4, -0.5]
    assert stability(grid, losing).spike  # type: ignore[union-attr]  # nothing positive
    with_nan = [math.nan] * 8 + [0.5]
    found = stability(grid, with_nan)
    assert found is not None and found.best_index == 8 and found.neighbours == 0 and found.spike
    assert stability(grid, [math.nan] * 9) is None


# ------------------------------------------------------------------ CLI and report


def test_cli_run_and_show(study: Path) -> None:
    path = spec(study)
    ran = runner.invoke(app, ["sweep", "run", str(path)])
    assert ran.exit_code == 0, ran.output
    assert "Trials in study mom_v1: 6" in ran.output and "Stability:" in ran.output
    assert "Sharpe: lookback (rows) x skip" in ran.output  # the heatmap
    sweep_id = registry.read_runs()["sweep_id"].dropna().iloc[0]
    shown = runner.invoke(app, ["sweep", "show", sweep_id[:12]])
    assert shown.exit_code == 0, shown.output
    assert "failed" in shown.output and sweep_id in shown.output
    page = next((study / "reports").glob("*_sweep.html")).read_text(encoding="utf-8")
    assert "Trials in study" in page and "<svg" in page and "Parameter stability" in page
    summary = summarize_sweep(sweep_id)
    assert summary.n_trials == 6 and summary.stability is not None
    sealed = runner.invoke(app, ["sweep", "run", str(spec(study, GRID + "segment: test\n"))])
    assert sealed.exit_code == 1 and "is sealed" in sealed.output
    missing = runner.invoke(app, ["sweep", "show", "sw-nope"])
    assert missing.exit_code == 1 and "no sweep" in missing.output
