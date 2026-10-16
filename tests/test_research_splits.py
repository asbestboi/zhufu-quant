"""Splits, the embargo, look-ahead across segment ends, and the sealed holdout (part 3)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from qr import storage
from qr.backtest import registry
from qr.backtest.config import BacktestConfig, load_backtest_config
from qr.backtest.runner import execute, run_backtest
from qr.cli import app
from qr.config import ConfigError, db_path
from qr.research import holdout
from qr.research.pairs import import_sealed_results, study_id
from qr.research.splits import (
    IN_SAMPLE,
    SEALED_TEST,
    VALIDATION,
    apply_split,
    load_split,
)

from .bt_helpers import synthetic_bars
from .conftest import AFTER_JAN_16

pytestmark = pytest.mark.usefixtures("engine_core")
runner = CliRunner()
REPO = Path(__file__).resolve().parents[1]

# 300 business days from 2015-01-02 run to 2016-02-26.
SPLIT = """
name: tiny
segments:
  train:      { start: 2015-01-01, end: 2015-06-30 }
  validation: { start: 2015-07-01, end: 2015-10-30 }
  test:       { start: 2015-11-02, end: null, sealed: true }
embargo_days: 10
"""
UNIVERSE = """
start_date: 2015-01-01
universe:
  - { symbol: AAA, source: yahoo, asset_class: equity }
  - { symbol: BBB, source: yahoo, asset_class: etf }
"""


def config_text(segment: str, **extra: str) -> str:
    fields = {
        "name": f"mom_{segment}",
        "engine": "vectorized",
        "strategy": "xs_momentum",
        "params": "{ lookback: 40, skip: 5, top_n: 1 }",
        "universe": "[AAA, BBB]",
        "split": "tiny",
        "segment": segment,
        "benchmark": "BBB",
        "rebalance": "weekly",
        **extra,
    }
    return "\n".join(f"{k}: {v}" for k, v in fields.items()) + "\n"


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temp project: a split, a universe, and a store with two synthetic symbols."""
    (tmp_path / "configs" / "splits").mkdir(parents=True)
    (tmp_path / "configs" / "splits" / "tiny.yaml").write_text(SPLIT, encoding="utf-8")
    (tmp_path / "universe.yaml").write_text(UNIVERSE, encoding="utf-8")
    monkeypatch.setenv("QR_CONFIG", str(tmp_path / "universe.yaml"))
    monkeypatch.setenv("QR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COLUMNS", "220")
    monkeypatch.delenv("QR_SPLITS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with storage.connect(db_path(tmp_path / "data")) as con:
        storage.init_schema(con)
        for seed, (symbol, asset_class) in enumerate([("AAA", "equity"), ("BBB", "etf")]):
            frame, actions = synthetic_bars(300, seed=seed)
            storage.upsert_instrument(con, symbol, "yahoo", asset_class, True, AFTER_JAN_16)
            storage.upsert_bars(con, symbol, "yahoo", frame, AFTER_JAN_16)
            storage.upsert_corporate_actions(con, symbol, actions)
    return tmp_path


def write(study: Path, name: str, text: str) -> Path:
    path = study / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------ split files


def split_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_split_loads_and_labels(study: Path) -> None:
    split = load_split("tiny")
    assert list(split.segments) == ["train", "validation", "test"]
    assert split.segment("test").end is None and split.segment("test").sealed
    assert [split.label(s) for s in split.segments] == [IN_SAMPLE, VALIDATION, SEALED_TEST]
    assert split.evaluate_from("train") is None  # nothing before train to embargo
    assert split.evaluate_from("validation") == date(2015, 7, 11)
    assert split.unsealed_range() == (date(2015, 1, 1), date(2015, 10, 30))
    with pytest.raises(ConfigError, match="has no segment 'nope'"):
        split.segment("nope")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (SPLIT.replace("end: 2015-06-30", "end: 2015-07-15"), "must not overlap"),
        (SPLIT.replace("start: 2015-07-01, end: 2015-10-30", "start: 2014-01-01, end: 2014-06-30"),
         "time order"),
        (SPLIT.replace("embargo_days: 10\n", ""), "embargo_days is missing"),
        (SPLIT.replace("embargo_days: 10", "embargo_days: -1"), "greater than or equal"),
        (SPLIT.replace("end: 2015-06-30", "end: null"), "open-ended but is not the last"),
        (SPLIT.replace("end: 2015-10-30 }", "end: 2015-10-30, sealed: true }")
         .replace("sealed: true }\nembargo", "}\nembargo"), "sealed segments must come last"),
        (SPLIT.replace("end: 2015-06-30", "end: 2014-06-30"), "ends before it starts"),
    ],
)  # fmt: skip
def test_invalid_splits_are_config_errors(tmp_path: Path, text: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_split(split_file(tmp_path, text))


def test_split_name_must_match_its_file(study: Path) -> None:
    (study / "configs" / "splits" / "other.yaml").write_text(SPLIT, encoding="utf-8")
    with pytest.raises(ConfigError, match="name is 'tiny', expected 'other'"):
        load_split("other")


def test_repo_splits_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QR_SPLITS_DIR", str(REPO / "configs" / "splits"))
    from qr.research import pairs

    split = load_split(pairs.SPLIT)
    assert split.embargo_days == 0
    assert (split.segment("train").start, split.segment("train").end) == pairs.TRAIN
    segment = split.segment("validation")
    assert (segment.start, segment.end) == pairs.VALIDATION
    assert split.segment("test").start == pairs.TEST_START and split.segment("test").sealed


# ------------------------------------------------------------------ configs on a segment


def test_apply_split_fills_the_dates(study: Path) -> None:
    raw = {"split": "tiny", "segment": "validation"}
    assert apply_split(raw) == raw | {
        "start": date(2015, 7, 1),
        "end": date(2015, 10, 30),
        "open_end": False,
        "evaluate_from": date(2015, 7, 11),
    }
    sealed = apply_split({"split": "tiny", "segment": "test"}, today=date(2026, 1, 5))
    assert sealed["end"] == date(2026, 1, 5) and sealed["open_end"] is True
    assert apply_split({"name": "x"}) == {"name": "x"}
    with pytest.raises(ConfigError, match="remove start"):
        apply_split({"split": "tiny", "segment": "train", "start": date(2015, 1, 1)})
    with pytest.raises(ConfigError, match="both split and segment"):
        apply_split({"split": "tiny"})


def test_config_files_on_a_segment(study: Path) -> None:
    config = load_backtest_config(write(study, "v", config_text("validation")))
    assert (config.start, config.end) == (date(2015, 7, 1), date(2015, 10, 30))
    assert config.evaluate_from == date(2015, 7, 11)
    with pytest.raises(ConfigError, match="remove start"):
        load_backtest_config(write(study, "bad", config_text("train", start="2015-01-01")))
    with pytest.raises(ConfigError, match="no segment 'oops'"):
        load_backtest_config(write(study, "bad2", config_text("oops")))


def test_an_open_end_is_left_out_of_the_hash(study: Path) -> None:
    raw = {"name": "t", "engine": "vectorized", "strategy": "buy_and_hold", "universe": ["AAA"],
           "split": "tiny", "segment": "test", "study_id": "s"}  # fmt: skip
    today = BacktestConfig.model_validate(apply_split(raw, today=date(2026, 1, 5)))
    later = BacktestConfig.model_validate(apply_split(raw, today=date(2026, 6, 1)))
    assert today.end != later.end and today.config_hash() == later.config_hash()
    fixed = BacktestConfig.model_validate(apply_split(raw | {"segment": "validation"}))
    moved = fixed.model_copy(update={"end": date(2015, 10, 29)})
    assert fixed.config_hash() != moved.config_hash()


def test_the_embargo_is_left_out_of_the_metrics(study: Path) -> None:
    config = load_backtest_config(write(study, "v", config_text("validation")))
    result = run_backtest(config, cost_stress=False)
    assert result.equity.index[0] == pd.Timestamp("2015-07-13")  # first session from Jul 11
    assert result.label == VALIDATION
    whole = run_backtest(config.model_copy(update={"evaluate_from": None}), cost_stress=False)
    assert whole.equity.index[0] == pd.Timestamp("2015-07-01")
    # The strategy traded through the embargo: the account carries into it unchanged.
    pd.testing.assert_series_equal(result.equity, whole.equity.loc["2015-07-13":])
    assert result.metrics != whole.metrics


def test_changing_data_after_a_segment_end_changes_nothing_in_it(study: Path) -> None:
    config = load_backtest_config(write(study, "t", config_text("train")))
    before = run_backtest(config)
    with storage.connect(db_path(study / "data")) as con:
        rng = np.random.default_rng(5)
        for symbol in ("AAA", "BBB"):
            bars = storage.read_bars(con, [symbol])
            later = bars["date"] > pd.Timestamp(config.end)
            bars = bars.loc[later].copy()
            for column in ("open", "high", "low", "close"):
                bars[column] = bars[column].to_numpy() * rng.uniform(0.5, 1.5, len(bars))
            storage.upsert_bars(con, symbol, "yahoo", bars, datetime(2026, 1, 1))
    after = run_backtest(config)
    assert after.metrics == before.metrics and after.stress_metrics == before.stress_metrics
    pd.testing.assert_series_equal(after.equity, before.equity, check_exact=True)


# ------------------------------------------------------------------ sealed holdout


def sealed_config(study: Path, name: str = "test", **extra: str) -> BacktestConfig:
    return load_backtest_config(
        write(study, name, config_text("test", study_id="mom_v1", cost_stress="[]", **extra))
    )


def test_a_sealed_segment_needs_a_study(study: Path) -> None:
    config = load_backtest_config(write(study, "t", config_text("test")))
    with pytest.raises(holdout.HoldoutError, match="set study_id"):
        execute(config)
    assert registry.read_runs().empty  # refused before anything ran


def test_first_run_stores_same_config_shows_it_different_config_refuses(study: Path) -> None:
    first = execute(sealed_config(study))
    assert first.error is None and first.sealed is None and first.result is not None
    assert first.result.label == SEALED_TEST
    stored = holdout.find("mom_v1", "tiny", "test")
    assert stored is not None and stored.run_id == first.record.run_id
    assert stored.metrics["sharpe"] == pytest.approx(first.result.metrics["sharpe"])
    assert registry.read_runs().iloc[0]["result_label"] == SEALED_TEST

    again = execute(sealed_config(study))  # the same config: no run, the stored result
    assert again.sealed == stored and again.result is None
    assert again.record.run_id == first.record.run_id
    assert len(registry.read_runs()) == 1

    other = sealed_config(study, "other", params="{ lookback: 30, skip: 5, top_n: 1 }")
    with pytest.raises(holdout.HoldoutError, match="already used sealed segment"):
        execute(other)
    assert len(registry.read_runs()) == 1


def test_an_override_is_recorded_and_marked(study: Path) -> None:
    execute(sealed_config(study))
    other = sealed_config(study, "other", params="{ lookback: 30, skip: 5, top_n: 1 }")
    reused = execute(other, unseal_reason="  checking a bug fix  ")
    assert reused.error is None and reused.result is not None
    assert reused.result.holdout_reused == "checking a bug fix"
    overrides = holdout.read_overrides()
    assert overrides["reason"].tolist() == ["checking a bug fix"]
    assert overrides["run_id"].tolist() == [reused.record.run_id]
    # The first result still stands.
    stored = holdout.find("mom_v1", "tiny", "test")
    assert stored is not None and stored.run_id != reused.record.run_id
    page = Path(str(reused.record.report_path)).read_text(encoding="utf-8")
    assert "Holdout reused" in page and "checking a bug fix" in page
    row = registry.read_runs(run_ids=[reused.record.run_id]).iloc[0]
    assert row["unseal_reason"] == "checking a bug fix"
    # A blank reason is no reason.
    with pytest.raises(holdout.HoldoutError):
        execute(other, unseal_reason="   ")


def test_cli_sealed_flow(study: Path) -> None:
    path = str(write(study, "test", config_text("test", study_id="mom_v1")))
    ran = runner.invoke(app, ["backtest", "run", path])
    assert ran.exit_code == 0, ran.output
    shown = runner.invoke(app, ["backtest", "run", path])
    assert shown.exit_code == 0 and "Sealed segment already run" in shown.output
    other = str(write(study, "other", config_text("test", study_id="mom_v1", rebalance="daily")))
    refused = runner.invoke(app, ["backtest", "run", other])
    assert refused.exit_code == 1 and "already used sealed segment" in refused.output
    forced = runner.invoke(app, ["backtest", "run", other, "--unseal-reason", "why not"])
    assert forced.exit_code == 0, forced.output
    listed = runner.invoke(app, ["holdout", "list"])
    assert "mom_v1" in listed.output and "Holdout overrides" in listed.output
    run_id = registry.read_runs().iloc[0]["run_id"]
    shown = runner.invoke(app, ["backtest", "show", run_id])
    assert "why not" in shown.output and "sealed test" in shown.output


# ------------------------------------------------------------------ pairs import


def test_import_pairs_results_without_rerunning(study: Path) -> None:
    from qr.research.pairs import COMPARISON_PAIRS, MAIN_PAIR

    results, created = {}, datetime(2026, 10, 14, 21, 58)
    for i, pair in enumerate([MAIN_PAIR, *COMPARISON_PAIRS]):
        name = f"pairs_{pair[0].lower()}_{pair[1].lower()}_test"
        results[f"{pair[0]} / {pair[1]} test"] = {"sharpe": 0.1 * i, "cagr": 0.01}
        registry.log_run(
            registry.RunRecord(
                run_id=f"20261014T2158{i:02d}-abc", created_at=created, strategy="pairs",
                engine="vectorized", config_json="{}", config_hash=f"hash{i}",
                git_commit=None, git_dirty=None, data_snapshot=None,
                start_date=date(2023, 1, 1), end_date=date(2026, 10, 14), sharpe=0.1 * i,
                cagr=0.01, max_drawdown=-0.1, turnover=1.0, report_path=None, notes="",
                name=name, status=registry.OK,
            )
        )  # fmt: skip
    sealed = study / "test_results.json"
    sealed.write_text(json.dumps({"sealed_at": "2026-10-14T21:58:24+00:00", "results": results}))
    assert import_sealed_results(sealed) == [study_id(p) for p in [MAIN_PAIR, *COMPARISON_PAIRS]]
    stored = holdout.find("pairs_ko_pep", "pairs_2010", "test")
    assert stored is not None and stored.config_hash == "hash0"
    assert stored.sealed_at == datetime(2026, 10, 14, 21, 58, 24)
    assert holdout.find("pairs_gs_ms", "pairs_2010", "test").metrics["sharpe"] == 0.4  # type: ignore[union-attr]
    assert import_sealed_results(sealed) == []  # already there: nothing changes
    assert len(holdout.read_results()) == 5


# ------------------------------------------------------------------ pairs migration


SEALED = REPO / "reports" / "pairs" / "test_results.json"
STORE = REPO / "data" / "market.duckdb"


@pytest.mark.skipif(
    not (SEALED.exists() and STORE.exists()), reason="needs the local pairs data and results"
)
def test_pairs_train_and_validation_reproduce_through_the_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running train and validation via split pairs_2010 matches the sealed file exactly."""
    from qr.backtest.data import load_market_data
    from qr.load import get_bars
    from qr.research import pairs

    monkeypatch.chdir(REPO)
    monkeypatch.delenv("QR_DATA_DIR", raising=False)
    monkeypatch.delenv("QR_SPLITS_DIR", raising=False)
    sealed = json.loads(SEALED.read_text())
    base = load_backtest_config("configs/backtests/pairs_ko_pep.yaml")
    raw = {k: v for k, v in base.model_dump().items() if k not in ("start", "end")}
    for pair in [pairs.MAIN_PAIR, *pairs.COMPARISON_PAIRS]:
        for segment in ("train", "validation"):
            config = BacktestConfig.model_validate(
                apply_split(
                    raw | {"universe": list(pair), "split": pairs.SPLIT, "segment": segment,
                           "params": base.params | sealed["params"] | {"y": pair[0], "x": pair[1]}}
                )
            )  # fmt: skip
            data = load_market_data(list(pair), config.end)
            spy = get_bars("SPY", end=config.end)["close"].droplevel("symbol")
            got = pairs.summarize(run_backtest(config, data, spy, cost_stress=False))
            assert got == sealed["results"][f"{pair[0]} / {pair[1]} {segment}"]
