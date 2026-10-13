# Project Spec: Learning Study — Pairs Trading (KO / PEP and friends)

> Hand-off document for Claude Code. Build this inside the existing `zhufu-quant` repo, using the Step 1 pipeline (`qr.load`). If the Step 2 backtester (`qr.backtest`) exists, use it for the final backtest section; otherwise implement a minimal, clearly-labelled vectorized backtest inside the study module and note that it should be replaced by `qr.backtest` later. Ask before deviating from scope.

## 1. Purpose

This is a **learning project first**. I'm new to quant and want to understand pairs trading by building it step by step. The main output is a set of notebooks that **teach**: every step has a short plain-language explanation of what we're doing, why, and what could go wrong. Clear explanations matter more than clever code.

Secondary goal: an honest answer to "does a simple pairs strategy on these stocks work after costs, without hindsight?" A "no" is a perfectly good result if it's reached honestly.

## 2. The Idea Being Studied

Two stocks driven by the same economic forces (e.g. Coca-Cola and PepsiCo) tend to move together. When the price gap between them (the **spread**) becomes unusually wide, bet on it closing: buy the laggard, short the leader. Profit comes from the gap closing, not from the market's direction.

## 3. Data

- Add these symbols to `config/universe.yaml` (asset_class `equity`), then run `qr ingest`:
  - Main pair: **KO / PEP**
  - Comparison pairs: **XOM / CVX**, **V / MA**, **HD / LOW**, **GS / MS**
  - Include **SPY** (already present) as a market reference.
- Load everything with `qr.load.get_bars(..., adjustment="total_return")`.
- Use log prices for spread calculations.

## 4. Code Structure

```
src/qr/research/
├── __init__.py
└── pairs.py              # reusable, tested functions (no plotting)
notebooks/pairs/
├── 01_intuition.ipynb
├── 02_cointegration.ipynb
├── 03_signals.ipynb
├── 04_honest_backtest.ipynb
└── 05_findings.ipynb
tests/test_pairs.py
```

Notebooks call functions from `pairs.py`; logic lives in the module so it can be tested and reused.

### `pairs.py` functions (type-hinted, docstrings, all past-data-only)

- `hedge_ratio_ols(y, x) -> float` — static OLS hedge ratio (for teaching only)
- `rolling_hedge_ratio(y, x, window) -> pd.Series` — estimated from past `window` days only, **shifted by one day** so day t uses data up to t-1
- `spread(y, x, beta) -> pd.Series`
- `rolling_zscore(s, window) -> pd.Series` — past-only mean and std
- `half_life(s) -> float` — mean-reversion half-life via AR(1) regression
- `engle_granger(y, x) -> dict` — wrapper around `statsmodels.tsa.stattools.coint` returning statistic, p-value, critical values
- `rolling_cointegration_pvalue(y, x, window, step) -> pd.Series`
- `pair_signals(z, entry, exit, stop) -> pd.Series` — position in {-1, 0, +1} for the spread; enter at |z| > entry, exit at |z| < exit, stop out at |z| > stop
- `pair_weights(signal, beta, gross=1.0) -> pd.DataFrame` — converts spread position into weights on the two legs, dollar-neutral-ish via the hedge ratio, normalized to `gross` exposure

## 5. Notebooks (each starts with a "What you'll learn" box and ends with "Key takeaways")

### 01_intuition
- Plot KO and PEP normalized to 100 at the start.
- Plot their ratio and log-spread over time.
- Show daily return correlation over the full period and a rolling 1-year correlation.
- **Explain:** what a spread is; why "it looks like it always comes back" on a chart is hindsight; PepsiCo's large snack business (Frito-Lay) means the two companies don't share an identical core.

### 02_cointegration
- **Explain** correlation vs. cointegration with a synthetic example: two random walks that are highly correlated but not cointegrated, and two series that are cointegrated. Plot both.
- Run the Engle-Granger test on KO/PEP and every comparison pair; show a table.
- Show the rolling cointegration p-value over time for KO/PEP — **explain that relationships come and go**.
- Compute the half-life of the spread; explain what it means for holding periods.
- **Explain** the multiple-testing problem: testing many pairs guarantees some look cointegrated by chance.

### 03_signals
- Compute the rolling hedge ratio and z-score (default windows: 252 days for beta, 60 days for z-score; make them parameters).
- Plot the z-score with entry/exit/stop bands and mark trades.
- **Explain** the difference between the static hedge ratio from notebook 02 (uses the future) and the rolling one (doesn't). Show how much the static version flatters results.

### 04_honest_backtest
- **Split by time and state it at the top:**
  - Train: 2010–2018 (choose parameters here only)
  - Validation: 2019–2022 (check, adjust at most a little)
  - Test: 2023–present (**run exactly once at the end**; the notebook must have a clearly marked final cell, and the text must say not to tweak after seeing it)
- Parameters to explore in training only: z-score window, entry threshold, exit threshold, stop threshold. Keep the grid small (≤ 50 combinations) and **report how many combinations were tried**.
- Costs: commission + spread + slippage from config (default 5 bps per leg per trade), plus a short borrow fee (default 0.5%/year on the short leg). Show results **before and after costs**.
- Execution: signal at close of day t → trade at open of day t+1 (or close of t+1 if opens are unavailable in the vectorized path; state which).
- Metrics: total return, CAGR, Sharpe, max drawdown, number of trades, average holding period, hit rate. Compare against simply holding SPY.
- If `qr.backtest` exists, run the final strategy through it and log runs in the experiment registry.

### 05_findings
- Plain-language summary: did it work on train, validation, test? After costs?
- How did the comparison pairs do with the **same** parameters (no re-tuning)?
- List what could be tried next and which new biases each idea would risk.

## 6. Tests (`tests/test_pairs.py`, offline, synthetic data)

- Hedge ratio recovers a known beta on synthetic cointegrated data.
- Rolling functions never use future data: perturb data after date t, results up to t unchanged.
- Half-life recovers a known value from a simulated AR(1) process (within tolerance).
- Engle-Granger: low p-value on a synthetic cointegrated pair, high p-value on independent random walks (fixed seeds).
- Signal state machine: hand-built z-score series produces the expected entries, exits, and stops.
- Weights: long-spread position produces long y / short x with the right hedge proportions.

`ruff`, `mypy`, and `pytest` must pass. Add `statsmodels` as a dependency.

## 7. Acceptance Criteria

- [ ] New symbols ingested and validated with no errors
- [ ] All five notebooks run top to bottom with saved outputs (charts visible on GitHub)
- [ ] Every notebook has "What you'll learn" and "Key takeaways" sections in plain language
- [ ] Test-period result is shown once, after costs, alongside SPY
- [ ] Number of parameter combinations tried is reported
- [ ] All tests, lint, and type checks pass
- [ ] README gets a short "Research studies" section linking the notebooks

## 8. Out of Scope

- Searching the whole market for pairs (too much multiple-testing risk for a first study)
- Kalman-filter hedge ratios, machine learning, intraday data
- Live or paper trading

## 9. Working Style

- Explanations are for a CS student new to finance: short, concrete, no unexplained jargon.
- Prefer honest, pessimistic assumptions. Never present a hindsight-based result without labelling it as such.
- Commit in small steps.
