# Project Spec: Step 1.1 — Pipeline Polish Before Going Public

> Hand-off document for Claude Code. Improve the existing Step 1 pipeline in `zhufu-quant` (package `qr`) so it is ready to be a public open-source repo. Keep all existing behaviour and tests passing. Ask before deviating from scope.

## 1. Context

Step 1 works: 112 tests pass, lint and mypy are clean, the default universe ingests fine, and re-runs add no duplicates. This step adds polish, robustness, and trust signals before making the repo public. The owner develops on **Windows** (Python 3.13 via `py`, uv installed).

## 2. Continuous Integration (GitHub Actions)

- Add `.github/workflows/ci.yml` running on push and pull request:
  - Matrix: `ubuntu-latest` and `windows-latest`, Python 3.12 and 3.13
  - Steps: install uv → `uv sync` → `ruff check` → `ruff format --check` → `mypy src` → `pytest`
  - Network tests stay skipped in CI.
- Add a CI status badge to the top of the README.

## 3. Pre-commit Hooks

- Add `.pre-commit-config.yaml` with ruff (lint + format) and basic hygiene hooks (trailing whitespace, end-of-file, large-file check, no committed `.env`, no `*.duckdb`).
- Document setup in the README (`uv run pre-commit install`).

## 4. Fail Fast When a Source Is Down

Current behaviour: if Yahoo is unreachable, each symbol retries ~6 times with backoff (~30 s per symbol), so a 50-symbol run wastes ~25 minutes.

- Add a **circuit breaker** per source: after N consecutive symbols fail with network-type errors (default N = 3, configurable in `universe.yaml`), mark the source as down, skip its remaining symbols with status `skipped` and a clear message, and finish the run with a non-zero exit code.
- Distinguish network errors (connection refused, DNS, timeouts, "no timezone found" empty responses across several symbols) from symbol-specific errors (one bad ticker) where practical; document the heuristic.
- Before ingesting, do a quick **connectivity probe** (one lightweight request); if it fails, abort early with a helpful message.
- Tests: simulate a dead source with mocks and assert the run stops after N symbols and logs skipped statuses.

## 5. Second Data Source for Cross-Checking

- Add a second daily source behind the existing `DataSource` interface. Preferred: **Tiingo** (free tier, needs an API key from `.env` as `TIINGO_API_KEY`). If Tiingo's free tier is unsuitable, propose an alternative before implementing.
- It is used for **verification, not as the primary source**:
  - New command: `qr crosscheck [--symbols X,Y] [--days 365]` fetches the comparison source for the window and compares unadjusted close prices (and split events) with stored Yahoo data.
  - Differences above a tolerance (default 0.5% on close, any mismatched split) are written to `quality_issues` with `check_name = 'crosscheck'`.
  - Print a summary table per symbol (days compared, mismatches, worst difference).
- If no API key is configured, the command explains how to get one and exits cleanly.
- Raw responses are snapshotted like Yahoo's.
- Tests with mocked responses, including a deliberate mismatch.

## 6. Named Universes

- Extend `universe.yaml` so instruments can belong to named groups:

```yaml
groups:
  core:  [SPY, QQQ, IWM, TLT, GLD, AAPL, MSFT]
  pairs: [KO, PEP, XOM, CVX, V, MA, HD, LOW, GS, MS]
```

- Every symbol in a group must be defined in `universe:` (validated with a clear error otherwise).
- CLI: `qr ingest --universe pairs`, `qr validate --universe pairs`, `qr status --universe core`. `--symbols` still works; using both is an error.
- Python API: `qr.config.universe_symbols("pairs") -> list[str]`.
- Add the pairs symbols above to the default config.

## 7. Windows Scheduling

- Add `scripts/windows/run_daily_ingest.ps1` that changes to the repo directory, runs `uv run qr ingest` then `uv run qr validate`, and appends output to `logs/ingest.log` (gitignore `logs/`).
- Add `scripts/windows/register_task.ps1` that registers a Windows Task Scheduler task running the script on weekdays at a configurable time (default 23:30 local), with instructions to remove it.
- README "Running it daily" section gets a Windows subsection alongside the existing Linux one.

## 8. Public-Readiness

- README:
  - Badge row (CI, Python version, license)
  - One-paragraph "What this is" at the top
  - **Disclaimer:** for research and education only, not financial advice, no warranty; data comes from third-party sources whose terms apply and is not redistributed by this repo
  - "Roadmap" section listing steps 1–4 with status
- Add `CONTRIBUTING.md` (how to set up, run checks, commit style) — short.
- Run `notebooks/01_explore_data.ipynb` and commit it **with outputs** so charts show on GitHub. Make sure no local file paths or personal info appear in the outputs.
- Confirm `git ls-files` contains no data, database, `.env`, or log files.

## 9. Acceptance Criteria

- [ ] CI passes on Ubuntu and Windows, Python 3.12 and 3.13; badge shows in README
- [ ] Pre-commit hooks installed and documented
- [ ] With the network to Yahoo blocked, `qr ingest` exits non-zero in under 1 minute for the full universe
- [ ] `qr crosscheck` works with a Tiingo key and explains itself without one
- [ ] `qr ingest --universe pairs` works; invalid group config fails with a clear error
- [ ] Windows scheduling scripts work and are documented
- [ ] README has disclaimer, roadmap, badges; notebook committed with outputs
- [ ] All existing and new tests, lint, and type checks pass

## 10. Out of Scope

- Making the second source the primary one or merging data from both
- Intraday data, fundamentals, new asset classes
- Publishing to PyPI

## 11. Working Style

- Small, logical commits.
- Keep code readable; this is a learning codebase.
- Note any non-obvious choices under "Design decisions" in the README.
