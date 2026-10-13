# Contributing

This is a personal learning project, but fixes and suggestions are welcome. Open an issue
before starting anything large, so we can agree on scope first.

## Set up

You need Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/asbestboi/zhufu-quant.git
cd zhufu-quant
uv sync                      # creates .venv with the runtime and dev dependencies
uv run pre-commit install    # run the checks automatically on every commit
```

## Run the checks

These are exactly what CI runs, on Ubuntu and Windows with Python 3.12 and 3.13:

```
uv run ruff check .
uv run ruff format --check .     # `uv run ruff format .` fixes it
uv run mypy src
uv run pytest
```

Tests must not touch the network: use the in-memory `FakeSource` in `tests/conftest.py`
and the small CSVs in `tests/fixtures/`. A test that needs a real provider gets
`@pytest.mark.network` and only runs with `uv run pytest --run-network`.

## Code style

- Readable beats clever: this is a codebase for learning from.
- Type hints and a docstring on every public function. Comments say *why*, not *what*.
- Correctness and reproducibility come before speed and features. If a choice is not
  obvious, add a line under "Design decisions" in the README.
- Changes to behaviour come with a test.

## Commits

- Small, logical commits: one idea each, with the tests that cover it.
- Subject line in the imperative, about 70 characters at most: "Add gap check for crypto".
  Use the body to explain why, if the subject does not already.
- Never commit market data, databases, logs, or secrets. `data/`, `logs/`, `*.duckdb`, and
  `.env` are gitignored, and the pre-commit hooks refuse `.env` and DuckDB files.

## Data and licensing

Code is MIT licensed. Market data comes from third-party providers under their own terms
and is not part of this repository; please do not add data files to pull requests beyond
tiny synthetic fixtures.
