r"""The Deflated Sharpe Ratio: how likely a selected Sharpe is real, given how many tried.

After trying ``N`` strategies, the best observed Sharpe ratio is high by luck alone. The
Deflated Sharpe Ratio (Bailey & López de Prado, 2014, "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality") is the
Probabilistic Sharpe Ratio of the selected run, measured not against zero but against
the Sharpe ratio the best of ``N`` worthless trials would be expected to reach.

All Sharpe ratios here are **per period** (daily, not annualized), the unit the formulas
are written in. ``qr`` stores annualized Sharpe ratios; they are divided by
:math:`\sqrt{252}` first.

1. **Expected maximum Sharpe of N trials with no skill** (the benchmark):

   .. math::

      SR_0 = \sqrt{V}\,\Big((1-\gamma)\,\Phi^{-1}\!\big(1 - \tfrac{1}{N}\big)
             + \gamma\,\Phi^{-1}\!\big(1 - \tfrac{1}{N e}\big)\Big)

   where :math:`V` is the variance of the ``N`` trials' Sharpe ratios, :math:`\gamma
   \approx 0.5772` the Euler-Mascheroni constant, :math:`\Phi^{-1}` the inverse standard
   normal CDF, and :math:`e` Euler's number. With one trial there is nothing to
   deflate for: :math:`SR_0 = 0`.

2. **DSR** is the PSR of the selected run against :math:`SR_0`:

   .. math::

      DSR = \Phi\!\left(\frac{(\widehat{SR} - SR_0)\sqrt{T-1}}
            {\sqrt{1 - \hat\gamma_3\,\widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\,\widehat{SR}^2}}
            \right)

   with :math:`\widehat{SR}` the selected run's Sharpe, :math:`T` its number of returns,
   :math:`\hat\gamma_3` their skewness, and :math:`\hat\gamma_4` their kurtosis (not
   excess: 3 for a normal distribution).

Assumptions, as in the paper: the trials are treated as independent draws (correlated
trials make ``N`` an overstatement, so DSR is conservative); the Sharpe ratios'
variance is estimated from the trials themselves; returns are i.i.d. apart from their
skew and fat tails. Moments are population moments, as in
:func:`qr.backtest.metrics.probabilistic_sharpe_ratio`, so that DSR equals PSR when
``N = 1``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from qr.backtest.metrics import PERIODS_PER_YEAR

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


@dataclass(frozen=True)
class ReturnMoments:
    """What DSR needs from one run's daily (excess) returns."""

    observations: int
    sharpe: float  # per period
    skew: float
    kurtosis: float  # not excess


def return_moments(returns: pd.Series) -> ReturnMoments:
    """Sample size, per-period Sharpe, skewness, and kurtosis of ``returns``."""
    values = np.asarray(returns.dropna(), dtype="float64")
    n = len(values)
    deviation = float(values.std(ddof=1)) if n > 2 else math.nan
    if not deviation > 1e-15:
        return ReturnMoments(n, math.nan, math.nan, math.nan)
    centered = values - values.mean()
    spread = float((centered**2).mean()) ** 0.5
    return ReturnMoments(
        observations=n,
        sharpe=float(values.mean() / deviation),
        skew=float((centered**3).mean()) / spread**3,
        kurtosis=float((centered**4).mean()) / spread**4,
    )


def per_period(annual_sharpe: float) -> float:
    """An annualized Sharpe ratio (as ``qr`` stores them) on a daily basis."""
    return annual_sharpe / math.sqrt(PERIODS_PER_YEAR)


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """:math:`SR_0`: the best Sharpe (per period) expected from ``n_trials`` with no skill."""
    if n_trials < 1:
        raise ValueError("need at least one trial")
    if n_trials == 1 or not variance > 0:
        return 0.0
    return math.sqrt(variance) * (
        (1.0 - EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def probabilistic_sharpe(moments: ReturnMoments, benchmark: float) -> float:
    """PSR of a run against a per-period ``benchmark`` Sharpe; NaN when undefined."""
    sr, n = moments.sharpe, moments.observations
    if not (math.isfinite(sr) and n > 2):
        return math.nan
    variance = 1.0 - moments.skew * sr + (moments.kurtosis - 1.0) / 4.0 * sr**2
    if not variance > 0:
        return math.nan
    return _NORMAL.cdf((sr - benchmark) * math.sqrt(n - 1) / math.sqrt(variance))


@dataclass(frozen=True)
class Deflated:
    """A DSR and what went into it."""

    dsr: float
    n_trials: int
    sharpe_variance: float  # of the trials' per-period Sharpe ratios
    benchmark: float  # SR_0, per period

    @property
    def benchmark_annualized(self) -> float:
        """:math:`SR_0` on the annual scale of the reports."""
        return self.benchmark * math.sqrt(PERIODS_PER_YEAR)


def deflated_sharpe(
    moments: ReturnMoments, trial_sharpes: Sequence[float], n_trials: int | None = None
) -> Deflated:
    """The DSR of the run described by ``moments``.

    ``trial_sharpes`` are the **annualized** Sharpe ratios of every trial (as stored in
    the registry); NaNs are dropped for the variance. ``n_trials`` defaults to their
    count; pass the study's full count when failed trials (which have no Sharpe) must be
    counted too.
    """
    finite = [per_period(s) for s in trial_sharpes if math.isfinite(s)]
    count = n_trials if n_trials is not None else len(trial_sharpes)
    count = max(count, 1)
    variance = float(np.var(finite, ddof=1)) if len(finite) > 1 else 0.0
    benchmark = expected_max_sharpe(count, variance)
    return Deflated(probabilistic_sharpe(moments, benchmark), count, variance, benchmark)
