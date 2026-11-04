"""The no-look-ahead suite — the tests this project exists to pass.

Timing contract: scores formed at month-end t earn the return of month
t+1. These tests attack that contract three ways:
  1. Corrupting every return dated on/before the first formation month
     must change nothing (returns are only read strictly after formation).
  2. A same-month return trap: huge returns in the formation month itself
     must never appear in the booked series.
  3. Relabelling time itself (scores AND returns shifted one month) must
     shift the IC series by exactly one month with identical values —
     alignment is mechanical, never calendar luck.
  4. The IC engine must pair scores with the *tradable* lag (t -> t+1):
     on data where next-month returns follow the score but same-month
     returns anti-follow it, the engine's IC is positive while the
     same-month pairing is negative.
"""
import numpy as np
import pandas as pd
import pytest

from factorlab.analysis import information_coefficient
from factorlab.portfolios import factor_returns

NAMES = [f"S{i:02d}" for i in range(20)]


def _panels(seed=21, n_score_months=8):
    rng = np.random.default_rng(seed)
    ret_months = pd.date_range("2021-01-31", periods=n_score_months + 2, freq=pd.offsets.MonthEnd())
    scores = pd.DataFrame(rng.normal(size=(n_score_months, 20)),
                          index=ret_months[1:-1], columns=NAMES)
    rets = pd.DataFrame(rng.normal(0, 0.04, size=(len(ret_months), 20)),
                        index=ret_months, columns=NAMES)
    return scores, rets


def test_returns_before_first_formation_are_never_read():
    scores, rets = _panels()
    base = factor_returns(scores, rets, cost_bps=10.0)
    poisoned = rets.copy()
    poisoned.loc[poisoned.index <= scores.index[0]] *= 100.0  # absurd returns
    out = factor_returns(scores, poisoned, cost_bps=10.0)
    pd.testing.assert_frame_equal(base, out)


def test_same_month_returns_are_a_trap_that_is_ignored():
    scores, rets = _panels()
    # In the formation month itself, make the *bottom*-scored stock soar:
    # a look-ahead engine would book it; a correct one cannot see it.
    trap_month = scores.index[0]
    bottom = scores.loc[trap_month].idxmin()
    rets.loc[trap_month, bottom] = 5.0
    # and give it a normal next month so booked values stay comparable
    base = factor_returns(scores, rets, cost_bps=0.0)
    rets2 = rets.copy()
    rets2.loc[trap_month, bottom] = -7.0  # change the trap; booked must not move
    out = factor_returns(scores, rets2, cost_bps=0.0)
    pd.testing.assert_frame_equal(base, out)


def test_gross_equals_weighted_next_month_return_exactly():
    months = pd.date_range("2021-01-31", periods=3, freq=pd.offsets.MonthEnd())
    scores = pd.DataFrame([np.arange(20.0)], index=[months[0]], columns=NAMES)
    rets = pd.DataFrame(np.zeros((3, 20)), index=months, columns=NAMES)
    rets.loc[months[1], "S19"] = 0.40   # top-scored stock, NEXT month: +40%
    rets.loc[months[1], "S00"] = -0.20  # bottom-scored stock, NEXT month: -20%
    rets.loc[months[0], "S19"] = 9.99   # same-month noise: must be ignored
    out = factor_returns(scores, rets, cost_bps=0.0)
    # top book: 4 names at 1/4 weight -> +0.40/4 ; short book subtracts -0.20/4
    assert out.iloc[0]["gross"] == pytest.approx(0.40 / 4 + 0.20 / 4)


def test_ic_relabeling_time_shifts_series_exactly():
    scores, rets = _panels(seed=33, n_score_months=10)
    ic = information_coefficient(scores, rets)
    # MonthEnd (not DateOffset(months=1), which lands on the 28th/30th and
    # collides return months) — a true one-month shift of month-ends.
    scores2 = scores.copy()
    scores2.index = scores.index + pd.offsets.MonthEnd(1)
    rets2 = rets.copy()
    rets2.index = rets.index + pd.offsets.MonthEnd(1)
    ic2 = information_coefficient(scores2, rets2)
    assert len(ic2) == len(ic)
    assert (ic2.index == (ic.index + pd.offsets.MonthEnd(1))).all()
    assert np.allclose(ic2.to_numpy(), ic.to_numpy())


def test_ic_uses_exactly_the_tradable_lag():
    from scipy import stats
    rng = np.random.default_rng(41)
    months = pd.date_range("2020-01-31", periods=27, freq=pd.offsets.MonthEnd())
    scores = pd.DataFrame(rng.normal(size=(26, 20)), index=months[:-1], columns=NAMES)
    rets = pd.DataFrame(rng.normal(0, 0.0005, size=(27, 20)), index=months, columns=NAMES)
    for m in range(1, 27):
        # next-month returns FOLLOW last month's score ...
        rets.iloc[m] += 0.04 * (scores.iloc[m - 1].rank(pct=True) - 0.5)
        if m <= 25:
            # ... while same-month returns ANTI-follow this month's score
            rets.iloc[m] -= 0.04 * (scores.iloc[m].rank(pct=True) - 0.5)
    ic = information_coefficient(scores, rets)
    assert ic.mean() > 0.3                       # engine books t -> t+1
    same_month = [stats.spearmanr(scores.iloc[t], rets.iloc[t]).statistic
                  for t in range(26)]
    assert np.mean(same_month) < -0.3            # the trap pairing is negative
