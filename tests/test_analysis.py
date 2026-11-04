import numpy as np
import pandas as pd
import pytest

from factorlab import analysis

NAMES = [f"S{i:02d}" for i in range(40)]


def test_performance_on_constant_growth():
    r = pd.Series(np.full(24, 0.01),
                  index=pd.date_range("2020-01-31", periods=24, freq=pd.offsets.MonthEnd()))
    p = analysis.performance(r)
    assert p["ann_return"] == pytest.approx(0.12)
    assert p["cum_return"] == pytest.approx(1.01 ** 24 - 1)
    assert p["max_drawdown"] == pytest.approx(0.0)
    assert p["hit_rate"] == pytest.approx(1.0)


def test_drawdown_window_finds_peak_and_trough():
    r = pd.Series([0.10, -0.50, 0.10, -0.10],
                  index=pd.date_range("2020-01-31", periods=4, freq=pd.offsets.MonthEnd()))
    dd = analysis.drawdown_window(r)
    # equity: 1.10, 0.55, 0.605, 0.5445 -> deepest point is the LAST month
    assert dd["max_drawdown"] == pytest.approx(-0.505, abs=1e-9)
    assert dd["peak"] == r.index[0]
    assert dd["trough"] == r.index[3]


def _planted_panel(seed=5, months=30, beta=1.0):
    """Scores, and next-month returns driven by the score's rank."""
    rng = np.random.default_rng(seed)
    mret = pd.date_range("2020-01-31", periods=months + 1, freq=pd.offsets.MonthEnd())
    scores = pd.DataFrame(rng.normal(size=(months, len(NAMES))),
                          index=mret[:-1], columns=NAMES)
    rets = pd.DataFrame(rng.normal(0, 0.001, size=(months + 1, len(NAMES))),
                        index=mret, columns=NAMES)
    for t in range(months):
        rank_pct = scores.iloc[t].rank(pct=True)
        rets.iloc[t + 1] += beta * 0.05 * (rank_pct - 0.5)
    return scores, rets


def test_ic_recovers_planted_positive_signal():
    scores, rets = _planted_panel(beta=1.0)
    ic = analysis.information_coefficient(scores, rets)
    s = analysis.ic_summary(ic)
    assert s["mean_ic"] > 0.5
    assert s["ic_tstat"] > 5


def test_ic_reversed_signal_is_negative():
    scores, rets = _planted_panel(beta=-1.0)
    s = analysis.ic_summary(analysis.information_coefficient(scores, rets))
    assert s["mean_ic"] < -0.5


def test_fama_macbeth_recovers_known_slopes():
    rng = np.random.default_rng(9)
    months = pd.date_range("2020-01-31", periods=37, freq=pd.offsets.MonthEnd())
    names = NAMES[:30]
    f1 = pd.DataFrame(rng.normal(size=(36, 30)), index=months[:-1], columns=names)
    f2 = pd.DataFrame(rng.normal(size=(36, 30)), index=months[:-1], columns=names)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(37, 30)), index=months, columns=names)
    for t in range(36):
        rets.iloc[t + 1] += 0.004 * f1.iloc[t] - 0.002 * f2.iloc[t]
    fm = analysis.fama_macbeth({"f1": f1, "f2": f2}, rets, min_names=20)
    assert fm.loc["f1", "n_months"] == 36
    assert fm.loc["f1", "mean_slope"] == pytest.approx(0.004, abs=0.0015)
    assert fm.loc["f2", "mean_slope"] == pytest.approx(-0.002, abs=0.0015)
    assert fm.loc["f1", "nw_tstat"] > 2


def test_newey_west_se_exceeds_plain_for_autocorrelated_series():
    rng = np.random.default_rng(13)
    x = np.zeros(240)
    e = rng.normal(size=240)
    for i in range(1, 240):
        x[i] = 0.7 * x[i - 1] + e[i]
    plain = x.std(ddof=1) / np.sqrt(len(x))
    nw = analysis._nw_se(x, lags=6)
    assert nw > 1.2 * plain
    # and on iid data the two roughly agree
    y = rng.normal(size=240)
    ratio = analysis._nw_se(y, lags=6) / (y.std(ddof=1) / np.sqrt(len(y)))
    assert 0.7 < ratio < 1.4
