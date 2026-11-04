import numpy as np
import pandas as pd
import pytest

from factorlab.portfolios import factor_returns, quintile_weights

NAMES = [f"S{i:02d}" for i in range(20)]


def _scores(months, seed=3):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(size=(len(months), 20)), index=months, columns=NAMES)


def _returns(months, seed=4):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.normal(0.0, 0.05, size=(len(months), 20)),
                        index=months, columns=NAMES)


def test_quintile_weights_pick_true_top_and_bottom():
    row = pd.Series(np.arange(1.0, 21.0), index=NAMES)
    w_long, w_short = quintile_weights(row)
    assert set(w_long[w_long > 0].index) == {"S16", "S17", "S18", "S19"}
    assert set(w_short[w_short > 0].index) == {"S00", "S01", "S02", "S03"}
    assert w_long.sum() == pytest.approx(1.0)
    assert w_short.sum() == pytest.approx(1.0)


def test_quintile_weights_refuse_tiny_cross_sections():
    row = pd.Series(np.arange(10.0), index=NAMES[:10])
    assert quintile_weights(row) == (None, None)


def test_first_month_turnover_is_full_then_zero_for_static_scores():
    months = pd.date_range("2021-01-31", periods=4, freq=pd.offsets.MonthEnd())
    scores = _scores(months[:3])
    scores.loc[:, :] = np.tile(np.arange(20.0), (3, 1))  # identical every month
    out = factor_returns(scores, _returns(months), cost_bps=10.0)
    assert out.iloc[0]["turnover_long"] == pytest.approx(1.0)   # entering from cash
    assert out.iloc[1]["turnover_long"] == pytest.approx(0.0)   # nothing changed
    assert out.iloc[0]["cost"] == pytest.approx(0.002)          # 10bps x (1 + 1)


def test_costs_strictly_reduce_net_returns():
    months = pd.date_range("2021-01-31", periods=13, freq=pd.offsets.MonthEnd())
    scores, rets = _scores(months[:12], seed=11), _returns(months, seed=12)
    free = factor_returns(scores, rets, cost_bps=0.0)
    dear = factor_returns(scores, rets, cost_bps=25.0)
    pd.testing.assert_series_equal(free["gross"], dear["gross"])
    assert dear["net"].sum() < free["net"].sum()
    assert (dear["cost"] > 0).all()


def test_zero_cost_net_equals_gross_and_columns_present():
    months = pd.date_range("2021-01-31", periods=5, freq=pd.offsets.MonthEnd())
    out = factor_returns(_scores(months[:4]), _returns(months), cost_bps=0.0)
    pd.testing.assert_series_equal(out["net"], out["gross"], check_names=False)
    assert list(out.columns) == ["gross", "cost", "net", "turnover_long",
                                 "turnover_short", "n_long", "n_short", "n_names"]
    empty = factor_returns(pd.DataFrame(columns=NAMES), _returns(months))
    assert len(empty) == 0
