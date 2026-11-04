import numpy as np
import pandas as pd
import pytest

from factorlab.factors import momentum_raw, standardise, winsorise, zscore

rng = np.random.default_rng(7)


def test_winsorise_clips_to_quantiles():
    s = pd.Series(np.concatenate([np.arange(1.0, 101.0), [10000.0]]))
    w = winsorise(s)
    assert w.max() == pytest.approx(s.quantile(0.99))
    assert w.min() == pytest.approx(s.quantile(0.01))
    assert (w >= w.min()).all() and (w <= w.max()).all()


def test_winsorise_leaves_small_samples_alone():
    s = pd.Series([1.0, 2.0, 100.0])
    pd.testing.assert_series_equal(winsorise(s), s)


def test_zscore_has_zero_mean_unit_std():
    s = pd.Series(rng.normal(size=40))
    z = zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-12)


def test_zscore_degenerate_sections_are_nan():
    assert zscore(pd.Series(np.full(30, 2.5))).isna().all()       # zero variance
    assert zscore(pd.Series(rng.normal(size=5))).isna().all()      # too few names


def test_momentum_raw_uses_skip_and_lookback_endpoints():
    months = pd.date_range("2019-01-31", periods=16, freq=pd.offsets.MonthEnd())
    growth = pd.DataFrame({"A": 100 * 1.01 ** np.arange(16),
                           "B": np.full(16, 50.0)}, index=months)
    raw = momentum_raw(growth)
    # at t=12 (index 12): P(11)/P(0) - 1 = 1.01**11 - 1 ; the most recent
    # month (t=12 itself) is skipped by construction
    assert raw.iloc[12]["A"] == pytest.approx(1.01 ** 11 - 1)
    assert raw.iloc[12]["B"] == pytest.approx(0.0)
    assert raw.iloc[10].isna().all()  # not enough history yet


def test_standardise_zscores_each_month_independently():
    panel = pd.DataFrame(rng.normal(size=(4, 30)) * np.array([[1.0], [10.0], [100.0], [1000.0]]))
    z = standardise(panel)
    for _, row in z.iterrows():
        assert row.mean() == pytest.approx(0.0, abs=1e-9)
        assert row.std(ddof=1) == pytest.approx(1.0, abs=1e-9)
