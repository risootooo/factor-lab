import numpy as np
import pandas as pd
import pytest

from factorlab.composite import bonferroni_bar, equal_weight_returns


def test_bonferroni_bar_tightens_with_more_tests():
    b1 = bonferroni_bar(1, 41)
    b4 = bonferroni_bar(4, 41)
    assert b4 > b1
    assert b1 == pytest.approx(2.02, abs=0.05)   # ~t(0.975, 40)
    assert b4 == pytest.approx(2.62, abs=0.07)   # ~t(1 - 0.00625, 40)


def test_equal_weight_returns_is_common_window_mean():
    idx = pd.date_range("2022-01-31", periods=5, freq=pd.offsets.MonthEnd())
    fa = pd.DataFrame({"net": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=idx)
    fb = pd.DataFrame({"net": [0.03, 0.02, 0.01, 0.00, -0.01]}, index=idx)
    fc = pd.DataFrame({"net": [0.02] * 4}, index=idx[1:])  # starts a month late
    out = equal_weight_returns({"a": fa, "b": fb, "c": fc}, names=["a", "b", "c"])
    assert len(out) == 4  # common window only
    assert out.iloc[0] == pytest.approx((0.02 + 0.02 + 0.02) / 3)
