"""Equal-weight composite, factor correlations, and the factor-zoo
multiple-testing reality check.

Four factors were tried. If you test enough factors at the 5% level, some
will look significant by luck — Harvey, Liu & Zhu (2016) is the canonical
warning. The simple lens used here: Bonferroni. With 4 tests, a per-test
two-sided alpha of 0.05/4 = 0.0125 replaces 0.05, i.e. |t| must clear the
corresponding critical value before we call a factor "surviving".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis

BASE_FACTORS = ["value", "momentum", "quality", "lowvol"]


def equal_weight_returns(factor_rets: dict[str, pd.DataFrame],
                         names: list[str] = BASE_FACTORS) -> pd.Series:
    """Monthly return of an equal-weight blend of the (net) factor
    long-short books, rebalanced monthly — the average of the books'
    returns over their common window."""
    nets = pd.DataFrame({n: factor_rets[n]["net"] for n in names})
    return nets.mean(axis=1, skipna=False).dropna().rename("composite_ew")


def correlation_matrix(factor_rets: dict[str, pd.DataFrame],
                       names: list[str] = BASE_FACTORS) -> pd.DataFrame:
    nets = pd.DataFrame({n: factor_rets[n]["net"] for n in names})
    return nets.corr()


def bonferroni_bar(n_tests: int, n_obs: int, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni critical |t| (t-distribution, n_obs-1 df)."""
    return float(stats.t.ppf(1 - (alpha / n_tests) / 2, df=max(n_obs - 1, 1)))


def zoo_check(fm: pd.DataFrame, ic_stats: dict[str, dict]) -> pd.DataFrame:
    """Score each factor against both significance lenses: Fama-MacBeth
    slope t-stats and IC t-stats, at 5% and at the Bonferroni bar."""
    rows = {}
    for f in BASE_FACTORS:
        row = {"fm_tstat": np.nan, "ic_tstat": np.nan, "n_months": np.nan}
        if f in fm.index and len(fm):
            row["fm_tstat"] = float(fm.loc[f, "nw_tstat"])
            row["n_months"] = int(fm.loc[f, "n_months"])
        if f in ic_stats:
            row["ic_tstat"] = float(ic_stats[f]["ic_tstat"])
        n = row["n_months"] if not np.isnan(row["n_months"]) else 60
        bar = bonferroni_bar(len(BASE_FACTORS), int(n))
        row["bonferroni_bar"] = bar
        row["survives_fm"] = bool(abs(row["fm_tstat"]) > bar) if not np.isnan(row["fm_tstat"]) else False
        row["survives_ic"] = bool(abs(row["ic_tstat"]) > bar) if not np.isnan(row["ic_tstat"]) else False
        row["survives_both"] = row["survives_fm"] and row["survives_ic"]
        rows[f] = row
    out = pd.DataFrame(rows).T
    out.attrs["n_surviving"] = int(out["survives_both"].sum())
    return out
