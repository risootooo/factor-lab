"""Performance statistics, information coefficients and Fama-MacBeth
cross-sectional regressions.

Conventions: monthly frequency; annualised return = 12 x mean, annualised
vol = sqrt(12) x std, Sharpe = annualised return / annualised vol with a
0% risk-free rate (stated, so numbers are comparable across factors).
Newey-West standard errors use the Bartlett kernel with the conventional
lag rule L = floor(4 * (T/100)^(2/9)) (Newey & West 1994 plug-in).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def performance(monthly: pd.Series) -> dict:
    """Headline stats for a monthly return series."""
    r = monthly.dropna()
    n = len(r)
    if n == 0:
        return {"n_months": 0}
    ann_ret = 12 * r.mean()
    ann_vol = np.sqrt(12) * r.std(ddof=1) if n > 1 else np.nan
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {"n_months": n,
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol else np.nan,
            "max_drawdown": float(dd.min()),
            "hit_rate": float((r > 0).mean()),
            "worst_month": float(r.min()),
            "best_month": float(r.max()),
            "cum_return": float(equity.iloc[-1] - 1)}


def drawdown_window(monthly: pd.Series) -> dict:
    """Peak/trough dates of the maximum drawdown of a monthly series."""
    r = monthly.dropna()
    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return {"peak": peak, "trough": trough, "max_drawdown": float(dd.min())}


def information_coefficient(scores: pd.DataFrame, monthly_ret: pd.DataFrame,
                            min_names: int = 15) -> pd.Series:
    """Monthly Spearman IC: score at t vs return at t+1 (the tradable lag)."""
    rets = monthly_ret.reindex(columns=scores.columns)
    ics = {}
    for t, row in scores.iterrows():
        nxt = rets.index[rets.index > t]
        if len(nxt) == 0:
            break
        pair = pd.DataFrame({"s": row, "r": rets.loc[nxt[0]]}).dropna()
        if len(pair) >= min_names and pair["s"].std() > 0 and pair["r"].std() > 0:
            ic, _ = stats.spearmanr(pair["s"], pair["r"])
            ics[nxt[0]] = ic
    return pd.Series(ics, name="ic")


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    n = len(ic)
    if n < 3:
        return {"n": n, "mean_ic": np.nan, "ic_tstat": np.nan, "icir": np.nan}
    sd = ic.std(ddof=1)
    return {"n": n, "mean_ic": float(ic.mean()),
            "ic_tstat": float(ic.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan,
            "icir": float(ic.mean() / sd) if sd > 0 else np.nan,
            "pct_positive": float((ic > 0).mean())}


def _nw_se(series: np.ndarray, lags: int) -> float:
    """Newey-West standard error of the mean of `series`."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    t = len(x)
    if t < 3:
        return np.nan
    d = x - x.mean()
    var = float(d @ d) / t
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        cov = float(d[lag:] @ d[:-lag]) / t
        var += 2.0 * w * cov
    return float(np.sqrt(max(var, 0.0) / t))


def nw_lags(t: int) -> int:
    return int(np.floor(4 * (t / 100.0) ** (2.0 / 9.0)))


def fama_macbeth(scores_by_factor: dict[str, pd.DataFrame],
                 monthly_ret: pd.DataFrame, min_names: int = 20) -> pd.DataFrame:
    """Joint Fama-MacBeth: each month regress next-month returns on all
    factor z-scores (cross-sectional OLS, numpy), then average the slopes
    over time with Newey-West t-statistics.

    Returns a DataFrame indexed by factor with columns:
    mean_slope (per month, in return units per 1 z-score), ann_slope,
    nw_tstat, plain_tstat, pct_positive, n_months."""
    names = list(scores_by_factor)
    common_idx = scores_by_factor[names[0]].index
    for f in names[1:]:
        common_idx = common_idx.intersection(scores_by_factor[f].index)
    rets = monthly_ret.reindex(columns=scores_by_factor[names[0]].columns)
    slopes = []
    for t in common_idx:
        nxt = rets.index[rets.index > t]
        if len(nxt) == 0:
            break
        X = pd.DataFrame({f: scores_by_factor[f].loc[t] for f in names})
        y = rets.loc[nxt[0]]
        dfm = X.join(y.rename("y")).dropna()
        if len(dfm) < min_names:
            continue
        Xm = np.column_stack([np.ones(len(dfm)), dfm[names].to_numpy()])
        beta, *_ = np.linalg.lstsq(Xm, dfm["y"].to_numpy(), rcond=None)
        slopes.append(beta[1:])
    if not slopes:
        return pd.DataFrame()
    S = pd.DataFrame(slopes, columns=names)
    rows = {}
    for f in names:
        s = S[f].dropna()
        n = len(s)
        se_nw = _nw_se(s.to_numpy(), nw_lags(n))
        se_plain = s.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows[f] = {"mean_slope": float(s.mean()),
                   "ann_slope": float(12 * s.mean()),
                   "nw_tstat": float(s.mean() / se_nw) if se_nw else np.nan,
                   "plain_tstat": float(s.mean() / se_plain) if se_plain else np.nan,
                   "pct_positive": float((s > 0).mean()),
                   "n_months": n}
    return pd.DataFrame(rows).T
