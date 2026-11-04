"""Monthly cross-sectional factor scores for the NIFTY 50 universe.

WHAT IS POINT-IN-TIME HERE — AND WHAT IS NOT
--------------------------------------------
* Momentum and Low-Volatility are functions of past prices only. A score
  dated month *t* uses prices up to and including the last trading day of
  month *t*, and is traded on month *t+1* returns. No look-ahead.
* Value and Quality use annual financial statements as served by Yahoo
  Finance. yfinance exposes only the latest ~4 annual reports per company
  and does not provide filing timestamps or as-originally-reported figures
  (statements are restated). We mitigate with a conservative **+3 month
  reporting lag**: the fiscal year ending 31 Mar becomes usable for scores
  dated from 30 Jun. That kills the crude look-ahead (using results before
  they could be public) but cannot remove restatement bias. Accounting
  factors therefore have coverage only from ~mid-2023 — an honest cost.
* The universe is **today's** NIFTY 50 held fixed backwards (survivorship
  bias; see README and docs/CASES.md Case C).

All raw components are winsorised cross-sectionally at the 1st/99th
percentile of each month and z-scored within the month. A stock needs a
valid price (and, for accounting factors, a usable statement) to be scored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPORTING_LAG_MONTHS = 3
MIN_NAMES = 10  # minimum cross-section size to z-score a month


# --------------------------------------------------------------------------
# cross-sectional helpers
# --------------------------------------------------------------------------
def winsorise(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Clip a cross-section to its [lo, hi] quantiles (NaNs untouched)."""
    x = s.dropna()
    if len(x) < 5:
        return s
    a, b = x.quantile(lo), x.quantile(hi)
    return s.clip(lower=a, upper=b)


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; a degenerate (zero-variance / tiny) section
    returns all-NaN rather than fake precision."""
    x = s.dropna()
    if len(x) < MIN_NAMES:
        return pd.Series(np.nan, index=s.index)
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - x.mean()) / sd


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (per month) winsorise + z-score of a raw factor panel."""
    return raw.apply(lambda row: zscore(winsorise(row)), axis=1)


# --------------------------------------------------------------------------
# price-based factors (fully point-in-time)
# --------------------------------------------------------------------------
def momentum_raw(month_end_px: pd.DataFrame, skip: int = 1, lookback: int = 12) -> pd.DataFrame:
    """12-1 momentum: P(t-skip) / P(t-skip-lookback+... ) — the standard
    Jegadeesh-Titman window: cumulative return from month t-12 to t-1.
    Both endpoints must be genuine observations (no forward-fill)."""
    end = month_end_px.shift(skip)
    start = month_end_px.shift(skip + lookback - 1)
    return end / start - 1.0


def lowvol_raw(daily_px: pd.DataFrame, month_ends: pd.Index,
               window: int = 252, min_obs: int = 120) -> pd.DataFrame:
    """Annualised volatility of daily returns over the trailing `window`
    trading days up to each month-end (rows aligned to `month_ends`)."""
    daily_ret = daily_px.pct_change(fill_method=None)
    roll = daily_ret.rolling(window, min_periods=min_obs).std() * np.sqrt(252)
    out = roll.reindex(month_ends, method="ffill")
    # a month-end is valid only if the last daily observation is close to it
    last_obs = daily_px.apply(lambda s: s.last_valid_index())
    valid = pd.DataFrame(
        {c: (month_ends - last_obs[c]).days <= 10 for c in daily_px.columns},
        index=month_ends)
    # rows where the trailing window had enough data but the stock has since
    # stopped trading are masked above; stale rows inside the window are
    # impossible because rolling std skips NaNs via min_periods on obs count
    out = out.where(valid)
    out.index = month_ends
    return out


# --------------------------------------------------------------------------
# accounting-based factors (statement-driven, +3m reporting lag)
# --------------------------------------------------------------------------
def fundamentals_asof(fund: pd.DataFrame, month_ends: pd.Index,
                      lag_months: int = REPORTING_LAG_MONTHS) -> dict[str, pd.DataFrame]:
    """Step each annual statement item onto the monthly grid: for month-end
    m, the value is the latest statement whose period_end + lag <= m.
    Returns {field: DataFrame(month x ticker)}."""
    fields = sorted(fund["field"].unique())
    out: dict[str, pd.DataFrame] = {}
    piv = {f: fund[fund.field == f].pivot_table(index="period_end", columns="ticker",
                                               values="value", aggfunc="last").sort_index()
           for f in fields}
    for f, p in piv.items():
        avail = p.copy()
        avail.index = p.index + pd.DateOffset(months=lag_months)
        merged = avail.reindex(avail.index.union(month_ends)).sort_index().ffill()
        out[f] = merged.reindex(month_ends)
    return out


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.div(b.replace(0, np.nan))


def value_raw(asof: dict[str, pd.DataFrame], month_end_px: pd.DataFrame) -> pd.DataFrame:
    """Raw value components: earnings yield (EPS/price) and book-to-market
    (book value per share / price). Components are z-scored separately and
    averaged; a stock needs at least one component."""
    eps = asof.get("diluted_eps")
    if eps is None:
        eps = asof.get("basic_eps")
    if eps is None:  # fall back to NI / diluted shares
        eps = _safe_div(asof["net_income"], asof["avg_shares_diluted"])
    shares = asof.get("avg_shares_diluted")
    if shares is None:
        shares = asof.get("avg_shares_basic")
    bvps = _safe_div(asof["equity"], shares)
    px = month_end_px.reindex(columns=eps.columns)
    ep = _safe_div(eps, px)
    bm = _safe_div(bvps, px)
    parts = [standardise(ep), standardise(bm)]
    stacked = pd.concat(parts)
    combined = stacked.groupby(level=0).mean()
    return combined.reindex(index=month_end_px.index, columns=month_end_px.columns)


def quality_raw(asof: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Raw quality: ROE (net income / average equity) and minus leverage
    (total debt / total assets). Z-scored separately, then averaged."""
    eq = asof["equity"]
    eq_avg = (eq + eq.shift(1)) / 2.0
    roe = _safe_div(asof["net_income"], eq_avg.where(eq_avg > 0))
    debt = asof.get("total_debt")
    if debt is None:
        debt = asof["lt_debt"].fillna(0) + asof["st_debt"].fillna(0)
    lev = _safe_div(debt, asof["total_assets"])
    parts = [standardise(roe), standardise(-lev)]
    stacked = pd.concat(parts)
    combined = stacked.groupby(level=0).mean()
    return combined.reindex(index=eq.index, columns=eq.columns)


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
FACTOR_NAMES = ["value", "momentum", "quality", "lowvol"]


def build_factor_scores(daily_px: pd.DataFrame, fund: pd.DataFrame,
                        start: str = "2016-01-01") -> dict[str, pd.DataFrame]:
    """Full pipeline. Returns {factor: z-score panel (month x ticker)} plus
    'composite' (mean of available factor z-scores, >= 2 factors required).
    Panels are indexed by formation month-end; the score formed at t is
    traded over month t+1 (see portfolios.py)."""
    me_px = daily_px.resample(pd.offsets.MonthEnd()).last()
    me_px = me_px[me_px.index >= "2015-01-01"]
    tickers = [c for c in daily_px.columns if c != "^NSEI"]
    me_px = me_px[tickers]

    mom = standardise(momentum_raw(me_px))
    lv = standardise(-lowvol_raw(daily_px[tickers], me_px.index))
    asof = fundamentals_asof(fund, me_px.index)
    val = value_raw(asof, me_px)
    qual = quality_raw(asof)

    scores = {"value": val, "momentum": mom, "quality": qual, "lowvol": lv}
    stacked = pd.concat([scores[f] for f in FACTOR_NAMES])
    count = sum(scores[f].notna() for f in FACTOR_NAMES)
    comp = stacked.groupby(level=0).mean()
    comp = comp.where(count >= 2)
    scores["composite"] = comp

    return {k: v[v.index >= start] for k, v in scores.items()}
