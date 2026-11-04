"""Quintile long-short portfolio construction.

Timing contract (the anti-look-ahead rule of this project):
    scores formed at month-end t  ->  weights fixed at t  ->  earn the
    return of month t+1.  `factor_returns` enforces it by shifting the
    *return* panel, never the scores.

Costs: each rebalance, one-way turnover per book is
tau = 0.5 * sum |w_new - w_old|  (entering from cash counts tau = 1).
The charge is `cost_bps` on every unit of turnover of each book, deducted
from the long-short return: cost = cost_bps/1e4 * (tau_long + tau_short).
At the default 10 bps that is at most 20 bps in a month in which both
books are completely replaced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_BPS = 10.0


def quintile_weights(score_row: pd.Series, n_groups: int = 5):
    """Equal weights for (long_book, short_book) of one formation month.
    Long = top quintile of score, short = bottom quintile. Returns
    (w_long, w_short) as weight Series (each sums to 1 / -1... no: both are
    positive-magnitude books; w_short holds the shorted names with
    weights summing to 1 and is *subtracted* in the return math).
    Returns (None, None) if the cross-section is too small."""
    s = score_row.dropna()
    if len(s) < n_groups * 3:
        return None, None
    ranks = s.rank(method="first")
    q = pd.qcut(ranks, n_groups, labels=False)
    top = s[q == n_groups - 1]
    bot = s[q == 0]
    if len(top) == 0 or len(bot) == 0:
        return None, None
    w_long = pd.Series(0.0, index=score_row.index)
    w_short = pd.Series(0.0, index=score_row.index)
    w_long[top.index] = 1.0 / len(top)
    w_short[bot.index] = 1.0 / len(bot)
    return w_long, w_short


def factor_returns(scores: pd.DataFrame, monthly_ret: pd.DataFrame,
                   cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    """Monthly long-short returns for one factor.

    Returns a DataFrame indexed by the *return* month with columns:
    gross, cost, net, turnover_long, turnover_short, n_long, n_short, n_names.
    """
    rows = []
    prev_long = prev_short = None
    rets = monthly_ret.reindex(columns=scores.columns)
    for t, row in scores.iterrows():
        nxt = rets.index[rets.index > t]
        if len(nxt) == 0:
            break
        r_month = nxt[0]
        w_long, w_short = quintile_weights(row)
        if w_long is None:
            continue
        r = rets.loc[r_month].fillna(0.0)  # a name with no return that month
        # (not yet listed / suspended) contributes 0 — disclosed simplification
        gross = float((w_long * r).sum() - (w_short * r).sum())
        if prev_long is None:
            tau_l = tau_s = 1.0
        else:
            tau_l = 0.5 * float((w_long - prev_long).abs().sum())
            tau_s = 0.5 * float((w_short - prev_short).abs().sum())
        cost = cost_bps / 1e4 * (tau_l + tau_s)
        rows.append({"month": r_month, "gross": gross, "cost": cost,
                     "net": gross - cost, "turnover_long": tau_l,
                     "turnover_short": tau_s,
                     "n_long": int((w_long > 0).sum()),
                     "n_short": int((w_short > 0).sum()),
                     "n_names": int(row.notna().sum())})
        prev_long, prev_short = w_long, w_short
    if not rows:
        return pd.DataFrame(columns=["gross", "cost", "net", "turnover_long",
                                     "turnover_short", "n_long", "n_short", "n_names"])
    return pd.DataFrame(rows).set_index("month")


def all_factor_returns(scores: dict[str, pd.DataFrame], monthly_ret: pd.DataFrame,
                       cost_bps: float = DEFAULT_COST_BPS) -> dict[str, pd.DataFrame]:
    return {name: factor_returns(panel, monthly_ret, cost_bps)
            for name, panel in scores.items()}
