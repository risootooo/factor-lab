"""End-to-end run: cached data -> factors -> portfolios -> statistics ->
case material. Pure computation; presentation lives in report.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis, composite, dataio, factors, portfolios

ALL_BOOKS = ["value", "momentum", "quality", "lowvol", "composite"]


def run(cost_bps: float = portfolios.DEFAULT_COST_BPS) -> dict:
    daily = dataio.load_daily_prices()
    fund = dataio.load_fundamentals()
    cons = dataio.load_constituents()
    log = dataio.load_download_log()

    tickers = [c for c in daily.columns if c != dataio.INDEX_TICKER]
    me_px = dataio.month_end_prices(daily)
    monthly_ret = dataio.monthly_returns(me_px)
    bench_ret = monthly_ret[dataio.INDEX_TICKER].rename("nifty50_index")

    scores = factors.build_factor_scores(daily, fund)
    frets = portfolios.all_factor_returns(scores, monthly_ret[tickers], cost_bps)

    # equal-weight return blend over the four base factors
    ew = composite.equal_weight_returns(frets)
    ew_df = pd.DataFrame({"gross": ew, "cost": 0.0, "net": ew})
    frets_ext = dict(frets)
    frets_ext["composite_ew"] = ew_df

    perf, ic_stats = {}, {}
    for name, fr in frets.items():
        if len(fr):
            perf[name] = analysis.performance(fr["net"])
            perf[name]["gross_sharpe"] = analysis.performance(fr["gross"])["sharpe"]
            perf[name]["mean_turnover"] = float(
                (fr["turnover_long"] + fr["turnover_short"]).mean())
            perf[name]["avg_names"] = float(fr["n_names"].mean())
        ic = analysis.information_coefficient(scores[name], monthly_ret[tickers])
        ic_stats[name] = analysis.ic_summary(ic)
        ic_stats[name]["series"] = ic
    perf["composite_ew"] = analysis.performance(ew)

    corr = composite.correlation_matrix(frets)
    fm = analysis.fama_macbeth({f: scores[f] for f in composite.BASE_FACTORS},
                               monthly_ret[tickers])
    zoo = composite.zoo_check(fm, {k: v for k, v in ic_stats.items()})

    mom = frets["momentum"]["net"]
    cases = build_cases(frets, perf, ic_stats, fm, zoo, mom, bench_ret,
                        monthly_ret, tickers)
    return {"perf": perf, "ic": ic_stats, "corr": corr, "fm": fm, "zoo": zoo,
            "frets": frets, "ew_returns": ew, "scores": scores,
            "bench_ret": bench_ret, "cases": cases, "cons": cons, "log": log,
            "monthly_ret": monthly_ret, "tickers": tickers}


def subperiod_sharpe(series: pd.Series, start: str, end: str) -> float:
    sub = series[(series.index >= start) & (series.index <= end)]
    if len(sub) < 12:
        return np.nan
    return analysis.performance(sub)["sharpe"]


def build_cases(frets, perf, ic_stats, fm, zoo, mom, bench_ret,
                monthly_ret, tickers) -> dict:
    cases: dict = {}
    # Case A — ranking of the real factors
    table = []
    for f in composite.BASE_FACTORS:
        table.append({"factor": f,
                      "sharpe": perf.get(f, {}).get("sharpe"),
                      "ann_return": perf.get(f, {}).get("ann_return"),
                      "max_drawdown": perf.get(f, {}).get("max_drawdown"),
                      "mean_ic": ic_stats[f]["mean_ic"],
                      "ic_tstat": ic_stats[f]["ic_tstat"],
                      "fm_tstat": float(fm.loc[f, "nw_tstat"]) if len(fm) else np.nan})
    rank = pd.DataFrame(table).sort_values("sharpe", ascending=False)
    cases["ranking"] = rank
    cases["winner"] = rank.iloc[0]["factor"]
    cases["loser"] = rank.iloc[-1]["factor"]

    # Case B — momentum crash anatomy
    dd = analysis.drawdown_window(mom)
    worst = mom.idxmin()
    covid = mom[(mom.index >= "2020-02-01") & (mom.index <= "2020-04-30")]
    cases["momentum_dd"] = dd
    cases["momentum_worst_month"] = (worst, float(mom.min()))
    cases["momentum_2022_min"] = float(mom[mom.index.year == 2022].min())
    cases["momentum_covid"] = {str(k.date()): float(v) for k, v in covid.items()}
    cases["bench_covid_min"] = float(bench_ret[(bench_ret.index >= "2020-02-01")
                                               & (bench_ret.index <= "2020-04-30")].min())

    # Case C — sub-period instability of the headline (momentum) factor and
    # of every other factor where coverage allows
    sub = {}
    for f in ALL_BOOKS:
        s = frets[f]["net"] if f != "composite" else frets["composite"]["net"]
        sub[f] = {"full": analysis.performance(s)["sharpe"],
                  "2016_2020": subperiod_sharpe(s, "2016-01-01", "2020-12-31"),
                  "2021_2026": subperiod_sharpe(s, "2021-01-01", "2026-12-31"),
                  "n_full": len(s)}
    cases["subperiod"] = pd.DataFrame(sub).T

    # Case C, quantified: an equal-weight portfolio of TODAY's constituents
    # (the backfilled universe itself) vs the actual investable index.
    ew_univ = monthly_ret[tickers].mean(axis=1)
    both = pd.DataFrame({"ew": ew_univ, "idx": bench_ret})
    both = both[both.index >= "2016-01-01"].dropna()
    cases["universe_vs_index"] = {
        "ew_ann_return": analysis.performance(both["ew"])["ann_return"],
        "index_ann_return": analysis.performance(both["idx"])["ann_return"],
        "n_months": len(both)}
    return cases
