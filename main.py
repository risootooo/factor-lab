#!/usr/bin/env python3
"""Factor Lab CLI — runs the full study OFFLINE from the cached data in
data/raw/ and prints the ranked factor table + the three cases. Also
rebuilds docs/results.json and docs/dashboard.html.

    python main.py            # cases + rebuild dashboard
    python main.py --no-build # cases only
    python main.py --cost 25  # rerun with 25 bps transaction costs
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from factorlab import composite, pipeline, report  # noqa: E402


def pct(x, d=1):
    return f"{100 * x:+.{d}f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--cost", type=float, default=10.0, help="bps per unit turnover")
    args = ap.parse_args()

    res = pipeline.run(cost_bps=args.cost)
    perf, ic, cases = res["perf"], res["ic"], res["cases"]
    fm, zoo = res["fm"], res["zoo"]

    print("=" * 88)
    print(f"FACTOR LAB — NIFTY 50 cross-sectional factors, net of {args.cost:.0f} bps costs")
    print("Universe: TODAY's NIFTY 50 backfilled (survivorship-biased — see README).")
    print("=" * 88)
    hdr = f"{'factor':<11}{'window':<22}{'Sharpe':>8}{'ann ret':>9}{'max DD':>9}{'IC':>8}{'IC t':>7}{'FM t':>7}"
    print(hdr)
    for name in ["value", "momentum", "quality", "lowvol", "composite", "composite_ew"]:
        fr = res["frets"].get(name)
        if name == "composite_ew":
            months = res["ew_returns"]
            window = f"{months.index.min():%Y-%m}..{months.index.max():%Y-%m}"
        elif fr is not None and len(fr):
            window = f"{fr.index.min():%Y-%m}..{fr.index.max():%Y-%m}"
        else:
            continue
        p = perf[name]
        fm_t = float(fm.loc[name, "nw_tstat"]) if name in getattr(fm, "index", []) else float("nan")
        icv = ic.get(name, {})
        fm_s = f"{fm_t:>+7.1f}" if fm_t == fm_t else "      —"
        ic_v, ic_t = icv.get('mean_ic', float('nan')), icv.get('ic_tstat', float('nan'))
        ic_s = f"{ic_v:>+8.3f}" if ic_v == ic_v else "       —"
        ict_s = f"{ic_t:>+7.1f}" if ic_t == ic_t else "      —"
        print(f"{name:<11}{window:<22}{p['sharpe']:>+8.2f}{pct(p['ann_return']):>9}"
              f"{pct(p['max_drawdown']):>9}{ic_s}{ict_s}{fm_s}")
    print(f"\nFactor return correlations (net):\n{res['corr'].round(2).to_string()}")
    print(f"\nFactor zoo: {int(zoo['survives_both'].sum())} of 4 factors survive BOTH "
          f"Bonferroni bars (FM and IC). Bars ~|t| > {float(zoo['bonferroni_bar'].iloc[0]):.2f}.")

    print("\n--- CASE A: ranking ---")
    print(cases["ranking"].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"Winner: {cases['winner']}   Loser: {cases['loser']}")

    dd = cases["momentum_dd"]
    print("\n--- CASE B: momentum crash anatomy ---")
    print(f"Max drawdown {pct(dd['max_drawdown'])}: {dd['peak']:%b %Y} -> {dd['trough']:%b %Y}")
    wm, wv = cases["momentum_worst_month"]
    print(f"Worst month {wm:%b %Y}: {pct(wv)}   | COVID window: " +
          ", ".join(f"{k[5:]} {pct(v)}" for k, v in cases["momentum_covid"].items()) +
          f"   | NIFTY worst COVID month {pct(cases['bench_covid_min'])}")

    print("\n--- CASE C: sub-period Sharpe instability (survivorship caveat inside) ---")
    print(cases["subperiod"].round(2).to_string())
    uvi = cases["universe_vs_index"]
    print(f"Survivorship in one number: EW of today's 50 backfilled = {pct(uvi['ew_ann_return'])}/yr "
          f"vs the real NIFTY index {pct(uvi['index_ann_return'])}/yr over {uvi['n_months']} months.")

    if not args.no_build:
        report.write_all(res)
        print("\nWrote docs/results.json and docs/dashboard.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
