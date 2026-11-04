"""Build docs/results.json and the self-contained docs/dashboard.html."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

COLORS = {"value": "#f0b429", "momentum": "#38bdf8", "quality": "#a78bfa",
          "lowvol": "#34d399", "composite": "#fb7185", "composite_ew": "#fb7185"}


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def results_json(res: dict) -> dict:
    perf = {k: {kk: vv for kk, vv in v.items()} for k, v in res["perf"].items()}
    ic = {k: {kk: vv for kk, vv in v.items() if kk != "series"}
          for k, v in res["ic"].items()}
    monthly = {name: {str(k.date()): float(v) for k, v in fr["net"].items()}
               for name, fr in res["frets"].items() if len(fr)}
    out = {"performance_net": perf,
           "monthly_net_returns": monthly,
           "information_coefficients": ic,
           "fama_macbeth": res["fm"].to_dict("index") if len(res["fm"]) else {},
           "zoo_check": res["zoo"].to_dict("index"),
           "factor_correlation": res["corr"].to_dict(),
           "ranking": res["cases"]["ranking"].to_dict("records"),
           "momentum_drawdown": res["cases"]["momentum_dd"],
           "subperiod_sharpe": res["cases"]["subperiod"].to_dict("index"),
           "download_log": res["log"]}
    return _clean(out)


def _svg_curves(series_map: dict[str, pd.Series], w=880, h=340) -> str:
    """Cumulative-growth chart, log y-scale. Each series rebased to 1.0 at
    its own first observation."""
    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 26
    all_idx = sorted(set().union(*[set(s.index) for s in series_map.values()]))
    x0, x1 = 0, len(all_idx) - 1
    curves = {}
    lo, hi = np.inf, -np.inf
    for name, s in series_map.items():
        eq = (1 + s.dropna()).cumprod()
        pts = {d: v for d, v in eq.items()}
        curves[name] = pts
        lo = min(lo, np.log(eq.min())); hi = max(hi, np.log(eq.max()))
    lo = min(lo, 0.0); hi = max(hi, 0.05)
    if hi - lo < 0.2:
        hi = lo + 0.2

    def X(i):
        return pad_l + (w - pad_l - pad_r) * (i / max(x1, 1))

    def Y(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (np.log(v) - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for gv in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        if lo <= np.log(gv) <= hi:
            y = Y(gv)
            parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" '
                         f'stroke="#2b3446" stroke-width="1"/>'
                         f'<text x="4" y="{y+4:.1f}" fill="#8b98b0" font-size="11">{gv:g}x</text>')
    for i, d in enumerate(all_idx):
        if i % 24 == 0:
            parts.append(f'<text x="{X(i):.0f}" y="{h-6}" fill="#8b98b0" font-size="11" '
                         f'text-anchor="middle">{pd.Timestamp(d).strftime("%Y")}</text>')
    for name, pts in curves.items():
        xy = " ".join(f"{X(all_idx.index(d)):.1f},{Y(v):.1f}" for d, v in pts.items())
        parts.append(f'<polyline points="{xy}" fill="none" stroke="{COLORS[name]}" '
                     f'stroke-width="2"/>')
    lx = pad_l + 6
    for name in curves:
        parts.append(f'<rect x="{lx}" y="6" width="10" height="10" fill="{COLORS[name]}"/>'
                     f'<text x="{lx+14}" y="15" fill="#dbe4f3" font-size="12">{name}</text>')
        lx += 14 + 9 * len(name) + 26
    parts.append("</svg>")
    return "".join(parts)


def _svg_ic_bars(ic_stats: dict, w=880, h=240) -> str:
    names = ["value", "momentum", "quality", "lowvol", "composite"]
    vals = {n: ic_stats[n]["mean_ic"] for n in names}
    mx = max(0.05, max(abs(v) for v in vals.values() if np.isfinite(v)))
    pad_l, pad_b, pad_t = 46, 24, 12
    mid = pad_t + (h - pad_t - pad_b) / 2
    scale = (h - pad_t - pad_b) / 2 / mx
    bw = (w - pad_l - 20) / len(names) * 0.55
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">',
             f'<line x1="{pad_l}" y1="{mid}" x2="{w-10}" y2="{mid}" stroke="#2b3446"/>']
    for i, n in enumerate(names):
        v = vals[n]
        cx = pad_l + 20 + (i + 0.5) * (w - pad_l - 20) / len(names)
        y = mid - v * scale
        parts.append(f'<rect x="{cx-bw/2:.0f}" y="{min(y, mid):.0f}" width="{bw:.0f}" '
                     f'height="{abs(v*scale):.0f}" fill="{COLORS[n]}"/>')
        parts.append(f'<text x="{cx:.0f}" y="{(y-6 if v >= 0 else y+abs(v*scale)+14):.0f}" '
                     f'fill="#dbe4f3" font-size="12" text-anchor="middle">{v:+.3f}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{h-6}" fill="#8b98b0" font-size="12" '
                     f'text-anchor="middle">{n} (t={ic_stats[n]["ic_tstat"]:+.1f})</text>')
    parts.append("</svg>")
    return "".join(parts)


def _corr_table(corr: pd.DataFrame) -> str:
    def cell(v):
        if pd.isna(v):
            return '<td class="num">—</td>'
        a = abs(v)
        bg = f"rgba(56,189,248,{0.15 + 0.55 * a:.2f})" if v >= 0 else f"rgba(251,113,133,{0.15 + 0.55 * a:.2f})"
        return f'<td class="num" style="background:{bg}">{v:+.2f}</td>'
    head = "".join(f"<th>{c}</th>" for c in corr.columns)
    rows = "".join("<tr><th>" + idx + "</th>" + "".join(cell(v) for v in row) + "</tr>"
                   for idx, row in corr.iterrows())
    return f"<table><tr><th></th>{head}</tr>{rows}</table>"


def _fmt_pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:+.{digits}f}%"


def dashboard_html(res: dict) -> str:
    perf, ic = res["perf"], res["ic"]
    cases = res["cases"]
    books = {n: res["frets"][n]["net"] for n in ["value", "momentum", "quality", "lowvol"]}
    books["composite"] = res["frets"]["composite"]["net"]
    curves_svg = _svg_curves(books)
    ic_svg = _svg_ic_bars(ic)
    rank = cases["ranking"]
    rank_rows = "".join(
        f'<tr><td><span class="dot" style="background:{COLORS[r.factor]}"></span>{r.factor}</td>'
        f'<td class="num">{r.sharpe:+.2f}</td><td class="num">{_fmt_pct(r.ann_return)}</td>'
        f'<td class="num">{_fmt_pct(r.max_drawdown)}</td><td class="num">{r.mean_ic:+.3f}</td>'
        f'<td class="num">{r.ic_tstat:+.1f}</td><td class="num">{r.fm_tstat:+.1f}</td></tr>'
        for r in rank.itertuples())
    sub = cases["subperiod"]
    sub_rows = "".join(
        f'<tr><td>{f}</td><td class="num">{sub.loc[f, "full"]:+.2f}</td>'
        f'<td class="num">{sub.loc[f, "2016_2020"]:+.2f}</td>'
        f'<td class="num">{sub.loc[f, "2021_2026"]:+.2f}</td>'
        f'<td class="num">{int(sub.loc[f, "n_full"])}</td></tr>'
        for f in ["momentum", "lowvol", "composite", "value", "quality"]
        if f in sub.index and np.isfinite(sub.loc[f, "full"]))
    dd = cases["momentum_dd"]
    ew_perf = perf["composite_ew"]
    zoo = res["zoo"]
    n_surv = int(zoo["survives_both"].sum())
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Factor Lab — NIFTY 50 factor research</title>
<style>
 body{{background:#0b0f17;color:#dbe4f3;font-family:ui-sans-serif,system-ui,'Segoe UI',Roboto,sans-serif;margin:0;padding:28px 4vw}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:17px;margin:26px 0 8px;color:#aebadd}}
 p.sub{{color:#8b98b0;margin:0 0 10px}} .card{{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px 18px;margin:14px 0}}
 table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:6px 8px;border-bottom:1px solid #1f2937;text-align:left}}
 td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} td.num{{text-align:right}}
 .warn{{border-color:#7c2d12;background:#1a1207}} .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
 .big{{font-size:20px;font-weight:700}} footer{{color:#64748b;font-size:12px;margin-top:26px}}
</style></head><body>
<h1>Factor Lab — what actually worked in Indian large caps</h1>
<p class="sub">Quintile long-short factor books on the NIFTY 50 universe, net of 10 bps/trade costs.
Research, not investment advice. Read the survivorship warning below before believing any number.</p>

<div class="card warn"><b>Survivorship warning.</b> The universe is <b>today's</b> NIFTY 50 projected
backwards to 2016. Companies that were dropped from the index after poor performance are missing,
so every return here is biased upward by an amount this dataset cannot measure. Value &amp; Quality
additionally rest on restated statements (Yahoo serves ~4 annual reports; a +3-month reporting lag
is applied). Momentum &amp; Low-Vol are fully point-in-time in prices.</div>

<div class="card"><h2>Cumulative growth of ₹1 — long-short books (log scale)</h2>{curves_svg}
<p class="sub">Books start when their factor has data: Momentum &amp; Low-Vol in 2016, Value &amp; Quality in 2023.
The charted composite re-ranks stocks on the average z-score. (Not charted: the equal-weight blend of the
four books' returns over their common 2023+ window, Sharpe {ew_perf['sharpe']:+.2f}.)</p></div>

<div class="card"><h2>Case A — the ranking (net of costs)</h2>
<table><tr><th>Factor</th><th class="num">Sharpe</th><th class="num">Ann. ret</th><th class="num">Max DD</th>
<th class="num">Mean IC</th><th class="num">IC t</th><th class="num">Fama-MacBeth t</th></tr>{rank_rows}</table>
<p class="sub">Winner: <b>{cases['winner']}</b>. Loser: <b>{cases['loser']}</b>. Fama-MacBeth is a joint regression
on the window where all four factors exist; IC is computed over each factor's own window.</p></div>

<div class="card"><h2>Mean monthly information coefficient</h2>{ic_svg}</div>

<div class="card"><h2>Case B — momentum crash anatomy</h2>
<p>Maximum drawdown <span class="big">{_fmt_pct(dd['max_drawdown'])}</span> from
{pd.Timestamp(dd['peak']).strftime('%b %Y')} to {pd.Timestamp(dd['trough']).strftime('%b %Y')}.
Worst single month: {pd.Timestamp(cases['momentum_worst_month'][0]).strftime('%b %Y')}
({_fmt_pct(cases['momentum_worst_month'][1])}). The NIFTY index's worst month in Feb–Apr 2020 was
{_fmt_pct(cases['bench_covid_min'])} — a long-short book cushions a crash, it does not escape one.</p></div>

<div class="card"><h2>Factor return correlations (net)</h2>{_corr_table(res['corr'])}</div>

<div class="card"><h2>Case C — sub-period instability (Sharpe)</h2>
<table><tr><th>Book</th><th class="num">Full</th><th class="num">2016–2020</th><th class="num">2021–2026</th><th class="num">Months</th></tr>{sub_rows}</table>
<p class="sub">Factors are regimes, not constants: the same book can halve or flip between decades.
Survivorship, quantified: an equal-weight portfolio of today's 50 constituents, backfilled,
returned {_fmt_pct(cases['universe_vs_index']['ew_ann_return'])}/yr vs the real, investable NIFTY
index at {_fmt_pct(cases['universe_vs_index']['index_ann_return'])}/yr over the same months.
That gap is the bias — no skill involved. Treat every long-short Sharpe here as an upper bound too.</p></div>

<div class="card"><h2>Factor zoo check</h2>
<p><b>{n_surv} of 4</b> factors clear a Bonferroni-adjusted bar on <i>both</i> the Fama-MacBeth
slope t-stat and the IC t-stat. With four factors tested, raw 5% significance is not enough —
this is the multiple-testing tax on factor mining.</p></div>

<footer>Factor Lab · data: Yahoo Finance via yfinance, cached 2026-11-04 · universe: NSE Indices
official constituent file of the same date · generated offline by report.py</footer>
</body></html>"""


def write_all(res: dict) -> None:
    DOCS.mkdir(exist_ok=True)
    with open(DOCS / "results.json", "w") as fh:
        json.dump(results_json(res), fh, indent=1, default=str)
    with open(DOCS / "dashboard.html", "w") as fh:
        fh.write(dashboard_html(res))
