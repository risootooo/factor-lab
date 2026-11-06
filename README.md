# Factor Lab — Cross-Sectional Factor Research on the NIFTY 50

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![tests](https://img.shields.io/badge/tests-24%20passing-brightgreen) ![data](https://img.shields.io/badge/data-Yahoo%20Finance%20(real)-orange) ![license](https://img.shields.io/badge/license-MIT-lightgrey) ![release](https://img.shields.io/github/v/release/risootooo/factor-lab)

**Four classic equity factors — Value, Momentum, Quality, Low-Volatility — built Fama-French style on real Indian large-cap data, run through quintile long-short portfolios, IC analysis and Fama-MacBeth regressions. Headline result: after costs and a Bonferroni correction, *none* of them survives in the NIFTY 50 over 2016–2026 — and Low-Volatility, the textbook anomaly, is a significantly *negative* premium in India. This repo publishes that, with the receipts.**

```bash
pip install -r requirements.txt   # numpy, pandas, scipy (data already cached)
python main.py                    # ranked factor table + Cases A/B/C, fully offline
python -m pytest tests/           # 24 tests, incl. a structural no-look-ahead test
```

Dashboard: [`docs/dashboard.html`](docs/dashboard.html) (self-contained, no network) · Cases: [`docs/CASES.md`](docs/CASES.md) · Raw numbers: [`docs/results.json`](docs/results.json)

---

## ⚠️ The honesty section (read this before any number)

This section is a feature of the project, not fine print.

**1. Survivorship bias — the big one.** The universe is **today's** NIFTY 50
(official NSE Indices constituent file, fetched 2026-11-04) projected
backwards to 2016. Stocks dropped from the index after poor performance are
missing; later joiners bring their winning histories with them. Every return
in this repo is biased **upward** by an amount the data cannot measure.
Quantified as far as possible in Case C: an equal-weight portfolio of these
50 survivors returned **+20.3%/yr** from 2016 while the real, investable
NIFTY 50 index returned **+11.4%/yr**. That gap is not skill. A point-in-time
constituent history (commercial data) is the proper fix.

**2. What is point-in-time and what is not.**
- *Momentum* and *Low-Volatility* are functions of past prices only — fully PIT. A score dated month *t* uses prices up to month-*t*'s last trading day and earns month *t+1*'s return. A structural test in the suite verifies the portfolio engine cannot see same-month returns.
- *Value* and *Quality* use annual financial statements as served by Yahoo Finance: only the latest ~4 reports per company, restated, with no filing dates. Mitigation: a conservative **+3-month reporting lag** after each fiscal year-end before a statement may influence a score. That removes crude look-ahead, not restatement bias, and it limits accounting-factor coverage to **Jul 2023 onwards**.
- Late listings (BSE 2017, SBI Life/HDFC Life 2017, Max Healthcare 2020, Eternal 2021, Jio Financial 2023) enter the cross-section only once listed — the universe grows from ~44 to 50 names honestly. TMPV.NS inherits pre-demerger Tata Motors prices from Yahoo; disclosed in `data/PROVENANCE.md`.

**3. Vendors restate, merge and rename.** HDFC Bank's history spans the HDFC Ltd merger; Eternal is the renamed Zomato. The cache and its provenance are committed so every number is reproducible against a fixed snapshot.

---

## The ranked table (real run, net of 10 bps costs, Rf = 0%)

| Factor | Window | Sharpe | Ann. ret | Max DD | Mean IC | IC t | Fama-MacBeth t |
|---|---|---|---|---|---|---|---|
| Momentum | 2016–2026 (130 mo) | **+0.25** | +4.9% | −43.4% | +0.020 | +1.0 | +1.2 |
| Value | 2023–2026 (41 mo) | +0.11 | +1.6% | −20.2% | +0.013 | +0.4 | +0.5 |
| Quality | 2023–2026 (41 mo) | −0.35 | −4.3% | −23.7% | −0.039 | −1.4 | −0.3 |
| Low-Volatility | 2016–2026 (130 mo) | **−0.95** | −18.3% | −89.1% | −0.042 | −2.2 | −2.0 |
| Composite (score) | 2016–2026 | −0.62 | −12.1% | −78.8% | −0.016 | −0.8 | — |
| Composite (EW blend, common window) | 2023–2026 | −0.62 | −3.8% | −12.3% | — | — | — |

**Factor zoo check:** 0 of 4 factors clear the Bonferroni bar (|t| ≳ 2.6 for 4 tests) on *both* the Fama-MacBeth slope and the IC. Low-volatility comes closest to significance — with the wrong sign.

## Methodology

1. **Signals** (monthly, cross-sectional): Value = mean z-score of earnings yield (diluted EPS ÷ price) and book-to-market (BVPS ÷ price). Momentum = 12-1 month return. Quality = mean z of ROE (NI ÷ average equity) and −(total debt ÷ total assets). Low-Vol = −(annualised σ of daily returns, trailing 252 trading days, ≥120 obs). Composite = mean of available factor z-scores (≥2 required).
2. **Standardisation**: each raw component is winsorised at the month's 1st/99th percentiles, then z-scored within the month (min 10 names).
3. **Portfolios**: equal-weight quintile long-short (top − bottom), rebalanced monthly, formed on month-*t* scores and held over *t+1*. Transaction costs: 10 bps per unit of one-way turnover per book (configurable: `python main.py --cost 25`).
4. **Statistics**: Sharpe from monthly net returns (Rf = 0); Spearman IC of score vs next-month return; Fama-MacBeth — monthly cross-sectional OLS of next-month returns on the four z-scores jointly, slopes averaged over time with Newey-West t-stats (Bartlett kernel, lag rule ⌊4(T/100)^(2/9)⌋).
5. **Multiple testing**: Bonferroni across the 4 factors on both the FM and IC lenses (`composite.zoo_check`).

## Cases

- **Case A** — the ranking above, with leg decompositions: low-vol's long leg Sharpe is 1.00 and its high-vol leg 1.26 — India paid for beta.
- **Case B** — momentum's crash is a *rotation*, not the COVID crash: Mar 2020 only −2.1% (index −23.2%), but Nov 2020 −23.4% (vaccine rotation), and a −43.4% peak-to-trough bleed from Feb 2020 to Jul 2023.
- **Case C** — the survivorship confession, quantified (above), plus sub-period Sharpe splits showing no factor is stable across 2016–2020 vs 2021–2026.

Full write-ups in [`docs/CASES.md`](docs/CASES.md).

## Layout

```
main.py                  CLI: cases + rebuilds docs/results.json & dashboard
scripts/download_data.py the only network code — refreshes data/raw/
data/raw/                cached prices, statements, constituents, provenance
src/factorlab/
  dataio.py              cache loaders, month-end panels
  factors.py             raw signals, winsorise/z-score, statement lag logic
  portfolios.py          quintile books, turnover, costs (timing contract)
  analysis.py            performance, IC, Newey-West, Fama-MacBeth
  composite.py           EW blend, correlations, Bonferroni zoo check
  pipeline.py            end-to-end run + case material
  report.py              results.json + self-contained dashboard.html
tests/                   20 tests (see below)
```

## Tests

24 tests, all passing (`/tmp/venv-test/bin/python -m pytest tests/ -q`, or any env with numpy/pandas/scipy/pytest): winsorise/z-score maths on toy panels, quintile assignment, IC recovery on planted signal, Fama-MacBeth recovery of a known slope, Newey-West ≥ plain SE on autocorrelated slopes, cost-drag monotonicity, score-panel invariants, and the **no-look-ahead suite**: returns dated on/before a formation month can be multiplied by 100 without changing any booked result; a same-month return trap is ignored; the booked gross equals the hand-computed next-month weighted return to the paisa; relabelling time shifts the IC series by exactly one month; and on data planted so next-month returns follow the score while same-month returns anti-follow it, the engine's IC is positive while the same-month pairing is negative.

## Limitations

- Survivorship bias (above) — the dominant one; magnitudes here are upper bounds, especially for long legs.
- Value/Quality windows are short (41 months) because free statement history is ~4 annual reports; their t-stats are indicative only.
- Yahoo statement fields vary by company type (banks vs non-banks); some debt fields are incomplete for insurers — recorded per-ticker in `data/raw/download_log.json`.
- Quintile books on ~50 names hold ~10 stocks per leg; idiosyncratic noise is large and no risk model is applied.
- Costs are a flat 10 bps on turnover; real impact in the least liquid names would be higher.

## Data provenance

Yahoo Finance via yfinance 1.7.0; universe from NSE Indices Ltd's official
constituent CSV; all cached 2026-11-04 with per-ticker logs in
`data/raw/` (`data/PROVENANCE.md`). No failed tickers in the download —
every one of the 50 constituents returned prices and statements.

## Disclaimer

Research and education only. Factor premia are historical descriptions, not
promises; past performance does not predict future returns. Not investment
advice.

*By Samrat Banerjee ([risootooo](https://github.com/risootooo)) — part of a quantitative-finance project series: EdgeGate, GreeksLab, VolSurface Atlas, StatArb Lab, Credit Scorecard Pro, Valuation Lab, VaR Risk Engine.*
