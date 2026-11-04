# Case studies — what the data actually said

All numbers below are produced by `python main.py` from the cached data
(downloaded 2026-11-04), net of 10 bps transaction costs per unit of
turnover, risk-free rate 0%. Books are equal-weight quintile long-short
(top minus bottom quintile of the factor score), rebalanced monthly.

**Before any case: the universe is today's NIFTY 50 projected backwards.**
Read Case C for what that does to the numbers. Nothing in this file should
be quoted without that caveat.

---

## Case A — Which factor actually worked in India, 2016–2026?

| Rank | Factor | Window | Sharpe | Ann. return | Max DD | Mean IC | IC t-stat | Fama-MacBeth t |
|---|---|---|---|---|---|---|---|---|
| 1 | **Momentum** | Feb 2016 – Nov 2026 (130 mo) | **+0.25** | +4.9% | −43.4% | +0.020 | +1.0 | +1.2 |
| 2 | Value | Jul 2023 – Nov 2026 (41 mo) | +0.11 | +1.6% | −20.2% | +0.013 | +0.4 | +0.5 |
| 3 | Quality | Jul 2023 – Nov 2026 (41 mo) | −0.35 | −4.3% | −23.7% | −0.039 | −1.4 | −0.3 |
| 4 | **Low-Volatility** | Feb 2016 – Nov 2026 (130 mo) | **−0.95** | −18.3% | −89.1% | −0.042 | −2.2 | −2.0 |

Composites do not rescue the picture: the composite score book has Sharpe
**−0.62** over 2016–2026, and the equal-weight blend of the four books over
their common window (2023–2026) has Sharpe **−0.62** (−3.8%/yr).

**Winner: Momentum** — the only factor with a positive Sharpe, positive IC
and positive Fama-MacBeth slope (+0.30%/month per z-score). But its
t-statistics (≈1.0–1.2) are far from significance: over 130 months, this is
a suggestion, not a discovery.

**Loser: Low-Volatility — and it is not noise-shaped.** The long leg
(genuinely low-vol stocks) earned a Sharpe of 1.00; the high-vol leg earned
1.26. India paid for beta for a decade: long-low-vol/short-high-vol lost
money with the most negative IC (−0.042, t = −2.2) and FM slope
(−0.50%/month per z, t = −2.0) of any factor. The famous low-vol "anomaly"
is, in this market and window, an anti-premium.

Factor return correlations (net) are mostly low — Value/Quality −0.46,
Quality/Low-Vol +0.30, Momentum nearly orthogonal to all — which makes the
composite's failure more damning: diversification across books did not
create a premium that the ingredients lacked.

**The factor-zoo verdict:** 0 of 4 factors clear a Bonferroni-adjusted bar
(|t| ≳ 2.6) on *both* the Fama-MacBeth slope and the IC. Low-vol comes
closest to significance — in the wrong direction. After costs, in the
world's most-tested factor set, on India's flagship index, there is no
free lunch in this window. That is the finding.

Caveat on window asymmetry: Momentum and Low-Vol are measured over
2016–2026; Value and Quality only over 2023–2026 (statement availability).
The ranking compares each factor over its own measurable window — the
honest comparison available — and Case C shows why window choice matters.

---

## Case B — Momentum crash anatomy

Momentum's maximum drawdown is **−43.4%, from Feb 2020 to Jul 2023** — a
three-and-a-half-year bleed, not a single crash. The anatomy:

| Month | Momentum LS | Context |
|---|---|---|
| Feb 2020 | **+10.6%** | Pre-crash positioning pays as the fall begins |
| Mar 2020 | −2.1% | NIFTY falls −23.2%; the LS book cushions, barely loses |
| Apr 2020 | −13.7% | First snap-back: beaten-down losers rebound hardest |
| **Nov 2020** | **−23.4%** | **Worst month in the sample.** Vaccine news triggers the classic momentum crash: prior losers (banks, cyclicals) rip, prior winners (pharma, IT defensives) stall |
| 2021–2023 | slow bleed | Drawdown does not bottom until Jul 2023 |

Two lessons the numbers force:

1. **A long-short book cushions a crash; it does not escape one.** In the
   actual crash month (Mar 2020) momentum lost only −2.1% while the index
   lost −23.2%. The catastrophic losses came *after*, in the rotations —
   Apr 2020 and, above all, Nov 2020. Momentum risk is rebound risk.
2. **Drawdown duration is the real cost.** Peak-to-trough of 41 months
   means any investor (or allocator) running this book through 2020–2023
   faced three years of underwater performance after a decade of the
   factor "working". Sharpe +0.25 over the full window hides that path.

For comparison, momentum's worst month of the 2022 global rate-shock
rotation was Nov 2022 at −7.7% — painful, but a third of the Nov 2020
crash. The drawdown window, not any single month, is the story. (The full
monthly series for every book is in `docs/results.json`.)

---

## Case C — The survivorship confession, quantified

This study ranks stocks from **today's** NIFTY 50 (official NSE Indices
constituent file, fetched 2026-11-04) and trades them back to 2016.
Companies that were in the index in 2016 but were later dropped — typically
*after* poor performance — are absent. Companies that joined later enter
with their (successful) pre-inclusion history included. Both effects bias
every return in this repo **upward**, by an amount this dataset cannot
measure directly. Two things *can* be shown:

**1. The backfilled universe vs the real index.** An equal-weight
portfolio of exactly these 50 survivors, held from Jan 2016, returned
**+20.3%/yr**. The actual NIFTY 50 index — investable, with its real-time
constituent changes — returned **+11.4%/yr** over the same 131 months.
The ~9 pp/yr gap involves zero skill and zero factor timing; part is
equal-weight vs cap-weight, but a large part is simply that we *chose the
universe knowing who won*. Anyone quoting this repo's long-leg numbers
without this paragraph is misleading you.

**2. Sub-period instability** (Sharpe, net):

| Book | Full window | 2016–2020 | 2021–2026 |
|---|---|---|---|
| Momentum | +0.25 | +0.15 | +0.37 |
| Low-Vol | −0.95 | −0.60 | −1.34 |
| Composite (score) | −0.62 | −0.46 | −0.82 |
| Value | +0.11 | no coverage | +0.11 |
| Quality | −0.35 | no coverage | −0.35 |

Splitting the window does not rescue any factor: momentum improves in the
second half but stays insignificant; low-vol gets *worse*; the composite
degrades. If a factor's Sharpe flips or halves when the window moves, the
full-window number was a regime description, not a law of nature.

**What we did about it (and what we could not):** late listings (BSE 2017,
the life insurers 2017, Max Healthcare 2020, Eternal 2021, Jio Financial
2023) enter only when their own history exists — the cross-section grows
honestly. What no amount of code can fix from free data is the absence of
the *departed*: Wipro, BPCL, Britannia, Hero MotoCorp and IndusInd Bank
were index members for much of this window and are not in this universe at
all. A point-in-time constituent history (commercial data) is the correct
fix, and until then this confession stands at the top of the README.

---

*Generated from a real run on 2026-11-04 data. Reproduce: `python main.py`.
Not investment advice.*
