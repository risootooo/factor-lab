# Data provenance

All raw data cached 2026-11-04 by `scripts/download_data.py`. The engine
(`src/factorlab/`) reads only this cache — delete it and rerun the script to
refresh.

| File | Source | Contents |
|---|---|---|
| `raw/nifty50_constituents.csv` | NSE Indices Ltd — `niftyindices.com/IndexConstituent/ind_nifty50list.csv`, fetched live 2026-11-04 | The 50 current NIFTY 50 constituents (symbol, company, industry, ISIN) |
| `raw/daily_prices.csv` | Yahoo Finance via yfinance 1.7.0, split/dividend-adjusted closes (`auto_adjust=True`) | Daily closes 2014-12-01 → 2026-11-03 for the 50 constituents (`.NS`) plus the NIFTY index (`^NSEI`) |
| `raw/fundamentals_long.csv` | Yahoo Finance via yfinance (`income_stmt`, `balance_sheet`) | Annual statement items (net income, EPS, avg shares, equity, total assets, debt) in long format; ~4 fiscal years per ticker (period ends 2021-12-31 → 2026-03-31) |
| `raw/info_snapshot.json` | Yahoo Finance `Ticker.info` | Current snapshot (trailing P/E, P/B, ROE, dividend yield, …) — kept for cross-checks only; the engine does **not** use it for historical factors |
| `raw/download_log.json` | The download run itself | Per-ticker failures, field coverage, first price date |

## Known data quirks (disclosed, not hidden)

- **No download failures**: all 50 tickers returned prices and statements.
- **Late listings** (price history starts at IPO): BSE (Feb 2017), SBI Life
  (Oct 2017), HDFC Life (Nov 2017), Max Healthcare (Aug 2020), Eternal/Zomato
  (Jul 2021), Jio Financial (Aug 2023). These stocks enter the cross-section
  only once they have enough history — the universe literally grows over time.
- **TMPV.NS** (Tata Motors Passenger Vehicles) carries Yahoo's continuous
  price history of the pre-demerger Tata Motors — a vendor convenience, not a
  tradable history for the demerged entity.
- **Statement depth is ~4 annual reports per ticker** (what Yahoo serves).
  With the +3-month reporting lag used by the engine, accounting-based
  factors (Value, Quality) have coverage only from ~mid-2023. Price-based
  factors (Momentum, Low-Vol) cover Jan 2016 → Oct 2026.
- Yahoo restates statements to the latest reported figures; the engine's
  reporting lag limits, but cannot eliminate, look-ahead in accounting data.
  See the README honesty section.
