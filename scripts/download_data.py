#!/usr/bin/env python3
"""Download ALL raw data for Factor Lab from Yahoo Finance (via yfinance)
and the official NSE Indices constituent file, caching everything into
data/raw/ so the research engine runs fully offline afterwards.

    python scripts/download_data.py            # full download
    python scripts/download_data.py --prices-only

Outputs (data/raw/):
    nifty50_constituents.csv   official constituent list (symbol/company/industry)
    daily_prices.csv           daily adjusted closes, constituents + ^NSEI
    fundamentals_long.csv      annual statement items, long format
    info_snapshot.json         current .info snapshot per ticker (cross-checks)
    download_log.json          per-ticker failures and coverage notes
    PROVENANCE.md              written by hand alongside this script's output

IMPORTANT (survivorship): the universe is TODAY's NIFTY 50. See README.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from datetime import date

import pandas as pd

ROOT = __file__.rsplit("/scripts/", 1)[0]
RAW = ROOT + "/data/raw"
LIST_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
START = "2014-12-01"  # prices from Dec-2014 -> 12-1 momentum usable from Jan-2016

# Fallback copy of the official list exactly as fetched from NSE Indices on
# 2026-11-04 (used only if the live file is unreachable at download time).
FALLBACK = [
    ("ADANIENT", "Adani Enterprises Ltd."), ("ADANIPORTS", "Adani Ports and Special Economic Zone Ltd."),
    ("APOLLOHOSP", "Apollo Hospitals Enterprise Ltd."), ("ASIANPAINT", "Asian Paints Ltd."),
    ("AXISBANK", "Axis Bank Ltd."), ("BAJAJ-AUTO", "Bajaj Auto Ltd."),
    ("BAJAJFINSV", "Bajaj Finserv Ltd."), ("BAJFINANCE", "Bajaj Finance Ltd."),
    ("BEL", "Bharat Electronics Ltd."), ("BHARTIARTL", "Bharti Airtel Ltd."),
    ("BSE", "BSE Ltd."), ("CIPLA", "Cipla Ltd."), ("COALINDIA", "Coal India Ltd."),
    ("DRREDDY", "Dr. Reddy's Laboratories Ltd."), ("EICHERMOT", "Eicher Motors Ltd."),
    ("ETERNAL", "Eternal Ltd."), ("GRASIM", "Grasim Industries Ltd."),
    ("HCLTECH", "HCL Technologies Ltd."), ("HDFCBANK", "HDFC Bank Ltd."),
    ("HDFCLIFE", "HDFC Life Insurance Company Ltd."), ("HINDALCO", "Hindalco Industries Ltd."),
    ("HINDUNILVR", "Hindustan Unilever Ltd."), ("ICICIBANK", "ICICI Bank Ltd."),
    ("INDIGO", "InterGlobe Aviation Ltd."), ("INFY", "Infosys Ltd."), ("ITC", "ITC Ltd."),
    ("JIOFIN", "Jio Financial Services Ltd."), ("JSWSTEEL", "JSW Steel Ltd."),
    ("KOTAKBANK", "Kotak Mahindra Bank Ltd."), ("LT", "Larsen & Toubro Ltd."),
    ("M&M", "Mahindra & Mahindra Ltd."), ("MARUTI", "Maruti Suzuki India Ltd."),
    ("MAXHEALTH", "Max Healthcare Institute Ltd."), ("NESTLEIND", "Nestle India Ltd."),
    ("NTPC", "NTPC Ltd."), ("ONGC", "Oil & Natural Gas Corporation Ltd."),
    ("POWERGRID", "Power Grid Corporation of India Ltd."), ("RELIANCE", "Reliance Industries Ltd."),
    ("SBILIFE", "SBI Life Insurance Company Ltd."), ("SBIN", "State Bank of India"),
    ("SHRIRAMFIN", "Shriram Finance Ltd."), ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd."),
    ("TATACONSUM", "Tata Consumer Products Ltd."), ("TATASTEEL", "Tata Steel Ltd."),
    ("TCS", "Tata Consultancy Services Ltd."), ("TECHM", "Tech Mahindra Ltd."),
    ("TITAN", "Titan Company Ltd."), ("TMPV", "Tata Motors Passenger Vehicles Ltd."),
    ("TRENT", "Trent Ltd."), ("ULTRACEMCO", "UltraTech Cement Ltd."),
]


def fetch_constituents() -> pd.DataFrame:
    hdr = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Accept": "text/csv,*/*", "Accept-Language": "en-US,en;q=0.9",
           "Referer": "https://www.niftyindices.com/"}
    for attempt in range(4):
        try:
            req = urllib.request.Request(LIST_URL, headers=hdr)
            body = urllib.request.urlopen(req, timeout=25).read().decode()
            rows = list(csv.DictReader(io.StringIO(body)))
            df = pd.DataFrame([{"symbol": r["Symbol"], "company": r["Company Name"],
                                "industry": r.get("Industry", ""), "isin": r.get("ISIN Code", "")}
                               for r in rows])
            if len(df) >= 45:
                print(f"constituents: live file OK ({len(df)} rows)")
                return df
        except Exception as e:  # noqa: BLE001
            print(f"constituents attempt {attempt + 1} failed: {e!r:.100}")
            time.sleep(4)
    print("constituents: live file unreachable -> FALLBACK list of 2026-11-04")
    return pd.DataFrame([{"symbol": s, "company": c, "industry": "", "isin": ""}
                         for s, c in FALLBACK])


def yf_tickers(symbols):
    return [s + ".NS" for s in symbols] + ["^NSEI"]


def download_prices(symbols, log):
    import yfinance as yf
    tickers = yf_tickers(symbols)
    px = yf.download(tickers, start=START, interval="1d", auto_adjust=True,
                     progress=False, group_by="ticker", threads=True)
    close = {}
    for t in tickers:
        try:
            s = px[t]["Close"].dropna()
        except Exception:  # noqa: BLE001
            s = pd.Series(dtype=float)
        if len(s) > 0:
            close[t] = s
        else:
            close[t] = pd.Series(dtype=float)
            log["prices_empty"].append(t)
    # refetch the empties one by one (bulk endpoint occasionally drops tickers)
    for t in list(log["prices_empty"]):
        try:
            s = yf.Ticker(t).history(start=START, auto_adjust=True)["Close"].dropna()
            if len(s) > 0:
                close[t] = s
                log["prices_empty"].remove(t)
                log["prices_refetched"].append(t)
        except Exception as e:  # noqa: BLE001
            log["prices_errors"][t] = repr(e)[:200]
        time.sleep(0.4)
    df = pd.DataFrame(close)
    df.index.name = "Date"
    return df


def _row(df, *names):
    """First row of a yfinance statement frame whose label matches any name
    (case-insensitive, exact after strip), else None."""
    if df is None or df.empty:
        return None
    lut = {str(i).strip().lower(): i for i in df.index}
    for n in names:
        if n.lower() in lut:
            return df.loc[lut[n.lower()]]
    return None


def download_fundamentals(symbols, log):
    import yfinance as yf
    fields = {
        "net_income": ("income", ("Net Income Common Stockholders", "Net Income",
                                  "Net Income From Continuing Operation Net Minority Interest")),
        "diluted_eps": ("income", ("Diluted EPS",)),
        "basic_eps": ("income", ("Basic EPS",)),
        "avg_shares_diluted": ("income", ("Diluted Average Shares",)),
        "avg_shares_basic": ("income", ("Basic Average Shares",)),
        "total_revenue": ("income", ("Total Revenue",)),
        "equity": ("balance", ("Stockholders Equity", "Common Stock Equity",
                               "Total Equity Gross Minority Interest")),
        "total_assets": ("balance", ("Total Assets",)),
        "total_debt": ("balance", ("Total Debt",)),
        "lt_debt": ("balance", ("Long Term Debt", "Long Term Debt And Capital Lease Obligation")),
        "st_debt": ("balance", ("Current Debt", "Current Debt And Capital Lease Obligation",
                               "Short Long Term Debt")),
    }
    records, infos = [], {}
    for sym in symbols:
        t = sym + ".NS"
        try:
            tk = yf.Ticker(t)
            frames = {"income": tk.income_stmt, "balance": tk.balance_sheet}
        except Exception as e:  # noqa: BLE001
            log["fundamentals_errors"][t] = repr(e)[:200]
            continue
        got = set()
        for fname, (which, names) in fields.items():
            row = _row(frames[which], *names)
            if row is None:
                continue
            for period, val in row.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if pd.isna(v):
                    continue
                records.append({"ticker": t, "period_end": pd.Timestamp(period).date().isoformat(),
                                "field": fname, "value": v})
                got.add(fname)
        log["fundamentals_fields"][t] = sorted(got)
        try:
            info = tk.info or {}
            keep = {k: info.get(k) for k in
                    ("trailingPE", "priceToBook", "returnOnEquity", "dividendYield",
                     "sharesOutstanding", "marketCap", "enterpriseValue",
                     "totalDebt", "trailingEps", "bookValue", "beta")}
            infos[t] = {k: (None if v is None else (float(v) if isinstance(v, (int, float)) else str(v)))
                        for k, v in keep.items()}
        except Exception as e:  # noqa: BLE001
            log["info_errors"][t] = repr(e)[:200]
        time.sleep(0.3)
        print(f"  fundamentals {t}: {len(got)} fields")
    return pd.DataFrame(records), infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices-only", action="store_true")
    args = ap.parse_args()
    log = {"download_date": date.today().isoformat(), "prices_empty": [],
           "prices_refetched": [], "prices_errors": {}, "fundamentals_errors": {},
           "fundamentals_fields": {}, "info_errors": {}}

    cons = fetch_constituents()
    cons.to_csv(f"{RAW}/nifty50_constituents.csv", index=False)
    symbols = cons["symbol"].tolist()
    print(f"universe: {len(symbols)} symbols")

    px = download_prices(symbols, log)
    px.to_csv(f"{RAW}/daily_prices.csv")
    cov = px.notna().sum()
    print("price coverage (days): min", int(cov.min()), "max", int(cov.max()))
    for t in px.columns:
        first = px[t].first_valid_index()
        log.setdefault("price_first_date", {})[t] = str(first.date()) if first is not None else None

    if not args.prices_only:
        fund, infos = download_fundamentals(symbols, log)
        fund.to_csv(f"{RAW}/fundamentals_long.csv", index=False)
        with open(f"{RAW}/info_snapshot.json", "w") as f:
            json.dump(infos, f, indent=1)
        print(f"fundamentals rows: {len(fund)}")

    with open(f"{RAW}/download_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print("DONE. Log:", json.dumps({k: (len(v) if isinstance(v, (list, dict)) else v)
                                    for k, v in log.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
