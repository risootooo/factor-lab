"""Load the cached raw data (data/raw/) into analysis-ready panels.

Nothing in this package touches the network: scripts/download_data.py is the
only downloader. Everything here is a pure function of the cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INDEX_TICKER = "^NSEI"


def load_constituents() -> pd.DataFrame:
    return pd.read_csv(RAW / "nifty50_constituents.csv")


def load_daily_prices() -> pd.DataFrame:
    """Daily adjusted closes; columns are Yahoo tickers ('RELIANCE.NS', '^NSEI')."""
    px = pd.read_csv(RAW / "daily_prices.csv", index_col=0, parse_dates=True)
    px.index.name = "date"
    return px.sort_index()


def load_fundamentals() -> pd.DataFrame:
    """Annual statement items in long format: ticker, period_end, field, value."""
    f = pd.read_csv(RAW / "fundamentals_long.csv", parse_dates=["period_end"])
    return f


def load_download_log() -> dict:
    with open(RAW / "download_log.json") as fh:
        return json.load(fh)


def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Last traded close of each calendar month (rows = month-end dates)."""
    me = daily.resample(pd.offsets.MonthEnd()).last()
    return me[me.index >= "2015-01-01"]


def monthly_returns(month_end_px: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns from month-end prices."""
    return month_end_px.pct_change(fill_method=None)
