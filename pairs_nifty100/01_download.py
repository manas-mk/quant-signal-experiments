# 01_download.py
# 1. Get the official NIFTY 100 constituent list (company, industry, NSE symbol) from NSE.
# 2. Download daily adjusted prices for all 100 stocks from Yahoo Finance.
#
# Stocks that listed after 2015 (IPOs) simply have missing prices before they listed.
# 02_walkforward.py only uses a stock in a window where it has complete data.

import io
import os
import sys

import pandas as pd
import requests
import yfinance as yf

START, END = "2015-01-01", "2026-09-01"
LIST_FILE = "ind_nifty100list.csv"
URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
]

# ---------------------------------------------------------------- constituent list
if not os.path.exists(LIST_FILE):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in URLS:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.ok and "Symbol" in r.text[:300]:
                open(LIST_FILE, "w", encoding="utf-8").write(r.text)
                print(f"downloaded the NIFTY 100 list from {url}")
                break
        except requests.RequestException:
            pass
    else:
        print("Could not download the NIFTY 100 list automatically.")
        print("Open this link in your browser, save the file into this folder as")
        print(f"'{LIST_FILE}', and run this script again:\n  {URLS[0]}")
        sys.exit(1)

lst = pd.read_csv(LIST_FILE)
lst.columns = [c.strip() for c in lst.columns]
lst["Symbol"] = lst["Symbol"].str.strip()
print(f"{len(lst)} stocks in the list, {lst['Industry'].nunique()} industries")

# ---------------------------------------------------------------- prices
tickers = [s + ".NS" for s in lst["Symbol"]]
data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=True)["Close"]
data.columns = [c.replace(".NS", "") for c in data.columns]

failed = [s for s in lst["Symbol"] if s not in data.columns or data[s].notna().sum() == 0]
if failed:
    print("no prices for:", ", ".join(failed))
data = data.drop(columns=[c for c in failed if c in data.columns])

# fill 1-2 day gaps inside a stock's history; leave the pre-listing period empty
data = data.ffill(limit=2).dropna(how="all")
data.to_csv("prices.csv")

lst[lst["Symbol"].isin(data.columns)][["Symbol", "Company Name", "Industry"]] \
    .rename(columns={"Symbol": "ticker"}).to_csv("industries.csv", index=False)

first = data.apply(lambda s: s.first_valid_index())
late = first[first > data.index[0] + pd.Timedelta(days=30)].sort_values()
print(f"\nsaved prices.csv: {data.shape[1]} stocks, {data.shape[0]} trading days, "
      f"{data.index[0].date()} to {data.index[-1].date()}")
print(f"{len(late)} stocks start later than 2015 (listed later), e.g. "
      + ", ".join(f"{t} ({d.year})" for t, d in late.tail(5).items()))
