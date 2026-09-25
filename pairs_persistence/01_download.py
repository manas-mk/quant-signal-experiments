# 01_download.py
# Download 11 years of daily prices for 20 large NSE stocks: 5 each from banking, IT, FMCG and metals.
# 20 stocks give 190 possible pairs: 40 inside the same sector and 150 across sectors.

import pandas as pd
import yfinance as yf

SECTORS = {
    "Banks":  ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
    "IT":     ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    "FMCG":   ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
    "Metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "SAIL", "JINDALSTEL"],
}
START, END = "2015-01-01", "2026-01-01"

tickers = [t + ".NS" for names in SECTORS.values() for t in names]
data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)["Close"]
data.columns = [c.replace(".NS", "") for c in data.columns]

missing = data.isna().mean().sort_values(ascending=False)
print("share of missing days per stock (top 5):")
print(missing.head().round(3).to_string())

# fill single-day gaps (holidays on one exchange feed), then keep only days where every stock has a price
data = data.ffill(limit=2).dropna()
data.to_csv("prices.csv")

sector_of = {t: s for s, names in SECTORS.items() for t in names}
pd.Series(sector_of, name="sector").to_csv("sectors.csv", index_label="ticker")

print(f"\nsaved prices.csv: {data.shape[1]} stocks, {data.shape[0]} trading days, "
      f"{data.index[0].date()} to {data.index[-1].date()}")
