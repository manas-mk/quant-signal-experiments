# 01_download.py
# Downloads daily prices for the pairs we want to study and saves them to a CSV file.

import yfinance as yf
import matplotlib.pyplot as plt

# The pair from Prof. Chakrabarty's paper that did well in training but lost money in testing,
# plus two "healthy" pairs to compare against.
tickers = [
    "GESHIP.NS", "SOMANYCERA.NS",      # the pair that broke down
    "INDUSINDBK.NS", "SHRIRAMFIN.NS",  # a pair that did well in his test period
    "HDFCBANK.NS", "ICICIBANK.NS",     # a backup healthy pair (two similar banks)
]

# Daily closing prices, adjusted for splits and dividends
prices = yf.download(tickers, start="2019-01-01", end="2025-01-01",
                     auto_adjust=True, progress=False)["Close"]

prices.to_csv("prices.csv")

# Check what we got: how many days of data per stock, and the first and last date
print("Rows (trading days):", len(prices))
print("\nDays of data per stock:")
print(prices.notna().sum())
print("\nFirst date with data for each stock:")
print(prices.apply(lambda col: col.first_valid_index()))

# Plot each stock divided by its first valid price, so they all start at 1 and are easy to compare
normalised = prices / prices.bfill().iloc[0]
normalised.plot(figsize=(11, 5), title="Prices relative to Jan 2019")
plt.ylabel("price / first price")
plt.tight_layout()
plt.savefig("01_prices.png", dpi=120)
print("\nSaved prices.csv and 01_prices.png")