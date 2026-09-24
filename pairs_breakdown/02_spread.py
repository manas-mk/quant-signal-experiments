# 02_spread.py
# For each pair: rolling hedge ratio, spread z-score, and rolling cointegration test.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
logp = np.log(prices)

PAIRS = [("GESHIP.NS", "SOMANYCERA.NS"),     # broke down in the paper's test period
         ("INDUSINDBK.NS", "SHRIRAMFIN.NS")] # did well in the test period

BETA_WIN = 120   # days used to estimate the hedge ratio
Z_WIN = 60       # days used to standardise the spread
COINT_WIN = 250  # days (about 1 year) used for each cointegration test
COINT_STEP = 20  # run the test every 20 days

for a, b in PAIRS:
    A, B = logp[a], logp[b]

    # Rolling hedge ratio: beta = cov(A, B) / var(B) over the past BETA_WIN days
    beta = A.rolling(BETA_WIN).cov(B) / B.rolling(BETA_WIN).var()
    alpha = A.rolling(BETA_WIN).mean() - beta * B.rolling(BETA_WIN).mean()

    # Use YESTERDAY's beta and alpha, so we never use information from the future
    spread = A - beta.shift(1) * B - alpha.shift(1)
    z = (spread - spread.rolling(Z_WIN).mean()) / spread.rolling(Z_WIN).std()

    # Rolling Engle-Granger cointegration test
    pvals = {}
    for end in range(COINT_WIN, len(A), COINT_STEP):
        window = slice(end - COINT_WIN, end)
        _, p, _ = coint(A.iloc[window], B.iloc[window])
        pvals[A.index[end - 1]] = p
    pvals = pd.Series(pvals)

    print(f"\n{a} / {b}")
    print(f"  share of test windows cointegrated (p < 0.05): {(pvals < 0.05).mean():.0%}")
    print(f"  hedge ratio range: {beta.min():.2f} to {beta.max():.2f}")

    # Three panels: hedge ratio, spread z-score, cointegration p-value
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(beta); ax[0].set_ylabel("hedge ratio β")
    ax[1].plot(z); ax[1].axhline(2, ls="--", c="grey"); ax[1].axhline(-2, ls="--", c="grey")
    ax[1].set_ylabel("spread z-score")
    ax[2].plot(pvals, marker="o"); ax[2].axhline(0.05, ls="--", c="red")
    ax[2].set_ylabel("coint. p-value"); ax[2].set_ylim(0, 1)
    fig.suptitle(f"{a} / {b}")
    fig.tight_layout()
    name = f"02_{a.split('.')[0]}_{b.split('.')[0]}.png"
    fig.savefig(name, dpi=120)
    print("  saved", name)