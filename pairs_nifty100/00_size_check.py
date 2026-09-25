# 00_size_check.py
# How often does each test say "cointegrated" when the two series are NOT related at all?
# A test at the 5% level should do this 5% of the time. We check with 2,000 pairs of independent
# random walks that have the same length (252 days) and daily volatility (1.5%) as stock prices.
#
# The rates measured here are the real "chance levels" used in the charts of 03_analyse.py.

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.simplefilter("ignore")
R, T, VOL = 2000, 252, 0.015
rng = np.random.default_rng(0)

eg = jo = 0
for _ in range(R):
    X = np.cumsum(rng.normal(0, VOL, size=(T, 2)), axis=0) + 5
    eg += coint(X[:, 0], X[:, 1])[1] < 0.05
    r = coint_johansen(X, 0, 1)
    jo += r.lr1[0] > r.cvt[0, 1]

se = lambda p: 1.96 * np.sqrt(p * (1 - p) / R)
eg, jo = eg / R, jo / R
print(f"False-positive rate on {R} pairs of unrelated random walks (nominal level 5%):")
print(f"  Engle-Granger: {100*eg:.1f}% (+/- {100*se(eg):.1f})")
print(f"  Johansen:      {100*jo:.1f}% (+/- {100*se(jo):.1f})")

os.makedirs("results", exist_ok=True)
pd.DataFrame({"test": ["eg", "jo"], "size": [eg, jo]}).to_csv("results/size_check.csv", index=False)
print("saved results/size_check.csv")
