# 03_trade.py
# Formation / trading test: fit the hedge ratio once on a cointegrated stretch,
# freeze it, trade the spread afterwards, and look for change-points.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from statsmodels.tsa.stattools import coint

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
logp = np.log(prices)


def best_formation(A, B, win=250, step=20, min_trade_days=250):
    """Find the 'win'-day window with the lowest Engle-Granger p-value, leaving at least
    'min_trade_days' of data after it to trade on. Returns (start date, end date, p-value)."""
    best = (None, None, 1.0)
    for end in range(win, len(A) - min_trade_days, step):
        w = slice(end - win, end)
        _, p, _ = coint(A.iloc[w], B.iloc[w])
        if p < best[2]:
            best = (A.index[end - win], A.index[end - 1], p)
    return best

# The formation window is chosen automatically: the 1-year window where the pair looks most cointegrated.
PAIRS = [("GESHIP.NS", "SOMANYCERA.NS"),
         ("INDUSINDBK.NS", "SHRIRAMFIN.NS")]

ENTRY, EXIT = 2.0, 0.0   # open a trade at |z| > 2, close it when z crosses 0
COST = 0.001             # 0.1% cost each time the position changes (same as the paper)
N_BREAKS = 2             # how many change-points to look for after formation

for a, b in PAIRS:
    A, B = logp[a], logp[b]
    f_start, f_end, p_best = best_formation(A, B)
    form = slice(f_start, f_end)

    # 1. Fit A = alpha + beta*B ONCE, on the formation period only
    beta, alpha = np.polyfit(B[form], A[form], 1)
    _, p_form, _ = coint(A[form], B[form])

    # 2. Spread from formation start onwards, standardised with FORMATION mean/std (frozen)
    spread = (A - beta * B - alpha)[f_start:]
    mu, sd = spread[form].mean(), spread[form].std()
    z = (spread - mu) / sd

    # 3. Trading rule, only after formation ends
    trade = z[z.index > pd.Timestamp(f_end)]
    pos = np.zeros(len(trade))           # +1 = long spread, -1 = short spread, 0 = flat
    for t in range(1, len(trade)):
        prev, zt = pos[t - 1], trade.iloc[t]
        if prev == 0:
            pos[t] = -1 if zt > ENTRY else (1 if zt < -ENTRY else 0)
        elif prev == 1:
            pos[t] = 0 if zt >= EXIT else 1
        else:
            pos[t] = 0 if zt <= EXIT else -1
    pos = pd.Series(pos, index=trade.index)

    # daily P&L: yesterday's position times today's change in the spread, minus costs
    d_spread = spread.loc[trade.index].diff().fillna(0)
    pnl = pos.shift(1).fillna(0) * d_spread - COST * pos.diff().abs().fillna(0)
    cum = pnl.cumsum()

    # 4. Change-points in the spread after formation (the N_BREAKS biggest shifts in its level)
    s = spread.loc[trade.index].values
    bkps = rpt.Binseg(model="l2", min_size=20).fit(s).predict(n_bkps=N_BREAKS)[:-1]
    cp_dates = [trade.index[i] for i in bkps]

    peak_date = cum.idxmax()
    n_trades = int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())   # number of times a trade is opened
    print(f"\n{a} / {b}")
    print(f"  formation {f_start.date()} to {f_end.date()}: beta = {beta:.2f}, cointegration p = {p_form:.3f}")
    print(f"  trades opened: {n_trades},  final P&L: {cum.iloc[-1]:+.3f},  best P&L {cum.max():+.3f} on {peak_date.date()}")
    print(f"  change-points: {[str(d.date()) for d in cp_dates]}")
    if cp_dates:
        gap = (peak_date - cp_dates[0]).days
        when = "BEFORE" if gap > 0 else "AFTER"
        print(f"  first change-point came {abs(gap)} days {when} the P&L peak")

    # 5. Plot: spread with bands and change-points / position / cumulative P&L
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(spread, lw=1)
    ax[0].axvspan(pd.Timestamp(f_start), pd.Timestamp(f_end), color="green", alpha=0.15, label="formation")
    for k in (-ENTRY, 0, ENTRY):
        ax[0].axhline(mu + k * sd, ls="--" if k else ":", c="grey")
    ax[0].set_ylabel("spread (log)"); ax[0].legend(loc="upper left")
    ax[1].plot(pos, drawstyle="steps-post"); ax[1].set_ylabel("position")
    ax[2].plot(cum); ax[2].set_ylabel("cumulative P&L")
    for axis in ax:
        for d in cp_dates:
            axis.axvline(d, c="red", lw=1)
    fig.suptitle(f"{a} / {b}: frozen hedge ratio (red lines = change-points)")
    fig.tight_layout()
    name = f"03_{a.split('.')[0]}_{b.split('.')[0]}.png"
    fig.savefig(name, dpi=120)
    print("  saved", name)
