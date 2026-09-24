# 04_alarm.py
# Can an ONLINE alarm catch the breakdown early enough to cut the losses?
# Same frozen-hedge-ratio setup as 03_trade.py, with three versions of the strategy:
#   1. no protection        (same as step 3)
#   2. stop-loss            exit and stop trading the pair once |z| > STOP_Z
#   3. CUSUM alarm          exit and stop trading once a CUSUM detector fires

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

PAIRS = [("GESHIP.NS", "SOMANYCERA.NS"),
         ("INDUSINDBK.NS", "SHRIRAMFIN.NS")]

ENTRY, EXIT = 2.0, 0.0
COST = 0.001
STOP_Z = 3.5      # stop-loss level, in formation standard deviations
K = 0.5           # CUSUM "slack": how far z must drift before it starts to count


def cusum(z, k):
    """Two-sided CUSUM on the z-score. Returns the larger of the up and down sums each day."""
    up, down, out = 0.0, 0.0, []
    for x in z:
        up = max(0.0, up + x - k)       # grows while z stays above +k
        down = max(0.0, down - x - k)   # grows while z stays below -k
        out.append(max(up, down))
    return pd.Series(out, index=z.index)


def run_strategy(z, spread, stop_date=None):
    """Mean-reversion rule on z; if stop_date is given, go flat for good from that day."""
    pos = np.zeros(len(z))
    for t in range(1, len(z)):
        if stop_date is not None and z.index[t] >= stop_date:
            pos[t] = 0
            continue
        prev, zt = pos[t - 1], z.iloc[t]
        if prev == 0:
            pos[t] = -1 if zt > ENTRY else (1 if zt < -ENTRY else 0)
        elif prev == 1:
            pos[t] = 0 if zt >= EXIT else 1
        else:
            pos[t] = 0 if zt <= EXIT else -1
    pos = pd.Series(pos, index=z.index)
    d = spread.diff().fillna(0)
    pnl = pos.shift(1).fillna(0) * d - COST * pos.diff().abs().fillna(0)
    return pnl.cumsum()


for a, b in PAIRS:
    A, B = logp[a], logp[b]
    f_start, f_end, p_best = best_formation(A, B)
    form = slice(f_start, f_end)
    beta, alpha = np.polyfit(B[form], A[form], 1)
    spread_all = (A - beta * B - alpha)[f_start:]
    mu, sd = spread_all[form].mean(), spread_all[form].std()
    z_all = (spread_all - mu) / sd

    after = z_all.index > pd.Timestamp(f_end)
    z, spread = z_all[after], spread_all[after]

    # Calibrate the CUSUM threshold on the FORMATION period (when the pair was "healthy"):
    # the alarm should never have fired there, so set it a bit above the largest value seen.
    h = max(5.0, 1.2 * cusum(z_all[form], K).max())
    c = cusum(z, K)

    stop_date = z.index[(z.abs() > STOP_Z).argmax()] if (z.abs() > STOP_Z).any() else None
    alarm_date = c.index[(c > h).argmax()] if (c > h).any() else None

    pnl_none = run_strategy(z, spread)
    pnl_stop = run_strategy(z, spread, stop_date)
    pnl_cusum = run_strategy(z, spread, alarm_date)

    start = z.index[0]
    def days(d):
        return f"{d.date()} ({(d - start).days} days into trading)" if d is not None else "never"

    print(f"\n{a} / {b}")
    print(f"  formation {f_start.date()} to {f_end.date()}: beta = {beta:.2f}, cointegration p = {p_best:.3f}")
    print(f"  trading starts {start.date()}")
    print(f"  CUSUM threshold h = {h:.1f}")
    print(f"  stop-loss (|z| > {STOP_Z}) fired: {days(stop_date)}")
    print(f"  CUSUM alarm fired:               {days(alarm_date)}")
    print(f"  final P&L  no protection: {pnl_none.iloc[-1]:+.3f}")
    print(f"             stop-loss:     {pnl_stop.iloc[-1]:+.3f}")
    print(f"             CUSUM alarm:   {pnl_cusum.iloc[-1]:+.3f}")

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(z, lw=1)
    for lvl, st in ((ENTRY, "--"), (STOP_Z, ":")):
        ax[0].axhline(lvl, ls=st, c="grey"); ax[0].axhline(-lvl, ls=st, c="grey")
    ax[0].set_ylabel("spread z-score")
    ax[1].plot(c, lw=1); ax[1].axhline(h, c="red", ls="--", label=f"threshold h = {h:.1f}")
    ax[1].set_ylabel("CUSUM"); ax[1].legend(loc="upper left")
    ax[2].plot(pnl_none, label="no protection")
    ax[2].plot(pnl_stop, label=f"stop-loss |z|>{STOP_Z}")
    ax[2].plot(pnl_cusum, label="CUSUM alarm")
    ax[2].set_ylabel("cumulative P&L"); ax[2].legend(loc="lower left")
    for axis in ax:
        if stop_date is not None:
            axis.axvline(stop_date, c="orange", lw=1)
        if alarm_date is not None:
            axis.axvline(alarm_date, c="green", lw=1)
    fig.suptitle(f"{a} / {b}: online alarms (orange = stop-loss, green = CUSUM)")
    fig.tight_layout()
    name = f"04_{a.split('.')[0]}_{b.split('.')[0]}.png"
    fig.savefig(name, dpi=120)
    print("  saved", name)
