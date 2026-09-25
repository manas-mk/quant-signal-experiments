# 02_walkforward.py
# Walk-forward test of every pair. Nothing here looks into the future.
#
# Every STEP trading days (one quarter) we stand at a date t and, for each of the 190 pairs:
#   1. FORMATION: run an Engle-Granger cointegration test on the PAST year (t-250 .. t-1)
#      and fit the hedge ratio beta on that same year.
#   2. PERSISTENCE: run the same test again on the NEXT year (t .. t+249).
#      Question: if a pair is cointegrated now, is it still cointegrated a year later?
#   3. TRADING: trade the next half-year (t .. t+124) with beta, mean and std FROZEN from formation,
#      using three exit rules (no protection, stop-loss, CUSUM alarm), as in pairs_breakdown/04_alarm.py.
#
# Output: results/walkforward.csv, one row per (formation date, pair).

import itertools
import os
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

FORM = 250        # formation window: about one year of trading days
TRADE = 125       # trading window: about six months
STEP = 63         # move forward one quarter at a time
ENTRY, EXIT = 2.0, 0.0
COST = 0.001      # 0.1% of gross position each time the position changes
STOP_Z = 3.5
K = 0.5           # CUSUM slack, same as pairs_breakdown/04_alarm.py

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
sector = pd.read_csv("sectors.csv", index_col=0)["sector"]
logp = np.log(prices)
dates = logp.index
N = len(logp)
pairs = list(itertools.combinations(logp.columns, 2))


def cusum(z, k=K):
    """Two-sided CUSUM on the z-score (numpy array). Returns max(up, down) each day."""
    up = down = 0.0
    out = np.empty(len(z))
    for i, x in enumerate(z):
        up = max(0.0, up + x - k)
        down = max(0.0, down - x - k)
        out[i] = max(up, down)
    return out


def trade(z, ds, gross, stop_at=None):
    """Mean-reversion rule on z. ds = daily change in the spread. If stop_at is an index,
    go flat for good from that day. Returns (daily P&L per unit of gross capital, positions)."""
    pos = np.zeros(len(z))
    for t in range(1, len(z)):
        if stop_at is not None and t >= stop_at:
            break
        prev, zt = pos[t - 1], z[t]
        if prev == 0:
            pos[t] = -1 if zt > ENTRY else (1 if zt < -ENTRY else 0)
        elif prev == 1:
            pos[t] = 0 if zt >= EXIT else 1
        else:
            pos[t] = 0 if zt <= EXIT else -1
    held = np.r_[0.0, pos[:-1]]                       # position held during each day
    changes = np.abs(np.diff(np.r_[0.0, pos]))
    changes[-1] += abs(pos[-1])                       # close whatever is open at the end
    return held * ds / gross - COST * changes, pos


def first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


rows = []
starts = list(range(FORM, N - TRADE + 1, STEP))
t0 = time.time()
for w, e in enumerate(starts):
    f = slice(e - FORM, e)
    nxt = slice(e, e + FORM) if e + FORM <= N else None
    tr = slice(e, e + TRADE)
    for a, b in pairs:
        A, B = logp[a].values, logp[b].values
        _, p_form, _ = coint(A[f], B[f])
        p_next = coint(A[nxt], B[nxt])[1] if nxt is not None else np.nan
        beta, alpha = np.polyfit(B[f], A[f], 1)

        s_form = A[f] - beta * B[f] - alpha
        mu, sd = s_form.mean(), s_form.std(ddof=1)
        s_tr = A[tr] - beta * B[tr] - alpha
        z = (s_tr - mu) / sd
        ds = np.r_[0.0, np.diff(s_tr)]
        gross = 1 + abs(beta)

        h = max(5.0, 1.2 * cusum((s_form - mu) / sd).max())   # calibrated on formation only
        # the detectors start when trading starts, so the first day never triggers from the past
        stop_day = first(np.abs(z) > STOP_Z)
        alarm_day = first(cusum(z) > h)

        pnl_none, pos = trade(z, ds, gross)
        pnl_stop, _ = trade(z, ds, gross, stop_day)
        pnl_cusum, _ = trade(z, ds, gross, alarm_day)
        path = np.cumsum(pnl_none)

        rows.append({
            "form_end": dates[e - 1].date(), "a": a, "b": b,
            "same_sector": sector[a] == sector[b],
            "p_form": p_form, "p_next": p_next, "beta": beta,
            "stop_day": stop_day, "alarm_day": alarm_day,
            "pnl_none": pnl_none.sum(), "pnl_stop": pnl_stop.sum(), "pnl_cusum": pnl_cusum.sum(),
            "peak_none": path.max(), "trough_none": path.min(),
            "n_trades": int(((np.r_[0.0, pos[:-1]] == 0) & (pos != 0)).sum()),
        })
    done = w + 1
    eta = (time.time() - t0) / done * (len(starts) - done)
    print(f"window {done}/{len(starts)} (formation ends {dates[e - 1].date()}), "
          f"about {eta / 60:.1f} min left", flush=True)

os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/walkforward.csv", index=False)
print(f"\nsaved results/walkforward.csv: {len(df)} rows "
      f"({len(starts)} formation dates x {len(pairs)} pairs)")
