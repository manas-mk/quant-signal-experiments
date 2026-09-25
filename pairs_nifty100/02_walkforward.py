# 02_walkforward.py
# Walk-forward test of every pair of NIFTY 100 stocks (about 4,950 pairs), with two cointegration tests.
#
# Every quarter (STEP = 63 trading days) we stand at a date t and, for every pair whose two stocks
# both have complete prices in the window:
#   1. FORMATION (t-252 .. t-1):
#        - Engle-Granger test: regress A on B, test the residual for a unit root (gives a p-value).
#        - Johansen trace test: tests both stocks jointly, so the answer does not depend on which
#          stock is called A (gives a pass/fail at 95%).
#        - hedge ratio beta, spread mean and std, fitted on this year only.
#   2. PERSISTENCE: the same two tests on the NEXT year (t .. t+251). Because 252 = 4 x 63, that is
#      exactly the formation window four quarters later, so we just look it up instead of re-running it.
#   3. TRADING (t .. t+125): z-score rule with everything frozen from formation, three exit rules
#      (no protection, stop-loss, CUSUM alarm), same as pairs_persistence/.
#
# Uses all CPU cores. Output: results/walkforward.csv, one row per (formation date, pair).

import itertools
import os
import time
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

FORM = 252        # formation window: one trading year
TRADE = 126       # trading window: six months
STEP = 63         # one quarter; FORM must be a multiple of STEP for the persistence lookup
ENTRY, EXIT = 2.0, 0.0
COST = 0.001      # 0.1% of gross position each time the position changes
STOP_Z = 3.5
K = 0.5           # CUSUM slack
assert FORM % STEP == 0


def cusum(z, k=K):
    up = down = 0.0
    out = np.empty(len(z))
    for i, x in enumerate(z):
        up = max(0.0, up + x - k)
        down = max(0.0, down - x - k)
        out[i] = max(up, down)
    return out


def trade(z, ds, gross, stop_at=None):
    """Mean-reversion rule on z; flat for good from index stop_at. Returns (daily P&L, positions)."""
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
    held = np.r_[0.0, pos[:-1]]
    changes = np.abs(np.diff(np.r_[0.0, pos]))
    changes[-1] += abs(pos[-1])
    return held * ds / gross - COST * changes, pos


def first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def init(logp, pairs):
    global LOGP, PAIRS
    LOGP, PAIRS = logp, pairs


def run_window(e):
    """All pairs for the formation window ending just before index e."""
    warnings.simplefilter("ignore")
    N = LOGP.shape[0]
    f = slice(e - FORM, e)
    ok_form = ~np.isnan(LOGP[f]).any(axis=0)
    can_trade = e + TRADE <= N
    ok_trade = ~np.isnan(LOGP[e:e + TRADE]).any(axis=0) if can_trade else np.zeros_like(ok_form)
    rows = []
    for i, j in PAIRS:
        if not (ok_form[i] and ok_form[j]):
            continue
        A, B = LOGP[f, i], LOGP[f, j]
        try:
            eg_p = coint(A, B)[1]
        except Exception:
            eg_p = np.nan
        try:
            jo = coint_johansen(np.column_stack([A, B]), 0, 1)
            jo_stat, jo_crit = jo.lr1[0], jo.cvt[0, 1]
        except Exception:
            jo_stat = jo_crit = np.nan
        beta, alpha = np.polyfit(B, A, 1)
        row = [e, i, j, eg_p, jo_stat, jo_crit, beta]

        if can_trade and ok_trade[i] and ok_trade[j]:
            s_form = A - beta * B - alpha
            mu, sd = s_form.mean(), s_form.std(ddof=1)
            s_tr = LOGP[e:e + TRADE, i] - beta * LOGP[e:e + TRADE, j] - alpha
            z = (s_tr - mu) / sd
            ds = np.r_[0.0, np.diff(s_tr)]
            gross = 1 + abs(beta)
            h = max(5.0, 1.2 * cusum((s_form - mu) / sd).max())
            stop_day = first(np.abs(z) > STOP_Z)
            alarm_day = first(cusum(z) > h)
            pnl_none, pos = trade(z, ds, gross)
            pnl_stop, _ = trade(z, ds, gross, stop_day)
            pnl_cusum, _ = trade(z, ds, gross, alarm_day)
            n_trades = int(((np.r_[0.0, pos[:-1]] == 0) & (pos != 0)).sum())
            row += [stop_day, alarm_day, pnl_none.sum(), pnl_stop.sum(), pnl_cusum.sum(), n_trades]
        else:
            row += [None, None, np.nan, np.nan, np.nan, np.nan]
        rows.append(row)
    return rows


if __name__ == "__main__":
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
    ind = pd.read_csv("industries.csv", index_col="ticker")["Industry"]
    tickers = list(prices.columns)
    logp = np.log(prices.values)
    dates = prices.index

    # Data breaks: Yahoo does not adjust some demergers (e.g. Vedanta 2026, Tata Motors 2025), which
    # show up as fake one-day drops of 40-65%. Any |daily log return| > BREAK is treated as missing
    # data, so that stock is left out of every window that contains the break.
    BREAK = 0.4
    jumps = np.abs(np.diff(logp, axis=0)) > BREAK
    for d, k in zip(*np.nonzero(jumps)):
        print(f"  data break: {tickers[k]} on {dates[d + 1].date()} "
              f"({100 * (np.exp(logp[d + 1, k] - logp[d, k]) - 1):+.0f}% in one day) - excluded around it")
    logp[1:][jumps] = np.nan
    N = len(dates)
    pairs = list(itertools.combinations(range(len(tickers)), 2))
    starts = list(range(FORM, N + 1, STEP))
    workers = max(1, cpu_count() - 1)
    print(f"{len(tickers)} stocks, {len(pairs)} pairs, {len(starts)} formation dates, {workers} worker processes")

    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    rows = []
    with Pool(workers, initializer=init, initargs=(logp, pairs)) as pool:
        for n, r in enumerate(pool.imap_unordered(run_window, starts), 1):
            rows.extend(r)
            eta = (time.time() - t0) / n * (len(starts) - n)
            print(f"  {n}/{len(starts)} windows done, about {eta / 60:.1f} min left", flush=True)

    cols = ["e", "i", "j", "eg_p", "jo_stat", "jo_crit", "beta",
            "stop_day", "alarm_day", "pnl_none", "pnl_stop", "pnl_cusum", "n_trades"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv("results/raw_rows.csv", index=False)          # safety copy of the slow part
    df["jo_pass"] = np.where(df["jo_stat"].isna(), np.nan, (df["jo_stat"] > df["jo_crit"]).astype(float))

    # persistence: the same pair's tests on the window that starts FORM days later
    nxt = df[["e", "i", "j", "eg_p", "jo_pass"]].copy()
    nxt["e"] -= FORM
    df = df.merge(nxt, on=["e", "i", "j"], how="left", suffixes=("", "_next"))

    df["form_end"] = dates[df["e"] - 1].date
    df["a"] = [tickers[k] for k in df["i"]]
    df["b"] = [tickers[k] for k in df["j"]]
    df["same_industry"] = [ind.get(a) == ind.get(b) for a, b in zip(df["a"], df["b"])]
    df = df.sort_values(["e", "i", "j"]).drop(columns=["i", "j"])

    df.to_csv("results/walkforward.csv", index=False)
    os.remove("results/raw_rows.csv")
    print(f"\nsaved results/walkforward.csv: {len(df)} pair-windows in {(time.time() - t0) / 60:.1f} min")
