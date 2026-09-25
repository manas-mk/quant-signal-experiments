# 03_analyse.py
# Answers from results/walkforward.csv (NIFTY 100, Engle-Granger vs Johansen):
#   Q1. How many pairs pass each test, compared with chance?
#   Q2. Does passing this year predict passing next year? (both tests, same vs different industry)
#   Q3. Is a smaller Engle-Granger p-value a better pair?
#   Q4. Do pairs selected by either test trade better than pairs that failed?
#   Q5. Exit rules: no protection vs stop-loss vs CUSUM alarm.
#
# 95% intervals: moving-block bootstrap over formation dates (blocks of 4 quarters). Every statistic
# here is a ratio of per-date sums, so the bootstrap only resamples small per-date tables (fast).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHA = 0.05
BLOCK, N_BOOT = 4, 2000
rng = np.random.default_rng(0)

df = pd.read_csv("results/walkforward.csv", parse_dates=["form_end"])
try:                                     # real false-positive rates from 00_size_check.py
    size = pd.read_csv("results/size_check.csv", index_col="test")["size"]
    CHANCE = {"Engle-Granger": size["eg"], "Johansen": size["jo"]}
except FileNotFoundError:
    CHANCE = {"Engle-Granger": ALPHA, "Johansen": ALPHA}
df["eg_pass"] = df["eg_p"] < ALPHA
df["eg_pass_next"] = np.where(df["eg_p_next"].notna(), df["eg_p_next"] < ALPHA, np.nan)
df["jo_pass"] = df["jo_pass"].astype(float)
df["both"] = df["eg_pass"] & (df["jo_pass"] == 1)
df["tradeable"] = (df["beta"] > 0) & df["pnl_none"].notna()
dates = np.sort(df["form_end"].unique())
pos = {d: k for k, d in enumerate(dates)}
df["d"] = df["form_end"].map(pos)
T = len(dates)

# block-bootstrap resamples of date positions, shared by every statistic
starts = rng.integers(0, T - BLOCK + 1, size=(N_BOOT, int(np.ceil(T / BLOCK))))
IDX = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :T]
out = []


def say(line=""):
    print(line)
    out.append(line)


def per_date(mask, value):
    """Per-date sum of value over rows in mask, and per-date count of rows in mask."""
    m = mask.values if hasattr(mask, "values") else mask
    v = np.asarray(value, dtype=float)
    num = np.bincount(df["d"][m], weights=v[m], minlength=T)
    den = np.bincount(df["d"][m], minlength=T).astype(float)
    return num, den


def ratio(mask, value):
    """Mean of value over rows in mask, with a 95% block-bootstrap interval."""
    num, den = per_date(mask, value)
    boots = num[IDX].sum(1) / den[IDX].sum(1)
    return num.sum() / den.sum(), *np.nanpercentile(boots, [2.5, 97.5]), int(den.sum())


def diff(mask1, mask2, value):
    n1, d1 = per_date(mask1, value)
    n2, d2 = per_date(mask2, value)
    boots = n1[IDX].sum(1) / d1[IDX].sum(1) - n2[IDX].sum(1) / d2[IDX].sum(1)
    return n1.sum() / d1.sum() - n2.sum() / d2.sum(), *np.nanpercentile(boots, [2.5, 97.5])


pct = lambda r: f"{100*r[0]:5.1f}% [{100*r[1]:.1f}, {100*r[2]:.1f}]"
pnl = lambda r: f"{100*r[0]:+5.2f}% [{100*r[1]:+.2f}, {100*r[2]:+.2f}]"

n_stocks = len(set(df["a"]) | set(df["b"]))
say(f"{n_stocks} stocks, {len(df)} pair-windows, {T} formation dates "
    f"({pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()})")
say(f"same-industry pairs: {100*df['same_industry'].mean():.1f}% of pair-windows\n")

# ---------------------------------------------------------------- Q1
per = df.groupby("form_end").agg(n=("eg_p", "size"), eg=("eg_pass", "mean"), jo=("jo_pass", "mean"),
                                  both=("both", "mean"))
say("Q1. Share of available pairs passing each test")
say(f"    Engle-Granger: {100*df['eg_pass'].mean():.1f}% (chance level {100*CHANCE['Engle-Granger']:.1f}%)   "
    f"Johansen: {100*df['jo_pass'].mean():.1f}% (chance level {100*CHANCE['Johansen']:.1f}%)   "
    f"both: {100*df['both'].mean():.1f}%")
say(f"    pairs available per date: {per['n'].min()} to {per['n'].max()}")
say(f"    same-industry share: {100*df['same_industry'].mean():.1f}% of all pairs, "
    f"{100*df.loc[df['eg_pass'], 'same_industry'].mean():.1f}% of Engle-Granger passes, "
    f"{100*df.loc[df['jo_pass'] == 1, 'same_industry'].mean():.1f}% of Johansen passes\n")

fig, ax = plt.subplots(figsize=(10, 4))
l1, = ax.plot(per.index, 100 * per["eg"], marker="o", ms=3, label="Engle-Granger p < 0.05")
l2, = ax.plot(per.index, 100 * per["jo"], marker="o", ms=3, label="Johansen trace, 95%")
ax.plot(per.index, 100 * per["both"], marker="o", ms=3, label="both tests")
ax.axhline(100 * CHANCE["Engle-Granger"], color=l1.get_color(), ls="--", lw=1,
           label=f"Engle-Granger chance level ({100*CHANCE['Engle-Granger']:.1f}%)")
ax.axhline(100 * CHANCE["Johansen"], color=l2.get_color(), ls="--", lw=1,
           label=f"Johansen chance level ({100*CHANCE['Johansen']:.1f}%)")
ax.set_ylabel("% of available pairs passing")
ax.set_title(f"How many of ~{int(per['n'].median())} NIFTY 100 pairs look cointegrated on each date")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("01_passing_vs_chance.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- Q2
say("Q2. Share cointegrated NEXT year, by the same test (95% block-bootstrap interval)")
q2 = []
for test, now, nxt in [("Engle-Granger", "eg_pass", "eg_pass_next"), ("Johansen", "jo_pass", "jo_pass_next")]:
    has = df[nxt].notna()
    for grp, gmask in [("all pairs", has), ("same industry", has & df["same_industry"]),
                       ("different industry", has & ~df["same_industry"])]:
        yes = ratio(gmask & (df[now] == 1), df[nxt])
        no = ratio(gmask & (df[now] == 0), df[nxt])
        q2.append((test, grp, yes, no))
        say(f"    {test:13s} {grp:18s} passed this year {pct(yes)}  failed {pct(no)}  (n = {yes[3]} vs {no[3]})")
say()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, test in zip(axes, ["Engle-Granger", "Johansen"]):
    rows = [r for r in q2 if r[0] == test]
    x = np.arange(len(rows))
    for off, k, col, lab in [(-0.2, 2, "#1f77b4", "passed this year"), (0.2, 3, "#9aa7b8", "failed this year")]:
        est = np.array([r[k][0] for r in rows]) * 100
        err = np.array([[r[k][0] - r[k][1], r[k][2] - r[k][0]] for r in rows]).T * 100
        ax.bar(x + off, est, 0.4, yerr=err, capsize=4, color=col, label=lab)
        for xi, v, top in zip(x + off, est, est + err[1]):
            ax.text(xi, top + 0.3, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(100 * CHANCE[test], color="red", ls="--", lw=1,
               label="chance level (measured false-positive rate)")
    ax.text(len(rows) - 0.55, 100 * CHANCE[test] + 0.3, f"{100*CHANCE[test]:.1f}%",
            color="red", fontsize=8, ha="right")
    ax.set_xticks(x, [r[1] for r in rows], fontsize=9)
    ax.set_title(test)
axes[0].set_ylabel("% passing the same test the following year")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.suptitle("Does cointegration last? NIFTY 100 pairs, walk-forward, 95% intervals")
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig("02_persistence.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- Q3
edges = [0, 0.01, 0.05, 0.2, 1.0001]
labels = ["< 0.01", "0.01-0.05", "0.05-0.2", "> 0.2"]
b = pd.cut(df["eg_p"], edges, labels=labels, right=False)
say("Q3. By Engle-Granger p-value: still cointegrated next year / mean 6-month P&L (no protection)")
q3 = []
for lab in labels:
    per_ = ratio((b == lab) & df["eg_pass_next"].notna(), df["eg_pass_next"])
    pl = ratio((b == lab) & df["tradeable"], df["pnl_none"])
    q3.append((lab, per_, pl))
    say(f"    p {lab:9s} persistence {pct(per_)}   P&L {pnl(pl)}   (n = {pl[3]})")
say()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, (k, title) in enumerate([(1, "% still cointegrated next year"), (2, "mean 6-month P&L, no protection (%)")]):
    est = np.array([r[k][0] for r in q3]) * 100
    err = np.array([[r[k][0] - r[k][1], r[k][2] - r[k][0]] for r in q3]).T * 100
    ax[i].bar(labels, est, yerr=err, capsize=4, color="#1f77b4")
    ax[i].axhline(0, color="black", lw=0.8)
    ax[i].set_title(title)
    ax[i].set_xlabel("Engle-Granger p-value in formation year")
ax[0].axhline(100 * CHANCE["Engle-Granger"], color="red", ls="--", lw=1)
fig.suptitle("Is a smaller p-value a better pair?")
fig.tight_layout()
fig.savefig("03_pvalue_buckets.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- Q4
tr = df["tradeable"]
groups = [("Engle-Granger passed", tr & df["eg_pass"]),
          ("Johansen passed", tr & (df["jo_pass"] == 1)),
          ("both passed", tr & df["both"]),
          ("both failed", tr & ~df["eg_pass"] & (df["jo_pass"] == 0))]
say("Q4. Mean 6-month P&L with no protection, by selection (beta > 0)")
q4 = []
for name, m in groups:
    r = ratio(m, df["pnl_none"])
    lose = (df.loc[m, "pnl_none"] < 0).mean()
    q4.append((name, r))
    say(f"    {name:21s} {pnl(r)}   losing {100*lose:.0f}%   (n = {r[3]})")
for name, m in groups[:3]:
    d = diff(m, groups[3][1], df["pnl_none"])
    say(f"    {name} minus both failed: {pnl(d)}")
say()

# ---------------------------------------------------------------- Q5
say("Q5. Exit rules on pairs that passed Engle-Granger (beta > 0), next 6 months")
sel = tr & df["eg_pass"]
rules = [("pnl_none", "no protection"), ("pnl_stop", "stop-loss |z|>3.5"), ("pnl_cusum", "CUSUM alarm")]
q5 = []
for col, name in rules:
    r = ratio(sel, df[col])
    s = df.loc[sel, col]
    q5.append((name, r, (s < 0).mean(), s.quantile(0.05)))
    say(f"    {name:18s} mean {pnl(r)}   losing {100*(s < 0).mean():.0f}%   worst 5% below {100*s.quantile(0.05):+.1f}%")
fired = df[sel & df["alarm_day"].notna()]
eff = fired["pnl_cusum"] - fired["pnl_none"]
say(f"    CUSUM fired in {100*len(fired)/sel.sum():.0f}% of trades (median day {fired['alarm_day'].median():.0f}); "
    f"helped {100*(eff > 0).mean():.0f}%, hurt {100*(eff < 0).mean():.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
names = [q[0].replace(" passed", "\npassed").replace(" failed", "\nfailed") for q in q4]
est = np.array([q[1][0] for q in q4]) * 100
err = np.array([[q[1][0] - q[1][1], q[1][2] - q[1][0]] for q in q4]).T * 100
ax[0].bar(names, est, yerr=err, capsize=4, color=["#1f77b4", "#9467bd", "#2ca02c", "#9aa7b8"])
ax[0].axhline(0, color="black", lw=0.8)
ax[0].set_ylabel("mean 6-month P&L (%), 95% interval")
ax[0].set_title("Does the test pick better pairs?")
ax[0].tick_params(axis="x", labelsize=8)
names = [f"{q[0]}\nworst 5%: {100*q[3]:.1f}%" for q in q5]
est = np.array([q[1][0] for q in q5]) * 100
err = np.array([[q[1][0] - q[1][1], q[1][2] - q[1][0]] for q in q5]).T * 100
ax[1].bar(names, est, yerr=err, capsize=4, color=["#9aa7b8", "#ff7f0e", "#2ca02c"])
ax[1].set_ylabel("mean 6-month P&L (%), 95% interval")
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_title("Exit rules (Engle-Granger pairs)")
ax[1].tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig("04_selection_and_exits.png", dpi=120)
plt.close(fig)

open("results/summary.txt", "w").write("\n".join(out) + "\n")
print("\nsaved 01_passing_vs_chance.png, 02_persistence.png, 03_pvalue_buckets.png, "
      "04_selection_and_exits.png and results/summary.txt")
