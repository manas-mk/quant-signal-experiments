# 03_analyse.py
# Turn results/walkforward.csv into the answers:
#   Q1. How many pairs pass the cointegration test just by chance?
#   Q2. If a pair is cointegrated this year, is it still cointegrated next year?
#   Q3. Does a smaller p-value mean a more reliable pair, or a more profitable one?
#   Q4. Across hundreds of trades, which exit rule works: none, stop-loss or CUSUM alarm?
#
# Confidence intervals: a block bootstrap over formation dates (blocks of 4 quarters). Pairs on the
# same date share stocks and market moves, and neighbouring dates overlap in time, so treating the
# rows as independent would make every interval look far too narrow.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALPHA = 0.05
BLOCK = 4          # bootstrap block length, in formation dates (4 quarters = 1 year)
N_BOOT = 2000
rng = np.random.default_rng(0)

df = pd.read_csv("results/walkforward.csv", parse_dates=["form_end"])
n_pairs = df.groupby("form_end").size().iloc[0]
dates = np.sort(df["form_end"].unique())
df["passes"] = df["p_form"] < ALPHA
df["passes_next"] = df["p_next"] < ALPHA
df["tradeable"] = df["beta"] > 0            # a negative hedge ratio isn't a long/short pair
out = []                                    # lines for results/summary.txt


def say(line=""):
    print(line)
    out.append(line)


def boot(stat, data):
    """Moving-block bootstrap over formation dates. stat(frame) -> number. Returns (estimate, lo, hi)."""
    groups = {d: g for d, g in data.groupby("form_end")}
    ds = [d for d in dates if d in groups]
    est = stat(data)
    vals = []
    for _ in range(N_BOOT):
        picks = []
        while len(picks) < len(ds):
            s = rng.integers(0, len(ds) - BLOCK + 1)
            picks.extend(ds[s:s + BLOCK])
        sample = pd.concat([groups[d] for d in picks[:len(ds)]])
        vals.append(stat(sample))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return est, lo, hi


def fmt(t, pct=True):
    e, lo, hi = t
    return f"{100*e:.1f}% [{100*lo:.1f}, {100*hi:.1f}]" if pct else f"{e:+.4f} [{lo:+.4f}, {hi:+.4f}]"


n_stocks = len(set(df["a"]) | set(df["b"]))
say(f"{n_stocks} stocks, {n_pairs} pairs, {len(dates)} formation dates, {len(df)} pair-windows")
say(f"formation dates run from {pd.Timestamp(dates[0]).date()} to {pd.Timestamp(dates[-1]).date()}\n")

# ---------------------------------------------------------------- Q1: chance
per_date = pd.DataFrame({
    "passing": df.groupby("form_end")["passes"].sum(),
    "same": (df["passes"] & df["same_sector"]).groupby(df["form_end"]).sum(),
})
expected = ALPHA * n_pairs
say("Q1. How many pairs pass the test (p < 0.05) on each date?")
say(f"    average {per_date['passing'].mean():.1f} of {n_pairs}; about {expected:.1f} would pass by chance alone")
say(f"    range {per_date['passing'].min()} to {per_date['passing'].max()} per date")
share_same_all = df["same_sector"].mean()
share_same_pass = df.loc[df["passes"], "same_sector"].mean()
say(f"    same-sector pairs: {100*share_same_all:.0f}% of all pairs, {100*share_same_pass:.0f}% of passing pairs\n")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(per_date.index, per_date["passing"], width=60, color="#9aa7b8", label="all passing pairs")
ax.bar(per_date.index, per_date["same"], width=60, color="#1f77b4", label="of which same sector")
ax.axhline(expected, color="red", ls="--", label=f"expected by chance ({expected:.1f})")
ax.set_ylabel(f"pairs with p < {ALPHA} (of {n_pairs})")
ax.set_title("How many pairs look cointegrated on each date, versus pure chance")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("01_passing_vs_chance.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- Q2: persistence
has_next = df.dropna(subset=["p_next"])
rate = lambda g: g["passes_next"].mean()
say("Q2. Share of pairs cointegrated NEXT year (95% block-bootstrap interval)")
rows = []
for label, sub in [("all pairs", has_next),
                   ("same sector", has_next[has_next["same_sector"]]),
                   ("cross sector", has_next[~has_next["same_sector"]])]:
    yes = boot(rate, sub[sub["passes"]])
    no = boot(rate, sub[~sub["passes"]])
    rows.append((label, yes, no))
    say(f"    {label:13s} cointegrated this year: {fmt(yes)}   not cointegrated: {fmt(no)}"
        f"   (n = {sub['passes'].sum()} vs {(~sub['passes']).sum()})")
say()

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(rows))
for off, key, col, lab in [(-0.2, 1, "#1f77b4", "cointegrated this year"),
                           (0.2, 2, "#9aa7b8", "not cointegrated this year")]:
    est = np.array([r[key][0] for r in rows]) * 100
    err = np.array([[r[key][0] - r[key][1], r[key][2] - r[key][0]] for r in rows]).T * 100
    ax.bar(x + off, est, 0.4, yerr=err, capsize=4, color=col, label=lab)
    for xi, v, top in zip(x + off, est, est + err[1]):
        ax.text(xi, top + 0.3, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
ax.axhline(100 * ALPHA, color="red", ls="--", lw=1, label="5% (chance level)")
ax.set_xticks(x, [r[0] for r in rows])
ax.set_ylabel("% cointegrated the following year")
ax.set_title("Does cointegration last? (walk-forward, 95% intervals)")
ax.legend()
fig.tight_layout()
fig.savefig("02_persistence.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- Q3: p-value buckets
edges = [0, 0.01, 0.05, 0.2, 1.0001]
labels = ["< 0.01", "0.01-0.05", "0.05-0.2", "> 0.2"]
df["bucket"] = pd.cut(df["p_form"], edges, labels=labels, right=False)
tr = df[df["tradeable"]]
say("Q3. By formation p-value: still cointegrated next year / mean 6-month P&L (no protection)")
bucket_rows = []
for lab in labels:
    g_next = has_next[pd.cut(has_next["p_form"], edges, labels=labels, right=False) == lab]
    g_tr = tr[tr["bucket"] == lab]
    per = boot(rate, g_next)
    pnl = boot(lambda g: g["pnl_none"].mean(), g_tr)
    bucket_rows.append((lab, per, pnl, len(g_tr)))
    say(f"    p {lab:9s}  persistence {fmt(per)}   P&L {fmt(pnl, pct=False)}   (n = {len(g_tr)})")
say()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, (key, title, scale) in enumerate([(1, "% still cointegrated next year", 100),
                                         (2, "mean 6-month P&L, no protection (%)", 100)]):
    est = np.array([r[key][0] for r in bucket_rows]) * scale
    err = np.array([[r[key][0] - r[key][1], r[key][2] - r[key][0]] for r in bucket_rows]).T * scale
    ax[i].bar(labels, est, yerr=err, capsize=4, color="#1f77b4")
    ax[i].axhline(0, color="black", lw=0.8)
    ax[i].set_title(title)
    ax[i].set_xlabel("cointegration p-value in formation year")
ax[0].axhline(100 * ALPHA, color="red", ls="--", lw=1)
fig.suptitle("Is a smaller p-value a better pair?")
fig.tight_layout()
fig.savefig("03_pvalue_buckets.png", dpi=120)
plt.close(fig)

# ---------------------------------------------------------------- Q4: exit rules
sel = tr[tr["passes"]]
say(f"Q4. Trading the {len(sel)} pair-windows that passed (p < 0.05, beta > 0), next 6 months")
rules = [("pnl_none", "no protection"), ("pnl_stop", "stop-loss |z|>3.5"), ("pnl_cusum", "CUSUM alarm")]
summary = []
for col, name in rules:
    m = boot(lambda g, c=col: g[c].mean(), sel)
    lose = (sel[col] < 0).mean()
    worst = sel[col].quantile(0.05)
    summary.append((name, m, lose, worst))
    say(f"    {name:18s} mean {fmt(m, pct=False)}   losing {100*lose:.0f}%   worst 5% below {100*worst:+.1f}%")
fired = sel[sel["alarm_day"].notna()]
diff = fired["pnl_cusum"] - fired["pnl_none"]
say(f"    CUSUM fired in {100*len(fired)/len(sel):.0f}% of trades; when it fired it helped in "
    f"{100*(diff > 0).mean():.0f}% and hurt in {100*(diff < 0).mean():.0f}% (mean effect {100*diff.mean():+.2f}%)")
not_sel = tr[~tr["passes"]]
cmp_ = boot(lambda g: g.loc[g["passes"], "pnl_none"].mean() - g.loc[~g["passes"], "pnl_none"].mean(), tr)
say(f"    passing minus non-passing pairs, mean P&L (no protection): {fmt(cmp_, pct=False)}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
bins = np.linspace(sel[[c for c, _ in rules]].min().min(), sel[[c for c, _ in rules]].max().max(), 50)
for (col, name), c in zip(rules, ["#9aa7b8", "#ff7f0e", "#2ca02c"]):
    ax[0].hist(sel[col] * 100, bins=bins * 100, histtype="step", lw=1.6, color=c, label=name)
ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_xlabel("6-month P&L (% of gross capital)")
ax[0].set_ylabel("number of pair-windows")
ax[0].legend()
names = [s[0] for s in summary]
means = np.array([s[1][0] for s in summary]) * 100
errs = np.array([[s[1][0] - s[1][1], s[1][2] - s[1][0]] for s in summary]).T * 100
ax[1].bar(names, means, yerr=errs, capsize=4, color=["#9aa7b8", "#ff7f0e", "#2ca02c"])
ax[1].axhline(0, color="black", lw=0.8)
ax[1].set_ylabel("mean P&L (%) with 95% interval")
ax[1].tick_params(axis="x", labelsize=8)
fig.suptitle(f"Exit rules across {len(sel)} walk-forward trades")
fig.tight_layout()
fig.savefig("04_exit_rules.png", dpi=120)
plt.close(fig)

open("results/summary.txt", "w").write("\n".join(out) + "\n")
print("\nsaved 01_passing_vs_chance.png, 02_persistence.png, 03_pvalue_buckets.png, "
      "04_exit_rules.png and results/summary.txt")
