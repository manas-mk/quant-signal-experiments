# When does a pair break? Online detection of cointegration breakdown in NSE pairs

Pairs trading assumes the relationship between two stocks stays stable. This project asks what happens when it doesn't, and whether an **online alarm** can catch the breakdown early enough to limit losses.

The main pair, **GESHIP / SOMANYCERA**, comes from Dutta, Diwan & Chakrabarty, *ESG driven pairs algorithm for sustainable trading: Analysis from the Indian market* ([arXiv:2401.14761](https://arxiv.org/abs/2401.14761)). In that paper it had the best Sharpe ratio in training (1.11) but lost money in testing. **INDUSINDBK / SHRIRAMFIN**, which did well in that paper's test period, is used for comparison.

## Method

| Step | Script | What it does |
|---|---|---|
| 1 | `01_download.py` | Daily adjusted closes from Yahoo Finance, Jan 2019 – Dec 2024 (1,481 trading days) |
| 2 | `02_spread.py` | Rolling hedge ratio (120 days), spread z-score, rolling Engle–Granger test (250-day windows, every 20 days) |
| 3 | `03_trade.py` | **Formation / trading split.** The formation window is the 250-day window with the lowest cointegration p-value. β is fitted once and frozen; the strategy then trades the spread afterwards (enter at \|z\| > 2, exit at z = 0, 0.1% cost per trade). Offline change-points (binary segmentation) are found on the spread. |
| 4 | `04_alarm.py` | Two **online** exits compared with no protection: a stop-loss at \|z\| > 3.5, and a two-sided **CUSUM** alarm on the z-score (slack k = 0.5). The CUSUM threshold h is calibrated on the formation period only: it is set 20% above the highest value reached while the pair was "healthy". |

## Results

**1. Rolling cointegration is fragile.** Over about 50 rolling one-year tests, GESHIP/SOMANYCERA was cointegrated (p < 0.05) in only 5–6 windows, and those windows were scattered. With 50 tests at the 5% level, about 2–3 would pass by chance alone. Its hedge ratio swung from −1.5 to +1.3 over the period. A rolling z-score of the spread still *looked* mean-reverting throughout, because re-normalising every day hides a breakdown.

**2. Passing a cointegration test did not protect either pair.** Both formation windows were genuinely cointegrated, and both pairs lost money after formation.

**3. The online CUSUM alarm fired earlier than a stop-loss, but "earlier" is not automatically "better".** Cumulative log P&L after formation:

| Pair | Formation (p-value) | No protection: final (best along the way) | Stop-loss \|z\| > 3.5 | CUSUM alarm |
|---|---|---|---|---|
| GESHIP / SOMANYCERA | Aug 2019 – Sep 2020 (p = 0.014) | −0.189 (peak about **+0.5** in mid-2022) | −0.137 (fired after 142 days) | −0.022 (fired after 56 days) |
| INDUSINDBK / SHRIRAMFIN | Jul 2022 – Jul 2023 (p = 0.004) | −0.437 (never meaningfully positive) | −0.104 (after 377 days) | **0.000** (after 330 days) |

Days are calendar days after trading starts.

- **INDUSINDBK / SHRIRAMFIN, a clean win for the alarm.** The spread drifted slowly downwards for almost a year. The CUSUM crossed its threshold on the same day the z-score first reached −2, so the strategy never opened the trade that went on to lose 0.44.
- **GESHIP / SOMANYCERA, the trade-off.** The alarm fired after 8 weeks and avoided the final loss. But the unprotected strategy went on to make about **+0.5** by mid-2022 before giving it all back. By the end-date number, the alarm "wins" (−0.02 vs −0.19). Measured against what was achievable, it also gave up two profitable years. Judging an exit rule by final P&L alone hides this.

**4. Offline change-point detection confirms breakdowns but does not warn of them.** In step 3, the first change-point found by binary segmentation came *after* the P&L peak. That makes it useful for explaining what happened, but not for trading.

## Limitations

- **Only two pairs.** GESHIP already shows the key open question: the **cost of exiting too early**. Measuring it properly needs many pairs and a metric beyond final P&L, for example P&L over fixed horizons or the drawdown avoided versus the profit forgone.
- **Some hindsight in the formation choice.** The best formation window was chosen by scanning the whole 2019–2024 period. Trading happens only after formation, but the choice of window used later data.
- **Simplified costs.** No slippage, no short-selling constraints, and P&L is measured in log-spread units.

## Next step

A **persistence study** on about 20 NIFTY stocks (190 pairs): how often does a pair that tests cointegrated in one year stay cointegrated the next? And across all pairs, does the formation-calibrated CUSUM alarm catch real breakdowns without exiting too often from pairs that remain profitable?

## Run it

```bash
pip install yfinance statsmodels pandas matplotlib ruptures
python 01_download.py
python 02_spread.py
python 03_trade.py
python 04_alarm.py
```
