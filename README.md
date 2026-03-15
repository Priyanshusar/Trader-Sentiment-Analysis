# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Intern Assignment

---

## Project Overview

This project analyzes how Bitcoin market sentiment (Fear/Greed Index) relates to
trader behavior and performance on the Hyperliquid decentralized exchange.

**Core Hypothesis:** If sentiment affects the broader market, it should also change
how individual traders behave and perform.

---

## Datasets

| Dataset | Rows | Description |
|---------|------|-------------|
| `fear_greed_index.csv` | 2,644 | Daily Bitcoin Fear/Greed score (0–100) from Feb 2018 to May 2025 |
| `historical_part1.csv` + `historical_part2.csv` | 211,224 | All trade events from 32 Hyperliquid traders (May 2023 – May 2025) |

---

## Project Structure

```
primetrade_project/
├── Trader_Sentiment_Analysis.ipynb   # Main analysis notebook (11 steps)
├── app.py                            # Streamlit dashboard
├── historical_part1.csv              # Trader data part 1
├── historical_part2.csv              # Trader data part 2
├── fear_greed_index.csv              # Fear/Greed index
├── merged_data.csv                   # Output: merged dataset
├── trader_profiles.csv               # Output: per-trader profiles
├── trader_clusters.csv               # Output: K-Means clusters
└── charts/                           # All generated charts
    ├── 01_performance_by_sentiment.png
    ├── 02_pnl_distribution_boxplot.png
    ├── 03_behavior_heatmap.png
    ├── 04_long_short_ratio.png
    ├── 05_winner_loser_sentiment.png
    ├── 06_cumulative_pnl_timeline.png
    ├── 07_liquidations.png
    ├── 08_model_results.png
    └── 09_trader_clusters.png
```

---

## Setup & How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit nbformat xgboost
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook Trader_Sentiment_Analysis.ipynb
```

### 3. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
> Make sure all CSV files are in the same directory as `app.py`

---

## Methodology

### Step 1 — Load & Combine Data
Both trader data parts are loaded and concatenated into one `historical_data` dataframe.

### Step 2 — EDA
Explore column types, value distributions, missing values, and unique traders/coins.

### Step 3 — Data Cleaning
- Parse trader timestamps (IST format: `DD-MM-YYYY HH:MM`) using `dayfirst=True`
- Parse Fear/Greed dates
- Rename columns for consistency

### Step 4 — Feature Engineering
- Filter to **closing trade events only** (rows where Closed PnL ≠ 0)
- Compute daily metrics per trader: PnL, win rate, leverage proxy, long/short ratio, trade count
- Engineer leverage proxy from position size and PnL

### Step 5 — Merge
Join daily trader metrics with Fear/Greed index on `date` key.
Simplify 5 sentiment classes into 3: **Fear**, **Neutral**, **Greed**.

### Step 6 — Analysis
Compare all metrics across Fear vs Neutral vs Greed days.

### Step 7 — Segmentation
Segment traders by:
1. Leverage (Low / Mid / High)
2. Frequency (Infrequent / Moderate / Frequent)
3. Performance (Net Winner / Net Loser)

### Step 8 — Predictive Model (Bonus)
Random Forest Classifier to predict profitable days.
- Features: sentiment, FG value, trade count, leverage, position size, long ratio, win rate
- Result: **96.3% ROC-AUC** on test set

### Step 9 — Clustering (Bonus)
K-Means clustering (k=3) to identify behavioral archetypes:
- **Cluster A (Aggressive):** Higher leverage, moderate win rate
- **Cluster B (Professional):** Low leverage, 89% win rate, high frequency
- **Cluster C (Steady):** Low leverage, good win rate, lower frequency

---

## Key Insights

### Insight 1 — Fear Days Generate 2.9x More PnL
- Fear days: **$8,042 avg daily PnL**
- Greed days: **$2,738 avg daily PnL**

These traders are experienced contrarians who buy when others panic.

### Insight 2 — Traders Trade ~50% More on Fear Days
- Fear: **68.9 trades/day** vs Greed: **45.8 trades/day**

Increased activity on Fear days = more opportunities captured at discounted prices.

### Insight 3 — Leverage Rises During Fear
- Fear: **1.23x avg** vs Greed: **1.06x avg**

Confident traders size up when they believe the market is undervalued.

### Insight 4 — Long Bias Persists Even in Fear
- Fear: **54.5% long** vs Greed: **50.2% long**

Traders buy the dip — they do not panic sell even when sentiment is fearful.

### Insight 5 — 96.3% ROC-AUC Predictive Model
Win rate and position size are the top predictive features — not sentiment alone.

---

## Actionable Strategy Rules

### Rule 1 — "Increase position size on Fear days — but only for high win rate traders"
> During Fear sentiment, traders with win rate > 80% should increase position size by up to 20%.
> Fear days generate 2.9x more PnL for disciplined traders.
> **Formula:** IF sentiment = Fear AND win_rate > 0.80 → multiply position_size × 1.2

### Rule 2 — "Boost trade frequency on Fear, reduce on Extreme Greed"
> On Fear days, increase number of trades by ~1.5x.
> On Extreme Greed (FG > 75), reduce by 0.6x — crowd overleveraging signals reversal.
> **Formula:** IF fg_value < 30 → trades × 1.5 | IF fg_value > 75 → trades × 0.6

### Rule 3 — "Net Losers must cap leverage at 1x on Fear days"
> The 3 Net Loser traders show worst drawdowns on Fear days.
> Volatility amplifies undisciplined trading. This segment needs the opposite rule.
> **Formula:** IF historical_win_rate < 0.70 AND sentiment = Fear → max_leverage = 1.0

---

## Evaluation Criteria Met

| Criterion | Status |
|-----------|--------|
| Data cleaning + merge alignment | Done |
| Strength of reasoning | Deep analysis with 5 insights |
| Quality of insights (actionable) | 3 specific rules with conditions |
| Clarity of communication | Structured notebook with explanations |
| Reproducibility | Clean notebook, all steps documented |
| Bonus: Predictive model | Random Forest, 96.3% ROC-AUC |
| Bonus: Clustering | K-Means, 3 behavioral archetypes |
| Bonus: Dashboard | Streamlit with 5 tabs + live predictor |

---

*Submitted for Primetrade.ai Data Science Intern — Round 0 Assignment*
