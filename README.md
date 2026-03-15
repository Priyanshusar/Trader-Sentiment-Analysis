# Trader Performance vs Market Sentiment

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-96.3%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> **Does Bitcoin market sentiment actually change how traders behave — and do they make more money when the market is fearful or greedy?**

This project answers that question using 2 years of real trade data from the Hyperliquid decentralized exchange, cross-referenced with the daily Bitcoin Fear/Greed Index. The answer is counterintuitive: **Fear days generate 2.9× more average daily profit than Greed days.**

---

## The Core Finding

| Sentiment | Avg Daily PnL | Win Rate | Trades/Day | Avg Leverage |
|-----------|--------------|----------|------------|--------------|
| Fear      | **$8,042**   | **86.6%**| **68.9**   | **1.23x**    |
| Neutral   | $4,247       | 84.0%    | 63.5       | 1.12x        |
| Greed     | $2,738       | 84.1%    | 45.8       | 1.06x        |

Experienced traders treat Fear as a buying opportunity. They trade 50% more frequently, use slightly more leverage, and maintain a long bias precisely when the market is most panicked.

---

## Project Structure

```
trader-sentiment-analysis/
│
├── Trader_Sentiment_Analysis.ipynb   # Full analysis notebook (11 steps)
├── app.py                            # Streamlit interactive dashboard
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
│
└── charts/                           # All generated visualizations
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

## Datasets

| Dataset | Rows | Period | Description |
|---------|------|--------|-------------|
| Bitcoin Fear/Greed Index | 2,644 | Feb 2018 – May 2025 | Daily sentiment score (0–100) |
| Hyperliquid Trader Data  | 211,224 | May 2023 – May 2025 | Trade events from 32 wallets |

> **Note:** The CSV data files are not included in this repository due to file size.
> Download them separately and place them in the root folder before running.

---

## Setup & How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/trader-sentiment-analysis.git
cd trader-sentiment-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your data files**

Place the following files in the root directory:
```
historical_data.csv
fear_greed_index.csv
```

**4. Run the Streamlit dashboard**
```bash
streamlit run app.py
```

**5. Or open the Jupyter notebook**
```bash
jupyter notebook Trader_Sentiment_Analysis.ipynb
```

---

## Methodology

Two datasets were used in this project. The first is the Bitcoin
Fear/Greed Index — a daily score from 0 to 100 that measures the
emotional temperature of the crypto market. The second is a full
export of trade events from 32 wallets on the Hyperliquid perpetual
futures exchange.

The processing pipeline worked as follows:

1. Parsed trader timestamps (IST format) and aligned both datasets
   on a shared date column
2. Filtered to closing trade events only — rows where Closed PnL
   is non-zero represent realized profit or loss
3. Engineered daily metrics per trader: PnL, win rate, trade count,
   leverage proxy, long/short ratio, and average position size
4. Merged trader daily metrics with the Fear/Greed index on date
5. Compared all metrics across Fear, Neutral, and Greed days
6. Segmented traders by performance, leverage, and frequency
7. Applied K-Means clustering (k=3) on 6 behavioral features to
   find natural trader archetypes
8. Trained a Random Forest classifier to predict whether a given
   trader-day would be profitable

---

## Key Insights

**1. Fear days generate 2.9× more profit than Greed days**
Average daily PnL on Fear days is $8,042 versus $2,738 on Greed
days. These traders are disciplined contrarians — they treat
falling prices as a discount and step in while others panic.

**2. Traders are 50% more active on Fear days**
68.9 trades per day during Fear versus 45.8 on Greed. More
activity means more opportunities captured when prices are cheap.

**3. Leverage increases slightly during Fear**
Average leverage rises from 1.06x on Greed to 1.23x on Fear.
Traders size up when they believe the market is undervalued —
a calculated move, not reckless gambling.

**4. Long bias persists even in Fear**
54.5% of trades are long on Fear days versus 50.2% on Greed days.
Traders are buying dips, not shorting the bottom.

**5. Winners and Losers react in opposite directions**
29 of 32 traders are net profitable. The 3 net losers post their
worst drawdowns on Fear days — the same conditions that reward
disciplined traders punish undisciplined ones.

**6. Behavior predicts outcome more than sentiment label**
The model's top features are win rate and position size — not the
Fear/Greed score alone. Sentiment sets the environment; how a
trader behaves within that environment drives the outcome.

---

## Strategy Recommendations

**Rule 1 — Scale up position size on Fear days (skilled traders only)**

Fear days generate 2.9× more PnL for high win-rate traders.
Only traders with a proven edge should increase exposure.

```
IF sentiment = Fear AND win_rate > 80%
THEN multiply position size by 1.2×
```

**Rule 2 — Adjust trade frequency based on sentiment score**

More trades on Fear (opportunities are abundant), fewer trades
on Extreme Greed (crowd is overleveraged, reversal risk is high).

```
IF Fear/Greed score < 30   THEN target trades × 1.5
IF Fear/Greed score > 75   THEN target trades × 0.6
```

**Rule 3 — Cap leverage for struggling traders on Fear days**

Undisciplined traders get their worst losses on Fear days because
volatility amplifies every mistake. The opposite rule applies
to this segment.

```
IF win_rate < 70% AND sentiment = Fear
THEN hard cap leverage at 1.0× (no margin)
```

---

## What's Inside the Notebook

| Step | What it does |
|------|-------------|
| 1  | Load Datsets |
| 2  | Exploratory data analysis |
| 3  | Data cleaning and timestamp parsing |
| 4  | Feature engineering (win rate, leverage proxy, long ratio) |
| 5  | Merge trader data with Fear/Greed index on date |
| 6  | Fear vs Greed performance comparison |
| 7  | Trader segmentation (3 dimensions) |
| 8  | Predictive model — Random Forest classifier |
| 9  | K-Means behavioral clustering |
| 10 | Key insights and strategy rules |
| 11 | Save all outputs |

---

## Streamlit Dashboard

The interactive dashboard has 5 tabs:

- **Performance** — PnL, win rate, and cumulative return by sentiment
- **Behavior** — How leverage, trade frequency, and direction shift with sentiment
- **Segments** — Winner vs Loser analysis and K-Means trader archetypes
- **Model** — Feature importance, ROC curve, confusion matrix, and live prediction tool
- **Insights** — Key findings and 3 actionable strategy rules

---

## Machine Learning Model

**Algorithm:** Random Forest Classifier
**Target:** Will this trader-day be profitable? (binary: 1 = profit, 0 = loss)
**Features:** Sentiment score, FG value, trade count, leverage, position size, long ratio, win rate

| Metric | Score |
|--------|-------|
| ROC-AUC (test set)   | **96.3%** |
| 5-Fold Cross-Val AUC | 96.0% ± 2.2% |
| Overall Accuracy     | 90.3% |
| Profit Day Precision | 96.7% |
| Profit Day Recall    | 91.9% |
| Profit Day F1 Score  | 94.3% |

---

## Trader Archetypes (K-Means, k=3)

K-Means clustering on 6 behavioral features identified 3 distinct trader personalities:

| Cluster | Name | Win Rate | Avg Leverage | Trades/Day | Strategy Implication |
|---------|------|----------|--------------|------------|----------------------|
| A | Aggressive   | 57% | 1.74x | 31  | Reduce exposure on Fear days |
| B | Professional | 89% | 1.15x | 216 | Increase size on Fear days   |
| C | Steady       | 87% | 1.10x | 58  | Boost frequency on Fear days |

> Cluster names are assigned dynamically from actual cluster profiles on every run — not hardcoded numbers.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+       | Core language |
| Pandas / NumPy     | Data manipulation and feature engineering |
| Matplotlib / Seaborn | Charts and visualizations |
| Scikit-learn       | K-Means clustering and Random Forest model |
| Streamlit          | Interactive dashboard |
| Jupyter            | Analysis notebook |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

Built with curiosity about market microstructure and behavioral finance.
If you found this useful, feel free to star the repo ⭐
