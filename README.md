[README (1).md](https://github.com/user-attachments/files/28164962/README.1.md)
# NFL Power Ratings Model
### Built by Shaan Jagtiani | Python, Pandas, nflfastR, scikit-learn

---

## About This Project

This is my second Python sports trading project, building directly on the market analysis work from Project 1. Where Project 1 analyzed existing betting market inefficiencies this project asks a different question: can I build my own team strength model from scratch and generate predicted spreads that have real predictive value?

The model uses EPA (Expected Points Added) to rate all 32 teams based on offensive and defensive efficiency. I train the model on weeks 1-14 of the 2024 NFL season, generate predicted spreads for weeks 15-17 and validate those predictions against actual Vegas closing lines and game outcomes.

This is called out-of-sample validation, the gold standard for testing whether a model has genuine predictive power rather than just fitting past data.

---

## Project Overview

**Data:** nflfastR 2024 play-by-play (49,492 plays, 372 columns) + spreadspoke historical lines (285 games)

**Training period:** Weeks 1-14 (36,180 plays)

**Test period:** Weeks 15-17 (8,310 plays, week 18 excluded due to teams resting starters)

**Core question:** Does an EPA-based power ratings model generate predicted spreads with real predictive value compared to Vegas closing lines?

---

## Setup

### Requirements
```bash
pip install pandas numpy matplotlib scikit-learn pyarrow
```

### Data
- Download `play_by_play_2024.parquet` from [nflverse GitHub](https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2024.parquet) and place it in the same folder as the notebook
- Download `spreadspoke_scores.csv` from [Kaggle](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data) and place it in the same folder as the notebook

---

## Model Architecture

### Cell 1: Imports and Data Load
Loads the nflfastR 2024 play-by-play dataset and the spreadspoke historical lines. Splits the data into training (weeks 1-14) and test (weeks 15-17) sets.

---

### Cell 2: EPA Metrics with Exponential Decay Weighting
Calculates 8 team metrics from weeks 1-14 with exponential decay weighting applied so recent games count more than early season games. Week 14 carries a weight of 1.0 and each prior week is worth 85% of the next; all the way back to week 1 at roughly 0.10.

**The 8 metrics:**

| Metric | Description |
|---|---|
| Offensive EPA per play | Overall offensive efficiency per snap |
| Passing EPA per dropback | Most predictive metric of future wins |
| Rushing EPA per carry | Ground game efficiency |
| Offensive success rate | % of plays that gained enough yards to stay on schedule |
| Defensive EPA allowed | How well the defense suppressed opponent efficiency |
| Defensive success rate allowed | % of opponent plays that succeeded |
| Turnover differential per game | Takeaways minus giveaways normalized per game |
| All metrics decay weighted | Exponential weighting applied to all of the above |

**Why decay weighting:** A team that started slow in September but peaked in November is genuinely different from one that was consistently mediocre. Simple averages treat both teams identically. Decay weighting captures current team quality while accounting for the full season.

---

### Cell 3: Ridge Regression Power Ratings
Uses Ridge Regression to learn the optimal weights for each metric rather than setting them manually. The regression takes the difference in each metric between home and away teams and learns which combination best predicted game margins across 208 training games.

**Why Ridge Regression:** Regular linear regression overfits when metrics are correlated. Offensive EPA and passing EPA are related so a standard regression assigns extreme weights to both. Ridge regression adds a penalty for large coefficients forcing the model to spread weight more sensibly.

**Home field advantage:** Calculated from 1,890 historical games between 2015 and 2023 excluding the 2020 COVID season. The data-derived value of 2.04 points is applied as a fixed adjustment when generating spreads.

**Learned feature weights:**

| Feature | Coefficient | Interpretation |
|---|---|---|
| Turnover differential | +1.801 | Strongest positive predictor |
| Passing EPA | +1.629 | Second most important |
| Offensive EPA | +1.532 | Overall offensive efficiency |
| Offensive success rate | +1.034 | Drive sustainability |
| Rushing EPA | +0.891 | Least important offensive metric |
| Defensive success rate | -0.711 | Stopping opponent drives |
| Defensive EPA allowed | -2.087 | Strongest overall predictor |

**R² = 0.334** on 208 training games.

**Team ratings heading into week 15:**

| Rank | Team | Rating |
|---|---|---|
| 1 | DET | +11.5 |
| 2 | BUF | +10.1 |
| 3 | PHI | +8.7 |
| 4 | BAL | +7.9 |
| 5 | WAS | +6.9 |
| ... | ... | ... |
| 30 | TEN | -5.2 |
| 31 | DAL | -6.2 |
| 32 | LV | -7.1 |

---

### Cell 4: Generate Predicted Spreads for Weeks 15-17
Applies the power ratings to every week 15-17 matchup using the formula:

```
Predicted spread = Home rating - Away rating + Home field advantage (2.04)
```

Merges predicted spreads against the spreadspoke consensus closing lines for direct comparison. Week 18 is excluded because teams rest starters making results unreliable for model validation.

---

### Cell 5: Model Evaluation and Value Games
Evaluates model performance across 48 out-of-sample games.

**Key results:**

| Metric | Value |
|---|---|
| Model MAE | 10.67 points |
| Vegas MAE | 10.39 points |
| Difference | 0.28 points |
| Value games (3+ point disagreement) | 22 games |
| Cover rate on value games | 54.5% |
| Break-even threshold | 52.4% |

The model came within 0.28 points of Vegas closing line accuracy using only 8 metrics and no real-time injury or roster information.

On the 22 games where the model disagreed with Vegas by 3 or more points the model's preferred side covered 54.5% of the time, above the 52.4% break-even threshold. This is an encouraging result but not a conclusion. 22 games is a small sample and a proper evaluation would require multiple seasons of out-of-sample testing.

---

### Cells 6-8: Visualizations

**Cell 6:** Power ratings bar chart with all 32 teams ranked from best to worst heading into week 15

**Cell 7:** Predicted vs actual scatter plot with each dot representing one game. The dots loosely follow the perfect prediction diagonal confirming the model has genuine predictive signal

**Cell 8:** Value games analysis: color coded scatter showing which direction the model disagreed with Vegas and whether the preferred side covered

---

## Key Findings

**Turnover differential was the strongest positive predictor** of game margin in the 2024 NFL season ahead of even passing EPA. Ball security was more predictive of winning margins than passing efficiency in this sample. Detroit and Buffalo led both the turnover differential and the final ratings.

**Defense was the strongest overall predictor.** Defensive EPA allowed had the largest coefficient in the regression at -2.087 confirming the "defense wins championships" principle in the data.

**The model came within 0.28 points of Vegas accuracy** with no injury data, no weather adjustments and no sharp money signals. Vegas employs full teams of analysts with access to all of those inputs, the gap being this small validates the EPA-based approach.

---

## Limitations

- No real-time injury or roster data: the model's biggest misses (like LV @ NO) were games where teams played significantly below their season metrics due to late-season injuries the model could not anticipate
- Season-long metrics have inherent lag: a team that changed dramatically after week 10 still carries early season data in their rating
- 48 game test sample is meaningful but not large enough for definitive conclusions about long-term predictive value
- The spreadspoke closing line is consensus-based; the exact source book is not specified

---

## Next Steps

Project 3 will apply this same framework to the NBA using net rating metrics. More games per season means a larger out-of-sample test set and more statistically robust conclusions about model accuracy and market efficiency.

---

## Data Sources

- [nflverse nflfastR](https://github.com/nflverse/nflverse-data) — 2024 NFL play-by-play data
- [Kaggle: NFL Scores and Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data) — historical scores and spread lines 1967-2024
