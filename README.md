# NCAA Tournament Matchup Predictor

This project uses advanced statistical features and machine learning (XGBoost) to predict the outcomes of NCAA men's basketball tournament matchups. It leverages historical KenPom and Barttorvik metrics and evaluates high vs. low seed team matchups to forecast results.

---

## Objective

Predict whether the higher-seeded team will win a given NCAA Tournament matchup using a feature-rich dataset derived from historical team stats.

---

## Data Sources

- **KenPom/Barttorvik Metrics**: Advanced team-level statistics (e.g., offensive efficiency, adjusted tempo).
- **Historical Matchups**: Seed and score data from previous NCAA tournaments.
- **2025 Matchups**: Provided CSV for generating current year predictions.

---

## Methodology

### 1. Data Preprocessing
- Filtered out data from 2025 for training.
- Joined each team in each matchup with its corresponding stats.
- Identified high seed vs. low seed teams in every matchup.

### 2. Feature Engineering
- Selected 30+ advanced metrics per team (tempo, efficiency, shooting, etc.).
- Converted matchup data to wide format (e.g., high_seed_feature1, low_seed_feature1, ...).

### 3. Model Training
- Model: **XGBoost** (gradient boosting decision trees).
- Optimized using **5-fold cross-validation**.
- Evaluated using:
  - **Log Loss** (confidence in predictions)
  - **Brier Score** (probability accuracy)

### 4. 2025 Predictions
- Cleaned and matched team names for 2025 bracket.
- Generated predictions for each matchup.
- Identified potential upsets (probability ≤ 0.5).

# Data Source

https://www.kaggle.com/datasets/nishaanamin/march-madness-data?resource=download&select=KenPom+Barttorvik.csv
