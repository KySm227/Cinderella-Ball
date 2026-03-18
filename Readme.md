# ⚽ Cinderella Ball — FIFA World Cup Upset Prediction

> _A Sports Analytics Club Project_

---

## 📌 Project Overview

**Cinderella Ball** is a data-driven sports analytics project centered on one of football's most thrilling phenomena — the **upset**. Named after the fairy-tale underdog who defied expectations, this project asks:

> _Can we predict which "smaller" nations are most likely to pull off a World Cup upset before the tournament even begins?_

Using historical FIFA World Cup **qualifying match data** and **international competition results**, we build a machine learning pipeline that assigns each team an **"upset probability score"** — a measure of how likely a lower-ranked team is to defeat a higher-ranked opponent. Think of it as finding the next Cinderella story before the ball even starts.

The project covers the full data science lifecycle:

- Collecting and cleaning real-world football match data
- Performing exploratory data analysis (EDA)
- Engineering meaningful features from raw match statistics
- Training and evaluating a machine learning model
- Visualizing and interpreting predictions

---

## 🌐 Data Collection

> ⚠️ **Note:** Web scraping is handled entirely by the project leads behind the scenes. Workshop participants will work directly with the pre-cleaned datasets.

### Sources

| Source                                                                      | What We Collect                                                            |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **[FBref.com](https://fbref.com)**                                          | Match-level stats: goals, xG, possession, shots, passes, defensive actions, World Cup qualification results, group standings, confederation brackets   |

### What Data We're Pulling

- **FIFA World Cup Qualifying Matches** (all confederations: UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC)
- **International Competition Results** (Nations League, Copa América, AFCON, etc.)
- **Team-level aggregate stats** per campaign: goals for/against, average xG, win/draw/loss records
- **FIFA World Rankings** at the time of each match

### How the Scraping Works (Overview)

Data is collected using `requests` and `BeautifulSoup` to parse FBref's match logs and Wikipedia's qualification tables. Raw data is exported as `.csv` files, then passed through a cleaning pipeline to handle missing values, normalize team names across sources, and standardize date formats before being handed off to workshop participants.

---

## 🧠 Machine Learning Model

### Problem Framing

This is a **binary classification** problem. For each match, we define an **upset** as:

> A team ranked **15+ positions lower** (per FIFA Rankings) defeating or drawing with a higher-ranked opponent.

The model predicts: `1 = Upset` | `0 = Expected Result`

---

### Primary Model: **XGBoost Classifier**

We use **XGBoost (Extreme Gradient Boosting)** as our primary model for the following reasons:

- Handles **tabular sports data** exceptionally well
- Robust to **missing values** (common in historical football data)
- Naturally captures **non-linear relationships** (e.g., a team's form matters more late in qualifying)
- Built-in **feature importance** scores, which help us interpret _why_ an upset is likely
- Strong baseline performance without heavy hyperparameter tuning

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=3,   # Handles class imbalance (upsets are rare)
    use_label_encoder=False,
    eval_metric='logloss'
)
```

---

### Feature Engineering

The following features are engineered from raw match and ranking data:

| Feature                 | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `fifa_rank_diff`        | Difference in FIFA rankings between the two teams |
| `avg_xg_last5`          | Average expected goals (xG) over last 5 matches   |
| `home_away_flag`        | Whether the match is home, away, or neutral       |
| `goals_scored_avg`      | Rolling average goals scored in qualifying        |
| `goals_conceded_avg`    | Rolling average goals conceded                    |
| `win_rate_last10`       | Win percentage over last 10 international matches |
| `confederation`         | Which FIFA confederation the team belongs to      |
| `qualification_stage`   | Group stage, playoff, intercontinental, etc.      |
| `days_since_last_match` | Rest/fatigue proxy                                |
| `rank_momentum`         | Change in FIFA ranking over past 6 months         |

---

### Secondary Model: **Logistic Regression** (Baseline Comparison)

We also train a **Logistic Regression** model as an interpretable baseline. This lets us:

- Validate that XGBoost is genuinely adding value over a simpler approach
- Communicate odds-style probabilities more intuitively to non-technical audiences

---

### Evaluation Metrics

Since upsets are **rare events** (~15–20% of matches), accuracy alone is misleading. We evaluate using:

- **ROC-AUC Score** — Overall discriminative power
- **Precision-Recall AUC** — Performance on the minority class (actual upsets)
- **F1 Score** — Balance of precision and recall
- **Confusion Matrix** — False positive/negative breakdown

---

## 🐍 Python Libraries

### Data Collection & Handling

| Library          | Purpose                            |
| ---------------- | ---------------------------------- |
| `requests`       | HTTP requests for scraping         |
| `beautifulsoup4` | HTML parsing for Wikipedia & FBref |
| `pandas`         | Data manipulation and analysis     |
| `numpy`          | Numerical operations               |

### Machine Learning

| Library            | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `scikit-learn`     | Preprocessing, model evaluation, Logistic Regression |
| `xgboost`          | Primary gradient boosting classifier                 |
| `imbalanced-learn` | Handling class imbalance (SMOTE oversampling)        |

### Visualization

| Library      | Purpose                           |
| ------------ | --------------------------------- |
| `matplotlib` | Base plotting                     |
| `seaborn`    | Statistical data visualization    |
| `plotly`     | Interactive charts and dashboards |

### Utilities

| Library   | Purpose                              |
| --------- | ------------------------------------ |
| `jupyter` | Notebook environment for workshops   |
| `tqdm`    | Progress bars for data loops         |
| `joblib`  | Model serialization (saving/loading) |

### Install All Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
pandas
numpy
requests
beautifulsoup4
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
plotly
jupyter
tqdm
joblib
```

---

## 📅 5-Week Workshop Curriculum

> Each session is approximately **60–90 minutes**. Participants should have basic Python familiarity. All datasets will be pre-loaded and ready to use.

---

### 🗓️ Week 1 — _Kickoff: Understanding the Problem & the Data_

**Goal:** Get everyone aligned on what we're building and comfortable with the dataset.

**Topics:**

- Intro to the Cinderella Ball project and its goals
- What is an "upset" in football? How do we define and measure it?
- Walkthrough of the dataset: what each column means, where it came from
- Introduction to `pandas`: loading, inspecting, and summarizing data
- Q&A on project expectations and team roles

**Hands-On Activity:**

```python
import pandas as pd
df = pd.read_csv('data/cleaned/qualifying_matches.csv')
df.head()
df.describe()
df['upset'].value_counts()
```

**Deliverable:** Each participant submits 3 observations about the data (e.g., "Group A teams had a higher average xG than Group C teams")

---

### 🗓️ Week 2 — _Exploratory Data Analysis (EDA)_

**Goal:** Let the data tell its story before we model anything.

**Topics:**

- Distributions of key variables (FIFA rank difference, goals scored, xG)
- Visualizing upset frequency by confederation, qualification stage, and year
- Correlation heatmaps between features
- Identifying data quality issues: missing values, outliers, encoding

**Hands-On Activity:**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['fifa_rank_diff'], bins=30)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
df.groupby('confederation')['upset'].mean().plot(kind='bar')
```

**Deliverable:** EDA notebook with at least 5 labeled visualizations and written interpretations

---

### 🗓️ Week 3 — _Feature Engineering & Data Prep_

**Goal:** Transform raw data into meaningful inputs the model can learn from.

**Topics:**

- What is feature engineering and why does it matter in sports analytics?
- Creating rolling averages (form metrics over last N games)
- Encoding categorical variables (`confederation`, `home_away_flag`)
- Train/test split and why we split _chronologically_ for sports data
- Handling class imbalance with SMOTE

**Hands-On Activity:**

```python
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Chronological split
train = df[df['year'] < 2018]
test  = df[df['year'] >= 2018]

X_train, y_train = train.drop('upset', axis=1), train['upset']
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

**Deliverable:** A `features.py` script that outputs a clean, model-ready dataframe

---

### 🗓️ Week 4 — _Model Training & Evaluation_

**Goal:** Train the XGBoost model, tune it, and honestly assess how well it works.

**Topics:**

- Intro to XGBoost: how gradient boosting works (conceptually)
- Baseline comparison with Logistic Regression
- Evaluation metrics: why accuracy fails us and what to use instead
- Cross-validation for sports data
- Feature importance: which variables matter most?

**Hands-On Activity:**

```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

model = XGBClassifier(n_estimators=200, scale_pos_weight=3, random_state=42)
model.fit(X_res, y_res)

preds = model.predict(X_test)
print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

**Deliverable:** Completed model notebook with ROC curve, confusion matrix, and feature importance bar chart

---

### 🗓️ Week 5 — _Predictions, Storytelling & Presentation_

**Goal:** Apply the model to current qualifying data and present findings like sports analysts.

**Topics:**

- Generating upset probability scores for upcoming/recent matches
- Building a "Cinderella Watchlist" — top 10 teams most likely to pull off an upset
- Creating an interactive Plotly dashboard of predictions
- How to communicate ML results to a non-technical audience
- Project retrospective: what worked, what didn't, what's next

**Hands-On Activity:**

```python
import plotly.express as px

predictions = pd.read_csv('outputs/predictions.csv')
fig = px.bar(
    predictions.sort_values('upset_prob', ascending=False).head(10),
    x='team', y='upset_prob',
    color='confederation',
    title='Top 10 Cinderella Contenders'
)
fig.show()
```

**Deliverable:** 5-minute group presentation — "Which team is our Cinderella pick and why?" backed by model output and EDA findings

---

## 🏆 Final Output

By the end of the workshop, each participant will have:

- ✅ Worked with real-world international football data
- ✅ Performed a full EDA and communicated findings visually
- ✅ Engineered features from raw match statistics
- ✅ Trained and evaluated an XGBoost classification model
- ✅ Generated data-backed predictions for real World Cup qualifiers
- ✅ Presented analytical findings to an audience

---
