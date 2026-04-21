# Spaceship Titanic — Binary Classification

Predicting which passengers were transported to an alternate dimension when the Spaceship Titanic collided with a spacetime anomaly. This is a Kaggle binary classification competition.

**Competition link:** https://www.kaggle.com/competitions/spaceship-titanic

---

## Project Overview

This project walks through an end-to-end binary classification workflow on the Spaceship Titanic dataset, covering:

- Exploratory data analysis (EDA) with visualizations
- Feature engineering (cabin parsing, passenger group detection, spending patterns)
- Principled missing value handling
- Multiple classification models (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning with Bayesian optimization
- Model explainability with SHAP
- Statistical comparison of models with hypothesis testing
- Kaggle submission

---

## Dataset

- ~8,700 training passengers, ~4,300 test passengers
- Target variable: `Transported` (True/False)
- Features: passenger demographics, cabin assignment, spending on ship amenities

**Note:** Raw data is not included in this repo per Kaggle competition rules. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the [competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and place them in `data/raw/`.

---

## Project Structure

```
Spaceship_Titanic_MP/
├── data/
│   ├── raw/                  # Original Kaggle files (gitignored)
│   └── processed/            # Cleaned / feature-engineered data (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb     # Model training and evaluation
├── submissions/              # Kaggle submission CSVs (gitignored)
├── models/                   # Saved trained models (gitignored)
├── src/                      # Reusable Python code
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Option 1: Using conda (recommended if you have Anaconda)

```bash
conda create -n spaceship_titanic python=3.11
conda activate spaceship_titanic
pip install -r requirements.txt
```

### Option 2: Using venv

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Download the data

1. Go to https://www.kaggle.com/competitions/spaceship-titanic/data
2. Download `train.csv`, `test.csv`, and `sample_submission.csv`
3. Place them in `data/raw/`

---

## Results

| Model | CV Accuracy | Kaggle Public Score |
|-------|-------------|---------------------|
| Logistic Regression (baseline) | 79.26% | — |
| Random Forest | 79.78% | — |
| XGBoost (tuned) | 80.41% | **0.80383** |

---

## Key Insights

*To be updated as the project progresses.*

---

## Tools & Libraries

- **Data manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Modeling:** scikit-learn, XGBoost
- **Hyperparameter tuning:** bayesian-optimization
- **Explainability:** SHAP
- **Statistical testing:** scipy

---

## Author

Built as a portfolio project to demonstrate end-to-end binary classification workflows.
