#  Titanic Survival Prediction — Machine Learning with Python

A machine learning project that predicts whether a passenger survived the Titanic disaster based on features like age, gender, passenger class, and fare.

---

##  Project Overview

The Titanic dataset is one of the most well-known datasets in data science. This project walks through a complete ML pipeline — from raw data to a trained model with **82%+ accuracy** — using Python and Scikit-learn.

---

##  Problem Statement

**Can we predict whether a passenger survived the Titanic, based on their personal details?**

This is a **binary classification** problem:
- `1` = Survived
- `0` = Did not survive

---

##  Dataset

- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **Size:** 891 passengers, 12 features
- **Key Features Used:**

| Feature | Description |
|---|---|
| Pclass | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Sex | Gender of passenger |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Fare | Ticket fare |
| Embarked | Port of embarkation (S, C, Q) |

---

##  Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
- **Tool:** Jupyter Notebook

---

##  Project Workflow

```
1. Data Loading       → Load train.csv using Pandas
2. Exploration        → Understand shape, columns, missing values
3. Data Cleaning      → Fix missing Age, drop Cabin, fill Embarked
4. Feature Engineering→ Convert Sex and Embarked to numbers
5. Model Training     → Random Forest Classifier (100 trees)
6. Evaluation         → Accuracy score + Confusion Matrix
7. Visualisation      → 4 charts saved as PNG
```

---

##  Results

| Metric | Score |
|---|---|
| Model | Random Forest Classifier |
| Accuracy | ~82% |
| Training Size | 712 passengers |
| Testing Size | 179 passengers |

### Key Findings:
- **Female passengers** had a significantly higher survival rate than males
- **1st class passengers** survived at a much higher rate than 3rd class
- **Fare and Age** were the most important features for the model

---

##  Visualisations

> Charts generated during the project:

- `chart1_survival_count.png` — Overall survival distribution
- `chart2_survival_by_gender.png` — Survival rate by gender
- `chart3_feature_importance.png` — Which features mattered most
- `chart4_confusion_matrix.png` — Model prediction accuracy breakdown

---

## 🚀 How to Run This Project

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/titanic-survival-prediction.git
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib scikit-learn
```

3. Open the notebook:
```bash
jupyter notebook titanic.ipynb
```

4. Run all cells from top to bottom.

---

## 📁 File Structure

```
titanic-survival-prediction/
│
├── titanic.ipynb                  ← Main notebook with all code
├── train.csv                      ← Training dataset
├── chart1_survival_count.png      ← Visualisation 1
├── chart2_survival_by_gender.png  ← Visualisation 2
├── chart3_feature_importance.png  ← Visualisation 3
├── chart4_confusion_matrix.png    ← Visualisation 4
└── README.md                      ← This file
```
