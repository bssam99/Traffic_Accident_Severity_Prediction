# 🚦 Traffic Accident Severity Prediction in Toronto

A machine learning project that predicts the severity of traffic accidents in Toronto using historical collision data from the Toronto Police Service Open Data Portal.

---

## 📌 Overview

Traffic accidents are a serious public safety issue in urban environments. This project frames accident severity prediction as a **binary classification task** — distinguishing between low-severity and high-severity collisions — using environmental, temporal, and road infrastructure features.

Several classical machine learning models were implemented, compared, and combined through ensemble learning to maximize predictive performance.

---

## 📂 Repository Structure

```
├── Traffic_Accident_Severity_Prediction.ipynb   # Main notebook (preprocessing, training, evaluation)
├── README.md
└── report/
    └── Traffic_Accident_Severity_Prediction_Report.pdf
```

---

## 📊 Dataset

**Source:** [Toronto Police Service — Motor Vehicle Collisions Open Data](https://data.torontopolice.on.ca/pages/motor-vehicle-collisions)

Features include:
- **Environmental:** weather conditions, temperature, humidity, visibility, wind speed, precipitation
- **Road infrastructure:** traffic signals, crossings, junctions, stop signs
- **Temporal:** hour of day, day of week, month, rush hour & nighttime indicators

Target variable: Binary accident severity (Low / High)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handled missing values
- One-hot encoded categorical variables (weather, lighting conditions)
- Grouped original severity levels into two binary classes

### 2. Feature Engineering
Extracted temporal features:
- Hour of the day
- Day of the week
- Month of the year
- Rush hour indicator
- Nighttime indicator

### 3. Models Implemented

| Model | Type |
|---|---|
| Naive Bayes | Baseline |
| Logistic Regression | Linear |
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| Hard Voting | Ensemble |
| Soft Voting | Ensemble |
| Stacking (RF + GB → LR) | Ensemble |

### 4. Evaluation
All models were evaluated using **5-fold cross-validation** with the following metrics:
- Accuracy
- Weighted F1-score
- Precision
- Recall
- ROC-AUC

---

## 📈 Results

| Model | Accuracy | F1-W | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| **Stacking (RF+GB→LR)** | **75.2%** | **0.752** | **0.753** | **0.752** | **0.843** |
| Gradient Boosting | 75.0% | 0.750 | 0.752 | 0.750 | 0.834 |
| Soft Voting | 73.4% | 0.733 | 0.741 | 0.734 | 0.819 |
| Hard Voting | 73.4% | 0.732 | 0.742 | 0.734 | 0.734 |
| Random Forest | 71.7% | 0.714 | 0.729 | 0.717 | 0.795 |
| Decision Tree | 69.4% | 0.687 | 0.704 | 0.694 | 0.764 |
| Logistic Regression | 65.6% | 0.656 | 0.657 | 0.656 | 0.710 |
| Naive Bayes | 61.2% | 0.606 | 0.612 | 0.612 | 0.688 |

> **Best model:** Stacking ensemble (RF + Gradient Boosting → Logistic Regression) with 75.2% accuracy and 0.843 ROC-AUC.

---

## 🔍 Key Findings

Feature importance analysis identified the most influential predictors of accident severity:

- ⏱️ **Accident duration**
- 📏 **Distance**
- 🚦 **Traffic signals**
- 🛤️ **Road crossings**
- 🕐 **Time of day**
- 📅 **Month of the year**

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run the Notebook
```bash
jupyter notebook Traffic_Accident_Severity_Prediction.ipynb
```

---

## 📚 References

- Toronto Police Service. [Motor Vehicle Collisions Open Data](https://data.torontopolice.on.ca/pages/motor-vehicle-collisions)
- Breiman, L. *Random Forests.* Machine Learning Journal, 2001.
- Chen, T., & Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- World Health Organization. [Global Status Report on Road Safety](https://www.who.int/publications/i/item/9789241565684)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
