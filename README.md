# Traffic Accident Severity Prediction

Predicting whether a traffic accident will be **low or high severity** from weather, time of day and road features, using classical ML models and ensembles. Built on a **7.7-million-record** accident dataset.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

> CSAI-801 Machine Learning course project · Contributors: Bassam Ahmed, Ahmed Magdi

## Results

Test set: 49,600 accidents (20% stratified hold-out, never used for EDA or tuning).

| Model | Accuracy | F1 (weighted) | ROC-AUC |
|---|---:|---:|---:|
| **Stacking (RF + GB → LR)** | **75.2%** | **0.752** | **0.834** |
| Gradient Boosting | 75.2% | 0.751 | 0.834 |
| Soft Voting | 73.4% | 0.732 | 0.819 |
| Random Forest | 71.7% | 0.714 | 0.795 |
| Decision Tree | 69.2% | 0.687 | 0.764 |
| Logistic Regression | 65.6% | 0.656 | 0.710 |
| Naive Bayes | 61.2% | 0.606 | 0.680 |

Gradient Boosting gets almost the same score as the stacking ensemble while training **12× faster** (3.5 s vs 43 s), which makes it the practical choice.

**Most important features:** accident duration, affected road distance, traffic signals, crossings, hour of day and month.

## Approach

1. **Split first** – the raw data is cleaned (duplicates, missing values), balanced to 62,000 accidents per severity level (248,000 rows), then split 80/20 *before* any EDA or feature engineering to avoid data leakage.
2. **Target** – severity levels 1–2 → Low, 3–4 → High (binary classification).
3. **Features** – weather (temperature, humidity, visibility, wind, precipitation, condition), road features (signals, crossings, junctions, stops…), daylight, and engineered time features (hour, weekday, month, rush hour, night).
4. **Preprocessing pipeline** – scaling and one-hot encoding fit on the training data only.
5. **Models** – baselines (Naive Bayes, Logistic Regression, Decision Tree), tuned Random Forest and Gradient Boosting (sensitivity analysis on trees, depth and learning rate), then hard/soft voting and stacking.
6. **Evaluation** – accuracy, weighted F1, precision, recall, ROC-AUC and confusion-matrix error analysis (missed high-severity accidents vs false alarms).

## Limitations

- Accident duration and affected distance are only known after the accident. A model meant to run at the moment of the accident should drop them and be re-evaluated.
- The data was balanced by sampling, so real-world precision for the rarer high-severity class will be lower than shown here.

## Project Structure

```text
├── Traffic_Accident_Severity_Prediction.ipynb   # full pipeline
├── report/
│   └── Traffic_Accident_Severity_Prediction_Report.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

1. Download the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle and save it next to the notebook as `Traffic_Collisions_Open_Data.csv` (the notebook uses the 33 columns listed in section 2.1).
2. Install and run:

```bash
git clone https://github.com/bssam99/Traffic_Accident_Severity_Prediction.git
cd Traffic_Accident_Severity_Prediction
pip install -r requirements.txt
jupyter notebook Traffic_Accident_Severity_Prediction.ipynb
```

## References

- Moosavi et al., *A Countrywide Traffic Accident Dataset*, 2019 – [arXiv:1906.05409](https://arxiv.org/abs/1906.05409)
- Breiman, *Random Forests*, Machine Learning, 2001
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of Statistics, 2001

## License

[MIT](LICENSE)
