# Anomaly Detection in Highly Imbalanced Transactional Data

This project focuses on detecting fraudulent transactions in a highly imbalanced credit card dataset using various machine learning algorithms. It includes comprehensive exploratory data analysis (EDA), data preprocessing, model training, and evaluation techniques to build a robust fraud detection pipeline.


---

## Dataset

- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (~0.17%)
- **Features**:
  - `Time`, `Amount`: raw numerical features
  - `V1` to `V28`: result of PCA transformation for privacy
  - `Class`: target variable (0 = non-fraud, 1 = fraud)

---

## Exploratory Data Analysis (EDA)

Key EDA steps included:
- Class imbalance analysis (only 0.17% fraud)
- Distribution plots for transaction amounts (fraud vs non-fraud)
- Correlation heatmap of anonymized features
- Visualizations of how fraud transactions differ in feature space
- Suggestions for feature scaling (`RobustScaler` for `Amount` and `Time`)
- Future suggestions: PCA or t-SNE visualization for separability

---

## Machine Learning Models

We experimented with the following classifiers:
- Logistic Regression

Future expansion:
- XGBoost, LightGBM, and CatBoost

---

## Handling Class Imbalance

Due to the severe class imbalance, several resampling techniques were applied:

- **Under-sampling**: NearMiss  
- **Over-sampling**: SMOTE, ADASYN  

All resampling was applied carefully to avoid data leakage by ensuring it's done only on the training data inside cross-validation.

---

## Evaluation Metrics

Given the imbalanced nature of the data, we evaluated models using metrics that better reflect performance in fraud detection:

- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- **ROC-AUC Score**
- **Precision-Recall Curve**
- **Cross-Validation Accuracy**

---

## Results Summary

| Model             | ROC-AUC | Precision | Recall | F1-Score |
|------------------|---------|-----------|--------|----------|
| LogisticRegression | ~0.93   | High      | Medium | Medium   |
| RandomForest      | ~0.97   | High      | High   | High     |
| KNN               | ~0.90   | Medium    | Low    | Low      |
| SVM               | ~0.94   | High      | Medium | Medium   |

> Random Forest performed best overall in terms of balanced precision and recall.

---
