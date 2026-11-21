# Machine Learning Assignment — Classification Model Comparison _ Nagaraju Bandaru

## Problem Statement
The objective of this project is to implement, evaluate, and compare five supervised machine learning classification models—Logistic Regression, Decision Tree, K-Nearest Neighbors (kNN), Gaussian Naive Bayes, and Random Forest—on a clinical diagnostic dataset to predict whether a patient has thyroid disease (`sick`) or is healthy (`negative`).

## Dataset Description
* **Dataset Source:** Thyroid Disease (`sick`) Dataset — UCI Machine Learning Repository (fetched via OpenML `version=1`).
* **Selection Rationale:** Clinical laboratory records naturally combine continuous hormone measurements with categorical medical history flags, contain real missing entries requiring imputation (`1,991` missing cells), and exhibit strong class imbalance—making them ideal for evaluating how different classifier families handle skewed distributions and preprocessing pipelines.
* **Instance & Feature Count:** `3,772` raw instances (`3,711` unique instances retained after removing `61` exact duplicates) and `28` predictive features (after dropping `TBG`, which is 100% empty).
* **Target Variable:** `target` (Binary: `0` = `negative` [healthy], `1` = `sick` [thyroid condition]).
* **Feature Types:**
  * **Numerical (6):** `age`, `TSH`, `T3`, `TT4`, `T4U`, `FTI`.
  * **Categorical (22):** `sex`, `on_thyroxine`, `query_on_thyroxine`, `on_antithyroid_medication`, `sick`, `pregnant`, `thyroid_surgery`, `I131_treatment`, `query_hypothyroid`, `query_hyperthyroid`, `lithium`, `goitre`, `tumor`, `hypopituitary`, `psych`, `TSH_measured`, `T3_measured`, `TT4_measured`, `T4U_measured`, `FTI_measured`, `TBG_measured`, and `referral_source`.
* **Class Distribution & Imbalance Examination:** After deduplication, the dataset is imbalanced with **93.78%** (`3,480` instances) belonging to class `0` (`negative`) and **6.22%** (`231` instances) belonging to class `1` (`sick`). Because a naive majority-class classifier achieves ~93.8% accuracy, **AUC, F1 Score, and Matthews Correlation Coefficient (MCC)** are evaluated alongside Accuracy to measure true predictive skill.

## GitHub Repository Link
* **Repository URL:** https://github.com/2026ct05017/AMLCCZG565-ML-Classification.git

## Data Preprocessing & Execution Instructions
* **Preprocessing Pipeline:** `61` duplicate rows are dropped, missing numerical features are imputed with the training median and standardized via `StandardScaler`, and missing categorical features are imputed with the most frequent value and encoded via `OneHotEncoder(handle_unknown="ignore")`. All transformers are wrapped in a `scikit-learn` `Pipeline` and fitted strictly on the 80% stratified training set (`random_state=5017`) to prevent data leakage.
* **How to Run:**
  ```bash
  pip install -r requirements.txt
  python train_models.py



Models Used & Evaluation Comparison TableML Model NameAccuracyAUCPrecisionRecall
F1MCCLogistic Regression0.96230.94020.76470.56520.65000.6386
Decision Tree0.98650.95220.87500.91300.89360.8867
kNN0.97170.92290.93100.58700.72000.7268
Naive Bayes0.37690.64870.08710.95650.15970.1524
Random Forest (Ensemble)0.98250.99660.88370.82610.85390.8452



Evaluation Metrics VisualizationModel Performance Observations

Logistic RegressionEstablishes a solid linear baseline with high Accuracy (0.9623) and strong ROC-AUC (0.9402), but its linear decision boundary struggles to capture non-linear hormone threshold interactions, resulting in moderate minority-class Recall (0.5652).

Decision Tree Excels at capturing sharp, non-linear clinical threshold rules (such as combined T3/TT4/TSH cutoffs), achieving the highest Accuracy (0.9865), Recall (0.9130), F1 (0.8936), and MCC (0.8867) on this test split.

kNN Benefits from StandardScaler normalization, achieving the highest Precision (0.9310) and strong Accuracy (0.9717), though one-hot encoding 22 categorical features increases dimensionality and limits minority-class Recall (0.5870).

Naive BayesAssumes conditional independence across features—an assumption violated by correlated thyroid panel tests (TT4, T3, FTI) and indicator flags (T3_measured). While it captures nearly all positive cases (Recall 0.9565), excessive false positives degrade Precision (0.0871), Accuracy (0.3769), and MCC (0.1524).

Random Forest (Ensemble)Achieves the highest ROC-AUC (0.9966) and second-best Accuracy (0.9825), F1 (0.8539), and MCC (0.8452) by aggregating 150 decorrelated trees, delivering the strongest probability ranking and lowest variance across unseen data.

Overall Winner for your dataset?
Decision Tree (highest F1 0.8936 and MCC 0.8867) and Random Forest (Ensemble) (highest ROC-AUC 0.9966 and lowest variance) are the top performers, with Decision Tree leading in point-classification metrics on this test split and Random Forest offering superior probabilistic discrimination.
  
