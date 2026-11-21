import os
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

# 1. Configuration (Unique random seed)
SEED = 5017
os.makedirs("model", exist_ok=True)

# 2. Load & Inspect Dataset (UCI Thyroid 'sick')
print("Fetching UCI Thyroid Disease (sick) dataset...")
X_raw, y_raw = fetch_openml(name="sick", version=1, as_frame=True, return_X_y=True)

df = X_raw.copy()
df["target"] = (y_raw == "sick").astype(int)

print(f"Raw Dataset Shape: {df.shape}")
print(f"Duplicate Rows Detected: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)

# Drop columns >95% missing (e.g., TBG is 100% NaN)
missing_fraction = df.isnull().mean()
drop_empty = missing_fraction[missing_fraction > 0.95].index.tolist()
df = df.drop(columns=drop_empty)

print(f"Cleaned Shape: {df.shape} -> {df.shape[1] - 1} Features, {df.shape[0]} Instances")
print(f"Missing Values Remaining (to be imputed): {df.isnull().sum().sum()}")
print("\nTarget Class Distribution (0 = Negative, 1 = Sick):")
print(df["target"].value_counts())

# 3. Stratified 80:20 Train-Test Split
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Save 20% test split for GitHub repository
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("\n[OK] Exported test_data.csv")

# 4. Preprocessing Pipeline (Fitted ONLY on Train)
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 5. Train & Evaluate All 5 Required Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=150, random_state=SEED),
}

metrics_records = []

for name, estimator in models.items():
    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    clf_pipeline.fit(X_train, y_train)

    filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(clf_pipeline, os.path.join("model", filename))

    y_pred = clf_pipeline.predict(X_test)
    y_prob = clf_pipeline.predict_proba(X_test)[:, 1]

    metrics_records.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    })

results_df = pd.DataFrame(metrics_records)

print("\n================ MODEL COMPARISON TABLE ================")
header = "| " + " | ".join(results_df.columns) + " |"
separator = "| " + " | ".join(["---"] * len(results_df.columns)) + " |"
rows = [
    "| " + " | ".join(str(val) for val in row) + " |"
    for row in results_df.itertuples(index=False)
]
print("\n".join([header, separator] + rows))
print("========================================================\n")

# 6. Generate Metrics Comparison Graph
metric_names = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
ax = results_df.set_index("ML Model Name")[metric_names].plot(
    kind="bar", figsize=(12, 6), width=0.82, colormap="viridis", edgecolor="black"
)
plt.title("ML Classification Models — Performance Comparison", fontsize=14, fontweight="bold")
plt.ylabel("Score", fontsize=11)
plt.xlabel("Classification Model", fontsize=11)
plt.ylim(0, 1.15)
plt.xticks(rotation=15, ha="right")
plt.legend(loc="upper right", ncol=3, frameon=True)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=7.5, padding=2, rotation=90)

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=300)
print("[OK] Saved metrics_comparison.png")

shutil.copy("train_models.py", os.path.join("model", "train_models.py"))
print("[OK] Copied train_models.py into model/ folder")