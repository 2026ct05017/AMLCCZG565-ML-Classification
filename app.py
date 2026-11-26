import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Thyroid Disease Classification", layout="wide")
st.title("Thyroid Disease Classification — ML Model Comparison")
st.write("Course: AMLCCZG565 | Student ID: 2026CT05017")

# Display evaluation metrics table
metrics_data = [
    {"Model": "Logistic Regression", "Accuracy": 0.9623, "AUC": 0.9402, "Precision": 0.7647, "Recall": 0.5652, "F1": 0.6500, "MCC": 0.6386},
    {"Model": "Decision Tree", "Accuracy": 0.9865, "AUC": 0.9522, "Precision": 0.8750, "Recall": 0.9130, "F1": 0.8936, "MCC": 0.8867},
    {"Model": "kNN", "Accuracy": 0.9717, "AUC": 0.9229, "Precision": 0.9310, "Recall": 0.5870, "F1": 0.7200, "MCC": 0.7268},
    {"Model": "Naive Bayes", "Accuracy": 0.3769, "AUC": 0.6487, "Precision": 0.0871, "Recall": 0.9565, "F1": 0.1597, "MCC": 0.1524},
    {"Model": "Random Forest (Ensemble)", "Accuracy": 0.9825, "AUC": 0.9966, "Precision": 0.8837, "Recall": 0.8261, "F1": 0.8539, "MCC": 0.8452},
]
st.subheader("Model Evaluation Comparison Table")
st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

# Interactive test-set prediction using saved models
st.subheader("Run Predictions on Holdout Test Data")
model_choice = st.selectbox(
    "Select Trained Model",
    ["decision_tree.pkl", "random_forest_ensemble.pkl", "knn.pkl", "logistic_regression.pkl", "naive_bayes.pkl"]
)

if st.button("Evaluate Selected Model on test_data.csv"):
    model = joblib.load(f"model/{model_choice}")
    test_df = pd.read_csv("test_data.csv")
    X_test = test_df.drop(columns=["target"])
    preds = model.predict(X_test)
    preview = test_df[["age", "TSH", "T3", "TT4", "target"]].copy()
    preview["Predicted"] = preds
    st.write("Sample Predictions (0 = Negative, 1 = Sick):")
    st.dataframe(preview.head(20), use_container_width=True)
