"""
STEP 2 of the project: train 4 models, compare them, and save the best one.

Run:   python train.py
Uses:  data/Synthetic_Financial_datasets_log.csv  (real PaySim file)
       or data/sample_data.csv if the real file is missing.
Makes: models/fraud_model.pkl      (the BEST model, used by app.py)
       reports/model_comparison.csv/.png  (comparison of all 4 models)
"""

import os
import sys

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.datahandler import DataHandler

REAL_FILE = "data/Synthetic_Financial_datasets_log.csv"
SAMPLE_FILE = "data/sample_data.csv"
MODEL_PATH = "models/fraud_model.pkl"
COMPARISON_CSV = "reports/model_comparison.csv"
COMPARISON_PNG = "reports/model_comparison.png"

# The real file has 6.3 million rows. We use a random part of it so training is quick.
MAX_ROWS = 500_000

# The model only understands numbers. These are the columns we give it.
NUMBER_COLUMNS = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
                  "oldbalanceDest", "newbalanceDest"]
TYPE_VALUES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def build_features(df):
    """
    Turns raw rows into the numeric table (X) the model needs.
    app.py uses this SAME function, so training and prediction always match.
    """
    X = df[NUMBER_COLUMNS].copy()

    # 'type' is text, so we make one 0/1 column per type (one-hot encoding).
    for t in TYPE_VALUES:
        X["type_" + t] = (df["type"] == t).astype(int)

    # Extra clues: does the money add up? In fraud it often does NOT.
    X["errorOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    X["errorDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return X


def main():
    path = REAL_FILE if os.path.exists(REAL_FILE) else SAMPLE_FILE
    if not os.path.exists(path):
        sys.exit("No data found. Put the PaySim CSV in data/ or run make_sample_data.py")
    print("Using data file:", path)

    # 1) Load + clean, using the classes you already wrote
    handler = DataHandler(path)
    df = handler.clean_data(handler.load_data())
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)

    # 2) Split into inputs (X) and the answer (y)
    X = build_features(df)
    y = df["isFraud"]

    # 3) Hold back 20% of the data for a fair test.
    #    stratify=y keeps the same tiny fraud percentage in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Four different models to compare.
    #    Logistic Regression needs feature SCALING (StandardScaler puts all
    #    columns on a similar range). Tree-based models do not need it.
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42),
    }

    # 5) Train each one and test it on data it has never seen
    results = []
    fitted = {}
    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        print("Confusion matrix [[true non-fraud, false alarms], [missed fraud, caught fraud]]")
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred, target_names=["Non-Fraud", "Fraud"], zero_division=0))
        results.append({
            "Model": name,
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "PR-AUC": average_precision_score(y_test, proba),
        })
        fitted[name] = model

    # 6) Compare. Accuracy is useless for rare fraud, so we rank by the
    #    F1-score of the fraud class (balance of precision and recall);
    #    PR-AUC breaks ties.
    table = pd.DataFrame(results).set_index("Model").round(4)
    table = table.sort_values(["F1", "PR-AUC"], ascending=False)
    print("\n=== MODEL COMPARISON (fraud class) ===")
    print(table)
    best_name = table.index[0]
    print(f"\nBest model: {best_name}")

    os.makedirs("reports", exist_ok=True)
    table.to_csv(COMPARISON_CSV)
    table[["Precision", "Recall", "F1", "ROC-AUC"]].plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))
    plt.title("Model comparison (fraud class)")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=200)
    plt.close()

    # 7) Feature importance of the best model, if it has one
    best = fitted[best_name]
    if hasattr(best, "feature_importances_"):
        imp = pd.Series(best.feature_importances_, index=X.columns)
        print("\nTop 5 important features:")
        print(imp.sort_values(ascending=False).head(5).round(3))

    # 8) Save ONLY the best model for the web app
    os.makedirs("models", exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print("\nSaved best model to", MODEL_PATH)


if __name__ == "__main__":
    main()
