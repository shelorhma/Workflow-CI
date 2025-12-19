import os
import pandas as pd
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def main():
    # Load preprocessed dataset
    df = pd.read_csv("bank_marketing_preprocessed.csv")
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Set experiment (MLflow Project akan mengatur run)
    mlflow.set_experiment("CI Bank Marketing")

    # 🔑 Gunakan SATU konfigurasi parameter (CI harus stabil)
    params = {"C": 1.0}

    # Train model
    model = LogisticRegression(C=params["C"], max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Manual logging (WAJIB untuk skilled/advanced)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", params["C"])
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # =========================
    # Artifact 1: Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title("Confusion Matrix")

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(cm_path)

    # =========================
    # Artifact 2: Summary File
    # =========================
    summary_path = "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Model: Logistic Regression\n")
        f.write(f"C: {params['C']}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")

    mlflow.log_artifact(summary_path)

    # Cleanup local files
    os.remove(cm_path)
    os.remove(summary_path)


if __name__ == "__main__":
    main()