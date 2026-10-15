"""
Generates publication-quality figures for the paper:
1. Bar chart comparing F1-scores: intra-patient vs inter-patient
2. Confusion matrix for the CV-selected model on the DS2 test patients
   recorded in models/model_meta.json

Run this AFTER train_models.py.
Output: PNG files in models/figures/
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from config import (BEST_MODEL_PATH, FEATURES_CSV, FIG_DIR, GROUP_COLUMN,
                    LABEL_COLUMN, MODEL_DIR, MODEL_META_PATH)

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 300, "savefig.bbox": "tight"})


def plot_f1_comparison():
    """Figure 1: F1-scores, intra-patient vs inter-patient."""
    df = pd.read_csv(os.path.join(MODEL_DIR, "model_comparison_both_splits.csv"))

    models = df["model"].unique()
    intra_f1 = df[df["evaluation_type"].str.startswith("Intra")].set_index("model")["f1"]
    inter_f1 = df[df["evaluation_type"].str.startswith("Inter")].set_index("model")["f1"]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width / 2, intra_f1.loc[models], width, label="Intra-patient", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, inter_f1.loc[models], width, label="Inter-patient", color="#DD8452")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score Comparison: Intra-patient vs Inter-patient Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.bar_label(bars1, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.2f", padding=3, fontsize=8)

    path = os.path.join(FIG_DIR, "fig1_f1_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved Figure 1 to {path}")


def plot_confusion_matrix_for_best_model():
    """Figure 2: confusion matrix of the saved best model on the held-out patients."""
    model = joblib.load(BEST_MODEL_PATH)
    with open(MODEL_META_PATH) as f:
        meta = json.load(f)

    df = pd.read_csv(FEATURES_CSV, dtype={GROUP_COLUMN: str})
    test_df = df[df[GROUP_COLUMN].isin(meta["test_records"])]
    X_test = test_df[meta["feature_columns"]]
    y_test = test_df[LABEL_COLUMN]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= meta.get("decision_threshold", 0.5)).astype(int)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Abnormal"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {meta['model_name']}\n(Inter-patient test set DS2, threshold {meta.get('decision_threshold', 0.5):.2f})")

    path = os.path.join(FIG_DIR, "fig2_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved Figure 2 to {path}")


if __name__ == "__main__":
    plot_f1_comparison()
    plot_confusion_matrix_for_best_model()
