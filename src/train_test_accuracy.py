"""
Train vs test accuracy for every model (shows over-fitting / generalisation gap).

For each model, using the thresholds tuned in train_models.py:
  * Inter-patient: fit on DS1, score on DS1 (train) and DS2 (test)
  * Intra-patient: fit on the random 80% split, score on that 80% (train) and the 20% (test)

Run AFTER train_models.py. Output: models/train_test_accuracy.csv
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DS1_RECORDS, DS2_RECORDS, FEATURE_COLUMNS, GROUP_COLUMN,
                    LABEL_COLUMN, MODEL_DIR, RANDOM_STATE, TEST_SIZE)
from train_models import build_models, compute_metrics, load_features


def score(model, X, y, thr):
    m = compute_metrics(y, model.predict_proba(X)[:, 1], thr)
    return m["accuracy"], m["f1"]


def main():
    df = load_features()
    comp = pd.read_csv(os.path.join(MODEL_DIR, "model_comparison_both_splits.csv"))
    thresholds = comp[comp["evaluation_type"].str.startswith("Inter")].set_index("model")["threshold"].to_dict()

    splits = {}
    tr = df[df[GROUP_COLUMN].isin(DS1_RECORDS)]
    te = df[df[GROUP_COLUMN].isin(DS2_RECORDS)]
    splits["Inter-patient (DS1 -> DS2)"] = (tr[FEATURE_COLUMNS], te[FEATURE_COLUMNS], tr[LABEL_COLUMN], te[LABEL_COLUMN])
    splits["Intra-patient (random split)"] = tuple(train_test_split(
        df[FEATURE_COLUMNS], df[LABEL_COLUMN], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN]))
    # train_test_split returns X_tr, X_te, y_tr, y_te - same order as above

    rows = []
    for split_name, (X_tr, X_te, y_tr, y_te) in splits.items():
        for name, model in build_models(y_tr).items():
            model.fit(X_tr, y_tr)
            thr = thresholds[name]
            acc_tr, f1_tr = score(model, X_tr, y_tr, thr)
            acc_te, f1_te = score(model, X_te, y_te, thr)
            rows.append({"evaluation_type": split_name, "model": name, "threshold": thr,
                         "train_accuracy": acc_tr, "test_accuracy": acc_te,
                         "accuracy_gap": acc_tr - acc_te,
                         "train_f1": f1_tr, "test_f1": f1_te, "f1_gap": f1_tr - f1_te})
            print(f"{split_name:30s} {name:20s} accuracy train {acc_tr:.4f} test {acc_te:.4f} | "
                  f"F1 train {f1_tr:.4f} test {f1_te:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(MODEL_DIR, "train_test_accuracy.csv"), index=False)
    base = 1 - te[LABEL_COLUMN].mean()
    print(f"\n'Always Normal' baseline accuracy on DS2: {base:.4f}")
    print("Saved models/train_test_accuracy.csv")


if __name__ == "__main__":
    main()
