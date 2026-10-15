"""
Part 3: Train, select and evaluate the abnormal-heartbeat detectors.

Evaluation protocol (what goes in the paper):

1. MODEL SELECTION - patient-grouped 5-fold cross-validation inside DS1,
   with the one-standard-error rule: the simplest model whose mean CV F1 is
   within one standard error of the best is deployed.
   The out-of-fold predictions are also used to pick each model's decision
   threshold (the one that maximises F1). The test set (DS2) is never used
   to choose the model or the threshold.
2. INTER-PATIENT TEST - every model is refit on all of DS1 and tested once
   on DS2 (de Chazal et al., 2004). DS1 and DS2 share no patients, so this
   is directly comparable with published MIT-BIH inter-patient results.
3. INTRA-PATIENT (random beat split) - reported only for contrast, to show
   how much a naive random split overestimates performance.

Outputs in models/:
    cv_results_ds1.csv               per-fold CV metrics (DS1)
    cv_summary_ds1.csv               mean +- std CV metrics per model
    model_comparison_both_splits.csv intra-patient vs inter-patient (DS1->DS2)
    per_class_detection_ds2.csv      share of each AAMI class flagged abnormal on DS2
    ds2_predictions.csv              per-beat DS2 probabilities for every model
    best_model.pkl + model_meta.json the selected model, refit on DS1

Run this AFTER feature_extraction.py.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (AAMI_CLASSES, AAMI_MAP, BEST_MODEL_PATH, CV_FOLDS,
                    DECISION_THRESHOLD, DS1_RECORDS, DS2_RECORDS,
                    FEATURE_COLUMNS, FEATURES_CSV, GROUP_COLUMN, LABEL_COLUMN,
                    MODEL_DIR, MODEL_META_PATH, RANDOM_STATE, TEST_SIZE)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

os.makedirs(MODEL_DIR, exist_ok=True)
METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auroc"]

# Model complexity order used by the one-standard-error rule (simplest first)
COMPLEXITY_ORDER = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]


def select_model(cv_summary):
    """One-standard-error rule (Breiman et al., 1984; Hastie et al., 2009).

    Among models whose mean CV F1 is within one standard error of the best
    mean CV F1, choose the simplest. SE = std across folds / sqrt(n_folds).
    """
    best = cv_summary.loc[cv_summary["f1_mean"].idxmax()]
    se = best["f1_std"] / np.sqrt(CV_FOLDS)
    eligible = cv_summary[cv_summary["f1_mean"] >= best["f1_mean"] - se]["model"].tolist()
    chosen = min(eligible, key=COMPLEXITY_ORDER.index)
    print(f"\nOne-SE rule: best mean CV F1 = {best['f1_mean']:.3f} ({best['model']}), SE = {se:.3f}; "
          f"eligible (>= {best['f1_mean'] - se:.3f}): {eligible}")
    return chosen, float(se), eligible


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_features():
    """Feature table with an AAMI class column added (derived from the beat symbol)."""
    df = pd.read_csv(FEATURES_CSV, dtype={GROUP_COLUMN: str})
    df["aami_class"] = df["symbol"].map(AAMI_MAP).fillna("Q")
    return df


def load_dataset():
    df = load_features()
    return df[FEATURE_COLUMNS], df[LABEL_COLUMN], df[GROUP_COLUMN]


def inter_patient_split(X, y, groups):
    """DS1 (train) / DS2 (test) indices - the single source of truth for the split."""
    groups = pd.Series(np.asarray(groups)).astype(str)
    train_idx = np.where(groups.isin(DS1_RECORDS))[0]
    test_idx = np.where(groups.isin(DS2_RECORDS))[0]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models(y_train, feature_count=None):
    """Fresh, unfitted pipelines. Class imbalance is handled for every model."""
    neg, pos = np.bincount(np.asarray(y_train), minlength=2)
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           min_samples_leaf=5, n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            # Platt-scaled probabilities (SVC(probability=True) is deprecated in sklearn >= 1.9)
            ("clf", CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
                method="sigmoid", cv=3, ensemble=False)),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  eval_metric="logloss", scale_pos_weight=neg / max(pos, 1),
                                  random_state=RANDOM_STATE, n_jobs=-1)),
        ])
    else:
        print("[Note] xgboost not installed - skipping XGBoost.")
    return models


def best_f1_threshold(y_true, probs):
    """Threshold that maximises F1 on (out-of-fold) predictions."""
    grid = np.unique(np.quantile(probs, np.linspace(0.5, 0.995, 200)))
    scores = [f1_score(y_true, probs >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def compute_metrics(y_true, probs, threshold=DECISION_THRESHOLD):
    preds = (np.asarray(probs) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "specificity": recall_score(y_true, preds, pos_label=0, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auroc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan,
    }


def print_metrics(name, m, y_true=None, probs=None, threshold=DECISION_THRESHOLD):
    print(f"--- {name} ---  " + "  ".join(f"{k}={m[k]:.4f}" for k in METRICS))
    if y_true is not None:
        preds = (np.asarray(probs) >= threshold).astype(int)
        print("    confusion [[TN FP] [FN TP]] =", confusion_matrix(y_true, preds).tolist())


# ---------------------------------------------------------------------------
# 1. Patient-grouped cross-validation on DS1 (model selection)
# ---------------------------------------------------------------------------
def grouped_cv_on_ds1(df):
    ds1 = df[df[GROUP_COLUMN].isin(DS1_RECORDS)].reset_index(drop=True)
    X, y, g = ds1[FEATURE_COLUMNS], ds1[LABEL_COLUMN], ds1[GROUP_COLUMN]

    print(f"\n{'=' * 70}\n1) MODEL SELECTION: {CV_FOLDS}-fold patient-grouped CV on DS1 "
          f"({g.nunique()} patients, {len(ds1)} beats)\n{'=' * 70}")
    folds = list(GroupKFold(n_splits=CV_FOLDS).split(X, y, g))
    oof = {}
    for fold, (tr, va) in enumerate(folds, start=1):
        for name, model in build_models(y.iloc[tr]).items():
            t0 = time.time()
            model.fit(X.iloc[tr], y.iloc[tr])
            oof.setdefault(name, np.zeros(len(ds1)))[va] = model.predict_proba(X.iloc[va])[:, 1]
            print(f"fold {fold} | {name:20s} fitted ({time.time() - t0:.0f}s)", flush=True)

    # Decision threshold per model, chosen on DS1 out-of-fold predictions only
    thresholds = {name: best_f1_threshold(y.values, p) for name, p in oof.items()}
    print("\nDecision thresholds tuned on DS1 out-of-fold predictions:",
          {k: round(v, 3) for k, v in thresholds.items()})

    rows = []
    for fold, (_, va) in enumerate(folds, start=1):
        for name, p in oof.items():
            rows.append({"fold": fold, "model": name,
                         **compute_metrics(y.iloc[va], p[va], thresholds[name])})
    cv = pd.DataFrame(rows)
    cv.to_csv(os.path.join(MODEL_DIR, "cv_results_ds1.csv"), index=False)
    summary = cv.groupby("model")[METRICS].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["threshold"] = summary["model"].map(thresholds)
    summary = summary.sort_values("f1_mean", ascending=False)
    summary.to_csv(os.path.join(MODEL_DIR, "cv_summary_ds1.csv"), index=False)

    print("\nCV summary (mean +- std across folds):")
    for _, r in summary.iterrows():
        print(f"  {r['model']:20s} F1 {r['f1_mean']:.3f} +- {r['f1_std']:.3f} | "
              f"AUROC {r['auroc_mean']:.3f} +- {r['auroc_std']:.3f} | threshold {r['threshold']:.3f}")
    return summary, thresholds


# ---------------------------------------------------------------------------
# 2. DS1 -> DS2 inter-patient test   3. intra-patient random split
# ---------------------------------------------------------------------------
def evaluate_ds1_to_ds2(df, thresholds):
    train_idx, test_idx = inter_patient_split(None, None, df[GROUP_COLUMN])
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    print(f"\n{'=' * 70}\n2) INTER-PATIENT TEST: train DS1 ({tr[GROUP_COLUMN].nunique()} patients, "
          f"{len(tr)} beats) -> test DS2 ({te[GROUP_COLUMN].nunique()} patients, {len(te)} beats)\n{'=' * 70}")

    results, fitted, per_class_rows = [], {}, []
    preds_df = te[[GROUP_COLUMN, "sample_index", "symbol", "aami_class", LABEL_COLUMN]].copy()
    for name, model in build_models(tr[LABEL_COLUMN]).items():
        t0 = time.time()
        model.fit(tr[FEATURE_COLUMNS], tr[LABEL_COLUMN])
        probs = model.predict_proba(te[FEATURE_COLUMNS])[:, 1]
        thr = thresholds[name]
        m = compute_metrics(te[LABEL_COLUMN], probs, thr)
        print_metrics(f"{name} (thr={thr:.3f}, {time.time() - t0:.0f}s)", m, te[LABEL_COLUMN], probs, thr)
        results.append({"evaluation_type": "Inter-patient (DS1 -> DS2)", "model": name,
                        "threshold": thr, **m})
        fitted[name] = model
        preds_df[f"prob_{name}"] = probs

        flagged = probs >= thr
        for cls in AAMI_CLASSES:
            mask = (te["aami_class"] == cls).values
            if mask.sum():
                per_class_rows.append({
                    "model": name, "aami_class": cls, "n_beats": int(mask.sum()),
                    "flagged_abnormal": int(flagged[mask].sum()),
                    # N: share correctly left as normal; S/V/F: share correctly flagged
                    "rate": 1 - flagged[mask].mean() if cls == "N" else flagged[mask].mean(),
                    "meaning": "specificity" if cls == "N" else "sensitivity",
                })

    preds_df.to_csv(os.path.join(MODEL_DIR, "ds2_predictions.csv"), index=False)
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(os.path.join(MODEL_DIR, "per_class_detection_ds2.csv"), index=False)
    print("\nPer-class detection on DS2 (N = specificity, S/V/F = sensitivity):")
    print(per_class.pivot(index="model", columns="aami_class", values="rate")
          .reindex(columns=AAMI_CLASSES).round(3).to_string())
    return pd.DataFrame(results), fitted


def evaluate_intra_patient(df, thresholds):
    X, y = df[FEATURE_COLUMNS], df[LABEL_COLUMN]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"\n{'=' * 70}\n3) INTRA-PATIENT (random beat split, for contrast only)\n{'=' * 70}")
    results = []
    for name, model in build_models(y_tr).items():
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, probs, thresholds[name])
        print_metrics(name, m)
        results.append({"evaluation_type": "Intra-patient (random split)", "model": name,
                        "threshold": thresholds[name], **m})
    return pd.DataFrame(results)


def main():
    df = load_features()
    print(f"Loaded {len(df)} beats from {df[GROUP_COLUMN].nunique()} records; "
          f"features = {FEATURE_COLUMNS}")
    print("AAMI class counts:", df["aami_class"].value_counts().to_dict())

    cv_summary, thresholds = grouped_cv_on_ds1(df)
    best_name, se, eligible = select_model(cv_summary)
    print(f"Selected (simplest model within one SE of the best): {best_name}")

    inter_results, inter_models = evaluate_ds1_to_ds2(df, thresholds)
    intra_results = evaluate_intra_patient(df, thresholds)

    combined = pd.concat([intra_results, inter_results], ignore_index=True)
    combined = combined[["evaluation_type", "model", "threshold"] + METRICS]
    combined.to_csv(os.path.join(MODEL_DIR, "model_comparison_both_splits.csv"), index=False)
    print(f"\n{'=' * 70}\nFINAL COMPARISON\n{'=' * 70}")
    print(combined.round(4).to_string(index=False))

    # Save the CV-selected model (already refit on all of DS1)
    joblib.dump(inter_models[best_name], BEST_MODEL_PATH)
    best_test = inter_results.set_index("model").loc[best_name]
    best_cv = cv_summary.set_index("model").loc[best_name]
    meta = {
        "model_name": best_name,
        "selection": (f"one-standard-error rule on {CV_FOLDS}-fold patient-grouped CV F1 (DS1): "
                      f"simplest model within 1 SE ({se:.3f}) of the best; eligible = {eligible}"),
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": thresholds[best_name],
        "train_records": DS1_RECORDS,
        "test_records": DS2_RECORDS,
        "cv_metrics_ds1": {m: {"mean": float(best_cv[f"{m}_mean"]), "std": float(best_cv[f"{m}_std"])}
                           for m in METRICS},
        "test_metrics_ds2": {m: float(best_test[m]) for m in METRICS},
    }
    with open(MODEL_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved {best_name} -> models/best_model.pkl (+ models/model_meta.json)")
    print("Next: python paper_experiments.py   (ablation, AAMI multi-class, CIs, ROC/PR figures)")


if __name__ == "__main__":
    main()
