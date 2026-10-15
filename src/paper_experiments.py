"""
Paper experiments - everything the Results section needs, on the standard
DS1 -> DS2 inter-patient protocol.

Run AFTER train_models.py (it reads models/ds2_predictions.csv,
models/model_comparison_both_splits.csv and models/model_meta.json).

Outputs in models/:
    ds2_bootstrap_ci.csv        95% CIs on DS2 (patient-level bootstrap)
    aami_multiclass_ds2.csv     4-class N/S/V/F results: Se, +P, F1 per class
    ablation_ds2.csv            feature-set ablation (threshold-free AUROC / AUPRC)
    permutation_importance.csv  feature importance of the selected model on DS2
Figures in models/figures/:
    fig3_roc_pr_ds2.png, fig4_per_class_detection.png, fig5_ablation.png,
    fig6_aami_confusion.png, fig7_feature_importance.png
"""

import json
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from config import (AAMI_CLASSES, BEST_MODEL_PATH, DS1_RECORDS, DS2_RECORDS,
                    FEATURE_COLUMNS, FIG_DIR, GROUP_COLUMN, LABEL_COLUMN,
                    MODEL_DIR, MODEL_META_PATH, RANDOM_STATE)
from train_models import load_features

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({
    "font.size": 10, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})

# Fixed categorical order: a model always keeps the same colour in every figure
MODEL_ORDER = ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
MODEL_COLORS = {"Logistic Regression": "#2a78d6", "Random Forest": "#eb6834",
                "SVM": "#1baf7a", "XGBoost": "#eda100"}
N_BOOT = 1000

FEATURE_SETS = {
    "RR only": ["rr_prev", "rr_local_avg", "rr_ratio"],
    "Morphology only": ["amplitude", "local_mean", "local_std"],
    "Baseline (rr_prev + morphology)": ["rr_prev", "amplitude", "local_mean", "local_std"],
    "Full (proposed)": FEATURE_COLUMNS,
}


def split(df):
    tr = df[df[GROUP_COLUMN].isin(DS1_RECORDS)]
    te = df[df[GROUP_COLUMN].isin(DS2_RECORDS)]
    return tr, te


# ---------------------------------------------------------------------------
# 1. Patient-level bootstrap confidence intervals on DS2
# ---------------------------------------------------------------------------
def bootstrap_ci(preds, thresholds):
    print(f"\n[1] Patient-level bootstrap 95% CIs on DS2 ({N_BOOT} resamples)")
    rng = np.random.default_rng(RANDOM_STATE)
    records = preds[GROUP_COLUMN].unique()
    by_rec = {r: preds.index[preds[GROUP_COLUMN] == r].values for r in records}
    samples = [np.concatenate([by_rec[r] for r in rng.choice(records, len(records))])
               for _ in range(N_BOOT)]
    y_all = preds[LABEL_COLUMN].values

    rows = []
    for name in [m for m in MODEL_ORDER if f"prob_{m}" in preds]:
        p_all = preds[f"prob_{name}"].values
        thr = thresholds[name]
        stats = {k: [] for k in ["f1", "recall", "precision", "auroc"]}
        for idx in samples:
            y, p = y_all[idx], p_all[idx]
            if y.min() == y.max():
                continue
            yhat = p >= thr
            stats["f1"].append(f1_score(y, yhat, zero_division=0))
            stats["recall"].append(recall_score(y, yhat, zero_division=0))
            stats["precision"].append(precision_score(y, yhat, zero_division=0))
            stats["auroc"].append(roc_auc_score(y, p))
        yhat = p_all >= thr
        point = {"f1": f1_score(y_all, yhat), "recall": recall_score(y_all, yhat),
                 "precision": precision_score(y_all, yhat), "auroc": roc_auc_score(y_all, p_all)}
        row = {"model": name}
        for k, v in stats.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            row.update({k: point[k], f"{k}_ci_low": lo, f"{k}_ci_high": hi})
        rows.append(row)
        print(f"  {name:20s} F1 {row['f1']:.3f} [{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}]  "
              f"AUROC {row['auroc']:.3f} [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(MODEL_DIR, "ds2_bootstrap_ci.csv"), index=False)
    return out


# ---------------------------------------------------------------------------
# 2. AAMI 4-class (N, S, V, F) classification, DS1 -> DS2
# ---------------------------------------------------------------------------
def aami_multiclass(df, show_model=None):
    print("\n[2] AAMI 4-class (N/S/V/F) classification, DS1 -> DS2")
    df = df[df["aami_class"].isin(AAMI_CLASSES)]
    tr, te = split(df)
    classes = AAMI_CLASSES
    y_tr = tr["aami_class"].map(classes.index).values
    y_te = te["aami_class"].map(classes.index).values

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                         ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        "Random Forest": Pipeline([("clf", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE))]),
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE))]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline([("clf", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, objective="multi:softprob",
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1))])

    rows, confusions = [], {}
    for name, model in models.items():
        t0 = time.time()
        fit_kw = {}
        if name == "XGBoost":
            fit_kw["clf__sample_weight"] = compute_sample_weight("balanced", y_tr)
        model.fit(tr[FEATURE_COLUMNS], y_tr, **fit_kw)
        pred = model.predict(te[FEATURE_COLUMNS])
        cm = confusion_matrix(y_te, pred, labels=range(len(classes)))
        confusions[name] = cm
        row = {"model": name, "accuracy": float(np.trace(cm) / cm.sum()),
               "macro_f1": f1_score(y_te, pred, average="macro", labels=range(len(classes)), zero_division=0)}
        for i, c in enumerate(classes):
            tp = cm[i, i]
            se = tp / cm[i].sum() if cm[i].sum() else np.nan
            ppv = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
            row[f"{c}_Se"], row[f"{c}_PPV"] = se, ppv
            row[f"{c}_F1"] = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
        rows.append(row)
        print(f"  {name:20s} acc {row['accuracy']:.3f} macroF1 {row['macro_f1']:.3f} | "
              + " | ".join(f"{c}: Se {row[f'{c}_Se']:.2f} +P {row[f'{c}_PPV']:.2f}" for c in classes)
              + f"  ({time.time() - t0:.0f}s)")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(MODEL_DIR, "aami_multiclass_ds2.csv"), index=False)

    # Confusion matrix of the deployed model (falls back to the best macro-F1 model)
    best = show_model if show_model in confusions else out.loc[out["macro_f1"].idxmax(), "model"]
    cm = confusions[best]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#1a1a1a")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"AAMI 4-class confusion on DS2\n{best} (row-normalised)", fontsize=10)
    plt.savefig(os.path.join(FIG_DIR, "fig6_aami_confusion.png"))
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 3. Feature ablation (threshold-free, so no tuning is needed)
# ---------------------------------------------------------------------------
def ablation(df):
    print("\n[3] Feature-set ablation, DS1 -> DS2 (AUROC / AUPRC)")
    tr, te = split(df)
    y_tr, y_te = tr[LABEL_COLUMN].values, te[LABEL_COLUMN].values
    neg, pos = np.bincount(y_tr)

    def make(name):
        if name == "Logistic Regression":
            return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=2000))])
        if name == "Random Forest":
            return RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=5,
                                          n_jobs=-1, random_state=RANDOM_STATE)
        if name == "SVM":
            return Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", class_weight="balanced",
                                                                random_state=RANDOM_STATE))])
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss",
                             scale_pos_weight=neg / pos, random_state=RANDOM_STATE, n_jobs=-1)

    rows = []
    for set_name, cols in FEATURE_SETS.items():
        for name in [m for m in MODEL_ORDER if HAS_XGBOOST or m != "XGBoost"]:
            model = make(name).fit(tr[cols], y_tr)
            # SVM ranks by its decision function (rank-based metrics need no calibration)
            score = (model.decision_function(te[cols]) if name == "SVM"
                     else model.predict_proba(te[cols])[:, 1])
            rows.append({"feature_set": set_name, "n_features": len(cols), "model": name,
                         "auroc": roc_auc_score(y_te, score),
                         "auprc": average_precision_score(y_te, score)})
            print(f"  {set_name:32s} {name:20s} AUROC {rows[-1]['auroc']:.3f}  AUPRC {rows[-1]['auprc']:.3f}",
                  flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(MODEL_DIR, "ablation_ds2.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    sets = list(FEATURE_SETS)
    models = [m for m in MODEL_ORDER if m in set(out["model"])]
    width = 0.8 / len(models)
    x = np.arange(len(sets))
    for k, m in enumerate(models):
        vals = out[out["model"] == m].set_index("feature_set").loc[sets, "auprc"]
        ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width * 0.92,
               color=MODEL_COLORS[m], label=m)
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in sets], fontsize=8)
    ax.set_ylabel("AUPRC on DS2")
    ax.set_ylim(0, 1)
    ax.set_title("Feature ablation (DS1 -> DS2)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper left", frameon=False)
    plt.savefig(os.path.join(FIG_DIR, "fig5_ablation.png"))
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 4. Permutation importance of the selected model
# ---------------------------------------------------------------------------
def feature_importance(df, meta):
    print("\n[4] Permutation importance (selected model, 10k sampled DS2 beats, AUROC drop)")
    _, te = split(df)
    model = joblib.load(BEST_MODEL_PATH)
    res = permutation_importance(model, te[FEATURE_COLUMNS], te[LABEL_COLUMN], scoring="roc_auc",
                                 n_repeats=5, max_samples=10000,
                                 random_state=RANDOM_STATE, n_jobs=1)
    out = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance_mean": res.importances_mean,
                        "importance_std": res.importances_std}).sort_values("importance_mean")
    out.to_csv(os.path.join(MODEL_DIR, "permutation_importance.csv"), index=False)
    print(out.sort_values("importance_mean", ascending=False).round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(out["feature"], out["importance_mean"], xerr=out["importance_std"],
            color=MODEL_COLORS.get(meta["model_name"], "#2a78d6"), height=0.6,
            error_kw={"elinewidth": 1, "ecolor": "#555555"})
    ax.set_xlabel("Drop in AUROC when feature is shuffled")
    ax.set_title(f"Feature importance - {meta['model_name']} (DS2)", fontsize=10)
    plt.savefig(os.path.join(FIG_DIR, "fig7_feature_importance.png"))
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 5. ROC / PR curves and per-class detection figure
# ---------------------------------------------------------------------------
def curves_and_per_class(preds):
    print("\n[5] ROC / PR curves and per-class detection figures")
    y = preds[LABEL_COLUMN].values
    models = [m for m in MODEL_ORDER if f"prob_{m}" in preds]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
    for m in models:
        p = preds[f"prob_{m}"].values
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        a1.plot(fpr, tpr, color=MODEL_COLORS[m], lw=1.6, label=f"{m} ({roc_auc_score(y, p):.3f})")
        a2.plot(rec, prec, color=MODEL_COLORS[m], lw=1.6, label=f"{m} ({average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
    a2.axhline(y.mean(), ls="--", lw=0.8, color="#999999")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (AUROC)")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (AUPRC)", ylim=(0, 1.02))
    a1.legend(fontsize=7, loc="lower right", frameon=False)
    a2.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    for a in (a1, a2):
        a.title.set_fontsize(10)
    fig.suptitle("Inter-patient test set (DS2)", fontsize=10, y=1.02)
    plt.savefig(os.path.join(FIG_DIR, "fig3_roc_pr_ds2.png"))
    plt.close()

    pc = pd.read_csv(os.path.join(MODEL_DIR, "per_class_detection_ds2.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    width = 0.8 / len(models)
    x = np.arange(len(AAMI_CLASSES))
    for k, m in enumerate(models):
        vals = pc[pc["model"] == m].set_index("aami_class").reindex(AAMI_CLASSES)["rate"]
        ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width * 0.92, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(x, ["N\n(specificity)", "S\n(sensitivity)", "V\n(sensitivity)", "F\n(sensitivity)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate on DS2")
    ax.set_title("Binary detector performance by AAMI beat class (DS2)", fontsize=10)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    plt.savefig(os.path.join(FIG_DIR, "fig4_per_class_detection.png"))
    plt.close()


def main():
    df = load_features()
    with open(MODEL_META_PATH) as f:
        meta = json.load(f)
    preds = pd.read_csv(os.path.join(MODEL_DIR, "ds2_predictions.csv"), dtype={GROUP_COLUMN: str})
    comp = pd.read_csv(os.path.join(MODEL_DIR, "model_comparison_both_splits.csv"))
    thresholds = (comp[comp["evaluation_type"].str.startswith("Inter")]
                  .set_index("model")["threshold"].to_dict())

    bootstrap_ci(preds, thresholds)
    curves_and_per_class(preds)
    feature_importance(df, meta)
    aami_multiclass(df, show_model=meta["model_name"])
    ablation(df)
    print(f"\nAll tables saved in {MODEL_DIR}, figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
