"""
End-to-end evaluation: QRS detector + classifier, with no reference beat positions.

1. The streaming QRS detector (qrs_detector.py) finds the beats in every record.
2. Detected beats are matched to the reference annotations (150 ms tolerance,
   AAMI EC57) to obtain detection Se / +P and labels for the matched beats.
3. The six causal features are computed from the DETECTED beat positions.
4. Two classifiers are evaluated on DS2:
     A. the deployed model (models/best_model.pkl, trained on annotated DS1 beats)
     B. logistic regression retrained on detected DS1 beats, threshold tuned by
        5-fold patient-grouped CV on DS1 (same protocol as train_models.py)
5. End-to-end metrics count every reference beat: a beat the detector misses is
   a miss, and a false detection that is flagged abnormal is a false alarm.

Run AFTER train_models.py. Outputs:
    data/ecg_features_detected.csv
    models/qrs_detection_results.csv      per-record detection results
    models/end_to_end_results.csv         classification + end-to-end metrics

Model A is what the API and dashboard use; model B is reported for reference only.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import (AAMI_MAP, BEST_MODEL_PATH, CV_FOLDS, DS1_RECORDS, DS2_RECORDS,
                    EXCLUDE_PACED_RECORDS, FEATURE_COLUMNS, MODEL_DIR, MODEL_META_PATH,
                    PACED_RECORDS, PROJECT_DIR)
from feature_extraction import features_from_beats, load_record
from load_data import available_records
from qrs_detector import detect_qrs, match_beats
from train_models import best_f1_threshold, build_models, compute_metrics

DETECTED_CSV = os.path.join(PROJECT_DIR, "data", "ecg_features_detected.csv")
TOLERANCE_S = 0.150


def detect_all(records):
    det_rows, feat_dfs = [], []
    for rec in records:
        signal, fs, ref, ref_sym = load_record(rec)
        det = detect_qrs(signal, fs)
        pairs = match_beats(det, ref, int(round(TOLERANCE_S * fs)))
        sym = [None] * len(det)
        matched_ref = np.zeros(len(ref), dtype=bool)
        for d, r in pairs:
            sym[d] = ref_sym[r]
            matched_ref[r] = True
        missed = pd.Series([AAMI_MAP.get(s, "Q") for s, m in zip(ref_sym, matched_ref) if not m])
        tp, fn, fp = len(pairs), len(ref) - len(pairs), len(det) - len(pairs)
        det_rows.append({"record": rec, "ref_beats": len(ref), "detected": len(det),
                         "TP": tp, "FN": fn, "FP": fp,
                         "Se": tp / len(ref), "PPV": tp / max(len(det), 1),
                         **{f"missed_{c}": int((missed == c).sum()) for c in ["N", "S", "V", "F", "Q"]}})
        df = features_from_beats(signal, fs, det, sym, rec)
        df["matched"] = df["symbol"].notna()
        feat_dfs.append(df)
        print(f"{rec}: Se={tp / len(ref):.4f} +P={tp / max(len(det), 1):.4f} (FN={fn}, FP={fp})", flush=True)
    return pd.DataFrame(det_rows), pd.concat(feat_dfs, ignore_index=True)


def tuned_lr_on_ds1(train):
    """LR + threshold chosen by patient-grouped CV on DS1 (same protocol as train_models.py)."""
    X, y, g = train[FEATURE_COLUMNS], train["label"].astype(int), train["record"]
    oof = np.zeros(len(train))
    for tr, va in GroupKFold(n_splits=CV_FOLDS).split(X, y, g):
        m = build_models(y.iloc[tr])["Logistic Regression"].fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    thr = best_f1_threshold(y.values, oof)
    model = build_models(y)["Logistic Regression"].fit(X, y)
    return model, thr


def evaluate(name, model, thr, test, n_ref_abnormal, missed_abnormal):
    matched = test[test["matched"]]
    probs = model.predict_proba(matched[FEATURE_COLUMNS])[:, 1]
    m = compute_metrics(matched["label"].astype(int), probs, thr)
    flagged = probs >= thr
    tp = int((flagged & (matched["label"] == 1)).sum())
    fp_matched = int((flagged & (matched["label"] == 0)).sum())
    false_det = test[~test["matched"]]
    fd_flagged = int((model.predict_proba(false_det[FEATURE_COLUMNS])[:, 1] >= thr).sum()) if len(false_det) else 0
    e2e_se = tp / n_ref_abnormal
    e2e_ppv = tp / max(tp + fp_matched + fd_flagged, 1)
    row = {"model": name, "threshold": thr,
           **{f"matched_{k}": v for k, v in m.items()},
           "false_detections": len(false_det), "false_detections_flagged": fd_flagged,
           "abnormal_beats_missed_by_detector": int(missed_abnormal),
           "e2e_sensitivity": e2e_se, "e2e_precision": e2e_ppv,
           "e2e_f1": 2 * e2e_se * e2e_ppv / max(e2e_se + e2e_ppv, 1e-12)}
    for c in ["N", "S", "V", "F"]:
        mask = matched["symbol"].map(AAMI_MAP).fillna("Q") == c
        if mask.any():
            rate = flagged[mask.values].mean()
            row[f"class_{c}"] = 1 - rate if c == "N" else rate
    print(f"\n[{name}] threshold {thr:.3f}")
    print("  on matched beats: " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    print(f"  false detections: {len(false_det)}, flagged abnormal: {fd_flagged}")
    print(f"  end-to-end: Se={e2e_se:.4f}  precision={e2e_ppv:.4f}  F1={row['e2e_f1']:.4f}")
    print("  per class (N=Sp, S/V/F=Se): " + ", ".join(f"{c}={row.get('class_' + c, float('nan')):.3f}"
                                                     for c in ['N', 'S', 'V', 'F']))
    return row


def main():
    records = available_records()
    if EXCLUDE_PACED_RECORDS:
        records = [r for r in records if r not in PACED_RECORDS]
    det_df, feats = detect_all(records)
    det_df.to_csv(os.path.join(MODEL_DIR, "qrs_detection_results.csv"), index=False)
    feats.to_csv(DETECTED_CSV, index=False)

    tot = det_df[["ref_beats", "detected", "TP", "FN", "FP"]].sum()
    print(f"\nQRS detection, {len(det_df)} records: Se={tot.TP / tot.ref_beats:.4f} "
          f"+P={tot.TP / tot.detected:.4f} (TP={tot.TP}, FN={tot.FN}, FP={tot.FP})")
    for split, recs in [("DS1", DS1_RECORDS), ("DS2", DS2_RECORDS)]:
        t = det_df[det_df.record.isin(recs)][["ref_beats", "detected", "TP"]].sum()
        print(f"  {split}: Se={t.TP / t.ref_beats:.4f} +P={t.TP / t.detected:.4f}")

    feats["record"] = feats["record"].astype(str)
    train = feats[feats.record.isin(DS1_RECORDS) & feats.matched]
    test = feats[feats.record.isin(DS2_RECORDS)]

    ds2_det = det_df[det_df.record.isin(DS2_RECORDS)]
    n_ref_abn = int(sum(ds2_det[f"missed_{c}"].sum() for c in "SVFQ"))       # missed abnormal
    # total abnormal reference beats in DS2 (within the feature edges this is approximate by < 0.1%)
    ref_abn_total = int(n_ref_abn + (test.matched & (test.label == 1)).sum())

    with open(MODEL_META_PATH) as f:
        meta = json.load(f)
    deployed = joblib.load(BEST_MODEL_PATH)
    rows = [evaluate(f"A: deployed {meta['model_name']} (trained on annotated beats)",
                     deployed, meta["decision_threshold"], test, ref_abn_total, n_ref_abn)]

    model_b, thr_b = tuned_lr_on_ds1(train)
    rows.append(evaluate("B: Logistic Regression retrained on detected beats",
                         model_b, thr_b, test, ref_abn_total, n_ref_abn))
    pd.DataFrame(rows).to_csv(os.path.join(MODEL_DIR, "end_to_end_results.csv"), index=False)
    print("\nSaved models/end_to_end_results.csv and models/qrs_detection_results.csv")


if __name__ == "__main__":
    main()
