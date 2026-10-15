"""
Compares training data vs testing data to verify the inter-patient split
is working correctly, and reports statistical differences between the two.

Run this AFTER feature_extraction.py has created ecg_features.csv.
"""

import os

import pandas as pd
from scipy import stats

from config import FEATURES_CSV, FEATURE_COLUMNS, GROUP_COLUMN, LABEL_COLUMN, MODEL_DIR
from train_models import inter_patient_split


def main():
    df = pd.read_csv(FEATURES_CSV, dtype={GROUP_COLUMN: str})
    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    groups = df[GROUP_COLUMN]

    # Exactly the same DS1 (train) / DS2 (test) split used in train_models.py
    train_idx, test_idx = inter_patient_split(X, y, groups)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # ---------------------------------------------------------------
    # CHECK 1: Confirm no patient overlap between train and test
    # ---------------------------------------------------------------
    train_patients = set(train_df["record"])
    test_patients = set(test_df["record"])
    overlap = train_patients & test_patients

    print("="*60)
    print("CHECK 1: Patient Overlap")
    print("="*60)
    print(f"Train patients: {sorted(train_patients)}")
    print(f"Test patients : {sorted(test_patients)}")
    print(f"Overlapping patients: {overlap if overlap else 'NONE (correct)'}")

    # ---------------------------------------------------------------
    # CHECK 2: Class balance comparison
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("CHECK 2: Class Balance (Normal=0 vs Abnormal=1)")
    print("="*60)
    train_balance = train_df["label"].value_counts(normalize=True).sort_index()
    test_balance = test_df["label"].value_counts(normalize=True).sort_index()

    balance_df = pd.DataFrame({
        "Train (%)": (train_balance * 100).round(2),
        "Test (%)": (test_balance * 100).round(2)
    })
    print(balance_df)

    # ---------------------------------------------------------------
    # CHECK 3: Feature distribution comparison (mean, std, and
    # statistical test for whether distributions differ significantly)
    # ---------------------------------------------------------------
    print("\n" + "="*60)
    print("CHECK 3: Feature Distributions (Train vs Test)")
    print("="*60)

    summary_rows = []
    for col in FEATURE_COLUMNS:
        train_vals = train_df[col]
        test_vals = test_df[col]

        # Kolmogorov-Smirnov test: checks if two distributions differ
        # p < 0.05 suggests a statistically significant difference
        ks_stat, p_value = stats.ks_2samp(train_vals, test_vals)

        summary_rows.append({
            "feature": col,
            "train_mean": train_vals.mean(),
            "test_mean": test_vals.mean(),
            "train_std": train_vals.std(),
            "test_std": test_vals.std(),
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "significantly_different": "Yes" if p_value < 0.05 else "No"
        })

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(summary_df.to_string(index=False))

    print("\nNote: A statistically significant difference (p < 0.05) between "
          "train and test feature distributions is EXPECTED and NORMAL in an "
          "inter-patient split, since different patients naturally have "
          "somewhat different heartbeat characteristics. This is not an error "
          "— it's part of what makes inter-patient evaluation a more "
          "realistic, harder test than a random split.")

    # Save for reference / inclusion in paper appendix if useful
    output_path = os.path.join(MODEL_DIR, "train_vs_test_distribution_check.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed comparison to {output_path}")


if __name__ == "__main__":
    main()
