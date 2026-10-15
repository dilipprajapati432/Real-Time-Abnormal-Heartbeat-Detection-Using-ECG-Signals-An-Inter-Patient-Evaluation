"""
Part 2: Extract one row of features per heartbeat from every complete
record in data/mitbih_data, and save them to data/ecg_features.csv.

Fixes vs the first version:
  * Non-beat annotations ('+', '~', '"', '|', 'x', '!', '[', ']') are no
    longer turned into "abnormal heartbeats", and are no longer used as
    neighbours when computing RR intervals.
  * Features are causal (no use of the NEXT beat), so the model can run
    in true real time. rr_next is still saved, but only for the dashboard's
    playback timing - it is NOT a model feature.
  * Adds patient-normalised RR features (rr_local_avg, rr_ratio).
  * Paced records are excluded (AAMI EC57), configurable in config.py.

Run this AFTER load_data.py.
"""

import numpy as np
import pandas as pd
import wfdb
import os

from config import (DATA_DIR, FEATURES_CSV, BEAT_SYMBOLS, NORMAL_LABELS,
                    PACED_RECORDS, EXCLUDE_PACED_RECORDS, WINDOW, RR_HISTORY)
from load_data import available_records


def features_for_beat(signal, fs, samples, i):
    """The six causal features of beat i, given the beat positions `samples`
    (annotated or detected) and the signal. Uses only beats 0..i and 250 ms of
    signal after beat i. Returns None if the beat is too close to the edges."""
    idx = int(samples[i])
    if i < 1 or idx - WINDOW < 0 or idx + WINDOW >= len(signal):
        return None
    rr = np.diff(np.asarray(samples[max(0, i - RR_HISTORY):i + 1])) / fs
    rr_prev = float(rr[-1])
    rr_local_avg = float(np.mean(rr))
    window_signal = signal[idx - WINDOW: idx + WINDOW]
    return {
        "rr_prev": rr_prev,
        "rr_local_avg": rr_local_avg,
        "rr_ratio": rr_prev / rr_local_avg if rr_local_avg > 0 else 1.0,
        "amplitude": float(signal[idx]),
        "local_mean": float(np.mean(window_signal)),
        "local_std": float(np.std(window_signal)),
    }


def features_from_beats(signal, fs, samples, symbols, record_name):
    """One row per beat (except the first and last). `symbols` may contain None
    for beats without a reference label (e.g. false detections)."""
    rows = []
    for i in range(1, len(samples) - 1):
        f = features_for_beat(signal, fs, samples, i)
        if f is None:
            continue
        symbol = symbols[i]
        rows.append({
            "record": record_name,
            "sample_index": int(samples[i]),
            **f,
            "rr_next": (samples[i + 1] - samples[i]) / fs,   # dashboard timing only
            "symbol": symbol,
            "label": (0 if symbol in NORMAL_LABELS else 1) if symbol is not None else np.nan,
        })
    return pd.DataFrame(rows)


def load_record(record_name, data_dir=DATA_DIR):
    """First ECG channel, sampling rate, and the reference beats (sample, symbol)."""
    record_path = os.path.join(data_dir, record_name)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, 'atr')
    beats = [(s, sym) for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS]
    return (record.p_signal[:, 0], record.fs,
            np.array([b[0] for b in beats]), [b[1] for b in beats])


def extract_features_for_record(record_name, data_dir=DATA_DIR):
    """Return a DataFrame with one row per real (annotated) heartbeat in a record."""
    signal, fs, samples, symbols = load_record(record_name, data_dir)
    if len(samples) < 3:
        return pd.DataFrame()
    return features_from_beats(signal, fs, samples, symbols, record_name)


def build_full_dataset(record_list):
    all_dfs = []
    for record_name in record_list:
        print(f"Extracting features for record {record_name} ...")
        try:
            all_dfs.append(extract_features_for_record(record_name))
        except Exception as e:
            print(f"  Skipped {record_name} due to error: {e}")
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    records = available_records()
    if EXCLUDE_PACED_RECORDS:
        skipped = [r for r in records if r in PACED_RECORDS]
        records = [r for r in records if r not in PACED_RECORDS]
        print(f"Excluding paced records (AAMI EC57): {skipped}")
    print(f"Using {len(records)} records.\n")

    dataset = build_full_dataset(records)

    print(f"\nTotal beats extracted: {len(dataset)}")
    print("Label distribution:")
    print(dataset["label"].value_counts())
    print("\nSymbol counts:")
    print(dataset["symbol"].value_counts())

    dataset.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved feature dataset to {FEATURES_CSV}")
