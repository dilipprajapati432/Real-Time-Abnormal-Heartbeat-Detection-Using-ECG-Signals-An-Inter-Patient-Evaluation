"""
Latency Benchmark: how fast the saved model classifies one heartbeat,
and how many heartbeats per second it can process in batch.

Run this AFTER train_models.py has created models/best_model.pkl.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from config import BEST_MODEL_PATH, FEATURES_CSV, MODEL_DIR, MODEL_META_PATH

N_SINGLE_PREDICTIONS = 1000   # one-at-a-time predictions to time
N_BATCH_PREDICTIONS = 10000   # predictions used for the throughput test


def load_model_and_data():
    model = joblib.load(BEST_MODEL_PATH)
    with open(MODEL_META_PATH) as f:
        meta = json.load(f)
    df = pd.read_csv(FEATURES_CSV)
    X = df[meta["feature_columns"]]
    return model, meta, X


def benchmark_single_prediction_latency(model, X, n=N_SINGLE_PREDICTIONS):
    """One beat at a time, the same way the FastAPI endpoint scores it (pipeline incl. scaling)."""
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(X), size=min(n, len(X)), replace=False)

    model.predict_proba(X.iloc[[sample_indices[0]]])   # warm-up, not counted

    latencies_ms = []
    for idx in sample_indices:
        row = X.iloc[[idx]]
        start = time.perf_counter()
        model.predict_proba(row)
        latencies_ms.append((time.perf_counter() - start) * 1000)
    return np.array(latencies_ms)


def benchmark_batch_throughput(model, X, n=N_BATCH_PREDICTIONS):
    batch = X.iloc[:min(n, len(X))]
    start = time.perf_counter()
    model.predict_proba(batch)
    total_time = time.perf_counter() - start
    return total_time, len(batch) / total_time


def main():
    print("Loading model and data...")
    model, meta, X = load_model_and_data()
    print(f"Model: {meta['model_name']} | {len(X)} heartbeat samples available.\n")

    latencies = benchmark_single_prediction_latency(model, X)
    print("=== Single-Prediction Latency (one beat at a time) ===")
    print(f"Mean   : {latencies.mean():.4f} ms")
    print(f"Median : {np.median(latencies):.4f} ms")
    print(f"P95    : {np.percentile(latencies, 95):.4f} ms")
    print(f"Max    : {latencies.max():.4f} ms")

    total_time, throughput = benchmark_batch_throughput(model, X)
    print("\n=== Batch Throughput ===")
    print(f"Throughput: {throughput:.1f} predictions/second ({total_time:.4f} s)")

    summary = pd.DataFrame([{
        "model": meta["model_name"],
        "mean_latency_ms": latencies.mean(),
        "median_latency_ms": np.median(latencies),
        "p95_latency_ms": np.percentile(latencies, 95),
        "max_latency_ms": latencies.max(),
        "throughput_per_sec": throughput,
    }])
    summary.to_csv(os.path.join(MODEL_DIR, "latency_benchmark.csv"), index=False)
    print("\nSaved models/latency_benchmark.csv")

    print("\n=== Suggested sentence for your paper ===")
    print(f'"The {meta["model_name"]} model achieves a mean inference latency of '
          f'{latencies.mean():.2f} ms per heartbeat (95th percentile: '
          f'{np.percentile(latencies, 95):.2f} ms) and a batch throughput of '
          f'{throughput:.0f} predictions per second. Because all features are causal, '
          f'each beat can be classified about 250 ms after its R peak."')
    print("Note: these numbers measure model inference only (no HTTP); "
          "measured on your machine - report the hardware in the paper.")


if __name__ == "__main__":
    main()
