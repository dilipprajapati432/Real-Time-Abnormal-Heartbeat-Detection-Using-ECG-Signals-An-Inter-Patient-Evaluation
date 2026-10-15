"""
Part 5: Streamlit dashboard — streams a raw ECG recording sample by sample,
detects each heartbeat with the causal QRS detector (no reference beat
positions are used), and shows real-time predictions from the FastAPI backend.

Upgrades in this version:
  1. Timing between beats uses the ACTUAL RR interval (scaled), so faster
     heart rhythms stream faster and slower rhythms stream slower — just
     like a real patient.
  2. A live "Heart Rate (BPM)" readout, calculated from the RR interval.
  3. A live latency counter showing real inference time per beat.
  4. The actual ECG waveform snippet around the current beat is plotted,
     not just the abnormal-probability line — this looks like a real
     ECG monitor instead of just a chart of numbers.

Run with:
    streamlit run dashboard.py

Make sure the FastAPI server (api.py) is already running in another
terminal at http://127.0.0.1:8000 before starting this.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
import time
import os
import json
import wfdb

from config import AAMI_MAP, DATA_DIR as RAW_DATA_DIR, MODEL_META_PATH, WINDOW
from feature_extraction import load_record
from qrs_detector import match_beats
from realtime_pipeline import RealtimeBeatStream

API_URL = "http://127.0.0.1:8000/predict"

MAX_ROWS_SHOWN = 10
MAX_ALERTS_SHOWN = 5
N_BEATS_TO_SIMULATE = 50
WAVEFORM_WINDOW = WINDOW    # same window as feature_extraction.py

CHUNK_SECONDS = 0.1          # the "sensor" delivers 100 ms of ECG at a time
WARMUP_SECONDS = 10          # signal fed (without display) so the detector can adapt
# Real time would be 1.0; >1 replays faster than real life to keep the demo short.
TIME_COMPRESSION = 4

st.set_page_config(page_title="Real-Time Heartbeat Monitor", layout="wide")
st.title("Real-Time Abnormal Heartbeat Detection Dashboard")

with open(MODEL_META_PATH) as f:
    META = json.load(f)
FEATURES = META["feature_columns"]
THRESHOLD = META.get("decision_threshold", 0.5)

# Only stream patients the model never saw during training (inter-patient demo)
test_records = META["test_records"]
selected_record = st.selectbox("Held-out patient record to stream", test_records)


@st.cache_data
def get_record(record_name):
    return load_record(str(record_name))


signal, fs, ref_samples, ref_symbols = get_record(selected_record)
ref_abnormal = [s for s, sym in zip(ref_samples, ref_symbols) if AAMI_MAP.get(sym, "Q") != "N"]

# Most recordings start with normal beats; by default start ~10 s before the
# first annotated abnormal beat so the demo reaches abnormal beats quickly.
default_start = max(0.0, (ref_abnormal[0] / fs - 10) if ref_abnormal else 0.0)
st.caption(f"Record {selected_record}: {len(ref_samples)} annotated beats, "
           f"{len(ref_abnormal)} abnormal"
           + (f" — first abnormal beat at {ref_abnormal[0] / fs:.1f} s" if ref_abnormal else ""))
start_time = st.number_input("Start time (seconds)", min_value=0.0,
                             max_value=float(len(signal) / fs - 30), value=float(round(default_start, 1)),
                             step=10.0, key=f"start_{selected_record}")

st.write(f"Model: **{META['model_name']}** — the raw ECG of record {selected_record} "
         "(never seen in training) is fed in 100 ms chunks. Beats are found by the "
         "streaming QRS detector; reference annotations are used only to show the true label.")

start = st.button("Start Live Simulation")

metrics_placeholder = st.empty()
alert_log_placeholder = st.empty()
waveform_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

history = []
alert_log = []


def reference_label(sample_index):
    """Expert label of the reference beat within 150 ms, for display only."""
    pairs = match_beats([sample_index], ref_samples, int(0.15 * fs))
    if not pairs:
        return "no reference beat (false detection)"
    sym = ref_symbols[pairs[0][1]]
    return f"{'Normal' if AAMI_MAP.get(sym, 'Q') == 'N' else 'Abnormal'} ({sym})"


if start:
    metrics_placeholder.info("Warming up the QRS detector...")
    stream = RealtimeBeatStream(fs=fs)
    chunk = int(CHUNK_SECONDS * fs)
    first = int(max(0, start_time - WARMUP_SECONDS) * fs)
    display_from = int(start_time * fs)
    shown = 0

    for k in range(first, len(signal), chunk):
        beats = stream.push(signal[k:k + chunk])
        live = k >= display_from
        if live:
            time.sleep(CHUNK_SECONDS / TIME_COMPRESSION)

        for beat in beats:
            if beat["sample_index"] < display_from:
                continue
            payload = {name: float(beat["features"][name]) for name in FEATURES}
            t0 = time.perf_counter()
            try:
                result = requests.post(API_URL, json=payload, timeout=2).json()
            except Exception as e:
                metrics_placeholder.error(f"Could not reach API: {e}")
                st.stop()
            latency_ms = (time.perf_counter() - t0) * 1000

            i = beat["beat_number"]
            rr_prev = beat["features"]["rr_prev"]
            bpm = 60.0 / rr_prev if rr_prev > 0 else 0
            true_label = reference_label(beat["sample_index"])
            history.append({"beat": i, "time_s": round(beat["time_s"], 2),
                            "prediction": result["prediction"], "true_label": true_label,
                            "abnormal_probability": result["abnormal_probability"]})

            with metrics_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Heart Rate", f"{bpm:.0f} BPM")
                c2.metric("Decision delay after R peak", f"{beat['decision_delay_ms']:.0f} ms")
                c3.metric("API round-trip", f"{latency_ms:.1f} ms")
                c4.metric("Beat # / reference", f"{i} / {true_label.split(' ')[0]}")
                msg = f"Beat #{i} at {beat['time_s']:.1f} s (probability {result['abnormal_probability']})"
                if result["prediction"] == "Abnormal":
                    st.error("ALERT: Abnormal heartbeat — " + msg)
                    alert_log.append(msg)
                else:
                    st.success("Normal — " + msg)

            if alert_log:
                alert_log_placeholder.warning(
                    "Recent alerts:\n" + "\n".join(f"- {a}" for a in alert_log[-MAX_ALERTS_SHOWN:][::-1]))

            r = beat["sample_index"]
            seg = signal[max(0, r - WINDOW): r + WINDOW]
            waveform_df = pd.DataFrame({"time_ms": (np.arange(len(seg)) - min(r, WINDOW)) / fs * 1000,
                                        "amplitude": seg})
            color = "#E24B4A" if result["prediction"] == "Abnormal" else "#5B9BD5"
            with waveform_placeholder.container():
                st.altair_chart(
                    alt.Chart(waveform_df).mark_line(color=color).encode(
                        x=alt.X("time_ms:Q", title="Time from detected R peak (ms)"),
                        y=alt.Y("amplitude:Q", title="ECG amplitude (mV)"))
                    .properties(height=200, title=f"ECG around detected beat #{i} ({result['prediction']})"),
                    width="stretch")

            history_df = pd.DataFrame(history)
            with chart_placeholder.container():
                base = alt.Chart(history_df)
                line = base.mark_line(color="#5B9BD5").encode(
                    x=alt.X("beat:Q", title="Beat number"),
                    y=alt.Y("abnormal_probability:Q", title="Abnormal probability",
                            scale=alt.Scale(domain=[0, 1])))
                points = base.mark_circle(size=80).encode(
                    x="beat:Q", y="abnormal_probability:Q",
                    color=alt.Color("prediction:N", scale=alt.Scale(domain=["Normal", "Abnormal"],
                                                                    range=["#5B9BD5", "#E24B4A"]),
                                    legend=alt.Legend(title="Prediction")),
                    tooltip=["beat", "time_s", "prediction", "true_label", "abnormal_probability"])
                threshold_line = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
                    color="#888880", strokeDash=[4, 4]).encode(y="y:Q")
                st.altair_chart((line + points + threshold_line).properties(
                    height=280, title="Live abnormal heartbeat probability"), width="stretch")
            with table_placeholder.container():
                st.dataframe(history_df.tail(MAX_ROWS_SHOWN), width="stretch")
            shown += 1

        if shown >= N_BEATS_TO_SIMULATE:
            break

    metrics_placeholder.info(f"Simulation complete — {shown} beats detected and classified.")
