"""
Real-time beat pipeline: raw ECG samples in, classified-ready beats out.

    stream = RealtimeBeatStream(fs=360)
    for chunk in ecg_chunks:                      # any chunk size, e.g. 0.1 s
        for beat in stream.push(chunk):
            beat["features"]  -> send to the API /predict

A beat is emitted as soon as (1) the QRS detector has confirmed it and
(2) 250 ms of signal after its R peak has arrived (needed by the morphology
features). Nothing from the future beyond that is used.
"""

import numpy as np

from config import WINDOW
from feature_extraction import features_for_beat
from qrs_detector import StreamingQRSDetector


class RealtimeBeatStream:
    def __init__(self, fs=360.0):
        self.fs = fs
        self.detector = StreamingQRSDetector(fs=fs)
        self.signal = []          # every sample received so far
        self.r_peaks = []         # detected R-peak sample indices
        self.next_beat = 1        # first beat that can have features (needs a previous beat)

    def push(self, chunk):
        chunk = np.asarray(chunk, dtype=float)
        self.signal.extend(chunk.tolist())
        for r in self.detector.process(chunk):
            if not self.r_peaks or r > self.r_peaks[-1]:
                self.r_peaks.append(r)

        out = []
        n = len(self.signal)
        while self.next_beat < len(self.r_peaks) and self.r_peaks[self.next_beat] + WINDOW < n:
            i = self.next_beat
            # only the +-WINDOW segment around the beat is needed; shift indices to it
            r = self.r_peaks[i]
            if r - WINDOW < 0:
                self.next_beat += 1
                continue
            offset = r - WINDOW
            sig = np.asarray(self.signal[offset: r + WINDOW + 1])
            local = [p - offset for p in self.r_peaks[: i + 1]]
            feats = features_for_beat(sig, self.fs, local, i)
            if feats is not None:
                out.append({
                    "beat_number": i,
                    "sample_index": self.r_peaks[i],
                    "time_s": self.r_peaks[i] / self.fs,
                    "decision_delay_ms": (n - self.r_peaks[i]) / self.fs * 1000,
                    "features": feats,
                })
            self.next_beat += 1
        return out
