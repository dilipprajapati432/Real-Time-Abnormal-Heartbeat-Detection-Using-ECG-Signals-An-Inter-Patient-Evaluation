"""
Causal, streaming QRS (R-peak) detector based on the Pan-Tompkins algorithm
(Pan & Tompkins, IEEE Trans. Biomed. Eng., 1985).

The detector consumes the raw ECG in chunks of any size (even one sample at a
time) and returns the sample index of every R peak it confirms. It only ever
looks at samples it has already received, so it can run on a live signal.

Pipeline (all causal):
  1. 5-15 Hz band-pass (2nd-order Butterworth, lfilter with carried state)
  2. 5-point derivative
  3. squaring
  4. 150 ms moving-window integration (MWI)
  5. adaptive signal/noise thresholds on MWI peaks, 200 ms refractory period,
     T-wave rejection (slope test) and search-back for missed beats
  6. R-peak localisation on the raw signal around the detected QRS

Typical confirmation delay: about 150-250 ms after the R peak.
"""

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


class StreamingQRSDetector:
    def __init__(self, fs=360.0, learning_seconds=2.0):
        self.fs = fs
        self.ms = lambda t: int(round(t * fs / 1000.0))       # milliseconds -> samples

        # 1) band-pass filter with persistent state
        self.b_bp, self.a_bp = butter(2, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype="band")
        self.zi_bp = None
        self.bp_delay = self.ms(42)                            # ~15 samples group delay at 360 Hz
        # 2) derivative  y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
        self.b_der = np.array([2, 1, 0, -1, -2]) * (fs / 8.0)
        self.zi_der = np.zeros(len(self.b_der) - 1)
        # 4) moving-window integration
        self.win = self.ms(150)
        self.b_mwi = np.ones(self.win) / self.win
        self.zi_mwi = np.zeros(self.win - 1)

        # history buffers (needed to localise R peaks and for search-back)
        self.raw, self.bp, self.der, self.mwi = [], [], [], []
        self.n = 0                                             # samples received so far

        # adaptive thresholds
        self.learning_samples = int(learning_seconds * fs)
        self.spki = self.npki = None
        self.refractory = self.ms(200)
        self.t_wave_window = self.ms(360)

        # peak tracking on the MWI signal
        self.tracking = False
        self.cand_val = 0.0
        self.cand_idx = 0
        self.prev_m = 0.0

        # beat bookkeeping
        self.last_qrs_mwi = -10 ** 9          # MWI index of last accepted QRS
        self.last_qrs_slope = None
        self.rr_recent = []                   # last 8 RR intervals (samples)
        self.searchback_cand = None           # (value, idx) of best sub-threshold peak since last QRS
        self.r_peaks = []

    # ------------------------------------------------------------------ helpers
    def _thresholds(self):
        thr1 = self.npki + 0.25 * (self.spki - self.npki)
        return thr1, 0.5 * thr1

    def _rr_avg(self):
        return np.mean(self.rr_recent) if self.rr_recent else self.fs  # default 1 s

    def _slope_at(self, mwi_idx):
        lo = max(0, mwi_idx - self.win)
        seg = self.der[lo:mwi_idx + 1]
        return float(np.max(np.abs(seg))) if len(seg) else 0.0

    def _locate_r(self, mwi_idx):
        """R-peak sample on the raw signal for a QRS whose MWI peak is at mwi_idx."""
        lo = max(0, mwi_idx - self.win - self.ms(30))
        seg = np.abs(self.bp[lo:mwi_idx + 1])
        if not len(seg):
            return mwi_idx
        j = lo + int(np.argmax(seg)) - self.bp_delay          # undo band-pass delay
        a, b = max(0, j - self.ms(60)), min(len(self.raw), j + self.ms(60) + 1)
        raw_seg = np.asarray(self.raw[a:b])
        if not len(raw_seg):
            return max(j, 0)
        dev = np.abs(raw_seg - np.median(raw_seg))
        return a + int(np.argmax(dev))

    def _accept_qrs(self, val, idx, searchback=False):
        if searchback:
            self.spki = 0.25 * val + 0.75 * self.spki
        else:
            self.spki = 0.125 * val + 0.875 * self.spki
        if self.last_qrs_mwi > 0:
            self.rr_recent.append(idx - self.last_qrs_mwi)
            self.rr_recent = self.rr_recent[-8:]
        self.last_qrs_mwi = idx
        self.last_qrs_slope = self._slope_at(idx)
        self.searchback_cand = None
        r = self._locate_r(idx)
        if not self.r_peaks or r - self.r_peaks[-1] > self.refractory:
            self.r_peaks.append(r)
            return r
        return None

    def _classify_peak(self, val, idx):
        """Decide whether an MWI peak is a QRS complex. Returns an R index or None."""
        if idx - self.last_qrs_mwi < self.refractory:
            return None
        thr1, thr2 = self._thresholds()
        if val > thr1:
            # possible T wave: close to the previous beat and much smaller slope
            if (idx - self.last_qrs_mwi < self.t_wave_window and self.last_qrs_slope
                    and self._slope_at(idx) < 0.5 * self.last_qrs_slope):
                self.npki = 0.125 * val + 0.875 * self.npki
                return None
            return self._accept_qrs(val, idx)
        self.npki = 0.125 * val + 0.875 * self.npki
        if val > thr2 and (self.searchback_cand is None or val > self.searchback_cand[0]):
            self.searchback_cand = (val, idx)
        return None

    # ------------------------------------------------------------------ public
    def process(self, chunk):
        """Feed new raw samples; returns the list of R-peak indices confirmed in this call."""
        x = np.asarray(chunk, dtype=float)
        if not len(x):
            return []
        if self.zi_bp is None:
            self.zi_bp = lfilter_zi(self.b_bp, self.a_bp) * x[0]
        bp, self.zi_bp = lfilter(self.b_bp, self.a_bp, x, zi=self.zi_bp)
        der, self.zi_der = lfilter(self.b_der, [1.0], bp, zi=self.zi_der)
        mwi, self.zi_mwi = lfilter(self.b_mwi, [1.0], der ** 2, zi=self.zi_mwi)

        self.raw.extend(x.tolist())
        self.bp.extend(bp.tolist())
        self.der.extend(der.tolist())
        self.mwi.extend(mwi.tolist())

        found = []
        start = self.n
        self.n += len(x)

        if self.spki is None:
            if self.n < self.learning_samples:
                self.prev_m = self.mwi[-1]
                return []
            first = np.asarray(self.mwi[:self.learning_samples])
            self.spki = 0.25 * float(first.max())
            self.npki = 0.5 * float(first.mean())
            start = max(start, self.learning_samples)

        for i in range(start, self.n):
            m = self.mwi[i]
            if not self.tracking:
                if m > self.prev_m:
                    self.tracking, self.cand_val, self.cand_idx = True, m, i
            elif m > self.cand_val:
                self.cand_val, self.cand_idx = m, i
            elif m < 0.5 * self.cand_val or i - self.cand_idx > self.ms(300):
                r = self._classify_peak(self.cand_val, self.cand_idx)
                if r is not None:
                    found.append(r)
                self.tracking = False
            self.prev_m = m

            # search-back for a missed beat
            if (self.searchback_cand is not None and self.last_qrs_mwi > 0
                    and i - self.last_qrs_mwi > 1.66 * self._rr_avg()):
                val, idx = self.searchback_cand
                r = self._accept_qrs(val, idx, searchback=True)
                if r is not None:
                    found.append(r)
        return found


def detect_qrs(signal, fs=360.0, chunk_size=360):
    """Run the streaming detector over a whole recording, chunk by chunk."""
    det = StreamingQRSDetector(fs=fs)
    peaks = []
    for s in range(0, len(signal), chunk_size):
        peaks.extend(det.process(signal[s:s + chunk_size]))
    return np.array(sorted(set(peaks)), dtype=int)


def match_beats(detected, reference, tolerance):
    """Greedy one-to-one matching (AAMI EC57 style). Returns (det_idx, ref_idx) pairs."""
    detected, reference = np.asarray(detected), np.asarray(reference)
    pairs, used = [], np.zeros(len(detected), dtype=bool)
    j0 = 0
    for ri, r in enumerate(reference):
        while j0 < len(detected) and detected[j0] < r - tolerance:
            j0 += 1
        best, best_d = None, tolerance + 1
        j = j0
        while j < len(detected) and detected[j] <= r + tolerance:
            d = abs(detected[j] - r)
            if not used[j] and d < best_d:
                best, best_d = j, d
            j += 1
        if best is not None:
            used[best] = True
            pairs.append((best, ri))
    return pairs
