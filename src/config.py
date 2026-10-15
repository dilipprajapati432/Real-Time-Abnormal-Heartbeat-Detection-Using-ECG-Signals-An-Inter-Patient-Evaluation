"""
Shared settings used by every script in the project.

Keeping paths, feature names and label rules in ONE place means the
feature extractor, trainer, API, dashboard and figure scripts can never
silently drift out of sync with each other.
"""

import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data", "mitbih_data")
FEATURES_CSV = os.path.join(PROJECT_DIR, "data", "ecg_features.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
FIG_DIR = os.path.join(MODEL_DIR, "figures")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")    # full sklearn Pipeline
MODEL_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")   # name, features, test patients

# ---------------------------------------------------------------------------
# Annotation symbols
# ---------------------------------------------------------------------------
# MIT-BIH .atr files contain BOTH heartbeat annotations and non-beat
# annotations (rhythm changes '+', noise '~', comments '"', artifacts '|',
# non-conducted P waves 'x', flutter waves '!', flutter start/end '[' ']').
# Only real heartbeats may become rows, and only real heartbeats may be
# used to compute RR intervals.
BEAT_SYMBOLS = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?',
}

# Beats treated as "Normal" (AAMI class N). Everything else in
# BEAT_SYMBOLS is "Abnormal".
NORMAL_LABELS = {'N', 'L', 'R', 'e', 'j'}

# Records 102, 104, 107 and 217 contain paced beats. The AAMI EC57
# recommendation (followed by most MIT-BIH papers) is to exclude them.
PACED_RECORDS = {'102', '104', '107', '217'}
EXCLUDE_PACED_RECORDS = True

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
WINDOW = 90            # samples before/after the beat for local shape stats (250 ms at 360 Hz)
RR_HISTORY = 10        # number of previous RR intervals used for the local RR average

# Every feature below is CAUSAL: it only uses the current beat, past beats
# and 250 ms of signal after the R peak. No feature needs the NEXT beat,
# so a beat can be classified ~250 ms after it occurs (true real time).
FEATURE_COLUMNS = [
    "rr_prev",        # time since previous beat (s)
    "rr_local_avg",   # mean of the last RR_HISTORY RR intervals (s)
    "rr_ratio",       # rr_prev / rr_local_avg  (patient-normalised prematurity)
    "amplitude",      # signal value at the annotated R peak (mV)
    "local_mean",     # mean of the +-WINDOW signal around the beat
    "local_std",      # std of the +-WINDOW signal around the beat
]

LABEL_COLUMN = "label"
GROUP_COLUMN = "record"

# ---------------------------------------------------------------------------
# AAMI EC57 beat classes (used for per-class reporting)
# ---------------------------------------------------------------------------
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',   # normal & bundle branch block
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',             # supraventricular ectopic
    'V': 'V', 'E': 'V',                                 # ventricular ectopic
    'F': 'F',                                           # fusion
    '/': 'Q', 'f': 'Q', 'Q': 'Q', '?': 'Q',             # paced / unclassifiable
}
AAMI_CLASSES = ['N', 'S', 'V', 'F']   # Q is too rare after excluding paced records

# ---------------------------------------------------------------------------
# Inter-patient split: de Chazal et al. (2004) DS1 / DS2
# ---------------------------------------------------------------------------
# The standard MIT-BIH inter-patient protocol. 22 records for training,
# 22 different records for testing; paced records are in neither set.
DS1_RECORDS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122',
               '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2_RECORDS = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210',
               '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

CV_FOLDS = 5             # patient-grouped CV folds inside DS1 (for model selection)
TEST_SIZE = 0.2          # only used for the intra-patient (random split) comparison
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5
