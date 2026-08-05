"""
Config/config.py

Central configuration for the Cybersickness ML pipeline.
Defines paths, constants, and the feature-group keyword mappings used to
identify which raw columns belong to which modality (per DATA_SCHEMA.md).

This is the ONLY file that should need to change if:
  - the raw data location moves (CS_PATIENT_DATA_DIR env var, or edit below)
  - new columns are added to a modality's raw export (extend FEATURE_GROUP_KEYWORDS)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMMON_DIR = Path(__file__).resolve().parent          # .../Machine_Learning/Config
PROJECT_ROOT = COMMON_DIR.parent                        # .../Machine_Learning
REPO_ROOT = PROJECT_ROOT.parent                         # .../Cybersickness_Project

# Root folder containing all raw per-patient data. Defaults to
# Cybersickness_Project/Patient_Data but can be overridden with an env var
# (useful for tests / alternate data drives) without touching any code.
PATIENT_DATA_DIR = Path(os.environ.get("CS_PATIENT_DATA_DIR", REPO_ROOT / "Patient_Data"))

RESULTS_DIR = PROJECT_ROOT / "Results"
SAVED_MODELS_DIR = PROJECT_ROOT / "Saved Models"
FIGURE_DIR = PROJECT_ROOT / "Figure"
CACHE_DIR = PROJECT_ROOT / "Common" / ".cache"

for _dir in (RESULTS_DIR, SAVED_MODELS_DIR, FIGURE_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Schema (per DATA_SCHEMA.md)
# ---------------------------------------------------------------------------

# Identifier columns - never treated as predictive features.
METADATA_COLUMNS = ["PatientID", "TrialName", "UnixTime_ms", "Source"]

# Sub-folder / file-name patterns used by the data loader to discover raw
# files without depending on the exact patient/session naming (those change
# per capture, e.g. Patient_1_0.0 -> Patient_2_0.3).
PATIENT_DIR_PATTERN = r"^Patient_\d+$"                     # e.g. Patient_1
SESSION_DIR_PATTERN = r"^Patient_\d{8}_\d{6}$"              # e.g. Patient_20260727_121113
MSSQ_FILE_PATTERN = r".*_MSSQ(_scored)?\.csv$"
COMBINED_FILE_PATTERN = r".*_combined\.csv$"

MODALITY_SUBDIRS = {
    "eye": "Eye_CSVs",
    "labscribe": "Labscribe_CSVs",
    "subjective": "Subjective_Results",
    "unity": "Unity",
}

# Feature group -> case-insensitive substring keywords used to auto-tag
# which columns in a loaded CSV belong to that modality's semantic group.
# Extend as new columns are added to raw exports; nothing else in the
# pipeline needs to change.
FEATURE_GROUP_KEYWORDS = {
    "eye": [
        "pupil", "blink", "confidence", "eye_open", "eye openness",
        "openness", "gaze",
    ],
    "labscribe": [
        "heart rate", "heartrate", "hr_", "bpm", "rr_interval",
        "rr interval", "hrv", "ecg", "balance", "cop_", "cop ",
    ],
    "unity": [
        "head_pos", "head position", "head_rot", "head rotation",
        "velocity", "acceleration", "reaction_time", "reaction time",
        "gameplay",
    ],
    "subjective": [
        "symptom", "ssq", "nausea", "oculomotor", "disorientation",
    ],
    "mssq": ["mssq"],
}

MISSING_DATA_DROP_THRESHOLD = 0.5   # drop columns missing more than 50% of rows

# ---------------------------------------------------------------------------
# Train / test split policy (DATA_SCHEMA.md: always split by PatientID)
# ---------------------------------------------------------------------------
GROUP_COLUMN = "PatientID"
TEST_SIZE = 0.2
N_SPLITS = 5
