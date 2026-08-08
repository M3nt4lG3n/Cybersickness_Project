"""
patient_config.py
------------------
Shared, per-patient parameter store used by crop.py / crop_batch.py /
pupils.py / pupils_batch.py.

All four scripts import this module instead of hard-coding crop boxes or
pupil-detection thresholds. Everything is keyed off a "Patient_<int>"
folder name found somewhere in a video's path, and split further by
eye side ("left" / "right"), since left/right eye videos need their own
crop box and their own detection thresholds.

Config file layout (patient_config.json, stored next to this file):

{
  "Patient_4": {
    "left":  {"crop_width": 109, "crop_height": 138, "start_x": 119,
              "start_y": 79, "border_size": 0,
              "RELAXED_THRESHOLD": 40, "MEDIUM_THRESHOLD": 50,
              "STRICT_THRESHOLD": 60, "SQUARE_SIZE": 200},
    "right": {...}
  },
  "Patient_5": {...}
}

border_color is intentionally never stored here -- it is always white,
per the tools' existing behavior.
"""

import json
import os
import re
import threading
import tkinter as tk
from tkinter import messagebox

# -----------------------------------------------------------------------
# Location / locking
# -----------------------------------------------------------------------

CONFIG_FILENAME = "patient_config.json"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)

_lock = threading.Lock()

# -----------------------------------------------------------------------
# Patient / side identification
# -----------------------------------------------------------------------

PATIENT_DIR_RE = re.compile(r"^Patient_(\d+)$", re.IGNORECASE)

SIDES = ("left", "right")

CROP_KEYS = ["crop_width", "crop_height", "start_x", "start_y", "border_size"]
PUPIL_KEYS = ["RELAXED_THRESHOLD", "MEDIUM_THRESHOLD", "STRICT_THRESHOLD", "SQUARE_SIZE"]
ALL_KEYS = CROP_KEYS + PUPIL_KEYS

# Fallback starting values for a brand-new patient/side that has nothing
# saved yet. These mirror the constants the original scripts used to
# hard-code at module level.
DEFAULT_CROP = {
    "crop_width": 109,
    "crop_height": 138,
    "start_x": 119,
    "start_y": 79,
    "border_size": 0,
}

DEFAULT_PUPIL = {
    "RELAXED_THRESHOLD": 40,
    "MEDIUM_THRESHOLD": 50,
    "STRICT_THRESHOLD": 60,
    "SQUARE_SIZE": 200,
}


def find_patient_dir(path):
    """Walk upward from `path` looking for a directory literally named
    Patient_<int> (case-insensitive). Returns the absolute path to that
    directory, or None if none of the ancestors match."""
    path = os.path.abspath(path)
    while True:
        name = os.path.basename(path)
        if PATIENT_DIR_RE.match(name):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def find_patient_id(path):
    """Same search as find_patient_dir(), but returns the canonical
    'Patient_X' id string (int-normalized) instead of a filesystem path."""
    patient_dir = find_patient_dir(path)
    if patient_dir is None:
        return None
    m = PATIENT_DIR_RE.match(os.path.basename(patient_dir))
    return f"Patient_{int(m.group(1))}"


def get_eye_side(filename):
    """Returns 'left', 'right', or None based on the filename alone
    (e.g. left_eye.mp4, right_eye_cropped.mp4, Left_Eye.MP4, ...)."""
    lower = os.path.basename(filename).lower()
    if "left_eye" in lower:
        return "left"
    if "right_eye" in lower:
        return "right"
    return None


# -----------------------------------------------------------------------
# Load / save
# -----------------------------------------------------------------------

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config):
    with _lock:
        tmp_path = CONFIG_PATH + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        os.replace(tmp_path, CONFIG_PATH)


# -----------------------------------------------------------------------
# Accessors
# -----------------------------------------------------------------------

def get_patient(config, patient_id):
    return config.get(patient_id, {})


def get_side(config, patient_id, side):
    return config.get(patient_id, {}).get(side, {})


def update_side(config, patient_id, side, values):
    """Merges `values` into config[patient_id][side], creating either
    level as needed. Returns the (mutated) config for convenience."""
    config.setdefault(patient_id, {})
    config[patient_id].setdefault(side, {})
    config[patient_id][side].update(values)
    return config


def has_all_keys(side_cfg, keys):
    return all(k in side_cfg and side_cfg[k] is not None for k in keys)


def has_crop_params(side_cfg):
    return has_all_keys(side_cfg, CROP_KEYS)


def has_pupil_params(side_cfg):
    return has_all_keys(side_cfg, PUPIL_KEYS)


def get_crop_values(config, patient_id, side):
    """Crop parameters for this patient/side, falling back to defaults
    for any key that hasn't been saved yet."""
    side_cfg = get_side(config, patient_id, side)
    result = dict(DEFAULT_CROP)
    result.update({k: side_cfg[k] for k in CROP_KEYS if k in side_cfg})
    return result


def get_pupil_values(config, patient_id, side):
    """Pupil-detection parameters for this patient/side, falling back to
    defaults for any key that hasn't been saved yet."""
    side_cfg = get_side(config, patient_id, side)
    result = dict(DEFAULT_PUPIL)
    result.update({k: side_cfg[k] for k in PUPIL_KEYS if k in side_cfg})
    return result


# -----------------------------------------------------------------------
# Small shared Tk helpers (simple yes/no + error dialogs)
# -----------------------------------------------------------------------

def ask_yes_no(title, message):
    root = tk.Tk()
    root.withdraw()
    try:
        return messagebox.askyesno(title, message)
    finally:
        root.destroy()


def show_error(title, message):
    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showerror(title, message)
    finally:
        root.destroy()


def show_info(title, message):
    root = tk.Tk()
    root.withdraw()
    try:
        messagebox.showinfo(title, message)
    finally:
        root.destroy()
