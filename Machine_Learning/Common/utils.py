"""
utils.py

Shared helper functions used throughout the ML pipeline.
"""

from pathlib import Path
from datetime import datetime
import json
import pickle

def timestamp():

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path):

    Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle(obj, filename):

    ensure_directory(Path(filename).parent)

    with open(filename, "wb") as f:

        pickle.dump(obj, f)


def load_pickle(filename):

    with open(filename, "rb") as f:

        return pickle.load(f)


def save_json(dictionary, filename):

    ensure_directory(Path(filename).parent)

    with open(filename, "w") as f:

        json.dump(dictionary, f, indent=4)