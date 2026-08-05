"""
Models/labscribe_ML.py

Thin, declarative entry point: wires a DataProvider into an Experiment and
runs it through the Pipeline. Edit TARGET_COLUMN / TASK_TYPE to match the
label you want to predict from this modality's labscribe features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Machine_Learning/

from Common.data_provider import LabscribeDataProvider
from Common.experiment import Experiment
from Common.pipeline import Pipeline

# TODO: set this to an actual column in your Subjective_Results / combined
# data, e.g. an SSQ total score, once you've confirmed the exact column name
# via Common.data_loader.discover_dataset() / provider.load().columns
TARGET_COLUMN = None
TASK_TYPE = "regression"  # "regression" or "classification"


def main():
    provider = LabscribeDataProvider()
    exp = Experiment(
        name="labscribe_ml",
        provider=provider,
        target_column=TARGET_COLUMN,
        task_type=TASK_TYPE,
    )
    Pipeline(exp).run()
    print(exp.summary())


if __name__ == "__main__":
    main()
