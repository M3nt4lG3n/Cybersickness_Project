"""
Models/Combined_ML.py

Thin, declarative entry point: wires a DataProvider into an Experiment and
runs it through the Pipeline. Edit TARGET_COLUMN / TASK_TYPE to match the
label you want to predict from this modality's combined features.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Machine_Learning/

from Common.data_provider import CombinedDataProvider
from Common.experiment import Experiment
from Common.pipeline import Pipeline

# reports__Reported_Value = the subjective symptom rating logged during the
# session (Subjective_Results/*_Reports.csv). Continuous -> regression.
# If you'd rather predict an SSQ subscale (nausea/oculomotor/disorientation)
# instead, those aren't in the pre-built *_combined.csv yet - say so and
# CombinedDataProvider can be extended to merge them in.
TARGET_COLUMN = "reports__Reported_Value"
TASK_TYPE = "regression"  # "regression" or "classification"


def main():
    provider = CombinedDataProvider()
    exp = Experiment(
        name="combined_ml",
        provider=provider,
        target_column=TARGET_COLUMN,
        task_type=TASK_TYPE,
    )
    Pipeline(exp).run()
    print(exp.summary())


if __name__ == "__main__":
    main()