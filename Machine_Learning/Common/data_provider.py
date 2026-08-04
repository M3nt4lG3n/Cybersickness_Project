"""
data_provider.py

Loads modality-specific datasets from the indexed patient structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

from .data_loader import DatasetLoader, PatientTrial

class DataProvider(ABC):
    """
    Base class for every modality-specific data provider.
    """

    def __init__(self, loader: DatasetLoader):
        self.loader = loader

    @abstractmethod
    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        pass

    def load_all(self) -> pd.DataFrame:
        dfs = []
        for trial in self.loader.iter_trials():
            try:
                df = self.load_trial(trial)
                if df is None or df.empty:
                    continue
                df["PatientID"] = trial.patient_id
                df["TrialName"] = trial.trial_name
                dfs.append(df)
            except Exception as e:
                print(f"Failed to load {trial.trial_name}: {e}")
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

class CombinedDataProvider(DataProvider):

    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        if trial.combined_file is None:
            return pd.DataFrame()
        return pd.read_csv(trial.combined_file)

class EyeDataProvider(DataProvider):

    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        dfs = []

        for csv in trial.eye_files:
            df = pd.read_csv(csv)

            if "left" in csv.stem.lower():
                prefix = "LeftEye_"
            elif "right" in csv.stem.lower():
                prefix = "RightEye_"
            else:
                prefix = ""

            df.rename(
                columns={
                    c: prefix + c
                    for c in df.columns
                    if c != "UnixTime_ms"
                },
                inplace=True,
            )

            dfs.append(df)

        if len(dfs) == 0:
            return pd.DataFrame()

        if len(dfs) == 1:
            return dfs[0]

        return pd.merge_asof(
            dfs[0].sort_values("UnixTime_ms"),
            dfs[1].sort_values("UnixTime_ms"),
            on="UnixTime_ms",
            direction="nearest",
        )

class UnityDataProvider(DataProvider):

    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        dfs = []

        for csv in trial.unity_files:
            dfs.append(pd.read_csv(csv))

        if not dfs:
            return pd.DataFrame()

        if len(dfs) == 1:
            return dfs[0]

        return pd.concat(dfs, axis=1)

class LabscribeDataProvider(DataProvider):

    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        dfs = []

        for csv in trial.labscribe_files:
            dfs.append(pd.read_csv(csv))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1)

class SubjectiveDataProvider(DataProvider):

    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        dfs = []

        for csv in trial.subjective_files:
            dfs.append(pd.read_csv(csv))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1)

class MSSQDataProvider(DataProvider):

    def load_trial(self, trial: PatientTrial) -> pd.DataFrame:
        patient = self.loader.get_patient(trial.patient_id)

        if patient is None:
            return pd.DataFrame()

        dfs = []

        for csv in patient.mssq_files:
            dfs.append(pd.read_csv(csv))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, axis=1)