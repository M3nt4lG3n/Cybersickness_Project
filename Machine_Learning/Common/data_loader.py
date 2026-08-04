"""
data_loader.py

Discovers the patient directory structure and provides a unified interface
for loading modality-specific datasets.

Expected directory structure:

Raw_Eye_Recordings/
│
├── Patient_1/
│   ├── Patient_20260727_121113/
│   ├── Patient_20260727_122654/
│   └── Patient_1_MSSQ.csv
│
├── Patient_2/
│   └── ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import RAW_DATA_DIR

@dataclass(slots=True)
class PatientTrial:
    """
    Represents one experimental VR trial.
    """

    patient_id: str
    trial_name: str
    trial_directory: Path

    eye_directory: Optional[Path] = None
    labscribe_directory: Optional[Path] = None
    unity_directory: Optional[Path] = None
    subjective_directory: Optional[Path] = None

    combined_file: Optional[Path] = None

    eye_files: list[Path] = field(default_factory=list)
    labscribe_files: list[Path] = field(default_factory=list)
    unity_files: list[Path] = field(default_factory=list)
    subjective_files: list[Path] = field(default_factory=list)

    def exists(self) -> bool:
        return self.trial_directory.exists()

@dataclass(slots=True)
class Patient:
    """
    Represents one participant.
    """

    patient_id: str

    patient_directory: Path

    trials: list[PatientTrial] = field(default_factory=list)

    mssq_files: list[Path] = field(default_factory=list)

@dataclass(slots=True)
class DatasetIndex:
    """
    Stores every discovered patient.
    """

    patients: list[Patient] = field(default_factory=list)

    def clear(self):

        self.patients.clear()

    @property
    def number_of_patients(self):

        return len(self.patients)

    @property
    def number_of_trials(self):

        return sum(len(p.trials) for p in self.patients)

class DatasetLoader:
    """
    Discovers every patient and trial in the dataset.

    Scanning only occurs once. Afterwards every model simply queries
    this object.
    """

    def __init__(self, root_directory: Path = RAW_DATA_DIR):

        self.root_directory = Path(root_directory)

        self.dataset = DatasetIndex()

    def scan(self):

        """
        Scan every patient directory.
        """

        self.dataset.clear()

        if not self.root_directory.exists():

            raise FileNotFoundError(
                f"Dataset directory not found:\n{self.root_directory}"
            )

        patient_directories = sorted(

            d for d in self.root_directory.iterdir()

            if d.is_dir()

        )

        for patient_directory in patient_directories:

            patient = self._scan_patient(patient_directory)

            self.dataset.patients.append(patient)

        return self.dataset

    def _scan_patient(self, patient_directory: Path) -> Patient:

        patient = Patient(

            patient_id=patient_directory.name,

            patient_directory=patient_directory,

        )

        for item in sorted(patient_directory.iterdir()):

            if item.is_dir():

                if item.name.startswith("Patient_20"):

                    patient.trials.append(

                        self._scan_trial(

                            patient.patient_id,

                            item,

                        )
                    )

            elif item.suffix.lower() == ".csv":

                if "MSSQ" in item.name.upper():

                    patient.mssq_files.append(item)

        return patient

    def _scan_trial(

        self,

        patient_id: str,

        trial_directory: Path,

    ) -> PatientTrial:

        trial = PatientTrial(

            patient_id=patient_id,

            trial_name=trial_directory.name,

            trial_directory=trial_directory,

        )

        for folder in trial_directory.iterdir():

            if not folder.is_dir():

                continue

            folder_name = folder.name.lower()

            if folder_name == "eye_csvs":

                trial.eye_directory = folder

                trial.eye_files = sorted(folder.glob("*.csv"))

            elif folder_name == "labscribe_csvs":

                trial.labscribe_directory = folder

                trial.labscribe_files = sorted(folder.glob("*.csv"))

            elif folder_name == "unity":

                trial.unity_directory = folder

                trial.unity_files = sorted(folder.glob("*.csv"))

            elif folder_name == "subjective_results":

                trial.subjective_directory = folder

                trial.subjective_files = sorted(folder.glob("*.csv"))

        combined = list(

            trial_directory.glob("*combined.csv")

        )

        if combined:

            trial.combined_file = combined[0]

        return trial

    def get_patient(

        self,

        patient_id: str,

    ) -> Patient | None:

        for patient in self.dataset.patients:

            if patient.patient_id == patient_id:

                return patient

        return None


    def iter_trials(self):

        for patient in self.dataset.patients:

            for trial in patient.trials:

                yield trial

    def summary(self):

        print("=" * 60)

        print("Dataset Summary")

        print("=" * 60)

        print(f"Patients : {self.dataset.number_of_patients}")

        print(f"Trials   : {self.dataset.number_of_trials}")

        print()

        for patient in self.dataset.patients:

            print(

                f"{patient.patient_id:15}"

                f"{len(patient.trials):3} trials"

            )