"""
Common/data_loader.py

Discovers and loads raw patient data from the Patient_Data folder.

Expected layout (exact numeric/timestamp tokens are irrelevant and change
per capture - discovery is done by STRUCTURE and REGEX PATTERN, never by
hardcoding a specific patient number or capture timestamp):

Patient_Data/
    Patient_<N>/                              <- patient dir   ("Patient_1")
        Patient_<YYYYMMDD>_<HHMMSS>/           <- session dir   ("Patient_20260727_121113")
            Eye_CSVs/
                *.csv                          (e.g. left_eye_readings.csv, right_eye_readings.csv)
            Labscribe_CSVs/
                *.csv                          (e.g. Patient_1_0.0_analysis.csv, ..._beats.csv, ...)
            Subjective_Results/
                *.csv                          (e.g. Patient_1_0.0_Reports.csv, ..._SSQ.csv, ...)
            Unity/
                *.csv                          (e.g. unity_biometrics.csv, unity_reports.csv)
            *_combined.csv                     <- preferred multimodal input for this session
        Patient_<YYYYMMDD>_<HHMMSS>/           <- more sessions ...
        Patient_<N>_MSSQ.csv                   <- patient-level susceptibility data
        Patient_<N>_MSSQ_scored.csv

Nothing in this module depends on the specific values of <N>, <YYYYMMDD>,
<HHMMSS>, or the trailing "_0.0" style suffix in filenames - only on the
directory nesting and the fixed sub-folder names from DATA_SCHEMA.md
(Eye_CSVs, Labscribe_CSVs, Subjective_Results, Unity) and file-name suffixes
(_combined.csv, _MSSQ.csv, _MSSQ_scored.csv).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from Config import config
from Common.utils import get_logger, safe_read_csv

logger = get_logger("data_loader")

PATIENT_DIR_RE = re.compile(config.PATIENT_DIR_PATTERN)
SESSION_DIR_RE = re.compile(config.SESSION_DIR_PATTERN)
MSSQ_FILE_RE = re.compile(config.MSSQ_FILE_PATTERN, re.IGNORECASE)
COMBINED_FILE_RE = re.compile(config.COMBINED_FILE_PATTERN, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Discovery data structures
# ---------------------------------------------------------------------------
@dataclass
class PatientSession:
    patient_id: str
    session_id: str
    session_dir: Path
    modality_dirs: Dict[str, Path] = field(default_factory=dict)
    combined_csv: Optional[Path] = None


@dataclass
class Patient:
    patient_id: str
    patient_dir: Path
    sessions: List[PatientSession] = field(default_factory=list)
    mssq_files: List[Path] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
class PatientDataLoader:
    """Discovers patients/sessions under Patient_Data and loads their CSVs.

    Discovery is purely structural (directory nesting + fixed sub-folder
    names + regex on the variable id/timestamp tokens), so adding, removing,
    or renumbering patients and capture sessions requires no code changes.
    """

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root is not None else config.PATIENT_DATA_DIR
        if not self.root.exists():
            logger.warning(f"Patient_Data root does not exist: {self.root}")

    # -- discovery -----------------------------------------------------
    def discover(self) -> List[Patient]:
        """Walk the Patient_Data tree and build the Patient/PatientSession
        index. Cheap (no CSV parsing) - safe to call repeatedly."""
        patients: List[Patient] = []
        if not self.root.exists():
            return patients

        for patient_dir in sorted(self.root.iterdir()):
            if not patient_dir.is_dir() or not PATIENT_DIR_RE.match(patient_dir.name):
                continue

            patient_id = self._extract_patient_id(patient_dir.name)
            patient = Patient(
                patient_id=patient_id,
                patient_dir=patient_dir,
                mssq_files=sorted(
                    f for f in patient_dir.glob("*.csv") if MSSQ_FILE_RE.match(f.name)
                ),
            )

            for session_dir in sorted(patient_dir.iterdir()):
                if not session_dir.is_dir() or not SESSION_DIR_RE.match(session_dir.name):
                    continue

                modality_dirs = {}
                for modality, subdir_name in config.MODALITY_SUBDIRS.items():
                    found = self._find_subdir_case_insensitive(session_dir, subdir_name)
                    if found is not None:
                        modality_dirs[modality] = found

                combined_matches = [
                    f for f in session_dir.glob("*.csv") if COMBINED_FILE_RE.match(f.name)
                ]
                combined_csv = combined_matches[0] if combined_matches else None
                if len(combined_matches) > 1:
                    logger.warning(
                        f"Multiple *_combined.csv found in {session_dir}, using {combined_csv}"
                    )

                patient.sessions.append(
                    PatientSession(
                        patient_id=patient_id,
                        session_id=session_dir.name,
                        session_dir=session_dir,
                        modality_dirs=modality_dirs,
                        combined_csv=combined_csv,
                    )
                )

            if not patient.sessions:
                logger.warning(f"No session folders found under {patient_dir}")

            patients.append(patient)

        return patients

    def manifest(self) -> pd.DataFrame:
        """Flat table describing what was discovered - useful for a quick
        sanity check that every expected file/folder was picked up."""
        rows = []
        for patient in self.discover():
            if not patient.sessions:
                rows.append({
                    "PatientID": patient.patient_id,
                    "TrialName": None,
                    "has_eye": False, "has_labscribe": False,
                    "has_subjective": False, "has_unity": False,
                    "has_combined": False,
                    "n_mssq_files": len(patient.mssq_files),
                })
                continue
            for session in patient.sessions:
                rows.append({
                    "PatientID": patient.patient_id,
                    "TrialName": session.session_id,
                    "has_eye": "eye" in session.modality_dirs,
                    "has_labscribe": "labscribe" in session.modality_dirs,
                    "has_subjective": "subjective" in session.modality_dirs,
                    "has_unity": "unity" in session.modality_dirs,
                    "has_combined": session.combined_csv is not None,
                    "n_mssq_files": len(patient.mssq_files),
                })
        return pd.DataFrame(rows)

    # -- loading: per-session raw modality data -------------------------
    def load_modality(self, session: PatientSession, modality: str) -> pd.DataFrame:
        """Load and concatenate every CSV in a session's modality folder
        (e.g. all CSVs under Eye_CSVs), tagging each row with PatientID,
        TrialName, and Source (originating filename)."""
        subdir = session.modality_dirs.get(modality)
        if subdir is None:
            logger.warning(
                f"No '{modality}' folder for session {session.session_id} "
                f"(patient {session.patient_id})"
            )
            return pd.DataFrame()

        frames = []
        for csv_path in sorted(subdir.glob("*.csv")):
            df = safe_read_csv(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df["PatientID"] = session.patient_id
            df["TrialName"] = session.session_id
            df["Source"] = csv_path.stem
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        # Different files within a modality (e.g. left/right eye, or
        # Labscribe's analysis/beats/summary/timestamped exports) are not
        # guaranteed to share a row-alignment key, so they are stacked
        # (concatenated) rather than joined. Columns unique to one file are
        # NaN-filled for the others - this is standard/expected here and is
        # handled by the imputation step in preprocessing.py.
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)

    def load_eye(self, session: PatientSession) -> pd.DataFrame:
        return self.load_modality(session, "eye")

    def load_labscribe(self, session: PatientSession) -> pd.DataFrame:
        return self.load_modality(session, "labscribe")

    def load_subjective(self, session: PatientSession) -> pd.DataFrame:
        return self.load_modality(session, "subjective")

    def load_unity(self, session: PatientSession) -> pd.DataFrame:
        return self.load_modality(session, "unity")

    def load_combined(self, session: PatientSession) -> pd.DataFrame:
        """Load the pre-built *_combined.csv for a session (preferred input
        for multimodal models per DATA_SCHEMA.md)."""
        if session.combined_csv is None:
            logger.warning(f"No combined CSV for session {session.session_id}")
            return pd.DataFrame()
        df = safe_read_csv(session.combined_csv)
        if df.empty:
            return df
        df = df.copy()
        if "PatientID" not in df.columns:
            df["PatientID"] = session.patient_id
        if "TrialName" not in df.columns:
            df["TrialName"] = session.session_id
        return df

    def load_mssq(self, patient: Patient) -> pd.DataFrame:
        """Load and merge a patient's MSSQ file(s) into a single one-row
        (per patient) DataFrame of susceptibility metrics."""
        if not patient.mssq_files:
            return pd.DataFrame()

        frames = [safe_read_csv(f) for f in patient.mssq_files]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()

        merged = frames[0].copy()
        for extra in frames[1:]:
            shared = [c for c in merged.columns if c in extra.columns]
            if shared:
                merged = merged.merge(extra, on=shared, how="outer", suffixes=("", "_dup"))
            else:
                merged = pd.concat(
                    [merged.reset_index(drop=True), extra.reset_index(drop=True)],
                    axis=1,
                )
        merged["PatientID"] = patient.patient_id
        return merged

    # -- loading: full-dataset aggregation ------------------------------
    def load_all_sessions_modality(self, modality: str) -> pd.DataFrame:
        """Aggregate one modality across every discovered patient/session."""
        frames = []
        for patient in self.discover():
            for session in patient.sessions:
                df = self.load_modality(session, modality)
                if not df.empty:
                    frames.append(df)
        if not frames:
            logger.warning(f"No data found for modality '{modality}' under {self.root}")
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)

    def load_all_combined(self) -> pd.DataFrame:
        """Aggregate the preferred *_combined.csv across every session."""
        frames = []
        for patient in self.discover():
            for session in patient.sessions:
                df = self.load_combined(session)
                if not df.empty:
                    frames.append(df)
        if not frames:
            logger.warning(f"No combined CSVs found under {self.root}")
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)

    def load_all_mssq(self) -> pd.DataFrame:
        """Aggregate MSSQ (patient-level) data across every patient."""
        frames = []
        for patient in self.discover():
            df = self.load_mssq(patient)
            if not df.empty:
                frames.append(df)
        if not frames:
            logger.warning(f"No MSSQ files found under {self.root}")
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _extract_patient_id(patient_dir_name: str) -> str:
        match = re.search(r"\d+", patient_dir_name)
        return match.group(0) if match else patient_dir_name

    @staticmethod
    def _find_subdir_case_insensitive(parent: Path, name: str) -> Optional[Path]:
        direct = parent / name
        if direct.is_dir():
            return direct
        lname = name.lower()
        for entry in parent.iterdir():
            if entry.is_dir() and entry.name.lower() == lname:
                return entry
        return None


def discover_dataset(root: Optional[Path] = None) -> pd.DataFrame:
    """Convenience function: return the discovery manifest as a DataFrame
    for a quick sanity check, e.g.:

        >>> from Common.data_loader import discover_dataset
        >>> discover_dataset()
    """
    return PatientDataLoader(root).manifest()
