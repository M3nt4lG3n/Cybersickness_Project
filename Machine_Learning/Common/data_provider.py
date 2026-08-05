"""
Common/data_provider.py

DataProviders load data ONLY (per ML_PIPELINE.md principle: "DataProviders
load data only") - they hand a clean, tagged DataFrame to the rest of the
pipeline (preprocessing -> feature_selection -> model_selection -> ...).

Each provider wraps a PatientDataLoader and knows how to aggregate one
modality (or the combined multimodal table) across the WHOLE Patient_Data
tree, regardless of how many patients/sessions exist or how they're named.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import pandas as pd

from Config import config
from Common.data_loader import PatientDataLoader
from Common.utils import get_logger, tag_feature_group

logger = get_logger("data_provider")


class BaseDataProvider(ABC):
    """Common interface for all modality data providers."""

    feature_group: str = "unknown"

    def __init__(self, loader: Optional[PatientDataLoader] = None):
        self.loader = loader or PatientDataLoader()
        self._cache: Optional[pd.DataFrame] = None

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """Return the raw aggregated DataFrame for this modality."""

    def load(self, refresh: bool = False) -> pd.DataFrame:
        if self._cache is None or refresh:
            self._cache = self._load_raw()
        return self._cache

    def get_data(
        self, target_column: Optional[str] = None, drop_metadata: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
        """Return (X, y, groups) ready for preprocessing/modeling.

        - X: feature columns (metadata columns removed by default)
        - y: the target_column if given and present, else None
        - groups: PatientID, for GroupShuffleSplit / GroupKFold
        """
        df = self.load()
        if df.empty:
            return df, None, pd.Series(dtype=object)

        if config.GROUP_COLUMN not in df.columns:
            raise KeyError(
                f"Expected group column '{config.GROUP_COLUMN}' not present "
                f"in {self.__class__.__name__} data."
            )
        groups = df[config.GROUP_COLUMN]

        y = None
        if target_column is not None:
            if target_column not in df.columns:
                raise KeyError(f"target_column '{target_column}' not found in data.")
            y = df[target_column]

        drop_cols = []
        if drop_metadata:
            drop_cols += [c for c in config.METADATA_COLUMNS if c in df.columns]
        if target_column is not None and target_column in df.columns:
            drop_cols.append(target_column)

        X = df.drop(columns=drop_cols, errors="ignore")
        return X, y, groups

    def feature_columns_by_group(self) -> dict:
        """Map feature_group -> list of column names, using the keyword
        rules in config.FEATURE_GROUP_KEYWORDS. Handy for feature_selection
        or for restricting a model to a subset of modalities."""
        df = self.load()
        groups: dict = {}
        for col in df.columns:
            if col in config.METADATA_COLUMNS:
                continue
            group = tag_feature_group(col, config.FEATURE_GROUP_KEYWORDS)
            groups.setdefault(group, []).append(col)
        return groups


class EyeDataProvider(BaseDataProvider):
    feature_group = "eye"

    def _load_raw(self) -> pd.DataFrame:
        return self.loader.load_all_sessions_modality("eye")


class LabscribeDataProvider(BaseDataProvider):
    feature_group = "labscribe"

    def _load_raw(self) -> pd.DataFrame:
        return self.loader.load_all_sessions_modality("labscribe")


class UnityDataProvider(BaseDataProvider):
    feature_group = "unity"

    def _load_raw(self) -> pd.DataFrame:
        return self.loader.load_all_sessions_modality("unity")


class SubjectiveDataProvider(BaseDataProvider):
    feature_group = "subjective"

    def _load_raw(self) -> pd.DataFrame:
        return self.loader.load_all_sessions_modality("subjective")


class MSSQDataProvider(BaseDataProvider):
    """Patient-level susceptibility metrics. No TrialName/session grain -
    one row per patient."""

    feature_group = "mssq"

    def _load_raw(self) -> pd.DataFrame:
        return self.loader.load_all_mssq()


class CombinedDataProvider(BaseDataProvider):
    """Preferred multimodal input (DATA_SCHEMA.md: 'Combined CSV - Preferred
    input for multimodal models'). Aggregates every session's *_combined.csv
    and left-joins patient-level MSSQ metrics on PatientID so susceptibility
    features are available alongside session-level readings."""

    feature_group = "combined"

    def __init__(self, loader: Optional[PatientDataLoader] = None, include_mssq: bool = True):
        super().__init__(loader)
        self.include_mssq = include_mssq

    def _load_raw(self) -> pd.DataFrame:
        combined = self.loader.load_all_combined()
        if combined.empty:
            logger.warning(
                "No *_combined.csv files found; falling back to an on-the-fly "
                "merge of Eye/Labscribe/Unity/Subjective modalities per session."
            )
            combined = self._build_combined_fallback()

        if self.include_mssq and not combined.empty:
            mssq = self.loader.load_all_mssq()
            if not mssq.empty and "PatientID" in combined.columns:
                mssq_cols = [c for c in mssq.columns if c == "PatientID" or c not in combined.columns]
                combined = combined.merge(mssq[mssq_cols], on="PatientID", how="left")

        return combined

    def _build_combined_fallback(self) -> pd.DataFrame:
        """If no pre-built combined CSV exists for a session, concatenate
        (not join, since sampling rates differ across modalities) the
        available modality tables so at least a usable dataset is produced."""
        frames = []
        for modality in ("eye", "labscribe", "unity", "subjective"):
            df = self.loader.load_all_sessions_modality(modality)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)


PROVIDERS = {
    "eye": EyeDataProvider,
    "labscribe": LabscribeDataProvider,
    "unity": UnityDataProvider,
    "subjective": SubjectiveDataProvider,
    "mssq": MSSQDataProvider,
    "combined": CombinedDataProvider,
}


def get_provider(name: str, loader: Optional[PatientDataLoader] = None) -> BaseDataProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown data provider '{name}'. Options: {list(PROVIDERS)}")
    return PROVIDERS[name](loader)
