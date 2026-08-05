"""
Common/preprocessing.py

Implements the missing-data and train/test split policy from DATA_SCHEMA.md:
  - Numeric: median imputation
  - Categorical: most-frequent imputation
  - Drop columns with excessive missingness
  - Always split by PatientID (GroupShuffleSplit / GroupKFold)
"""

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from Config import config
from Common.utils import get_logger

logger = get_logger("preprocessing")


def drop_high_missingness(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    threshold = config.MISSING_DATA_DROP_THRESHOLD if threshold is None else threshold
    missing_frac = df.isna().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} columns with >{threshold:.0%} missing: {drop_cols}")
    return df.drop(columns=drop_cols)


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns, most-frequent-impute categorical
    columns. Operates on a copy; returns a new DataFrame with the same
    column order."""
    if df.empty:
        return df

    df = drop_high_missingness(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    out = df.copy()
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        out[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        out[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    return out


def encode_categoricals(df: pd.DataFrame, max_categories: int = 50) -> pd.DataFrame:
    """One-hot encode object/categorical columns so every column handed to
    a model is numeric (sklearn estimators reject raw strings, e.g. a
    'Red'/'Green' fixation-color column or 'Gender').

    High-cardinality string columns (more distinct values than
    max_categories) are dropped instead of exploded into hundreds of
    dummy columns, with a warning - if that column is actually meant to be
    numeric, the warning is a good pointer to a data-quality issue (stray
    non-numeric values) rather than a genuine category.
    """
    if df.empty:
        return df

    cat_cols = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if not cat_cols:
        return df

    df = df.copy()
    drop_cols = []
    for col in cat_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > max_categories:
            logger.warning(
                f"Column '{col}' has {n_unique} distinct values (>{max_categories}); "
                "dropping rather than one-hot encoding. If this is meant to be "
                "numeric, check for stray non-numeric values in the raw CSV."
            )
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)
        cat_cols = [c for c in cat_cols if c not in drop_cols]

    if cat_cols:
        logger.info(f"One-hot encoding {len(cat_cols)} categorical column(s): {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

    return df


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = None,
    random_state: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split by PatientID so no patient appears in both train and test."""
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = config.RANDOM_STATE if random_state is None else random_state

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx] if y is not None else None
    y_test = y.iloc[test_idx] if y is not None else None
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    logger.info(
        f"Group split: {groups_train.nunique()} train patients / "
        f"{groups_test.nunique()} test patients "
        f"({len(X_train)} / {len(X_test)} rows)"
    )
    return X_train, X_test, y_train, y_test, groups_train, groups_test


def group_kfold_splits(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) folds grouped by PatientID, for use in
    cross_val_score / GridSearchCV via cv=list(group_kfold_splits(...))."""
    n_splits = config.N_SPLITS if n_splits is None else n_splits
    n_groups = groups.nunique()

    if n_groups < 2:
        # Small pilot cohorts (common early in VR/cybersickness studies) can
        # leave only one patient in the training fold. GroupKFold requires
        # >=2 groups, so fall back to a plain (non-grouped) KFold on rows and
        # warn loudly - the PatientID-isolation guarantee does not hold for
        # this particular comparison until more patients are collected.
        logger.warning(
            f"Only {n_groups} unique patient(s) available for cross-validation; "
            "cannot group-split. Falling back to an ungrouped KFold - treat "
            "these CV scores as provisional until more patients are added."
        )
        if len(X) < 2:
            raise ValueError(
                f"Only {len(X)} training row(s) available - not enough to run "
                "any cross-validation. Add more patients/sessions to Patient_Data "
                "before running model comparison."
            )
        n_splits = max(2, min(n_splits, len(X)))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        yield from kfold.split(X, y)
        return

    if n_groups < n_splits:
        logger.warning(
            f"Only {n_groups} unique patients but n_splits={n_splits}; "
            f"reducing n_splits to {n_groups}."
        )
        n_splits = n_groups

    kfold = GroupKFold(n_splits=n_splits)
    yield from kfold.split(X, y, groups=groups)