"""
config.py

Global configuration for the Cybersickness ML pipeline.

Every model imports settings from this file so that changing
one parameter updates the entire project.
"""

from pathlib import Path

###############################################################################
# PROJECT PATHS
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "Raw_Eye_Recordings"

RESULTS_DIR = PROJECT_ROOT / "MachineLearning" / "Results"

MODELS_DIR = PROJECT_ROOT / "MachineLearning" / "SavedModels"

FIGURES_DIR = PROJECT_ROOT / "MachineLearning" / "Figures"

CACHE_DIR = PROJECT_ROOT / "MachineLearning" / "Cache"

###############################################################################
# RANDOMNESS
###############################################################################

RANDOM_SEED = 42

###############################################################################
# TRAIN / TEST
###############################################################################

TEST_SIZE = 0.20

VALIDATION_SPLITS = 5

STRATIFIED_SPLITS = True

###############################################################################
# FEATURE FILTERING
###############################################################################

MAX_MISSING_PERCENT = 0.30

MIN_VARIANCE = 1e-6

MAX_CORRELATION = 0.95

###############################################################################
# SCALING
###############################################################################

NUMERIC_SCALER = "standard"

IMPUTATION = "median"

###############################################################################
# FEATURE IMPORTANCE
###############################################################################

TOP_FEATURES = 30

SHAP_SAMPLE_SIZE = 1000

PERMUTATION_REPEATS = 20

###############################################################################
# MODEL COMPARISON
###############################################################################

COMPARE_MODELS = [

    "LogisticRegression",

    "RandomForest",

    "ExtraTrees",

    "GradientBoosting",

    "XGBoost",

    "LightGBM",

    "CatBoost",

    "SVM",

    "MLP"

]

###############################################################################
# HYPERPARAMETER SEARCH
###############################################################################

OPTIMIZER = "bayesian"

SEARCH_ITERATIONS = 50

###############################################################################
# TARGET VARIABLE
###############################################################################

TARGET_COLUMN = "Cybersickness"

###############################################################################
# OUTPUT OPTIONS
###############################################################################

SAVE_MODELS = True

SAVE_FIGURES = True

SAVE_FEATURE_IMPORTANCE = True

SAVE_SHAP = True

VERBOSE = True