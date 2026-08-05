"""
launch_ML.py

Entry point for the Cybersickness ML pipeline.

Runs fine two ways:

1. Hit VS Code's Run button (or `python launch_ML.py`) with no arguments -
   this runs DEFAULT_MODEL below (printing the discovery manifest first as a
   sanity check). Change DEFAULT_MODEL to switch what the Run button does.

2. From a terminal, with explicit flags:

    # Sanity-check what was discovered under Patient_Data, then exit
    python launch_ML.py --discover

    # Run one modality's experiment
    python launch_ML.py --model combined

    # Run every modality
    python launch_ML.py --model all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Common.data_loader import discover_dataset
from Common.utils import get_logger

logger = get_logger("launch_ML")

# Edit this to change what happens when the script is run with no CLI args
# (e.g. VS Code's Run button) - one of MODEL_SCRIPTS' keys, or "all".
DEFAULT_MODEL = "combined"

MODEL_SCRIPTS = {
    "eye": "Models.Eye_ML",
    "labscribe": "Models.Labscribe_ML",
    "unity": "Models.Unity_ML",
    "subjective": "Models.Subjective_ML",
    "mssq": "Models.MSSQ_ML",
    "combined": "Models.Combined_ML",
}


def run_model(key: str):
    import importlib
    module = importlib.import_module(MODEL_SCRIPTS[key])
    try:
        module.main()
    except ValueError as exc:
        # Most common cause: TARGET_COLUMN is still None (the default) in
        # Models/<key>_ML.py. Surface a clear pointer instead of a raw
        # stack trace from deep inside the pipeline.
        logger.error(
            f"Experiment '{key}' failed: {exc}\n"
            f"  -> Check that TARGET_COLUMN is set at the top of "
            f"Models/{key.capitalize()}_ML.py to an actual column name "
            f"(run with --discover, or inspect the relevant DataProvider's "
            f".load().columns, to find it)."
        )
        raise


def print_manifest():
    manifest = discover_dataset()
    if manifest.empty:
        logger.warning("No patients/sessions discovered under Patient_Data.")
    else:
        print(manifest.to_string(index=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Cybersickness ML pipeline launcher")
    parser.add_argument(
        "--model", choices=list(MODEL_SCRIPTS) + ["all"], default=None,
        help=f"Which modality experiment to run. Defaults to '{DEFAULT_MODEL}' "
             "when no flags are given at all (e.g. VS Code's Run button).",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Print the Patient_Data discovery manifest and exit (no modeling).",
    )
    args = parser.parse_args()

    if args.discover:
        print_manifest()
        return

    # No CLI args at all (VS Code Run button, or a bare `python launch_ML.py`)
    # -> show what was discovered, then fall through to DEFAULT_MODEL below.
    if args.model is None:
        print_manifest()
        args.model = DEFAULT_MODEL
        logger.info(f"No --model given; running default model '{DEFAULT_MODEL}'.")

    keys = list(MODEL_SCRIPTS) if args.model == "all" else [args.model]
    for key in keys:
        run_model(key)


if __name__ == "__main__":
    main()
