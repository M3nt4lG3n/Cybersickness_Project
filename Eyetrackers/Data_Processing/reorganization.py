"""
reorganization.py

Reorganizes a completed recording/analysis session folder
(e.g. Patient_20260727_121113) into subfolders, for easier
navigation.

Must be run AFTER every pipeline output (eye tracking files,
raw LabScribe export, ECG/balance/summary CSVs, videos, plot,
merged "readings" CSVs, unity biometrics) already exists in the
session folder - this module only moves files, it never
generates them.

Resulting layout inside a session folder:

    Raw_Eye_Videos/
        left_eye.mp4
        right_eye.mp4

    Cropped_Eye_Videos/
        left_eye_cropped.mp4
        right_eye_cropped.mp4

    Eye_CSVs/
        left_eye.csv
        left_eye_readings.csv
        left_pupil.csv
        right_eye.csv
        right_eye_readings.csv
        right_pupil.csv

    Raw_Labscribe/
        <stem>.xls
        <stem>.iwxdata
        <stem>.csv

    Labscribe_CSVs/
        <stem>_analysis.csv
        <stem>_beats.csv
        <stem>_summary.csv
        <stem>_timestamped.csv

    Unity/
        unity_biometrics.csv
        unity_reports.csv

    Visualization/
        <stem>_balance.mp4
        <stem>_unity.mp4
        <stem>_ecg.png
        <stem>_Hop_Count.png
        <stem>_Percentile_Curve.png
        <stem>_Total_Severity.png

    Subjective_Results/
        <stem>_Reports.csv
        <stem>_SSQ.csv

`<stem>` is the LabScribe file stem for this session (e.g.
"Patient_1_0.0"), taken from the session's .iwxdata file name -
this is what main.py's `create_output_paths()` already uses to
name every generated analysis file.

Note on the MSSQ csv: subjective_processing.py writes
`<mssq_stem>_scored.csv` (and expects the raw MSSQ csv, e.g.
"Patient_1_MSSQ.csv") directly in the super-folder that holds a
patient's session subfolders, not inside any individual session
folder -- since the MSSQ is collected once per patient rather than
once per session. reorganize_session() only ever looks at files
directly inside the session folder it's given, so it never touches
those MSSQ files; they're intentionally left where they are.

Can be imported and called directly from main.py:

    from . import reorganization
    reorganization.reorganize_session(session_dir)

or run standalone from VSCode, in which case it prompts for a
Patient super-folder or an individual session folder, the same
way cleanup_generated_files.py does.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil


# ---------------------------------------------------------------------
# Target subfolders
# ---------------------------------------------------------------------

SUBFOLDERS = (
    "Raw_Eye_Videos",
    "Cropped_Eye_Videos",
    "Eye_CSVs",
    "Raw_Labscribe",
    "Labscribe_CSVs",
    "Unity",
    "Visualization",
    "Subjective_Results",
)

# Files whose name never changes between sessions.
FIXED_NAME_DESTINATIONS = {
    "left_eye.mp4": "Raw_Eye_Videos",
    "right_eye.mp4": "Raw_Eye_Videos",

    "left_eye_cropped.mp4": "Cropped_Eye_Videos",
    "right_eye_cropped.mp4": "Cropped_Eye_Videos",

    "left_eye.csv": "Eye_CSVs",
    "right_eye.csv": "Eye_CSVs",
    "left_pupil.csv": "Eye_CSVs",
    "right_pupil.csv": "Eye_CSVs",
    "left_eye_readings.csv": "Eye_CSVs",
    "right_eye_readings.csv": "Eye_CSVs",

    "unity_biometrics.csv": "Unity",
    "unity_reports.csv": "Unity",
}

# Suffixes appended to the session's LabScribe stem
# (e.g. "Patient_1_0.0"), mapped to their destination. Order
# matters: more specific suffixes must be checked before the
# bare ".csv"/".xls" ones so e.g. "_analysis.csv" isn't matched
# as a generic ".csv".
#
# Note: the MSSQ csv and its scored counterpart are NOT listed here --
# they live in the super-folder above the session subfolders, not inside
# any individual session folder, so reorganize_session() never
# encounters them (see the module docstring's MSSQ note).
STEM_SUFFIX_DESTINATIONS = (
    ("_analysis.csv", "Labscribe_CSVs"),
    ("_beats.csv", "Labscribe_CSVs"),
    ("_summary.csv", "Labscribe_CSVs"),
    ("_timestamped.csv", "Labscribe_CSVs"),
    ("_balance.mp4", "Visualization"),
    ("_unity.mp4", "Visualization"),
    ("_ecg.png", "Visualization"),
    ("_Hop_Count.png", "Visualization"),
    ("_Percentile_Curve.png", "Visualization"),
    ("_Total_Severity.png", "Visualization"),
    ("_Reports.csv", "Subjective_Results"),
    ("_SSQ_scored.csv", "Subjective_Results"),
    ("_SSQ.csv", "Subjective_Results"),
    (".xls", "Raw_Labscribe"),
    (".iwxdata", "Raw_Labscribe"),
    (".csv", "Raw_Labscribe"),
)

# Same pattern used by main.py / cleanup_generated_files.py to
# recognize session folders.
PATIENT_FOLDER_PATTERN = re.compile(r"^Patient_\d{8}_\d{6}$")


# ---------------------------------------------------------------------
# Stem discovery
# ---------------------------------------------------------------------

def find_labscribe_stem(session_dir: Path) -> str | None:
    """
    Determine the LabScribe stem (e.g. "Patient_1_0.0") for a
    session folder from its .iwxdata file name. Looks directly
    inside `session_dir` and, if it's already been reorganized,
    inside Raw_Labscribe/ as well.

    Returns None if no .iwxdata file can be found.
    """

    search_dirs = [session_dir, session_dir / "Raw_Labscribe"]

    for directory in search_dirs:

        if not directory.is_dir():
            continue

        matches = sorted(directory.glob("*.iwxdata"))

        if not matches:
            continue

        if len(matches) > 1:
            print(
                f"  Note: multiple .iwxdata files found in {directory}; "
                f"using {matches[0].name} as the session stem."
            )

        return matches[0].stem

    return None


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------

def classify_file(item: Path, stem: str | None) -> str | None:
    """
    Return the destination subfolder name for `item`, or None if
    it doesn't match any known pattern.
    """

    if item.name in FIXED_NAME_DESTINATIONS:
        return FIXED_NAME_DESTINATIONS[item.name]

    if stem is not None and item.name.startswith(stem):

        remainder = item.name[len(stem):]

        for suffix, destination in STEM_SUFFIX_DESTINATIONS:

            if remainder == suffix:
                return destination

    return None


# ---------------------------------------------------------------------
# Reorganization
# ---------------------------------------------------------------------

def reorganize_session(session_dir: Path) -> tuple[int, list[str]]:
    """
    Sort every recognized file directly inside `session_dir` into
    its destination subfolder. Files already sitting inside one
    of the target subfolders (e.g. from a previous run) are left
    alone. Unrecognized files are left in place at the session
    root.

    Returns (files_moved, unclassified_file_names).
    """

    session_dir = Path(session_dir)

    stem = find_labscribe_stem(session_dir)

    if stem is None:
        print(
            f"  Warning: no .iwxdata file found in {session_dir}; "
            "Raw_Labscribe files (.xls/.iwxdata/.csv) can't be "
            "identified and will be left unclassified."
        )

    moved = 0
    unclassified: list[str] = []

    for item in sorted(session_dir.iterdir()):

        if item.is_dir():
            # Subfolders (including ones this function already
            # created on a previous run) are left as-is.
            continue

        destination_name = classify_file(item, stem)

        if destination_name is None:
            unclassified.append(item.name)
            continue

        dest_dir = session_dir / destination_name
        dest_dir.mkdir(exist_ok=True)

        target = dest_dir / item.name

        if target.exists():
            print(
                f"  Skipped {item.name}: {target} already exists."
            )
            continue

        shutil.move(str(item), str(target))
        moved += 1
        print(f"  {item.name} -> {destination_name}/")

    return moved, unclassified


def reorganize_sessions(session_dirs: list[Path]) -> None:
    """
    Reorganize a batch of session folders, printing a summary
    per folder.
    """

    for session_dir in session_dirs:

        print(f"\nReorganizing: {session_dir}")

        moved, unclassified = reorganize_session(session_dir)

        print(f"  Moved {moved} file(s) into subfolders.")

        if unclassified:
            print(
                "  Left unclassified at session root: "
                + ", ".join(unclassified)
            )


# ---------------------------------------------------------------------
# Standalone usage (VSCode / direct run)
# ---------------------------------------------------------------------

def main():

    from tkinter import Tk, filedialog, messagebox

    root = Tk()
    root.withdraw()

    selected = filedialog.askdirectory(
        title="Select Patient folder or session folder to reorganize"
    )

    if not selected:
        print("No folder selected.")
        return

    selected = Path(selected)

    session_dirs = sorted(
        d for d in selected.iterdir()
        if d.is_dir() and PATIENT_FOLDER_PATTERN.fullmatch(d.name)
    )

    if session_dirs:
        print(f"\nDetected Patient superfolder: {selected.name}")
        print(f"Found {len(session_dirs)} session folder(s).")
    else:
        print(f"\nTreating as an individual session folder: {selected.name}")
        session_dirs = [selected]

    reorganize_sessions(session_dirs)

    messagebox.showinfo(
        "Reorganization Complete",
        f"Reorganized {len(session_dirs)} session folder(s).",
    )


if __name__ == "__main__":
    main()