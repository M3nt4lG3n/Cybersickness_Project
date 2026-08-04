"""
cleanup_generated_files.py

Removes generated analysis/processing files from eye-tracking recordings.

Behavior
--------
- Prompts the user to select either:
    1. A Patient superfolder (e.g. Patient_1)
    2. An individual recording folder (e.g. Patient_20260727_163406)

- If a Patient superfolder is selected:
    Searches for subfolders matching:
        Patient_YYYYMMDD_HHMMSS
    and cleans each one.

- If an individual recording folder is selected:
    Cleans only that folder.

Recording folders may be flat, or reorganized into the
subfolders produced by reorganization.py (Raw_Eye_Videos,
Cropped_Eye_Videos, Eye_CSVs, Raw_Labscribe, Labscribe_CSVs,
Unity, Visualization) - this script searches recursively either
way.

The following files are ALWAYS preserved:
    left_eye_cropped.mp4
    left_eye.csv
    left_eye.mp4
    left_pupil.csv
    right_eye_cropped.mp4
    right_eye.csv
    right_eye.mp4
    right_pupil.csv
    unity_biometrics.csv
    *.iwxdata
    *.xls

Also always preserved are the subjective-input files produced by
subjective_inputs.py:
    Patient_<N>_<trial>_Reports.csv   (e.g. Patient_1_0.0_Reports.csv)
    Patient_<N>_<trial>_SSQ.csv       (e.g. Patient_1_0.0_SSQ.csv)
    Patient_<N>_MSSQ.csv

The Reports/SSQ files live inside a recording folder (directly,
or under a Subjective_Results subfolder) and are flattened to
that recording folder's root along with the other kept files.
The MSSQ file lives one level up, directly under the Patient
superfolder, alongside the recording folders rather than inside
one - since this script only cleans within each recording folder
it selects, that file is never touched either way, but it is
still recognized as a keeper in case it's ever found nested
inside a recording folder.

Preserved files are moved back to the top level of the
recording folder (flattening any reorganized subfolders), and
everything else - including all generated CSVs (e.g.
*_analysis.csv, *_readings.csv, *_summary.csv), videos, plots,
and every subdirectory itself - is deleted. The end result is
always the flat layout:

    Patient_1/
        Patient_1_MSSQ.csv
        Patient_20260727_163406/
            left_eye_cropped.mp4
            left_eye.csv
            left_eye.mp4
            left_pupil.csv
            right_eye_cropped.mp4
            right_eye.csv
            right_eye.mp4
            right_pupil.csv
            unity_biometrics.csv
            Patient_1_0.0_Reports.csv
            Patient_1_0.0_SSQ.csv
            *.iwxdata
            *.xls

Run from VSCode:
    python cleanup_generated_files.py
"""

from pathlib import Path
from tkinter import Tk, filedialog, messagebox
import shutil
import re

# ---------------------------------------------------------------------
# Files that should always remain
# ---------------------------------------------------------------------

KEEP_FILES = {
    "left_eye_cropped.mp4",
    "left_eye.csv",
    "left_eye.mp4",
    "left_pupil.csv",
    "right_eye_cropped.mp4",
    "right_eye.csv",
    "right_eye.mp4",
    "right_pupil.csv",
    "unity_biometrics.csv",
}

KEEP_EXTENSIONS = {
    ".iwxdata",
    ".xls",
}

# The Reports/SSQ/MSSQ csv filenames produced by subjective_inputs.py embed
# a patient name and a trial identifier (either a 0.X value or, as a
# fallback, a folder name), so they're matched by pattern rather than by
# exact name, e.g.:
#   Patient_1_0.0_Reports.csv
#   Patient_1_0.0_SSQ.csv
#   Patient_1_MSSQ.csv
SUBJECTIVE_INPUT_PATTERNS = (
    re.compile(r"^Patient_.+_(Reports|SSQ)\.csv$", re.IGNORECASE),
    re.compile(r"^Patient_.+_MSSQ\.csv$", re.IGNORECASE),
)

# Recording folder names look like:
# Patient_20260727_163406
PATIENT_FOLDER_PATTERN = re.compile(r"^Patient_\d{8}_\d{6}$")


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def should_keep(path: Path) -> bool:
    """Return True if the file should be preserved."""

    if path.name in KEEP_FILES:
        return True

    if path.suffix.lower() in KEEP_EXTENSIONS:
        return True

    if any(pattern.match(path.name) for pattern in SUBJECTIVE_INPUT_PATTERNS):
        return True

    return False


def clean_recording_folder(folder: Path):
    """
    Clean a recording folder, whether it's still flat or has
    been reorganized into subfolders (Raw_Eye_Videos,
    Cropped_Eye_Videos, Eye_CSVs, Raw_Labscribe, Labscribe_CSVs,
    Unity, Visualization, ...).

    Recurses into every subfolder:
      - Files that should be kept are moved back up to the
        recording folder root (a no-op if already there).
      - Every other file is deleted.
    Afterwards, every subdirectory of the recording folder is
    removed, leaving a flat folder containing only the
    preserved files.
    """

    print(f"\nCleaning: {folder}")

    deleted_files = 0
    moved_files = 0

    # ------------------------------------------------------------
    # Pass 1: walk every file at any depth, keep-or-delete it.
    # ------------------------------------------------------------

    for item in sorted(folder.rglob("*")):

        if not item.is_file():
            continue

        if should_keep(item):

            target = folder / item.name

            if item == target:
                continue

            if target.exists():
                print(
                    f"Warning: {target.name} already exists at the "
                    f"recording folder root; deleting duplicate "
                    f"{item.relative_to(folder)} instead of moving it."
                )
                try:
                    item.unlink()
                    deleted_files += 1
                except Exception as e:
                    print(f"Could not delete file {item}: {e}")
                continue

            try:
                shutil.move(str(item), str(target))
                moved_files += 1
                print(
                    f"Moved kept file to root: "
                    f"{item.relative_to(folder)} -> {item.name}"
                )
            except Exception as e:
                print(f"Could not move file {item}: {e}")

            continue

        try:
            item.unlink()
            deleted_files += 1
            print(f"Deleted file: {item.relative_to(folder)}")

        except Exception as e:
            print(f"Could not delete file {item}: {e}")

    # ------------------------------------------------------------
    # Pass 2: remove all subdirectories now that any files worth
    # keeping have been pulled up to the root.
    # ------------------------------------------------------------

    deleted_dirs = 0

    for item in sorted(folder.iterdir()):

        if item.is_dir():

            try:
                shutil.rmtree(item)
                deleted_dirs += 1
                print(f"Deleted folder: {item.name}")

            except Exception as e:
                print(f"Could not delete folder {item}: {e}")

    return deleted_files, deleted_dirs, moved_files


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    root = Tk()
    root.withdraw()

    selected = filedialog.askdirectory(
        title="Select Patient folder or recording folder"
    )

    if not selected:
        print("No folder selected.")
        return

    selected = Path(selected)

    # Look for recording folders inside the selected directory
    recording_folders = sorted(
        d for d in selected.iterdir()
        if d.is_dir() and PATIENT_FOLDER_PATTERN.fullmatch(d.name)
    )

    total_files = 0
    total_dirs = 0
    total_moved = 0

    if recording_folders:

        print(f"\nDetected Patient superfolder: {selected.name}")
        print(f"Found {len(recording_folders)} recording folder(s).")

        for folder in recording_folders:
            files, dirs, moved = clean_recording_folder(folder)
            total_files += files
            total_dirs += dirs
            total_moved += moved

    else:

        if not PATIENT_FOLDER_PATTERN.fullmatch(selected.name):
            print(
                "\nNo recording subfolders were found.\n"
                "Treating the selected folder as an individual recording."
            )
        else:
            print(f"\nDetected recording folder: {selected.name}")

        files, dirs, moved = clean_recording_folder(selected)
        total_files += files
        total_dirs += dirs
        total_moved += moved

    print("\n--------------------------------")
    print("Cleanup complete")
    print(f"Files deleted   : {total_files}")
    print(f"Files kept/moved: {total_moved}")
    print(f"Folders deleted : {total_dirs}")
    print("--------------------------------")

    messagebox.showinfo(
        "Cleanup Complete",
        f"Deleted {total_files} file(s)\n"
        f"Flattened {total_moved} kept file(s)\n"
        f"Deleted {total_dirs} folder(s)"
    )


if __name__ == "__main__":
    main()