"""
merger.py

Combines the various CSV outputs produced by the LabScribe
cybersickness analysis pipeline (main.py) with the Unity
biometrics log and the eye-tracker (pupil + eye) recordings
into a single time-aligned dataset.

Merge rules
-----------
1. Most files are merged against each other using their
   UnixTime_ms column (exact match). This includes:

       - the labscribe *_timestamped.csv
       - the ECG *_analysis.csv (with beats folded in, see #2)
       - unity_biometrics.csv
       - the generated *_eye_readings.csv files (see #4)

   The leftmost column of the combined file is a generated
   UnixTime_ms column, not any one source file's own time
   column: it starts at the smallest UnixTime_ms found across
   every input file and increments by 1 for every row up to
   the largest. Every other file's data is then dropped into
   whichever row has a matching UnixTime_ms; rows with no
   match anywhere stay blank in that file's columns.

   Once the full grid is built, it's pruned down to:

       - only rows within unity_biometrics.csv's own
         UnixTime_ms range (its earliest and latest timestamp)
       - and, within that range, only rows that actually have
         at least one column of data populated (fully empty
         timestamps are dropped)

2. *_beats.csv is NOT timestamp-merged. It has one row per
   detected heartbeat, not one row per sample, so it has no
   UnixTime_ms of its own. Instead, each row of *_beats.csv is
   attached, in order, to the row of *_analysis.csv where the
   beat-flag column == 1. The Nth beat in *_beats.csv lines up
   with the Nth row flagged as a beat in *_analysis.csv.

3. *_summary.csv is never merged. It's an aggregate summary
   table, not a per-sample/per-event table, and is left
   untouched by this script.

4. left_pupil.csv / left_eye.csv and right_pupil.csv /
   right_eye.csv are merged first, separately, before joining
   the rest of the data:

       - the pupil file's frame number is matched against the
         eye file's row number (its position in the file)
       - the result is written out as left_eye_readings.csv /
         right_eye_readings.csv
       - CaptureTimestampMs (from the eye file, carried through
         the pupil+eye merge) is then used as that stream's
         timestamp when it gets folded into the combined
         UnixTime_ms dataset.

Expected input layout
----------------------
All of the following are expected in the same parent folder as
the user-selected LabScribe CSV:

    unity_biometrics.csv
    right_eye.csv
    right_pupil.csv
    left_eye.csv
    left_pupil.csv

Any of these that are missing are skipped with a warning rather
than raising, since not every recording session necessarily has
eye-tracking or biometrics data attached.

Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration / assumptions
#
# These column names reflect the LabScribe / Unity / eye-
# tracker export formats as described. If a given export uses
# different column names, update the constants here rather
# than the merge logic below.
# ============================================================

# Column in *_analysis.csv that marks a detected heartbeat with
# a 1 (0 otherwise).
BEAT_FLAG_COLUMN = "Beat"

# Column in *_pupil.csv identifying which video frame a pupil
# reading belongs to.
PUPIL_FRAME_COLUMN = "Frame"

# Column (in the eye files, carried through into the merged
# eye-reading files) holding the capture timestamp in
# milliseconds. Used to line the eye-tracking stream up
# against LabScribe's UnixTime_ms.
CAPTURE_TIMESTAMP_COLUMN = "CaptureTimestampMs"

# Fixed input file names expected alongside the user-selected
# LabScribe CSV.
UNITY_BIOMETRICS_FILENAME = "unity_biometrics.csv"
RIGHT_EYE_FILENAME = "right_eye.csv"
RIGHT_PUPIL_FILENAME = "right_pupil.csv"
LEFT_EYE_FILENAME = "left_eye.csv"
LEFT_PUPIL_FILENAME = "left_pupil.csv"

@dataclass(slots=True)
class MergeInputPaths:
    """
    Every path merger.py needs. `timestamped_csv`, `analysis_csv`
    and `beats_csv` are the pipeline outputs main.py already
    produced; the rest are expected to sit alongside the
    original user-selected LabScribe CSV, in `input_dir`.
    """

    input_dir: Path

    timestamped_csv: Path
    analysis_csv: Path
    beats_csv: Path

    unity_biometrics_csv: Path = None
    right_eye_csv: Path = None
    right_pupil_csv: Path = None
    left_eye_csv: Path = None
    left_pupil_csv: Path = None

    def __post_init__(self):

        if self.unity_biometrics_csv is None:
            self.unity_biometrics_csv = self.input_dir / UNITY_BIOMETRICS_FILENAME

        if self.right_eye_csv is None:
            self.right_eye_csv = self.input_dir / RIGHT_EYE_FILENAME

        if self.right_pupil_csv is None:
            self.right_pupil_csv = self.input_dir / RIGHT_PUPIL_FILENAME

        if self.left_eye_csv is None:
            self.left_eye_csv = self.input_dir / LEFT_EYE_FILENAME

        if self.left_pupil_csv is None:
            self.left_pupil_csv = self.input_dir / LEFT_PUPIL_FILENAME


# ============================================================
# Small helpers
# ============================================================

def _read_csv_if_exists(path: Path | None) -> pd.DataFrame | None:
    """
    Read a CSV if it exists, otherwise print a note and return
    None so callers can skip it cleanly.
    """

    if path is None or not Path(path).exists():
        print(f"  (skipping, not found: {path})")
        return None

    return pd.read_csv(path)


# ============================================================
# Rule #2: fold beats into the analysis dataframe
# ============================================================

def merge_beats_into_analysis(
    analysis_df: pd.DataFrame,
    beats_df: pd.DataFrame,
    flag_column: str = BEAT_FLAG_COLUMN,
) -> pd.DataFrame:
    """
    Attach each row of `beats_df`, in order, to the row of
    `analysis_df` where `flag_column` == 1.

    `beats_df` has one row per detected heartbeat and carries
    no UnixTime_ms of its own, so it can't be timestamp-merged
    like everything else. Instead the Nth beat lines up with
    the Nth sample flagged as a beat.
    """

    merged = analysis_df.copy()

    if flag_column not in merged.columns:
        raise ValueError(
            f"'{flag_column}' column not found in the analysis dataframe. "
            "Update BEAT_FLAG_COLUMN in merger.py if ecg.py names the "
            "beat-flag column something else."
        )

    beat_row_indices = merged.index[merged[flag_column] == 1].tolist()

    if len(beat_row_indices) != len(beats_df):
        print(
            "  Warning: found "
            f"{len(beat_row_indices)} rows flagged as beats in *_analysis.csv "
            f"but {len(beats_df)} rows in *_beats.csv. Merging as many as "
            "line up, in order, and leaving the rest blank."
        )

    beat_columns = [f"beat_{col}" for col in beats_df.columns]

    for col in beat_columns:
        merged[col] = pd.NA

    n = min(len(beat_row_indices), len(beats_df))

    for i in range(n):

        row_idx = beat_row_indices[i]

        for orig_col, new_col in zip(beats_df.columns, beat_columns):
            merged.at[row_idx, new_col] = beats_df.iloc[i][orig_col]

    return merged


# ============================================================
# Rule #4: pupil + eye merges
# ============================================================

def merge_pupil_and_eye(
    pupil_df: pd.DataFrame,
    eye_df: pd.DataFrame,
    frame_column: str = PUPIL_FRAME_COLUMN,
) -> pd.DataFrame:
    """
    Merge a pupil-tracking file with its matching eye-tracking
    file.

    The pupil file's frame number is matched against the eye
    file's row number (its position in the file) since the eye
    file has no frame-number column of its own.
    """

    if frame_column not in pupil_df.columns:
        raise ValueError(
            f"'{frame_column}' column not found in the pupil dataframe. "
            "Update PUPIL_FRAME_COLUMN in merger.py if the eye tracker "
            "export uses a different frame-number column."
        )

    eye = eye_df.copy().reset_index(drop=True)
    eye["_row_number"] = eye.index

    merged = pupil_df.merge(
        eye,
        left_on=frame_column,
        right_on="_row_number",
        how="inner",
        suffixes=("_pupil", "_eye"),
    )

    merged.drop(columns=["_row_number"], inplace=True)

    return merged


def build_eye_readings(
    paths: MergeInputPaths,
    output_directory: Path,
) -> dict[str, Path]:
    """
    Build left_eye_readings.csv and right_eye_readings.csv by
    merging each side's pupil file with its eye file (rule #4),
    writing them out to `output_directory`.

    Returns the resulting file paths keyed by "left" / "right";
    a side is omitted if its pupil or eye file is missing.
    """

    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}

    sides = {
        "left": (paths.left_pupil_csv, paths.left_eye_csv),
        "right": (paths.right_pupil_csv, paths.right_eye_csv),
    }

    for side, (pupil_path, eye_path) in sides.items():

        print(f"Building {side}_eye_readings.csv...")

        pupil_df = _read_csv_if_exists(pupil_path)
        eye_df = _read_csv_if_exists(eye_path)

        if pupil_df is None or eye_df is None:
            print(f"  Skipping {side} eye readings (missing input file).")
            continue

        merged = merge_pupil_and_eye(pupil_df, eye_df)

        out_path = output_directory / f"{side}_eye_readings.csv"
        merged.to_csv(out_path, index=False)

        results[side] = out_path

        print(f"  Wrote {out_path.name} ({len(merged)} rows)")

    return results


# ============================================================
# Rule #1: UnixTime_ms based merging
# ============================================================

def _prepare_for_timestamp_merge(
    df: pd.DataFrame,
    timestamp_column: str,
    label: str,
) -> pd.DataFrame:
    """
    Sort a dataframe by its timestamp column, rename that
    column to UnixTime_ms, and prefix every other column with
    `label__` so columns from different sources stay
    distinguishable once everything is joined together.
    """

    df = df.copy()

    if timestamp_column not in df.columns:
        raise ValueError(
            f"'{timestamp_column}' column not found for stream '{label}'."
        )

    if timestamp_column != "UnixTime_ms":
        df.rename(columns={timestamp_column: "UnixTime_ms"}, inplace=True)

    df["UnixTime_ms"] = pd.to_numeric(df["UnixTime_ms"], errors="coerce")
    df.dropna(subset=["UnixTime_ms"], inplace=True)
    df.sort_values("UnixTime_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)

    rename_map = {
        col: f"{label}__{col}"
        for col in df.columns
        if col != "UnixTime_ms"
    }
    df.rename(columns=rename_map, inplace=True)

    return df


def build_unixtime_grid(streams: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the master timeline: a single UnixTime_ms column
    starting at the smallest UnixTime_ms found across every
    stream and incrementing by 1 for every row up to the
    largest.

    This is the leftmost column of the combined output - it is
    generated, not taken from any one source file's own time
    column (e.g. labscribe's TimeOfDay).
    """

    start = min(int(df["UnixTime_ms"].min()) for df in streams.values())
    end = max(int(df["UnixTime_ms"].max()) for df in streams.values())

    return pd.DataFrame({"UnixTime_ms": range(start, end + 1)})


def merge_on_unixtime(
    streams: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Left-merge every stream onto a generated UnixTime_ms grid
    (see build_unixtime_grid), matching each source row into
    the grid row with the exact same UnixTime_ms. Grid rows
    with no matching source row stay blank for that source's
    columns.
    """

    combined = build_unixtime_grid(streams)

    for label, df in streams.items():

        df = df.copy()

        # Round to whole milliseconds so exact-match merging is
        # reliable even if a source stored fractional ms values.
        df["UnixTime_ms"] = df["UnixTime_ms"].round().astype("int64")

        # A stream shouldn't have two rows claiming the same
        # millisecond; if it does, keep the first and drop the
        # rest rather than duplicating grid rows.
        duplicate_count = df["UnixTime_ms"].duplicated().sum()

        if duplicate_count:
            print(
                f"  Warning: stream '{label}' has {duplicate_count} rows "
                "sharing a UnixTime_ms with another row in the same stream; "
                "keeping the first occurrence of each."
            )
            df = df.drop_duplicates(subset="UnixTime_ms", keep="first")

        combined = combined.merge(df, on="UnixTime_ms", how="left")

    return combined


def prune_combined(
    combined: pd.DataFrame,
    unity_min: int | None,
    unity_max: int | None,
) -> pd.DataFrame:
    """
    Trim the combined grid down to unity_biometrics.csv's own
    UnixTime_ms range, then drop any remaining row that has no
    data at all (every non-UnixTime_ms column is blank).
    """

    pruned = combined

    if unity_min is not None and unity_max is not None:

        pruned = pruned[
            (pruned["UnixTime_ms"] >= unity_min)
            & (pruned["UnixTime_ms"] <= unity_max)
        ]

    else:

        print(
            "  Warning: no unity_biometrics.csv timestamps available; "
            "skipping start/end pruning."
        )

    data_columns = [
        column
        for column in pruned.columns
        if column != "UnixTime_ms"
    ]

    pruned = pruned.dropna(subset=data_columns, how="all")

    pruned = pruned.reset_index(drop=True)

    return pruned


# ============================================================
# Orchestration
# ============================================================

def merge_all(
    paths: MergeInputPaths,
    output_directory: Path,
) -> Path:
    """
    Run the full merge pipeline and write out a single combined
    CSV alongside the two eye-reading CSVs.

    Returns the path to the combined output file.
    """

    print("Merging pipeline outputs...")

    # ---- Fold beats into the ECG analysis dataframe --------
    analysis_df = pd.read_csv(paths.analysis_csv)
    beats_df = pd.read_csv(paths.beats_csv)

    ecg_with_beats = merge_beats_into_analysis(analysis_df, beats_df)

    # ---- Base timeline: the full timestamped LabScribe export
    timestamped_df = pd.read_csv(paths.timestamped_csv)

    streams: dict[str, pd.DataFrame] = {
        "labscribe": _prepare_for_timestamp_merge(
            timestamped_df, "UnixTime_ms", "labscribe"
        ),
        "ecg": _prepare_for_timestamp_merge(
            ecg_with_beats, "UnixTime_ms", "ecg"
        ),
    }

    # ---- Unity biometrics ------------------------------------
    unity_df = _read_csv_if_exists(paths.unity_biometrics_csv)

    unity_min = None
    unity_max = None

    if unity_df is not None:
        streams["unity"] = _prepare_for_timestamp_merge(
            unity_df, "UnixTime_ms", "unity"
        )
        unity_min = int(streams["unity"]["UnixTime_ms"].min())
        unity_max = int(streams["unity"]["UnixTime_ms"].max())

    # ---- Eye tracking (pupil + eye, per side) -----------------
    eye_reading_paths = build_eye_readings(paths, output_directory)

    for side, path in eye_reading_paths.items():

        eye_df = pd.read_csv(path)

        if CAPTURE_TIMESTAMP_COLUMN not in eye_df.columns:
            print(
                f"  Warning: {path.name} has no "
                f"'{CAPTURE_TIMESTAMP_COLUMN}' column, so it can't be lined "
                "up with UnixTime_ms and will be left out of the combined file."
            )
            continue

        streams[f"{side}_eye"] = _prepare_for_timestamp_merge(
            eye_df, CAPTURE_TIMESTAMP_COLUMN, f"{side}_eye"
        )

    # ---- Build the 1ms grid and drop everything onto it -------
    # (*_summary.csv is intentionally never included here - it's
    # an aggregate table, not a per-sample one.)
    combined = merge_on_unixtime(streams)

    print(f"  Full grid: {len(combined)} rows before pruning")

    # ---- Prune to unity_biometrics' range, drop empty rows ----
    combined = prune_combined(combined, unity_min, unity_max)

    stem = paths.timestamped_csv.stem.replace("_timestamped", "")
    output_path = output_directory / f"{stem}_combined.csv"

    combined.to_csv(output_path, index=False)

    print(f"Combined dataset written to {output_path.name} ({len(combined)} rows)")

    return output_path


def run_merge_from_analysis_paths(
    input_csv: Path,
    output_directory: Path,
    paths: dict[str, Path],
) -> Path:
    """
    Convenience wrapper for main.py: build a MergeInputPaths from
    the `paths` dict main.py's create_output_paths() already
    produced (so we don't need to re-derive file names), then
    run the full merge.
    """

    merge_paths = MergeInputPaths(
        input_dir=input_csv.parent,
        timestamped_csv=paths["timestamped"],
        analysis_csv=paths["analysis"],
        beats_csv=paths["beats"],
    )

    return merge_all(merge_paths, output_directory)


# ============================================================
# Standalone entry point
#
# merger.py is meant to run as part of main.py's pipeline (see
# run_merge_from_analysis_paths, called from run_analysis), but
# can also be run on its own against an already-processed
# LabScribe folder.
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Merge LabScribe pipeline outputs with Unity biometrics and "
            "eye-tracker data into one combined, UnixTime_ms-aligned CSV."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the original LabScribe CSV (used to find the "
        "*_timestamped/_analysis/_beats files and the folder containing "
        "unity_biometrics.csv, left/right_eye.csv, left/right_pupil.csv).",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Folder containing main.py's pipeline outputs.",
    )

    args = parser.parse_args()

    stem = args.input_csv.stem

    derived_paths = {
        "timestamped": args.output_directory / f"{stem}_timestamped.csv",
        "analysis": args.output_directory / f"{stem}_analysis.csv",
        "beats": args.output_directory / f"{stem}_beats.csv",
    }

    run_merge_from_analysis_paths(
        args.input_csv,
        args.output_directory,
        derived_paths,
    )