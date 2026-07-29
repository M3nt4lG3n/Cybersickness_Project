"""
main.py

Main execution pipeline for LabScribe
cybersickness analysis.

Pipeline:

    LabScribe CSV
        |
        v
    io.py
        |
        +----------------+
        |                |
        v                v
     ecg.py          balance.py
        |                |
        +----------------+
                 |
                 v
             summary.py
                 |
                 v
          visualization.py


Outputs:

    *_analysis.csv

    *_beats.csv

    *_summary.csv

    *_ecg.png

    *_balance.mp4


Author:
    Brian Bizon / OpenAI
"""

from __future__ import annotations


from dataclasses import dataclass, field

from datetime import datetime

from pathlib import Path


import pandas as pd

import tkinter as tk

from tkinter import (
    filedialog,
    messagebox,
)



# ============================================================
# Internal Imports
# ============================================================

from .labscribe_io import (

    convert_labview_export_to_csv,

    load_labscribe_csv,

    prepare_analysis_dataframe,

    get_ecg_signal,

    get_balance_data,

    export_dataframe,

)


from .timestamps import (

    parse_patient_folder_datetime,

)


from .ecg import analyze_ecg, plot_ecg


from .balance import analyze_balance


from .summary import (

    calculate_combined_summary,

    export_final_summary,

)


from .visualization import (

    render_balance_video,

    VisualizationConfig,

)


from .unity_visualization import (

    load_unity_csv,

    render_unity_video,

    UnityVisualizationConfig,

)


from . import merger



# ============================================================
# Configuration
# ============================================================

@dataclass(slots=True)
class AnalysisConfig:
    """
    User configurable analysis settings.
    """

    input_csv: Path

    output_directory: Path

    folder_datetime: datetime

    generate_video: bool = True

    video_config: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )

    # Unity biometrics CSV (PosX/Y/Z, QuatX/Y/Z/W, ...).
    # Optional: if not provided, the position/rotation video
    # is skipped regardless of generate_unity_video.
    unity_csv: Path | None = None

    generate_unity_video: bool = True

    unity_video_config: UnityVisualizationConfig = field(
        default_factory=UnityVisualizationConfig
    )

    run_merge: bool = True



# ============================================================
# Output Paths
# ============================================================

def create_output_paths(
    config: AnalysisConfig,
) -> dict[str, Path]:
    """
    Create output file locations.
    """

    config.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    stem = (
        config.input_csv
        .stem
    )


    return {

        "timestamped":
            config.output_directory
            /
            f"{stem}_timestamped.csv",


        "analysis":
            config.output_directory
            /
            f"{stem}_analysis.csv",


        "beats":
            config.output_directory
            /
            f"{stem}_beats.csv",


        "ecg_png":
            config.output_directory
            /
            f"{stem}_ecg.png",


        "summary":
            config.output_directory
            /
            f"{stem}_summary.csv",


        "video":
            config.output_directory
            /
            f"{stem}_balance.mp4",


        "unity_video":
            config.output_directory
            /
            f"{stem}_unity.mp4",
    }

# ============================================================
# Balance Visualization Data Conversion
# ============================================================

def balance_result_to_dataframe(
    balance_result,
) -> pd.DataFrame:
    """
    Convert BalanceAnalysisResult into the dataframe
    format expected by visualization.py.

    Handles conversion from analysis objects
    into rendering format.
    """

    df = balance_result.dataframe.copy()


    rename_map = {

        "timestamp_ms":
            "UnixTime_ms",

        "tl":
            "TL",

        "tr":
            "TR",

        "bl":
            "BL",

        "br":
            "BR",

        "total_weight":
            "TotalWeight",

        "cop_x":
            "COP_X",

        "cop_y":
            "COP_Y",
    }


    for old, new in rename_map.items():

        if old in df.columns:

            df.rename(
                columns={
                    old: new
                },
                inplace=True,
            )


    return df



# ============================================================
# ECG Visualization Data Conversion
# ============================================================

def create_visualization_ecg_dataframe(
    ecg_result,
) -> pd.DataFrame:
    """
    Convert ECGAnalysisResult into visualization format.

    visualization.py expects:

        UnixTime_ms

        ECG

        optional HeartRate
    """

    df = pd.DataFrame()


    df["UnixTime_ms"] = (
        ecg_result.analysis_df[
            "UnixTime_ms"
        ]
    )


    df["ECG"] = (
        ecg_result.clean_signal
    )


    if (
        "HeartRate"
        in
        ecg_result.analysis_df.columns
    ):

        df["HeartRate"] = (
            ecg_result.analysis_df[
                "HeartRate"
            ]
        )


    return df



# ============================================================
# Main Analysis Pipeline
# ============================================================

def run_analysis(
    config: AnalysisConfig,
) -> None:
    """
    Execute complete LabScribe analysis pipeline.
    """

    print(
        "Loading LabScribe CSV..."
    )


    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    lab_data = load_labscribe_csv(

        config.input_csv

    )


    print(
        f"Loaded {len(lab_data.dataframe)} samples"
    )



    # --------------------------------------------------------
    # Prepare dataframe
    # --------------------------------------------------------

    df = prepare_analysis_dataframe(

        lab_data,

        config.folder_datetime,
    )



    paths = create_output_paths(
        config
    )



    # --------------------------------------------------------
    # Export timestamped CSV
    #
    # This is the full LabScribe dataframe (all original
    # columns) with the UnixTime_ms column added, based on
    # the recording date/time parsed from the standardized
    # Patient_YYYYMMDD_HHMMSS parent folder. Exported before
    # ECG/balance processing so it is always available even
    # if later stages fail.
    # --------------------------------------------------------

    print(
        "Exporting timestamped CSV..."
    )


    export_dataframe(

        df,

        paths["timestamped"],
    )



    # --------------------------------------------------------
    # ECG analysis
    # --------------------------------------------------------

    print(
        "Processing ECG..."
    )


    ecg_signal = get_ecg_signal(
        df
    )


    ecg_result = analyze_ecg(

        ecg_signal,

        df["UnixTime_ms"],
    )



    print(
        "ECG complete."
    )



    # --------------------------------------------------------
    # Balance analysis
    # --------------------------------------------------------

    print(
        "Processing balance..."
    )


    balance_df = get_balance_data(
        df
    )


    balance_result = analyze_balance(
        balance_df
    )



    print(
        "Balance complete."
    )



    # --------------------------------------------------------
    # Export analysis files
    # --------------------------------------------------------

    print(
        "Exporting CSV files..."
    )


    export_dataframe(

        ecg_result.analysis_df,

        paths["analysis"],
    )


    export_dataframe(

        ecg_result.beats_df,

        paths["beats"],
    )



    # --------------------------------------------------------
    # ECG plot
    # --------------------------------------------------------

    print(
        "Rendering ECG plot..."
    )


    plot_ecg(

        ecg_result.raw_signal,

        ecg_result.timestamps,

        ecg_result.beats_df,

        paths["ecg_png"],

        title=lab_data.metadata.file_name,
    )



    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    experiment_summary = (
        calculate_combined_summary(

            ecg_result,

            balance_result,

            metadata={

                "FileName":
                    lab_data.metadata.file_name,

                "SampleCount":
                    lab_data.metadata.sample_count,

            },

        )
    )


    export_final_summary(

        experiment_summary,

        paths["summary"],

    )



    print(
        "Summary exported."
    )



    # --------------------------------------------------------
    # Balance board video
    # --------------------------------------------------------

    if config.generate_video:

        generate_balance_video(

            config,

            ecg_result,

            balance_result,

            paths["video"],
        )

    else:

        print(
            "Video generation skipped."
        )


    # --------------------------------------------------------
    # Unity position/rotation video
    # --------------------------------------------------------

    if config.generate_unity_video and config.unity_csv is not None:

        generate_unity_position_video(

            config,

            paths["unity_video"],
        )

    elif config.generate_unity_video and config.unity_csv is None:

        print(
            "Unity video skipped (no Unity biometrics CSV "
            "provided)."
        )

    else:

        print(
            "Unity video generation skipped."
        )


    # --------------------------------------------------------
    # Merge with Unity biometrics + eye tracking data
    # --------------------------------------------------------

    if config.run_merge:

        print(
            "Merging with biometrics and eye-tracking data..."
        )

        merger.run_merge_from_analysis_paths(

            config.input_csv,

            config.output_directory,

            paths,
        )

        print(
            "Merge complete."
        )

    else:

        print(
            "Merge step skipped."
        )

# ============================================================
# Video Rendering
# ============================================================

def generate_balance_video(
    config: AnalysisConfig,
    ecg_result,
    balance_result,
    output_file: Path,
) -> None:
    """
    Generate synchronized ECG + balance video.
    """

    print(
        "Preparing video rendering..."
    )


    balance_df = balance_result_to_dataframe(
        balance_result
    )


    ecg_df = create_visualization_ecg_dataframe(
        ecg_result
    )


    render_balance_video(

        balance_df,

        ecg_df,

        str(output_file),

        config.video_config,

    )


    print(
        "Video complete."
    )


def generate_unity_position_video(
    config: AnalysisConfig,
    output_file: Path,
) -> None:
    """
    Generate the Unity position/rotation video: a sphere that
    translates and rotates through 3D space according to the
    recorded headset position and orientation.
    """

    print(
        "Loading Unity biometrics CSV..."
    )


    unity_df = load_unity_csv(
        config.unity_csv
    )


    print(
        f"Loaded {len(unity_df)} Unity samples."
    )


    print(
        "Rendering Unity position/rotation video..."
    )


    render_unity_video(

        unity_df,

        str(output_file),

        config.unity_video_config,

    )


    print(
        "Unity video complete."
    )



# ============================================================
# Session Discovery
#
# A "session folder" is a standardized Patient_YYYYMMDD_HHMMSS
# folder (see timestamps.parse_patient_folder_datetime). Batch
# mode processes every session subfolder inside a selected
# super-folder (e.g. "Patient_1"); individual mode processes
# exactly one session folder selected directly.
# ============================================================

def find_session_subfolders(super_folder: Path) -> list[Path]:
    """
    Find every immediate subfolder of `super_folder` whose name
    matches the standardized Patient_YYYYMMDD_HHMMSS pattern.
    """

    sessions = []

    for entry in sorted(super_folder.iterdir()):

        if not entry.is_dir():
            continue

        try:
            parse_patient_folder_datetime(entry.name)
        except ValueError:
            continue

        sessions.append(entry)

    return sessions


# ============================================================
# LabScribe .xls Discovery
#
# Each session folder is expected to contain a raw LabVIEW/
# LabScribe export (e.g. "Patient_1_0.1.xls"). If it can't be
# found, the user gets one chance to fix it and retry before
# the run is aborted.
# ============================================================

def find_xls_file(session_dir: Path) -> Path | None:
    """
    Find a .xls file directly inside `session_dir`.

    If more than one is found, the alphabetically-first match
    is used and a note is printed.
    """

    matches = sorted(
        entry
        for entry in session_dir.iterdir()
        if entry.is_file() and entry.suffix.lower() == ".xls"
    )

    if len(matches) > 1:

        print(
            f"  Note: multiple .xls files found in {session_dir}; "
            f"using {matches[0].name}."
        )

    return matches[0] if matches else None


def locate_xls_with_retry(
    session_dir: Path,
    max_retries: int = 1,
) -> Path | None:
    """
    Search `session_dir` for a .xls file, giving the user one
    retry (via a Tkinter Retry/Cancel prompt) if it isn't found
    the first time. Returns None - meaning the caller should
    stop - if it's still missing after the retry, or if the
    user cancels.
    """

    attempt = 0

    while True:

        xls_path = find_xls_file(session_dir)

        if xls_path is not None:
            return xls_path

        if attempt >= max_retries:

            print(
                f"Stopping: no .xls file found in {session_dir} "
                "after retry."
            )

            return None

        attempt += 1

        retry = messagebox.askretrycancel(

            "LabScribe File Not Found",

            f".xls file not found in ./{session_dir.name}. Make "
            "sure the file is named properly and present in the "
            "folder.",

        )

        if not retry:

            print(
                f"Stopping: .xls search cancelled for {session_dir}."
            )

            return None

        # Loop back and search again.


# ============================================================
# Unity Biometrics CSV Discovery
#
# Unlike the old flow (which asked the user to browse for it),
# unity_biometrics.csv is expected to already live directly in
# the session folder alongside the LabScribe export.
# ============================================================

def find_unity_csv(session_dir: Path) -> Path | None:
    """
    Look for unity_biometrics.csv directly inside `session_dir`.
    """

    candidate = session_dir / "unity_biometrics.csv"

    return candidate if candidate.exists() else None


# ============================================================
# Tkinter User Interface
# ============================================================

def ask_batch_or_individual(root: tk.Tk) -> str | None:
    """
    Ask whether to run a batch (multiple sessions) or an
    individual (single session) analysis.

    Returns "batch", "individual", or None if the dialog was
    closed without a choice.
    """

    choice = {"value": None}

    dialog = tk.Toplevel(root)
    dialog.title("Analysis Mode")
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(
        dialog,
        text=(
            "Run a batch analysis (a super-folder containing "
            "multiple\nPatient_YYYYMMDD_HHMMSS session folders) or "
            "an\nindividual analysis (a single session folder)?"
        ),
        padx=20,
        pady=20,
        justify="left",
    ).pack()

    button_frame = tk.Frame(dialog)
    button_frame.pack(pady=(0, 15))

    def pick(value):
        choice["value"] = value
        dialog.destroy()

    tk.Button(
        button_frame,
        text="Batch",
        width=12,
        command=lambda: pick("batch"),
    ).pack(side="left", padx=10)

    tk.Button(
        button_frame,
        text="Individual",
        width=12,
        command=lambda: pick("individual"),
    ).pack(side="left", padx=10)

    dialog.protocol("WM_DELETE_WINDOW", lambda: pick(None))

    root.wait_window(dialog)

    return choice["value"]


def get_run_options() -> tuple[bool, bool]:
    """
    Ask the balance-video and Unity-video yes/no questions.
    Reused as-is from the previous single-file flow.
    """

    generate_video = messagebox.askyesno(

        "Generate Video",

        "Generate balance board video?",

    )

    generate_unity_video = messagebox.askyesno(

        "Generate Unity Video",

        "Generate Unity position/rotation video?",

    )

    return generate_video, generate_unity_video


def build_analysis_config(
    session_dir: Path,
    xls_path: Path,
    generate_video: bool,
    generate_unity_video: bool,
) -> AnalysisConfig | None:
    """
    Convert a session folder's .xls export and assemble its
    AnalysisConfig. Returns None (after showing an error) if
    the folder name can't be parsed as a recording date/time.
    """

    # --------------------------------------------------------
    # Raw LabVIEW export conversion
    #
    # LabScribe exports are saved with a ".xls" extension but
    # are actually tab-delimited text files. Convert to a
    # proper CSV up front so the rest of the pipeline can keep
    # using its existing CSV-based architecture unchanged.
    # --------------------------------------------------------

    print(
        f"Converting raw LabVIEW export to CSV: {xls_path.name}"
    )

    csv_path = convert_labview_export_to_csv(
        xls_path
    )

    print(
        f"  Saved: {csv_path}"
    )

    # --------------------------------------------------------
    # Recording date/time (from the standardized session
    # folder name: Patient_YearMonthDay_HourMinuteSecond)
    # --------------------------------------------------------

    try:

        folder_datetime = parse_patient_folder_datetime(
            session_dir.name
        )

    except ValueError as error:

        messagebox.showerror(

            "Invalid Patient Folder",

            "Could not read the recording date/time from the "
            "session folder name.\n\n"
            f"{error}",

        )

        return None

    # --------------------------------------------------------
    # Unity biometrics CSV - auto-discovered rather than
    # browsed for. If missing, treat this session as if the
    # user had said "no" to the Unity video question.
    # --------------------------------------------------------

    unity_csv = None
    session_generate_unity_video = generate_unity_video

    if generate_unity_video:

        unity_csv = find_unity_csv(session_dir)

        if unity_csv is None:

            print(
                f"  unity_biometrics.csv not found in {session_dir}; "
                "skipping Unity video for this session."
            )

            session_generate_unity_video = False

    return AnalysisConfig(

        input_csv=csv_path,

        output_directory=session_dir,

        folder_datetime=folder_datetime,

        generate_video=generate_video,

        unity_csv=unity_csv,

        generate_unity_video=session_generate_unity_video,

    )


def get_batch_configs() -> list[AnalysisConfig]:
    """
    Run the full Tkinter selection flow and return one
    AnalysisConfig per session to process (one for individual
    mode, one per discovered session subfolder for batch mode).

    Returns an empty list if the user cancels at any point, or
    if a required file couldn't be found.
    """

    root = tk.Tk()

    root.withdraw()

    # --------------------------------------------------------
    # Batch vs individual
    # --------------------------------------------------------

    mode = ask_batch_or_individual(root)

    if mode is None:

        root.destroy()

        return []

    if mode == "batch":

        super_folder = filedialog.askdirectory(

            title="Select Patient Super-Folder (e.g. Patient_1)",

        )

        if not super_folder:

            root.destroy()

            return []

        session_dirs = find_session_subfolders(
            Path(super_folder)
        )

        if not session_dirs:

            messagebox.showerror(

                "No Session Folders Found",

                "No Patient_YYYYMMDD_HHMMSS session subfolders "
                f"were found in:\n{super_folder}",

            )

            root.destroy()

            return []

        if len(session_dirs) != 4:

            print(
                "Note: expected 4 session subfolders, found "
                f"{len(session_dirs)}. Continuing anyway."
            )

    else:

        session_folder = filedialog.askdirectory(

            title="Select Session Folder (e.g. Patient_20260727_121113)",

        )

        if not session_folder:

            root.destroy()

            return []

        session_dir = Path(session_folder)

        try:

            parse_patient_folder_datetime(session_dir.name)

        except ValueError as error:

            messagebox.showerror(

                "Invalid Session Folder",

                "Folder name doesn't match the expected "
                f"Patient_YYYYMMDD_HHMMSS format.\n\n{error}",

            )

            root.destroy()

            return []

        session_dirs = [session_dir]

    # --------------------------------------------------------
    # Balance video / Unity video questions (asked once, applied
    # to every session in this run)
    # --------------------------------------------------------

    generate_video, generate_unity_video = get_run_options()

    # --------------------------------------------------------
    # Locate each session's .xls export and build its config
    # --------------------------------------------------------

    configs: list[AnalysisConfig] = []

    for session_dir in session_dirs:

        xls_path = locate_xls_with_retry(session_dir)

        if xls_path is None:

            root.destroy()

            return []

        config = build_analysis_config(
            session_dir,
            xls_path,
            generate_video,
            generate_unity_video,
        )

        if config is None:

            root.destroy()

            return []

        configs.append(config)

    root.destroy()

    return configs


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Application entry point.
    """

    configs = get_batch_configs()


    if not configs:

        print(
            "Analysis cancelled."
        )

        return


    for index, config in enumerate(configs, start=1):

        print(
            f"\n=== Session {index}/{len(configs)}: "
            f"{config.output_directory.name} ==="
        )

        try:

            run_analysis(
                config
            )


        except Exception as error:

            import traceback


            traceback.print_exc()


            print(
                f"\nAnalysis failed for {config.output_directory.name}: "
                f"{error}"
            )



if __name__ == "__main__":

    main()