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

from Eyetrackers.Labview.labscribe_io import (

    load_labscribe_csv,

    prepare_analysis_dataframe,

    get_ecg_signal,

    get_balance_data,

    export_dataframe,

)


from Eyetrackers.Labview.timestamps import (

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



# ============================================================
# Tkinter User Interface
# ============================================================

def get_user_config() -> AnalysisConfig | None:
    """
    Collect analysis settings using Tkinter.
    """

    root = tk.Tk()

    root.withdraw()



    # --------------------------------------------------------
    # Select CSV
    # --------------------------------------------------------

    csv_file = filedialog.askopenfilename(

        title="Select LabScribe CSV",

        filetypes=[
            (
                "CSV Files",
                "*.csv",
            )
        ],
    )


    if not csv_file:

        return None



    # --------------------------------------------------------
    # Recording date/time (from the standardized parent
    # folder name: Patient_YearMonthDay_HourMinuteSecond)
    # --------------------------------------------------------

    try:

        folder_datetime = parse_patient_folder_datetime(
            Path(csv_file).parent.name
        )

    except ValueError as error:

        messagebox.showerror(

            "Invalid Patient Folder",

            "Could not read the recording date/time from the "
            "parent folder name.\n\n"
            f"{error}",

        )

        return None



    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_directory = Path(csv_file).parent

    # --------------------------------------------------------
    # Video option
    # --------------------------------------------------------

    generate_video = messagebox.askyesno(

        "Generate Video",

        "Generate balance board video?",

    )



    root.destroy()



    return AnalysisConfig(

        input_csv=Path(csv_file),

        output_directory=output_directory,

        folder_datetime=folder_datetime,

        generate_video=generate_video,

    )

# ============================================================
# Main Entry Point
# ============================================================

def main():
    """
    Application entry point.
    """

    config = get_user_config()


    if config is None:

        print(
            "Analysis cancelled."
        )

        return



    try:

        run_analysis(
            config
        )


    except Exception as error:

        import traceback


        traceback.print_exc()


        print(
            f"\nAnalysis failed: {error}"
        )



if __name__ == "__main__":

    main()