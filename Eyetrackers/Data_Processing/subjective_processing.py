"""
subjective_processing.py

Scores Simulator Sickness Questionnaire (SSQ) and Motion Sickness
Susceptibility Questionnaire (MSSQ) CSV exports found in a LabScribe
session folder, using the scoring methods from the source papers:

  - Kennedy, R. S., Lane, N. E., Berbaum, K. S., & Lilienthal, M. G. (1993).
    Simulator Sickness Questionnaire: An Enhanced Method for Quantifying
    Simulator Sickness. The International Journal of Aviation Psychology,
    3(3), 203-220.

  - Golding, J. F. (1998). Motion sickness susceptibility questionnaire
    revised and its relationship to other forms of sickness. Brain Research
    Bulletin, 47(5), 507-516.

Figure rendering lives in subjective_visualization.py, not here -- this
module only scores questionnaires and writes csvs. That mirrors the rest
of the pipeline's separation of data processing from visualization
(cf. ecg.py / balance.py / summary.py -> visualization.py). This module
imports subjective_visualization and calls into it after scoring each
session.

Pipeline integration
---------------------
This module is designed to run early in the LabScribe pipeline -- before
reorganization.py touches a session folder -- so it can find the raw
MSSQ/SSQ CSV exports in their original location:

    LabScribe CSV
        |
        v
    io.py -> ecg.py / balance.py -> summary.py -> visualization.py
        |
        v
    subjective_processing.py -> subjective_visualization.py   <-- (this module + its figure renderer)
        |
        v
    reorganization.py

`process_session(session_dir)` is the integration entry point called from
main.py for each session. It has NO tolerance for missing questionnaire
data: if the SSQ csv can't be found directly inside `session_dir`, or the
MSSQ csv can't be found directly inside `session_dir`'s parent folder (the
super-folder that holds the patient session subfolders -- the MSSQ is
collected once per super-folder, not per session), it prints a console
message describing what's missing and raises
MissingQuestionnaireFileError, which the pipeline stops the entire run
for (not just this session), rather than silently continuing.

Because the MSSQ is shared by every session under the same super-folder,
`process_session()` scores it fresh each time (so each session's figures
always reflect it) but only ever writes `<mssq_stem>_scored.csv` into
that shared super-folder -- never into the individual session folder.
Every session for the same patient writes to that same path, so exactly
one MSSQ scored csv ends up on disk per patient rather than one per
session.

Standalone usage
-----------------
    python -m analysis.subjective_processing <path>

`<path>` may be either:
  - a super-folder containing multiple Patient_YYYYMMDD_HHMMSS session
    subfolders (batch mode), or
  - a single Patient_YYYYMMDD_HHMMSS session folder (individual mode).

The script auto-detects which one it was given, exactly like the
"batch or individual" choice at the start of the main pipeline.

Outputs
-------
Written directly into the session subfolder:
    <ssq_stem>_scored.csv
    <Patient_prefix>[_<iwxdata tag>]_Percentile_Curve.png   (MSSQ Fig. 1, via subjective_visualization.py)
    <Patient_prefix>[_<iwxdata tag>]_Total_Severity.png     (SSQ Fig. 1, via subjective_visualization.py)
    <Patient_prefix>[_<iwxdata tag>]_Hop_Count.png           (SSQ Fig. 3, via subjective_visualization.py)

Written into the super-folder above the session subfolders (shared
across all of a patient's sessions, written once):
    <mssq_stem>_scored.csv

If a ".iwxdata" raw-data file is present in the session folder (matching
the same naming convention as the SSQ/MSSQ CSVs, e.g.
"Patient_1_0.0.iwxdata" / "Patient_1_0_0_SSQ.csv"), its "N.N" identifier
is inserted into the image filenames so images from different sessions of
the same patient don't look identical at a glance even if later gathered
into one place -- e.g. "Patient_1_0.0_Total_Severity.png". If no
.iwxdata file is found, the tag is simply omitted and a note is printed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from pathlib import Path
from statistics import mean

try:
    # Reuse the pipeline's own session-folder-name parser when available
    # (this module lives in the same package as main.py in production).
    from .timestamps import parse_patient_folder_datetime
except ImportError:  # pragma: no cover - allows standalone script use
    parse_patient_folder_datetime = None

try:
    # Figure rendering is a separate module -- see the module docstring.
    from . import subjective_visualization as viz
except ImportError:  # pragma: no cover - allows standalone script use
    import subjective_visualization as viz


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MissingQuestionnaireFileError(RuntimeError):
    """Raised when a session folder is missing its SSQ and/or MSSQ csv.

    The pipeline treats this as fatal: main.py stops the whole run rather
    than skipping the affected session, since subjective_processing runs
    before reorganization.py and downstream steps assume it succeeded.
    """


# ---------------------------------------------------------------------------
# SSQ (Kennedy, Lane, Berbaum, & Lilienthal, 1993) scoring definitions
# ---------------------------------------------------------------------------

# Table 4: unit weights of each symptom onto the three SSQ subscales --
# Nausea (N), Oculomotor (O), Disorientation (D).
SSQ_WEIGHTS = {
    "general discomfort":       {"N": 1, "O": 1},
    "fatigue":                  {"O": 1},
    "headache":                 {"O": 1},
    "eyestrain":                {"O": 1},
    "difficulty focusing":      {"O": 1, "D": 1},
    "increased salivation":     {"N": 1},
    "sweating":                 {"N": 1},
    "nausea":                   {"N": 1, "D": 1},
    "difficulty concentrating": {"N": 1, "O": 1},
    "fullness of head":         {"D": 1},
    "blurred vision":           {"O": 1, "D": 1},
    "dizzy (eyes open)":        {"D": 1},
    "dizzy (eyes closed)":      {"D": 1},
    "vertigo":                  {"D": 1},
    "stomach awareness":        {"N": 1},
    "burping":                  {"N": 1},
}

# Conversion constants from raw (unit-weighted) subscale sums to scaled
# scores, and for the Total Severity (TS) score (Table 4 / p. 212).
SSQ_N_CONST = 9.54
SSQ_O_CONST = 7.58
SSQ_D_CONST = 13.92
SSQ_TS_CONST = 3.74

# Calibration-sample means, printed alongside each session's own scores
# for context (Kennedy et al., 1993, Table 5).
SSQ_CALIBRATION_MEANS = {"N": 7.7, "O": 10.6, "D": 6.4, "TS": 9.8}
SSQ_CALIBRATION_SD = 15.0


def _normalize(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def score_ssq_file(path):
    """Read one SSQ csv and return an ordered dict of per-trial scores."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        trial_cols = [c for c in fieldnames if c.strip().lower().startswith("trial_response")]
        if not trial_cols:
            raise ValueError(f"No 'Trial_Response_*' columns found in {path}")
        trial_cols = sorted(trial_cols, key=_trial_number)

        symptom_col = next((c for c in fieldnames if c.strip().lower() == "symptom"), None)
        if symptom_col is None:
            raise ValueError(f"No 'Symptom' column found in {path}")

        raw_sums = {trial: {"N": 0.0, "O": 0.0, "D": 0.0} for trial in trial_cols}
        unmatched_symptoms = []

        for row in reader:
            symptom_raw = row.get(symptom_col, "")
            key = _normalize(symptom_raw)
            weights = SSQ_WEIGHTS.get(key)
            if weights is None:
                if symptom_raw.strip():
                    unmatched_symptoms.append(symptom_raw)
                continue
            for trial in trial_cols:
                val = row.get(trial, "")
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    val = 0.0
                for subscale, w in weights.items():
                    raw_sums[trial][subscale] += w * val

    if unmatched_symptoms:
        print(f"  [warning] Unrecognized symptom rows in {os.path.basename(path)}: "
              f"{sorted(set(unmatched_symptoms))} (ignored in scoring)")

    results = {}
    for trial in trial_cols:
        raw = raw_sums[trial]
        results[trial] = {
            "N_raw": raw["N"], "O_raw": raw["O"], "D_raw": raw["D"],
            "N": round(raw["N"] * SSQ_N_CONST, 2),
            "O": round(raw["O"] * SSQ_O_CONST, 2),
            "D": round(raw["D"] * SSQ_D_CONST, 2),
            "TS": round((raw["N"] + raw["O"] + raw["D"]) * SSQ_TS_CONST, 2),
        }
    return results


def _trial_number(colname):
    m = re.search(r"(\d+)\s*$", colname)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# MSSQ (Golding, 1998) scoring definitions
# ---------------------------------------------------------------------------

TRANSPORT_TYPES = [
    "Car", "Buses/Coaches", "Trains", "Aircraft", "Small_Boats",
    "Ships", "Swings", "Roundabouts", "Big_Dippers",
]

MSSQ_SCALE_FACTOR = 2.64  # simplified-score -> original-score conversion (Golding, 1998, Fig. 2)
MSSQ_NUM_TYPES = 9

# Approximate percentile curve digitized from Figure 1 of Golding (1998).
# These points are read off the published cumulative-distribution plot
# (Golding never published an exact lookup table), so treat the percentile
# as an APPROXIMATION for descriptive/contextual purposes only.
MSSQ_PERCENTILE_CURVE = [
    (0, 0), (10, 8), (20, 15), (30, 22), (40, 30), (50, 38),
    (60, 48), (70, 62), (80, 80), (90, 100), (95, 130), (99, 190), (100, 200),
]


def _mssq_percentile(score):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    pts = MSSQ_PERCENTILE_CURVE
    if score <= pts[0][1]:
        return pts[0][0]
    if score >= pts[-1][1]:
        return pts[-1][0]
    for (p_lo, s_lo), (p_hi, s_hi) in zip(pts, pts[1:]):
        if s_lo <= score <= s_hi:
            if s_hi == s_lo:
                return p_lo
            frac = (score - s_lo) / (s_hi - s_lo)
            return round(p_lo + frac * (p_hi - p_lo), 1)
    return None


def _section_score(row, suffix):
    """Compute MSSQA or MSSQB (suffix='Child' or '10') for one respondent row."""
    total = 0.0
    n_types = 0
    for t in TRANSPORT_TYPES:
        col = f"{t}_{suffix}"
        if col not in row:
            continue
        raw = (row[col] or "").strip()
        if raw == "" or raw.lower() == "t":
            continue  # blank / not applicable -- excluded from numerator and denominator
        try:
            val = float(raw)
        except ValueError:
            continue
        total += val
        n_types += 1

    if n_types == 0:
        return None  # division by zero -- cannot estimate (Golding, 1998, Appendix)
    return MSSQ_SCALE_FACTOR * total * MSSQ_NUM_TYPES / n_types


def score_mssq_file(path):
    """Read one MSSQ csv (one row per respondent) and return a list of
    per-row score dicts."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows_out = []
        for i, row in enumerate(reader, start=1):
            mssqa = _section_score(row, "Child")
            mssqb = _section_score(row, "10")
            raw_total = None if (mssqa is None and mssqb is None) else (mssqa or 0) + (mssqb or 0)
            rows_out.append({
                "row": i,
                "Age": row.get("Age"),
                "Gender": row.get("Gender"),
                "MSSQA_child": None if mssqa is None else round(mssqa, 2),
                "MSSQB_adult": None if mssqb is None else round(mssqb, 2),
                "MSSQ_raw": None if raw_total is None else round(raw_total, 2),
                "MSSQ_approx_percentile": _mssq_percentile(raw_total) if raw_total is not None else None,
            })
    return rows_out


# ---------------------------------------------------------------------------
# Scored CSV output
# ---------------------------------------------------------------------------

def write_ssq_output(path, results, out_dir):
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = Path(out_dir) / f"{base}_scored.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "N_raw", "O_raw", "D_raw", "N", "O", "D", "TS"])
        for trial, r in results.items():
            writer.writerow([trial, r["N_raw"], r["O_raw"], r["D_raw"],
                              r["N"], r["O"], r["D"], r["TS"]])
    return out_path


def write_mssq_output(path, rows_out, out_dir):
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = Path(out_dir) / f"{base}_scored.csv"
    fieldnames = ["row", "Age", "Gender", "MSSQA_child", "MSSQB_adult",
                  "MSSQ_raw", "MSSQ_approx_percentile"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    return out_path


def print_ssq_summary(path, results):
    print(f"\nSSQ file: {os.path.basename(path)}")
    print(f"  {'Trial':<18}{'N':>8}{'O':>8}{'D':>8}{'TS':>8}")
    for trial, r in results.items():
        print(f"  {trial:<18}{r['N']:>8}{r['O']:>8}{r['D']:>8}{r['TS']:>8}")
    if results:
        print(f"  {'Mean across trials':<18}"
              f"{mean(r['N'] for r in results.values()):>8.2f}"
              f"{mean(r['O'] for r in results.values()):>8.2f}"
              f"{mean(r['D'] for r in results.values()):>8.2f}"
              f"{mean(r['TS'] for r in results.values()):>8.2f}")
    print(f"  (Calibration-sample means [Kennedy et al., 1993, Table 5]: "
          f"N={SSQ_CALIBRATION_MEANS['N']}, O={SSQ_CALIBRATION_MEANS['O']}, "
          f"D={SSQ_CALIBRATION_MEANS['D']}, TS={SSQ_CALIBRATION_MEANS['TS']}, SD={SSQ_CALIBRATION_SD})")


def print_mssq_summary(path, rows_out):
    print(f"\nMSSQ file: {os.path.basename(path)}")
    for r in rows_out:
        print(f"  Row {r['row']} (Age={r['Age']}, Gender={r['Gender']}): "
              f"MSSQA(child)={r['MSSQA_child']}  MSSQB(adult)={r['MSSQB_adult']}  "
              f"MSSQ_raw={r['MSSQ_raw']}  ~percentile={r['MSSQ_approx_percentile']}")
    print("  (Percentile is an approximation digitized from Golding (1998) Figure 1; "
          "treat as indicative only.)")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _match_csvs(session_dir, needle, exclude=None):
    matches = sorted(
        f for f in glob.glob(os.path.join(str(session_dir), "*.csv"))
        if needle in os.path.basename(f).lower()
        and (exclude is None or exclude not in os.path.basename(f).lower())
    )
    return matches


def discover_files(session_dir):
    """Return (ssq_files, mssq_files) without raising -- a non-fatal
    lookup used by resolve_session_dirs() for batch/individual
    auto-detection. SSQ csvs are found directly inside `session_dir`.
    MSSQ csvs are found directly inside `session_dir`'s parent folder --
    the super-folder that holds the patient session subfolders -- since
    the MSSQ is collected once per super-folder rather than once per
    session. See find_ssq_file()/find_mssq_file() for the fatal,
    single-file lookups used by process_session()."""
    session_dir = Path(session_dir)
    ssq_files = _match_csvs(session_dir, "ssq", exclude="mssq")
    mssq_files = _match_csvs(session_dir.parent, "mssq")
    return ssq_files, mssq_files


def find_ssq_file(session_dir):
    """Locate exactly one SSQ csv directly inside `session_dir`. Prints a
    console message and raises MissingQuestionnaireFileError if none is
    found -- this is the hard stop condition required by the pipeline
    (there are no fallback checks upstream)."""
    session_dir = Path(session_dir)
    ssq_files, _ = discover_files(session_dir)

    if not ssq_files:
        print(
            f"\nStopping: SSQ csv file not found directly inside session "
            f"folder:\n  {session_dir}\n"
            "Every patient session folder must contain a '*_SSQ.csv' "
            "file before the pipeline can continue."
        )
        raise MissingQuestionnaireFileError(f"Missing SSQ csv file in {session_dir}")

    if len(ssq_files) > 1:
        print(f"  Note: multiple SSQ csv files found in {session_dir}; using {ssq_files[0]}.")

    return Path(ssq_files[0])


def find_mssq_file(super_folder):
    """Locate exactly one MSSQ csv directly inside `super_folder` -- the
    folder holding the patient's session subfolders, one level above any
    individual session_dir. Prints a console message and raises
    MissingQuestionnaireFileError if none is found."""
    super_folder = Path(super_folder)
    mssq_files = _match_csvs(super_folder, "mssq")

    if not mssq_files:
        print(
            f"\nStopping: MSSQ csv file not found directly inside:\n"
            f"  {super_folder}\n"
            "The folder containing the patient's session subfolders must "
            "contain a '*_MSSQ.csv' file before the pipeline can continue."
        )
        raise MissingQuestionnaireFileError(f"Missing MSSQ csv file in {super_folder}")

    if len(mssq_files) > 1:
        print(f"  Note: multiple MSSQ csv files found in {super_folder}; using {mssq_files[0]}.")

    return Path(mssq_files[0])


# ---------------------------------------------------------------------------
# Session/patient identification (used both for csv naming here and for
# image naming over in subjective_visualization.py)
# ---------------------------------------------------------------------------

def _extract_patient_prefix(filename):
    """Pull the 'Patient_N' prefix off a questionnaire csv filename."""
    m = re.match(r"(Patient_\d+)", os.path.basename(filename), re.IGNORECASE)
    return m.group(1) if m else Path(filename).stem


def find_iwxdata_tag(session_dir):
    """Look for a '*.iwxdata' raw-data file directly inside session_dir and
    pull its trailing 'N.N' session identifier (matching the same
    convention used by the SSQ/MSSQ csv filenames, e.g.
    "Patient_1_0.0.iwxdata" <-> "Patient_1_0_0_SSQ.csv"). Returns None
    (with a console note) if no .iwxdata file is present or no identifier
    could be parsed from its name."""
    matches = sorted(glob.glob(os.path.join(str(session_dir), "*.iwxdata")))
    if not matches:
        print(f"  Note: no .iwxdata file found in {session_dir}; "
              "output image filenames will not include a session tag.")
        return None

    if len(matches) > 1:
        print(f"  Note: multiple .iwxdata files found in {session_dir}; using {matches[0]}.")

    stem = Path(matches[0]).stem
    m = re.search(r"(\d+)[._](\d+)$", stem)
    if not m:
        print(f"  Note: could not parse a session identifier from {matches[0]}; "
              "output image filenames will not include a session tag.")
        return None

    return f"{m.group(1)}.{m.group(2)}"


# ---------------------------------------------------------------------------
# Session processing (main pipeline integration point)
# ---------------------------------------------------------------------------

def process_session(session_dir):
    """
    Process one session folder's SSQ/MSSQ questionnaires: score them,
    write the scored CSVs, and (via subjective_visualization.py) render
    the reference figures.

    This is the function main.py calls for each session, before
    reorganization.reorganize_session(). Raises
    MissingQuestionnaireFileError (after printing a console message) if
    either csv can't be found -- callers should treat that as fatal and
    stop the whole run.

    The SSQ scored csv is written into `session_dir`, same as before. The
    MSSQ scored csv is written into `session_dir.parent` instead -- the
    super-folder shared by every session for this patient -- since the
    MSSQ itself lives (and is scored) there rather than per-session. Every
    session writes to that same shared path, so only one MSSQ scored csv
    ends up on disk per patient no matter how many sessions are processed.
    """
    session_dir = Path(session_dir)
    super_folder = session_dir.parent

    print(f"Scoring subjective questionnaires in {session_dir}...")

    ssq_path = find_ssq_file(session_dir)
    mssq_path = find_mssq_file(super_folder)

    ssq_results = score_ssq_file(ssq_path)
    mssq_rows = score_mssq_file(mssq_path)

    print_ssq_summary(ssq_path, ssq_results)
    print_mssq_summary(mssq_path, mssq_rows)

    ssq_out = write_ssq_output(ssq_path, ssq_results, session_dir)
    mssq_out = write_mssq_output(mssq_path, mssq_rows, super_folder)
    print(f"  -> {ssq_out}")
    print(f"  -> {mssq_out} (shared across this patient's sessions)")

    # ----------------------------------------------------------------
    # Figures -- handed off to subjective_visualization.py
    # ----------------------------------------------------------------

    patient_prefix = _extract_patient_prefix(ssq_path.name)
    tag = find_iwxdata_tag(session_dir)

    figures = viz.render_session_figures(
        session_dir, patient_prefix, tag, ssq_results, mssq_rows
    )

    print("  Figures written:")
    for p in figures.values():
        print(f"    -> {p}")

    return {
        "ssq_results": ssq_results,
        "mssq_rows": mssq_rows,
        "ssq_scored_csv": ssq_out,
        "mssq_scored_csv": mssq_out,
        "figures": figures,
    }


# ---------------------------------------------------------------------------
# Session/super-folder discovery (for standalone CLI use, mirroring the
# batch-vs-individual choice at the start of the main pipeline)
# ---------------------------------------------------------------------------

_SESSION_FOLDER_PATTERN = re.compile(r"^Patient_\d{8}_\d{6}$")


def _is_session_folder(path):
    if not path.is_dir():
        return False
    if parse_patient_folder_datetime is not None:
        try:
            parse_patient_folder_datetime(path.name)
            return True
        except ValueError:
            return False
    return bool(_SESSION_FOLDER_PATTERN.match(path.name))


def resolve_session_dirs(input_dir):
    """Given either a super-folder or a single session folder, return the
    list of session directories to process -- the same choice main.py asks
    for up front ("batch" vs. "individual")."""
    input_dir = Path(input_dir)

    ssq_files, mssq_files = discover_files(input_dir)
    if ssq_files or mssq_files:
        # Questionnaire csvs live directly here -- this IS a session folder.
        return [input_dir]

    session_dirs = [p for p in sorted(input_dir.iterdir()) if _is_session_folder(p)]
    if session_dirs:
        return session_dirs

    # Fall back to any subdirectories at all, in case folder names don't
    # match the standardized pattern.
    subdirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if subdirs:
        print(f"  Note: no Patient_YYYYMMDD_HHMMSS subfolders matched in {input_dir}; "
              f"treating all {len(subdirs)} subfolder(s) as sessions.")
        return subdirs

    return [input_dir]


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score SSQ/MSSQ questionnaires and render reference figures "
                    "for one session folder or a super-folder of session folders."
    )
    parser.add_argument(
        "path", nargs="?", default=".",
        help="Session folder (contains the SSQ/MSSQ csvs directly) or a "
             "super-folder containing Patient_YYYYMMDD_HHMMSS session subfolders.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    session_dirs = resolve_session_dirs(root)

    for index, session_dir in enumerate(session_dirs, start=1):
        print(f"\n=== Session {index}/{len(session_dirs)}: {session_dir.name} ===")
        try:
            process_session(session_dir)
        except MissingQuestionnaireFileError:
            # Message already printed inside find_ssq_file()/find_mssq_file().
            sys.exit(1)


if __name__ == "__main__":
    main()