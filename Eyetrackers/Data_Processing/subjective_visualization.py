"""
subjective_visualization.py

Renders the SSQ/MSSQ reference figures for one session, from the scored
data subjective_processing.py already produced. Kept separate from
subjective_processing.py so that data processing and figure rendering
stay split, matching the rest of the pipeline's architecture (cf.
ecg.py / balance.py / summary.py -> visualization.py).

Not meant to be run standalone -- subjective_processing.py imports this
module and calls render_session_figures() once per session, after
scoring that session's SSQ and (shared, per-patient) MSSQ data.

Figures rendered (see render_session_figures()):
    <Patient_prefix>[_<iwxdata tag>]_Percentile_Curve.png   (MSSQ Fig. 1, Golding, 1998)
    <Patient_prefix>[_<iwxdata tag>]_Total_Severity.png     (SSQ Fig. 1, Kennedy et al., 1993)
    <Patient_prefix>[_<iwxdata tag>]_Hop_Count.png           (SSQ Fig. 3, Kennedy et al., 1993)

A fourth figure -- the SSQ Fig. 2 / Table 5 percentile plot -- is not
currently rendered here; it's expected to be added in a later pass.
SSQ_TABLE5 / SSQ_TABLE5_PERCENTILES / _table5_percentile() below are the
reference data and interpolation helper it will need, kept here (rather
than in subjective_processing.py) since they exist solely to support
that figure.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend -- safe to import anywhere
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reference constants used by the figures below
# ---------------------------------------------------------------------------

SSQ_SUBSCALE_COLORS = {"N": "#1f77b4", "O": "#ff7f0e", "D": "#2ca02c", "TS": "#d62728"}

# Calibration-sample means (Kennedy et al., 1993, Table 5), used as the
# dashed reference line on the Total_Severity figure.
SSQ_CALIBRATION_MEANS = {"N": 7.7, "O": 10.6, "D": 6.4, "TS": 9.8}

# Table 5: percentile points for each SSQ scale in the calibration sample
# (N ~ 1,100 observations). Not currently plotted (see module docstring)
# -- kept here for the not-yet-reinstated SSQ Fig. 2 / Percentile figure.
SSQ_TABLE5_PERCENTILES = [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
SSQ_TABLE5 = {
    "N":  [0.0, 0.0, 0.0, 0.0, 0.0, 9.5, 9.5, 9.5, 9.5, 19.7, 28.6, 38.2, 38.2, 47.7, 57.2, 66.8],
    "O":  [0.0, 0.0, 7.6, 7.6, 7.6, 7.6, 15.2, 15.2, 22.7, 27.7, 30.3, 45.5, 45.5, 53.1, 53.1, 60.7],
    "D":  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.9, 13.9, 27.8, 41.7, 41.7, 55.7, 55.7, 83.5],
    "TS": [0.0, 3.7, 3.7, 3.7, 7.5, 7.5, 11.2, 15.0, 22.5, 22.5, 30.0, 44.9, 44.9, 48.7, 56.2, 75.9],
}


def _table5_percentile(subscale, score):
    """Interpolate an approximate percentile for `score` on `subscale`
    using the published Table 5 points (Kennedy et al., 1993)."""
    xs, ys = SSQ_TABLE5_PERCENTILES, SSQ_TABLE5[subscale]
    if score is None:
        return None
    if score <= ys[0]:
        return xs[0]
    if score >= ys[-1]:
        return xs[-1]
    for (p_lo, s_lo), (p_hi, s_hi) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if s_lo <= score <= s_hi:
            if s_hi == s_lo:
                return p_lo
            frac = (score - s_lo) / (s_hi - s_lo)
            return round(p_lo + frac * (p_hi - p_lo), 1)
    return None


def _trial_number(colname):
    m = re.search(r"(\d+)\s*$", colname)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# MSSQ percentile curve reference (Golding, 1998, Fig. 1), digitized.
# Duplicated here (rather than imported from subjective_processing.py)
# so this module has no dependency back on it -- the two only communicate
# through render_session_figures()'s plain-data arguments.
# ---------------------------------------------------------------------------

MSSQ_PERCENTILE_CURVE = [
    (0, 0), (10, 8), (20, 15), (30, 22), (40, 30), (50, 38),
    (60, 48), (70, 62), (80, 80), (90, 100), (95, 130), (99, 190), (100, 200),
]


# ---------------------------------------------------------------------------
# Output filename construction
# ---------------------------------------------------------------------------

def build_image_filename(patient_prefix, tag, suffix):
    if tag:
        return f"{patient_prefix}_{tag}_{suffix}.png"
    return f"{patient_prefix}_{suffix}.png"


# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------

def plot_mssq_percentile_curve(rows_out, out_path):
    """MSSQ Figure 1 analogue: cumulative percentile curve (Golding, 1998),
    with this session's respondent(s) marked on it."""
    xs = [s for _, s in MSSQ_PERCENTILE_CURVE]
    ys = [p for p, _ in MSSQ_PERCENTILE_CURVE]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs, ys, marker="o", color="#1f77b4", label="Digitized reference curve\n(Golding, 1998, Fig. 1)")

    for r in rows_out:
        score, pct = r["MSSQ_raw"], r["MSSQ_approx_percentile"]
        if score is None:
            continue
        ax.plot(score, pct, marker="*", markersize=16, color="#d62728", linestyle="none",
                 label=f"Row {r['row']} score={score} (~{pct}th pct.)")
        ax.axvline(score, color="#d62728", linestyle=":", alpha=0.5)
        ax.axhline(pct, color="#d62728", linestyle=":", alpha=0.5)

    ax.set_xlabel("MSSQ Raw Score")
    ax.set_ylabel("Percentile")
    ax.set_title("Motion Sickness Susceptibility Questionnaire\nApproximate Percentile Curve")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.01, 0.01,
              "Note: curve digitized from Golding (1998) Fig. 1; approximate, not the "
              "original published lookup table.",
              fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ssq_total_severity(results, out_path):
    """SSQ Figure 1 analogue: severity by subscale for each trial in this
    session (cf. Kennedy et al., 1993, Fig. 1 population histogram)."""
    trials = list(results.keys())
    subscales = ["N", "O", "D", "TS"]
    x = range(len(trials))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sub in enumerate(subscales):
        values = [results[t][sub] for t in trials]
        offsets = [xi + (i - 1.5) * width for xi in x]
        ax.bar(offsets, values, width=width, label=sub, color=SSQ_SUBSCALE_COLORS[sub])

    for sub in subscales:
        ax.axhline(SSQ_CALIBRATION_MEANS[sub], color=SSQ_SUBSCALE_COLORS[sub],
                    linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("Trial_Response_", "Trial ") for t in trials])
    ax.set_ylabel("SSQ Score")
    ax.set_title("Simulator Sickness Questionnaire\nSeverity by Trial (dashed = calibration-sample mean)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ssq_hop_count(results, out_path):
    """SSQ Figure 3 analogue: scores as a function of trial ('hop') number."""
    trials = list(results.keys())
    hop_numbers = [_trial_number(t) or (i + 1) for i, t in enumerate(trials)]

    fig, ax = plt.subplots(figsize=(7, 5))
    for sub in ["N", "O", "D", "TS"]:
        values = [results[t][sub] for t in trials]
        ax.plot(hop_numbers, values, marker="o", color=SSQ_SUBSCALE_COLORS[sub], label=sub)

    ax.set_xlabel("Hop / Trial Number")
    ax.set_ylabel("SSQ Score")
    ax.set_title("SSQ Scores Across Trials\n(cf. Kennedy et al., 1993, Fig. 3)")
    ax.set_xticks(hop_numbers)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration -- the entry point subjective_processing.py calls
# ---------------------------------------------------------------------------

def render_session_figures(session_dir, patient_prefix, tag, ssq_results, mssq_rows):
    """
    Render all of one session's reference figures into `session_dir`,
    using the naming convention build_image_filename(patient_prefix, tag,
    suffix). Returns a dict of figure name -> Path, in the same shape
    subjective_processing.process_session() returns under "figures".
    """
    session_dir = Path(session_dir)

    mssq_curve_path = session_dir / build_image_filename(patient_prefix, tag, "Percentile_Curve")
    ssq_total_severity_path = session_dir / build_image_filename(patient_prefix, tag, "Total_Severity")
    ssq_hop_count_path = session_dir / build_image_filename(patient_prefix, tag, "Hop_Count")

    plot_mssq_percentile_curve(mssq_rows, mssq_curve_path)
    plot_ssq_total_severity(ssq_results, ssq_total_severity_path)
    plot_ssq_hop_count(ssq_results, ssq_hop_count_path)

    return {
        "mssq_percentile_curve": mssq_curve_path,
        "ssq_total_severity": ssq_total_severity_path,
        "ssq_hop_count": ssq_hop_count_path,
    }