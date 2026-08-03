#!/usr/bin/env python3
"""
subjective_inputs.py

Collects Patient Report, SSQ (Simulator Sickness Questionnaire), and MSSQ
(Motion Sickness Susceptibility Questionnaire - Short form) data through a
Tkinter UI, and writes the results out as CSV files into the appropriate
patient folders.

Usage:
    python subjective_inputs.py

On launch you will be asked to select either:
  - A "Patient X" superfolder that contains the (up to four) trial
    subfolders, or
  - A single trial subfolder itself.

See the header comments on each function below for details on the folder
structures this script expects and produces.
"""

import os
import re
import csv
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SSQ_SYMPTOMS = [
    "General Discomfort", "Fatigue", "Headache", "Eyestrain",
    "Difficulty Focusing", "Increased Salivation", "Sweating", "Nausea",
    "Difficulty Concentrating", "Fullness of Head", "Blurred Vision",
    "Dizzy (eyes open)", "Dizzy (eyes closed)", "Vertigo",
    "Stomach Awareness", "Burping",
]

# Internal keys used for MSSQ CSV column names (Car_Child, Car_10, etc.)
MSSQ_ACTIVITIES = [
    "Car", "Buses/Coaches", "Trains", "Aircraft", "Small_Boats",
    "Ships", "Swings", "Roundabouts", "Big_Dippers",
]

# Human readable labels shown in the UI for each activity row
MSSQ_ACTIVITY_DISPLAY = {
    "Car": "Cars",
    "Buses/Coaches": "Buses or Coaches",
    "Trains": "Trains",
    "Aircraft": "Aircraft",
    "Small_Boats": "Small Boats",
    "Ships": "Ships (e.g. Channel Ferries)",
    "Swings": "Swings in Playgrounds",
    "Roundabouts": "Roundabouts in Playgrounds",
    "Big_Dippers": "Big Dippers / Funfair Rides",
}

# (stored_code, ui_label)
MSSQ_RESPONSE_OPTIONS = [
    ("t", "N/A -\nNever Travelled"),
    ("0", "Never\nFelt Sick"),
    ("1", "Rarely\nFelt Sick"),
    ("2", "Sometimes\nFelt Sick"),
    ("3", "Frequently\nFelt Sick"),
]

AGE_OPTIONS = [str(i) for i in range(18, 26)]
REPORT_VALUE_OPTIONS = [str(i) for i in range(1, 11)]
RAW_FOLDER_NAME = "Raw_Labscribe"
SUBJ_RESULTS_FOLDER_NAME = "Subjective_Results"
NUM_REPORT_ROWS = 15

# Maps the 0.X value found in an .iwxdata filename to a trial number 1-4
TRIAL_VALUE_TO_INDEX = {"0.0": 1, "0.1": 2, "0.2": 3, "0.3": 4}


# ---------------------------------------------------------------------------
# Folder / file detection helpers
# ---------------------------------------------------------------------------

def find_iwxdata_and_processed(folder):
    """
    Look for a .iwxdata file directly inside `folder`. If none is found,
    check inside a Raw_Labscribe subfolder (this is the "processed" file
    structure, where the raw file is buried one level deeper).

    Returns (path_or_None, processed_bool) where processed_bool reflects
    whether a Raw_Labscribe folder exists at all (independent of whether an
    .iwxdata file was actually found in it).
    """
    iwx_path = None
    try:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".iwxdata"):
                iwx_path = os.path.join(folder, f)
                break
    except OSError:
        pass

    raw_dir = os.path.join(folder, RAW_FOLDER_NAME)
    processed = os.path.isdir(raw_dir)

    if iwx_path is None and processed:
        try:
            for f in sorted(os.listdir(raw_dir)):
                if f.lower().endswith(".iwxdata"):
                    iwx_path = os.path.join(raw_dir, f)
                    break
        except OSError:
            pass

    return iwx_path, processed


def parse_trial_value(iwxdata_path):
    """Pull the 0.X value out of a 'Patient_Y_0.X....iwxdata' filename."""
    if not iwxdata_path:
        return None
    name = os.path.basename(iwxdata_path)
    match = re.search(r"(\d\.\d+)", name)
    return match.group(1) if match else None


def get_output_dir(trial_folder, processed):
    """Where Reports/SSQ csvs should be written for this trial folder."""
    if not processed:
        return trial_folder
    out_dir = os.path.join(trial_folder, SUBJ_RESULTS_FOLDER_NAME)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_patient_name_and_root(selected_path, is_superfolder):
    """
    Returns (patient_folder_name, patient_root_path).
    patient_root_path is always the "Patient_N" folder that directly
    contains the trial subfolders and the MSSQ csv.
    """
    if is_superfolder:
        return os.path.basename(os.path.normpath(selected_path)), selected_path
    parent = os.path.dirname(os.path.normpath(selected_path))
    return os.path.basename(parent), parent


class TrialColumn:
    """Represents one spindown/comment/SSQ column -- i.e. one trial folder."""

    def __init__(self, folder, header, value, trial_index, processed, notice=None):
        self.folder = folder
        self.header = header          # display text: the 0.X value, or folder name as fallback
        self.value = value            # the raw 0.X string, or None
        self.trial_index = trial_index  # 1-4, which Trial_Response_N slot this feeds
        self.processed = processed
        self.notice = notice
        self.output_dir = get_output_dir(folder, processed)


def analyze_selection(selected_path):
    """
    Determine whether `selected_path` is a Patient superfolder or a single
    trial subfolder, and gather everything needed to build the UI.

    Returns a dict:
        mode: "superfolder" | "subfolder"
        patient_name: str
        patient_root: str
        columns: list[TrialColumn]
        mssq_existing: bool
        mssq_path: str
    Raises ValueError if the selection can't be interpreted.
    """
    selected_path = os.path.normpath(selected_path)
    iwx, processed = find_iwxdata_and_processed(selected_path)

    if iwx is not None:
        mode = "subfolder"
    else:
        try:
            children = [
                d for d in sorted(os.listdir(selected_path))
                if os.path.isdir(os.path.join(selected_path, d))
            ]
        except OSError:
            children = []
        if children:
            mode = "superfolder"
        else:
            raise ValueError(
                "The selected folder has no .iwxdata file and no "
                "subfolders. Please select a Patient superfolder or a "
                "trial subfolder."
            )

    if mode == "superfolder":
        patient_name, patient_root = get_patient_name_and_root(selected_path, True)
        subfolders = [
            os.path.join(selected_path, d) for d in sorted(os.listdir(selected_path))
            if os.path.isdir(os.path.join(selected_path, d))
        ]
        if len(subfolders) < 4:
            print(f"[WARNING] Expected 4 subfolders under '{selected_path}', "
                  f"found {len(subfolders)}. Continuing anyway.")

        columns = []
        for idx, sf in enumerate(subfolders, start=1):
            iwx_f, proc_f = find_iwxdata_and_processed(sf)
            value = parse_trial_value(iwx_f)
            if value is not None:
                header = value
                notice = None
            else:
                header = os.path.basename(sf)
                notice = (f"[WARNING] No .iwxdata file found in '{sf}'; "
                           f"using the folder name as the column header.")
                print(notice)
            columns.append(TrialColumn(sf, header, value, idx, proc_f, notice))

        mssq_path = os.path.join(patient_root, f"{patient_name}_MSSQ.csv")
        return {
            "mode": "superfolder",
            "patient_name": patient_name,
            "patient_root": patient_root,
            "columns": columns,
            "mssq_existing": os.path.isfile(mssq_path),
            "mssq_path": mssq_path,
        }

    # mode == "subfolder"
    patient_name, patient_root = get_patient_name_and_root(selected_path, False)
    value = parse_trial_value(iwx)
    if value is not None:
        header = value
        trial_index = TRIAL_VALUE_TO_INDEX.get(value, 1)
        notice = None
    else:
        header = os.path.basename(selected_path)
        trial_index = 1
        notice = (f"[WARNING] No .iwxdata file found in '{selected_path}'; "
                   f"using the folder name as the column header.")
        print(notice)
    column = TrialColumn(selected_path, header, value, trial_index, processed, notice)

    mssq_path = os.path.join(patient_root, f"{patient_name}_MSSQ.csv")
    return {
        "mode": "subfolder",
        "patient_name": patient_name,
        "patient_root": patient_root,
        "columns": [column],
        "mssq_existing": os.path.isfile(mssq_path),
        "mssq_path": mssq_path,
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def make_spindown(parent, options, default="NULL", width=6):
    """
    Builds a dropdown (OptionMenu) that starts on `default`. The first time
    the user picks something other than `default`, that option is removed
    from the menu permanently (per-widget), so it can't be re-selected.
    Returns (widget, StringVar).
    """
    var = tk.StringVar(value=default)
    om = tk.OptionMenu(parent, var, default, *options)
    om.config(width=width)
    state = {"armed": True}

    def on_write(*_args):
        if state["armed"] and var.get() != default:
            state["armed"] = False
            menu = om["menu"]
            try:
                menu.delete(0)  # default is always index 0 until removed
            except tk.TclError:
                pass

    var.trace_add("write", on_write)
    return om, var


class ScrollableFrame(tk.Frame):
    """A vertically scrollable frame; put widgets in .scrollable_frame."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("Subjective Inputs")

        selected = filedialog.askdirectory(
            title="Select a Patient superfolder or a trial subfolder"
        )
        if not selected:
            self.destroy()
            sys.exit(0)

        try:
            self.info = analyze_selection(selected)
        except ValueError as e:
            messagebox.showerror("Invalid Selection", str(e))
            self.destroy()
            sys.exit(1)

        # MSSQ defaults ON, except in subfolder mode when a MSSQ csv
        # already exists for this patient -- then it defaults OFF and a
        # toggle button is shown.
        default_mssq_on = not (self.info["mode"] == "subfolder" and self.info["mssq_existing"])
        self.mssq_enabled = tk.BooleanVar(value=default_mssq_on)

        self.report_vars = {}   # col_index -> list[(spindown_var, comment_var)]
        self.ssq_vars = {}      # symptom_index -> {col_index: IntVar}
        self.mssq_age_var = tk.StringVar(value="NULL")
        self.mssq_gender_var = tk.StringVar(value="")
        self.mssq_eyesight_var = tk.StringVar(value="")
        self.mssq_child_vars = {}
        self.mssq_ten_vars = {}
        self._mssq_toggle_btn = None

        self.deiconify()
        self.geometry("1300x850")
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = ScrollableFrame(self)
        outer.pack(fill="both", expand=True)
        content = outer.scrollable_frame

        title = f"Patient: {self.info['patient_name']}  |  Mode: {self.info['mode']}"
        tk.Label(content, text=title, font=("Arial", 12, "italic")).pack(anchor="w", padx=10, pady=(10, 0))

        self._build_patient_reports(content)
        self._build_ssq(content)
        self._build_mssq_toggle(content)

        self.mssq_frame_holder = tk.Frame(content)
        self.mssq_frame_holder.pack(fill="x", padx=10, pady=10)
        self._render_mssq_section()

        tk.Button(
            content, text="Save and Generate CSV Files",
            font=("Arial", 12, "bold"), command=self._on_save,
            bg="#4a90d9", fg="white", padx=10, pady=6,
        ).pack(pady=20)

    def _build_patient_reports(self, parent):
        frame = tk.LabelFrame(parent, text="Patient Reports", font=("Arial", 14, "bold"), padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=10)

        tk.Label(frame, text="", width=16).grid(row=0, column=0)
        for r in range(NUM_REPORT_ROWS):
            tk.Label(frame, text=f"Patient Report {r + 1}", anchor="w", width=16) \
                .grid(row=r + 2, column=0, sticky="w", padx=4, pady=2)

        col_pos = 1
        for ci, col in enumerate(self.info["columns"]):
            tk.Label(frame, text=col.header, font=("Arial", 11, "bold")) \
                .grid(row=0, column=col_pos, columnspan=2, pady=4)
            tk.Label(frame, text="Value", font=("Arial", 9)).grid(row=1, column=col_pos)
            tk.Label(frame, text="Comment", font=("Arial", 9)).grid(row=1, column=col_pos + 1)

            rows = []
            for r in range(NUM_REPORT_ROWS):
                om, sv = make_spindown(frame, REPORT_VALUE_OPTIONS, default="NULL", width=6)
                om.grid(row=r + 2, column=col_pos, padx=4, pady=2)
                cv = tk.StringVar(value="")
                tk.Entry(frame, textvariable=cv, width=20).grid(row=r + 2, column=col_pos + 1, padx=4, pady=2)
                rows.append((sv, cv))
            self.report_vars[ci] = rows
            col_pos += 2

    def _build_ssq(self, parent):
        frame = tk.LabelFrame(parent, text="SSQ Results", font=("Arial", 14, "bold"), padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=10)

        tk.Label(frame, text="Symptom", width=24, anchor="w", font=("Arial", 10, "bold")) \
            .grid(row=0, column=0, sticky="w")

        for ci, col in enumerate(self.info["columns"]):
            label = f"Trial {col.trial_index}" if col.trial_index else col.header
            tk.Label(frame, text=label, font=("Arial", 10, "bold")).grid(row=0, column=ci + 1, padx=14)

        for si, symptom in enumerate(SSQ_SYMPTOMS):
            tk.Label(frame, text=symptom, anchor="w", width=24).grid(row=si + 1, column=0, sticky="w", pady=1)
            for ci in range(len(self.info["columns"])):
                var = tk.IntVar(value=0)
                btn = tk.Checkbutton(frame, variable=var, indicatoron=False,
                                      onvalue=1, offvalue=0, width=6, text="No")

                def make_cmd(v=var, b=btn):
                    return lambda: b.config(
                        text="Yes" if v.get() else "No",
                        bg=("#9fd89f" if v.get() else "#d89f9f"),
                    )

                btn.config(command=make_cmd())
                btn.grid(row=si + 1, column=ci + 1, padx=14, pady=1)
                self.ssq_vars.setdefault(si, {})[ci] = var

    def _build_mssq_toggle(self, parent):
        if self.info["mode"] == "subfolder" and self.info["mssq_existing"]:
            bar = tk.Frame(parent)
            bar.pack(fill="x", padx=10, pady=(10, 0))
            tk.Label(
                bar, text="An MSSQ file already exists for this patient.",
                font=("Arial", 9, "italic"),
            ).pack(side="left", padx=(0, 10))
            btn = tk.Button(
                bar, text=f"MSSQ Entry: {'ON' if self.mssq_enabled.get() else 'OFF'}",
                width=20, command=self._toggle_mssq,
            )
            btn.pack(side="left")
            self._mssq_toggle_btn = btn

    def _toggle_mssq(self):
        new_state = not self.mssq_enabled.get()
        self.mssq_enabled.set(new_state)
        if self._mssq_toggle_btn is not None:
            self._mssq_toggle_btn.config(text=f"MSSQ Entry: {'ON' if new_state else 'OFF'}")
        self._render_mssq_section()

    def _render_mssq_section(self):
        for w in self.mssq_frame_holder.winfo_children():
            w.destroy()

        if not self.mssq_enabled.get():
            return

        frame = tk.LabelFrame(self.mssq_frame_holder, text="MSSQ Responses",
                               font=("Arial", 14, "bold"), padx=10, pady=10)
        frame.pack(fill="x")

        top = tk.Frame(frame)
        top.pack(fill="x", pady=6)

        tk.Label(top, text="Age:").pack(side="left", padx=(0, 4))
        om, self.mssq_age_var = make_spindown(top, AGE_OPTIONS, default="NULL", width=6)
        om.pack(side="left", padx=(0, 20))

        tk.Label(top, text="Gender:").pack(side="left", padx=(0, 4))
        for label in ("Male", "Female"):
            tk.Radiobutton(top, text=label, variable=self.mssq_gender_var, value=label).pack(side="left")

        tk.Label(top, text="    Eyesight:").pack(side="left", padx=(10, 4))
        for label in ("Nearsighted", "Farsighted", "None"):
            tk.Radiobutton(top, text=label, variable=self.mssq_eyesight_var, value=label).pack(side="left")

        self.mssq_child_vars = {}
        self._build_mssq_grid(frame, "As a Child (before age 12) - How Often You Felt Sick or Nauseated",
                               self.mssq_child_vars)

        self.mssq_ten_vars = {}
        self._build_mssq_grid(frame, "Over the Last 10 Years - How Often You Felt Sick or Nauseated",
                               self.mssq_ten_vars)

    def _build_mssq_grid(self, parent, title, var_store):
        section = tk.LabelFrame(parent, text=title, padx=8, pady=8)
        section.pack(fill="x", pady=8)

        tk.Label(section, text="", width=22).grid(row=0, column=0)
        for ci, (_code, label) in enumerate(MSSQ_RESPONSE_OPTIONS):
            tk.Label(section, text=label, font=("Arial", 8), justify="center") \
                .grid(row=0, column=ci + 1, padx=6)

        for ri, activity in enumerate(MSSQ_ACTIVITIES):
            tk.Label(section, text=MSSQ_ACTIVITY_DISPLAY[activity], anchor="w", width=22) \
                .grid(row=ri + 1, column=0, sticky="w", pady=1)
            var = tk.StringVar(value="")
            var_store[activity] = var
            for ci, (code, _label) in enumerate(MSSQ_RESPONSE_OPTIONS):
                tk.Radiobutton(section, variable=var, value=code).grid(row=ri + 1, column=ci + 1, padx=6, pady=1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_reports(self):
        for ci, rows in self.report_vars.items():
            for ri, (sv, _cv) in enumerate(rows):
                if sv.get() == "NULL":
                    col = self.info["columns"][ci]
                    messagebox.showerror(
                        "Missing Input",
                        f"Patient Report {ri + 1} for column '{col.header}' "
                        f"has not been selected."
                    )
                    return False
        return True

    def _ssq_all_blank(self):
        for symptom_vars in self.ssq_vars.values():
            for var in symptom_vars.values():
                if var.get() == 1:
                    return False
        return True

    def _validate_mssq(self):
        if not self.mssq_enabled.get():
            return True
        if self.mssq_age_var.get() == "NULL":
            messagebox.showerror("Missing Input", "Age has not been selected in the MSSQ section.")
            return False
        if not self.mssq_gender_var.get():
            messagebox.showerror("Missing Input", "Gender has not been selected in the MSSQ section.")
            return False
        if not self.mssq_eyesight_var.get():
            messagebox.showerror("Missing Input", "Eyesight has not been selected in the MSSQ section.")
            return False
        for activity in MSSQ_ACTIVITIES:
            if not self.mssq_child_vars[activity].get():
                messagebox.showerror(
                    "Missing Input",
                    f"{MSSQ_ACTIVITY_DISPLAY[activity]} row from the child MSSQ section has no input."
                )
                return False
        for activity in MSSQ_ACTIVITIES:
            if not self.mssq_ten_vars[activity].get():
                messagebox.showerror(
                    "Missing Input",
                    f"{MSSQ_ACTIVITY_DISPLAY[activity]} row from the last-10-years MSSQ section has no input."
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Save / CSV writing
    # ------------------------------------------------------------------

    def _on_save(self):
        if not self._validate_reports():
            return

        if self._ssq_all_blank():
            proceed = messagebox.askyesno(
                "Confirm SSQ Results",
                "No SSQ symptoms were marked 'Yes' for any trial. "
                "Are you sure this is correct?"
            )
            if not proceed:
                return  # back to the entry window, nothing is saved

        if not self._validate_mssq():
            return

        try:
            self._write_reports_csvs()
            self._write_ssq_csvs()
            if self.mssq_enabled.get():
                self._write_mssq_csv()
        except OSError as e:
            messagebox.showerror("File Error", f"Could not write CSV files:\n{e}")
            return

        messagebox.showinfo("Done", "CSV files have been generated successfully.")
        self.destroy()

    def _write_reports_csvs(self):
        for ci, col in enumerate(self.info["columns"]):
            rows = self.report_vars[ci]
            filename = f"{self.info['patient_name']}_{col.header}_Reports.csv"
            path = os.path.join(col.output_dir, filename)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Patient_Report_Number", "Reported_Value", "Comments"])
                for ri, (sv, cv) in enumerate(rows):
                    writer.writerow([ri + 1, sv.get(), cv.get()])

    def _write_ssq_csvs(self):
        # One SSQ file is written per trial column (per the naming spec),
        # but each file contains the full set of Trial_Response_1..4 slots
        # so that whichever trial(s) were entered end up recorded. Slots
        # for trials not present in this session are left blank.
        for ci, col in enumerate(self.info["columns"]):
            filename = f"{self.info['patient_name']}_{col.header}_SSQ.csv"
            path = os.path.join(col.output_dir, filename)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Symptom", "Trial_Response_1", "Trial_Response_2",
                                  "Trial_Response_3", "Trial_Response_4"])
                for si, symptom in enumerate(SSQ_SYMPTOMS):
                    row = [symptom, "", "", "", ""]
                    for cj, other_col in enumerate(self.info["columns"]):
                        if other_col.trial_index:
                            row[other_col.trial_index] = self.ssq_vars[si][cj].get()
                    writer.writerow(row)

    def _write_mssq_csv(self):
        headers = ["Age", "Gender", "Eyesight"]
        headers += [f"{a}_Child" for a in MSSQ_ACTIVITIES]
        headers += [f"{a}_10" for a in MSSQ_ACTIVITIES]

        row = [self.mssq_age_var.get(), self.mssq_gender_var.get(), self.mssq_eyesight_var.get()]
        row += [self.mssq_child_vars[a].get() for a in MSSQ_ACTIVITIES]
        row += [self.mssq_ten_vars[a].get() for a in MSSQ_ACTIVITIES]

        path = self.info["mssq_path"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerow(row)


if __name__ == "__main__":
    app = App()
    app.mainloop()