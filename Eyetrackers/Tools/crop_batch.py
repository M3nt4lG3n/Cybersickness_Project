import cv2
import os
from tkinter import Tk, filedialog

import patient_config as pc

# -----------------------------
# Border Settings
# -----------------------------
BORDER_COLOR = (255, 255, 255)   # White (B, G, R) -- always white, not stored in config

# Filenames to look for (case-insensitive) and which eye side they are.
TARGET_FILENAMES = {
    "left_eye.mp4": "left",
    "right_eye.mp4": "right",
}


def crop_video(input_file, x, y, crop_width, crop_height, border_size):
    base, ext = os.path.splitext(input_file)
    output_file = base + "_cropped" + ext

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"  Could not open video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (crop_width, crop_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = frame[y:y + crop_height, x:x + crop_width].copy()

        if border_size > 0:
            cv2.rectangle(
                cropped,
                (0, 0),
                (crop_width - 1, crop_height - 1),
                BORDER_COLOR,
                border_size
            )

        out.write(cropped)

    cap.release()
    out.release()
    print(f"  Saved: {output_file}")


def find_patient_videos(patient_dir):
    """Walks patient_dir looking for left_eye.mp4 / right_eye.mp4.
    Returns a list of (dirpath, filename, side) tuples."""
    matches = []
    for dirpath, _dirnames, filenames in os.walk(patient_dir):
        for filename in filenames:
            side = TARGET_FILENAMES.get(filename.lower())
            if side:
                matches.append((dirpath, filename, side))
    return matches


def run_batch_for_patient(patient_dir):
    """Crops every left_eye.mp4 / right_eye.mp4 found under patient_dir,
    using that patient's saved crop parameters from patient_config.json.
    Can be called directly (e.g. from crop.py) without going through the
    folder-selection dialog."""
    if not patient_dir:
        print("No patient folder given.")
        return

    patient_id = pc.find_patient_id(patient_dir)
    if patient_id is None:
        pc.show_error(
            "Patient Not Recognized",
            f"Could not determine a Patient_<number> id from:\n{patient_dir}"
        )
        return

    matches = find_patient_videos(patient_dir)
    if not matches:
        print("No left_eye.mp4 or right_eye.mp4 files found.")
        return

    config = pc.load_config()
    needed_sides = sorted({side for _, _, side in matches})
    missing = [s for s in needed_sides if not pc.has_crop_params(pc.get_side(config, patient_id, s))]

    if missing:
        pc.show_error(
            "Missing Crop Parameters",
            f"No saved crop parameters for {patient_id} ({', '.join(missing)} eye).\n\n"
            f"Please set the parameters using crop.py first."
        )
        return

    print(f"Found {len(matches)} file(s) to process for {patient_id}.\n")

    for dirpath, filename, side in matches:
        side_cfg = pc.get_side(config, patient_id, side)
        input_file = os.path.join(dirpath, filename)
        print(f"Processing [{side}]: {input_file}")
        crop_video(
            input_file,
            side_cfg["start_x"],
            side_cfg["start_y"],
            side_cfg["crop_width"],
            side_cfg["crop_height"],
            side_cfg["border_size"],
        )

    print("\nFinished!")


def select_patient_directory():
    root = Tk()
    root.withdraw()
    parent_dir = filedialog.askdirectory(title="Select Patient Folder")
    root.destroy()
    return parent_dir


def main():
    patient_dir = select_patient_directory()
    if not patient_dir:
        print("No folder selected.")
        return

    run_batch_for_patient(patient_dir)


if __name__ == "__main__":
    main()