import cv2
import tkinter as tk
from tkinter import Tk, filedialog

import patient_config as pc
import crop_batch

# Import the pupil-detection logic from pupils.py (used only for the
# 'm' snap-to-pupil hotkey below). pupils.py must live in the same
# folder as this script.
try:
    import pupils
except ImportError:
    pupils = None

# -----------------------------
# Internal working size used by pupils.process_frame()
# (crop_to_aspect_ratio() in pupils.py defaults to this)
# -----------------------------
PUPIL_TARGET_W = 640
PUPIL_TARGET_H = 480

WINDOW_NAME = "Crop Selector"

# How far the border-width trackbar can go, in pixels.
BORDER_TRACKBAR_MAX = 50


def print_instructions():
    print("=" * 60)
    print("EYE VIDEO CROP TOOL")
    print("=" * 60)
    print("This tool no longer saves a cropped .mp4 -- it only updates")
    print("that patient's crop parameters in patient_config.json.")
    print("Use crop_batch.py (or the prompt this tool offers you) to")
    print("actually produce the cropped videos.")
    print()
    print("Hotkeys (once the preview window is open):")
    print("  Left / Right arrow   - step one frame back / forward")
    print("  Frame slider         - jump to any frame")
    print("  Border slider        - set the border width (preview only)")
    print("  Click + drag box     - move the crop box")
    print("  m                    - snap crop box center to detected pupil")
    print("  c                    - confirm crop box / border and continue")
    print("  q / Esc              - quit without saving")
    print("=" * 60)


def get_pupil_center_original_coords(frame):
    """Runs pupils.process_frame() on a raw frame and maps the resulting
    ellipse center from pupils.py's internal 640x480 working space back
    into the pixel coordinates of the original (raw) frame."""
    if pupils is None:
        print("pupils.py could not be imported -- cannot snap to pupil.")
        return None

    orig_h, orig_w = frame.shape[:2]

    try:
        rotated_rect = pupils.process_frame(frame)
    except Exception as e:
        print(f"Pupil detection failed on this frame: {e}")
        return None

    (cx, cy), (w, h), _angle = rotated_rect
    if w <= 0 or h <= 0:
        print("No pupil detected on this frame.")
        return None

    desired_ratio = PUPIL_TARGET_W / PUPIL_TARGET_H
    current_ratio = orig_w / orig_h

    # Reverse pupils.crop_to_aspect_ratio()'s crop + resize so the
    # detected center maps back onto the original frame.
    if current_ratio > desired_ratio:
        cropped_w = int(desired_ratio * orig_h)
        cropped_h = orig_h
        offset_x = (orig_w - cropped_w) // 2
        offset_y = 0
    else:
        cropped_h = int(orig_w / desired_ratio)
        cropped_w = orig_w
        offset_x = 0
        offset_y = (orig_h - cropped_h) // 2

    scale_x = cropped_w / PUPIL_TARGET_W
    scale_y = cropped_h / PUPIL_TARGET_H

    orig_x = cx * scale_x + offset_x
    orig_y = cy * scale_y + offset_y

    return orig_x, orig_y


def select_video():
    root = Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(
        title="Select Raw Eye MP4 (left_eye.mp4 or right_eye.mp4)",
        filetypes=[("MP4 files", "*.mp4")]
    )
    root.destroy()
    return input_file


class CropSelector:
    def __init__(self, video_path, crop_values):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.frame_count <= 0:
            raise RuntimeError("Could not determine frame count for video.")

        # Crop box size is fixed for the session (comes from the config /
        # defaults); only its position and the border width are adjusted
        # interactively here.
        self.crop_width = crop_values["crop_width"]
        self.crop_height = crop_values["crop_height"]

        if self.crop_width > self.frame_w or self.crop_height > self.frame_h:
            print("Warning: crop_width/crop_height are larger than the video frame.")

        self.idx = 0
        self.x = max(0, min(crop_values["start_x"], max(0, self.frame_w - self.crop_width)))
        self.y = max(0, min(crop_values["start_y"], max(0, self.frame_h - self.crop_height)))
        self.border_max = max(1, min(self.crop_width, self.crop_height) // 2, BORDER_TRACKBAR_MAX)
        self.border_size = max(0, min(crop_values["border_size"], self.border_max))

        self.dragging = False
        self.drag_offset = (0, 0)
        self._suppress_trackbar_cb = False
        self.confirmed = False

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(1, self.frame_count - 1), self._on_frame_trackbar)
        # OpenCV always stacks trackbars at the top of the window (there is
        # no way to force one to the bottom), so this is the second control
        # under the frame slider -- as close to "at the bottom" as HighGUI
        # allows.
        cv2.createTrackbar("Border", WINDOW_NAME, self.border_size, self.border_max, self._on_border_trackbar)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_frame_trackbar(self, val):
        if self._suppress_trackbar_cb:
            return
        self.idx = val

    def _on_border_trackbar(self, val):
        if self._suppress_trackbar_cb:
            return
        self.border_size = val

    def _set_idx(self, new_idx):
        new_idx = max(0, min(new_idx, self.frame_count - 1))
        self.idx = new_idx
        self._suppress_trackbar_cb = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, new_idx)
        self._suppress_trackbar_cb = False

    def _on_mouse(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.x <= mx <= self.x + self.crop_width and self.y <= my <= self.y + self.crop_height:
                self.dragging = True
                self.drag_offset = (mx - self.x, my - self.y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                new_x = mx - self.drag_offset[0]
                new_y = my - self.drag_offset[1]
                self.x = max(0, min(new_x, max(0, self.frame_w - self.crop_width)))
                self.y = max(0, min(new_y, max(0, self.frame_h - self.crop_height)))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def _get_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _draw_overlay(self, frame):
        display = frame.copy()
        cv2.rectangle(
            display, (self.x, self.y),
            (self.x + self.crop_width, self.y + self.crop_height),
            (0, 255, 0), 2
        )
        # Preview the border that crop_batch.py would draw inside the box.
        if self.border_size > 0:
            cv2.rectangle(
                display,
                (self.x, self.y),
                (self.x + self.crop_width - 1, self.y + self.crop_height - 1),
                (255, 255, 255),
                self.border_size
            )
        center_x = self.x + self.crop_width // 2
        center_y = self.y + self.crop_height // 2
        cv2.drawMarker(
            display, (center_x, center_y), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2
        )
        cv2.putText(
            display,
            f"Frame {self.idx}/{self.frame_count - 1}  pos=({self.x},{self.y})  border={self.border_size}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        return display

    def run(self):
        last_idx = -1
        frame = None
        while True:
            if self.idx != last_idx:
                new_frame = self._get_frame(self.idx)
                if new_frame is None:
                    # Couldn't read at this index (e.g. past the end) -- back off.
                    self._set_idx(max(0, self.idx - 1))
                    continue
                frame = new_frame
                last_idx = self.idx

            display = self._draw_overlay(frame)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKeyEx(30)
            if key == -1:
                continue
            raw_key = key & 0xFF

            if raw_key == ord('q') or raw_key == 27:  # q or Esc
                break
            elif raw_key == ord('c'):
                self.confirmed = True
                break
            elif raw_key == ord('m'):
                center = get_pupil_center_original_coords(frame)
                if center is not None:
                    cx, cy = center
                    new_x = int(round(cx - self.crop_width / 2))
                    new_y = int(round(cy - self.crop_height / 2))
                    self.x = max(0, min(new_x, max(0, self.frame_w - self.crop_width)))
                    self.y = max(0, min(new_y, max(0, self.frame_h - self.crop_height)))
                    print(f"Snapped to pupil center ({cx:.1f}, {cy:.1f}) -> box top-left ({self.x}, {self.y})")
            elif key in (2424832, 65361, 63234, 81):   # Left arrow (Win/Linux/Mac/other)
                self._set_idx(self.idx - 1)
            elif key in (2555904, 65363, 63235, 83):   # Right arrow (Win/Linux/Mac/other)
                self._set_idx(self.idx + 1)

        cv2.destroyAllWindows()
        return self.confirmed, self.x, self.y, self.border_size


def show_batch_prompt(patient_id, config, changed_side):
    """Two-column (left eye / right eye) summary of this patient's saved
    crop parameters, with the side that was just changed shown in red.
    Returns True if the user wants to run crop_batch now."""
    result = {"run": False}

    root = tk.Tk()
    root.title(f"Run Batch Crop for {patient_id}?")

    tk.Label(
        root,
        text=f"Both eyes now have saved crop parameters for {patient_id}.",
        font=("Segoe UI", 10, "bold"),
    ).grid(row=0, column=0, columnspan=2, padx=12, pady=(12, 6))

    headers = {"left": "Left Eye", "right": "Right Eye"}
    for col, side in enumerate(("left", "right")):
        color = "red" if side == changed_side else "black"
        tk.Label(
            root, text=headers[side], font=("Segoe UI", 10, "bold"), fg=color
        ).grid(row=1, column=col, padx=24, pady=(0, 4))

    for r, key in enumerate(pc.CROP_KEYS, start=2):
        for col, side in enumerate(("left", "right")):
            val = config.get(patient_id, {}).get(side, {}).get(key, "-")
            color = "red" if side == changed_side else "black"
            tk.Label(root, text=f"{key} = {val}", fg=color, anchor="w").grid(
                row=r, column=col, sticky="w", padx=24
            )

    def on_yes():
        result["run"] = True
        root.destroy()

    def on_no():
        result["run"] = False
        root.destroy()

    btn_row = len(pc.CROP_KEYS) + 2
    btn_frame = tk.Frame(root)
    btn_frame.grid(row=btn_row, column=0, columnspan=2, pady=12)
    tk.Button(btn_frame, text="Yes, run batch crop", command=on_yes, width=18).pack(side="left", padx=6)
    tk.Button(btn_frame, text="No", command=on_no, width=10).pack(side="left", padx=6)

    root.mainloop()
    return result["run"]


def main():
    print_instructions()

    input_file = select_video()
    if not input_file:
        print("No file selected.")
        return

    patient_id = pc.find_patient_id(input_file)
    side = pc.get_eye_side(input_file)

    if patient_id is None:
        print("Could not determine a Patient_X folder from the selected file's path.")
        pc.show_error(
            "Patient Not Recognized",
            "Could not determine a Patient_<number> folder for this file.\n\n"
            "Make sure the video lives somewhere under a 'Patient_<number>' folder."
        )
        return

    if side is None:
        print("Could not determine eye side (left/right) from the filename.")
        pc.show_error(
            "Eye Side Not Recognized",
            "Could not tell whether this is a left or right eye video.\n\n"
            "The filename should contain 'left_eye' or 'right_eye'."
        )
        return

    config = pc.load_config()
    crop_values = pc.get_crop_values(config, patient_id, side)

    selector = CropSelector(input_file, crop_values)
    confirmed, x, y, border_size = selector.run()
    selector.cap.release()

    if not confirmed:
        print("Cancelled -- no changes were made.")
        return

    values = {
        "crop_width": selector.crop_width,
        "crop_height": selector.crop_height,
        "start_x": x,
        "start_y": y,
        "border_size": border_size,
    }

    print(f"Selected crop box for {patient_id} ({side} eye): {values}")

    if not pc.ask_yes_no(
        "Save Crop Parameters?",
        f"Save these crop parameters for {patient_id} ({side} eye)?\n\n"
        + "\n".join(f"{k} = {v}" for k, v in values.items())
    ):
        print("Not saved.")
        return

    config = pc.update_side(config, patient_id, side, values)
    pc.save_config(config)
    print(f"Saved crop parameters for {patient_id} ({side} eye) to {pc.CONFIG_PATH}")

    other_side = "right" if side == "left" else "left"
    both_ready = (
        pc.has_crop_params(pc.get_side(config, patient_id, side))
        and pc.has_crop_params(pc.get_side(config, patient_id, other_side))
    )

    if both_ready:
        if show_batch_prompt(patient_id, config, changed_side=side):
            patient_dir = pc.find_patient_dir(input_file)
            print(f"Running batch crop for {patient_id} ...")
            crop_batch.run_batch_for_patient(patient_dir)
        else:
            print("Batch crop not run.")
    else:
        print(f"({other_side} eye crop parameters not yet saved for {patient_id} -- "
              f"run crop.py on a {other_side}_eye.mp4 for this patient to enable batch cropping.)")


if __name__ == "__main__":
    main()