import cv2
import os
from tkinter import Tk, filedialog

# Import the pupil-detection logic from pupils.py.
# pupils.py must live in the same folder as this script.
try:
    import pupils
except ImportError:
    pupils = None

# -----------------------------
# Crop Settings
# -----------------------------
crop_width = 109
crop_height = 138

# Starting position for the overlay box (top-left corner).
# This is just a starting point -- drag the box in the preview window
# to reposition it, or press 'm' to snap it to the detected pupil.
start_x = 119
start_y = 79

# -----------------------------
# Border Settings
# -----------------------------
border_size = 0                  # Border thickness in pixels
border_color = (255, 255, 255)   # White (B, G, R)

# -----------------------------
# Internal working size used by pupils.process_frame()
# (crop_to_aspect_ratio() in pupils.py defaults to this)
# -----------------------------
PUPIL_TARGET_W = 640
PUPIL_TARGET_H = 480

WINDOW_NAME = "Crop Selector"


def print_instructions():
    print("=" * 60)
    print("EYE VIDEO CROP TOOL")
    print("=" * 60)
    print("Hotkeys (once the preview window is open):")
    print("  Left / Right arrow  - step one frame back / forward")
    print("  Slider              - jump to any frame")
    print("  Click + drag box    - move the crop box")
    print("  m                   - snap crop box center to detected pupil")
    print("  c                   - confirm crop box and process the video")
    print("  q / Esc             - quit without cropping")
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
        title="Select Raw Eye MP4",
        filetypes=[("MP4 files", "*.mp4")]
    )
    root.destroy()
    return input_file


class CropSelector:
    def __init__(self, video_path):
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

        if crop_width > self.frame_w or crop_height > self.frame_h:
            print("Warning: crop_width/crop_height are larger than the video frame.")

        self.idx = 0
        self.x = max(0, min(start_x, max(0, self.frame_w - crop_width)))
        self.y = max(0, min(start_y, max(0, self.frame_h - crop_height)))

        self.dragging = False
        self.drag_offset = (0, 0)
        self._suppress_trackbar_cb = False
        self.confirmed = False

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(1, self.frame_count - 1), self._on_trackbar)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_trackbar(self, val):
        if self._suppress_trackbar_cb:
            return
        self.idx = val

    def _set_idx(self, new_idx):
        new_idx = max(0, min(new_idx, self.frame_count - 1))
        self.idx = new_idx
        self._suppress_trackbar_cb = True
        cv2.setTrackbarPos("Frame", WINDOW_NAME, new_idx)
        self._suppress_trackbar_cb = False

    def _on_mouse(self, event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.x <= mx <= self.x + crop_width and self.y <= my <= self.y + crop_height:
                self.dragging = True
                self.drag_offset = (mx - self.x, my - self.y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                new_x = mx - self.drag_offset[0]
                new_y = my - self.drag_offset[1]
                self.x = max(0, min(new_x, max(0, self.frame_w - crop_width)))
                self.y = max(0, min(new_y, max(0, self.frame_h - crop_height)))
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
            (self.x + crop_width, self.y + crop_height),
            (0, 255, 0), 2
        )
        center_x = self.x + crop_width // 2
        center_y = self.y + crop_height // 2
        cv2.drawMarker(
            display, (center_x, center_y), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2
        )
        cv2.putText(
            display, f"Frame {self.idx}/{self.frame_count - 1}  pos=({self.x},{self.y})",
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
                    new_x = int(round(cx - crop_width / 2))
                    new_y = int(round(cy - crop_height / 2))
                    self.x = max(0, min(new_x, max(0, self.frame_w - crop_width)))
                    self.y = max(0, min(new_y, max(0, self.frame_h - crop_height)))
                    print(f"Snapped to pupil center ({cx:.1f}, {cy:.1f}) -> box top-left ({self.x}, {self.y})")
            elif key in (2424832, 65361, 63234, 81):   # Left arrow (Win/Linux/Mac/other)
                self._set_idx(self.idx - 1)
            elif key in (2555904, 65363, 63235, 83):   # Right arrow (Win/Linux/Mac/other)
                self._set_idx(self.idx + 1)

        cv2.destroyAllWindows()
        return self.confirmed, self.x, self.y


def crop_video(video_path, x, y):
    base, ext = os.path.splitext(video_path)
    output_file = base + "_cropped" + ext

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for cropping.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (crop_width, crop_height))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped = frame[y:y + crop_height, x:x + crop_width].copy()

        cv2.rectangle(
            cropped,
            (0, 0),
            (crop_width - 1, crop_height - 1),
            border_color,
            border_size
        )

        out.write(cropped)
        frame_num += 1

    cap.release()
    out.release()
    return output_file, frame_num


def main():
    print_instructions()

    input_file = select_video()
    if not input_file:
        print("No file selected.")
        return

    selector = CropSelector(input_file)
    confirmed, x, y = selector.run()
    selector.cap.release()

    if not confirmed:
        print("Cancelled -- no video was cropped.")
        return

    print(f"Cropping video with box at x={x}, y={y}, width={crop_width}, height={crop_height} ...")
    output_file, frame_num = crop_video(input_file, x, y)

    print(f"Finished!\nSaved to:\n{output_file}")
    print("-" * 60)
    print("Batch-crop parameters:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"crop_width = {crop_width}")
    print(f"crop_height = {crop_height}")
    print("-" * 60)


if __name__ == "__main__":
    main()