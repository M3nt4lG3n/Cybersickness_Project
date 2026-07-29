"""
frame_debug.py
===============
Standalone single-frame diagnostic. Pulls ONE exact frame from an
*_eye_cropped.mp4, draws that frame's pupil ellipse on it using the same
crop_to_aspect_ratio-inversion math as pupil_visualizer.py, and saves it
full-size (no grid-cell stretching) so alignment can be checked cleanly.

Usage:
    python frame_debug.py <trial_folder> <left|right> <frame_number>

Example:
    python frame_debug.py "Raw_Eye_Recordings/Patient_1/Patient_20260727_121113" right 3182

Writes debug_frame_<eye>_<frame>.png next to this script, at native video
resolution, with:
    - the raw frame
    - the CSV's fitted ellipse (green), placed via the same inverse
      transform pupil_visualizer.py uses
    - a red crosshair at the exact pixel the transform computes as the
      ellipse center, plus that value printed on the image
"""

import sys
import os
import cv2
import pandas as pd

DETECTION_FRAME_W = 640
DETECTION_FRAME_H = 480


def crop_to_aspect_ratio(image, width=640, height=480):
    """Exact copy of pupils_batch.py's crop_to_aspect_ratio."""
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height
    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]
    return cv2.resize(cropped_img, (width, height))


def compute_detection_to_video_transform(video_w, video_h):
    """Exact copy of pupil_visualizer.py's inversion."""
    desired_ratio = DETECTION_FRAME_W / DETECTION_FRAME_H
    current_ratio = video_w / video_h
    if current_ratio > desired_ratio:
        crop_w = desired_ratio * video_h
        crop_h = float(video_h)
        offset_x = (video_w - crop_w) / 2.0
        offset_y = 0.0
    else:
        crop_w = float(video_w)
        crop_h = video_w / desired_ratio
        offset_x = 0.0
        offset_y = (video_h - crop_h) / 2.0
    scale = crop_w / DETECTION_FRAME_W
    return offset_x, offset_y, scale


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    trial_folder, eye, frame_number = sys.argv[1], sys.argv[2].lower(), int(sys.argv[3])

    video_path = os.path.join(trial_folder, f"{eye}_eye_cropped.mp4")
    csv_path_candidates = [
        os.path.join(trial_folder, f"{eye}_pupil_readings.csv"),
        os.path.join(trial_folder, f"{eye}_eye_readings.csv"),
    ]
    csv_path = next((p for p in csv_path_candidates if os.path.exists(p)), None)
    if csv_path is None:
        print(f"Could not find a readings CSV for {eye} in {trial_folder}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    row = df[df["Frame"] == frame_number]
    if row.empty:
        print(f"Frame {frame_number} not found in {csv_path}")
        sys.exit(1)
    row = row.iloc[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read frame {frame_number} from {video_path}")
        sys.exit(1)

    video_h, video_w = frame.shape[:2]
    print(f"Native frame size: {video_w}x{video_h}")

    # Also render what the detector actually saw for this exact frame,
    # for a side-by-side sanity check.
    detection_view = crop_to_aspect_ratio(frame.copy(), DETECTION_FRAME_W, DETECTION_FRAME_H)
    cx_det, cy_det = float(row["CenterX"]), float(row["CenterY"])
    maj, minr, angle = float(row["MajorDiameter"]), float(row["MinorDiameter"]), float(row["Angle"])
    cv2.ellipse(detection_view, (int(round(cx_det)), int(round(cy_det))),
                (int(round(maj / 2)), int(round(minr / 2))), angle, 0, 360, (55, 255, 0), 2)
    cv2.imwrite(f"debug_detection_view_{eye}_{frame_number}.png", detection_view)

    # Inverse transform onto the native frame.
    ox, oy, scale = compute_detection_to_video_transform(video_w, video_h)
    vx = ox + cx_det * scale
    vy = oy + cy_det * scale
    vmaj = maj * scale
    vminr = minr * scale
    print(f"offset=({ox:.2f},{oy:.2f}) scale={scale:.5f}")
    print(f"Detection-space center: ({cx_det:.1f},{cy_det:.1f})  ->  Video-space center: ({vx:.1f},{vy:.1f})")

    out = frame.copy()
    cv2.ellipse(out, (int(round(vx)), int(round(vy))),
                (max(1, int(round(vmaj / 2))), max(1, int(round(vminr / 2)))),
                angle, 0, 360, (55, 255, 0), 1)
    cv2.drawMarker(out, (int(round(vx)), int(round(vy))), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
    label = f"{eye} frame {frame_number}  center=({vx:.1f},{vy:.1f})"
    cv2.putText(out, label, (2, video_h - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1, cv2.LINE_AA)

    # Upscale 4x (nearest-neighbor-free) so a 108x138 frame is actually
    # visible/inspectable, without changing the underlying alignment.
    out_big = cv2.resize(out, (video_w * 4, video_h * 4), interpolation=cv2.INTER_NEAREST)
    out_path = f"debug_frame_{eye}_{frame_number}.png"
    cv2.imwrite(out_path, out_big)
    print(f"Wrote {out_path} (4x upscaled, native alignment)")
    print(f"Wrote debug_detection_view_{eye}_{frame_number}.png (what the detector saw)")


if __name__ == "__main__":
    main()