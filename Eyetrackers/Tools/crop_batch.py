import cv2
import os
from tkinter import Tk, filedialog

# -----------------------------
# Crop Settings
# -----------------------------

# Right eye
right_x = 118
right_y = 66
right_crop_width = 109
right_crop_height = 138

# Left eye
left_x = 115
left_y = 47
left_crop_width = 109
left_crop_height = 138

# -----------------------------
# Border Settings
# -----------------------------
border_size = 0                  # Border thickness in pixels
border_color = (255, 255, 255)   # White (B, G, R)

# Filenames to look for (case-insensitive) and their crop settings
TARGETS = {
    "left_eye.mp4": {
        "x": left_x, "y": left_y,
        "w": left_crop_width, "h": left_crop_height,
    },
    "right_eye.mp4": {
        "x": right_x, "y": right_y,
        "w": right_crop_width, "h": right_crop_height,
    },
}


def crop_video(input_file, x, y, crop_width, crop_height):
    base, ext = os.path.splitext(input_file)
    output_file = base + "_cropped" + ext

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"  Could not open video: {input_file}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, first_frame = cap.read()
    if not ret:
        print(f"  Could not read first frame: {input_file}")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_file,
        fourcc,
        fps,
        (crop_width, crop_height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped = frame[y:y + crop_height, x:x + crop_width].copy()

        # Draw a border INSIDE the cropped image.
        # The image size remains unchanged.
        if border_size > 0:
            cv2.rectangle(
                cropped,
                (0, 0),
                (crop_width - 1, crop_height - 1),
                border_color,
                border_size
            )

        out.write(cropped)

    cap.release()
    out.release()
    print(f"  Saved: {output_file}")


def main():
    root = Tk()
    root.withdraw()

    parent_dir = filedialog.askdirectory(
        title="Select Parent Folder"
    )

    if not parent_dir:
        print("No folder selected.")
        return

    matches = []
    for dirpath, _dirnames, filenames in os.walk(parent_dir):
        for filename in filenames:
            if filename.lower() in TARGETS:
                matches.append((dirpath, filename))

    if not matches:
        print("No left_eye.mp4 or right_eye.mp4 files found.")
        return

    print(f"Found {len(matches)} file(s) to process.\n")

    for dirpath, filename in matches:
        settings = TARGETS[filename.lower()]
        input_file = os.path.join(dirpath, filename)
        print(f"Processing: {input_file}")
        crop_video(
            input_file,
            settings["x"],
            settings["y"],
            settings["w"],
            settings["h"],
        )

    print("\nFinished!")


if __name__ == "__main__":
    main()