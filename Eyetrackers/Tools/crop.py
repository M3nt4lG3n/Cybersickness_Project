import cv2
import os
from tkinter import Tk, filedialog
# -----------------------------
# Crop Settings
# -----------------------------

# Right
x = 119
y = 79
crop_width = 109
crop_height = 138

# Left
# x = 130
# y = 70
# crop_width = 109
# crop_height = 138

# -----------------------------
# Border Settings
# -----------------------------
border_size = 0                  # Border thickness in pixels
border_color = (255, 255, 255)   # White (B, G, R)

# -----------------------------
# Select Video
# -----------------------------
root = Tk()
root.withdraw()

input_file = filedialog.askopenfilename(
    title="Select MP4 Video",
    filetypes=[("MP4 files", "*.mp4")]
)

if not input_file:
    print("No file selected.")
    quit()

base, ext = os.path.splitext(input_file)
output_file = base + "_cropped" + ext

# -----------------------------
# Open Video
# -----------------------------
cap = cv2.VideoCapture(input_file)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

fps = cap.get(cv2.CAP_PROP_FPS)

# Read first frame to verify the video
ret, first_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read first frame.")

# Reset back to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -----------------------------
# Video Writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    output_file,
    fourcc,
    fps,
    (crop_width, crop_height)
)

# -----------------------------
# Process Video
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Crop the frame
    cropped = frame[y:y + crop_height, x:x + crop_width].copy()

    # Draw a border INSIDE the cropped image.
    # The image size remains unchanged.
    cv2.rectangle(
        cropped,
        (0, 0),
        (crop_width - 1, crop_height - 1),
        border_color,
        border_size
    )

    # Write frame
    out.write(cropped)

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

print(f"Finished!\nSaved to:\n{output_file}")