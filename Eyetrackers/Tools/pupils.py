import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

import patient_config as pc

# These start out as the shared defaults and get overwritten with a
# patient/side's saved values (or left at defaults for a brand-new
# patient/side) as soon as a video is selected. They are then further
# adjustable at runtime with the on-screen +/- controls.
RELAXED_THRESHOLD = pc.DEFAULT_PUPIL["RELAXED_THRESHOLD"]
MEDIUM_THRESHOLD = pc.DEFAULT_PUPIL["MEDIUM_THRESHOLD"]
STRICT_THRESHOLD = pc.DEFAULT_PUPIL["STRICT_THRESHOLD"]
SQUARE_SIZE = pc.DEFAULT_PUPIL["SQUARE_SIZE"]

# =========================================================================
# Detection logic below is unmodified from the original pupils.py.
# =========================================================================

# Crop the image to maintain a specific aspect ratio (width:height) before resizing. 
def crop_to_aspect_ratio(image, width=640, height=480):
    
    # Calculate current aspect ratio
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]

    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

#Finds a square area of dark pixels in the image
#@param I input image (converted to grayscale during search process)
#@return a point within the pupil region
def get_darkest_area(image):

    ignoreBounds = 20 #don't search the boundaries of the image for ignoreBounds pixels
    imageSkipSize = 10 #only check the darkness of a block for every Nth x and y pixel (sparse sampling)
    searchArea = 20 #the size of the block to search
    internalSkipSize = 3 #skip every Nth x and y pixel in the local search area (sparse sampling)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    min_sum = float('inf')
    darkest_point = None

    # Loop over the image with spacing defined by imageSkipSize, ignoring the boundaries
    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Calculate sum of pixel values in the search area, skipping pixels based on internalSkipSize
            current_sum = np.int64(0)
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            # Update the darkest point if the current block is darker
            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)  # Center of the block

    return darkest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

#returns the largest contour that is not extremely long or tall
#contours is the list of contours, pixel_thresh is the max pixels to filter, and ratio_thresh is the max ratio
def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length = max(w, h)
            width = min(w, h)

            # Calculate the length-to-width ratio and width-to-length ratio
            length_to_width_ratio = length / width
            width_to_length_ratio = width / length

            # Pick the higher of the two ratios
            current_ratio = max(length_to_width_ratio, width_to_length_ratio)

            # Check if highest ratio is within the acceptable threshold
            if current_ratio <= ratio_thresh:
                # Update the largest contour if the current one is bigger
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    # Return a list with only the largest contour, or an empty list if no contour was found
    if largest_contour is not None:
        return [largest_contour]
    else:
        return []

#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
  
    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]
    final_goodness = 0
    
    #iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        # Dilate the binary image
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)#medium
        
        # Find contours
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            #gray_copy = gray_frame.copy()
            #cv2.drawContours(gray_copies[i-1], reduced_contours, -1, (255), 1)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            final_goodness = current_goodness[0]*total_pixels[0]*total_pixels[0]*total_pixels[1]
            
            #show intermediary images with text output
            if debug_mode_on:
                cv2.putText(gray_copies[i-1], "%filled:     " + str(current_goodness[0])[:5] + " (percentage of filled contour pixels inside ellipse)", (10,30), font, .55, (255,255,255), 1) #%filled
                cv2.putText(gray_copies[i-1], "abs. pix:   " + str(total_pixels[0]) + " (total pixels under fit ellipse)", (10,50), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "pix ratio:  " + str(total_pixels[1]) + " (total pix under fit ellipse / contour border pix)", (10,70), font, .55, (255,255,255), 1    ) #abs pix
                cv2.putText(gray_copies[i-1], "final:     " + str(final_goodness) + " (filled*ratio)", (10,90), font, .55, (255,255,255), 1) #skewedness
                cv2.imshow(name_array[i-1] + " threshold", image_array[i-1])
                cv2.imshow(name_array[i-1], gray_copies[i-1])

        if final_goodness > 0 and final_goodness > goodness: 
            goodness = final_goodness
            ellipse_reduced_contours = total_pixels[2]
            best_image = image_array[i-1]
            final_contours = reduced_contours
            final_image = dilated_image
    
    if debug_mode_on:
        cv2.imshow("Reduced contours of best thresholded image", ellipse_reduced_contours)

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0] > 5):
        #cv2.drawContours(test_frame, final_contours, -1, (255, 255, 255), 1)
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
        #cv2.circle(test_frame, darkest_point, 3, (255, 125, 125), -1)
        center_x, center_y = map(int, ellipse[0])
        cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
        cv2.putText(test_frame, "SPACE = play/pause", (10,410), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #space
        cv2.putText(test_frame, "Q      = quit", (10,430), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #quit
        cv2.putText(test_frame, "D      = show debug", (10,450), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,90,30), 2) #debug

    if render_cv_window:
        cv2.imshow('best_thresholded_image_contours_on_frame', test_frame)
    
    # Create an empty image to draw contours
    contour_img3 = np.zeros_like(image_array[i-1])
    
    if len(final_contours[0]) >= 5:
        contour = np.array(final_contours[0], dtype=np.int32).reshape((-1, 1, 2)) #format for cv2.fitEllipse
        ellipse = cv2.fitEllipse(contour) # Fit ellipse
        cv2.ellipse(gray_frame, ellipse, (255,255,255), 2)  # Draw with white color and thickness of 2

    #process_frames now returns a rotated rectangle for the ellipse for easy access
    return final_rotated_rect


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, STRICT_THRESHOLD)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, SQUARE_SIZE)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, MEDIUM_THRESHOLD)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, SQUARE_SIZE)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, RELAXED_THRESHOLD)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, SQUARE_SIZE)
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
    return final_rotated_rect

# =========================================================================
# Everything below this line is the interactive driver -- this is what
# changed. The detection functions above are untouched.
# =========================================================================

CONTROL_WINDOW = "Pupil Detection Controls"

PANEL_W = 560
ROW_H = 46
TOP_PAD = 55
BOTTOM_PAD = 55

# (display label / global variable name, min, max)
CONTROL_ROWS = [
    ("RELAXED_THRESHOLD", 0, 255),
    ("MEDIUM_THRESHOLD", 0, 255),
    ("STRICT_THRESHOLD", 0, 255),
    ("SQUARE_SIZE", 10, 640),
]


class PupilTuner:
    """Drives a video frame-by-frame through the (unmodified) detection
    pipeline above, with a slider to rewind/scrub and on-screen +/-
    controls to adjust the threshold / square-size globals live."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video.")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_count <= 0:
            raise RuntimeError("Could not determine frame count for video.")

        self.idx = 0
        self.debug_mode_on = False
        self.paused = False
        self._suppress_trackbar_cb = False

        self.panel_h = TOP_PAD + ROW_H * len(CONTROL_ROWS) + BOTTOM_PAD

        self._open_control_window()

    def _open_control_window(self):
        cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Frame", CONTROL_WINDOW, self.idx, max(1, self.frame_count - 1), self._on_trackbar)
        cv2.setMouseCallback(CONTROL_WINDOW, self._on_mouse)

    def _on_trackbar(self, val):
        if self._suppress_trackbar_cb:
            return
        self.idx = val
        self.paused = True  # scrubbing pauses playback

    def _set_idx(self, new_idx):
        new_idx = max(0, min(new_idx, self.frame_count - 1))
        self.idx = new_idx
        self._suppress_trackbar_cb = True
        cv2.setTrackbarPos("Frame", CONTROL_WINDOW, new_idx)
        self._suppress_trackbar_cb = False

    def _button_rects(self):
        rects = {}
        for i, (varname, lo, hi) in enumerate(CONTROL_ROWS):
            y0 = TOP_PAD + i * ROW_H
            rects[varname] = {
                "down": (20, y0, 60, y0 + 32),
                "up": (90, y0, 130, y0 + 32),
                "lo": lo,
                "hi": hi,
            }
        return rects

    def _on_mouse(self, event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for varname, r in self._button_rects().items():
            for direction in ("down", "up"):
                x0, y0, x1, y1 = r[direction]
                if x0 <= mx <= x1 and y0 <= my <= y1:
                    delta = -1 if direction == "down" else 1
                    self._adjust(varname, delta, r["lo"], r["hi"])
                    return

    def _adjust(self, varname, delta, lo, hi):
        global RELAXED_THRESHOLD, MEDIUM_THRESHOLD, STRICT_THRESHOLD, SQUARE_SIZE
        current = globals()[varname]
        globals()[varname] = max(lo, min(hi, current + delta))

    def _draw_panel(self):
        panel = np.full((self.panel_h, PANEL_W, 3), 40, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, f"Frame {self.idx}/{self.frame_count - 1}"
                            f"  ({'paused' if self.paused else 'playing'})",
                    (20, 30), font, 0.6, (0, 255, 255), 2)

        for varname, r in self._button_rects().items():
            y0 = r["down"][1]
            value = globals()[varname]

            cv2.rectangle(panel, r["down"][0:2], r["down"][2:4], (70, 70, 220), -1)
            cv2.putText(panel, "-", (r["down"][0] + 14, r["down"][3] - 7), font, 0.9, (255, 255, 255), 2)

            cv2.rectangle(panel, r["up"][0:2], r["up"][2:4], (70, 190, 70), -1)
            cv2.putText(panel, "+", (r["up"][0] + 11, r["up"][3] - 7), font, 0.9, (255, 255, 255), 2)

            cv2.putText(panel, f"{varname}: {value}", (145, y0 + 24), font, 0.6, (255, 255, 255), 1)

        instr_y = TOP_PAD + ROW_H * len(CONTROL_ROWS) + 28
        cv2.putText(panel, "SPACE=play/pause  Left/Right=step  D=debug  Q/Enter/Esc=finish",
                    (20, instr_y), font, 0.48, (200, 200, 200), 1)
        return panel

    def _get_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def run(self):
        last_idx = -1
        frame = None

        while True:
            if self.idx != last_idx:
                new_frame = self._get_frame(self.idx)
                if new_frame is None:
                    self._set_idx(max(0, self.idx - 1))
                    continue
                frame = new_frame
                last_idx = self.idx

            proc_frame = crop_to_aspect_ratio(frame)
            darkest_point = get_darkest_area(proc_frame)
            gray_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
            darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

            thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, STRICT_THRESHOLD)
            thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, SQUARE_SIZE)

            thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, MEDIUM_THRESHOLD)
            thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, SQUARE_SIZE)

            thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, RELAXED_THRESHOLD)
            thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, SQUARE_SIZE)

            process_frames(
                thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed,
                proc_frame, gray_frame, darkest_point, self.debug_mode_on, True
            )

            cv2.imshow(CONTROL_WINDOW, self._draw_panel())

            key = cv2.waitKeyEx(30)
            raw_key = (key & 0xFF) if key != -1 else -1

            if raw_key in (ord('q'), 13, 27):  # q, Enter, Esc -> finish
                break
            elif raw_key == ord('d'):
                self.debug_mode_on = not self.debug_mode_on
                if not self.debug_mode_on:
                    cv2.destroyAllWindows()
                    self._open_control_window()
            elif raw_key == ord(' '):
                self.paused = not self.paused
            elif key in (2424832, 65361, 63234):  # Left arrow
                self.paused = True
                self._set_idx(self.idx - 1)
            elif key in (2555904, 65363, 63235):  # Right arrow
                self.paused = True
                self._set_idx(self.idx + 1)
            elif key == -1 and not self.paused:
                if self.idx >= self.frame_count - 1:
                    break  # reached the end of the video naturally
                self._set_idx(self.idx + 1)

        cv2.destroyAllWindows()


def run_on_video(video_path):
    global RELAXED_THRESHOLD, MEDIUM_THRESHOLD, STRICT_THRESHOLD, SQUARE_SIZE

    patient_id = pc.find_patient_id(video_path)
    side = pc.get_eye_side(video_path)

    config = pc.load_config()
    if patient_id and side:
        pupil_values = pc.get_pupil_values(config, patient_id, side)
        RELAXED_THRESHOLD = pupil_values["RELAXED_THRESHOLD"]
        MEDIUM_THRESHOLD = pupil_values["MEDIUM_THRESHOLD"]
        STRICT_THRESHOLD = pupil_values["STRICT_THRESHOLD"]
        SQUARE_SIZE = pupil_values["SQUARE_SIZE"]
    else:
        print("Could not determine Patient_X / eye side from this file's path -- "
              "starting from default threshold values (these won't be saveable).")

    tuner = PupilTuner(video_path)
    tuner.run()
    tuner.cap.release()

    print(f"Final values -- RELAXED_THRESHOLD={RELAXED_THRESHOLD}, "
          f"MEDIUM_THRESHOLD={MEDIUM_THRESHOLD}, STRICT_THRESHOLD={STRICT_THRESHOLD}, "
          f"SQUARE_SIZE={SQUARE_SIZE}")

    if patient_id is None or side is None:
        pc.show_error(
            "Not Saved",
            "Could not determine a Patient_<number> folder and/or eye side "
            "(left_eye / right_eye) from this file's path, so these values "
            "were not saved."
        )
        return

    if not pc.ask_yes_no(
        "Save Threshold Values?",
        f"Save these pupil-detection parameters for {patient_id} ({side} eye)?\n\n"
        f"RELAXED_THRESHOLD = {RELAXED_THRESHOLD}\n"
        f"MEDIUM_THRESHOLD = {MEDIUM_THRESHOLD}\n"
        f"STRICT_THRESHOLD = {STRICT_THRESHOLD}\n"
        f"SQUARE_SIZE = {SQUARE_SIZE}"
    ):
        print("Not saved.")
        return

    values = {
        "RELAXED_THRESHOLD": RELAXED_THRESHOLD,
        "MEDIUM_THRESHOLD": MEDIUM_THRESHOLD,
        "STRICT_THRESHOLD": STRICT_THRESHOLD,
        "SQUARE_SIZE": SQUARE_SIZE,
    }
    config = pc.update_side(config, patient_id, side, values)
    pc.save_config(config)
    print(f"Saved pupil-detection parameters for {patient_id} ({side} eye) to {pc.CONFIG_PATH}")

    if pc.ask_yes_no(
        "Run Batch Detection?",
        f"Run pupil detection on the rest of {patient_id}'s eye videos "
        f"using these saved parameters?"
    ):
        patient_dir = pc.find_patient_dir(video_path)
        print(f"Running pupil-detection batch for {patient_id} ...")
        try:
            import pupils_batch
        except ImportError as e:
            pc.show_error("Could Not Start Batch", f"Could not load pupils_batch.py:\n{e}")
            return
        pupils_batch.run_batch_for_patient(patient_dir)
    else:
        print("Batch detection not run.")


#Prompts the user to select a single (already-cropped) video file.
def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Cropped Eye Video File",
        filetypes=[("Video Files", "*.mp4;*.avi")]
    )
    root.destroy()
    if not video_path:
        print("No file selected. Exiting.")
        return

    run_on_video(video_path)


if __name__ == "__main__":
    select_video()