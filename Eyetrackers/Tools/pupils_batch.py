import cv2
import numpy as np
import random
import math
import tkinter as tk
import os
import sys
from pathlib import Path
from tkinter import filedialog
import matplotlib.pyplot as plt
import csv

# merger.py (Eyetrackers/Data_Processing/merger.py) already owns the logic
# for turning a pupil CSV + eye CSV into the "readings" CSVs
# (build_eye_readings -> left_eye_readings.csv / right_eye_readings.csv).
# It's imported here rather than reimplemented so both stay in sync.
# Assumes merger.py sits alongside this script; adjust this path if not.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Eyetrackers.Data_Processing.merger import build_eye_readings, MergeInputPaths

import patient_config as pc

# -----------------------------
# Pupil Detection Configuration Settings
# -----------------------------
# No longer hard-coded: these are set per-video, from that video's
# patient/eye-side entry in patient_config.json, just before each call
# to process_video() (see _apply_pupil_globals() / run_batch_for_patient()
# below). They start out at the shared defaults purely so the module
# still imports cleanly and the detection functions below always have
# *something* to read.
RELAXED_THRESHOLD = pc.DEFAULT_PUPIL["RELAXED_THRESHOLD"]
MEDIUM_THRESHOLD = pc.DEFAULT_PUPIL["MEDIUM_THRESHOLD"]
STRICT_THRESHOLD = pc.DEFAULT_PUPIL["STRICT_THRESHOLD"]
SQUARE_SIZE = pc.DEFAULT_PUPIL["SQUARE_SIZE"]

# -----------------------------
# Blinking Detection Tolerance
# -----------------------------
CENTER_TOLERANCE = 0.40  # Percentage value 0.0 - 1.0; Lower values are more sensitive

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
        # Reset per-iteration so a rejected candidate can't accidentally
        # inherit a previous iteration's final_goodness value.
        final_goodness = 0

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

            # cv2.fitEllipse() does a least-squares fit assuming the input
            # points trace a full ellipse perimeter. If reduced_contours[0]
            # is actually a thin curved arc (e.g. the eyelid crease, not a
            # closed pupil boundary) rather than a filled pupil blob,
            # fitEllipse can extrapolate a center point well outside the
            # masked search region around darkest_point -- even though
            # every contour pixel it was fit from lies inside that region.
            # A real pupil fit should stay within the square we searched.
            ellipse_center_dist = math.hypot(ellipse[0][0] - darkest_point[0],
                                              ellipse[0][1] - darkest_point[1])
            fitted_outside_search_region = ellipse_center_dist > (SQUARE_SIZE / 2.0)

            if debug_mode_on: #show contours 
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])
                
            #in total pixels, first element is pixel total, next is ratio
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)                 
            
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)  # Draw with specified color and thickness of 2
            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            
            if fitted_outside_search_region:
                final_goodness = 0
                if debug_mode_on:
                    print(f"  [{name_array[i-1]}] rejected: fitted ellipse center "
                          f"{ellipse_center_dist:.1f}px from darkest_point "
                          f"(> {SQUARE_SIZE/2.0:.0f}px search radius) -- likely a "
                          f"partial arc, not the pupil")
            else:
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

# Loads a video and finds the pupil in each frame
# video_path: path to the input .mp4
# output_csv_path: full path (including filename) to write the per-frame pupil CSV to
# input_method: 1 for video file, 2 for webcam
# show_window: whether to show the live cv2 preview window
# save_annotated_video: whether to also save an annotated debug .mp4 alongside the CSV
def process_video(video_path, output_csv_path, input_method=1, show_window=True, save_annotated_video=False):

    out = None
    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        annotated_path = os.path.join(os.path.dirname(output_csv_path), "annotated_output.mp4")
        out = cv2.VideoWriter(annotated_path, fourcc, 30.0, (640, 480))  # Output video filename, codec, frame rate, and frame size

    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(00, cv2.CAP_DSHOW)  # Camera input
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    else:
        print("Invalid video source.")
        return

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    debug_mode_on = False
    
    temp_center = (0,0)

    csv_file = open(output_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "Frame",
        "Time_ms",
        "CenterX",
        "CenterY",
        "MajorDiameter",
        "MinorDiameter",
        "Area",
        "Angle",
        "isEyeClosed"
    ])

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop and resize frame
        frame = crop_to_aspect_ratio(frame)

        #find the darkest point
        darkest_point = get_darkest_area(frame)

        # Eyes-closed detection: when the eye is closed, the darkest region in
        # frame is usually the eyelashes/eyelid margin rather than the pupil,
        # which sits away from the center of the frame. Reuses SQUARE_SIZE -
        # the same constant already used elsewhere as the expected pupil
        # region size - as the "near the middle" distance threshold, and the
        # frame's own (already cropped/resized) dimensions for the center.
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        distance_from_center = math.hypot(darkest_point[0] - frame_center_x, darkest_point[1] - frame_center_y)
        # Maximum possible distance from center to a corner
        max_distance = math.hypot(frame_width / 2, frame_height / 2)

        # Eye is considered closed if the darkest point is more than
        # 40% of the way from the center toward a corner.
        is_eye_closed = distance_from_center > (CENTER_TOLERANCE * max_distance)

        if debug_mode_on:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)

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
        pupil_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, show_window)

        ((cx, cy), (w, h), angle) = pupil_rotated_rect

        time_ms = frame_number * 1000.0 / fps

        # Frames with no fittable ellipse (commonly closed-eye frames) still
        # get a row now, so isEyeClosed is never silently dropped - the
        # ellipse fields are just left blank when there's nothing to report.
        if w > 0 and h > 0:
            area = math.pi * (w / 2) * (h / 2)
            csv_writer.writerow([
                frame_number,
                round(time_ms, 3),
                round(cx, 2),
                round(cy, 2),
                round(w, 2),
                round(h, 2),
                round(area, 2),
                round(angle, 2),
                is_eye_closed
            ])
        else:
            csv_writer.writerow([
                frame_number,
                round(time_ms, 3),
                "",
                "",
                "",
                "",
                "",
                "",
                is_eye_closed
            ])

        frame_number += 1
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d') and debug_mode_on == False:  # Press 'q' to start debug mode
            debug_mode_on = True
        elif key == ord('d') and debug_mode_on == True:
            debug_mode_on = False
            cv2.destroyAllWindows()
        if key == ord('q'):  # Press 'q' to quit
            if out is not None:
                out.release()
            break   
        elif key == ord(' '):  # Press spacebar to start/stop
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Press spacebar again to resume
                    break
                elif key == ord('q'):  # Press 'q' to quit
                    break

    csv_file.close()
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    print(f"  -> saved {output_csv_path}")


#Prompts the user to select a single video file.
#Kept around for quick manual debugging of one video at a time.
def select_video():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
    if not video_path:
        print("No file selected. Exiting.")
        return

    csv_path = os.path.join(os.path.dirname(video_path), "pupil.csv")
    #second parameter is 1 for video 2 for webcam
    process_video(video_path, csv_path, input_method=1)


# Prompts for a top-level "Patient" directory, then walks its subfolders
# looking for left_eye_cropped.mp4 / right_eye_cropped.mp4.
# Returns a list of (video_path, output_csv_path, eye_label) tuples.
def find_patient_eye_videos(patient_dir):
    jobs = []
    for root, dirs, files in os.walk(patient_dir):
        if os.path.normpath(root) == os.path.normpath(patient_dir):
            continue  # only look inside subfolders, not loose files in the patient dir itself

        for fname in files:
            lower = fname.lower()
            if lower == "left_eye_cropped.mp4":
                jobs.append((os.path.join(root, fname), os.path.join(root, "left_pupil.csv"), "left"))
            elif lower == "right_eye_cropped.mp4":
                jobs.append((os.path.join(root, fname), os.path.join(root, "right_pupil.csv"), "right"))

    return jobs


def select_patient_directory():
    root = tk.Tk()
    root.withdraw()
    patient_dir = filedialog.askdirectory(title="Select Patient Directory")
    root.destroy()
    return patient_dir


# Copies one eye side's saved pupil-detection parameters from
# patient_config.json onto the module-level globals that the (unmodified)
# detection functions above read at call time.
def _apply_pupil_globals(side_cfg):
    global RELAXED_THRESHOLD, MEDIUM_THRESHOLD, STRICT_THRESHOLD, SQUARE_SIZE
    RELAXED_THRESHOLD = side_cfg["RELAXED_THRESHOLD"]
    MEDIUM_THRESHOLD = side_cfg["MEDIUM_THRESHOLD"]
    STRICT_THRESHOLD = side_cfg["STRICT_THRESHOLD"]
    SQUARE_SIZE = side_cfg["SQUARE_SIZE"]


# Reuses merger.py's build_eye_readings() to (re)generate a session
# folder's left_eye_readings.csv / right_eye_readings.csv - referred to
# elsewhere as the "left/right pupil readings" files - now that this
# script has (re)written its left_pupil.csv / right_pupil.csv.
#
# build_eye_readings() merges each side's *_pupil.csv against its
# matching *_eye.csv (expected to already exist in the same folder,
# written by the eye-tracker's own eye-video pipeline) and writes the
# merged result back out to that same folder. A side is silently skipped
# if either of its input files is missing.
def build_pupil_readings_for_session(session_dir):
    session_dir = Path(session_dir)

    # MergeInputPaths requires timestamped_csv/analysis_csv/beats_csv,
    # but build_eye_readings() never reads them - only the
    # *_pupil_csv / *_eye_csv paths it derives from input_dir matter
    # here. These are unused placeholders to satisfy the dataclass.
    merge_paths = MergeInputPaths(
        input_dir=session_dir,
        timestamped_csv=session_dir / "_unused_timestamped.csv",
        analysis_csv=session_dir / "_unused_analysis.csv",
        beats_csv=session_dir / "_unused_beats.csv",
    )

    print(f"  Updating pupil readings files in: {session_dir}")
    return build_eye_readings(merge_paths, output_directory=session_dir)


# Runs every discovered video through the (unmodified) pupil-detection pipeline
# and writes each result to left_pupil.csv / right_pupil.csv next to its video.
#
# Videos are processed one at a time, in sequence, rather than in parallel.
# This is deliberate for accuracy/reliability, not just simplicity:
#   - OpenCV already parallelizes its own per-frame ops (dilate, cvtColor, etc.)
#     internally across CPU cores/threads. Running several videos at once with
#     Python threads or multiprocessing would have those internal thread pools
#     compete with each other for the same cores, which tends to slow things
#     down rather than speed them up, and can make per-video timing unpredictable.
#   - The pipeline opens live cv2.imshow debug windows and reads keyboard input
#     via cv2.waitKey() inside the loop. OpenCV's HighGUI window/event handling
#     is not safe to run from multiple threads/processes simultaneously, so
#     doing that risks window conflicts.
# A strict queue keeps every video's frame-by-frame detection behaving exactly
# like running this script once per video by hand.
#
# Uses that patient's saved parameters from patient_config.json instead of
# hard-coded constants. Can be called directly (e.g. from pupils.py) with an
# already-known patient_dir, skipping the folder-selection dialog.
def run_batch_for_patient(patient_dir, show_window=True):
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

    jobs = find_patient_eye_videos(patient_dir)

    if not jobs:
        print(f"No left_eye_cropped.mp4 or right_eye_cropped.mp4 files found in subfolders of: {patient_dir}")
        return

    config = pc.load_config()
    needed_sides = sorted({eye_label for _, _, eye_label in jobs})
    missing = [s for s in needed_sides if not pc.has_pupil_params(pc.get_side(config, patient_id, s))]

    if missing:
        pc.show_error(
            "Missing Pupil-Detection Parameters",
            f"No saved pupil-detection parameters for {patient_id} ({', '.join(missing)} eye).\n\n"
            f"Please set the parameters using pupils.py first."
        )
        return

    print(f"Found {len(jobs)} video(s) to process for {patient_id}:")
    for video_path, csv_path, eye_label in jobs:
        print(f"  [{eye_label}] {video_path}")

    for video_path, csv_path, eye_label in jobs:
        side_cfg = pc.get_side(config, patient_id, eye_label)
        _apply_pupil_globals(side_cfg)
        print(f"\nProcessing {eye_label} eye video: {video_path}")
        process_video(video_path, csv_path, input_method=1, show_window=show_window)

    # Now that every session folder's left_pupil.csv / right_pupil.csv
    # reflects this run, refresh that folder's pupil readings CSVs too.
    session_dirs = sorted({os.path.dirname(csv_path) for _, csv_path, _ in jobs})
    print(f"\nUpdating pupil readings files for {len(session_dirs)} session folder(s)...")
    for session_dir in session_dirs:
        build_pupil_readings_for_session(session_dir)

    print("\nDone. Processed all videos.")


def run_batch(show_window=True):
    patient_dir = select_patient_directory()
    if not patient_dir:
        print("No directory selected. Exiting.")
        return

    run_batch_for_patient(patient_dir, show_window=show_window)


if __name__ == "__main__":
    run_batch()