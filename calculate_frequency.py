import cv2
import numpy as np
import math
import os

def process_frame(frame, roi_size=64):
    center = (roi_size // 2, roi_size // 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(
        thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None

    contours = [
        contour for contour in contours 
        if 50 < cv2.contourArea(contour) < 15000]

    selected_contour = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            selected_contour = contour
            break
    
    if selected_contour is None:
        return None, None

    max_distance = 0
    furthest_point = None
    for point in selected_contour:
        point = point[0]
        distance = np.linalg.norm(np.array(point) - np.array(center))
        if distance > max_distance:
            max_distance = distance
            furthest_point = point

    return selected_contour, tuple(furthest_point)

def calculate_freq(prev_point, curr_point):
    direction = np.arctan2(
        curr_point[1] - prev_point[1], curr_point[0] - prev_point[0])
    return direction

def calculate_frequency(video_path, roi_size=64, frame_interval=20, output_fps=None):
    cap = cv2.VideoCapture(video_path)
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = output_fps if output_fps else original_fps

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(os.path.dirname(video_path), f"{video_name}_tracked.avi")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    frame_count = 0
    prev_point = None
    frequencies = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read any valid frames.")
            cap.release()
            out.release()
            return []

        selected_contour, furthest_point = process_frame(frame, roi_size)
        if furthest_point is not None:
            prev_point = furthest_point
            break

    if prev_point is None:
        print("No valid contour found in any frame.")
        cap.release()
        out.release()
        return []

    total_angle_change = 0
    grouped_frequencies = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        selected_contour, furthest_point = process_frame(frame, roi_size)
        if furthest_point is not None:
            direction = calculate_freq(prev_point, furthest_point)
            
            total_angle_change += direction
            frequencies.append(direction / (2 * math.pi) * fps)

            prev_point = furthest_point

            if selected_contour is not None:
                cv2.drawContours(frame, [selected_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, furthest_point, 5, (0, 0, 255), -1)

        out.write(frame)
        frame_count += 1

        if frame_count % frame_interval == 0:
            time_interval = frame_interval / fps
            frequency = total_angle_change / (2 * math.pi) / time_interval
            grouped_frequencies.append(frequency)
            total_angle_change = 0

    if frame_count % frame_interval != 0 and total_angle_change != 0:
        time_interval = (frame_count % frame_interval) / fps
        frequency = total_angle_change / (2 * math.pi) / time_interval
        grouped_frequencies.append(frequency)

    cap.release()
    out.release()

    frequency_file_path = os.path.join(os.path.dirname(video_path), f"{video_name}_frequencies.txt")
    with open(frequency_file_path, 'w') as f:
        for freq in frequencies:
            f.write(f"{freq}\n")

    return grouped_frequencies
