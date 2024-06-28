import cv2
import numpy as np

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    return cap

def create_max_projection(cap):
    ret, frame1 = cap.read()
    if not ret:
        raise ValueError("Error: Could not read the first frame.")
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    max_projection = np.zeros_like(frame1_gray, dtype=np.uint8)

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        abs_diff = cv2.absdiff(frame1_gray, frame2_gray)
        max_projection = np.maximum(max_projection, abs_diff)
        frame1_gray = frame2_gray

    cap.release()
    return max_projection

def preprocess_image(max_projection):
    inverted_image = cv2.bitwise_not(max_projection)
    blurred_image = cv2.GaussianBlur(inverted_image, (15, 15), 0)
    edges = cv2.Canny(blurred_image, 30, 60)
    _, thresholded_image = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    return thresholded_image
