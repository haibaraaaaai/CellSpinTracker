import cv2
import numpy as np

def detect_circles(thresholded_image):
    circles = cv2.HoughCircles(
        thresholded_image, cv2.HOUGH_GRADIENT, dp=0.5, 
        minDist=5, param1=15, param2=15, minRadius=2, maxRadius=150)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles

def filter_overlapping_circles(circles):
    def is_overlapping(circle1, circle2):
        x1, y1, r1 = circle1
        x2, y2, r2 = circle2
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance < r1 + r2

    if circles is None:
        return []

    filtered_circles = []
    for circle in circles:
        if not any(
            is_overlapping(circle, existing_circle) 
            for existing_circle in filtered_circles):
            filtered_circles.append(circle)
    return filtered_circles
