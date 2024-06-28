import cv2
import numpy as np

def resize_to_fit_screen(image, screen_width, screen_height):
    height, width = image.shape[:2]
    scale = min(screen_width / width, screen_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))

def is_overlapping(circle1, circle2):
    x1, y1, r1 = circle1
    x2, y2, r2 = circle2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance < r1 + r2

# Read the video file
video_path = '16_51_50.avi'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video successfully opened.")

# Initialize variables
ret, frame1 = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    cap.release()
else:
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

max_projection = np.zeros_like(frame1_gray, dtype=np.uint8)

# Process each frame to create max projection
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break
    
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    abs_diff = cv2.absdiff(frame1_gray, frame2_gray)
    
    # Update the max projection
    max_projection = np.maximum(max_projection, abs_diff)
    
    frame1_gray = frame2_gray

# Release the video capture object
cap.release()

# Invert the image
inverted_image = cv2.bitwise_not(max_projection)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(inverted_image, (15, 15), 0)

# Apply Canny edge detection with adjusted thresholds
edges = cv2.Canny(blurred_image, 30, 60)  # Adjusted thresholds

# Apply thresholding with adjusted threshold value
_, thresholded_image = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)  # Adjusted threshold

# Apply Hough Circle Transform
circles = cv2.HoughCircles(thresholded_image, cv2.HOUGH_GRADIENT, dp=0.5, minDist=5, param1=15, param2=15, minRadius=2, maxRadius=150)

# Ensure circles were found
output_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    
    # Filter out overlapping circles
    filtered_circles = []
    for circle in circles:
        if not any(is_overlapping(circle, existing_circle) for existing_circle in filtered_circles):
            filtered_circles.append(circle)
    
    for (x, y, r) in filtered_circles:
        # Draw the outer circle
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)
else:
    print("No circles were detected.")

# Print the number of detected circles
if circles is not None:
    print(f"Number of detected circles: {len(filtered_circles)}")
else:
    print("No circles were detected.")

# Get the screen resolution
screen_width = 1920  # Replace with your screen width
screen_height = 1080  # Replace with your screen height

# Resize the images to fit the screen
max_projection_resized = resize_to_fit_screen(max_projection, screen_width, screen_height)
inverted_resized = resize_to_fit_screen(inverted_image, screen_width, screen_height)
blurred_resized = resize_to_fit_screen(blurred_image, screen_width, screen_height)
edges_resized = resize_to_fit_screen(edges, screen_width, screen_height)
output_image_resized = resize_to_fit_screen(output_image, screen_width, screen_height)

# Save the results for inspection
cv2.imwrite('max_projection.png', max_projection)
cv2.imwrite('inverted_image.png', inverted_image)
cv2.imwrite('blurred_image.png', blurred_image)
cv2.imwrite('edges.png', edges)
cv2.imwrite('detected_circles.png', output_image)

# Display the results
cv2.imshow('Max Projection', max_projection_resized)
cv2.imshow('Inverted Image', inverted_resized)
cv2.imshow('Blurred Image', blurred_resized)
cv2.imshow('Edges', edges_resized)
cv2.imshow('Detected Circles', output_image_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract and save ROIs into new videos with fixed size 64x64
if circles is not None:
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_fps = original_fps // 2  # Slower frame rate, half of the original fps

    roi_size = 64

    for idx, (x, y, r) in enumerate(filtered_circles):
        # Define the fixed size ROI rectangle
        x1 = max(0, x - roi_size // 2)
        y1 = max(0, y - roi_size // 2)
        x2 = min(frame_width, x + roi_size // 2)
        y2 = min(frame_height, y + roi_size // 2)
        
        # Create a video writer for the ROI
        roi_video_path = f'roi_{idx+1}.avi'
        out = cv2.VideoWriter(roi_video_path, cv2.VideoWriter_fourcc(*'XVID'), new_fps, (roi_size, roi_size))

        # Reset the video capture to the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Extract the ROI from the frame
            roi = frame[y1:y2, x1:x2]
            # Ensure the ROI is the correct size by padding if necessary
            if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
                roi = cv2.copyMakeBorder(roi, 
                                         top=0, bottom=roi_size - roi.shape[0], 
                                         left=0, right=roi_size - roi.shape[1], 
                                         borderType=cv2.BORDER_CONSTANT, 
                                         value=[0, 0, 0])
            out.write(roi)
        
        out.release()

    cap.release()
    print("ROIs saved into new videos.")
else:
    print("No circles were detected to extract ROIs.")
