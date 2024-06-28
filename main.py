import cv2
import os
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from preprocessing import read_video, create_max_projection, preprocess_image
from circle_detection import detect_circles, filter_overlapping_circles
from roi_extraction import extract_and_save_rois

def resize_to_fit_screen(image, screen_width, screen_height):
    height, width = image.shape[:2]
    scale = min(screen_width / width, screen_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))

def analyze_frequencies(frequencies):
    total_samples = len(frequencies)
    cw_count = sum(
        1 for freq_list in frequencies if all(freq > 0 for freq in freq_list))
    ccw_count = sum(
        1 for freq_list in frequencies if all(freq < 0 for freq in freq_list))
    switch_count = total_samples - cw_count - ccw_count
    return total_samples, cw_count, ccw_count, switch_count

def main():
    root = Tk()
    root.withdraw()

    video_path = filedialog.askopenfilename(
        title="Select Video File", 
        filetypes=[("AVI files", "*.avi"), ("All files", "*.*")])
    if not video_path:
        print("No file selected. Exiting.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(
        os.path.dirname(video_path), f"{video_name} results")
    os.makedirs(output_folder, exist_ok=True)

    cap = read_video(video_path)
    max_projection = create_max_projection(cap)
    thresholded_image = preprocess_image(max_projection)

    circles = detect_circles(thresholded_image)
    filtered_circles = filter_overlapping_circles(circles)

    print(f"Number of detected circles: {len(filtered_circles)}")

    screen_width = 1920
    screen_height = 1080

    max_projection_resized = resize_to_fit_screen(
        max_projection, screen_width, screen_height)
    cv2.imwrite(
        os.path.join(output_folder, 'max_projection.png'), 
        max_projection_resized)

    thresholded_resized = resize_to_fit_screen(
        thresholded_image, screen_width, screen_height)
    cv2.imwrite(
        os.path.join(output_folder, 'thresholded_image.png'), 
        thresholded_resized)

    output_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in filtered_circles:
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)
    output_image_resized = resize_to_fit_screen(
        output_image, screen_width, screen_height)
    cv2.imwrite(
        os.path.join(output_folder, 'detected_circles.png'), 
        output_image_resized)

    # adjust frame_interval to decide how many frame are combined together for 
    # a frequency calculation, higher frame_interval will be more robust 
    # against Brownian noise and error in tracking, but require longer video
    # and may miss certain switching event.
    frame_interval=20
    roi_size=64

    frequencies, fps = extract_and_save_rois(
        video_path, filtered_circles, 
        output_folder, roi_size, frame_interval)

    plt.figure()
    for freq_list in frequencies:
        times = [(frame_interval / fps) * i for i in range(len(freq_list))]
        plt.scatter(times, freq_list)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Frequencies of Detected Spiners')
    plt.savefig(os.path.join(output_folder, 'frequencies.png'))
    plt.show()

    total_samples, cw_count, ccw_count, switch_count = analyze_frequencies(
        frequencies)
    with open(os.path.join(output_folder, 'frequency_analysis.txt'), 'w') as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"CW rotations: {cw_count}\n")
        f.write(f"CCW rotations: {ccw_count}\n")
        f.write(f"Switches in direction: {switch_count}\n")

if __name__ == "__main__":
    main()
