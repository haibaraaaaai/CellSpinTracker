import cv2
import os
from calculate_frequency import calculate_frequency

def extract_and_save_rois(
        video_path, circles, output_folder,
          roi_size=64, frame_interval=20, output_fps=None):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps = output_fps if output_fps else original_fps

    roi_frequencies = []

    for idx, (x, y, _) in enumerate(circles):
        x1 = max(0, x - roi_size // 2)
        y1 = max(0, y - roi_size // 2)
        x2 = min(frame_width, x + roi_size // 2)
        y2 = min(frame_height, y + roi_size // 2)

        roi_video_path = os.path.join(output_folder, f'roi_{idx+1}.avi')
        out = cv2.VideoWriter(
            roi_video_path, cv2.VideoWriter_fourcc(*'XVID'), 
            fps, (roi_size, roi_size))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] < roi_size or roi.shape[1] < roi_size:
                roi = cv2.copyMakeBorder(
                    roi, top=0, bottom=roi_size - roi.shape[0], 
                    left=0, right=roi_size - roi.shape[1], 
                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
            out.write(roi)
        out.release()

        frequencies = calculate_frequency(
            roi_video_path, roi_size, frame_interval, fps)
        roi_frequencies.append(frequencies)

    cap.release()
    print("ROIs saved into new videos.")
    return roi_frequencies, fps
