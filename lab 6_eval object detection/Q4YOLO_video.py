import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model (automatically downloads if not present)
model = YOLO("yolov8n.pt")  # Smallest model for faster inference

# Set video source: 0 for webcam or provide video file path
video_source = 0  # Change to "video.mp4" for a recorded video

# Open video capture
cap = cv2.VideoCapture(video_source)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

prev_time = 0  # Initialize time for FPS calculation

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is read

    # Run YOLO detection
    results = model(frame, stream=True)  # Stream mode for efficiency

    # Annotate frame with detected objects
    for r in results:
        frame = r.plot()  # Draws bounding boxes on the frame

    # Compute FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Display FPS on video
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
