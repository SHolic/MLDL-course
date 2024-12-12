"""
YOLOv8 
Some portions of the code in this example were generated using ChatGPT-4o

You may need to "pip install" these packages:
pip install opencv-python ultralytics

"""

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (sizes): yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt 
model = YOLO('yolov8n.pt')  # Use a smaller model for speed, or larger for better accuracy

# Open the webcam or use a video file
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' to use a video file

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 detection
    results = model(frame)

    # Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            color = (0, 255, 0)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
