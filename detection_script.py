import cv2
import math
import json
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('best.pt')

# Open video capture from a file or webcam (change the source as needed)
input_video_path = 'input.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Dictionary to store detections
detections = {
    "Pepsi": [],
    "CocaCola": []
}

# Initialize frame number
frame_number = 0

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Process each detection
    for result in results:
        for bbox, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if int(cls) < 2:  # Assuming Pepsi and CocaCola are classes 0 and 1
                x1, y1, x2, y2 = map(int, bbox)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_size = bbox_width * bbox_height
                bbox_center_x = x1 + bbox_width // 2
                bbox_center_y = y1 + bbox_height // 2
                # Calculate distance (if needed)
                frame_center_x = width // 2
                frame_center_y = height // 2
                distance = math.sqrt((bbox_center_x - frame_center_x) ** 2 + (bbox_center_y - frame_center_y) ** 2)
                detection_info = {
                    "timestamp": frame_number / fps,
                    "size": bbox_size,
                    "distance": distance
                }
                if int(cls) == 0:
                    detections["Pepsi"].append(detection_info)
                elif int(cls) == 1:
                    detections["CocaCola"].append(detection_info)

    # Display the frame
    cv2.imshow('Real-Time Detection', frame)

    # Exit key (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame number
    frame_number += 1

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Save detections to a JSON file
json_output_path = 'detections.json'
with open(json_output_path, 'w') as json_file:
    json.dump(detections, json_file, indent=4)

print(f"Real-time video processing completed.")
print(f"Detections saved to {json_output_path}")
