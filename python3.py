import cv2
from ultralytics import YOLO
import math
import json

model = YOLO('best.pt')


input_video_path = 'input.mp4'
output_video_path = 'video.mp4'

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

detections = {
    "Pepsi": [],
    "CocaCola": []
}

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = round(frame_number / fps, 2)


    frame_center_x = width // 2
    frame_center_y = height // 2

    results = model(frame)

    for result in results:
        for bbox, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if int(cls) < 2:  
                x1, y1, x2, y2 = map(int, bbox)
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_size = bbox_width * bbox_height
                bbox_center_x = x1 + bbox_width // 2
                bbox_center_y = y1 + bbox_height // 2
                distance = math.sqrt((bbox_center_x - frame_center_x) ** 2 + (bbox_center_y - frame_center_y) ** 2)
                detection_info = {
                    "timestamp": timestamp,
                    "size": bbox_size,
                    "distance": distance
                }
                if int(cls) == 0:
                    detections["Pepsi"].append(detection_info)
                elif int(cls) == 1:
                    detections["CocaCola"].append(detection_info)

    out.write(frame)
    frame_number += 1


cap.release()
out.release()
cv2.destroyAllWindows()

json_output_path = 'detections.json'
with open(json_output_path, 'w') as json_file:
    json.dump(detections, json_file, indent=4)

print(f"Processed video saved to {output_video_path}")
print(f"Detections saved to {json_output_path}")
