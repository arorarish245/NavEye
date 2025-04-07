import cv2
import pyttsx3
import time
from ultralytics import YOLO
from collections import defaultdict

# Initialize voice engine
engine = pyttsx3.init()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Flag to only announce once
objects_announced = False

# Timer start
start_time = time.time()

# Track how many frames each object appears in
object_frame_counts = defaultdict(int)

# Scanning phase for 5 seconds
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    names = model.names

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            object_frame_counts[class_name] += 1

    # Only announce after 5 seconds
    if not objects_announced and (time.time() - start_time >= 8):
        # Filter objects that appeared in ≥ 15 frames
        consistent_objects = [obj for obj, count in object_frame_counts.items() if count >= 30]

        if consistent_objects:
            sentence = "I see " + ", ".join(consistent_objects)
        else:
            sentence = "I could not detect any consistent objects."

        print("🔊", sentence)
        engine.say(sentence)
        engine.runAndWait()
        objects_announced = True

    # Draw bounding boxes
    annotated_frame = results[0].plot()
    cv2.imshow("Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()