import cv2
import time
import pyttsx3
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
names = model.names

# Voice engine
engine = pyttsx3.init()

# Video capture
cap = cv2.VideoCapture(0)

# Timing
start_time = time.time()
object_detection_duration = 8  # seconds

static_objects = {}  # Store static object name and position (center)
person_position = None

print("Scanning for static objects...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - start_time

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    frame_objects = {}

    for box in boxes:
        cls_id = int(box.cls[0])
        name = names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if name == "person":
            person_position = (cx, cy)
        else:
            frame_objects[name] = (cx, cy)

    # After 8 seconds, lock static object positions
    if elapsed_time >= object_detection_duration and not static_objects:
        static_objects = frame_objects.copy()
        print("🔒 Static objects locked:", static_objects)

    # If static objects are locked and person is being tracked
    if static_objects and person_position:
        for obj_name, obj_pos in static_objects.items():
            dx = obj_pos[0] - person_position[0]
            dy = obj_pos[1] - person_position[1]
            pixel_distance = int((dx**2 + dy**2) ** 0.5)

            direction = "center"
            if dx > 40:
                direction = "right"
            elif dx < -40:
                direction = "left"

            print(f"{obj_name.capitalize()}: ~{pixel_distance} px away to your {direction}")

    # Annotate frame
    annotated = results[0].plot()
    cv2.imshow("Tracking", annotated)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
