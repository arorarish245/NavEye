import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize voice engine
engine = pyttsx3.init()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Flag to only announce once
objects_announced = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)
    names = model.names  # Class names like 'bottle', 'chair'

    # Get all detected classes from the frame
    detected_objects = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detected_objects.add(names[cls_id])

    # Only announce once
    if not objects_announced and detected_objects:
        sentence = "I see " + ", ".join(detected_objects)
        print("🔊", sentence)
        engine.say(sentence)
        engine.runAndWait()
        objects_announced = True

    # Draw bounding boxes
    annotated_frame = results[0].plot()
    cv2.imshow("Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
