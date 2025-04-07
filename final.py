import cv2
import time
import pyttsx3
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Initialize voice engine
engine = pyttsx3.init()
speaking = False  # Flag to avoid double runAndWait crashes

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
names = model.names

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

start_time = time.time()
object_detection_duration = 8
frame_stability_threshold = 30

object_appearances = {}
static_objects = {}
person_position = None
thumbs_up_triggered = False
last_thumbs_time = 0

# ✅ Helper function for calculating distance and direction
def get_distance_and_direction(obj_pos, person_pos):
    dx = obj_pos[0] - person_pos[0]
    dy = obj_pos[1] - person_pos[1]
    pixel_distance = int((dx**2 + dy**2) ** 0.5)

    # Convert pixel distance to centimeters
    pixels_to_cm = 0.25
    distance_cm = pixel_distance * pixels_to_cm

    # Readable unit
    if distance_cm < 100:
        distance_str = f"{int(distance_cm)} centimeters"
    else:
        distance_str = f"{distance_cm / 100:.1f} meters"

    # Determine direction
    direction = "center"
    if dx > 40:
        direction = "right"
    elif dx < -40:
        direction = "left"

    return distance_str, direction

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and
            thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y)

print("Scanning for static objects...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    elapsed_time = current_time - start_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)

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
            if name not in object_appearances:
                object_appearances[name] = []
            object_appearances[name].append((cx, cy))

    if elapsed_time >= object_detection_duration and not static_objects:
        for name, positions in object_appearances.items():
            if len(positions) >= frame_stability_threshold:
                avg_x = int(np.mean([p[0] for p in positions]))
                avg_y = int(np.mean([p[1] for p in positions]))
                static_objects[name] = (avg_x, avg_y)

        print("🔒 Static objects locked:", static_objects)
        if static_objects and person_position:
            description_parts = []
            for name, pos in static_objects.items():
                distance_str, direction = get_distance_and_direction(pos, person_position)
                description_parts.append(f"{name} about {distance_str} to your {direction}")

            sentence = "I see " + ", and ".join(description_parts)
        else:
            sentence = "I could not detect any consistent objects."
        print("🔊", sentence)
        engine.stop()
        engine.say(" ")
        engine.runAndWait()
        engine.say(sentence)
        time.sleep(0.2)
        engine.runAndWait()

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_thumbs_up(hand_landmarks):
                if not thumbs_up_triggered and not speaking:
                    thumbs_up_triggered = True
                    last_thumbs_time = time.time()
                    if static_objects and person_position:
                        speak_queue = []
                        for obj_name, obj_pos in static_objects.items():
                            distance_str, direction = get_distance_and_direction(obj_pos, person_position)
                            speak = f"{obj_name} is approximately {distance_str} to your {direction}"
                            print("🔊", speak)
                            speak_queue.append(speak)

                        # Speak only once
                        speaking = True
                        engine.stop()
                        for sentence in speak_queue:
                            engine.say(sentence)
                        try:
                            engine.runAndWait()
                            time.sleep(0.5)
                        except RuntimeError:
                            print("⚠️ Voice engine was already running, skipping.")
                        speaking = False
            else:
                thumbs_up_triggered = False

    annotated = results[0].plot()
    if thumbs_up_triggered:
        cv2.putText(annotated, "👍 Thumbs-Up Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Smart Assistant", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
