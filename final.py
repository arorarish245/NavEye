import cv2
import time
import pyttsx3
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# Initialize voice engine
engine = pyttsx3.init()
speaking = False  

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
frame_stability_threshold = 45

object_appearances = {}
static_objects = {}
person_position = None
thumbs_up_triggered = False
last_thumbs_time = 0
last_seen = {}
object_timeout = 10
initial_static_locked = False
announced_new_objects = set() 
selected_object_name = None


# Helper function for calculating distance and direction
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
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y

    fingers_folded = (
        middle_tip.y > index_mcp.y and
        ring_tip.y > index_mcp.y and
        pinky_tip.y > index_mcp.y
    )

    return thumb_up and fingers_folded

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
            last_seen[name] = current_time

    # Lock initial static objects at 8 seconds only once
    if not initial_static_locked and elapsed_time >= object_detection_duration:
        for name, positions in object_appearances.items():
            if len(positions) >= frame_stability_threshold:
                avg_x = int(np.mean([p[0] for p in positions]))
                avg_y = int(np.mean([p[1] for p in positions]))
                static_objects[name] = (avg_x, avg_y) 
        initial_static_locked = True  # Mark that we’ve done initial locking
        print("Initial static objects locked:", static_objects)

        if static_objects:
            object_list_str = ", ".join(static_objects.keys())
            print("🔊 I see:", object_list_str)
            engine.say(f"I see: {object_list_str}")
            engine.runAndWait()
            cv2.waitKey(1)
            engine.say("I will now name each object. Show thumbs-up when I say the one you want to track.")
            engine.runAndWait()
            cv2.waitKey(1)

            
            for name in static_objects:
                engine.say(name)
                engine.runAndWait()
                cv2.waitKey(1)

                thumbs_start = time.time()
                while time.time() - thumbs_start < 8:
                    ret_check, frame_check = cap.read()
                    if not ret_check:
                        print("Camera read failed.")

                        break

                    frame_check = cv2.flip(frame_check, 1)
                    rgb_check = cv2.cvtColor(frame_check, cv2.COLOR_BGR2RGB)
                    hand_check = hands.process(rgb_check)

                    if hand_check.multi_hand_landmarks:
                        for hand_landmarks in hand_check.multi_hand_landmarks:
                            if is_thumbs_up(hand_landmarks):
                                selected_object_name = name
                                break
                    if selected_object_name:
                        break

                    

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if selected_object_name:
                    engine.say(f"Object selected: {selected_object_name}")
                    engine.runAndWait()
                    cv2.waitKey(1)
                    print(f"Object selected for tracking: {selected_object_name}")
                    break
                else:
                    print(f"No thumbs-up for: {name}, moving to next.")
            
            if not selected_object_name:
                engine.say("No object was selected.")
                engine.runAndWait()
                cv2.waitKey(1)
                print("No object selected.")
        else:
            sentence = "I could not detect any consistent objects."
            print("🔊", sentence)
            engine.say(sentence)
            engine.runAndWait()
            cv2.waitKey(1)
                    

    # Dynamically add new stable objects after initial lock
    if initial_static_locked:
        for name, positions in object_appearances.items():
            if name not in static_objects and len(positions) >= frame_stability_threshold:
                recent_positions = positions[-frame_stability_threshold:]
                avg_x = int(np.mean([p[0] for p in recent_positions]))
                avg_y = int(np.mean([p[1] for p in recent_positions]))
                static_objects[name] = (avg_x, avg_y)
                print(f"New static object added: {name} at ({avg_x}, {avg_y})")

                if name not in announced_new_objects and person_position:
                    distance_str, direction = get_distance_and_direction((avg_x, avg_y), person_position)
                    new_sentence = f"New object {name}, about {distance_str} to your {direction}"
                    print("🔊", new_sentence)
                    engine.say(new_sentence)
                    engine.say(f"Do you want me to track {name}? Show thumbs-up.")
                    try:
                        engine.runAndWait()
                        cv2.waitKey(1)
                    except RuntimeError:
                        print("Voice engine already running.")
                    
                    # Check for thumbs-up within 8 seconds without interfering camera
                    thumbs_up_detected = False
                    start_time = time.time()
                    while time.time() - start_time < 8:
                        success_check, frame_check = cap.read()
                        if not success_check:
                            break

                        frame_check = cv2.flip(frame_check, 1)
                        rgb_frame_check = cv2.cvtColor(frame_check, cv2.COLOR_BGR2RGB)
                        hand_check = hands.process(rgb_frame_check)

                        if hand_check.multi_hand_landmarks:
                            for hand_landmarks in hand_check.multi_hand_landmarks:
                                if is_thumbs_up(hand_landmarks):
                                    thumbs_up_detected = True
                                    break
                        if thumbs_up_detected:
                            break

                        

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if thumbs_up_detected:
                        selected_object_name = name
                        print(f"New object selected for tracking: {selected_object_name}")
                        engine.say(f"Tracking {selected_object_name} now.")
                        try:
                            engine.runAndWait()
                            cv2.waitKey(1)
                        except RuntimeError:
                            print("Voice engine already running.")
                    else:
                        print(f"User did not select {name}")

                    announced_new_objects.add(name)

    for name in list(static_objects.keys()):
        if name not in last_seen or (current_time - last_seen[name]) > object_timeout:
            print(f" Removing lost object: {name}")
            del static_objects[name]
            if name in object_appearances:
                del object_appearances[name]
            if name in announced_new_objects:
                announced_new_objects.remove(name)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if is_thumbs_up(hand_landmarks):
                if not thumbs_up_triggered and not speaking:
                    thumbs_up_triggered = True
                    last_thumbs_time = time.time()

                    if selected_object_name and selected_object_name in static_objects and person_position:
                        obj_pos = static_objects[selected_object_name]
                        distance_str, direction = get_distance_and_direction(obj_pos, person_position)
                        sentence = f"{selected_object_name} is approximately {distance_str} to your {direction}"
                        print("🔊", sentence)

                        # Speak only once
                        speaking = True
                        engine.stop()
                        engine.say(sentence)
                        try:
                            engine.runAndWait()
                            time.sleep(0.5)
                        except RuntimeError:
                            print("Voice engine was already running, skipping.")
                        speaking = False
                    else:
                        print("🔊 Object not available to track.")
                        engine.stop()
                        engine.say("Object not available to track.")
                        try:
                            engine.runAndWait()
                        except RuntimeError:
                            print("Voice engine was already running, skipping.")
                        speaking = False
            else:
                thumbs_up_triggered = False
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