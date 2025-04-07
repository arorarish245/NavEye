import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# State variable to avoid repeating print
thumbs_up_detected = False
last_detection_time = 0

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Thumb should be higher than all fingers (lower y value)
    return (thumb_tip.y < index_tip.y and
            thumb_tip.y < middle_tip.y and
            thumb_tip.y < ring_tip.y and
            thumb_tip.y < pinky_tip.y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_thumbs_up(hand_landmarks):
                if not thumbs_up_detected:
                    thumbs_up_detected = True
                    last_detection_time = time.time()
                    print("✅ Thumbs-up detected!")
            else:
                thumbs_up_detected = False

    # Reset after 2 seconds to allow re-detection
    if thumbs_up_detected and (time.time() - last_detection_time > 5):
        thumbs_up_detected = False

    if thumbs_up_detected:
        cv2.putText(frame, "👍 Thumbs-Up Detected!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Thumbs-Up Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()