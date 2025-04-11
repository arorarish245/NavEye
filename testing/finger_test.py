import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,  # Enable both hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        total_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Only count fingers (skip thumb)
                finger_tip_ids = [8, 12, 16, 20]
                for tip_id in finger_tip_ids:
                    if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                        total_fingers += 1

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f'Fingers (No Thumb): {total_fingers}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Finger Count (No Thumbs)", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
