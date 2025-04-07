import cv2
import pyttsx3

# Initialize voice engine
engine = pyttsx3.init()
engine.say("Testing your camera and voice. Please wait.")
engine.runAndWait()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Feed", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
