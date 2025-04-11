# 🤖NAVEYE :- Smart Object Tracker with Gesture-Controlled Voice Guidance

A computer vision project that detects static objects using YOLOv8, tracks the person, and provides **real-time voice feedback** on object distance and direction when the user shows a **thumbs-up gesture**. 

🔊 Designed to be assistive and intuitive!

---

## 📸 Features

- ✅ **YOLOv8 Object Detection** – Detects and locks stable objects (e.g., teddy bear, chair).
- ✋ **Thumbs-Up Gesture Trigger** – Uses MediaPipe to detect thumbs-up gesture.
- 🧠 **Object Stability Check** – Locks only those objects that stay visible for 5 seconds.
- 🧍 **Person Tracking** – Tracks the user's position in real-time.
- 🧭 **Distance & Direction** – Calculates relative distance & left/right/front from the user.
- 🎙️ **Voice Output** – Speaks object info only when user shows a thumbs-up.
- 👀 **Real-time Display** – Visualizes detection, tracking, and gesture overlay via OpenCV.

---
## 🚀 How to Run

1. **Install dependencies**:

```bash
pip install ultralytics opencv-python mediapipe pyttsx3
```

2. **Download YOLOv8n model (if not cached automatically):**
```bash
from ultralytics import YOLO
YOLO("yolov8n.pt")
```

3. **Run the script:**
```bash
python final.py
```


