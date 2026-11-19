# Real-Time Driver Drowsiness Detection System

Real-time driver fatigue monitoring system built with OpenCV, Dlib facial landmarks, and automated alarm alerting.

```
Pipeline: OpenCV (Camera Feed) -> Dlib (68 Landmarks) -> EAR / Eye State Classifier -> Alert (Airhorn) & Incident Recording
```

---

## 1. Project Status & Current Maturity Level

**Current Status:** Working Computer Vision Prototype.

This repository is a functional computer vision prototype developed for driver fatigue detection experiments. It runs locally as a standalone Python desktop application.

### Pipeline Implementation:
1. **OpenCV:** Captures webcam frames at 30 FPS, normalizes lighting, and manages real-time video display.
2. **Dlib (68 Facial Landmarks):** Detects facial boundaries via HOG and localizes 68 landmark points to isolate the left and right eye contours.
3. **EAR & Eye State Classification:** Evaluates eye closure state using Eye Aspect Ratio (EAR) geometry and a MobileNet neural network classifier.
4. **Alert & Incident Recording:** Tracks consecutive closed frames. When threshold is exceeded, it triggers an audible airhorn alert (`veryloudairhorn.mp3`) and automatically saves an incident video clip to `./detected/`.

### Current Limitations:
- Single-camera desktop script: Designed for local webcam execution without cloud telemetry.
- Environment sensitivity: Accuracy is affected by extreme low-light environments and oblique face angles.

---

## 2. System Architecture

```
+-------------------------------------------------------------+
|                     1. OPENCV VIDEO INGEST                  |
|  - Real-time webcam frame acquisition                       |
|  - Grayscale & LAB color space preprocessing                |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 2. DLIB 68 FACIAL LANDMARKS                 |
|  - Face detection via HOG classifier                        |
|  - 68-point landmark localization                           |
|  - Left/Right eye boundary segmentation                     |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|            3. EAR & EYE STATE CLASSIFICATION                |
|  - Eye Aspect Ratio (EAR) contour calculation               |
|  - MobileNet binary classifier (Open vs Closed)             |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             4. THRESHOLD ALERT & INCIDENT RECORDING         |
|  - Safety frame closure counter                             |
|  - Real-time airhorn alarm trigger (playsound)              |
|  - Automatic MP4 clip export to ./detected/                 |
+-------------------------------------------------------------+
```

---

## 3. Tech Stack

- Language: Python 3.8+
- Computer Vision: OpenCV, Dlib, face_recognition, imutils
- Machine Learning: TensorFlow, Keras (MobileNet)
- Audio: playsound

---

## 4. Setup & Running Locally

### Prerequisites

- Python 3.8 or 3.9
- CMake and C++ compiler (required for Dlib compilation)
- Working webcam

### Installation

```bash
git clone https://github.com/yangcoi/detect-face-eye.git
cd detect-face-eye

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Execution

```bash
python sleep_detect.py
```

Press `q` to terminate the application.

---

## License

This project is licensed under the MIT License.
