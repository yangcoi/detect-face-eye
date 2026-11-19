# Real-Time Driver Drowsiness Detection System

A computer vision desktop application designed to detect driver drowsiness and closed-eye fatigue in real time using Dlib facial landmarks, MobileNet eye-state classification, and automated audio alarms.

---

## 1. Project Status & Current Maturity Level

**Current Status:** Working Computer Vision Prototype.

This repository is a functional computer vision prototype created to evaluate real-time driver fatigue monitoring. It runs as a standalone Python desktop application connected to a local camera feed.

### What is implemented:
- Real-time face detection using HOG (Histogram of Oriented Gradients) via `face_recognition`.
- 68-point facial landmark alignment using Dlib to extract eye regions.
- MobileNet-based binary eye-state classification (open vs. closed) from cropped grayscale eye patches.
- Consecutive frame closure counter: triggers an acoustic horn warning (`veryloudairhorn.mp3`) when closure exceeds the safety threshold.
- Automatic incident video recording: captures and saves short MP4 clips of detected drowsiness events into `./detected/`.

### Current limitations & Scope:
- Desktop script: Designed for single-camera desktop/laptop environments without a web dashboard.
- Model dependencies: Relies on pre-trained Dlib 68-landmark weights (`shape_predictor_68_face_landmarks.dat`) and a fine-tuned MobileNet model (`weights.149-0.01.hdf5`).
- Environment sensitivity: Detection accuracy depends on lighting conditions and camera angle relative to the driver's face.

---

## 2. System Flow

```
+-------------------------------------------------------------+
|                     WEBCAM VIDEO FEED                       |
|  - Frame capture at 30 FPS                                  |
|  - Downsampling & LAB color space conversion                |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               FACE DETECTION & LANDMARK EXTRACTION          |
|  - HOG face bounding box detection                          |
|  - Dlib 68 facial landmarks locator                         |
|  - Left and right eye bounding box segmentation             |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                 EYE STATE CLASSIFICATION                    |
|  - Preprocess eye patches (20x10 grayscale / MobileNet)     |
|  - Binary classification: Open vs Closed                    |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  FATIGUE LOGIC & RESPONSE                   |
|  - Closed frame counter > Threshold                         |
|  - Trigger audible horn alarm (playsound)                   |
|  - Save incident video clip to ./detected/                  |
+-------------------------------------------------------------+
```

---

## 3. Tech Stack

- Language: Python 3.8+
- Computer Vision: OpenCV, Dlib, face_recognition, imutils
- Deep Learning: TensorFlow, Keras (MobileNet)
- Audio: playsound

---

## 4. Setup & Running Locally

### Prerequisites

- Python 3.8 or 3.9 recommended
- CMake and C++ build tools (required to compile `dlib`)
- Working webcam

### Installation

```bash
# Clone the repository
git clone https://github.com/yangcoi/detect-face-eye.git
cd detect-face-eye

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Execution

```bash
python sleep_detect.py
```

Press `q` in the OpenCV window to exit the application.

---

## License

This project is licensed under the MIT License.
