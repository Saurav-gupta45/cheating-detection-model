# ProctorAI 🛡️

ProctorAI is a real-time, multi-modal automated exam proctoring system designed to monitor and preserve academic integrity during online tests. Built on top of deep learning classifiers and computer vision models, it tracks student attention, audio events, and visual queues through a glassmorphic dashboard.

---

## 🌟 Key Features

*   **👁️ Gaze & Attention Tracking:** Maps iris and eye coordinates to flag when a candidate looks off-screen.
*   **👥 Multi-Face Detection:** Tracks up to 5 faces in real time and detects mouth-opening states to identify Whispering/Secondary voices.
*   **📱 Prohibited Object Detection:** Employs a YOLOv8 network optimized to detect mobile phones and unauthorized devices.
*   **🎙️ Dynamic Speaker Verification:** Calibrates a voiceprint enrollment using an ECAPA-TDNN embedding model to verify identity and flag third-party talkers.
*   **🔊 Acoustic Event Detection:** Recognizes critical environment sounds (Speech, Whispering, Laughter, Yawns) via YAMNet.
*   **⚡ Active Anomaly Panel:** Visually tracks active alerts on-screen with real-time video stream overlay notifications.
*   **📈 Suspicion Scoring Engine:** Combines time-independent calculations, decay cooldowns, and duration rules (e.g., voice flag triggers after >3s continuous detection).

---

## 🏗️ Project Architecture

```
proctor_project/
├── app.py                      # Flask Server Entry point
├── requirements.txt            # Python Dependencies
├── setup.py                    # Package setup configurations
├── artifacts/                  # Local directory for model weights
├── static/
│   ├── css/style.css           # Glassmorphic Dark UI Theme stylesheet
│   └── js/app.js               # Real-time SSE listener and Gauge animator
└── src/
    ├── components/
    │   ├── face_detector.py    # MediaPipe FaceMesh wrapper
    │   ├── gaze_detector.py    # PyTorch MLP Gaze neural network
    │   ├── phone_detector.py   # YOLOv8 target detector
    │   ├── light_detector.py   # Environmental brightness check
    │   └── audio_proctor.py    # Multi-threaded WeSpeaker & YAMNet worker
    └── pipeline/
        └── proctor_pipeline.py # Suspicion grading engine
```

---

## 🚀 Getting Started

### 📋 Prerequisites

Install python dependencies inside a clean virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🧠 Model Weights Setup
Download the required pre-trained weights and drop them into the `artifacts/` folder:
- **MediaPipe Face Mesh Task:** `face_landmarker.task`
- **WeSpeaker ONNX:** `wespeaker_en_voxceleb_resnet34.onnx`
- **YAMNet ONNX:** `yamnet.onnx`
- **YOLOv8 Weights:** `yolov8s.pt`
- **Gaze Model Weights:** `proctor_model.pth`

### 💻 Running the Server

Start the Flask application on port `5001`:
```bash
python3 app.py
```
Open **[http://localhost:5001/login](http://localhost:5001/login)** in your browser.

---

## 📄 License
This project is licensed under the MIT License.
