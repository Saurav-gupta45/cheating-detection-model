"""
ProctorAI Flask Application

Google Meet-style proctoring dashboard with:
  - MJPEG video stream from webcam
  - Server-Sent Events (SSE) for real-time telemetry
  - Feature toggle API endpoints
  - Voice calibration status endpoint

Routes:
  GET  /                  → Serve meet.html dashboard
  GET  /video_feed        → MJPEG webcam stream with HUD overlays
  GET  /status_feed       → SSE stream (JSON: suspicion %, feature status, logs)
  POST /toggle            → Toggle master proctoring or individual features
  GET  /calibration_status → Voice calibration progress
"""

import os
import sys
import time
import json
import threading

import cv2
from flask import Flask, Response, render_template, request, jsonify

# Add project root to path for src imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.components.face_detector import FaceDetector
from src.components.gaze_detector import GazeDetector
from src.components.light_detector import LightDetector
from src.pipeline.proctor_pipeline import ProctorPipeline

# Lazy-loaded heavy modules (phone, audio) to speed up startup
phone_detector = None
audio_proctor = None


def create_app():
    """Factory function to create and configure the Flask app."""
    app = Flask(__name__,
                template_folder=os.path.join(PROJECT_ROOT, 'templates'),
                static_folder=os.path.join(PROJECT_ROOT, 'static'))

    # ──────────────────────────────────────────
    # Initialize detection components
    # ──────────────────────────────────────────
    print("\n=== ProctorAI System Initialization ===")

    face_detector = FaceDetector(max_faces=5)
    gaze_detector = GazeDetector()
    light_detector = LightDetector()
    pipeline = ProctorPipeline()

    # Lazy load phone detector (YOLOv8 is heavy)
    global phone_detector
    try:
        from src.components.phone_detector import PhoneDetector
        phone_detector = PhoneDetector()
    except Exception as e:
        print(f"[WARNING] Phone detector failed to load: {e}")
        phone_detector = None

    # Lazy load audio proctor
    global audio_proctor
    try:
        from src.components.audio_proctor import AudioProctor
        audio_proctor = AudioProctor()
        audio_proctor.start()
    except Exception as e:
        print(f"[WARNING] Audio proctor failed to load: {e}")
        audio_proctor = None

    print("=== All modules loaded ===\n")

    # ──────────────────────────────────────────
    # Camera Stream Thread
    # ──────────────────────────────────────────
    class CameraStream:
        """Background thread for webcam capture to prevent hardware conflicts."""

        def __init__(self):
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[ERROR] Could not open webcam!")
                self.frame = None
            else:
                ret, self.frame = self.cap.read()
                if ret and self.frame is not None:
                    self.frame = cv2.flip(self.frame, 1)
            self.lock = threading.Lock()
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print("[CameraStream] Background capture thread started")

        def _capture_loop(self):
            while self.running:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        with self.lock:
                            self.frame = frame
                time.sleep(0.016)  # ~60fps capture rate

        def get_frame(self):
            with self.lock:
                return self.frame.copy() if self.frame is not None else None

        def stop(self):
            self.running = False
            if self.cap.isOpened():
                self.cap.release()

    camera = CameraStream()

    # ──────────────────────────────────────────
    # Processing state (shared between routes)
    # ──────────────────────────────────────────
    processing_lock = threading.Lock()
    latest_processed_frame = [None]
    latest_face_result = [None]
    latest_gaze_result = [None]
    latest_phone_result = [None]
    latest_light_result = [None]

    def processing_loop():
        """Background loop that runs all detection modules on each frame."""
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # Face detection (always runs - shared by gaze & multi-face)
            face_result = face_detector.process(frame)

            # Gaze detection
            gaze_result = None
            if pipeline.feature_toggles.get('gaze', False) and pipeline.proctoring_enabled:
                if face_result['face_detected'] and face_result['landmarks_list']:
                    gaze_result = gaze_detector.process(face_result['landmarks_list'][0])

            # Phone detection
            phone_result = None
            if pipeline.feature_toggles.get('phone', False) and pipeline.proctoring_enabled and phone_detector:
                phone_result = phone_detector.process(frame)

            # Light detection
            light_result = None
            if pipeline.feature_toggles.get('light', False) and pipeline.proctoring_enabled:
                light_result = light_detector.process(frame)

            # Audio state
            audio_status = None
            if audio_proctor:
                audio_status = audio_proctor.get_status()
                # Update face/mouth state for lip-sync validation
                audio_proctor.update_face_state(
                    face_detected=face_result['face_detected'],
                    mouth_open=face_result['mouth_open']
                )

            # Update pipeline
            if pipeline.proctoring_enabled:
                pipeline.update(face_result, gaze_result, phone_result, light_result, audio_status)

            # Render HUD on frame
            display_frame = frame.copy()
            if pipeline.proctoring_enabled:
                display_frame = pipeline.render_hud(display_frame, face_result, phone_result)

            with processing_lock:
                latest_processed_frame[0] = display_frame
                latest_face_result[0] = face_result
                latest_gaze_result[0] = gaze_result
                latest_phone_result[0] = phone_result
                latest_light_result[0] = light_result

            time.sleep(0.033)  # ~30fps processing rate

    proc_thread = threading.Thread(target=processing_loop, daemon=True)
    proc_thread.start()

    # ──────────────────────────────────────────
    # Routes
    # ──────────────────────────────────────────

    @app.route('/login')
    def login():
        """Serve the ProctorAI login page."""
        return render_template('login.html')

    @app.route('/')
    def index():
        """Serve the Google Meet-style proctoring dashboard."""
        if audio_proctor:
            audio_proctor.reset_calibration()
        return render_template('meet.html')

    @app.route('/video_feed')
    def video_feed():
        """MJPEG stream endpoint for live video."""
        def generate():
            while True:
                with processing_lock:
                    frame = latest_processed_frame[0]

                if frame is None:
                    time.sleep(0.05)
                    continue

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # ~20fps stream to browser

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/status_feed')
    def status_feed():
        """Server-Sent Events endpoint for real-time telemetry."""
        def generate():
            while True:
                state = pipeline.get_state()

                # Add audio calibration info
                if audio_proctor:
                    audio_status = audio_proctor.get_status()
                    state['audio_calibrated'] = audio_status['is_calibrated']
                    state['calibration_progress'] = audio_status['calibration_progress']
                    state['is_recording_speech'] = audio_status['is_recording_speech']
                else:
                    state['audio_calibrated'] = True
                    state['calibration_progress'] = 1.0
                    state['is_recording_speech'] = False

                yield f"data: {json.dumps(state)}\n\n"
                time.sleep(0.5)  # Push updates 2x per second

        return Response(generate(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.route('/toggle', methods=['POST'])
    def toggle():
        """Toggle master proctoring or individual feature ON/OFF."""
        data = request.get_json()
        feature = data.get('feature', '')
        enabled = data.get('enabled', True)

        pipeline.set_toggle(feature, enabled)
        return jsonify({'success': True, 'feature': feature, 'enabled': enabled})

    @app.route('/calibration_status')
    def calibration_status():
        """Voice calibration progress endpoint."""
        if audio_proctor:
            status = audio_proctor.get_status()
            return jsonify({
                'is_calibrated': status['is_calibrated'],
                'progress': status['calibration_progress']
            })
        return jsonify({'is_calibrated': True, 'progress': 1.0})

    return app


# ──────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────
app = create_app()

if __name__ == "__main__":
    print("\n🚀 ProctorAI starting at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
