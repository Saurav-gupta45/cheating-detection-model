"""
ProctorPipeline: Central state manager, suspicion score calculator, and HUD renderer.

Integrates all 6 detection modules under a time-based decay algorithm
to produce a unified suspicion percentage (0-100%).

Suspicion rates are time-delta-based (per second) to avoid frame rate dependency.
Warning logs are throttled to one per type every 3 seconds.
"""

import time
import threading
import cv2
import numpy as np


class ProctorPipeline:
    """
    Orchestrates all detection modules and computes a unified suspicion score.
    
    Suspicion Rates (per second):
        Gaze off-screen >3s:    +15%/sec
        Multiple faces:         +15%/sec  + 20% one-time penalty
        Phone detected:         +25%/sec  + 40% one-time penalty
        Unknown speaker:        +10%/sec  + 20% one-time penalty
        Lip-sync mismatch:      +5%/sec
        Lighting change:        +15% one-time penalty
        All clear (decay):      -2%/sec
    """

    # Suspicion increment rates (per second)
    RATE_GAZE = 15.0
    RATE_MULTIFACE = 15.0
    RATE_PHONE = 25.0
    RATE_UNKNOWN_SPEAKER = 10.0
    RATE_LIP_SYNC = 5.0
    RATE_DECAY = 2.0

    # One-time penalties (applied once per detection event)
    PENALTY_MULTIFACE = 20.0
    PENALTY_PHONE = 40.0
    PENALTY_LIGHT = 15.0
    PENALTY_UNKNOWN_SPEAKER = 20.0

    # Gaze off-screen grace period
    GAZE_GRACE_SECONDS = 3.0

    # No-face grace period
    NO_FACE_GRACE_SECONDS = 2.0

    # Log throttling interval
    LOG_THROTTLE_SECONDS = 3.0

    def __init__(self):
        self._lock = threading.Lock()

        # --- Master & feature toggles ---
        self.proctoring_enabled = True
        self.feature_toggles = {
            'gaze': True,
            'face': True,
            'phone': True,
            'light': True,
            'voice': True,
            'sound': True,
        }

        # --- Suspicion state ---
        self.suspicion = 0.0
        self._last_update_time = time.time()

        # --- Per-feature status ---
        self.feature_status = {
            'gaze': {'status': 'OK', 'detail': ''},
            'face': {'status': 'OK', 'detail': ''},
            'phone': {'status': 'OK', 'detail': ''},
            'light': {'status': 'OK', 'detail': ''},
            'voice': {'status': 'OK', 'detail': ''},
            'sound': {'status': 'OK', 'detail': ''},
        }

        # --- Gaze timing ---
        self._gaze_off_start = None  # Timestamp when gaze first went off-screen
        self._gaze_off_duration = 0.0

        # --- Face timing ---
        self._no_face_start = None

        # --- Voice/Speaker timing and frequency tracking ---
        self._unknown_voice_start_time = None
        self._unknown_voice_duration = 0.0
        self._unknown_voice_duration_penalty_applied = False
        self._unknown_voice_timestamps = []
        self._last_frequency_penalty_time = 0.0

        # --- Penalty tracking (to apply one-time penalties only once per event) ---
        self._penalties_applied = {
            'multiface': False,
            'phone': False,
            'light': False,
            'unknown_speaker': False,
        }

        # --- Warning log ---
        self._warning_logs = []  # List of { timestamp, message, type }
        self._last_log_times = {}  # { violation_type: last_timestamp }
        self._max_logs = 100

        print("[ProctorPipeline] Initialized")

    def set_toggle(self, feature, enabled):
        """Toggle a specific feature or master proctoring ON/OFF."""
        with self._lock:
            if feature == 'master':
                self.proctoring_enabled = enabled
            elif feature in self.feature_toggles:
                self.feature_toggles[feature] = enabled

    def get_state(self):
        """Thread-safe retrieval of full pipeline state for SSE broadcast."""
        with self._lock:
            return {
                'proctoring_enabled': self.proctoring_enabled,
                'feature_toggles': dict(self.feature_toggles),
                'suspicion': round(self.suspicion, 1),
                'feature_status': {k: dict(v) for k, v in self.feature_status.items()},
                'warning_logs': list(self._warning_logs[-20:]),  # Last 20 logs
                'gaze_off_duration': round(self._gaze_off_duration, 1),
            }

    def _add_warning(self, violation_type, message):
        """Add a warning log with throttling (3s cooldown per type)."""
        now = time.time()
        last = self._last_log_times.get(violation_type, 0)
        if now - last >= self.LOG_THROTTLE_SECONDS:
            self._last_log_times[violation_type] = now
            log_entry = {
                'timestamp': time.strftime('%H:%M:%S'),
                'message': message,
                'type': violation_type
            }
            self._warning_logs.append(log_entry)
            if len(self._warning_logs) > self._max_logs:
                self._warning_logs = self._warning_logs[-self._max_logs:]

    def update(self, face_result, gaze_result, phone_result, light_result, audio_status):
        """
        Main update tick. Called once per frame by the Flask app.
        
        Args:
            face_result: dict from FaceDetector.process()
            gaze_result: dict from GazeDetector.process() or None
            phone_result: dict from PhoneDetector.process() or None
            light_result: dict from LightDetector.process()
            audio_status: dict from AudioProctor.get_status()
        """
        with self._lock:
            if not self.proctoring_enabled:
                return

            now = time.time()
            dt = now - self._last_update_time
            self._last_update_time = now

            # Clamp dt to avoid huge jumps on lag
            dt = min(dt, 1.0)

            any_violation = False

            # ──────────────────────────────────────────
            # FEATURE 1: Gaze Detection
            # ──────────────────────────────────────────
            if self.feature_toggles.get('gaze', False) and gaze_result is not None:
                if gaze_result['is_cheating']:
                    if self._gaze_off_start is None:
                        self._gaze_off_start = now
                    self._gaze_off_duration = now - self._gaze_off_start

                    if self._gaze_off_duration > self.GAZE_GRACE_SECONDS:
                        self.suspicion += self.RATE_GAZE * dt
                        any_violation = True
                        self.feature_status['gaze'] = {
                            'status': 'ALERT',
                            'detail': f'Off-screen for {self._gaze_off_duration:.1f}s'
                        }
                        self._add_warning('gaze', f'Gaze off-screen detected ({self._gaze_off_duration:.1f}s)')
                    else:
                        self.feature_status['gaze'] = {
                            'status': 'WARNING',
                            'detail': f'Looking away ({self._gaze_off_duration:.1f}s)'
                        }
                else:
                    self._gaze_off_start = None
                    self._gaze_off_duration = 0.0
                    self.feature_status['gaze'] = {
                        'status': 'OK',
                        'detail': f'Safe ({gaze_result["safe_probability"]:.0%})'
                    }
            elif self.feature_toggles.get('gaze', False) and gaze_result is None and face_result['face_detected']:
                self.feature_status['gaze'] = {'status': 'OK', 'detail': 'Processing...'}

            # ──────────────────────────────────────────
            # FEATURE 2: Multi-Face / No-Face Detection
            # ──────────────────────────────────────────
            if self.feature_toggles.get('face', False):
                num_faces = face_result['num_faces']

                if num_faces == 0:
                    if self._no_face_start is None:
                        self._no_face_start = now
                    no_face_duration = now - self._no_face_start

                    if no_face_duration > self.NO_FACE_GRACE_SECONDS:
                        self.suspicion += self.RATE_MULTIFACE * dt
                        any_violation = True
                        self.feature_status['face'] = {
                            'status': 'ALERT',
                            'detail': f'No face for {no_face_duration:.1f}s'
                        }
                        self._add_warning('no_face', f'No face detected for {no_face_duration:.1f}s')
                    else:
                        self.feature_status['face'] = {
                            'status': 'WARNING',
                            'detail': 'Face not detected'
                        }

                elif num_faces == 1:
                    self._no_face_start = None
                    self.feature_status['face'] = {'status': 'OK', 'detail': '1 face detected'}
                    self._penalties_applied['multiface'] = False

                else:  # Multiple faces
                    self._no_face_start = None
                    self.suspicion += self.RATE_MULTIFACE * dt
                    any_violation = True

                    if not self._penalties_applied['multiface']:
                        self.suspicion += self.PENALTY_MULTIFACE
                        self._penalties_applied['multiface'] = True

                    self.feature_status['face'] = {
                        'status': 'ALERT',
                        'detail': f'{num_faces} faces detected!'
                    }
                    self._add_warning('multiface', f'Multiple faces detected ({num_faces})')

            # ──────────────────────────────────────────
            # FEATURE 3: Phone Detection
            # ──────────────────────────────────────────
            if self.feature_toggles.get('phone', False) and phone_result is not None:
                if phone_result['detected']:
                    self.suspicion += self.RATE_PHONE * dt
                    any_violation = True

                    if not self._penalties_applied['phone']:
                        self.suspicion += self.PENALTY_PHONE
                        self._penalties_applied['phone'] = True

                    conf = max(b['confidence'] for b in phone_result['boxes']) if phone_result['boxes'] else 0
                    self.feature_status['phone'] = {
                        'status': 'ALERT',
                        'detail': f'Phone detected ({conf:.0%})'
                    }
                    self._add_warning('phone', f'Cell phone detected ({conf:.0%} confidence)')
                else:
                    self.feature_status['phone'] = {'status': 'OK', 'detail': 'No phone'}
                    self._penalties_applied['phone'] = False

            # ──────────────────────────────────────────
            # FEATURE 4: Lighting Monitor
            # ──────────────────────────────────────────
            if self.feature_toggles.get('light', False) and light_result is not None:
                if light_result['alert'] is not None:
                    if not self._penalties_applied['light']:
                        self.suspicion += self.PENALTY_LIGHT
                        self._penalties_applied['light'] = True

                    any_violation = True
                    alert_text = light_result['alert'].replace('_', ' ').title()
                    self.feature_status['light'] = {
                        'status': 'ALERT',
                        'detail': alert_text
                    }
                    self._add_warning('light', f'Lighting alert: {alert_text}')
                else:
                    self.feature_status['light'] = {
                        'status': 'OK',
                        'detail': f'Brightness: {light_result["brightness"]:.0f}'
                    }
                    self._penalties_applied['light'] = False

            # ──────────────────────────────────────────
            # FEATURE 5: Voice Verification
            # ──────────────────────────────────────────
            if self.feature_toggles.get('voice', False) and audio_status is not None:
                if audio_status['is_calibrated']:
                    is_speech = audio_status['event'] in ["Speech", "Whispering", "Laughter", "Yawn", "Scream"]

                    if is_speech and audio_status['score'] >= 0.35:
                        if not audio_status['is_user_voice']:
                            any_violation = True

                            # Calculate duration of unknown speaker
                            if self._unknown_voice_start_time is None:
                                self._unknown_voice_start_time = now
                            self._unknown_voice_duration = now - self._unknown_voice_start_time

                            if self._unknown_voice_duration > 3.0 and not self._unknown_voice_duration_penalty_applied:
                                self.suspicion += 10.0
                                self._unknown_voice_duration_penalty_applied = True
                                self._add_warning('unknown_speaker_time', 'Unknown speaker detected for >3 seconds (+10%)')

                            self.feature_status['voice'] = {
                                'status': 'ALERT',
                                'detail': f'Unknown speaker ({self._unknown_voice_duration:.1f}s, sim: {audio_status["similarity"]:.2f})'
                            }
                            self._add_warning('unknown_speaker', f'Unknown speaker detected (similarity: {audio_status["similarity"]:.2f})')
                        else:
                            # Reset trackers when user is speaking
                            self._unknown_voice_start_time = None
                            self._unknown_voice_duration = 0.0
                            self._unknown_voice_duration_penalty_applied = False

                            if audio_status['lip_sync_warning']:
                                self.suspicion += self.RATE_LIP_SYNC * dt
                                any_violation = True
                                self.feature_status['voice'] = {
                                    'status': 'WARNING',
                                    'detail': 'Speech detected, mouth closed'
                                }
                                self._add_warning('lip_sync', 'Lip-sync mismatch: speech without mouth movement')
                            else:
                                self.feature_status['voice'] = {
                                    'status': 'OK',
                                    'detail': f'User speaking (sim: {audio_status["similarity"]:.2f})'
                                }
                    else:
                        # Reset trackers when silent
                        self._unknown_voice_start_time = None
                        self._unknown_voice_duration = 0.0
                        self._unknown_voice_duration_penalty_applied = False

                        self.feature_status['voice'] = {'status': 'OK', 'detail': 'Silent'}
                else:
                    pct = int(audio_status['calibration_progress'] * 100)
                    self.feature_status['voice'] = {
                        'status': 'CALIBRATING',
                        'detail': f'Voice calibration: {pct}%'
                    }

            # ──────────────────────────────────────────
            # FEATURE 6: Environmental Sound Detection
            # ──────────────────────────────────────────
            if self.feature_toggles.get('sound', False) and audio_status is not None:
                if audio_status['is_calibrated']:
                    cat = audio_status.get('sound_category', 'SAFE')
                    if cat != 'SAFE':
                        any_violation = True
                        self.suspicion += 10.0 * dt  # +10%/sec for suspicious sounds
                        label = cat.replace('_', ' ').title()
                        self.feature_status['sound'] = {
                            'status': 'ALERT',
                            'detail': f'{label}: {audio_status["event"]}'
                        }
                        self._add_warning('sound', f'Suspicious sound: {audio_status["event"]} ({label})')
                    else:
                        self.feature_status['sound'] = {
                            'status': 'OK',
                            'detail': f'{audio_status["event"]}'
                        }
                else:
                    self.feature_status['sound'] = {'status': 'OK', 'detail': 'Waiting for calibration'}

            # ──────────────────────────────────────────
            # DECAY: Reduce suspicion when everything is OK
            # ──────────────────────────────────────────
            if not any_violation:
                self.suspicion -= self.RATE_DECAY * dt

            # Clamp 0-100
            self.suspicion = max(0.0, min(100.0, self.suspicion))

    def render_hud(self, frame, face_result, phone_result):
        """
        Draw detection overlays on the video frame (bounding boxes, face mesh).
        This is optional visualization for the MJPEG stream.
        """
        h, w = frame.shape[:2]

        # Draw phone bounding boxes
        if phone_result and phone_result['detected']:
            for box in phone_result['boxes']:
                cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 0, 255), 3)
                label = f"Phone: {box['confidence']:.0%}"
                cv2.putText(frame, label, (box['x1'], box['y1'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw alert border based on suspicion level
        if self.suspicion >= 60:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)  # Red border
        elif self.suspicion >= 30:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 4)  # Orange border

        return frame
