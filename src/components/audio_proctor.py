"""
AudioProctor: Combined voice calibration, speaker verification, and sound classification.

Runs as a daemon thread combining:
  - WeSpeaker: 20-second voice enrollment + cosine similarity verification
  - YAMNet: Environmental sound classification (phones, alarms, music, tools)
  - Lip-sync validation: Cross-checks speech with mouth state

Thread-safe status retrieval via get_status().
"""

import os
import csv
import time
import threading
import numpy as np
import sounddevice as sd
import onnxruntime as ort

from src.utils import get_artifact_path

# Try to import sherpa_onnx for speaker embedding
try:
    import sherpa_onnx
    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False
    print("[AudioProctor] WARNING: sherpa_onnx not installed. Voice verification disabled.")


class AudioProctor:
    """
    Combined audio proctoring engine running in a background thread.
    
    Handles:
      1. Voice calibration (20s enrollment)
      2. Speaker verification (cosine similarity)
      3. Environmental sound classification (YAMNet)
      4. Lip-sync cross-validation
    
    Usage:
        proctor = AudioProctor()
        proctor.start()
        status = proctor.get_status()
        proctor.stop()
    """

    SAMPLE_RATE = 16000
    WINDOW_SIZE = 15600  # 0.975 seconds
    CALIBRATION_SECONDS = 20
    CALIBRATION_SAMPLES = CALIBRATION_SECONDS * SAMPLE_RATE
    LIP_OPEN_THRESHOLD = 0.025

    # Cheating sound categories
    PHONE_ALARM_KEYWORDS = ["ringtone", "alarm", "buzzer", "telephone", "beep"]
    PHONE_ALARM_EVENTS = [
        "Telephone bell ringing", "Telephone dialing, DTMF",
        "Smoke detector, smoke alarm", "Fire alarm",
        "Beep, bleep", "Car alarm", "Reversing beeps"
    ]
    MUSIC_KEYWORDS = ["music", "singing", "choir"]
    MUSIC_EVENTS = ["Radio", "Television"]
    TOOL_KEYWORDS = ["drill", "hammer", "sawing", "chainsaw", "power tool", "jackhammer", "tools", "construction"]

    def __init__(self):
        # Thread-safe state
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Audio classification state
        self._latest_event = "Silence"
        self._latest_score = 0.0
        self._latest_similarity = 1.0
        self._is_user_voice = True
        self._lip_sync_warning = False

        # Calibration state
        self._enrolled_embedding = None
        self._is_calibrated = False
        self._calibration_progress = 0.0
        self._similarity_threshold = 0.60
        self._smoothed_similarity = None
        self._calibration_start_time = None  # Track when calibration started
        self._mock_calibration_seconds = 10  # Simulated calibration when WeSpeaker unavailable
        self._non_speech_frames = 0

        # External state set by pipeline (mouth open status)
        self._mouth_open = False
        self._face_detected = False
        self._calibration_audio_buffer = []
        self._collect_calibration = True
        self._pause_calibration = True
        self._is_recording_speech = False

        # Sound category cache
        self._sound_category = "SAFE"

        # Load models
        self._load_models()
        print("[AudioProctor] Initialized")

    def _load_models(self):
        """Load YAMNet and WeSpeaker models."""
        # YAMNet
        yamnet_path = get_artifact_path("yamnet.onnx")
        classmap_path = get_artifact_path("yamnet_class_map.csv")

        if not os.path.exists(yamnet_path):
            raise FileNotFoundError(f"YAMNet model not found at {yamnet_path}")
        if not os.path.exists(classmap_path):
            raise FileNotFoundError(f"YAMNet class map not found at {classmap_path}")

        self._yamnet_session = ort.InferenceSession(yamnet_path)
        self._yamnet_input_name = self._yamnet_session.get_inputs()[0].name
        self._class_names = self._load_class_map(classmap_path)

        # WeSpeaker
        if HAS_SHERPA:
            wespeaker_path = get_artifact_path("wespeaker_en_voxceleb_resnet34.onnx")
            if os.path.exists(wespeaker_path):
                config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=wespeaker_path, num_threads=1, debug=False
                )
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
            else:
                self._extractor = None
                print("[AudioProctor] WeSpeaker model not found, voice verification disabled.")
        else:
            self._extractor = None

    @staticmethod
    def _load_class_map(path):
        """Load YAMNet class index mapping from CSV."""
        class_names = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 3:
                    class_names[int(row[0])] = row[2]
        return class_names

    def start(self):
        """Start the audio proctoring daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()
        print("[AudioProctor] Background thread started")

    def stop(self):
        """Stop the audio proctoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        print("[AudioProctor] Stopped")

    def update_face_state(self, face_detected, mouth_open):
        """Called by the pipeline each frame to update face/lip state."""
        with self._lock:
            self._face_detected = face_detected
            self._mouth_open = mouth_open

    def get_status(self):
        """Thread-safe retrieval of current audio proctoring state."""
        with self._lock:
            return {
                'event': self._latest_event,
                'score': self._latest_score,
                'similarity': self._latest_similarity,
                'is_user_voice': self._is_user_voice,
                'is_calibrated': self._is_calibrated,
                'calibration_progress': self._calibration_progress,
                'lip_sync_warning': self._lip_sync_warning,
                'sound_category': self._sound_category,
                'similarity_threshold': self._similarity_threshold,
                'is_recording_speech': self._is_recording_speech
            }

    def _classify_sound(self, event, score):
        """Categorize a YAMNet event into cheating categories."""
        if score < 0.35:
            return "SAFE"

        event_lower = event.lower()

        is_phone = (
            any(k in event_lower for k in self.PHONE_ALARM_KEYWORDS)
            or event in self.PHONE_ALARM_EVENTS
        )
        if is_phone:
            return "PHONE_ALARM"

        is_music = (
            any(k in event_lower for k in self.MUSIC_KEYWORDS)
            or event in self.MUSIC_EVENTS
        )
        if is_music:
            return "MUSIC_PLAYBACK"

        is_tool = any(k in event_lower for k in self.TOOL_KEYWORDS)
        if is_tool:
            return "CONSTRUCTION_TOOL"

        return "SAFE"

    def _audio_loop(self):
        """Main audio processing loop (runs in background thread)."""
        buffer = np.zeros(self.WINDOW_SIZE, dtype=np.float32)
        audio_data_list = []

        def callback(indata, frames, time_info, status):
            if status:
                pass  # Suppress warnings to avoid console spam
            data_copy = indata.copy().flatten()
            audio_data_list.append(data_copy)

            # Collect calibration samples when face is present and user is actually speaking (RMS gate)
            if self._collect_calibration and not self._pause_calibration:
                rms = np.sqrt(np.mean(data_copy**2) + 1e-9)
                if rms > 0.008:
                    self._calibration_audio_buffer.append(data_copy)
                    self._is_recording_speech = True
                else:
                    self._is_recording_speech = False
            else:
                self._is_recording_speech = False

        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, callback=callback):
                print("[AudioProctor] Microphone stream active")
                last_analysis_time = time.time()

                while self._running:
                    time.sleep(0.05)

                    # Update calibration pause based on face detection
                    with self._lock:
                        self._pause_calibration = not self._face_detected

                    # --- CALIBRATION PHASE ---
                    if not self._is_calibrated and self._extractor is not None:
                        current_samples = sum(len(x) for x in self._calibration_audio_buffer)
                        with self._lock:
                            self._calibration_progress = min(1.0, current_samples / float(self.CALIBRATION_SAMPLES))

                        if current_samples >= self.CALIBRATION_SAMPLES:
                            all_samples = np.concatenate(self._calibration_audio_buffer)[:self.CALIBRATION_SAMPLES]

                            # Compute master voiceprint
                            stream = self._extractor.create_stream()
                            stream.accept_waveform(self.SAMPLE_RATE, all_samples)
                            stream.input_finished()

                            if self._extractor.is_ready(stream):
                                master_emb = np.array(self._extractor.compute(stream))

                                # Dynamic threshold calibration via self-similarity analysis
                                self_similarities = []
                                seg_length = self.SAMPLE_RATE  # 1 second chunks
                                for i in range(self.CALIBRATION_SECONDS):
                                    segment = all_samples[i * seg_length: (i + 1) * seg_length]
                                    seg_stream = self._extractor.create_stream()
                                    seg_stream.accept_waveform(self.SAMPLE_RATE, segment)
                                    seg_stream.input_finished()
                                    if self._extractor.is_ready(seg_stream):
                                        seg_emb = np.array(self._extractor.compute(seg_stream))
                                        dot = np.dot(master_emb, seg_emb)
                                        norm_m = np.linalg.norm(master_emb)
                                        norm_s = np.linalg.norm(seg_emb)
                                        sim = float(dot / (norm_m * norm_s + 1e-9))
                                        self_similarities.append(sim)

                                if len(self_similarities) > 0:
                                    mean_sim = float(np.mean(self_similarities))
                                    std_sim = float(np.std(self_similarities))
                                    # Clip threshold to a stable, robust range for speaker verification
                                    threshold = float(np.clip(mean_sim - 2.5 * std_sim, 0.52, 0.63))
                                else:
                                    threshold = 0.56

                                with self._lock:
                                    self._enrolled_embedding = master_emb
                                    self._similarity_threshold = threshold
                                    self._is_calibrated = True
                                    collect_calibration = False
                                print(f"[AudioProctor] Voice calibrated! Threshold: {threshold:.3f}")

                    elif not self._is_calibrated and self._extractor is None:
                        # No WeSpeaker — run simulated calibration so UI overlay is visible
                        if self._calibration_start_time is None:
                            self._calibration_start_time = time.time()
                            print("[AudioProctor] WeSpeaker unavailable — running warm-up calibration")

                        elapsed = time.time() - self._calibration_start_time
                        with self._lock:
                            self._calibration_progress = min(1.0, elapsed / float(self._mock_calibration_seconds))

                        if elapsed >= self._mock_calibration_seconds:
                            with self._lock:
                                self._is_calibrated = True
                                self._calibration_progress = 1.0
                                self._collect_calibration = False
                            print("[AudioProctor] Warm-up calibration complete (no voice verification)")


                    # --- PROCTORING PHASE (run YAMNet + speaker verification every 0.3s) ---
                    current_time = time.time()
                    if self._is_calibrated and (current_time - last_analysis_time >= 0.3):
                        last_analysis_time = current_time

                        if not audio_data_list:
                            continue

                        new_data = np.concatenate(audio_data_list, axis=0).flatten()
                        audio_data_list.clear()

                        # Roll sliding buffer
                        buffer = np.roll(buffer, -len(new_data))
                        if len(new_data) >= self.WINDOW_SIZE:
                            buffer[:] = new_data[-self.WINDOW_SIZE:]
                        else:
                            buffer[-len(new_data):] = new_data

                        input_tensor = buffer.astype(np.float32)

                        # YAMNet classification
                        outputs = self._yamnet_session.run(None, {self._yamnet_input_name: input_tensor})
                        scores = outputs[0][0]
                        max_idx = np.argmax(scores)
                        event = self._class_names.get(max_idx, "Unknown")
                        score = float(scores[max_idx])

                        # Sound categorization
                        sound_cat = self._classify_sound(event, score)

                        # Speaker verification
                        similarity = 1.0
                        is_user = True
                        lip_sync_warn = False
                        is_speech = event in ["Speech", "Whispering", "Laughter", "Yawn", "Scream"]

                        if is_speech and score >= 0.35 and self._extractor is not None and self._enrolled_embedding is not None:
                            self._non_speech_frames = 0
                            stream = self._extractor.create_stream()
                            stream.accept_waveform(self.SAMPLE_RATE, input_tensor)
                            stream.input_finished()

                            if self._extractor.is_ready(stream):
                                curr_emb = np.array(self._extractor.compute(stream))
                                dot = np.dot(self._enrolled_embedding, curr_emb)
                                norm_e = np.linalg.norm(self._enrolled_embedding)
                                norm_c = np.linalg.norm(curr_emb)
                                raw_similarity = float(dot / (norm_e * norm_c + 1e-9))

                                if self._smoothed_similarity is None:
                                    self._smoothed_similarity = raw_similarity
                                else:
                                    self._smoothed_similarity = 0.4 * raw_similarity + 0.6 * self._smoothed_similarity

                                similarity = self._smoothed_similarity

                                with self._lock:
                                    current_threshold = self._similarity_threshold
                                    mouth_active = self._mouth_open
                                is_user = similarity >= current_threshold

                                # Adaptive voiceprint update
                                if is_user and mouth_active:
                                    curr_norm = curr_emb / (norm_c + 1e-9)
                                    self._enrolled_embedding = 0.98 * self._enrolled_embedding + 0.02 * curr_norm
                                    self._enrolled_embedding = self._enrolled_embedding / (np.linalg.norm(self._enrolled_embedding) + 1e-9)

                                # Lip-sync warning: speech verified but mouth closed
                                if is_user and not mouth_active:
                                    lip_sync_warn = True
                        else:
                            self._non_speech_frames += 1
                            if self._non_speech_frames >= 5:
                                self._smoothed_similarity = None
                                similarity = 1.0

                        with self._lock:
                            self._latest_event = event
                            self._latest_score = score
                            self._latest_similarity = similarity
                            self._is_user_voice = is_user
                            self._lip_sync_warning = lip_sync_warn
                            self._sound_category = sound_cat

        except Exception as e:
            print(f"[AudioProctor] Error in audio loop: {e}")
            with self._lock:
                self._is_calibrated = True
                self._calibration_progress = 1.0

    def reset_calibration(self):
        """Thread-safe reset of the voice calibration state."""
        with self._lock:
            self._is_calibrated = False
            self._calibration_progress = 0.0
            self._calibration_start_time = None
            self._calibration_audio_buffer.clear()
            self._collect_calibration = True
            self._pause_calibration = True
            self._enrolled_embedding = None
            self._is_recording_speech = False
            print("[AudioProctor] Voice calibration state reset by request")

