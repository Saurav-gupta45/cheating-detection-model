"""
FaceDetector: MediaPipe FaceLandmarker multi-face coordinator.

Uses the new MediaPipe Tasks API (mediapipe >= 0.10.x).
Detects up to 5 simultaneous faces with 478 landmarks (including iris).
Shared across gaze detection, multi-face tracking, and lip-sync features.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as bo

from src.utils import get_artifact_path


class FaceDetector:
    """
    Wraps MediaPipe FaceLandmarker for multi-face detection.
    
    Uses the new Tasks API which requires a .task model file.
    
    Usage:
        detector = FaceDetector()
        result = detector.process(frame)
    """

    # Lip distance threshold (mouth open vs closed)
    LIP_OPEN_THRESHOLD = 0.025

    def __init__(self, max_faces=5):
        model_path = get_artifact_path("face_landmarker.task")

        if not os.path.exists(model_path):
            # Auto-download the model
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            print(f"[FaceDetector] Downloading face_landmarker.task...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
            print(f"[FaceDetector] Downloaded successfully")

        options = vision.FaceLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=model_path),
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.max_faces = max_faces
        print(f"[FaceDetector] Initialized (max_faces={max_faces})")

    def process(self, frame):
        """
        Detect faces in a BGR frame.
        
        Args:
            frame: BGR OpenCV frame
            
        Returns:
            dict with keys:
                num_faces: int
                face_detected: bool
                landmarks_list: list of landmark lists (one per face)
                    Each landmark has .x, .y, .z attributes (normalized 0-1)
                mouth_open: bool (for first face only)
                lip_dist_norm: float (normalized lip distance for first face)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)

        num_faces = len(result.face_landmarks) if result.face_landmarks else 0
        landmarks_list = result.face_landmarks if result.face_landmarks else []
        mouth_open = False
        lip_dist_norm = 0.0

        # Calculate lip distance for the first (primary) face
        if num_faces >= 1:
            landmarks = landmarks_list[0]
            try:
                forehead = np.array([landmarks[10].x, landmarks[10].y])
                chin = np.array([landmarks[152].x, landmarks[152].y])
                face_height = np.linalg.norm(forehead - chin)

                upper_lip = np.array([landmarks[13].x, landmarks[13].y])
                lower_lip = np.array([landmarks[14].x, landmarks[14].y])
                lip_distance = np.linalg.norm(upper_lip - lower_lip)

                lip_dist_norm = lip_distance / (face_height + 1e-6)
                mouth_open = lip_dist_norm >= self.LIP_OPEN_THRESHOLD
            except (IndexError, AttributeError):
                pass

        return {
            'num_faces': num_faces,
            'face_detected': num_faces > 0,
            'landmarks_list': landmarks_list,
            'mouth_open': mouth_open,
            'lip_dist_norm': lip_dist_norm
        }

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
