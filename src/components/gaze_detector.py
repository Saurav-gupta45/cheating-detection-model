"""
GazeDetector: PyTorch MLP gaze classifier component.

Loads a pre-trained MLP model that maps 6 normalized gaze/head-pose features
to a single sigmoid probability (Safe vs Cheating).

Features extracted from MediaPipe FaceMesh landmarks:
  - left_gaze_x, left_gaze_y  (iris 468 relative to eye corners 33/133/159/145)
  - right_gaze_x, right_gaze_y (iris 473 relative to eye corners 362/263/386/374)
  - head_yaw, head_pitch       (nose 1 relative to head edges 234/454/10/152)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from src.utils import get_artifact_path


class GazeClassifier(nn.Module):
    """Multi-Layer Perceptron: 6 inputs → 32 → 16 → 1 sigmoid output."""

    def __init__(self, input_dim=6):
        super(GazeClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class GazeDetector:
    """
    Wraps the PyTorch gaze classifier for real-time inference.
    
    Usage:
        detector = GazeDetector()
        result = detector.process(landmarks)
        # result = { 'safe_probability': 0.92, 'is_cheating': False, 'features': [...] }
    """

    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model_path = get_artifact_path("proctor_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gaze model not found at {model_path}")

        self.model = GazeClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"[GazeDetector] Model loaded on {self.device}")

    @staticmethod
    def extract_features(landmarks):
        """
        Extract 6 normalized gaze features from MediaPipe FaceMesh landmarks.
        
        Args:
            landmarks: MediaPipe face_landmarks.landmark list
            
        Returns:
            List of 6 floats, or None if landmarks are invalid.
        """
        if landmarks is None:
            return None

        try:
            # Left Eye: corners 33/133, top/bottom 159/145, iris 468
            lx33, rx133 = landmarks[33], landmarks[133]
            ty159, by145 = landmarks[159], landmarks[145]
            iris468 = landmarks[468]

            left_gaze_x = (iris468.x - lx33.x) / (rx133.x - lx33.x + 1e-6)
            left_gaze_y = (iris468.y - ty159.y) / (by145.y - ty159.y + 1e-6)

            # Right Eye: corners 362/263, top/bottom 386/374, iris 473
            lx362, rx263 = landmarks[362], landmarks[263]
            ty386, by374 = landmarks[386], landmarks[374]
            iris473 = landmarks[473]

            right_gaze_x = (iris473.x - lx362.x) / (rx263.x - lx362.x + 1e-6)
            right_gaze_y = (iris473.y - ty386.y) / (by374.y - ty386.y + 1e-6)

            # Head pose: nose 1, edges 234/454, forehead 10, chin 152
            nose = landmarks[1]
            left_edge, right_edge = landmarks[234], landmarks[454]
            forehead, chin = landmarks[10], landmarks[152]

            head_yaw = (nose.x - left_edge.x) / (right_edge.x - left_edge.x + 1e-6)
            head_pitch = (nose.y - forehead.y) / (chin.y - forehead.y + 1e-6)

            return [left_gaze_x, left_gaze_y, right_gaze_x, right_gaze_y, head_yaw, head_pitch]
        except (IndexError, AttributeError):
            return None

    def process(self, landmarks):
        """
        Run gaze classification on extracted landmarks.
        
        Args:
            landmarks: MediaPipe face_landmarks.landmark list
            
        Returns:
            dict with keys: safe_probability, is_cheating, features
            Returns None if no valid features could be extracted.
        """
        features = self.extract_features(landmarks)
        if features is None:
            return None

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            safe_probability = output.item()

        return {
            'safe_probability': safe_probability,
            'is_cheating': safe_probability < 0.5,
            'features': features
        }
