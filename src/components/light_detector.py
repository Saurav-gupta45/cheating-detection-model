"""
LightDetector: Grayscale frame brightness analyzer.

Detects camera-covered conditions and sudden brightness changes
(indicative of tab switching or environmental lighting shifts).

Uses a rolling deque of 30 brightness samples as baseline.
"""

import cv2
import numpy as np
from collections import deque


class LightDetector:
    """
    Monitors frame brightness for suspicious lighting changes.
    
    Usage:
        detector = LightDetector()
        result = detector.process(frame)
        # result = { 'brightness': 120.5, 'baseline': 118.2, 'alert': None }
    """

    DARKNESS_LIMIT = 8.0       # Below this → camera covered / pitch black
    CHANGE_THRESHOLD = 10.0    # Sudden deviation from baseline

    def __init__(self, history_size=30):
        self.brightness_history = deque(maxlen=history_size)
        self.history_size = history_size
        self._initialized = False
        print("[LightDetector] Initialized")

    def process(self, frame):
        """
        Analyze frame brightness and compare to rolling baseline.
        
        Args:
            frame: BGR OpenCV frame
            
        Returns:
            dict with keys:
                brightness: float (current mean brightness 0-255)
                baseline: float (rolling average brightness)
                alert: str or None ('CAMERA_COVERED', 'BRIGHTNESS_SPIKE', 'BRIGHTNESS_DROP')
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_brightness = float(np.mean(gray))

        # Pre-fill history on first few frames to avoid false alarms
        if not self._initialized:
            for _ in range(self.history_size):
                self.brightness_history.append(current_brightness)
            self._initialized = True

        baseline_brightness = float(np.mean(self.brightness_history))

        alert = None
        if current_brightness < self.DARKNESS_LIMIT:
            alert = 'CAMERA_COVERED'
        elif current_brightness - baseline_brightness > self.CHANGE_THRESHOLD:
            alert = 'BRIGHTNESS_SPIKE'
        elif baseline_brightness - current_brightness > self.CHANGE_THRESHOLD:
            alert = 'BRIGHTNESS_DROP'

        # Push current brightness to rolling buffer
        self.brightness_history.append(current_brightness)

        return {
            'brightness': current_brightness,
            'baseline': baseline_brightness,
            'alert': alert
        }
