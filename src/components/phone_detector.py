"""
PhoneDetector: YOLOv8 cell phone detection wrapper.

Uses YOLOv8 Small model filtered exclusively for COCO Class 67 (cell phone).
Generates alerts if confidence >= 30%.
"""

import os
from ultralytics import YOLO
from src.utils import get_artifact_path


class PhoneDetector:
    """
    Wraps YOLOv8s for real-time cell phone detection.
    
    Usage:
        detector = PhoneDetector()
        result = detector.process(frame)
        # result = { 'detected': True, 'boxes': [{'x1':..., 'y1':..., 'x2':..., 'y2':..., 'confidence': 0.85}] }
    """

    PHONE_CLASS_ID = 67
    CONFIDENCE_THRESHOLD = 0.30

    def __init__(self):
        model_path = get_artifact_path("yolov8s.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLOv8 model not found at {model_path}")

        self.model = YOLO(model_path)
        print("[PhoneDetector] YOLOv8s model loaded")

    def process(self, frame):
        """
        Detect cell phones in a video frame.
        
        Args:
            frame: BGR OpenCV frame
            
        Returns:
            dict with keys:
                detected: bool
                boxes: list of dicts with x1, y1, x2, y2, confidence
        """
        results = self.model(frame, classes=[self.PHONE_CLASS_ID], verbose=False)
        
        detected = False
        boxes = []
        
        detections = results[0].boxes
        for box in detections:
            confidence = float(box.conf[0])
            if confidence >= self.CONFIDENCE_THRESHOLD:
                detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append({
                    'x1': x1, 'y1': y1,
                    'x2': x2, 'y2': y2,
                    'confidence': confidence
                })

        return {
            'detected': detected,
            'boxes': boxes
        }
