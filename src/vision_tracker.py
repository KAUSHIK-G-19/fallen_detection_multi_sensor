"""
Real-Time Vision Pose Tracker Module
Uses MediaPipe Pose Landmarker with Online 2D Kalman Filtering for smooth trajectory estimation,
aspect ratio tracking, torso angle calculation, and vertical descent velocity estimation.
"""
import os
import math
import numpy as np
import cv2
from src.kalman_filter import OnlineKalmanFilter2D


class VisionPoseTracker:
    """
    Real-time vision tracker for human pose, aspect ratio, torso orientation, and centroid descent.
    """
    def __init__(self, model_path="models/pose_landmarker_lite.task", min_detection_confidence=0.5):
        self.model_path = model_path
        self.min_confidence = min_detection_confidence
        self.landmarker = None
        self.has_mediapipe = False

        # Online Kalman filters for key points to eliminate tracking jitter
        self.centroid_kf = OnlineKalmanFilter2D(dt=1.0/30.0, process_noise=1e-3, measurement_noise=1e-2)
        self.head_kf = OnlineKalmanFilter2D(dt=1.0/30.0)
        self.hip_kf = OnlineKalmanFilter2D(dt=1.0/30.0)

        # Posture state variables
        self.prev_centroid_y = None
        self.prev_time = None
        self.centroid_vy = 0.0  # Downward velocity

        self._init_mediapipe()

    def _init_mediapipe(self):
        """Initialize MediaPipe Pose Landmarker Tasks API"""
        if not os.path.exists(self.model_path):
            print(f"Warning: MediaPipe model not found at {self.model_path}. Vision tracker will use OpenCV silhouette fallback.")
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self.min_confidence,
                min_pose_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
                num_poses=1
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.has_mediapipe = True
            print("VisionPoseTracker: MediaPipe Pose Landmarker successfully initialized!")
        except Exception as e:
            print(f"VisionPoseTracker: Failed to initialize MediaPipe: {e}. Using fallback.")
            self.has_mediapipe = False

    def process_frame(self, frame, timestamp_sec=None):
        """
        Process a single BGR video frame.
        Returns:
            features: dict of geometric and kinematic indicators
            annotated_frame: frame with skeleton and HUD overlay
        """
        h, w, _ = frame.shape
        annotated_frame = frame.copy()

        default_features = {
            'detected': False,
            'aspect_ratio': 0.5,
            'torso_angle': 0.0,
            'centroid_x': 0.5,
            'centroid_y': 0.5,
            'centroid_vy': 0.0,
            'head_y': 0.2,
            'is_laying': False,
            'posture': "UNKNOWN",
            'bbox': [0, 0, 0, 0]
        }

        if not self.has_mediapipe or self.landmarker is None:
            return self._process_fallback(frame, annotated_frame, default_features)

        try:
            import mediapipe as mp
            # Convert OpenCV BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = self.landmarker.detect(mp_image)

            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return default_features, annotated_frame

            landmarks = detection_result.pose_landmarks[0]

            # Extract normalized coordinates (x, y in [0, 1])
            coords = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks])

            # Key landmark indices (COCO/MediaPipe standard)
            # 0: nose, 11: left shoulder, 12: right shoulder, 23: left hip, 24: right hip
            # 25: left knee, 26: right knee, 27: left ankle, 28: right ankle
            nose = coords[0, :2]
            left_shoulder = coords[11, :2]
            right_shoulder = coords[12, :2]
            left_hip = coords[23, :2]
            right_hip = coords[24, :2]
            left_ankle = coords[27, :2]
            right_ankle = coords[28, :2]

            # Midpoints
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            hip_mid = (left_hip + right_hip) / 2.0
            raw_centroid = (shoulder_mid + hip_mid) / 2.0

            # Online Kalman Filter for Centroid
            smooth_cx, smooth_cy, vx, vy = self.centroid_kf.update(raw_centroid[0], raw_centroid[1])
            smooth_head_x, smooth_head_y, _, _ = self.head_kf.update(nose[0], nose[1])
            smooth_hip_x, smooth_hip_y, _, _ = self.hip_kf.update(hip_mid[0], hip_mid[1])

            # Centroid vertical velocity (downward positive)
            self.centroid_vy = vy

            # Bounding Box
            visible_mask = coords[:, 2] > 0.3
            if np.any(visible_mask):
                valid_coords = coords[visible_mask, :2]
            else:
                valid_coords = coords[:, :2]

            x_min = max(0.0, float(np.min(valid_coords[:, 0])))
            y_min = max(0.0, float(np.min(valid_coords[:, 1])))
            x_max = min(1.0, float(np.max(valid_coords[:, 0])))
            y_max = min(1.0, float(np.max(valid_coords[:, 1])))

            box_w = max(0.05, x_max - x_min)
            box_h = max(0.05, y_max - y_min)

            # Aspect Ratio: Width / Height
            # Normal standing: 0.3 - 0.7. Lying on floor: 1.3 - 3.0
            aspect_ratio = float(box_w / box_h)

            # Torso Angle relative to vertical
            dx = float(shoulder_mid[0] - hip_mid[0])
            dy = float(shoulder_mid[1] - hip_mid[1])
            # angle with vertical axis (0 deg = standing upright, 90 deg = horizontal lying)
            angle_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
            torso_angle = math.degrees(angle_rad)

            # Posture Classification
            is_laying = (aspect_ratio >= 1.15) or (torso_angle >= 50.0 and abs(smooth_head_y - smooth_hip_y) < 0.2)

            if is_laying:
                posture = "LYING"
            elif aspect_ratio <= 0.65 and torso_angle < 30.0:
                posture = "STANDING"
            elif 0.65 < aspect_ratio < 1.15 and torso_angle < 45.0:
                posture = "SITTING"
            else:
                posture = "TRANSITION"

            # Pixel bounding box
            px_bbox = [int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)]

            features = {
                'detected': True,
                'aspect_ratio': aspect_ratio,
                'torso_angle': torso_angle,
                'centroid_x': smooth_cx,
                'centroid_y': smooth_cy,
                'centroid_vy': self.centroid_vy,
                'head_y': smooth_head_y,
                'is_laying': is_laying,
                'posture': posture,
                'bbox': px_bbox
            }

            # Render overlay
            self._draw_pose(annotated_frame, coords, px_bbox, features)
            return features, annotated_frame

        except Exception as e:
            # In case of any inference glitch, fallback gracefully
            return default_features, annotated_frame

    def _process_fallback(self, frame, annotated_frame, default_features):
        """Lightweight OpenCV contour fallback if MediaPipe is unavailable"""
        return default_features, annotated_frame

    def _draw_pose(self, frame, coords, bbox, features):
        """Draw skeletal joints, bounding box, and telemetry on frame"""
        h, w, _ = frame.shape

        # Draw Bounding Box (Red if lying, Green if standing, Amber if transition)
        if features['is_laying']:
            box_color = (0, 0, 255)  # Red
        elif features['posture'] == "STANDING":
            box_color = (0, 255, 0)  # Green
        else:
            box_color = (0, 215, 255)  # Amber

        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Draw posture tag
        label = f"{features['posture']} (AR:{features['aspect_ratio']:.2f}, {features['torso_angle']:.0f}deg)"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # Draw skeletal connections
        connections = [
            (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
            (11, 13), (13, 15),                       # Left Arm
            (12, 14), (14, 16),                       # Right Arm
            (23, 25), (25, 27),                       # Left Leg
            (24, 26), (26, 28),                       # Right Leg
            (0, 11), (0, 12)                          # Head
        ]

        for s_idx, e_idx in connections:
            if s_idx < len(coords) and e_idx < len(coords):
                if coords[s_idx, 2] > 0.4 and coords[e_idx, 2] > 0.4:
                    pt1 = (int(coords[s_idx, 0] * w), int(coords[s_idx, 1] * h))
                    pt2 = (int(coords[e_idx, 0] * w), int(coords[e_idx, 1] * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

        # Draw joints
        for i in range(len(coords)):
            if coords[i, 2] > 0.4:
                pt = (int(coords[i, 0] * w), int(coords[i, 1] * h))
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)

    def close(self):
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
