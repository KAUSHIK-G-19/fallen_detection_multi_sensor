"""
Real-Time Multimodal Fall Detection System
Production-ready deployment application fusing live camera pose tracking,
streaming IMU sensor ring buffer, online Kalman filtering, and spatiotemporal decision engine.

Usage:
    python realtime_deploy.py --source 0               # Run with physical webcam
    python realtime_deploy.py --source synthetic        # Run with built-in realistic animated human simulation
    python realtime_deploy.py --source video.mp4        # Run with video file
"""
import os
import sys
import time
import argparse
import threading
import pickle
import numpy as np
import cv2

# Project modules
from src.vision_tracker import VisionPoseTracker
from src.sensor_streamer import RealtimeSensorRingBuffer, RealisticSensorStreamer
from src.decision_engine import FallDecisionEngine

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
ALERT_DIR = os.path.join(PROJECT_DIR, "alerts")
os.makedirs(ALERT_DIR, exist_ok=True)


class RealtimeFallDetectorApp:
    def __init__(self, source=0, enable_sound=True, inactivity_sec=1.8, record_path=None):
        self.source = source
        self.enable_sound = enable_sound
        self.inactivity_sec = inactivity_sec
        self.record_path = record_path
        self.running = False

        print("\n" + "="*60)
        print("REAL-TIME MULTIMODAL FALL DETECTION SYSTEM")
        print("="*60)

        # 1. Initialize Vision Tracker (MediaPipe Pose + 2D Kalman Filter)
        model_task_path = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
        self.vision_tracker = VisionPoseTracker(model_path=model_task_path)

        # 2. Initialize Sensor Streamer and Ring Buffer (100 Hz, 2.0s = 200 samples)
        self.sensor_streamer = RealisticSensorStreamer()
        self.ring_buffer = RealtimeSensorRingBuffer(capacity=200, sr=100)

        # 3. Initialize Spatiotemporal Decision Engine
        self.decision_engine = FallDecisionEngine(
            inactivity_timeout_sec=self.inactivity_sec,
            alert_dir=ALERT_DIR
        )

        # 4. Load Trained Machine Learning Classifier
        self.scaler = None
        self.ml_classifier = None
        self._load_ml_model()

        # Telemetry
        self.fps = 0.0
        self.latency_ms = 0.0
        self.alert_count = 0
        self.synthetic_angle = 0.0
        self.synthetic_fall = False
        self.synthetic_fall_progress = 0

    def _load_ml_model(self):
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        model_path = os.path.join(MODELS_DIR, "fall_classifier.pkl")

        if os.path.exists(scaler_path) and os.path.exists(model_path):
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(model_path, "rb") as f:
                    self.ml_classifier = pickle.load(f)
                print("[SYSTEM] Trained Machine Learning Fall Classifier successfully loaded.")
            except Exception as e:
                print(f"[WARNING] Could not load ML model: {e}. Relying on Spatiotemporal Decision Engine.")
        else:
            print("[INFO] Model weights not found in models/; using Spatiotemporal Decision Engine.")

    def _play_alert_sound(self):
        """Asynchronous non-blocking emergency siren / alert sound"""
        if not self.enable_sound:
            return

        def _sound_thread():
            try:
                import winsound
                # Play urgent two-tone alert
                winsound.Beep(1200, 200)
                winsound.Beep(800, 200)
                winsound.Beep(1200, 300)
            except Exception:
                try:
                    import sounddevice as sd
                    fs = 44100
                    duration = 0.5
                    t = np.linspace(0, duration, int(fs * duration), False)
                    tone = np.sin(2 * np.pi * 880 * t) * 0.3
                    sd.play(tone, fs)
                    sd.wait()
                except Exception:
                    pass

        threading.Thread(target=_sound_thread, daemon=True).start()

    def _generate_synthetic_frame(self, w=720, h=480):
        """
        Creates a realistic simulated human stick figure walking, transitioning, or falling.
        Used when running with `--source synthetic` or when no webcam hardware is attached.
        """
        frame = np.full((h, w, 3), 30, dtype=np.uint8)

        # Floor grid line
        floor_y = int(h * 0.82)
        cv2.line(frame, (0, floor_y), (w, floor_y), (80, 80, 80), 2)
        cv2.putText(frame, "[SYNTHETIC SIMULATION MODE - PRESS 'F' TO SIMULATE FALL]", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        center_x = int(w * 0.45)

        if self.synthetic_fall:
            self.synthetic_fall_progress += 1
            # Angle tilts from 0 to 90 degrees as body drops
            tilt = min(90.0, self.synthetic_fall_progress * 4.5)
            y_offset = min(int(h * 0.35), int(self.synthetic_fall_progress * 8.0))
        else:
            tilt = 0.0
            y_offset = int(6.0 * np.sin(self.synthetic_angle))
            self.synthetic_angle += 0.12

        # Draw simulated human figure
        rad = np.radians(tilt)
        torso_len = 100
        head_radius = 20

        base_y = floor_y - 20
        hip_x = center_x
        hip_y = base_y - 60 + y_offset

        shoulder_x = int(hip_x + torso_len * np.sin(rad))
        shoulder_y = int(hip_y - torso_len * np.cos(rad))

        head_x = int(shoulder_x + 30 * np.sin(rad))
        head_y = int(shoulder_y - 30 * np.cos(rad))

        # Color: Cyan if normal, Red if falling
        fig_color = (0, 0, 255) if tilt > 45 else (255, 200, 0)

        # Head
        cv2.circle(frame, (head_x, head_y), head_radius, fig_color, -1)
        # Torso
        cv2.line(frame, (shoulder_x, shoulder_y), (hip_x, hip_y), fig_color, 6)
        # Arms
        cv2.line(frame, (shoulder_x, shoulder_y), (shoulder_x - 30, shoulder_y + 40), fig_color, 4)
        cv2.line(frame, (shoulder_x, shoulder_y), (shoulder_x + 30, shoulder_y + 40), fig_color, 4)
        # Legs
        if tilt > 45:
            cv2.line(frame, (hip_x, hip_y), (hip_x - 60, floor_y), fig_color, 4)
            cv2.line(frame, (hip_x, hip_y), (hip_x - 40, floor_y), fig_color, 4)
        else:
            leg_swing = int(20 * np.sin(self.synthetic_angle))
            cv2.line(frame, (hip_x, hip_y), (hip_x - 20 + leg_swing, floor_y), fig_color, 4)
            cv2.line(frame, (hip_x, hip_y), (hip_x + 20 - leg_swing, floor_y), fig_color, 4)

        return frame

    def run(self):
        self.running = True
        is_synthetic = (str(self.source).lower() == "synthetic")

        cap = None
        if not is_synthetic:
            try:
                src_val = int(self.source)
            except ValueError:
                src_val = self.source
            cap = cv2.VideoCapture(src_val)
            if not cap.isOpened():
                print(f"[WARNING] Unable to open video source '{self.source}'. Switching to SYNTHETIC mode.")
                is_synthetic = True
            else:
                print(f"[SYSTEM] Live video stream opened from source: {self.source}")

        writer = None
        prev_time = time.time()

        print("\n[READY] System active. Press 'F' to trigger simulated fall shock, 'R' to reset, 'Q' to quit.\n")

        try:
            while self.running:
                loop_start = time.time()

                # 1. Acquire Frame
                if is_synthetic:
                    frame = self._generate_synthetic_frame()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video file if ended
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break

                # 2. Acquire Sensor Sample & Update Ring Buffer (100 Hz)
                raw_sensor = self.sensor_streamer.get_next_sample()
                sensor_feats = self.ring_buffer.add_sample(raw_sensor, timestamp=loop_start)

                # 3. Vision Pose Tracking (MediaPipe + 2D Kalman Filter)
                vision_feats, annotated_frame = self.vision_tracker.process_frame(frame, timestamp_sec=loop_start)

                # In synthetic mode, map geometry from simulated figure
                if is_synthetic:
                    if self.synthetic_fall:
                        vision_feats['is_laying'] = (self.synthetic_fall_progress > 15)
                        vision_feats['aspect_ratio'] = 1.8 if self.synthetic_fall_progress > 15 else 0.8
                        vision_feats['torso_angle'] = min(90.0, self.synthetic_fall_progress * 4.5)
                        vision_feats['posture'] = "LYING" if self.synthetic_fall_progress > 15 else "TRANSITION"
                    else:
                        vision_feats['is_laying'] = False
                        vision_feats['aspect_ratio'] = 0.45
                        vision_feats['torso_angle'] = 8.0
                        vision_feats['posture'] = "STANDING"

                # 4. Spatiotemporal Decision Engine Update
                decision = self.decision_engine.update(vision_feats, sensor_feats, frame=annotated_frame)

                # 5. Handle Alert Trigger
                if decision['trigger_alert']:
                    self.alert_count += 1
                    print(f"\n🚨 [EMERGENCY DISPATCH] FALL CONFIRMED! Alert #{self.alert_count} triggered.")
                    self._play_alert_sound()

                # 6. Compute FPS & Latency
                loop_end = time.time()
                self.latency_ms = (loop_end - loop_start) * 1000.0
                dt = loop_end - prev_time
                prev_time = loop_end
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

                # 7. Render Modern HUD
                hud_frame = self._render_hud(annotated_frame, vision_feats, sensor_feats, decision)

                # Write to video if recording
                if self.record_path:
                    if writer is None:
                        h, w, _ = hud_frame.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(self.record_path, fourcc, 25.0, (w, h))
                    writer.write(hud_frame)

                # Display Window
                cv2.imshow("Multimodal Real-Time Fall Detection System", hud_frame)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('f') or key == ord('F'):
                    print("[EVENT] User injected simulated fall shock!")
                    self.sensor_streamer.trigger_fall()
                    if is_synthetic:
                        self.synthetic_fall = True
                        self.synthetic_fall_progress = 0
                elif key == ord('r') or key == ord('R'):
                    print("[EVENT] Resetting state to NORMAL.")
                    self.decision_engine.reset()
                    self.synthetic_fall = False
                    self.synthetic_fall_progress = 0

        finally:
            self.running = False
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            self.vision_tracker.close()
            cv2.destroyAllWindows()
            print("[SHUTDOWN] Real-time detection system cleanly terminated.")

    def _render_hud(self, frame, vision_feats, sensor_feats, decision):
        """Render high-tech heads-up display overlay"""
        h, w, _ = frame.shape
        out = frame.copy()

        state = decision['state']
        confidence = decision['confidence']

        # Determine state color scheme
        if state == FallDecisionEngine.STATE_CONFIRMED:
            banner_bg = (0, 0, 200)  # Crimson Red
            banner_text = f"EMERGENCY: FALL DETECTED ({confidence:.0f}%)"
            box_border = (0, 0, 255)
        elif state == FallDecisionEngine.STATE_INACTIVITY:
            banner_bg = (0, 100, 255)  # Dark Orange
            banner_text = f"VERIFYING IMMOBILITY ({confidence:.0f}%)"
            box_border = (0, 140, 255)
        elif state == FallDecisionEngine.STATE_COLLAPSE:
            banner_bg = (0, 165, 255)  # Orange
            banner_text = f"POSTURE COLLAPSED ({confidence:.0f}%)"
            box_border = (0, 180, 255)
        elif state == FallDecisionEngine.STATE_IMPACT:
            banner_bg = (0, 215, 255)  # Amber Yellow
            banner_text = f"IMPACT SHOCK DETECTED ({confidence:.0f}%)"
            box_border = (0, 215, 255)
        else:
            banner_bg = (30, 120, 30)  # Deep Emerald Green
            banner_text = "MONITORING ACTIVE: NORMAL"
            box_border = (0, 255, 0)

        # 1. Top Banner
        cv2.rectangle(out, (0, 0), (w, 45), banner_bg, -1)
        cv2.putText(out, banner_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(out, f"FPS: {self.fps:.1f}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 2. Side Telemetry Glass Panel
        panel_w = 260
        panel_h = 240
        panel_x = w - panel_w - 15
        panel_y = 60

        # Semi-transparent dark overlay
        overlay = out.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)
        cv2.rectangle(out, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), box_border, 1)

        # Telemetry Text
        lines = [
            ("TELEMETRY", f"{self.latency_ms:.1f} ms"),
            ("Posture:", vision_feats.get('posture', 'N/A')),
            ("Aspect Ratio:", f"{vision_feats.get('aspect_ratio', 0.0):.2f}"),
            ("Torso Angle:", f"{vision_feats.get('torso_angle', 0.0):.0f} deg"),
            ("Accel SVM:", f"{sensor_feats.get('svm_acc', 9.81):.2f} m/s2 ({sensor_feats.get('svm_acc', 9.81)/9.81:.2f}g)"),
            ("Gyro Mag:", f"{sensor_feats.get('svm_gyro', 0.0):.1f} deg/s"),
            ("Heart Rate:", f"{int(sensor_feats.get('heart_rate', 75))} BPM"),
            ("Alerts Fired:", str(self.alert_count))
        ]

        ty = panel_y + 22
        for title, val in lines:
            if title == "TELEMETRY":
                cv2.putText(out, f"{title} [{val}]", (panel_x + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                cv2.putText(out, title, (panel_x + 10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                cv2.putText(out, val, (panel_x + 115, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            ty += 24

        # Risk Confidence Meter Bar
        bar_x = panel_x + 10
        bar_y = panel_y + panel_h - 22
        bar_w = panel_w - 20
        bar_h = 10
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
        fill_w = int((confidence / 100.0) * bar_w)
        meter_color = (0, 0, 255) if confidence > 70 else (0, 200, 255) if confidence > 30 else (0, 255, 0)
        if fill_w > 0:
            cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), meter_color, -1)

        # Flashing Border in Confirmed Fall state
        if state == FallDecisionEngine.STATE_CONFIRMED and (int(time.time() * 3) % 2 == 0):
            cv2.rectangle(out, (0, 0), (w, h), (0, 0, 255), 8)

        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Multimodal Fall Detection System")
    parser.add_argument("--source", type=str, default="synthetic",
                        help="Video source: '0' for webcam, 'synthetic' for simulation, or path to video file")
    parser.add_argument("--sound", action="store_true", default=True,
                        help="Enable audio alarm on fall detection")
    parser.add_argument("--no-sound", action="store_false", dest="sound",
                        help="Disable audio alarm")
    parser.add_argument("--inactivity", type=float, default=1.8,
                        help="Post-fall inactivity confirmation timeout in seconds (default: 1.8)")
    parser.add_argument("--record", type=str, default=None,
                        help="Path to save output video (.mp4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = RealtimeFallDetectorApp(
        source=args.source,
        enable_sound=args.sound,
        inactivity_sec=args.inactivity,
        record_path=args.record
    )
    app.run()
