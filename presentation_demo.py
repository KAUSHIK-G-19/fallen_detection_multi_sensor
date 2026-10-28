"""
Interactive 5-Stage Multimodal Fall Detection Presentation System
Designed for Project Demonstrations, Vivas, and Defense Presentations.

Progression Stages:
  Stage 1: Single Sensor Analysis (Only IMU Sensor) -> Detects Human Presence & Motion (Active vs. Stationary)
  Stage 2: Two Sensor Integration (IMU + MPU-6050) -> Detects Body Position & 3D Tilt Orientation
  Stage 3: Tri-Sensor Integration (IMU + MPU-6050 + IR) -> Detects Position & Posture (Stand, Sit, Bend, Lie)
  Stage 4: Camera Stick Figure / Skeleton Analysis -> Pure Vision Fall Detection via Aspect Ratio & Descent Vy
  Stage 5: Full Multimodal Late-Fusion System -> Fuses All Sensors + Camera Stick Pose + Live Video + Emergency Alert

Usage:
  python presentation_demo.py             # Launches interactive suite (switch stages with keys 1-5)
  python presentation_demo.py --stage 1   # Launch directly into Stage 1
  python presentation_demo.py --stage 2   # Launch directly into Stage 2
  python presentation_demo.py --stage 3   # Launch directly into Stage 3
  python presentation_demo.py --stage 4   # Launch directly into Stage 4
  python presentation_demo.py --stage 5   # Launch directly into Stage 5
"""
import os
import sys
import time
import math
import argparse
import collections
import numpy as np
import cv2

# Project Modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.sensor_streamer import RealisticSensorStreamer, RealtimeSensorRingBuffer, SerialSensorStreamer, FallAllDStreamer
from src.vision_tracker import VisionPoseTracker
from src.decision_engine import FallDecisionEngine
from realtime_deploy import RealtimeFallDetectorApp

WIDTH = 1000
HEIGHT = 650


class PresentationDemoApp:
    def __init__(self, initial_stage=1, source="synthetic", port=None, dataset=None):
        self.stage = int(initial_stage)
        self.source = source
        self.port = port
        self.dataset = dataset
        self.running = True

        # Initialize sensors & streams
        if port:
            self.streamer = SerialSensorStreamer(port=port)
        elif self.dataset and str(self.dataset).lower() == "fallalld":
            self.streamer = FallAllDStreamer()
        else:
            self.streamer = RealisticSensorStreamer()
        self.buffer = RealtimeSensorRingBuffer(capacity=200, sr=100)

        # Vision tracker
        model_task = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_lite.task")
        self.vision_tracker = VisionPoseTracker(model_path=model_task)
        self.decision_engine = FallDecisionEngine(inactivity_timeout_sec=1.8, alert_dir="alerts")

        # Rolling waveforms for Stage 1 & 2
        self.history_raw_svm = collections.deque(maxlen=160)
        self.history_filt_svm = collections.deque(maxlen=160)
        self.history_tilt = collections.deque(maxlen=160)
        self.history_ir = collections.deque(maxlen=160)

        # Stage 4 & 5 controls
        self.synthetic_fall = False
        self.fall_progress = 0
        self.anim_t = 0.0

        # Video capture if not synthetic
        self.cap = None
        self.is_synthetic = (str(source).lower() == "synthetic")
        if not self.is_synthetic:
            if os.path.isdir(str(source)):
                from src.vision_tracker import ImageSequenceCapture
                self.cap = ImageSequenceCapture(source)
                if not self.cap.isOpened():
                    print(f"[WARN] No image files found in {source}. Falling back to synthetic.")
                    self.is_synthetic = True
                else:
                    print(f"[SYSTEM] Image sequence stream opened from: {source} ({len(self.cap.images)} frames)")
            else:
                try:
                    src_val = int(source)
                except ValueError:
                    src_val = source
                self.cap = cv2.VideoCapture(src_val)
                if not self.cap.isOpened():
                    print(f"[WARN] Cannot open camera {source}. Falling back to synthetic mode.")
                    self.is_synthetic = True

    def run(self):
        print("\n" + "="*70)
        print("  MULTIMODAL FALL DETECTION - 5-STAGE PRESENTATION SUITE")
        print("="*70)
        print("Hotkeys to switch stages instantly:")
        print("  [1] Stage 1: Single IMU (Human Presence & Motion)")
        print("  [2] Stage 2: Dual IMU + MPU-6050 (Position & 3D Tilt Angle)")
        print("  [3] Stage 3: Tri-Sensor IMU + MPU + IR (Position & Posture Classification)")
        print("  [4] Stage 4: Camera Stick Figure (Vision-Only Fall Detection)")
        print("  [5] Stage 5: Full Multimodal Late-Fusion Integration")
        print("  [F] Trigger Fall Shock   [R] Reset   [Q] Quit\n")

        cv2.namedWindow("Multimodal Fall Detection - Presentation Suite", cv2.WINDOW_AUTOSIZE)

        while self.running:
            loop_start = time.time()
            self.anim_t += 0.03

            # Acquire sensor data
            raw_sample = self.streamer.get_next_sample()
            ir_dist = getattr(self.streamer, 'latest_ir_distance', 1.65)
            sensor_feats = self.buffer.add_sample(raw_sample, timestamp=loop_start, ir_distance=ir_dist)

            # Update rolling waveforms
            raw_acc_mag = float(np.linalg.norm(raw_sample[:3]))
            self.history_raw_svm.append(raw_acc_mag)
            self.history_filt_svm.append(sensor_feats['svm_acc'])
            self.history_tilt.append(sensor_feats['tilt_angle'])
            self.history_ir.append(sensor_feats['ir_distance'])

            # Render stage canvas
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            canvas[:] = (22, 24, 30)  # Slate dark background

            # Render persistent top header & bottom controls
            self._render_header(canvas)
            self._render_bottom_nav(canvas)

            # Render active stage content
            if self.stage == 1:
                self._render_stage_1(canvas, raw_sample, sensor_feats)
            elif self.stage == 2:
                self._render_stage_2(canvas, raw_sample, sensor_feats)
            elif self.stage == 3:
                self._render_stage_3(canvas, sensor_feats)
            elif self.stage == 4:
                self._render_stage_4(canvas, loop_start)
            elif self.stage == 5:
                self._render_stage_5(canvas, loop_start, raw_sample, sensor_feats)

            cv2.imshow("Multimodal Fall Detection - Presentation Suite", canvas)

            # Process key events
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                self.stage = 1
                print("[SWITCH] -> Stage 1: Single IMU Sensor (Presence & Motion)")
            elif key == ord('2'):
                self.stage = 2
                print("[SWITCH] -> Stage 2: Dual IMU + MPU-6050 (Position & 3D Tilt)")
            elif key == ord('3'):
                self.stage = 3
                print("[SWITCH] -> Stage 3: Tri-Sensor IMU + MPU + IR (Position & Posture)")
            elif key == ord('4'):
                self.stage = 4
                print("[SWITCH] -> Stage 4: Camera Stick Figure (Vision Fall Detection)")
            elif key == ord('5'):
                self.stage = 5
                print("[SWITCH] -> Stage 5: Full Multimodal Late-Fusion Integration")
            elif key == ord('f') or key == ord('F'):
                print("[EVENT] Simulated Fall Triggered!")
                self.streamer.trigger_fall()
                self.synthetic_fall = True
                self.fall_progress = 0
            elif key == ord('r') or key == ord('R'):
                print("[EVENT] Reset to Normal / Standing")
                self.streamer.set_posture_mode('standing')
                self.decision_engine.reset()
                self.synthetic_fall = False
                self.fall_progress = 0
            elif key == ord('w') or key == ord('W'):
                self.streamer.set_posture_mode('walking')
            elif key == ord('s') or key == ord('S'):
                self.streamer.set_posture_mode('standing')
            elif key == ord('b') or key == ord('B'):
                self.streamer.set_posture_mode('bending')
            elif key == ord('t') or key == ord('T'):
                self.streamer.set_posture_mode('sitting')
            elif key == ord('l') or key == ord('L'):
                self.streamer.set_posture_mode('lying')

        if self.cap:
            self.cap.release()
        self.vision_tracker.close()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Presentation suite closed cleanly.")

    # -----------------------------------------------------------------
    # UI Component: Top Header
    # -----------------------------------------------------------------
    def _render_header(self, canvas):
        cv2.rectangle(canvas, (0, 0), (WIDTH, 52), (14, 16, 20), -1)
        cv2.line(canvas, (0, 52), (WIDTH, 52), (50, 60, 75), 1)

        stage_titles = {
            1: "STAGE 1 / 5: SINGLE IMU SENSOR ANALYSIS (PRESENCE & MOTION DETECTION)",
            2: "STAGE 2 / 5: DUAL SENSOR INTEGRATION: IMU + MPU-6050 (POSITION & ORIENTATION)",
            3: "STAGE 3 / 5: TRI-SENSOR INTEGRATION: IMU + MPU + IR (POSITION & POSTURE CLASSIFICATION)",
            4: "STAGE 4 / 5: CAMERA STICK SKELETON ANALYSIS (VISION-ONLY FALL DETECTION)",
            5: "STAGE 5 / 5: FULL MULTIMODAL LATE-FUSION (SENSORS + STICK POSE + VIDEO + ALARM)"
        }
        title = stage_titles.get(self.stage, "FALL DETECTION ANALYSIS")
        cv2.putText(canvas, title, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)

        # Stage pills
        for s in range(1, 6):
            px = WIDTH - 190 + (s - 1) * 35
            py = 14
            pill_color = (0, 200, 100) if s == self.stage else (45, 50, 60)
            text_color = (255, 255, 255) if s == self.stage else (140, 140, 140)
            cv2.rectangle(canvas, (px, py), (px + 28, py + 24), pill_color, -1)
            cv2.putText(canvas, str(s), (px + 9, py + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # -----------------------------------------------------------------
    # UI Component: Bottom Navigation Bar
    # -----------------------------------------------------------------
    def _render_bottom_nav(self, canvas):
        cv2.rectangle(canvas, (0, HEIGHT - 45), (WIDTH, HEIGHT), (14, 16, 20), -1)
        cv2.line(canvas, (0, HEIGHT - 45), (WIDTH, HEIGHT - 45), (50, 60, 75), 1)

        nav_txt = "Press [1-5]: Switch Stages  |  [W] Walk  [S] Stand  [T] Sit  [B] Bend  [L] Lie  |  [F] Fall  [R] Reset  [Q] Exit"
        cv2.putText(canvas, nav_txt, (20, HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 200, 220), 1)

    # -----------------------------------------------------------------
    # STAGE 1: Single Sensor Analysis (Only IMU Sensor)
    # Detects Human Presence & Motion (Active vs Stationary)
    # -----------------------------------------------------------------
    def _render_stage_1(self, canvas, raw_sample, sensor_feats):
        # 1. Telemetry Card
        cx, cy, cw, ch = 30, 80, 360, 480
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (30, 34, 44), -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (65, 75, 95), 1)

        cv2.putText(canvas, "SENSOR 1: PRIMARY BODY IMU", (cx + 15, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2)
        cv2.putText(canvas, "Single Tri-Axial Accelerometer (a_x, a_y, a_z)", (cx + 15, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 185), 1)

        ax, ay, az = raw_sample[0], raw_sample[1], raw_sample[2]
        svm_raw = float(np.linalg.norm([ax, ay, az]))
        svm_filt = sensor_feats['svm_acc']
        dyn_acc = abs(svm_filt - 9.81)

        if len(self.history_filt_svm) > 2 and np.std(list(self.history_raw_svm)) > 1e-4:
            noise_att = max(0.0, (1.0 - np.std(list(self.history_filt_svm)) / np.std(list(self.history_raw_svm)))) * 100.0
        else:
            noise_att = 0.0

        stats = [
            ("Raw Accel X (ax):", f"{ax:+.2f} m/s2"),
            ("Raw Accel Y (ay):", f"{ay:+.2f} m/s2"),
            ("Raw Accel Z (az):", f"{az:+.2f} m/s2"),
            ("Resultant SVM (Raw):", f"{svm_raw:.2f} m/s2 ({svm_raw/9.81:.2f}g)"),
            ("Kalman Filtered SVM:", f"{svm_filt:.2f} m/s2 ({svm_filt/9.81:.2f}g)"),
            ("Dynamic Acceleration:", f"{dyn_acc:.2f} m/s2"),
            ("Noise Attenuation:", f"{noise_att:.1f} %")
        ]

        ty = cy + 105
        for lbl, val in stats:
            cv2.putText(canvas, lbl, (cx + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 205), 1)
            cv2.putText(canvas, val, (cx + 205, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            ty += 30

        # Human Presence Decision Box
        is_active = (dyn_acc > 1.0 or svm_filt > 11.2 or svm_filt < 8.2)
        status_box_y = cy + ch - 110
        cv2.rectangle(canvas, (cx + 15, status_box_y), (cx + cw - 15, status_box_y + 80), (20, 24, 30), -1)
        status_color = (0, 230, 100) if is_active else (255, 180, 0)
        cv2.rectangle(canvas, (cx + 15, status_box_y), (cx + cw - 15, status_box_y + 80), status_color, 2)

        cv2.putText(canvas, "STAGE 1 OUTPUT: HUMAN PRESENCE", (cx + 25, status_box_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        state_txt = "ACTIVE (WALKING / MOTION)" if is_active else "STATIONARY (RESTING / IDLE)"
        cv2.putText(canvas, state_txt, (cx + 25, status_box_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)

        # 2. Real-Time Oscillogram Plot (Right Pane)
        px, py, pw, ph = 420, 80, 550, 480
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 20, 26), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (65, 75, 95), 1)

        cv2.putText(canvas, "REAL-TIME ACCELERATION OSCILLOGRAM (100 Hz)", (px + 20, py + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(canvas, "Red: Raw Sensor Noise  |  Cyan: Online 1D Kalman Denoised (Zero-Lag)", (px + 20, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)

        # Grid lines
        graph_y = py + 90
        graph_h = ph - 120
        graph_w = pw - 40
        mid_y = graph_y + graph_h // 2

        cv2.line(canvas, (px + 20, mid_y), (px + 20 + graph_w, mid_y), (50, 60, 75), 1)
        cv2.putText(canvas, "1.0g (9.81 m/s2)", (px + 25, mid_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 110, 130), 1)

        # Plot waveforms
        raw_list = list(self.history_raw_svm)
        filt_list = list(self.history_filt_svm)
        n_pts = len(raw_list)

        if n_pts > 1:
            dx = graph_w / max(1, self.history_raw_svm.maxlen)
            for i in range(1, n_pts):
                x1 = int(px + 20 + (i - 1) * dx)
                x2 = int(px + 20 + i * dx)

                # Scale: 9.81 is mid_y, +/- 15 m/s2 range
                scale = (graph_h / 2.0) / 18.0
                y_raw1 = int(mid_y - (raw_list[i-1] - 9.81) * scale)
                y_raw2 = int(mid_y - (raw_list[i] - 9.81) * scale)
                y_filt1 = int(mid_y - (filt_list[i-1] - 9.81) * scale)
                y_filt2 = int(mid_y - (filt_list[i] - 9.81) * scale)

                y_raw1 = np.clip(y_raw1, graph_y, graph_y + graph_h)
                y_raw2 = np.clip(y_raw2, graph_y, graph_y + graph_h)
                y_filt1 = np.clip(y_filt1, graph_y, graph_y + graph_h)
                y_filt2 = np.clip(y_filt2, graph_y, graph_y + graph_h)

                cv2.line(canvas, (x1, y_raw1), (x2, y_raw2), (0, 0, 255), 1)
                cv2.line(canvas, (x1, y_filt1), (x2, y_filt2), (255, 255, 0), 2)

    # -----------------------------------------------------------------
    # STAGE 2: Two Sensor Integration (IMU + MPU-6050)
    # Detects Body Position & 3D Tilt Orientation
    # -----------------------------------------------------------------
    def _render_stage_2(self, canvas, raw_sample, sensor_feats):
        cx, cy, cw, ch = 30, 80, 440, 480
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (30, 34, 44), -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (65, 75, 95), 1)

        cv2.putText(canvas, "DUAL SENSOR: IMU + MPU-6050 FUSION", (cx + 15, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2)
        cv2.putText(canvas, "Sensor 1 (Chest IMU) + Sensor 2 (Waist MPU-6050 6-DOF)", (cx + 15, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 185), 1)

        pitch = sensor_feats['pitch']
        roll = sensor_feats['roll']
        tilt = sensor_feats['tilt_angle']
        gyro_mag = sensor_feats['svm_gyro']

        # Determine spatial alignment
        if tilt < 25.0:
            pos_label = "UPRIGHT VERTICAL (STANDING / SITTING)"
            pos_color = (0, 230, 100)
        elif tilt < 60.0:
            pos_label = "INCLINED / LEANING (TRANSITION)"
            pos_color = (0, 200, 255)
        else:
            pos_label = "HORIZONTAL RECUMBENT (PRONE / LYING)"
            pos_color = (0, 0, 255)

        stats = [
            ("Chest IMU Accel:", f"{sensor_feats['svm_acc']:.2f} m/s2"),
            ("Waist MPU Gyro X:", f"{raw_sample[3]:+.1f} deg/s"),
            ("Waist MPU Gyro Y:", f"{raw_sample[4]:+.1f} deg/s"),
            ("Waist MPU Gyro Z:", f"{raw_sample[5]:+.1f} deg/s"),
            ("Resultant Angular Rate:", f"{gyro_mag:.1f} deg/s"),
            ("Calculated Pitch Angle:", f"{pitch:+.1f} deg"),
            ("Calculated Roll Angle:", f"{roll:+.1f} deg"),
            ("3D Body Tilt Angle:", f"{tilt:.1f} deg from Vertical")
        ]

        ty = cy + 100
        for lbl, val in stats:
            cv2.putText(canvas, lbl, (cx + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 190, 205), 1)
            cv2.putText(canvas, val, (cx + 250, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            ty += 28

        # Position Result Box
        res_y = cy + ch - 95
        cv2.rectangle(canvas, (cx + 15, res_y), (cx + cw - 15, res_y + 75), (20, 24, 30), -1)
        cv2.rectangle(canvas, (cx + 15, res_y), (cx + cw - 15, res_y + 75), pos_color, 2)
        cv2.putText(canvas, "STAGE 2 OUTPUT: HUMAN POSITION", (cx + 25, res_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, pos_label, (cx + 25, res_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, pos_color, 2)

        # 2. Right Pane: Visual 3D Attitude Horizon / Tilt Meter
        px, py, pw, ph = 500, 80, 470, 480
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 20, 26), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (65, 75, 95), 1)

        cv2.putText(canvas, "BODY SPATIAL ORIENTATION GAUGE", (px + 20, py + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(canvas, "Artificial Horizon & Body Incline Needle", (px + 20, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)

        # Dial Center
        dial_x = px + pw // 2
        dial_y = py + 230
        dial_r = 130

        # Dial Circles
        cv2.circle(canvas, (dial_x, dial_y), dial_r, (40, 50, 65), 2)
        cv2.circle(canvas, (dial_x, dial_y), 4, (0, 230, 255), -1)

        # Angle markings
        for deg in [0, 30, 45, 60, 90]:
            rad = math.radians(deg - 90)
            mx = int(dial_x + (dial_r - 10) * math.cos(rad))
            my = int(dial_y + (dial_r - 10) * math.sin(rad))
            cv2.circle(canvas, (mx, my), 3, (80, 95, 115), -1)
            cv2.putText(canvas, f"{deg}d", (mx - 10, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 150, 170), 1)

        # Needle for tilt
        needle_rad = math.radians(tilt - 90)
        nx = int(dial_x + (dial_r - 15) * math.cos(needle_rad))
        ny = int(dial_y + (dial_r - 15) * math.sin(needle_rad))
        cv2.line(canvas, (dial_x, dial_y), (nx, ny), pos_color, 4)

        # Tilt Bar Gauge below dial
        bar_y = py + ph - 65
        bar_w = pw - 60
        cv2.rectangle(canvas, (px + 30, bar_y), (px + 30 + bar_w, bar_y + 16), (40, 45, 55), 1)
        fill_w = int((tilt / 90.0) * bar_w)
        cv2.rectangle(canvas, (px + 30, bar_y), (px + 30 + fill_w, bar_y + 16), pos_color, -1)
        cv2.putText(canvas, f"Tilt: {tilt:.1f} / 90.0 deg", (px + 30, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)

    # -----------------------------------------------------------------
    # STAGE 3: Tri-Sensor Integration (IMU + MPU-6050 + IR Sensor)
    # Detects Position & Posture Classification
    # -----------------------------------------------------------------
    def _render_stage_3(self, canvas, sensor_feats):
        cx, cy, cw, ch = 30, 80, 450, 480
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (30, 34, 44), -1)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (65, 75, 95), 1)

        cv2.putText(canvas, "TRI-SENSOR INTEGRATION (IMU + MPU + IR)", (cx + 15, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2)
        cv2.putText(canvas, "Fusing Acceleration, Tilt Angle, and IR Floor Distance", (cx + 15, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 185), 1)

        tilt = sensor_feats['tilt_angle']
        ir_dist = sensor_feats['ir_distance']
        posture = sensor_feats['posture_sensor']
        svm_acc = sensor_feats['svm_acc']

        stats = [
            ("Sensor 1 (IMU Accel):", f"{svm_acc:.2f} m/s2"),
            ("Sensor 1 (IMU Presence):", sensor_feats['imu_activity']),
            ("Sensor 2 (MPU-6050 Tilt):", f"{tilt:.1f} deg"),
            ("Sensor 2 (MPU Pitch/Roll):", f"P:{sensor_feats['pitch']:+.1f} R:{sensor_feats['roll']:+.1f}"),
            ("Sensor 3 (IR Floor Distance):", f"{ir_dist:.2f} meters"),
            ("Clearance Threshold (Floor):", "< 0.35 m (Prone Alert)"),
            ("Clearance Threshold (Sit):", "0.60 m - 1.05 m")
        ]

        ty = cy + 105
        for lbl, val in stats:
            cv2.putText(canvas, lbl, (cx + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 190, 205), 1)
            cv2.putText(canvas, val, (cx + 250, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            ty += 30

        # Posture Result Box
        p_colors = {
            "STANDING": (0, 230, 100),
            "SITTING": (0, 200, 255),
            "BENDING": (0, 140, 255),
            "LYING": (0, 0, 255)
        }
        p_color = p_colors.get(posture, (200, 200, 200))

        res_y = cy + ch - 110
        cv2.rectangle(canvas, (cx + 15, res_y), (cx + cw - 15, res_y + 85), (20, 24, 30), -1)
        cv2.rectangle(canvas, (cx + 15, res_y), (cx + cw - 15, res_y + 85), p_color, 2)
        cv2.putText(canvas, "STAGE 3 OUTPUT: POSITION & POSTURE", (cx + 25, res_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, f"CLASSIFIED POSTURE: {posture}", (cx + 25, res_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, p_color, 2)

        # 2. Right Pane: IR Distance Proximity Pillar & Posture Matrix
        px, py, pw, ph = 510, 80, 460, 480
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 20, 26), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (65, 75, 95), 1)

        cv2.putText(canvas, "IR SENSOR CLEARANCE TO FLOOR", (px + 20, py + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(canvas, "Real-Time Ultrasonic / IR Height Measurement", (px + 20, py + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)

        # Vertical Pillar Gauge
        pillar_x = px + 60
        pillar_y = py + 100
        pillar_w = 40
        pillar_h = 280

        cv2.rectangle(canvas, (pillar_x, pillar_y), (pillar_x + pillar_w, pillar_y + pillar_h), (40, 45, 55), 1)

        # Fill based on height (0.0m to 2.0m max)
        fill_h = int(np.clip(ir_dist / 2.0, 0.0, 1.0) * pillar_h)
        bar_top = pillar_y + pillar_h - fill_h
        cv2.rectangle(canvas, (pillar_x, bar_top), (pillar_x + pillar_w, pillar_y + pillar_h), p_color, -1)

        # Tick marks
        for m_mark, label in [(1.8, "1.8m - Standing"), (0.9, "0.9m - Sitting"), (0.3, "0.3m - Floor Alert")]:
            mark_y = int(pillar_y + pillar_h - (m_mark / 2.0) * pillar_h)
            cv2.line(canvas, (pillar_x + pillar_w, mark_y), (pillar_x + pillar_w + 12, mark_y), (100, 120, 140), 1)
            cv2.putText(canvas, label, (pillar_x + pillar_w + 18, mark_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 205), 1)

        # Posture Matrix Info Box
        info_y = py + ph - 80
        cv2.rectangle(canvas, (px + 20, info_y), (px + pw - 20, info_y + 60), (25, 30, 38), -1)
        cv2.putText(canvas, "Decision Logic: Posture = f(Tilt MPU, Height IR, Accel IMU)", (px + 30, info_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)
        cv2.putText(canvas, "Eliminates false alarms from simply dropping sensor or sitting.", (px + 30, info_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1)

    # -----------------------------------------------------------------
    # STAGE 4: Camera Stick Figure Analysis (Vision-Only Fall Detection)
    # -----------------------------------------------------------------
    def _render_stage_4(self, canvas, timestamp_sec):
        # Generate or capture video frame
        if self.is_synthetic:
            frame = self._draw_stick_man_frame()
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    frame = self._draw_stick_man_frame()

        # Run vision pose tracker
        vision_feats, annotated = self.vision_tracker.process_frame(frame, timestamp_sec=timestamp_sec)

        if self.is_synthetic and self.synthetic_fall:
            self.fall_progress += 1
            vision_feats['is_laying'] = (self.fall_progress > 15)
            vision_feats['aspect_ratio'] = 1.8 if self.fall_progress > 15 else 0.8
            vision_feats['torso_angle'] = min(90.0, self.fall_progress * 4.5)
            vision_feats['centroid_vy'] = 1.6 if self.fall_progress < 15 else 0.0
            vision_feats['posture'] = "LYING" if self.fall_progress > 15 else "TRANSITION"

        # Embed annotated video frame onto left canvas
        disp_w, disp_h = 560, 480
        resized_vid = cv2.resize(annotated, (disp_w, disp_h))
        canvas[80:80+disp_h, 30:30+disp_w] = resized_vid
        cv2.rectangle(canvas, (30, 80), (30 + disp_w, 80 + disp_h), (65, 75, 95), 1)

        # Right Telemetry Pane
        px, py, pw, ph = 620, 80, 350, 480
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (30, 34, 44), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (65, 75, 95), 1)

        cv2.putText(canvas, "CAMERA STICK SKELETON ANALYSIS", (px + 15, py + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)
        cv2.putText(canvas, "33-Point MediaPipe Pose Tracking (2D Kalman)", (px + 15, py + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 185), 1)

        ar = vision_feats.get('aspect_ratio', 0.45)
        torso = vision_feats.get('torso_angle', 10.0)
        vy = vision_feats.get('centroid_vy', 0.0)
        posture = vision_feats.get('posture', 'STANDING')

        stats = [
            ("Posture Classification:", posture),
            ("Bounding Box AR (W/H):", f"{ar:.2f}"),
            ("AR Threshold (Floor):", "> 1.15 (Inversion)"),
            ("Torso Inclination Angle:", f"{torso:.0f} deg"),
            ("Centroid Descent Vy:", f"{vy:.2f} m/s"),
            ("Descent Spike Trigger:", "> 1.20 m/s")
        ]

        ty = py + 100
        for lbl, val in stats:
            cv2.putText(canvas, lbl, (px + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 190, 205), 1)
            cv2.putText(canvas, val, (px + 205, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
            ty += 30

        # Fall status from vision
        is_vision_fall = (ar > 1.15 and torso > 50.0)
        status_color = (0, 0, 255) if is_vision_fall else (0, 230, 100)

        res_y = py + ph - 110
        cv2.rectangle(canvas, (px + 15, res_y), (px + pw - 15, res_y + 85), (20, 24, 30), -1)
        cv2.rectangle(canvas, (px + 15, res_y), (px + pw - 15, res_y + 85), status_color, 2)
        cv2.putText(canvas, "STAGE 4 OUTPUT: VISION DETECTION", (px + 25, res_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        vis_txt = "FALL DETECTED (POSTURE COLLAPSED)" if is_vision_fall else "NORMAL / UPRIGHT POSTURE"
        cv2.putText(canvas, vis_txt, (px + 25, res_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 2)

    # -----------------------------------------------------------------
    # STAGE 5: Full Multimodal Late-Fusion Integration
    # -----------------------------------------------------------------
    def _render_stage_5(self, canvas, timestamp_sec, raw_sample, sensor_feats):
        # 1. Video frame
        if self.is_synthetic:
            frame = self._draw_stick_man_frame()
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    frame = self._draw_stick_man_frame()

        vision_feats, annotated = self.vision_tracker.process_frame(frame, timestamp_sec=timestamp_sec)

        if self.is_synthetic and self.synthetic_fall:
            self.fall_progress += 1
            vision_feats['is_laying'] = (self.fall_progress > 15)
            vision_feats['aspect_ratio'] = 1.8 if self.fall_progress > 15 else 0.8
            vision_feats['torso_angle'] = min(90.0, self.fall_progress * 4.5)
            vision_feats['centroid_vy'] = 1.6 if self.fall_progress < 15 else 0.0
            vision_feats['posture'] = "LYING" if self.fall_progress > 15 else "TRANSITION"

        # Update Spatiotemporal Decision Engine
        decision = self.decision_engine.update(vision_feats, sensor_feats, frame=annotated)

        # Embed video
        disp_w, disp_h = 560, 480
        resized_vid = cv2.resize(annotated, (disp_w, disp_h))
        canvas[80:80+disp_h, 30:30+disp_w] = resized_vid
        cv2.rectangle(canvas, (30, 80), (30 + disp_w, 80 + disp_h), (65, 75, 95), 1)

        # Right Telemetry Glass Panel
        px, py, pw, ph = 620, 80, 350, 480
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (30, 34, 44), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (65, 75, 95), 1)

        cv2.putText(canvas, "STAGE 5: FULL MULTIMODAL FUSION", (px + 15, py + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 2)

        # Telemetry
        state = decision['state']
        conf = decision['confidence']

        items = [
            ("FSM Decision State:", state[:18]),
            ("Fall Risk Confidence:", f"{conf:.0f} %"),
            ("Vision Posture:", str(vision_feats.get('posture', 'N/A'))),
            ("Sensor Posture:", str(sensor_feats.get('posture_sensor', 'STANDING'))),
            ("Vision Aspect Ratio:", f"{vision_feats.get('aspect_ratio', 0.0):.2f}"),
            ("Sensor 1 (IMU Motion):", str(sensor_feats.get('imu_activity', 'ACTIVE')[:14])),
            ("Sensor 2 (MPU Tilt):", f"{sensor_feats.get('tilt_angle', 0.0):.0f} deg"),
            ("Sensor 3 (IR Clearance):", f"{sensor_feats.get('ir_distance', 1.65):.2f} m"),
            ("Accel SVM:", f"{sensor_feats.get('svm_acc', 9.81):.2f} m/s2 ({sensor_feats.get('svm_acc', 9.81)/9.81:.2f}g)"),
            ("Gyro Mag:", f"{sensor_feats.get('svm_gyro', 0.0):.1f} deg/s")
        ]

        ty = py + 68
        for lbl, val in items:
            cv2.putText(canvas, lbl, (px + 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1)
            cv2.putText(canvas, val, (px + 195, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            ty += 24

        # Risk Progress Meter
        bar_x = px + 15
        bar_y = py + ph - 85
        bar_w = pw - 30
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (50, 55, 65), 1)
        fill_w = int((conf / 100.0) * bar_w)
        m_color = (0, 0, 255) if conf > 70 else (0, 200, 255) if conf > 30 else (0, 230, 100)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + 14), m_color, -1)
        cv2.putText(canvas, f"Emergency Confidence: {conf:.0f}%", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Flashing Emergency Border
        if state == FallDecisionEngine.STATE_CONFIRMED:
            cv2.rectangle(canvas, (bar_x, bar_y + 24), (bar_x + bar_w, bar_y + 60), (0, 0, 255), -1)
            cv2.putText(canvas, "EMERGENCY: FALL CONFIRMED!", (bar_x + 10, bar_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
            if int(time.time() * 3) % 2 == 0:
                cv2.rectangle(canvas, (0, 0), (WIDTH, HEIGHT), (0, 0, 255), 8)

    # -----------------------------------------------------------------
    # Helper: Synthetic Animated Stick-Figure Frame
    # -----------------------------------------------------------------
    def _draw_stick_man_frame(self):
        frame = np.zeros((480, 560, 3), dtype=np.uint8)
        frame[:] = (35, 38, 48)  # Room background

        # Floor line
        cv2.line(frame, (40, 420), (520, 420), (80, 90, 105), 3)

        tilt = min(90.0, self.fall_progress * 5.0) if self.synthetic_fall else 0.0
        tilt_rad = math.radians(tilt)

        base_x = 280
        floor_y = 420
        torso_len = 110
        leg_len = 90
        head_r = 22

        if tilt > 45:
            hip_y = floor_y - 30
            hip_x = base_x + int(40 * (tilt / 90.0))
        else:
            hip_y = floor_y - leg_len
            hip_x = base_x

        neck_x = hip_x + int(torso_len * math.sin(tilt_rad))
        neck_y = hip_y - int(torso_len * math.cos(tilt_rad))

        head_x = neck_x + int(head_r * math.sin(tilt_rad))
        head_y = neck_y - int(head_r * math.cos(tilt_rad))

        # Color: Green normal, Red if fall
        color = (0, 0, 255) if tilt > 45 else (0, 230, 100)

        # Head & Torso
        cv2.circle(frame, (head_x, head_y), head_r, color, -1)
        cv2.line(frame, (neck_x, neck_y), (hip_x, hip_y), color, 6)

        # Arms
        cv2.line(frame, (neck_x, neck_y), (neck_x - 40, neck_y + 40), color, 4)
        cv2.line(frame, (neck_x, neck_y), (neck_x + 40, neck_y + 40), color, 4)

        # Legs
        if tilt > 45:
            cv2.line(frame, (hip_x, hip_y), (hip_x - 60, floor_y), color, 5)
            cv2.line(frame, (hip_x, hip_y), (hip_x - 40, floor_y), color, 5)
        else:
            swing = int(18 * math.sin(self.anim_t * 6.0))
            cv2.line(frame, (hip_x, hip_y), (hip_x - 20 + swing, floor_y), color, 5)
            cv2.line(frame, (hip_x, hip_y), (hip_x + 20 - swing, floor_y), color, 5)

        return frame


def parse_args():
    parser = argparse.ArgumentParser(description="5-Stage Multimodal Fall Detection Presentation System")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="Initial presentation stage to display (1 to 5)")
    parser.add_argument("--source", type=str, default="synthetic",
                        help="Camera source: 'synthetic', '0' for webcam, or video path")
    parser.add_argument("--port", type=str, default=None,
                        help="Optional serial port for Arduino/ESP32 (e.g., COM3)")
    parser.add_argument("--dataset", type=str, default=None, choices=["fallalld", "synthetic"],
                        help="Optional dataset stream source: 'fallalld' to stream genuine FallAllD clinical trials")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = PresentationDemoApp(
        initial_stage=args.stage,
        source=args.source,
        port=args.port,
        dataset=args.dataset
    )
    app.run()
