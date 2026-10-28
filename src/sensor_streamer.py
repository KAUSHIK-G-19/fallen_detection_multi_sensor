"""
Real-Time Sensor Ring Buffer and Streamer Module
Manages a continuous circular buffer (100 Hz, 2.0s = 200 samples) of wearable IMU and HR signals,
applies sample-by-sample 1D Kalman filtering, calculates dynamic acceleration, and detects impact spikes.
"""
import math
import time
import numpy as np
import pandas as pd
from src.kalman_filter import OnlineKalmanFilter1D


class RealtimeSensorRingBuffer:
    """
    Circular sliding buffer maintaining the latest N sensor samples (default 200 for 2.0s @ 100Hz).
    Applies real-time online Kalman filtering to raw signals.
    """
    def __init__(self, capacity=200, sr=100):
        self.capacity = capacity
        self.sr = sr
        self.channels = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate']
        self.n_channels = len(self.channels)

        # Pre-allocate buffer arrays
        self.raw_buffer = np.zeros((self.capacity, self.n_channels), dtype=np.float32)
        self.filtered_buffer = np.zeros((self.capacity, self.n_channels), dtype=np.float32)
        self.timestamps = np.zeros(self.capacity, dtype=np.float64)

        self.index = 0
        self.count = 0

        # Online Kalman filters for each channel
        self.kalman_filters = {
            col: OnlineKalmanFilter1D(q=1e-4, r=0.04) for col in self.channels
        }
        # Dedicated filter for IR sensor distance
        self.ir_kf = OnlineKalmanFilter1D(q=1e-3, r=0.02)

        # Rolling statistics
        self.latest_svm_acc = 9.81
        self.latest_svm_gyro = 0.0
        self.latest_hr = 75.0
        self.latest_ir_dist = 1.65
        self.latest_pitch = 0.0
        self.latest_roll = 0.0
        self.latest_tilt = 0.0
        self.max_impact_in_window = 9.81
        self.has_impact_spike = False

        # Physical constants
        self.G = 9.81
        self.IMPACT_THRESHOLD_MS2 = 2.0 * self.G  # ~19.6 m/s^2 (2.0g impact)
        self.FREEFALL_THRESHOLD_MS2 = 0.45 * self.G  # Freefall weightlessness trigger

    def add_sample(self, raw_sample, timestamp=None, ir_distance=None):
        """
        Add a single incoming sensor packet.
        raw_sample: list or 1D array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate]
        Optional ir_distance: float in meters
        """
        if timestamp is None:
            timestamp = time.time()

        sample_arr = np.array(raw_sample[:self.n_channels], dtype=np.float32)
        filtered_sample = np.zeros(self.n_channels, dtype=np.float32)

        # Apply online Kalman filter channel by channel
        for i, col in enumerate(self.channels):
            filtered_sample[i] = self.kalman_filters[col].update(sample_arr[i])

        # Insert into circular buffer
        self.raw_buffer[self.index] = sample_arr
        self.filtered_buffer[self.index] = filtered_sample
        self.timestamps[self.index] = timestamp

        self.index = (self.index + 1) % self.capacity
        self.count += 1

        # Compute signal vector magnitudes from raw and filtered signals
        raw_acc_mag = float(np.linalg.norm(sample_arr[:3]))
        acc = filtered_sample[:3]
        gyro = filtered_sample[3:6]
        hr = filtered_sample[6]

        self.latest_svm_acc = float(np.linalg.norm(acc))
        self.latest_svm_gyro = float(np.linalg.norm(gyro))
        self.latest_hr = float(hr)

        # -------------------------------------------------------------
        # Orientation Estimation (MPU-6050 Pitch, Roll, 3D Tilt)
        # -------------------------------------------------------------
        ax, ay, az = float(acc[0]), float(acc[1]), float(acc[2])
        norm_xy = math.sqrt(ax * ax + ay * ay)
        self.latest_pitch = math.degrees(math.atan2(ax, math.sqrt(ay * ay + az * az) + 1e-6))
        self.latest_roll = math.degrees(math.atan2(ay, math.sqrt(ax * ax + az * az) + 1e-6))
        
        # Tilt angle from vertical (assuming Z is vertical 1g in upright stance)
        ratio = max(0.0, min(1.0, abs(az) / (self.latest_svm_acc + 1e-6)))
        self.latest_tilt = math.degrees(math.acos(ratio))

        # -------------------------------------------------------------
        # IR Proximity / Floor Distance Estimation
        # -------------------------------------------------------------
        if ir_distance is not None:
            self.latest_ir_dist = self.ir_kf.update(ir_distance)
        elif len(raw_sample) > 7:
            self.latest_ir_dist = self.ir_kf.update(float(raw_sample[7]))
        else:
            # If not explicitly provided, estimate inferred IR floor distance from tilt
            # Standing (~1.65m) -> Prone (~0.15m)
            tilt_fraction = min(1.0, self.latest_tilt / 85.0)
            inferred_ir = 1.65 * (1.0 - 0.90 * tilt_fraction)
            self.latest_ir_dist = self.ir_kf.update(inferred_ir)

        # Evaluate window impact across both filtered and raw signals
        window = self.get_ordered_window()
        window_acc_mag = np.linalg.norm(window[:, :3], axis=1)
        raw_window = self.get_ordered_raw_window()
        raw_window_acc_mag = np.linalg.norm(raw_window[:, :3], axis=1)

        self.max_impact_in_window = float(max(np.max(window_acc_mag), np.max(raw_window_acc_mag)))
        self.has_impact_spike = bool(self.max_impact_in_window >= self.IMPACT_THRESHOLD_MS2 or raw_acc_mag >= self.IMPACT_THRESHOLD_MS2)

        # Sensor Posture Classification (Tri-Sensor Fusion: IMU + MPU + IR)
        if self.latest_tilt > 65.0 or self.latest_ir_dist < 0.35:
            posture_sensor = "LYING"
        elif self.latest_tilt > 40.0:
            posture_sensor = "BENDING"
        elif self.latest_ir_dist < 1.05 and self.latest_tilt < 35.0:
            posture_sensor = "SITTING"
        else:
            posture_sensor = "STANDING"

        # Sensor 1 Presence/Activity Status
        dyn_acc = abs(self.latest_svm_acc - self.G)
        if dyn_acc > 1.2 or self.latest_svm_gyro > 20.0:
            imu_activity = "ACTIVE (WALKING / MOTION)"
        else:
            imu_activity = "STATIONARY (RESTING)"

        return {
            'svm_acc': self.latest_svm_acc,
            'svm_gyro': self.latest_svm_gyro,
            'heart_rate': self.latest_hr,
            'max_impact': self.max_impact_in_window,
            'has_impact': self.has_impact_spike,
            'is_freefall': bool(self.latest_svm_acc <= self.FREEFALL_THRESHOLD_MS2),
            'pitch': self.latest_pitch,
            'roll': self.latest_roll,
            'tilt_angle': self.latest_tilt,
            'ir_distance': self.latest_ir_dist,
            'posture_sensor': posture_sensor,
            'imu_activity': imu_activity
        }

    def get_ordered_window(self):
        """Returns the current filtered window ordered chronologically (oldest to newest)"""
        if self.count < self.capacity:
            return self.filtered_buffer[:self.count]
        else:
            return np.vstack([self.filtered_buffer[self.index:], self.filtered_buffer[:self.index]])

    def get_ordered_raw_window(self):
        """Returns the current raw window ordered chronologically (oldest to newest)"""
        if self.count < self.capacity:
            return self.raw_buffer[:self.count]
        else:
            return np.vstack([self.raw_buffer[self.index:], self.raw_buffer[:self.index]])

    def is_ready(self):
        """Returns True once the buffer is full (2.0s of data)"""
        return self.count >= self.capacity


class RealisticSensorStreamer:
    """
    Simulates real-time 100 Hz sensor data stream across 3 distinct sensor tiers:
    1. Sensor 1: Primary IMU (human dynamic motion & presence)
    2. Sensor 2: MPU-6050 6-axis IMU (triaxial accelerometer + gyroscope for pitch/roll/tilt orientation)
    3. Sensor 3: IR Proximity / Distance Sensor (clearance to floor: standing ~1.65m, sitting ~0.85m, floor ~0.15m)
    """
    def __init__(self, dataset_csv_path=None, sr=100):
        self.sr = sr
        self.dt = 1.0 / sr
        self.time_step = 0
        self.data_df = None
        self.row_index = 0
        self.posture_mode = "walking"  # "walking", "standing", "sitting", "bending", "lying"

        if dataset_csv_path and pd.io.common.file_exists(dataset_csv_path):
            try:
                self.data_df = pd.read_csv(dataset_csv_path)
                print(f"RealisticSensorStreamer: Loaded {len(self.data_df)} records from dataset.")
            except Exception:
                self.data_df = None

        # Fall injection controls
        self.inject_fall = False
        self.fall_progress = 0
        self.latest_ir_distance = 1.65

    def set_posture_mode(self, mode):
        """Set human posture state: 'walking', 'standing', 'sitting', 'bending', 'lying'"""
        self.posture_mode = str(mode).lower()
        if self.posture_mode != "lying":
            self.inject_fall = False

    def trigger_fall(self):
        """Manually trigger a simulated fall in the sensor stream"""
        self.inject_fall = True
        self.fall_progress = 0

    def get_next_sample(self):
        """
        Generate or stream the next 10ms sensor sample (100 Hz).
        Returns: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate]
        """
        t = self.time_step * self.dt
        self.time_step += 1

        # If a fall is triggered: simulate fall kinematics
        if self.inject_fall:
            self.fall_progress += 1
            # Phase 1: Freefall (0 - 0.2s = 20 samples)
            if self.fall_progress < 20:
                acc_x = float(np.random.normal(0.0, 0.5))
                acc_y = float(np.random.normal(0.0, 0.5))
                acc_z = float(np.random.normal(1.5, 0.5))  # Drop towards 0g
                gyro_x = float(np.random.normal(30.0, 10.0))
                gyro_y = float(np.random.normal(40.0, 10.0))
                gyro_z = float(np.random.normal(20.0, 5.0))
                hr = 85.0
                self.latest_ir_distance = float(max(0.6, 1.65 - (self.fall_progress / 20.0) * 1.0))
            # Phase 2: Impact shock (0.2s - 0.35s = 15 samples)
            elif self.fall_progress < 35:
                acc_x = float(np.random.normal(12.0, 3.0))
                acc_y = float(np.random.normal(8.0, 2.0))
                acc_z = float(np.random.normal(24.0, 4.0))  # ~2.8g Impact!
                gyro_x = float(np.random.normal(150.0, 30.0))
                gyro_y = float(np.random.normal(180.0, 40.0))
                gyro_z = float(np.random.normal(90.0, 20.0))
                hr = 105.0
                self.latest_ir_distance = 0.22
            # Phase 3: Lying stationary on floor (post-impact)
            else:
                acc_x = float(np.random.normal(9.7, 0.1))  # Gravity along X (lying sideways)
                acc_y = float(np.random.normal(0.3, 0.1))
                acc_z = float(np.random.normal(1.0, 0.1))
                gyro_x = float(np.random.normal(0.0, 1.0))
                gyro_y = float(np.random.normal(0.0, 1.0))
                gyro_z = float(np.random.normal(0.0, 1.0))
                hr = 95.0
                self.latest_ir_distance = float(max(0.05, 0.15 + np.random.normal(0.0, 0.02)))
                if self.fall_progress > 250:  # Reset after 2.5 seconds if needed
                    pass

            return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, hr]

        # -------------------------------------------------------------
        # Non-fall kinematic simulation based on posture mode
        # -------------------------------------------------------------
        step_freq = 1.6
        if self.posture_mode == "walking":
            vertical_bounce = 1.2 * np.sin(2 * np.pi * step_freq * t)
            lateral_sway = 0.5 * np.cos(2 * np.pi * (step_freq / 2) * t)

            acc_x = float(lateral_sway + np.random.normal(0.0, 0.15))
            acc_y = float(np.random.normal(0.2, 0.15))
            acc_z = float(9.81 + vertical_bounce + np.random.normal(0.0, 0.2))

            gyro_x = float(10.0 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0.0, 2.0))
            gyro_y = float(15.0 * np.cos(2 * np.pi * step_freq * t) + np.random.normal(0.0, 2.0))
            gyro_z = float(np.random.normal(0.0, 1.5))
            hr = float(74 + int(3.0 * np.sin(2 * np.pi * 0.1 * t)))
            self.latest_ir_distance = float(1.65 + 0.04 * np.sin(2 * np.pi * step_freq * t))

        elif self.posture_mode == "standing":
            acc_x = float(np.random.normal(0.0, 0.05))
            acc_y = float(np.random.normal(0.1, 0.05))
            acc_z = float(9.81 + np.random.normal(0.0, 0.06))
            gyro_x = float(np.random.normal(0.0, 0.8))
            gyro_y = float(np.random.normal(0.0, 0.8))
            gyro_z = float(np.random.normal(0.0, 0.5))
            hr = 70.0
            self.latest_ir_distance = float(1.70 + np.random.normal(0.0, 0.01))

        elif self.posture_mode == "sitting":
            # Slight forward tilt (acc_y ~ 2.0, acc_z ~ 9.5)
            acc_x = float(np.random.normal(0.0, 0.05))
            acc_y = float(2.2 + np.random.normal(0.0, 0.08))
            acc_z = float(9.5 + np.random.normal(0.0, 0.08))
            gyro_x = float(np.random.normal(0.0, 0.6))
            gyro_y = float(np.random.normal(0.0, 0.6))
            gyro_z = float(np.random.normal(0.0, 0.5))
            hr = 68.0
            self.latest_ir_distance = float(0.85 + np.random.normal(0.0, 0.01))

        elif self.posture_mode == "bending":
            # Pronounced forward tilt (acc_y ~ 7.0, acc_z ~ 6.5)
            acc_x = float(np.random.normal(0.0, 0.1))
            acc_y = float(6.9 + np.random.normal(0.0, 0.15))
            acc_z = float(6.8 + np.random.normal(0.0, 0.15))
            gyro_x = float(np.random.normal(0.0, 1.2))
            gyro_y = float(np.random.normal(0.0, 1.2))
            gyro_z = float(np.random.normal(0.0, 0.8))
            hr = 78.0
            self.latest_ir_distance = float(1.05 + np.random.normal(0.0, 0.02))

        elif self.posture_mode == "lying":
            # Horizontal (acc_x ~ 9.8, acc_z ~ 0.5)
            acc_x = float(9.6 + np.random.normal(0.0, 0.08))
            acc_y = float(np.random.normal(0.2, 0.08))
            acc_z = float(np.random.normal(0.8, 0.08))
            gyro_x = float(np.random.normal(0.0, 0.5))
            gyro_y = float(np.random.normal(0.0, 0.5))
            gyro_z = float(np.random.normal(0.0, 0.5))
            hr = 65.0
            self.latest_ir_distance = float(0.15 + np.random.normal(0.0, 0.01))

        else:
            acc_x, acc_y, acc_z = 0.0, 0.0, 9.81
            gyro_x, gyro_y, gyro_z = 0.0, 0.0, 0.0
            hr = 72.0
            self.latest_ir_distance = 1.65

        return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, hr]

    def get_multi_sensor_packet(self):
        """
        Returns full multi-sensor packet including Sensor 1, Sensor 2, and Sensor 3 channels.
        """
        sample = self.get_next_sample()
        return {
            'sample': sample,
            'ir_distance': self.latest_ir_distance,
            'imu1_acc': sample[:3],
            'mpu_acc': sample[:3],
            'mpu_gyro': sample[3:6],
            'heart_rate': sample[6]
        }


class SerialSensorStreamer:
    """
    Reads hardware sensor stream from physical serial port (e.g. Arduino / ESP32 via USB).
    Expected line format: 'acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,heart_rate,ir_distance'
    Falls back gracefully to RealisticSensorStreamer if serial port is unavailable.
    """
    def __init__(self, port="COM3", baud=115200, fallback_streamer=None):
        self.port = port
        self.baud = baud
        self.serial_conn = None
        self.is_connected = False
        self.fallback = fallback_streamer or RealisticSensorStreamer()
        self.latest_ir_distance = 1.65

        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baud, timeout=0.05)
            self.is_connected = True
            print(f"[SERIAL] Successfully opened hardware sensor on port {self.port} @ {self.baud} baud.")
        except Exception as e:
            print(f"[SERIAL WARNING] Hardware port '{self.port}' not available: {e}. Using synthetic fallback.")
            self.is_connected = False

    def get_next_sample(self):
        if not self.is_connected or self.serial_conn is None:
            sample = self.fallback.get_next_sample()
            self.latest_ir_distance = self.fallback.latest_ir_distance
            return sample

        try:
            line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            if line:
                parts = [float(p.strip()) for p in line.split(',') if p.strip()]
                if len(parts) >= 7:
                    if len(parts) >= 8:
                        self.latest_ir_distance = parts[7]
                    return parts[:7]
        except Exception:
            pass

        sample = self.fallback.get_next_sample()
        self.latest_ir_distance = self.fallback.latest_ir_distance
        return sample

    def trigger_fall(self):
        self.fallback.trigger_fall()

    def set_posture_mode(self, mode):
        self.fallback.set_posture_mode(mode)


class FallAllDStreamer:
    """
    Streams genuine clinical human fall & ADL sensor records from the FallAllD dataset.
    Default path: C:\\Users\\KAUSHIK G\\Downloads\\fall_all_id\\FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl
    """
    def __init__(self, pkl_path=r"C:\Users\KAUSHIK G\Downloads\fall_all_id\FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl", device="Waist"):
        import os
        self.pkl_path = pkl_path
        self.device = device
        self.trials = []
        self.trial_idx = 0
        self.sample_idx = 0
        self.latest_ir_distance = 1.65
        self.inject_fall = False
        self.fallback = RealisticSensorStreamer()
        self._load()

    def _load(self):
        import os
        if os.path.exists(self.pkl_path):
            try:
                import pandas as pd
                df = pd.read_pickle(self.pkl_path)
                subset = df[df['Device'] == self.device] if 'Device' in df.columns else df
                self.trials = subset.to_dict('records')
                print(f"[FallAllD] Loaded {len(self.trials)} clinical trials for device '{self.device}'.")
            except Exception as e:
                print(f"[FallAllD] Could not load pkl: {e}. Using fallback.")
                self.trials = []
        else:
            self.trials = []

    def get_next_sample(self):
        if not self.trials:
            sample = self.fallback.get_next_sample()
            self.latest_ir_distance = self.fallback.latest_ir_distance
            return sample

        trial = self.trials[self.trial_idx % len(self.trials)]
        acc_mat = trial.get('Acc', None)
        gyr_mat = trial.get('Gyr', None)

        if acc_mat is None or len(acc_mat) == 0:
            return self.fallback.get_next_sample()

        idx = self.sample_idx % len(acc_mat)
        self.sample_idx += 1
        if self.sample_idx >= len(acc_mat):
            self.sample_idx = 0
            self.trial_idx = (self.trial_idx + 1) % len(self.trials)

        # Normalize FallAllD 4096 LSB/g counts to m/s^2
        ax = float(acc_mat[idx, 0]) / 4096.0 * 9.81
        ay = float(acc_mat[idx, 1]) / 4096.0 * 9.81
        az = float(acc_mat[idx, 2]) / 4096.0 * 9.81

        gx = float(gyr_mat[idx, 0]) / 32.8 if gyr_mat is not None else 0.0
        gy = float(gyr_mat[idx, 1]) / 32.8 if gyr_mat is not None else 0.0
        gz = float(gyr_mat[idx, 2]) / 32.8 if gyr_mat is not None else 0.0

        svm = math.sqrt(ax * ax + ay * ay + az * az)
        tilt = math.degrees(math.acos(max(0.0, min(1.0, abs(az) / (svm + 1e-5)))))
        self.latest_ir_distance = float(max(0.15, 1.65 * (1.0 - 0.88 * (tilt / 90.0))))

        return [ax, ay, az, gx, gy, gz, 75.0]

    def trigger_fall(self):
        fall_trials = [i for i, t in enumerate(self.trials) if t.get('ActivityID', 0) > 100]
        if fall_trials:
            self.trial_idx = fall_trials[0]
            self.sample_idx = 0
            print(f"[FallAllD] Switched to clinical fall trial #{self.trial_idx} (Activity {self.trials[self.trial_idx].get('ActivityID')})")
        else:
            self.fallback.trigger_fall()

    def set_posture_mode(self, mode):
        self.fallback.set_posture_mode(mode)
