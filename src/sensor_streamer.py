"""
Real-Time Sensor Ring Buffer and Streamer Module
Manages a continuous circular buffer (100 Hz, 2.0s = 200 samples) of wearable IMU and HR signals,
applies sample-by-sample 1D Kalman filtering, calculates dynamic acceleration, and detects impact spikes.
"""
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

        # Rolling statistics
        self.latest_svm_acc = 9.81
        self.latest_svm_gyro = 0.0
        self.latest_hr = 75.0
        self.max_impact_in_window = 9.81
        self.has_impact_spike = False

        # Physical constants
        self.G = 9.81
        self.IMPACT_THRESHOLD_MS2 = 2.0 * self.G  # ~19.6 m/s^2 (2.0g impact)
        self.FREEFALL_THRESHOLD_MS2 = 0.45 * self.G  # Freefall weightlessness trigger

    def add_sample(self, raw_sample, timestamp=None):
        """
        Add a single incoming sensor packet.
        raw_sample: list or 1D array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate]
        """
        if timestamp is None:
            timestamp = time.time()

        sample_arr = np.array(raw_sample, dtype=np.float32)
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

        # Evaluate window impact across both filtered and raw signals
        window = self.get_ordered_window()
        window_acc_mag = np.linalg.norm(window[:, :3], axis=1)
        raw_window = self.get_ordered_raw_window()
        raw_window_acc_mag = np.linalg.norm(raw_window[:, :3], axis=1)

        self.max_impact_in_window = float(max(np.max(window_acc_mag), np.max(raw_window_acc_mag)))
        self.has_impact_spike = bool(self.max_impact_in_window >= self.IMPACT_THRESHOLD_MS2 or raw_acc_mag >= self.IMPACT_THRESHOLD_MS2)

        return {
            'svm_acc': self.latest_svm_acc,
            'svm_gyro': self.latest_svm_gyro,
            'heart_rate': self.latest_hr,
            'max_impact': self.max_impact_in_window,
            'has_impact': self.has_impact_spike,
            'is_freefall': bool(self.latest_svm_acc <= self.FREEFALL_THRESHOLD_MS2)
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
    Simulates real-time 100 Hz sensor data stream.
    Can replay from dataset or generate realistic synthetic human motion:
    - Normal Walking (acc ~ 1g + periodic stepping oscillations, gyro low)
    - Sitting Down (smooth deceleration)
    - Fall Event (free-fall weightlessness -> sudden high impact spike -> prone static acceleration)
    """
    def __init__(self, dataset_csv_path=None, sr=100):
        self.sr = sr
        self.dt = 1.0 / sr
        self.time_step = 0
        self.data_df = None
        self.row_index = 0

        if dataset_csv_path and pd.io.common.file_exists(dataset_csv_path):
            try:
                self.data_df = pd.read_csv(dataset_csv_path)
                print(f"RealisticSensorStreamer: Loaded {len(self.data_df)} records from dataset.")
            except Exception:
                self.data_df = None

        # Fall injection controls
        self.inject_fall = False
        self.fall_progress = 0

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
                acc_x = np.random.normal(0.0, 0.5)
                acc_y = np.random.normal(0.0, 0.5)
                acc_z = np.random.normal(1.5, 0.5)  # Drop towards 0g
                gyro_x = np.random.normal(30.0, 10.0)
                gyro_y = np.random.normal(40.0, 10.0)
                gyro_z = np.random.normal(20.0, 5.0)
                hr = 85
            # Phase 2: Impact shock (0.2s - 0.35s = 15 samples)
            elif self.fall_progress < 35:
                acc_x = np.random.normal(12.0, 3.0)
                acc_y = np.random.normal(8.0, 2.0)
                acc_z = np.random.normal(24.0, 4.0)  # ~2.8g Impact!
                gyro_x = np.random.normal(150.0, 30.0)
                gyro_y = np.random.normal(180.0, 40.0)
                gyro_z = np.random.normal(90.0, 20.0)
                hr = 105
            # Phase 3: Lying stationary on floor (post-impact)
            else:
                acc_x = np.random.normal(9.7, 0.1)  # Gravity along X (lying sideways)
                acc_y = np.random.normal(0.3, 0.1)
                acc_z = np.random.normal(1.0, 0.1)
                gyro_x = np.random.normal(0.0, 1.0)
                gyro_y = np.random.normal(0.0, 1.0)
                gyro_z = np.random.normal(0.0, 1.0)
                hr = 95
                if self.fall_progress > 200:  # Reset after 2 seconds
                    self.inject_fall = False

            return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, hr]

        # Normal Walking / Sitting human kinematics
        # 1.5 Hz walking cadence rhythm
        step_freq = 1.6
        vertical_bounce = 1.2 * np.sin(2 * np.pi * step_freq * t)
        lateral_sway = 0.5 * np.cos(2 * np.pi * (step_freq / 2) * t)

        acc_x = lateral_sway + np.random.normal(0.0, 0.15)
        acc_y = np.random.normal(0.2, 0.15)
        acc_z = 9.81 + vertical_bounce + np.random.normal(0.0, 0.2)

        gyro_x = 10.0 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0.0, 2.0)
        gyro_y = 15.0 * np.cos(2 * np.pi * step_freq * t) + np.random.normal(0.0, 2.0)
        gyro_z = np.random.normal(0.0, 1.5)

        hr = 72 + int(3.0 * np.sin(2 * np.pi * 0.1 * t))

        return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, hr]
