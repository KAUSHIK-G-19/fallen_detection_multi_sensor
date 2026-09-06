"""
Online Kalman Filter Module for Real-Time Streaming Fall Detection
Implements both 1D (sensor signals) and 2D (coordinate tracking) Kalman filters.
"""
import numpy as np


class OnlineKalmanFilter1D:
    """
    Online 1D Kalman Filter for real-time sensor stream denoising (accel, gyro, HR).
    Maintains internal state and error covariance; updates sample-by-sample.
    """
    def __init__(self, q=1e-4, r=0.04, initial_value=0.0):
        self.q = float(q)  # Process variance
        self.r = float(r)  # Measurement variance
        self.x = float(initial_value)  # State estimate
        self.p = 1.0  # Error covariance
        self.initialized = False

    def reset(self, initial_value=0.0):
        self.x = float(initial_value)
        self.p = 1.0
        self.initialized = False

    def update(self, measurement):
        """
        Process a single incoming scalar measurement in real time.
        Returns the filtered estimate.
        """
        if np.isnan(measurement):
            return self.x

        if not self.initialized:
            self.x = float(measurement)
            self.p = 1.0
            self.initialized = True
            return self.x

        # Time Update (Predict)
        x_pred = self.x
        residual = abs(measurement - x_pred)
        # Adaptive innovation: If sudden physical impact shock occurs, allow fast tracking
        adaptive_q = self.q + (0.05 if residual > 3.0 else 0.0)
        p_pred = self.p + adaptive_q

        # Measurement Update (Correct)
        k = p_pred / (p_pred + self.r)
        self.x = x_pred + k * (measurement - x_pred)
        self.p = (1.0 - k) * p_pred

        return self.x


class OnlineKalmanFilter2D:
    """
    Online 2D Kalman Filter for tracking and smoothing 2D point trajectories (e.g. pose keypoints).
    Uses a constant-velocity kinematic model: state = [x, y, vx, vy]^T.
    """
    def __init__(self, dt=1.0/30.0, process_noise=1e-3, measurement_noise=1e-2):
        self.dt = dt
        # State transition matrix
        self.A = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Measurement matrix (measures x, y)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        # Covariances
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.state = np.zeros(4, dtype=np.float32)
        self.initialized = False

    def reset(self, x=0.0, y=0.0):
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1.0
        self.initialized = False

    def update(self, x, y):
        """
        Process a single incoming (x, y) coordinate.
        Returns smoothed (x, y, vx, vy).
        """
        if not self.initialized:
            self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
            self.initialized = True
            return float(x), float(y), 0.0, 0.0

        # Predict
        state_pred = self.A @ self.state
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Correct
        z = np.array([x, y], dtype=np.float32)
        y_residual = z - (self.H @ state_pred)
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.state = state_pred + K @ y_residual
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ P_pred

        return float(self.state[0]), float(self.state[1]), float(self.state[2]), float(self.state[3])
