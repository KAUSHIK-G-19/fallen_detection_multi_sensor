"""
Automated Test Suite for Real-Time Fall Detection System
Verifies:
1. Online Kalman Filter 1D & 2D stability
2. Real-Time Sensor Ring Buffer circular indexing and impact detection
3. Spatiotemporal Decision Engine finite state transitions
4. Machine Learning Model loading and prediction
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np


from src.kalman_filter import OnlineKalmanFilter1D, OnlineKalmanFilter2D
from src.sensor_streamer import RealtimeSensorRingBuffer, RealisticSensorStreamer
from src.decision_engine import FallDecisionEngine


def test_kalman_1d():
    kf = OnlineKalmanFilter1D(q=1e-4, r=0.04)
    # Feed step input with noise
    measurements = [10.0 + np.random.normal(0, 0.5) for _ in range(50)]
    filtered = [kf.update(m) for m in measurements]

    assert len(filtered) == 50
    # Filtered variance should be smaller than noisy raw variance
    assert np.std(filtered[-20:]) < np.std(measurements[-20:])
    print("[TEST PASSED] OnlineKalmanFilter1D operates correctly and denoises signal.")


def test_kalman_2d():
    kf = OnlineKalmanFilter2D(dt=1.0/30.0)
    # Moving point from (0,0) to (1,1)
    trajectory = [(i * 0.02, i * 0.02) for i in range(50)]
    estimates = [kf.update(x, y) for x, y in trajectory]

    assert len(estimates) == 50
    final_x, final_y, vx, vy = estimates[-1]
    assert abs(final_x - trajectory[-1][0]) < 0.1
    assert abs(final_y - trajectory[-1][1]) < 0.1
    print("[TEST PASSED] OnlineKalmanFilter2D tracks 2D point kinematics correctly.")


def test_sensor_ring_buffer():
    buffer = RealtimeSensorRingBuffer(capacity=200, sr=100)
    assert not buffer.is_ready()

    # Ingest 250 normal samples
    for i in range(250):
        buffer.add_sample([0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 75])

    assert buffer.is_ready()
    assert buffer.count == 250
    window = buffer.get_ordered_window()
    assert window.shape == (200, 7)
    assert not buffer.has_impact_spike

    # Ingest high impact shock sample (2.8g)
    buffer.add_sample([15.0, 10.0, 25.0, 120.0, 150.0, 80.0, 110])
    assert buffer.has_impact_spike
    assert buffer.max_impact_in_window > 25.0
    print("[TEST PASSED] RealtimeSensorRingBuffer maintains sliding window and detects impact.")


def test_decision_engine_state_machine():
    engine = FallDecisionEngine(inactivity_timeout_sec=0.2, impact_window_sec=0.5)

    # 1. Normal state
    normal_vision = {'is_laying': False, 'aspect_ratio': 0.45, 'torso_angle': 10.0, 'centroid_vy': 0.0}
    normal_sensor = {'svm_acc': 9.81, 'max_impact': 9.81, 'has_impact': False, 'is_freefall': False}
    d1 = engine.update(normal_vision, normal_sensor)
    assert d1['state'] == FallDecisionEngine.STATE_NORMAL

    # 2. Impact shock occurs
    impact_sensor = {'svm_acc': 26.0, 'max_impact': 26.0, 'has_impact': True, 'is_freefall': False}
    d2 = engine.update(normal_vision, impact_sensor)
    assert d2['state'] == FallDecisionEngine.STATE_IMPACT

    # 3. Posture collapses to horizontal lying
    lying_vision = {'is_laying': True, 'aspect_ratio': 1.6, 'torso_angle': 75.0, 'centroid_vy': 0.0}
    d3 = engine.update(lying_vision, normal_sensor)
    assert d3['state'] == FallDecisionEngine.STATE_COLLAPSE

    # 4. Inactivity begins
    time.sleep(0.1)
    d4 = engine.update(lying_vision, normal_sensor)
    assert d4['state'] == FallDecisionEngine.STATE_INACTIVITY

    # 5. Inactivity sustained beyond timeout -> CONFIRMED
    time.sleep(0.25)
    d5 = engine.update(lying_vision, normal_sensor)
    assert d5['state'] == FallDecisionEngine.STATE_CONFIRMED
    assert d5['is_fall_active'] is True

    # 6. Recovery reset
    engine.reset()
    assert engine.current_state == FallDecisionEngine.STATE_NORMAL
    print("[TEST PASSED] FallDecisionEngine finite state machine transitions verified.")


def test_false_alarm_guard_requires_posture_and_impact():
    engine = FallDecisionEngine(inactivity_timeout_sec=0.2, impact_window_sec=0.5)

    normal_vision = {'is_laying': False, 'aspect_ratio': 0.45, 'torso_angle': 10.0, 'centroid_vy': 0.0}
    normal_sensor = {'svm_acc': 9.81, 'max_impact': 9.81, 'has_impact': False, 'is_freefall': False}
    impact_only_sensor = {'svm_acc': 26.0, 'max_impact': 26.0, 'has_impact': True, 'is_freefall': False}

    decision = engine.update(normal_vision, impact_only_sensor)
    assert decision['state'] == FallDecisionEngine.STATE_IMPACT
    time.sleep(0.55)
    decision = engine.update(normal_vision, normal_sensor)
    assert decision['state'] == FallDecisionEngine.STATE_NORMAL
    print("[TEST PASSED] Decision engine suppresses impact-only false alarms without posture collapse.")


def test_event_logging_created_for_alerts():
    engine = FallDecisionEngine(inactivity_timeout_sec=0.05, impact_window_sec=0.2, alert_dir='alerts')
    assert engine.event_log_path.endswith('.jsonl')
    assert os.path.exists(os.path.dirname(engine.event_log_path))

    visual = {'is_laying': True, 'aspect_ratio': 1.6, 'torso_angle': 75.0, 'centroid_vy': 2.0}
    sensor = {'svm_acc': 30.0, 'max_impact': 30.0, 'has_impact': True, 'is_freefall': False}
    engine.update(visual, sensor)

    assert os.path.exists(engine.event_log_path)
    print("[TEST PASSED] Decision engine writes state transition events to the alert log.")


def test_saved_model_inference():
    scaler_path = "models/scaler.pkl"
    model_path = "models/fall_classifier.pkl"

    assert os.path.exists(scaler_path), "scaler.pkl missing"
    assert os.path.exists(model_path), "fall_classifier.pkl missing"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Test normal standing sample
    normal_feats = np.array([[0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 72, 9.81, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    pred_normal = model.predict(scaler.transform(normal_feats))[0]
    assert pred_normal == 0

    # Test violent fall impact sample
    fall_feats = np.array([[15.0, 8.0, 28.0, 150.0, 180.0, 90.0, 115, 32.7, 250.0, 22.9, 0.5, 0.3, 1.0]], dtype=np.float32)
    pred_fall = model.predict(scaler.transform(fall_feats))[0]
    assert pred_fall == 1
    print("[TEST PASSED] Saved ML model makes accurate predictions on test vectors.")


def test_multisensor_integration():
    buffer = RealtimeSensorRingBuffer(capacity=50, sr=100)
    streamer = RealisticSensorStreamer()

    # 1. Test standing posture
    streamer.set_posture_mode("standing")
    sample_standing = streamer.get_next_sample()
    feats_standing = buffer.add_sample(sample_standing, ir_distance=streamer.latest_ir_distance)
    assert feats_standing['posture_sensor'] in ["STANDING", "SITTING"]
    assert feats_standing['tilt_angle'] < 30.0
    assert feats_standing['ir_distance'] > 1.2

    # 2. Test lying posture
    streamer.set_posture_mode("lying")
    sample_lying = streamer.get_next_sample()
    feats_lying = buffer.add_sample(sample_lying, ir_distance=streamer.latest_ir_distance)
    assert feats_lying['posture_sensor'] == "LYING"
    assert feats_lying['tilt_angle'] > 60.0
    assert feats_lying['ir_distance'] < 0.4
    print("[TEST PASSED] Multi-sensor integration (IMU, MPU-6050, IR) correctly classifies posture.")


if __name__ == "__main__":
    test_kalman_1d()
    test_kalman_2d()
    test_sensor_ring_buffer()
    test_decision_engine_state_machine()
    test_saved_model_inference()
    test_multisensor_integration()
    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY!")
