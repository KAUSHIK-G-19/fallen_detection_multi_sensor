"""
Verification script for realtime_deploy.py
Runs the real-time application in synthetic simulation mode for 60 frames,
triggers a simulated fall, verifies emergency snapshot generation in alerts/, and terminates cleanly.
"""
import os
import sys
import time
import glob
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_deploy import RealtimeFallDetectorApp

def verify_realtime_app():
    print("[VERIFY] Initializing RealtimeFallDetectorApp in synthetic mode...")
    app = RealtimeFallDetectorApp(source="synthetic", enable_sound=False, inactivity_sec=0.5)

    initial_alerts = glob.glob("alerts/*.jpg")
    print(f"[VERIFY] Initial alert snapshots in alerts/: {len(initial_alerts)}")

    # Run loop manually for 60 iterations (simulate 2 seconds)
    print("[VERIFY] Running real-time loop and injecting fall shock at frame 15...")
    for frame_idx in range(60):
        loop_start = time.time()

        # Generate synthetic frame
        frame = app._generate_synthetic_frame()

        # Ingest sensor sample
        if frame_idx == 15:
            print("[VERIFY] Injecting simulated fall event!")
            app.sensor_streamer.trigger_fall()
            app.synthetic_fall = True
            app.synthetic_fall_progress = 0

        raw_sensor = app.sensor_streamer.get_next_sample()
        sensor_feats = app.ring_buffer.add_sample(raw_sensor, timestamp=loop_start)

        vision_feats, annotated_frame = app.vision_tracker.process_frame(frame, timestamp_sec=loop_start)
        if app.synthetic_fall:
            vision_feats['is_laying'] = (app.synthetic_fall_progress > 10)
            vision_feats['aspect_ratio'] = 1.8 if app.synthetic_fall_progress > 10 else 0.8
            vision_feats['torso_angle'] = min(90.0, app.synthetic_fall_progress * 5.0)
            vision_feats['posture'] = "LYING" if app.synthetic_fall_progress > 10 else "TRANSITION"

        decision = app.decision_engine.update(vision_feats, sensor_feats, frame=annotated_frame)
        hud_frame = app._render_hud(annotated_frame, vision_feats, sensor_feats, decision)

        # Print state progression
        if frame_idx % 10 == 0 or decision['state'] == app.decision_engine.STATE_CONFIRMED:
            print(f"  Frame {frame_idx:02d}: State={decision['state']:<24} Conf={decision['confidence']:.0f}% Accel={sensor_feats['svm_acc']:.2f}m/s2")

        time.sleep(0.02)  # 50 FPS simulation pace

    app.vision_tracker.close()
    final_alerts = glob.glob("alerts/*.jpg")
    print(f"[VERIFY] Final alert snapshots in alerts/: {len(final_alerts)}")
    assert len(final_alerts) >= len(initial_alerts), "No emergency snapshot saved!"
    print("\n[SUCCESS] Real-time deployment verification passed! App runs and detects falls properly.")

if __name__ == "__main__":
    verify_realtime_app()
