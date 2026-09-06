"""
Spatiotemporal Decision Engine Module
Implements a 4-stage finite state machine combining real-time vision pose geometry
and wearable sensor dynamics to eliminate false alarms and confirm genuine falls.
"""
import time
import os
import cv2


class FallDecisionEngine:
    """
    Finite State Machine for Multimodal Fall Verification:
    - NORMAL (0): Normal upright or seated activity
    - IMPACT_SUSPECTED (1): Sudden sensor acceleration spike (>2.0g) or camera descent velocity
    - TRANSITION_COLLAPSE (2): Orientation inversion from vertical to horizontal (AR > 1.15)
    - POST_FALL_INACTIVITY (3): Prone immobility on ground for sustained duration (1.5 - 2.0s)
    - FALL_CONFIRMED (4): Emergency alert dispatched, audio alarm triggered, snapshot logged
    """
    STATE_NORMAL = "NORMAL"
    STATE_IMPACT = "IMPACT_DETECTED"
    STATE_COLLAPSE = "POSTURE_COLLAPSED"
    STATE_INACTIVITY = "INACTIVITY_VERIFICATION"
    STATE_CONFIRMED = "FALL_CONFIRMED"

    def __init__(self, inactivity_timeout_sec=1.8, impact_window_sec=1.5, alert_dir="alerts"):
        self.inactivity_timeout = inactivity_timeout_sec
        self.impact_window = impact_window_sec
        self.alert_dir = alert_dir
        os.makedirs(self.alert_dir, exist_ok=True)

        self.current_state = self.STATE_NORMAL
        self.time_impact = 0.0
        self.time_collapse = 0.0
        self.fall_confidence = 0.0  # 0.0 to 100.0%

        # Alert cooldown to prevent flooding
        self.last_alert_time = 0.0
        self.alert_cooldown = 10.0
        self.snapshot_saved = False

        # Physical constants
        self.G = 9.81
        self.ACC_IMPACT_THRESHOLD = 2.0 * self.G  # 19.6 m/s^2

    def update(self, vision_features, sensor_features, frame=None):
        """
        Evaluate multimodal inputs for the current time frame.
        vision_features: dict from VisionPoseTracker
        sensor_features: dict from RealtimeSensorRingBuffer
        frame: current BGR frame (for capturing emergency snapshot if fall is confirmed)

        Returns:
            decision: dict with state, confidence, alert flags, and telemetry
        """
        now = time.time()
        is_laying = vision_features.get('is_laying', False)
        aspect_ratio = vision_features.get('aspect_ratio', 0.5)
        torso_angle = vision_features.get('torso_angle', 0.0)
        centroid_vy = vision_features.get('centroid_vy', 0.0)

        svm_acc = sensor_features.get('svm_acc', self.G)
        max_impact = sensor_features.get('max_impact', self.G)
        has_sensor_impact = sensor_features.get('has_impact', False)
        is_freefall = sensor_features.get('is_freefall', False)

        trigger_alert = False
        emergency_snapshot_path = None

        # -------------------------------------------------------------
        # STATE 0: NORMAL
        # -------------------------------------------------------------
        if self.current_state == self.STATE_NORMAL:
            self.fall_confidence = max(0.0, self.fall_confidence - 5.0)

            # Impact trigger: High sensor acceleration spike OR sharp camera descent
            vision_impact = (centroid_vy > 1.2 and aspect_ratio > 0.8)
            sensor_impact = has_sensor_impact or (svm_acc > self.ACC_IMPACT_THRESHOLD)

            if sensor_impact or vision_impact:
                self.current_state = self.STATE_IMPACT
                self.time_impact = now
                self.fall_confidence = 40.0
                self.snapshot_saved = False

            # Direct collapse trigger (in case sensor wasn't worn but vision sees sudden prone posture)
            elif is_laying and aspect_ratio > 1.3:
                self.current_state = self.STATE_COLLAPSE
                self.time_collapse = now
                self.fall_confidence = 55.0

        # -------------------------------------------------------------
        # STATE 1: IMPACT_DETECTED
        # -------------------------------------------------------------
        elif self.current_state == self.STATE_IMPACT:
            time_since_impact = now - self.time_impact
            self.fall_confidence = 60.0

            # Did posture collapse into lying position within the impact window?
            if is_laying or aspect_ratio > 1.15 or torso_angle > 45.0:
                self.current_state = self.STATE_COLLAPSE
                self.time_collapse = now
                self.fall_confidence = 75.0

            # If impact happened but person remains standing upright -> False alarm, return to normal
            elif time_since_impact > self.impact_window:
                self.current_state = self.STATE_NORMAL
                self.fall_confidence = 10.0

        # -------------------------------------------------------------
        # STATE 2: POSTURE_COLLAPSED
        # -------------------------------------------------------------
        elif self.current_state == self.STATE_COLLAPSE:
            time_in_collapse = now - self.time_collapse
            self.fall_confidence = 80.0

            # Check if person recovered and stood back up
            if not is_laying and aspect_ratio < 0.75 and torso_angle < 30.0:
                # Quick recovery (e.g. picked up dropped item)
                self.current_state = self.STATE_NORMAL
                self.fall_confidence = 15.0
            collapse_delay = min(0.4, self.inactivity_timeout * 0.3)
            if time_in_collapse >= collapse_delay:
                # Person remained down -> Proceed to inactivity verification
                self.current_state = self.STATE_INACTIVITY
                self.fall_confidence = 88.0

        # -------------------------------------------------------------
        # STATE 3: INACTIVITY_VERIFICATION
        # -------------------------------------------------------------
        elif self.current_state == self.STATE_INACTIVITY:
            time_prone = now - self.time_collapse

            # If still down on the floor for sustained duration -> CONFIRM FALL!
            if is_laying or aspect_ratio > 1.1:
                progress = min(1.0, time_prone / self.inactivity_timeout)
                self.fall_confidence = 88.0 + (12.0 * progress)

                if time_prone >= self.inactivity_timeout:
                    self.current_state = self.STATE_CONFIRMED
                    self.fall_confidence = 100.0

                    if (now - self.last_alert_time) > self.alert_cooldown:
                        trigger_alert = True
                        self.last_alert_time = now

                        # Save incident snapshot
                        if frame is not None and not self.snapshot_saved:
                            filename = f"fall_incident_{int(now)}.jpg"
                            emergency_snapshot_path = os.path.join(self.alert_dir, filename)
                            cv2.imwrite(emergency_snapshot_path, frame)
                            self.snapshot_saved = True
                            print(f"[EMERGENCY ALERT] Fall incident snapshot saved to: {emergency_snapshot_path}")

            else:
                # Person stood up before timeout
                self.current_state = self.STATE_NORMAL
                self.fall_confidence = 10.0

        # -------------------------------------------------------------
        # STATE 4: FALL_CONFIRMED
        # -------------------------------------------------------------
        elif self.current_state == self.STATE_CONFIRMED:
            self.fall_confidence = 100.0
            # Remain in confirmed state until person recovers/stands up or is assisted
            if not is_laying and aspect_ratio < 0.7 and torso_angle < 25.0:
                print("[SYSTEM] Person has recovered and is upright. Resetting state to NORMAL.")
                self.current_state = self.STATE_NORMAL
                self.fall_confidence = 0.0
                self.snapshot_saved = False

        return {
            'state': self.current_state,
            'confidence': float(self.fall_confidence),
            'trigger_alert': trigger_alert,
            'snapshot_path': emergency_snapshot_path,
            'is_fall_active': (self.current_state == self.STATE_CONFIRMED)
        }

    def reset(self):
        self.current_state = self.STATE_NORMAL
        self.fall_confidence = 0.0
        self.snapshot_saved = False
