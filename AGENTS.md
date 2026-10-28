# Agent Instructions

## Project Shape

- This is a script-driven Python 3.10+ fall-detection system; there is no `pyproject.toml`, `requirements.txt`, or package build command.
- Run commands from the repository root because runtime paths such as `models/` and `alerts/` are relative.
- Read [README.md](README.md) for the complete architecture, installation, and deployment description before changing user-facing behavior.

## Architecture Boundaries

- [realtime_deploy.py](realtime_deploy.py) owns the CLI, input sources, model loading, HUD, recording, and alert presentation.
- [src/vision_tracker.py](src/vision_tracker.py) owns MediaPipe/OpenCV pose extraction, geometry, posture classification, and pose smoothing.
- [src/sensor_streamer.py](src/sensor_streamer.py) owns synthetic 100 Hz sensor data and the 200-sample sliding buffer.
- [src/kalman_filter.py](src/kalman_filter.py) owns the online 1D sensor and 2D pose filters.
- [src/decision_engine.py](src/decision_engine.py) owns the spatiotemporal fall FSM, event JSONL logging, and alert snapshots.
- [src/train_model.py](src/train_model.py) owns synthetic training data and persisted model generation.
- Treat [sample.py](sample.py), [sample_multimodal.py](sample_multimodal.py), and [sample_3.py](sample_3.py) as legacy or experimental workflows unless a task explicitly targets them.

## Validation

- Install the documented dependencies in an activated `.venv`; TensorFlow and MediaPipe are optional for some paths, but deployment capabilities vary without them.
- Run `python tests/test_system.py` for the direct assertion-based system checks.
- Run `python tests/verify_deployment.py` for the headless synthetic deployment smoke test.
- Run `python -m src.train_model` only when changing training or model persistence behavior.
- Prefer `python realtime_deploy.py --source synthetic` for a hardware-free manual check before webcam or video input.
- The test script's `__main__` block does not call every test function; use pytest discovery only if pytest is available and full discovery is needed.

## Change Rules

- Preserve the public data dictionaries and state constants exchanged between vision, sensors, and the decision engine unless the task requires a coordinated contract change.
- Keep imports rooted at `src.*` and preserve root execution assumptions.
- Do not commit generated alert snapshots, event logs, caches, or other runtime artifacts; inspect `git status` before and after validation.
- Tests may append to `alerts/fall_event_log.jsonl` or create alert artifacts. Avoid deleting unrelated existing artifacts or reverting user changes in the worktree.
- Keep changes focused and add or update a nearby assertion when changing thresholds, FSM transitions, feature ordering, or model behavior.