"""
Multimodal Fall Detection System
Combines sensor data (accelerometer, gyroscope, HR) with image pose keypoints
Uses Kalman filtering for both modalities
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    print("WARNING: TensorFlow not found.")
    TF_AVAILABLE = False

# ================= CONFIGURATION =================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SENSOR_DATA_PATH = os.path.join(PROJECT_DIR, "fall_detection_dataset.csv")
IMAGE_DATASET_PATH = r"C:\Users\KAUSHIK G\Downloads\archive (3)\laying_dataset"
OUTPUT_FILE_PATH = os.path.join(PROJECT_DIR, "C:\\Users\\KAUSHIK G\\OneDrive\\Desktop\\pd project\\output.xlsx")
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE_PATH)

SR = 100                    # Sampling rate (Hz)
WINDOW_SEC = 2.0
STEP_SEC = 1.0
NUM_EPOCHS = 1000
BATCH_SIZE = 32
RANDOM_SEED = 42
N_KEYPOINTS = 17           # YOLO pose has 17 keypoints

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ================= KALMAN FILTER IMPLEMENTATION =================
def apply_kalman_filter(signal):
    """
    Simple 1D Kalman Filter to smooth sensor noise
    """
    n_iter = len(signal)
    sz = (n_iter,)

    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)

    Q = 1e-5          # Process variance
    R = 0.1 ** 2      # Measurement variance

    xhat[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (signal[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


def apply_kalman_2d(signal_x, signal_y):
    """Apply Kalman filter to 2D coordinates (x, y)"""
    filtered_x = apply_kalman_filter(signal_x)
    filtered_y = apply_kalman_filter(signal_y)
    return filtered_x, filtered_y


# ================= IMAGE KEYPOINT PROCESSING =================
def extract_keypoints_from_labels(label_file):
    """
    Parse YOLO keypoint labels
    Format: class_id x y w h kpt1_x kpt1_y kpt1_conf ... kpt17_x kpt17_y kpt17_conf
    Returns: keypoints (17, 2), label (0=laying, 1=standing)
    """
    try:
        with open(label_file, 'r') as f:
            line = f.read().strip()
        
        values = list(map(float, line.split()))
        label = int(values[0])  # 0=laying (fall), 1=standing (normal)
        
        # Extract 17 keypoints (x, y pairs)
        keypoints = []
        # Skip first 5 values (class, x, y, w, h) and take every 3rd value (x, y, conf)
        for i in range(5, 5 + N_KEYPOINTS * 3, 3):
            if i + 1 < len(values):
                kpt_x = values[i]
                kpt_y = values[i + 1]
                # conf = values[i + 2]  # Confidence score
                keypoints.append([kpt_x, kpt_y])
        
        keypoints = np.array(keypoints)
        return keypoints, label
    except Exception as e:
        print(f"Error reading {label_file}: {e}")
        return None, None


def compute_pose_features(keypoints):
    """
    Compute hand-crafted features from keypoints
    - Joint distances
    - Joint angles
    - Body height
    - Symmetry scores
    """
    if keypoints is None or len(keypoints) < N_KEYPOINTS:
        return None
    
    try:
        # Limb connections (COCO format)
        limbs = [
            (5, 6), (7, 5), (7, 9), (9, 11),  # Left arm
            (6, 8), (8, 10), (10, 12),         # Right arm
            (5, 11), (6, 12),                  # Torso-arm
            (11, 13), (13, 15),                # Left leg
            (12, 14), (14, 16),                # Right leg
        ]
        
        features = []
        
        # 1. Joint distances (limb lengths)
        for start, end in limbs:
            if start < len(keypoints) and end < len(keypoints):
                dist = np.linalg.norm(keypoints[start] - keypoints[end])
                features.append(dist)
        
        # 2. Body position (centroid X, Y)
        centroid = np.mean(keypoints, axis=0)
        features.extend(centroid)
        
        # 3. Body height (vertical span)
        height = np.max(keypoints[:, 1]) - np.min(keypoints[:, 1])
        features.append(height)
        
        # 4. Body width (horizontal span)
        width = np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])
        features.append(width)
        
        # 5. Horizontal pose indicator (head position relative to center)
        head_idx = 0
        head_pos_y = keypoints[head_idx, 1]
        features.append(head_pos_y)
        
        return np.array(features)
    except Exception as e:
        print(f"Error computing pose features: {e}")
        return None


def load_image_dataset():
    """
    Load image keypoints from dataset
    Returns sequences of keypoint features
    """
    images_dir = os.path.join(IMAGE_DATASET_PATH, "images")
    labels_dir = os.path.join(IMAGE_DATASET_PATH, "labels")
    
    print(f"Loading image dataset from {IMAGE_DATASET_PATH}", flush=True)
    
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    print(f"Found {len(label_files)} label files", flush=True)
    
    X_image = []
    y_image = []
    
    for i, label_file in enumerate(label_files):
        if i % 20 == 0:
            print(f"  Processing image {i+1}/{len(label_files)}...", flush=True)
            
        keypoints, label = extract_keypoints_from_labels(label_file)
        
        if keypoints is not None:
            # Apply Kalman filter to keypoints
            filtered_kpts_x, filtered_kpts_y = apply_kalman_2d(
                keypoints[:, 0], keypoints[:, 1]
            )
            filtered_keypoints = np.column_stack([filtered_kpts_x, filtered_kpts_y])
            
            # Compute features
            features = compute_pose_features(filtered_keypoints)
            
            if features is not None:
                X_image.append(features)
                y_image.append(label)
    
    print(f"✅ Loaded {len(X_image)} image samples", flush=True)
    return np.array(X_image), np.array(y_image)


# ================= SENSOR DATA PROCESSING =================
def load_and_clean_data(path):
    """Load and preprocess sensor data"""
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine='openpyxl')
    except FileNotFoundError:
        print(f"Error: Input file not found at {path}")
        raise
    
    df.columns = df.columns.str.strip().str.lower()
    required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate', 'fall_event']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    df.dropna(subset=required_cols, inplace=True)

    print("Applying Kalman Filtering to sensor signals...")
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in sensor_cols:
        df[col] = apply_kalman_filter(df[col].values)

    return df


def preprocess_and_segment(df):
    """Segment sensor data into windows"""
    feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate']
    window_steps = int(WINDOW_SEC * SR)
    step_size = int(STEP_SEC * SR)
    X, y = [], []
    
    for i in range(0, len(df) - window_steps, step_size):
        window = df.iloc[i: i + window_steps]
        X.append(window[feature_cols].values)
        y.append(1 if window['fall_event'].max() == 1 else 0)
    
    return np.array(X), np.array(y)


# ================= MULTIMODAL FUSION =================
def fuse_multimodal_data(X_sensor, X_image, y_sensor, y_image):
    """
    Fuse sensor and image data
    Combines both modalities while handling different sample counts
    """
    print(f"Sensor samples: {len(X_sensor)}, Image samples: {len(X_image)}")
    
    # Normalize to same scale
    sensor_scaler = StandardScaler()
    image_scaler = StandardScaler()
    
    X_sensor_scaled = sensor_scaler.fit_transform(
        X_sensor.reshape(-1, X_sensor.shape[-1])
    ).reshape(X_sensor.shape)
    
    X_image_scaled = image_scaler.fit_transform(X_image)
    
    # Since they have different temporal dimensions, use image data as additional features
    # Expand image features to match sensor sample count
    if len(y_image) > 0:
        # Assign image labels based on proximity
        y_combined = y_sensor.copy()
        
        # Convert image features to match sensor window format
        # Repeat image features to create "sequences"
        X_image_repeated = np.repeat(
            X_image_scaled[:, np.newaxis, :],
            X_sensor.shape[1],
            axis=1
        )
        
        # Match lengths by duplication or truncation
        n_samples = min(len(X_sensor), len(X_image_repeated))
        X_sensor_matched = X_sensor_scaled[:n_samples]
        X_image_matched = X_image_repeated[:n_samples]
        y_combined = y_sensor[:n_samples]
        
        return X_sensor_matched, X_image_matched, y_combined
    else:
        return X_sensor_scaled, None, y_sensor


# ================= MODEL BUILDING =================
def build_multimodal_lstm_model(sensor_shape, image_shape=None):
    """
    Build LSTM model for multimodal data
    """
    if image_shape is None:
        # Sensor-only model
        model = Sequential([
            Input(shape=sensor_shape),
            LSTM(32, return_sequences=True),
            GlobalAveragePooling1D(),
            Dropout(0.5),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    else:
        # Multimodal model
        # Sensor branch
        sensor_input = Input(shape=sensor_shape, name='sensor_input')
        sensor_lstm = LSTM(32, return_sequences=True)(sensor_input)
        sensor_pool = GlobalAveragePooling1D()(sensor_lstm)
        sensor_dropout = Dropout(0.5)(sensor_pool)
        
        # Image branch
        image_input = Input(shape=image_shape, name='image_input')
        image_lstm = LSTM(24, return_sequences=True)(image_input)
        image_pool = GlobalAveragePooling1D()(image_lstm)
        image_dropout = Dropout(0.5)(image_pool)
        
        # Fusion
        merged = Concatenate()([sensor_dropout, image_dropout])
        dense1 = Dense(24, activation='relu')(merged)
        dense2 = Dense(12, activation='relu')(dense1)
        output = Dense(1, activation='sigmoid')(dense2)
        
        model = tf.keras.Model(
            inputs=[sensor_input, image_input],
            outputs=output
        )
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def kalman_anomaly_detection(X_data, threshold=2.5):
    """Kalman-based anomaly detector"""
    predictions = []
    for sample in X_data:
        acc_mag = np.sqrt(np.sum(sample[:, :3] ** 2, axis=1))
        if np.max(acc_mag) > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)


# ================= MAIN PIPELINE =================
def main():
    print("\n" + "="*50, flush=True)
    print("MULTIMODAL FALL DETECTION SYSTEM", flush=True)
    print("="*50, flush=True)
    
    results = []
    
    # 1. Load sensor data
    print("\n" + "="*50, flush=True)
    print("LOADING SENSOR DATA", flush=True)
    print("="*50, flush=True)
    df_sensor = load_and_clean_data(SENSOR_DATA_PATH)
    X_sensor, y_sensor = preprocess_and_segment(df_sensor)
    print(f"Sensor data shape: {X_sensor.shape}, Labels: {y_sensor.shape}", flush=True)
    
    # 2. Load image keypoint data
    print("\n" + "="*50)
    print("LOADING IMAGE KEYPOINT DATA")
    print("="*50)
    X_image, y_image = load_image_dataset()
    
    # 3. Fuse multimodal data
    print("\n" + "="*50)
    print("FUSING MULTIMODAL DATA")
    print("="*50)
    X_sensor_matched, X_image_matched, y_combined = fuse_multimodal_data(
        X_sensor, X_image, y_sensor, y_image
    )
    print(f"Sensor matched: {X_sensor_matched.shape}")
    if X_image_matched is not None:
        print(f"Image matched: {X_image_matched.shape}")
    print(f"Combined labels: {y_combined.shape}")
    
    # 4. Train-test split
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    X_train_s, X_test_s, y_train, y_test = train_test_split(
        X_sensor_matched, y_combined,
        test_size=0.3,
        stratify=y_combined if len(np.unique(y_combined)) > 1 else None,
        random_state=RANDOM_SEED
    )
    
    if X_image_matched is not None:
        X_train_i, X_test_i = train_test_split(
            X_image_matched,
            test_size=0.3,
            random_state=RANDOM_SEED
        )
        print(f"Train sensor: {X_train_s.shape}, Train image: {X_train_i.shape}")
        print(f"Test sensor: {X_test_s.shape}, Test image: {X_test_i.shape}")
    else:
        X_train_i = X_test_i = None
        print(f"Train sensor: {X_train_s.shape}, Test sensor: {X_test_s.shape}")
    
    print(f"Train labels: {y_train.shape}, Test labels: {y_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # 5. Train Multimodal LSTM
    if TF_AVAILABLE and X_train_i is not None:
        print("\n" + "="*50)
        print("TRAINING MULTIMODAL LSTM")
        print("="*50)
        
        model = build_multimodal_lstm_model(
            (X_train_s.shape[1], X_train_s.shape[2]),
            (X_train_i.shape[1], X_train_i.shape[2])
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            [X_train_s, X_train_i], y_train,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        y_pred_prob = model.predict([X_test_s, X_test_i])
        y_pred_multimodal = (y_pred_prob > 0.5).astype(int).flatten()
        
        acc_mm = accuracy_score(y_test, y_pred_multimodal)
        _, _, f1_mm, _ = precision_recall_fscore_support(
            y_test, y_pred_multimodal, average='binary', zero_division=0
        )
        results.append({'Model': 'Multimodal LSTM', 'Accuracy': acc_mm, 'F1_Score': f1_mm})
        
        # Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Multimodal LSTM Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Multimodal LSTM Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'multimodal_training.png'))
        plt.close()
        
        print(f"Multimodal LSTM - Accuracy: {acc_mm:.4f}, F1: {f1_mm:.4f}")
    
    # 6. Kalman Anomaly Detector
    print("\n" + "="*50)
    print("KALMAN ANOMALY DETECTOR")
    print("="*50)
    y_pred_kalman = kalman_anomaly_detection(X_test_s)
    acc_k = accuracy_score(y_test, y_pred_kalman)
    _, _, f1_k, _ = precision_recall_fscore_support(
        y_test, y_pred_kalman, average='binary', zero_division=0
    )
    results.append({'Model': 'Kalman Algorithm', 'Accuracy': acc_k, 'F1_Score': f1_k})
    print(f"Kalman - Accuracy: {acc_k:.4f}, F1: {f1_k:.4f}")
    
    # 7. Save results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame({'True_Label': y_test, 'Kalman_Pred': y_pred_kalman})
    
    if 'y_pred_multimodal' in locals():
        preds_df['Multimodal_LSTM_Pred'] = y_pred_multimodal
    
    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Metrics', index=False)
        preds_df.to_excel(writer, sheet_name='Predictions', index=False)
    
    print(f"\n✅ Success! Results saved to {OUTPUT_FILE_PATH}")
    print("\n" + "="*50)
    print("FINAL MODEL METRICS")
    print("="*50)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
