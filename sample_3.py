import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_class_weight
# Optional: SMOTE for oversampling engineered features (install imbalanced-learn to enable)
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras import regularizers
    from tensorflow.keras.losses import BinaryCrossentropy

    TF_AVAILABLE = True
    tf.random.set_seed(42)
except ImportError:
    print("WARNING: TensorFlow not found.")
    TF_AVAILABLE = False

# ================= CONFIGURATION =================
# Use project-relative paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_PATH = os.path.join(PROJECT_DIR, "fall_detection_dataset.csv")
OUTPUT_FILE_PATH = os.path.join(PROJECT_DIR, "Fallall_output.xlsx")
OUTPUT_DIR = os.path.dirname(OUTPUT_FILE_PATH)

SR = 100
WINDOW_SEC = 2.0
STEP_SEC = 1.0
NUM_EPOCHS = 200
BATCH_SIZE = 32
RANDOM_SEED = 42

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ================= KALMAN ALGORITHM FUNCTIONS =================
def apply_kalman_filter(signal):
    """
    Simple 1D Kalman Filter implementation to smooth sensor noise.
    """
    n_iter = len(signal)
    sz = (n_iter,)

    # Allocate space for arrays
    xhat = np.zeros(sz)       # a posteri estimate of x
    P = np.zeros(sz)          # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)     # a priori error estimate
    K = np.zeros(sz)          # gain or blending factor

    Q = 1e-5      # process variance
    R = 0.1 ** 2  # estimate of measurement variance

    # Initial guesses
    xhat[0] = signal[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Time Update (Predict)
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # Measurement Update (Correct)
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (signal[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


def kalman_anomaly_detection(X_data, threshold=2.5):
    """
    Detects falls by looking for high variance in the Kalman-smoothed signals.
    """
    predictions = []
    for sample in X_data:
        # Calculate the magnitude of acceleration (SVM - Signal Vector Magnitude)
        acc_mag = np.sqrt(np.sum(sample[:, :3] ** 2, axis=1))
        # If the smoothed acceleration exceeds the threshold, label as Fall
        if np.max(acc_mag) > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)


# ================= DATA PROCESSING =================
def load_and_clean_data(path):
    """Load data from Excel or CSV file"""
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine='openpyxl')
    except FileNotFoundError as e:
        print(f"Error: Input file not found at {path}")
        raise
    
    df.columns = df.columns.str.strip().str.lower()
    required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate', 'fall_event']
    
    # Check if all required columns exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
    
    df.dropna(subset=required_cols, inplace=True)

    # --- APPLY KALMAN FILTER TO ALL SENSOR COLUMNS ---
    print("Applying Kalman Filtering to signals...")
    sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    for col in sensor_cols:
        df[col] = apply_kalman_filter(df[col].values)

    return df


def engineer_window_features(window_df, feature_cols):
    """Create compact statistical features for each sensor window."""
    features = []

    for col in feature_cols:
        values = window_df[col].astype(float).to_numpy()
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values),
            np.percentile(values, 25),
            np.percentile(values, 75),
            np.max(np.abs(values)),
            np.mean(np.abs(np.diff(values))) if len(values) > 1 else 0.0,
        ])

    acc_vals = window_df[['acc_x', 'acc_y', 'acc_z']].astype(float).to_numpy()
    gyro_vals = window_df[['gyro_x', 'gyro_y', 'gyro_z']].astype(float).to_numpy()
    acc_mag = np.sqrt(np.sum(acc_vals ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro_vals ** 2, axis=1))

    features.extend([
        np.mean(acc_mag),
        np.std(acc_mag),
        np.max(acc_mag),
        np.mean(gyro_mag),
        np.std(gyro_mag),
        np.max(gyro_mag),
        np.mean(window_df['heart_rate'].astype(float).to_numpy()),
        np.std(window_df['heart_rate'].astype(float).to_numpy()),
        np.max(window_df['heart_rate'].astype(float).to_numpy()),
        np.min(window_df['heart_rate'].astype(float).to_numpy()),
    ])

    return np.array(features, dtype=float)


def preprocess_and_segment(df):
    feature_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'heart_rate']
    window_steps = int(WINDOW_SEC * SR)
    step_size = int(STEP_SEC * SR)
    X_seq, X_features, y = [], [], []
    for i in range(0, len(df) - window_steps, step_size):
        window = df.iloc[i: i + window_steps]
        X_seq.append(window[feature_cols].values)
        X_features.append(engineer_window_features(window, feature_cols))
        y.append(1 if window['fall_event'].max() == 1 else 0)
    return np.array(X_seq), np.array(X_features), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        LSTM(16, return_sequences=False, kernel_regularizer=regularizers.l2(1e-4)),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model


def build_tree_model():
    return ExtraTreesClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        class_weight='balanced_subsample',
        n_jobs=-1,
        min_samples_leaf=1,
    )


def tune_tree_model(X, y, random_state=RANDOM_SEED):
    """Run a small GridSearchCV to find better tree hyperparameters."""
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [None, 6, 12],
        'min_samples_leaf': [1, 2, 4]
    }
    et = ExtraTreesClassifier(random_state=random_state, class_weight='balanced_subsample', n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    gs = GridSearchCV(et, param_grid, scoring='f1', cv=cv, n_jobs=-1, verbose=0)
    gs.fit(X, y)
    print(f"GridSearchCV best params: {gs.best_params_}")
    return gs.best_estimator_


# ================= MAIN =================
def main():
    # 1. Load, Clean, and Kalman Filter
    print(f"Loading data from {INPUT_FILE_PATH}")
    df = load_and_clean_data(INPUT_FILE_PATH)

    # 2. Segment
    print("Segmenting data into windows...")
    X_seq, X_features, y = preprocess_and_segment(df)

    if len(X_seq) == 0:
        print("Error: No valid data segments created!")
        return

    # Quick dataset diagnostics
    classes, counts = np.unique(y, return_counts=True)
    print(f"Total segments: {len(y)}; class distribution: {dict(zip(classes, counts))}")

    # 3. Split
    X_train_seq_raw, X_test_seq_raw, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_seq, X_features, y,
        test_size=0.3,
        stratify=y if len(np.unique(y)) > 1 else None,
        random_state=RANDOM_SEED
    )

    # Recompute train distribution
    t_classes, t_counts = np.unique(y_train, return_counts=True)
    print(f"Train distribution: {dict(zip(t_classes, t_counts))}")

    # Optionally apply SMOTE to engineered features to help tree models (if available)
    X_train_feat_res, y_train_res = X_train_feat, y_train
    if IMBLEARN_AVAILABLE and len(np.unique(y_train)) > 1:
        try:
            # only apply if imbalance exists
            minority = np.min(t_counts)
            if minority < (0.5 * np.max(t_counts)):
                print("Applying SMOTE to engineered features to reduce class imbalance...")
                sm = SMOTE(random_state=RANDOM_SEED)
                X_train_feat_res, y_train_res = sm.fit_resample(X_train_feat, y_train)
                print(f"After SMOTE train distribution: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")
        except Exception as e:
            print("SMOTE failed, proceeding without it:", e)

    # compute class weight mapping for use in LSTM training
    class_weight_map = None
    if len(np.unique(y_train)) > 1:
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_map = {int(cls): float(w) for cls, w in zip(np.unique(y_train), cw)}
        print(f"Computed class weights for LSTM: {class_weight_map}")

    # 4. Scale sequential inputs for LSTM
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_seq_raw.reshape(-1, X_train_seq_raw.shape[-1])).reshape(X_train_seq_raw.shape)
    X_test = scaler.transform(X_test_seq_raw.reshape(-1, X_test_seq_raw.shape[-1])).reshape(X_test_seq_raw.shape)

    results = []

    # --- MODEL 1: LSTM (Deep Learning) ---
    y_pred_lstm = None
    if TF_AVAILABLE:
        print("Training LSTM Model...")
        lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=12,
            min_delta=1e-4,
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        history = lstm_model.fit(
            X_train, y_train,
            epochs=NUM_EPOCHS,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr],
            class_weight=class_weight_map,
            verbose=1,
            shuffle=True
        )

        y_pred_prob = lstm_model.predict(X_test)
        y_pred_lstm = (y_pred_prob > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred_lstm)
        _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred_lstm, average='binary', zero_division=0)
        results.append({'Model': 'LSTM', 'Accuracy': acc, 'F1_Score': f1})

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('LSTM Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- MODEL 2: EXTRA TREES (ENGINEERED FEATURES) ---
    print("Tuning and training ExtraTrees classifier on engineered features...")
    try:
        best_tree = tune_tree_model(X_train_feat_res, y_train_res)
    except Exception:
        print("GridSearch failed or skipped; falling back to default ExtraTrees settings.")
        best_tree = build_tree_model()

    best_tree.fit(X_train_feat_res, y_train_res)
    y_pred_tree = best_tree.predict(X_test_feat)
    acc_tree = accuracy_score(y_test, y_pred_tree)
    _, _, f1_tree, _ = precision_recall_fscore_support(y_test, y_pred_tree, average='binary', zero_division=0)
    results.append({'Model': 'ExtraTrees', 'Accuracy': acc_tree, 'F1_Score': f1_tree})

    # --- MODEL 3: KALMAN ANOMALY DETECTOR ---
    print("Running Kalman Anomaly Algorithm...")
    y_pred_kalman = kalman_anomaly_detection(X_test_seq_raw)

    acc_k = accuracy_score(y_test, y_pred_kalman)
    _, _, f1_k, _ = precision_recall_fscore_support(y_test, y_pred_kalman, average='binary', zero_division=0)
    results.append({'Model': 'Kalman Algorithm', 'Accuracy': acc_k, 'F1_Score': f1_k})

    # --- SAVE RESULTS ---
    results_df = pd.DataFrame(results)
    preds_df = pd.DataFrame({'True_Label': y_test, 'Kalman_Pred': y_pred_kalman})
    if y_pred_lstm is not None:
        preds_df['LSTM_Pred'] = y_pred_lstm
    preds_df['ExtraTrees_Pred'] = y_pred_tree

    with pd.ExcelWriter(OUTPUT_FILE_PATH, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Metrics', index=False)
        preds_df.to_excel(writer, sheet_name='Predictions', index=False)

    print(f"\nSuccess! Results saved to {OUTPUT_FILE_PATH}")
    print("\n--- Final Model Metrics ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
