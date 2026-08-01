import pandas as pd
import numpy as np

# Configuration to match your script
FILE_NAME = r"C:\Users\KAUSHIK G\Downloads\fall_detection 100.xlsx"
SAMPLING_RATE = 100 
DURATION_SEC = 20  # Total 20 seconds of data
TOTAL_ROWS = SAMPLING_RATE * DURATION_SEC

def generate_sample_data():
    np.random.seed(42)
    
    # Create timestamps (not required by your code, but good for context)
    time = np.linspace(0, DURATION_SEC, TOTAL_ROWS)
    
    # Generate random sensor noise (Normal Activity)
    acc_x = np.random.normal(0, 0.1, TOTAL_ROWS)
    acc_y = np.random.normal(1.0, 0.1, TOTAL_ROWS) # Gravity on Y
    acc_z = np.random.normal(0, 0.1, TOTAL_ROWS)
    
    gyro_x = np.random.normal(0, 5, TOTAL_ROWS)
    gyro_y = np.random.normal(0, 5, TOTAL_ROWS)
    gyro_z = np.random.normal(0, 5, TOTAL_ROWS)
    
    heart_rate = np.random.randint(70, 85, TOTAL_ROWS)
    fall_event = np.zeros(TOTAL_ROWS)

    # Simulate 2 FALL EVENTS (Spikes in data)
    # Fall 1: at 5 seconds
    fall_idx_1 = 5 * SAMPLING_RATE
    acc_x[fall_idx_1 : fall_idx_1 + 50] += np.random.uniform(3, 8, 50)
    gyro_z[fall_idx_1 : fall_idx_1 + 50] += np.random.uniform(100, 200, 50)
    fall_event[fall_idx_1 : fall_idx_1 + 100] = 1 # Mark fall duration
    
    # Fall 2: at 15 seconds
    fall_idx_2 = 15 * SAMPLING_RATE
    acc_z[fall_idx_2 : fall_idx_2 + 50] -= np.random.uniform(3, 8, 50)
    heart_rate[fall_idx_2 : fall_idx_2 + 100] += 20
    fall_event[fall_idx_2 : fall_idx_2 + 100] = 1

    # Create DataFrame
    df = pd.DataFrame({
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'heart_rate': heart_rate,
        'fall_event': fall_event
    })

    # Save to Excel
    try:
        df.to_excel(FILE_NAME, index=False, engine='openpyxl')
        print(f"Success! Sample dataset created at: {FILE_NAME}")
    except Exception as e:
        print(f"Error saving file: {e}")
        print("Make sure the folder path exists and you have permission to write to it.")

if __name__ == "__main__":
    generate_sample_data()
