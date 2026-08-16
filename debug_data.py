"""
Debug version to test data loading
"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SENSOR_DATA_PATH = os.path.join(PROJECT_DIR, "fall_detection_dataset.csv")
IMAGE_DATASET_PATH = r"C:\Users\KAUSHIK G\Downloads\archive (3)\laying_dataset"

print("="*60)
print("DEBUG: Testing data loading")
print("="*60)

# Check sensor data
print(f"\n1. Checking sensor data at: {SENSOR_DATA_PATH}")
if os.path.exists(SENSOR_DATA_PATH):
    print(f"   ✅ File exists")
    df = pd.read_csv(SENSOR_DATA_PATH)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
else:
    print(f"   ❌ File not found")

# Check image dataset
print(f"\n2. Checking image dataset at: {IMAGE_DATASET_PATH}")
if os.path.exists(IMAGE_DATASET_PATH):
    print(f"   ✅ Directory exists")
    
    labels_dir = os.path.join(IMAGE_DATASET_PATH, "labels")
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    
    print(f"   Found {len(label_files)} label files")
    
    if label_files:
        sample_file = label_files[0]
        print(f"\n   Sample label file: {os.path.basename(sample_file)}")
        with open(sample_file, 'r') as f:
            content = f.read().strip()
        print(f"   Content: {content[:100]}...")
        
        values = list(map(float, content.split()))
        print(f"   Label: {int(values[0])} (0=laying, 1=standing)")
        print(f"   Total values: {len(values)}")
        print(f"   Expected: 5 (class, x, y, w, h) + 17*3 (keypoints) = {5 + 17*3}")
else:
    print(f"   ❌ Directory not found")

print("\n" + "="*60)
print("Debug complete!")
print("="*60)
