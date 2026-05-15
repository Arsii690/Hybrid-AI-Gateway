import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys
import multiprocessing

# --- 1. CONFIGURATION & DIRECTORY CHECK ---
DATA_PATH = 'data/training_data.xlsx'
MODEL_DIR = 'models'
MODEL_NAME = 'slicer_ai_model.pkl'
NUM_CORES = multiprocessing.cpu_count()  # Use all available CPU cores

if not os.path.exists(DATA_PATH):
    print(f"ERROR: Could not find {DATA_PATH}. Ensure you ran AnyLogic first!")
    sys.exit()

# --- 2. LOAD DATA ---
print(f"Loading simulation data... (using {NUM_CORES} CPU cores)")
df = pd.read_excel(DATA_PATH, header=None, names=['p_size', 'p_urgency', 'queue_size'])

# Check if the file is empty
if df.empty:
    print("ERROR: The Excel file is empty. Run the simulation for at least 60 seconds!")
    sys.exit()

print(f"Loaded {len(df)} rows.")
print(f"  p_size   : {df['p_size'].min()} – {df['p_size'].max()}")
print(f"  p_urgency: {df['p_urgency'].min()} – {df['p_urgency'].max()}")
print(f"  queue_size: {df['queue_size'].min()} – {df['queue_size'].max()}")

# --- 3. LABELING LOGIC (Expert-Tuned QoS Rules) ---
# Thresholds calibrated to actual data ranges:
#   p_size   : 256 – 1024 bytes
#   p_urgency: 1 – 10
#   queue_size: 0 – 31217
def assign_slice(row):
    # Slice 0: GOLD (Ultra-Reliable Low Latency)
    # High urgency packets get priority unless the network is congested
    if row['p_urgency'] >= 7 and row['queue_size'] < 15000:
        return 0
    # Slice 2: BRONZE (Background/Massive IoT)
    # Very large packets (top ~25%) or very low urgency go to the slow lane
    elif row['p_size'] > 800 or row['p_urgency'] <= 2:
        return 2
    # Slice 1: SILVER (Enhanced Mobile Broadband)
    # Everything else falls into the standard lane
    else:
        return 1

df['slice'] = df.apply(assign_slice, axis=1)

# --- 4. DATA VALIDATION (Check for Class Imbalance) ---
print("\n--- Data Distribution ---")
class_dist = df['slice'].value_counts(normalize=True).sort_index()
slice_names = {0: 'Gold', 1: 'Silver', 2: 'Bronze'}
for cls, pct in class_dist.items():
    print(f"  Slice {cls} ({slice_names.get(cls, '?')}): {pct*100:.1f}%")

unique_classes = sorted(df['slice'].unique())
if len(unique_classes) < 2:
    print("ERROR: Only one class found. Adjust labeling thresholds or collect more data!")
    sys.exit()

# --- 5. PREPARE FOR AI TRAINING ---
X = df[['p_size', 'p_urgency', 'queue_size']]
y = df['slice']

# Encode labels to consecutive integers [0, 1, ...] as required by XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Compute class weights to handle imbalance
class_counts = np.bincount(y_train)
total = len(y_train)
sample_weights = np.array([total / (len(class_counts) * class_counts[c]) for c in y_train])

# --- 6. TRAIN XGBOOST (Optimized for Apple Silicon) ---
print(f"\nTraining XGBoost Classifier on {NUM_CORES} cores...")
num_classes = len(label_encoder.classes_)
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    objective='multi:softprob',
    num_class=num_classes,
    tree_method='hist',       # Fastest method for CPU
    n_jobs=NUM_CORES,         # Use all CPU cores
    random_state=42,
    eval_metric='mlogloss',
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# --- 7. EVALUATION & INSIGHTS ---
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Map encoded labels back to original names for the report
present_names = [slice_names[c] for c in label_encoder.classes_]

print(f"\n--- Model Evaluation ---")
print(f"AI Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=present_names))

# Feature Importance: Tells you which variable (Size, Urgency, or Queue) the AI cares about most
importance = dict(zip(X.columns, model.feature_importances_))
print("\nFeature Importance (What the AI learned):")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {score:.4f}")

# --- 8. SAVE THE BRAIN (NATIVE XGBOOST WAY) ---
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    
# We use .save_model() instead of joblib
model.save_model(f"{MODEL_DIR}/slicer_ai_model.json")
print(f"\nSUCCESS: Model saved to {MODEL_DIR}/slicer_ai_model.json")