import pandas as pd
import numpy as np
import xgboost as xgb
import os
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- HARDWARE OPTIMIZATION ---
# Automatically detect how many cores the Mac has (usually 8 for M1)
total_cores = multiprocessing.cpu_count()
use_cores = total_cores  # Change this to 4 if you want to leave room for other apps
print(f"Hardware Detected: Mac with {total_cores} CPU cores. Utilizing {use_cores} cores for training...\n")

# 1. LOAD THE DATA
print("Loading Security Data (This might take 10 seconds)...")
data_path = 'data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
df = pd.read_csv(data_path, low_memory=False)

df.columns = df.columns.str.strip()

# 2. LABEL THE HACKERS
print("Labeling Normal vs Malicious Traffic...")
df['Target'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

y = df['Target']
X = df.drop(columns=['Label', 'Target'])

X = X.replace([np.inf, -np.inf, 'Infinity', 'infinity'], np.nan)
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(0)

# 3. THE HUNTER: FIND TOP 4 FEATURES
print(f"Scanning 80 features using {use_cores} cores...")
# n_jobs forces the Random Forest to split the work across multiple cores
hunter = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=use_cores)
hunter.fit(X, y)

importances = pd.Series(hunter.feature_importances_, index=X.columns)
top_4_features = importances.sort_values(ascending=False).head(4).index.tolist()
print(f"\nSUCCESS! Top 4 Features Discovered:\n{top_4_features}\n")

# 4. TRAIN THE BOUNCER
print(f"Training the Security Bouncer using {use_cores} cores...")
X_final = X[top_4_features]

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# n_jobs forces XGBoost to engage the libomp multi-threading engine
security_ai = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    n_jobs=use_cores 
)
security_ai.fit(X_train, y_train)

# 5. THE REPORT CARD
y_pred = security_ai.predict(X_test)
print("\n--- Security AI Report Card ---")
print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Hacker (1)"]))

# 6. SAVE THE BRAIN
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

security_ai.save_model(f"{MODEL_DIR}/security_ai_model.json")
print(f"\nBrain Saved: {MODEL_DIR}/security_ai_model.json")
