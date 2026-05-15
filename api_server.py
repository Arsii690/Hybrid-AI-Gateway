from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

print("Waking up the AI Brains...")

# 1. Load the Security Bouncer
bouncer_model = xgb.XGBClassifier()
bouncer_model.load_model("models/security_ai_model.json")
print(" Security Bouncer is awake!")

# 2. Load the Slicer
slicer_model = xgb.XGBClassifier()
slicer_model.load_model("models/slicer_ai_model.json")
print("✅ Network Slicer is awake!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # --- LAYER 1: THE BOUNCER ---
        # Extract the exact 4 features your Random Forest found
        security_features = pd.DataFrame([{
            'Fwd IAT Min': data.get('Fwd IAT Min', 0),
            'Init_Win_bytes_backward': data.get('Init_Win_bytes_backward', 0),
            'Flow IAT Min': data.get('Flow IAT Min', 0),
            'Fwd Header Length': data.get('Fwd Header Length', 0)
        }])
        
        # Ask the Bouncer: Is this a hacker?
        is_hacker = bouncer_model.predict(security_features)[0]
        
        if is_hacker == 1:
            print("🚨 HACKER DETECTED! Dropping packet.")
            # 99 is the kill-code we will tell AnyLogic to look for
            return jsonify({"slice_decision": 99})
            
        # --- LAYER 2: THE SLICER ---
        # If it's safe, send it to the Slicer
        slicer_features = pd.DataFrame([{
            'p_size': data.get('p_size', 0),
            'p_urgency': data.get('p_urgency', 0),
            'queue_size': data.get('queue_size', 0)
        }])
        
        slice_lane = int(slicer_model.predict(slicer_features)[0])
        print(f"✅ Packet Safe. Routing to lane: {slice_lane}")
        
        return jsonify({"slice_decision": slice_lane})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Hybrid AI Gateway Listening on Port 8000...")
    app.run(port=8000, debug=True, use_reloader=False)
