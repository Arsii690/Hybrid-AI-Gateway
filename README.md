# Hybrid AI Gateway: 5G Security & QoS Slicing

A dual-layer AI defense system integrating **AnyLogic Discrete Event Simulation** with **XGBoost Machine Learning** to secure 5G Edge nodes.

## 🚀 Overview
This project solves the decoupling of security and QoS in 5G networks. It features a mandatory "Security Bouncer" that intercepts threats before they enter the slicing pipeline, preserving resources for critical URLLC traffic.

## 🛠️ Tech Stack
- **Simulation:** AnyLogic 8.8 (Process Modeling Library)
- **AI Backend:** Python 3.9+, Flask, XGBoost, Scikit-Learn
- **Deployment:** Hardware-aware optimization for Apple Silicon (M1/M2)

## 📁 Project Structure
- `core2.0/`: AnyLogic source files and 3D assets.
- `src/`: Python scripts for training and the live API server.
- `models/`: Pre-trained XGBoost models (JSON/Joblib format).
- `data/`: Training datasets (Note: Large PCAP-converted CSVs excluded from standard clone).

## 🚦 How to Run
1. **Initialize API:** ```bash
   python api_server.py