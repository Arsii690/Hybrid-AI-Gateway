import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
NUM_ROWS = 50000  # Number of packets to simulate (Adjust this if you want more/less data)
OUTPUT_DIR = 'data'
OUTPUT_FILE = f'{OUTPUT_DIR}/training_data.xlsx'

def generate_simulation_data(num_samples):
    print(f"Generating {num_samples} simulated packet records...")
    
    # 1. Packet Size (p_size): 64 to 1500 bytes (standard MTU limits)
    # Using a normal distribution centered around 512 to mimic real network traffic
    p_sizes = np.random.normal(loc=512, scale=256, size=num_samples)
    p_sizes = np.clip(p_sizes, 64, 1500).astype(int)
    
    # 2. Urgency Score (p_urgency): 1 to 10
    # Uniformly distributed so the AI learns all priority levels equally
    p_urgencies = np.random.randint(1, 11, size=num_samples)
    
    # 3. Queue Size (queue_size): 0 to ~31217 (Based on your script's max range)
    # Using an exponential distribution so most of the time the queue is low, 
    # but it occasionally spikes to simulate heavy network congestion.
    queue_sizes = np.random.exponential(scale=6000, size=num_samples)
    queue_sizes = np.clip(queue_sizes, 0, 31217).astype(int)
    
    # Assemble into a Pandas DataFrame
    df = pd.DataFrame({
        'p_size': p_sizes,
        'p_urgency': p_urgencies,
        'queue_size': queue_sizes
    })
    
    return df

if __name__ == "__main__":
    # Ensure the 'data' folder exists before trying to save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created missing directory: {OUTPUT_DIR}/")
        
    # Generate the simulated data
    sim_df = generate_simulation_data(NUM_ROWS)
    
    # Display a quick preview
    print("\nData Preview:")
    print(sim_df.head())
    
    # Save to Excel
    # IMPORTANT: index=False and header=False ensures it matches train_slicer.py exactly!
    print(f"\nSaving to {OUTPUT_FILE}...")
    sim_df.to_excel(OUTPUT_FILE, index=False, header=False)
    
    print("\nSUCCESS: Simulated data generation complete!")
    print("Next Step: You can now run 'python train_slicer.py' to label this data and train the AI.")