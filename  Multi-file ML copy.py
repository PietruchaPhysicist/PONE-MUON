import h5py
import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# =========================================================
# 1. DEFINE YOUR FILES & MODEL SAVE PATH
# =========================================================
file_list = [
    '40906960_gen_001.i3.zst.h5',
    '40906960_gen_002.i3.zst.h5',
    '40906960_gen_003.i3.zst.h5',
    '40906960_gen_004.i3.zst.h5',
    '40906960_gen_005.i3.zst.h5',
    '40906960_gen_006.i3.zst.h5',
    '40906960_gen_007.i3.zst.h5',
    '40906960_gen_008.i3.zst.h5',
    '40906960_gen_009.i3.zst.h5',
    '40906960_gen_010.i3.zst.h5',
    '40917491_gen_001.i3.zst.h5',
    '40917491_gen_002.i3.zst.h5',
    '40917491_gen_003.i3.zst.h5',
    '40917491_gen_004.i3.zst.h5',
    '40917491_gen_005.i3.zst.h5',
    '40917491_gen_006.i3.zst.h5',
    '40917491_gen_007.i3.zst.h5',
    '40917491_gen_008.i3.zst.h5',
    '40917491_gen_009.i3.zst.h5',
    '40917491_gen_010.i3.zst.h5',
]

# This is where the model's "brain" will be saved after each file
model_save_path = 'neutrino_model.json'

# =========================================================
# 2. INITIALIZE THE MODEL TEMPLATE
# =========================================================
# Note: Every time this model looks at a new file, it will add 100 NEW trees 
# to its brain, making it smarter and smarter with each file it sees.
model = XGBClassifier(
    n_estimators=100,      
    learning_rate=0.1,     
    max_depth=6,           
    scale_pos_weight=5,    # Keep prioritizing rare muon signals!
    random_state=42,
    n_jobs=-1              
)

# =========================================================
# 3. INCREMENTAL TRAINING LOOP (METHOD A)
# =========================================================
for i, file_name in enumerate(file_list):
    print(f"\n=======================================================")
    print(f" PROCESSING FILE {i+1}/{len(file_list)}: {file_name}")
    print(f"=======================================================")
    
    try:
        # --- A. Load the Data ---
        print(" -> Extracting tables from HDF5...")
        with h5py.File(file_name, 'r') as f:
            signal_df = pd.DataFrame(f['Accepted_MCPEMap'][:])
            signal_df['label'] = 1 
            
            k40_df = pd.DataFrame(f['Noise_K40'][:])
            k40_df['label'] = 0 
            
            dark_df = pd.DataFrame(f['Noise_Dark'][:])
            dark_df['label'] = 0
            
        file_df = pd.concat([signal_df, k40_df, dark_df], ignore_index=True)
        ml_columns = ['Event', 'string', 'om', 'pmt', 'time', 'npe', 'label']
        file_df = file_df[ml_columns].dropna()
        
        # (Optional) We still keep a little bit of noise subsampling just to make sure 
        # a single 5-million row file doesn't freeze your laptop during the train_test_split.
        signal_hits = file_df[file_df['label'] == 1]
        noise_hits = file_df[file_df['label'] == 0].sample(frac=0.10, random_state=42)
        final_df = pd.concat([signal_hits, noise_hits], ignore_index=True)
        
        # --- B. Prepare for the test ---
        X = final_df[['string', 'om', 'pmt', 'time', 'npe']]
        y = final_df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f" -> Training on {len(X_train)} hits...")
        
        # --- C. Train (or Update) the Model ---
        if os.path.exists(model_save_path):
            print(" -> Found existing brain! Updating model with new data...")
            # The magic 'xgb_model' parameter tells it to load previous knowledge
            model.fit(X_train, y_train, xgb_model=model_save_path)
        else:
            print(" -> No existing brain found. Training from scratch...")
            model.fit(X_train, y_train)
            
        # --- D. Save the new, smarter brain! ---
        model.save_model(model_save_path)
        print(f" -> Success! Knowledge saved to {model_save_path}")
        
        # --- E. Evaluate how well it learned THIS file ---
        print("\n -> Report Card for this specific file:")
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)']))
        
    except FileNotFoundError:
        print(f"    [ERROR] Could not find {file_name}. Did you misspell it?")
        
print("\n=======================================================")
print(f"PIPELINE COMPLETE! Final model saved safely as: {model_save_path}")
print("=======================================================")