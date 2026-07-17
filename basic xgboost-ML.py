import h5py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# 1. DEFINE YOUR FILES
# Add as many files as you want to this list!
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

# We will store the condensed versions of each file in this list
all_condensed_data = []

# =========================================================
# 2. LOAD AND CONDENSE MULTIPLE FILES (SMART SUBSAMPLING)
# =========================================================
print(f"Loading and condensing {len(file_list)} file(s)...")

for file_name in file_list:
    print(f" -> Processing {file_name}...")
    
    try:
        with h5py.File(file_name, 'r') as f:
            # Load Signal (Label 1)
            signal_df = pd.DataFrame(f['Accepted_MCPEMap'][:])
            signal_df['label'] = 1 
            
            # Load Noise (Label 0)
            k40_df = pd.DataFrame(f['Noise_K40'][:])
            k40_df['label'] = 0 
            dark_df = pd.DataFrame(f['Noise_Dark'][:])
            dark_df['label'] = 0
            
        # Combine just for this file
        file_df = pd.concat([signal_df, k40_df, dark_df], ignore_index=True)
        
        # Keep only ML columns
        ml_columns = ['Event', 'string', 'om', 'pmt', 'time', 'npe', 'label']
        file_df = file_df[ml_columns].dropna()
        
        # --- THE MEMORY SAVING TRICK ---
        # Keep 100% of the valuable signal hits
        signal_hits = file_df[file_df['label'] == 1]
        
        # Randomly sample only 5% of the background noise to save RAM
        # (Change 0.05 to a lower number if your laptop still runs out of memory)
        noise_hits = file_df[file_df['label'] == 0].sample(frac=0.05, random_state=42)
        
        condensed_file = pd.concat([signal_hits, noise_hits])
        all_condensed_data.append(condensed_file)
        
    except FileNotFoundError:
        print(f"    [WARNING] Could not find {file_name}. popraw nazwe pliku")

# =========================================================
# 3. PREPARE THE MASTER DATASET
# =========================================================
print("\nCombining all files into a Master Dataset...")
master_df = pd.concat(all_condensed_data, ignore_index=True)
print(f"Total rows for Machine Learning: {len(master_df)}")

# Separate Features (X) and Target (y)
# Notice we drop 'Event' here! We don't want the model memorizing event ID numbers.
X = master_df[['string', 'om', 'pmt', 'time', 'npe']]
y = master_df['label']

# Split into Training Data (80%) and taking a final Exam on Test Data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} hits, Testing on {len(X_test)} hits.")

# =========================================================
# 4. TRAIN THE XGBOOST MODEL
# =========================================================
print("\nInitializing XGBoost Model...")
# scale_pos_weight forces the model to pay extra attention to the rare signal hits
model = XGBClassifier(
    n_estimators=100,      # Number of decision trees
    learning_rate=0.1,     # How fast it learns
    max_depth=6,           # How complex each tree can be
    scale_pos_weight=5,    # Balances the fact that we still have more noise than signal
    random_state=42,
    n_jobs=-1              # Uses all your Mac's CPU cores to train faster!
)

print("Training model... ")
model.fit(X_train, y_train)

# =========================================================
# 5. EVALUATE THE MODEL
# =========================================================
print("Making predictions on the unseen test data...")
predictions = model.predict(X_test)

print("\n=======================================================")
print("                 MODEL REPORT CARD")
print("=======================================================")
print(classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)']))

# Note: In the future, we will calculate the 'density_feature' here before training 
# to make the model significantly smarter!  
