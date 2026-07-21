import h5py
import pandas as pd

# Updated to your new file!
file_name = '40906960_gen_001.i3.zst.h5'

print(f"Extracting data from {file_name} and building ML dataset...")

with h5py.File(file_name, 'r') as f:
    # 1. Extract Signal Hits and label them '1'
    signal_df = pd.DataFrame(f['Accepted_MCPEMap'][:])
    signal_df['label'] = 1 
    
    # 2. Extract Potassium-40 Noise and label it '0'
    k40_df = pd.DataFrame(f['Noise_K40'][:])
    k40_df['label'] = 0 
    
    # 3. Extract Sensor Dark Noise and label it '0'
    dark_df = pd.DataFrame(f['Noise_Dark'][:])
    dark_df['label'] = 0

# 4. Glue them all together into one big dataset
df = pd.concat([signal_df, k40_df, dark_df], ignore_index=True)

# 5. Keep only the physical features, the grouping IDs, and our label
ml_columns = ['Event', 'string', 'om', 'pmt', 'time', 'npe', 'label']
df_clean = df[ml_columns].copy()

# 6. Check for missing data (NaNs) and drop them if they exist
df_clean = df_clean.dropna()

print("\n--- CLEAN DATASET PREVIEW ---")
print(df_clean.head())
print(f"\nTotal rows ready for Machine Learning: {len(df_clean)}")
