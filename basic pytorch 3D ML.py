import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# =========================================================
# 1. LOAD AND CONDENSE MULTIPLE FILES
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

all_condensed_data = []

print(f"1. Loading and condensing {len(file_list)} file(s)...")

for file_name in file_list:
    print(f"   -> Processing {file_name}...")
    try:
        with h5py.File(file_name, 'r') as f:
            signal_df = pd.DataFrame(f['Accepted_MCPEMap'][:])
            signal_df['label'] = 1 
            
            k40_df = pd.DataFrame(f['Noise_K40'][:])
            k40_df['label'] = 0 
            
            dark_df = pd.DataFrame(f['Noise_Dark'][:])
            dark_df['label'] = 0

        file_df = pd.concat([signal_df, k40_df, dark_df], ignore_index=True)
        df_clean = file_df[['string', 'om', 'time', 'npe', 'label']].dropna()

        # Downsample noise for fast training (Keep 100% signal, 5% noise)
        signal_hits = df_clean[df_clean['label'] == 1]
        noise_hits = df_clean[df_clean['label'] == 0].sample(frac=0.05, random_state=42)
        
        condensed_file = pd.concat([signal_hits, noise_hits], ignore_index=True)
        all_condensed_data.append(condensed_file)
        
    except FileNotFoundError:
        print(f"      [WARNING] Could not find {file_name}. Skipping...")

print("\nCombining all files into a Master Dataset...")
master_df = pd.concat(all_condensed_data, ignore_index=True)
print(f"Total rows for Machine Learning: {len(master_df)}")

# =========================================================
# 2. THE GEOMETRY TRANSLATOR
# =========================================================
print("2. Translating String/OM IDs into 3D Physical Space...")

def apply_geometry(data):
    """
    Translates abstract IDs into true P-ONE Cartesian coordinates (X, Y, Z)
    Based on detector schematics: 
    - 40m Hexagonal Grid in XY plane
    - 20 modules per 1km (50m vertical spacing)
    """
    df = data.copy()
    
    # --- XY PLANE: 40m Hexagonal Grid ---
    # We approximate the hex grid using staggered rows.
    # Assuming a rough 40x40 grid to cover the ~1000+ strings shown in the plot
    grid_size = 40
    
    # Calculate row and column based on string ID
    row = df['string'] // grid_size
    col = df['string'] % grid_size
    
    # X spacing is 40m. Every odd row is shifted by half the spacing (+20m) to form hexagons
    df['x_coord'] = (col * 40.0) + ((row % 2) * 20.0)
    
    # Y spacing in a perfect hexagon is spacing * sqrt(3)/2
    df['y_coord'] = row * (40.0 * (np.sqrt(3) / 2.0))
    
    # --- Z AXIS: 50m Vertical Spacing ---
    # 20 modules along 1km of cable = 50m per module
    # We use negative to represent going deeper into the ocean
    df['z_coord'] = df['om'] * -50.0 
    
    return df

# Apply the geometry translation to our new master dataset
geo_df = apply_geometry(master_df)

# Our features are now explicitly Geometric and Temporal!
X_raw = geo_df[['x_coord', 'y_coord', 'z_coord', 'time', 'npe']].values
y_raw = geo_df['label'].values

# =========================================================
# 3. PYTORCH PREPARATION
# =========================================================
print("3. Scaling data and moving to PyTorch Tensors...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=0.2, random_state=42)

# Notice we DO NOT use `.unsqueeze(1)` here like we did with the CNN.
# We are passing a flat array of [X, Y, Z, Time, NPE] directly to the neurons.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# =========================================================
# 4. BUILD THE GEOMETRIC MLP (Multi-Layer Perceptron)
# =========================================================
class GeometricMLP(nn.Module):
    def __init__(self):
        super(GeometricMLP, self).__init__()
        
        # Layer 1: Looks at our 5 physical features (X, Y, Z, T, Charge)
        self.layer1 = nn.Linear(5, 32)
        
        # Layer 2: Learns relationships (e.g., correlations between Z and T)
        self.layer2 = nn.Linear(32, 16)
        
        # Layer 3: Final decision (Muon or Noise)
        self.output_layer = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

print("4. Initializing PyTorch MLP...")
model = GeometricMLP()
criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================================================
# 5. TRAINING LOOP
# =========================================================
epochs = 5
print(f"5. Training the model for {epochs} epochs...")

for epoch in range(epochs):
    model.train() 
    current_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
        
    print(f"   -> Epoch {epoch+1}/{epochs} | Loss: {current_loss/len(train_loader):.4f}")

# =========================================================
# 6. EVALUATION
# =========================================================
print("\n6. Evaluating on unseen test data...")
model.eval()

with torch.no_grad(): 
    test_outputs = model(X_test_tensor)
    predictions = torch.sigmoid(test_outputs).round().numpy()

print("\n=======================================================")
print("            PYTORCH 3D GEOMETRY REPORT")
print("=======================================================")
print(classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)']))
print("=======================================================")