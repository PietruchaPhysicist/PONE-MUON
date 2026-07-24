import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os 


# =========================================================
# 1. LOAD AND CONDENSE MULTIPLE FILES
# =========================================================
noise_fraction = 0.5
batch_size = 64
learning_rate = 0.00001
epochs = 7
size_test = 0.2

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
        # ADDED PMT HERE so we can calculate directions later
        df_clean = file_df[['string', 'om', 'pmt', 'time', 'npe', 'label']].dropna()

        # Downsample noise for fast training 
        signal_hits = df_clean[df_clean['label'] == 1]
        noise_hits = df_clean[df_clean['label'] == 0].sample(frac=noise_fraction, random_state=42)
        
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
print("2. Translating String/OM/PMT IDs into 3D Physical Space...")

def apply_geometry(data):
    """
    Translates abstract IDs into true P-ONE Cartesian coordinates and vectors
    Based on detector schematics: 
    - 40m Hexagonal Grid in XY plane
    - 20 modules per 1km (50m vertical spacing)
    - 16 PMTs arranged spherically
    """
    df = data.copy()
    
    # --- XY PLANE: 40m Hexagonal Grid ---
    grid_size = 40
    row = df['string'] // grid_size
    col = df['string'] % grid_size
    
    df['x_coord'] = (col * 40.0) + ((row % 2) * 20.0)
    df['y_coord'] = row * (40.0 * (np.sqrt(3) / 2.0))
    
    # --- Z AXIS: 50m Vertical Spacing ---
    df['z_coord'] = df['om'] * -50.0 
    
    # --- PMT 3D Direction Vectors (dx, dy, dz) ---
    n_pmts = 16
    phi = np.pi * (3. - np.sqrt(5.))  # Golden Angle
    
    pmt_idx = df['pmt'].values % n_pmts 
    
    # Fibonacci Sphere calculation
    df['pmt_dz'] = 1 - (pmt_idx / float(n_pmts - 1)) * 2  
    radius = np.sqrt(1 - df['pmt_dz']**2)
    theta = phi * pmt_idx
    
    df['pmt_dx'] = np.cos(theta) * radius
    df['pmt_dy'] = np.sin(theta) * radius
    
    return df

# Apply the geometry translation to our new master dataset
geo_df = apply_geometry(master_df)

# Our features are now explicitly Geometric, Temporal, and Directional!
# We extract all 8 features
X_raw = geo_df[['x_coord', 'y_coord', 'z_coord', 'time', 'npe', 'pmt_dx', 'pmt_dy', 'pmt_dz']].values
y_raw = geo_df['label'].values

# =========================================================
# 3. PYTORCH PREPARATION
# =========================================================
print("3. Scaling data and moving to PyTorch Tensors...")
# We scale the features to have better training performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=size_test, random_state=42)

# We are passing a flat array of [X, Y, Z, Time, NPE] directly to the neurons.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# =========================================================
# 4. BUILD THE GEOMETRIC MLP (Multi-Layer Perceptron)
# =========================================================

# Define the architecture of the MLP
class GeometricMLP(nn.Module):
    def __init__(self):
        super(GeometricMLP, self).__init__()
        
        # Layer 1: Now looks at 8 features (X, Y, Z, T, Charge, DX, DY, DZ)
        self.layer1 = nn.Linear(8, 32)
        
        # Layer 2: combines the learned features into a smaller representation
        self.layer2 = nn.Linear(32, 24)

        self.layer3 = nn.Linear(24, 16)  

        self.layer4 = nn.Linear(16, 8) 

        self.layer5 = nn.Linear(8, 4) 
        
        # Layer 4: Final decision (Muon or Noise)
        self.output_layer = nn.Linear(4, 1)

        # Activation function: ReLU is a common choice for hidden layers 
        self.relu = nn.ReLU()

    # Forward pass defines how data flows through the network
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))   
        x = self.output_layer(x)
        return x


# Calculate pos_weight for the loss function to handle class imbalance
num_noise = (y_train == 0).sum()
num_muons = (y_train == 1).sum()
weight_ratio = num_noise / num_muons
print(f"-> Calculated pos_weight ratio: {weight_ratio:.2f}")

# Convert to a PyTorch float32 tensor
pos_weight_tensor = torch.tensor([weight_ratio], dtype=torch.float32)
print("4. Initializing PyTorch MLP...")

# Choosing geometry-based MLP model, loss function (calculate how well the model is performing), and optimizer
model = GeometricMLP()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor) # Used to handle class imbalance by giving more importance to the minority class (signal hits) during training. This helps the model learn to detect rare events more effectively.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Learning rate, how long it takes to converge

# =========================================================
# 5. TRAINING LOOP
# =========================================================

print(f"5. Training the model for {epochs} epochs...")

# Create an empty list to store the data for our plot
epoch_losses = []

for epoch in range(epochs):
    #model.train() sets the model to training mode, which is important for layers like dropout and batch normalization that behave differently during training and evaluation.
    model.train() 
    #current_loss keeps track of the loss for the current epoch, which is useful for monitoring training progress and diagnosing issues like overfitting or underfitting.
    current_loss = 0.0
    
    for batch_X, batch_y in train_loader:
            optimizer.zero_grad()#clears old gradients from the last step (otherwise they would accumulate)
            outputs = model(batch_X)#outputs are the raw predictions from the model for the current batch of input data
            loss = criterion(outputs, batch_y)#calculates the loss between the model's predictions and the true labels for the current batch
            loss.backward()#computes the gradient of the loss with respect to the model's parameters (weights and biases) using backpropagation
            optimizer.step()#updates the model's parameters based on the computed gradients and the learning rate defined in the optimizer
            current_loss += loss.item()#converts the loss tensor to a Python float and adds it to the current_loss for tracking the total loss over the epoch 
        
    avg_loss = current_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    
    print(f"   -> Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# =========================================================
# 6. EVALUATION
# =========================================================
print("\n6. Evaluating on unseen test data...")
model.eval()

with torch.no_grad(): 
    test_outputs = model(X_test_tensor)
    
    probabilities = torch.sigmoid(test_outputs).numpy()
    predictions = probabilities.round()

print("\n=======================================================")
print("            PYTORCH 3D GEOMETRY REPORT")
print("=======================================================")
print(classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)']))
print("=======================================================")

# =========================================================
# 7. GENERATING CONFERENCE PLOTS
# =========================================================
print("\n7. Generating and saving visualization plots...")
sns.set_theme(style="whitegrid")

# Plot 1: Receiver Operating Characteristic (ROC) Curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f}(area under the curve))')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.005, 1.0])
plt.ylim([0.0, 1.005])
plt.xlabel('FP - False Positive Rate (Noise falsely flagged as Signal)', fontsize=12)
plt.ylabel('TP - True Positive Rate (Muons correctly detected)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join("pytorch_results", "pytorch_roc_curve.png"), dpi=300)


#Plot 2: Training Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='b', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title("Model Training Loss Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
plt.xticks(range(1, epochs + 1))
plt.tight_layout()
plt.savefig(os.path.join("pytorch_results", "pytorch_training_loss.png"), dpi=300)
plt.close()



#Plot 3: Confusion Matrix Heatmap
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Noise', 'Predicted Muon'], 
            yticklabels=['Actual Noise', 'Actual Muon'],
            annot_kws={"size": 14})
plt.title('Model Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join("pytorch_results", "pytorch_confusion_matrix.png"), dpi=300)
plt.close()

print("-> Successfully saved 'pytorch_training_loss.png', 'pytorch_roc_curve.png', and 'pytorch_confusion_matrix.png' to your directory.")


# =========================================================
# 8. EXPERIMENT TRACKING LOG
# =========================================================
print("\n8. Saving results to experiment log...")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Create a report log entry with all relevant details
log_entry = f"""
=======================================================
EXPERIMENT RUN: {timestamp}
=======================================================
Noise Fraction: {noise_fraction}
Batch Size:     {batch_size}
Learning Rate:  {learning_rate}
Epochs:         {epochs}
ROC AUC Score:  {roc_auc:.4f}
Test size:      {size_test}

Classification Report

{classification_report(y_test, predictions, target_names=['Noise (0)', 'Muon Signal (1)'])}"""

# The 'a' means "append". It adds to the bottom instead of overwriting!
with open(os.path.join("pytorch_results", "pytorch_experiment_log.txt"), "a") as file:
    file.write(log_entry)

print("-> Journal entry added to 'pytorch_experiment_log.txt'!")